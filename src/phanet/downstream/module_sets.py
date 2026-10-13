"""Member-set inference architecture; seven auxiliary types are independent of nine functional classes."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn

@dataclass(frozen=True)
class ModuleSetConfig:
    input_dim: int = 1280
    hidden_dim: int = 192
    slots: int = 32
    heads: int = 4
    width: int = 30
    dropout: float = 0.0
    network_dim: int = 0

    def __post_init__(self):
        if min(self.input_dim, self.hidden_dim, self.slots, self.heads) < 1:
            raise ValueError('Dimensions must be positive')
        if self.hidden_dim % self.heads or not 2 <= self.width <= 30:
            raise ValueError('Invalid attention dimensions/window')
        if not 0 <= self.dropout < 1:
            raise ValueError('Invalid dropout')
        if self.network_dim < 0:
            raise ValueError('Invalid network dimension')


class NetworkResidual(nn.Module):
    """Adapt external features to locus representations; missing inputs contribute zero."""
    def __init__(self, hidden_dim, evidence_dim):
        super().__init__()
        self.evidence_dim = evidence_dim
        self.project = nn.Sequential(nn.Linear(evidence_dim, 64), nn.GELU(),
                                     nn.Linear(64, hidden_dim))
        self.gate = nn.Linear(hidden_dim + evidence_dim, 1)
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, states, evidence, available, valid):
        if evidence.shape != (*valid.shape, self.evidence_dim) or available.shape != valid.shape:
            raise ValueError('Network input shape mismatch')
        if available.dtype != torch.bool:
            raise ValueError('Network availability must be boolean')
        active = available & valid
        # Mask missing features in both adapter branches.
        x = torch.where(active[..., None], evidence.to(states.dtype), 0.)
        if not torch.isfinite(x).all():
            raise ValueError('Nonfinite active network evidence')
        gate = self.gate(torch.cat((states, x), -1)).sigmoid()
        residual = gate * self.project(x)
        return states + torch.where(active[..., None], residual, 0.)


class AnchorSetHead(nn.Module):
    """Lightweight content queries, with a direct frozen ESM2 feature path."""
    def __init__(self, config=ModuleSetConfig()):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.context = nn.Sequential(nn.LayerNorm(config.input_dim), nn.Linear(config.input_dim, d))
        self.protein = nn.Sequential(nn.LayerNorm(config.input_dim), nn.Linear(config.input_dim, d))
        self.memory_norm = nn.LayerNorm(d)
        self.network = NetworkResidual(d, config.network_dim) if config.network_dim else None
        self.slot = nn.Parameter(torch.empty(config.slots, d))
        nn.init.normal_(self.slot, std=0.02)
        self.attention = nn.MultiheadAttention(d, config.heads, dropout=config.dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2*d), nn.GELU(), nn.Dropout(config.dropout), nn.Linear(2*d, d))
        self.output_norm = nn.LayerNorm(d)
        self.mask_query = nn.Linear(d, d)
        self.mask_key = nn.Linear(d, d)
        self.distance_bias = nn.Embedding(config.width, 1)
        self.strand_bias = nn.Embedding(2, 1)
        self.object_head = nn.Linear(d, 1)
        self.type_head = nn.Linear(d, 7)
        nn.init.zeros_(self.distance_bias.weight)
        nn.init.zeros_(self.strand_bias.weight)
        nn.init.constant_(self.object_head.bias, -2.0)

    def forward(self, states, proteins, strand, valid, network_features=None, network_available=None):
        if states.shape != proteins.shape or states.shape[:2] != valid.shape:
            raise ValueError('Feature shape mismatch')
        b, length, _ = states.shape
        if length > self.config.width or strand.shape != valid.shape or not valid.any(1).all():
            raise ValueError('Invalid context')
        # Right padding only; anchors are native positions.
        if (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError('Noncontiguous padding')
        mem = self.memory_norm((self.context(states) + self.protein(proteins)) / math.sqrt(2))
        if (network_features is None) != (network_available is None):
            raise ValueError('Supply both network features and availability')
        if network_features is not None:
            if self.network is None:
                raise ValueError('Head has no network adapter')
            mem = self.network(mem, network_features, network_available, valid)
        q = (mem[:, :, None] + self.slot[None, None]).reshape(b, -1, self.config.hidden_dim)
        attended = self.attention(q, mem, mem, key_padding_mask=~valid, need_weights=False)[0]
        q = self.query_norm(q + attended)
        q = self.output_norm(q + self.ff(q)).reshape(b, length, self.config.slots, -1)
        logits = torch.einsum('bakd,bjd->bakj', self.mask_query(q), self.mask_key(mem)) / math.sqrt(self.config.hidden_dim)
        pos = torch.arange(length, device=states.device)
        distance = pos[None, :] - pos[:, None]
        bias = self.distance_bias(distance.clamp_min(0)).squeeze(-1)
        same = (strand[:, :, None] == strand[:, None, :]).long()
        logits = logits + bias[None, :, None, :] + self.strand_bias(same).squeeze(-1)[:, :, None, :]
        return {'mask_logits': logits, 'object_logits': self.object_head(q).squeeze(-1),
                'type_logits': self.type_head(q), 'valid': valid}


class GenomicModuleSetModel(nn.Module):
    """Predict member sets using a supplied genomic-context encoder."""
    def __init__(self, glm, head_config, *, masked_state, padding_state, train_last_layers=0):
        super().__init__()
        self.glm = glm
        self.head = AnchorSetHead(head_config)
        self.masked_state = masked_state
        self.padding_state = padding_state
        if glm.config.hidden_dim != head_config.input_dim:
            raise ValueError('GLM and module head dimensions differ')
        if glm.annotation_projection.in_features != 9:
            raise ValueError('Supply the frozen nine-category GLM implementation')
        self.set_trainable_layers(train_last_layers)

    def set_trainable_layers(self, count):
        if not 0 <= count <= len(self.glm.layers):
            raise ValueError('Invalid trainable layer count')
        self.train_last_layers = count
        self.glm.requires_grad_(False)
        if count:
            for layer in self.glm.layers[-count:]:
                layer.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.glm.eval()
        if mode and self.train_last_layers:
            for layer in self.glm.layers[-self.train_last_layers:]:
                layer.train()
        return self

    def encode(self, embeddings, strand, valid):
        annotations = embeddings.new_zeros((*valid.shape, 9))
        state = torch.where(valid, self.masked_state, self.padding_state)
        return self.glm(embeddings, annotations, state, strand, valid, member_mask=None)['states']

    def forward(self, embeddings, strand, valid, network_features=None, network_available=None):
        return self.head(self.encode(embeddings, strand, valid), embeddings, strand, valid,
                         network_features, network_available)
