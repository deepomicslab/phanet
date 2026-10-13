"""Genomic Transformer with simultaneous rank and relative-position encoding."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from phanet.common.constants import CATEGORIES
from phanet.datasets.splits import VISIBLE, MASKED, UNRESOLVED, PADDING


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int = 1280
    hidden_dim: int = 1280
    num_layers: int = 6
    num_heads: int = 8
    feedforward_dim: int = 2560
    dropout: float = .1
    max_context: int = 30
    projection_dim: int = 256
    use_rank: bool = True
    use_relative: bool = True
    use_strand: bool = True
    protein_only: bool = False
    centered_context: bool = False
    module_objective: str = "consistency"

    def validate(self):
        if min(self.input_dim, self.hidden_dim, self.num_layers, self.num_heads, self.feedforward_dim,
               self.max_context, self.projection_dim) < 1 or self.hidden_dim % self.num_heads:
            raise ValueError("Invalid Transformer dimensions")
        if not 0 <= self.dropout < 1:
            raise ValueError("Invalid dropout")
        if self.module_objective not in ("consistency", "composition"):
            raise ValueError("Unknown module objective")
        if self.protein_only and self.module_objective == "composition":
            raise ValueError("Composition requires the genomic Transformer and module token")
        if self.protein_only and self.input_dim != self.hidden_dim:
            raise ValueError("Protein-only control requires the same 1280-dimensional linear head as the full model")

    @classmethod
    def tiny(cls, input_dim=1280):
        return cls(input_dim=input_dim, hidden_dim=64, num_layers=2, num_heads=4,
                   feedforward_dim=128, projection_dim=32)


class RelativeEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.qkv = nn.Linear(config.hidden_dim, 3 * config.hidden_dim)
        self.output = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ff1 = nn.Linear(config.hidden_dim, config.feedforward_dim)
        self.ff2 = nn.Linear(config.feedforward_dim, config.hidden_dim)
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.relative = nn.Embedding(2 * config.max_context - 1, config.num_heads) if config.use_relative else None
        self.special_bias = nn.Parameter(torch.zeros(config.num_heads, 3))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid, *, member_mask=None):
        batch, length, hidden = x.shape
        heads = self.config.num_heads
        qkv = self.qkv(x).reshape(batch, length, 3, heads, hidden // heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hidden // heads)
        has_module = member_mask is not None
        offset = 1 if has_module else 0
        genes = length - offset
        if self.relative is not None:
            pos = torch.arange(genes, device=x.device)
            relative = (pos[:, None] - pos[None, :]).clamp(1 - self.config.max_context, self.config.max_context - 1)
            bias = self.relative(relative + self.config.max_context - 1).permute(2, 0, 1)
            scores[:, :, offset:, offset:] = scores[:, :, offset:, offset:] + bias[None]
        if has_module:
            scores[:, :, 0, 1:] += self.special_bias[:, 0][None, :, None]
            scores[:, :, 1:, 0] += self.special_bias[:, 1][None, :, None]
            scores[:, :, 0, 0] += self.special_bias[:, 2][None]
            scores[:, :, 0, 1:] = scores[:, :, 0, 1:].masked_fill(~member_mask[:, None, :], -torch.inf)
            # Encode each gene in its native genomic context.
            scores[:, :, 1:, 0] = -torch.inf
        scores = scores.masked_fill(~valid[:, None, None, :], -torch.inf)
        probabilities = F.softmax(scores, dim=-1)
        attended = (self.dropout(probabilities) @ v).transpose(1, 2).reshape(batch, length, hidden)
        x = self.norm1(x + self.dropout(self.output(attended)))
        x = self.norm2(x + self.dropout(self.ff2(self.dropout(F.gelu(self.ff1(x))))))
        return x * valid.unsqueeze(-1)


class PhaNetModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.input_projection = nn.Identity() if config.input_dim == config.hidden_dim else nn.Linear(config.input_dim, config.hidden_dim)
        self.annotation_projection = nn.Linear(len(CATEGORIES), config.hidden_dim, bias=False)
        self.state_embedding = nn.Embedding(4, config.hidden_dim, padding_idx=PADDING)
        self.rank_embedding = nn.Embedding(config.max_context, config.hidden_dim) if config.use_rank else None
        self.strand_embedding = nn.Embedding(2, config.hidden_dim) if config.use_strand else None
        self.module_token = nn.Parameter(torch.empty(config.hidden_dim))
        self.protein_mask = nn.Parameter(torch.empty(config.input_dim))
        nn.init.normal_(self.module_token, std=.02)
        nn.init.normal_(self.protein_mask, std=.02)
        self.layers = nn.ModuleList([RelativeEncoderLayer(config) for _ in range(config.num_layers)])
        self.annotation_head = nn.Linear(config.hidden_dim, len(CATEGORIES))
        # Default head dimensions match the checkpoint parameter keys.
        self.projection = None
        self.composition_head = None
        if config.module_objective == "consistency":
            self.projection = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
                                            nn.Linear(config.hidden_dim, config.projection_dim))
        else:
            self.composition_head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.projection_dim), nn.GELU(),
                nn.Linear(config.projection_dim, 1, bias=False))
        self.family_head = nn.Linear(config.hidden_dim, config.input_dim)
        self.order_head = nn.Linear(config.hidden_dim, 1)
        if config.protein_only:
            self.requires_grad_(False)
            self.annotation_head.requires_grad_(True)

    def forward(self, embeddings, annotation_input, state, strand, valid, member_mask=None):
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.input_dim:
            raise ValueError("Protein embedding shape disagrees with model")
        b, l, _ = embeddings.shape
        if l > self.config.max_context or annotation_input.shape != (b, l, len(CATEGORIES)):
            raise ValueError("Context length or annotation shape mismatch")
        if state.shape != (b, l) or strand.shape != (b, l) or valid.shape != (b, l):
            raise ValueError("Token metadata shapes disagree")
        valid = valid.bool()
        if not valid.any(dim=1).all():
            raise ValueError("Empty padded examples must not enter the Transformer")
        if ((state < 0) | (state > PADDING)).any() or ((strand[valid] != -1) & (strand[valid] != 1)).any():
            raise ValueError("Invalid token state or strand")
        if not torch.isfinite(embeddings).all() or not torch.isfinite(annotation_input).all():
            raise ValueError("Nonfinite model input")
        if self.config.protein_only:
            genes = embeddings.float() * valid.unsqueeze(-1)
            mean = genes.sum(dim=1) / valid.sum(dim=1, keepdim=True)
            result = {"states": genes, "logits": self.annotation_head(genes), "genome": mean}
            if member_mask is not None:
                result.update(module=mean, projection=F.normalize(mean, dim=-1),
                              family=F.normalize(mean, dim=-1), order=mean.sum(dim=-1) * 0)
            return result
        # Apply the state mask to annotation inputs.
        annotation = annotation_input * ((state == VISIBLE) & valid).unsqueeze(-1)
        h = self.input_projection(embeddings.float()) + self.annotation_projection(annotation.float()) + self.state_embedding(state.long())
        if self.rank_embedding is not None:
            h = h + self.rank_embedding(torch.arange(l, device=h.device))[None]
        if self.strand_embedding is not None:
            h = h + self.strand_embedding((strand > 0).long())
        if member_mask is not None:
            member_mask = member_mask.bool()
            if member_mask.shape != valid.shape or (member_mask & ~valid).any() or not member_mask.any(dim=1).all():
                raise ValueError("A module token needs valid member positions")
            h = torch.cat([self.module_token.expand(b, 1, -1), h], dim=1)
            attention_valid = torch.cat([torch.ones((b, 1), dtype=torch.bool, device=h.device), valid], dim=1)
        else:
            attention_valid = valid
        for layer in self.layers:
            h = layer(h, attention_valid, member_mask=member_mask)
        genes = h[:, 1:] if member_mask is not None else h
        result = {"states": genes, "logits": self.annotation_head(genes),
                  "genome": (genes * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)}
        if member_mask is not None:
            module = h[:, 0]
            result.update(module=module, family=F.normalize(self.family_head(module), dim=-1),
                          order=self.order_head(module).squeeze(-1))
            if self.projection is not None:
                result["projection"] = F.normalize(self.projection(module), dim=-1)
            else:
                # Compute ranking differences in FP32.
                with torch.autocast(device_type=module.device.type, enabled=False):
                    result["composition"] = self.composition_head(module.float()).squeeze(-1)
        return result


def forward_batch(model, batch, *, members=True):
    return model(batch["embeddings"], batch["annotation_input"], batch["state"],
                 batch["strand"], batch["valid"], batch.get("member_mask") if members else None)
