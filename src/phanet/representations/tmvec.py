"""Full-length ProtT5/TM-Vec protein representations.

TMVecEncoder follows TM-Vec by Tymor Hamamsy (2022), BSD-3-Clause.
See THIRD_PARTY_NOTICES.txt."""
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from phanet.common.io import digest_file, read_json
from phanet.datasets.data import normalize_protein

UPSTREAM_SHA256 = "5a45f3a7a72431a75d27f60376a17d9dff5a27d830cf4dcafa73861002e770a9"
INFERENCE_SHA256 = "edf3a7f74c62af176508e62d5b8639539915d26768d8aecd9ee954e92fe24a85"


def weight_identity(root, meta):
    name = meta.get("weights")
    if name not in ("weights.pt", "upstream.ckpt"):
        raise ValueError("Unsupported TM-Vec weights filename")
    actual = digest_file(root / name)
    if actual != meta.get("weights_sha256"):
        raise ValueError("TM-Vec weights checksum mismatch")
    if name == "upstream.ckpt":
        # Load the encoder parameters from the upstream checkpoint.
        if actual != UPSTREAM_SHA256:
            raise ValueError("Unsupported upstream TM-Vec checkpoint")
        return INFERENCE_SHA256
    return actual


class TMVecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        keys = ("d_model", "nhead", "dim_feedforward", "dropout", "activation")
        layer = nn.TransformerEncoderLayer(batch_first=True, **{k: config[k] for k in keys})
        self.encoder = nn.TransformerEncoder(layer, num_layers=config["num_layers"])
        self.dropout = nn.Dropout(config["dropout"])
        self.mlp = nn.Linear(config["d_model"], config["out_dim"])

    def forward(self, residues):
        mask = torch.zeros(residues.shape[:2], dtype=torch.bool, device=residues.device)
        hidden = self.encoder(residues, src_key_padding_mask=mask)
        lengths = torch.logical_not(mask).sum(dim=1).float()
        return self.mlp(self.dropout(hidden.sum(dim=1) / lengths.unsqueeze(1)))


def model_identity(directory, prott5):
    root, t5 = Path(directory).expanduser().resolve(), Path(prott5).expanduser().resolve()
    meta = read_json(root / "model.json")
    if (meta.get("schema_version") != 1 or meta.get("model") != "tm-vec-cath-large"
            or meta.get("weights") not in ("weights.pt", "upstream.ckpt")):
        raise ValueError("Invalid TM-Vec inference model bundle")
    cfg = meta["architecture"]
    if (cfg.get("d_model") != 1024 or cfg.get("out_dim") != 512 or cfg.get("num_layers") != 4
            or cfg.get("nhead") != 4 or cfg.get("dim_feedforward") != 2048
            or cfg.get("activation") != "relu" or cfg.get("dropout") != .1):
        raise ValueError("Unsupported TM-Vec architecture")
    names = ("config.json", "pytorch_model.bin", "spiece.model", "tokenizer_config.json", "special_tokens_map.json")
    hashes = {name: digest_file(t5 / name) for name in names if (t5 / name).is_file()}
    if not set(names[:4]) <= set(hashes):
        raise ValueError("Incomplete local ProtT5 model")
    return {"tmvec": weight_identity(root, meta), "architecture": cfg, "prott5": hashes}


def place_protein_encoder(protein, device):
    """Run every ProtT5 layer on CUDA while keeping inactive weights on CPU.

    Only weight residency changes. Activations, attention and normalization stay
    on the requested GPU in FP32; no sequence truncation or CPU inference fallback.
    """
    if device != "cuda":
        return protein.to(device)
    protein.shared.to(device)
    protein.encoder.final_layer_norm.to(device)

    def before_layer(module, inputs):
        module.to(device)

    def after_layer(module, inputs, output):
        module.to("cpu")

    for layer in protein.encoder.block:
        layer.register_forward_pre_hook(before_layer)
        layer.register_forward_hook(after_layer)
    return protein


def full_length_vectors(sequences, directory, prott5, *, device="cpu", progress=print):
    """Encode full-length proteins individually using local ProtT5 and TM-Vec models."""
    from transformers import T5EncoderModel, T5Tokenizer
    root = Path(directory).expanduser().resolve()
    meta = read_json(root / "model.json")
    weight_identity(root, meta)
    structural = TMVecEncoder(meta["architecture"])
    weights = torch.load(root / meta["weights"], map_location="cpu", weights_only=True)
    if meta["weights"] == "upstream.ckpt":
        weights = weights["state_dict"]
    structural.load_state_dict(weights, strict=True)
    del weights
    structural.eval().to(device)
    tokenizer = T5Tokenizer.from_pretrained(str(prott5), local_files_only=True)
    protein = T5EncoderModel.from_pretrained(str(prott5), local_files_only=True).float().eval()
    protein = place_protein_encoder(protein, device)
    values = []
    with torch.inference_mode():
        for index, original in enumerate(sequences):
            sequence = re.sub("[UZOBJ]", "X", normalize_protein(original))
            tokens = tokenizer(" ".join(sequence), add_special_tokens=True, return_tensors="pt")
            tokens = {k: v.to(device) for k, v in tokens.items()}
            hidden = protein(**tokens).last_hidden_state
            if int(tokens["attention_mask"].sum()) != len(sequence)+1 or hidden.shape[1] != len(sequence)+1:
                raise ValueError("Unexpected ProtT5 residue/token correspondence")
            vector = structural(hidden[:, :len(sequence)].float())[0].cpu().numpy()
            # Match the published full-length feature-store precision before normalization.
            vector = vector.astype(np.float16).astype(np.float64)
            norm = np.linalg.norm(vector)
            if vector.shape != (512,) or not np.isfinite(vector).all() or norm == 0:
                raise ValueError("Invalid TM-Vec protein representation")
            values.append((vector/norm).astype(np.float32))
            progress(f"Full-length TM-Vec: {index+1}/{len(sequences)} proteins")
    return np.asarray(values, np.float32).reshape(-1, 512)
