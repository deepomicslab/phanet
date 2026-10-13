"""Load task-specific inference model bundles."""
from pathlib import Path
import re

import numpy as np
import torch
from torch import nn

from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json
from phanet.datasets.splits import MASKED, PADDING
from phanet.learning.model import ModelConfig, PhaNetModel
from .module_sets import ModuleSetConfig, GenomicModuleSetModel

EMBEDDING = "esm2_t33_650M_UR50D:layer33:full_length:mean:float16"


class GOModel(nn.Module):
    """Independently fine-tuned encoder and ontology head."""

    def __init__(self, encoder, width):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.config.hidden_dim, width)

    def forward(self, batch, offsets):
        x, annotation, state, strand, valid = batch
        rows = torch.arange(len(offsets), device=x.device)
        columns = torch.as_tensor(offsets, device=x.device)
        if (not valid[rows, columns].all() or
                not (state[rows, columns] == MASKED).all() or
                (annotation[rows, columns] != 0).any()):
            raise ValueError("GO targets must be valid and annotation-masked")
        states = self.encoder(x, annotation, state, strand, valid, member_mask=None)["states"]
        return self.head(states[rows, columns])


class BinaryContextModel(GOModel):
    """Binary classifier over masked native protein contexts."""

    def __init__(self, encoder):
        super().__init__(encoder, 1)
        # Preserved for exact checkpoint compatibility. The validation-selected
        # decision threshold is stored separately in model.json.
        self.register_buffer("thresholds", torch.full((1,), .5, dtype=torch.float32))

    def forward(self, batch, offsets):
        _, annotation, state, _, valid = batch
        if annotation.any() or not (state[valid] == MASKED).all() or any(i != 14 for i in offsets):
            raise ValueError("Binary context requires fully hidden annotations and target position 14")
        return super().forward(batch, offsets)


def metadata(directory, task):
    root = Path(directory).expanduser().resolve()
    meta = read_json(root / "model.json")
    if not isinstance(meta, dict) or not isinstance(meta.get("model_config"), dict):
        raise ValueError("Missing downstream architecture configuration")
    if (meta.get("schema_version") != 1 or meta.get("task") != task or
            meta.get("categories") != list(CATEGORIES) or meta.get("embedding") != EMBEDDING):
        raise ValueError("Incompatible downstream model contract")
    if meta.get("weights") != "weights.pt" or digest_file(root / "weights.pt") != meta.get("weights_sha256"):
        raise ValueError("Downstream weights path or checksum mismatch")
    config = ModelConfig(**meta["model_config"])
    config.validate()
    if config.input_dim != 1280 or config.max_context != 30:
        raise ValueError("Expected full-length ESM2 and 30-position context")
    if task == "module":
        if not isinstance(meta.get("head_config"), dict) or not isinstance(meta.get("prediction"), dict):
            raise ValueError("Missing module head or prediction configuration")
        head = ModuleSetConfig(**meta["head_config"])
        if head.width != 30 or head.input_dim != 1280 or config.hidden_dim != head.input_dim:
            raise ValueError("Incompatible module architecture")
        if meta.get("network_mode") != "off" or meta.get("auxiliary_type_count") != 7:
            raise ValueError("Expected network-off module model with seven auxiliary types")
        from .module_prediction import PredictionConfig
        PredictionConfig(**meta["prediction"])
    elif task == "go":
        terms = meta.get("terms")
        if (meta.get("namespace") not in ("BP", "CC", "MF") or not isinstance(terms, list) or not terms or
                any(not isinstance(t, str) or not re.fullmatch(r"GO:\d{7}", t) for t in terms) or
                len(set(terms)) != len(terms)):
            raise ValueError("Invalid ordered GO vocabulary")
        threshold = meta.get("threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                not np.isfinite(threshold) or not 0 < threshold < 1 or
                meta.get("annotation_mode") != "neighbors_target_masked"):
            raise ValueError("Invalid GO decision or annotation policy")
    elif task == "vf":
        threshold = meta.get("threshold")
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or
                not np.isfinite(threshold) or not 0 < threshold < 1 or
                meta.get("annotation_mode") != "all_masked" or
                meta.get("context_mode") != "fixed_center_padded" or meta.get("target_offset") != 14 or
                meta.get("threshold_source") != "validation"):
            raise ValueError("Invalid binary context or validation decision policy")
    else:
        raise ValueError("Unknown downstream task")
    return root, meta


def load(directory, task, *, device="cpu"):
    root, meta = metadata(directory, task)
    encoder = PhaNetModel(ModelConfig(**meta["model_config"]))
    if task == "module":
        model = GenomicModuleSetModel(encoder, ModuleSetConfig(**meta["head_config"]),
                                     masked_state=MASKED, padding_state=PADDING)
    elif task == "go":
        model = GOModel(encoder, len(meta["terms"]))
    else:
        model = BinaryContextModel(encoder)
    weights = torch.load(str(root / "weights.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite downstream model weights")
    return model.requires_grad_(False).eval().to(device), meta
