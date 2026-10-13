"""Sliding-window member-set prediction over native genomes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from phanet.downstream.module_decode import compact_candidates, members_from_bits


@dataclass(frozen=True)
class PredictionConfig:
    object_threshold: float = .75
    member_threshold: float = .5
    window_support: float = .75
    batch_size: int = 16
    max_candidates_per_genome: int = 100000

    def __post_init__(self):
        for value in (self.object_threshold, self.member_threshold):
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError("Prediction thresholds must lie in (0, 1)")
        if not np.isfinite(self.window_support) or not 0 <= self.window_support <= 1:
            raise ValueError("Invalid window support")
        if any(type(v) is not int or v < 1 for v in
               (self.batch_size, self.max_candidates_per_genome)):
            raise ValueError("Positive integer inference budgets required")


def covering_windows(length, members, width=30):
    lo = max(0, members[-1] - width + 1)
    hi = min(members[0], max(0, length - width))
    return max(0, hi - lo + 1)


def predict_genome(model, vectors, vector_rows, strands, config=PredictionConfig(), *, device="cpu"):
    """Scan every native start, consolidate exact sets and count one vote/window."""
    rows = np.asarray(vector_rows)
    strand = np.asarray(strands)
    width = model.head.config.width
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or not len(rows)
            or vectors.ndim != 2 or vectors.shape[1] != model.head.config.input_dim
            or np.any(rows < 0) or np.any(rows >= len(vectors))
            or strand.shape != rows.shape or not np.isin(strand, (-1, 1)).all()):
        raise ValueError("Invalid genome/vector mapping")
    if width != 30:
        raise ValueError("Only the accepted 30-position scan is supported")
    candidates = {}
    count = max(1, len(rows) - width + 1)
    model.eval()
    with torch.inference_mode():
        for first in range(0, count, config.batch_size):
            starts = np.arange(first, min(first + config.batch_size, count))
            offsets = starts[:, None] + np.arange(width)[None]
            valid = offsets < len(rows)
            offsets = np.minimum(offsets, len(rows) - 1)
            ids, inverse = np.unique(rows[offsets], return_inverse=True)
            unique = np.asarray(vectors[ids], dtype=np.float32)
            if not np.isfinite(unique).all():
                raise ValueError("Nonfinite protein embeddings")
            x = torch.as_tensor(unique, device=device)[torch.as_tensor(inverse, device=device)]
            x = x.reshape(len(starts), width, -1)
            v = torch.as_tensor(valid, device=device)
            x = x.masked_fill(~v[..., None], 0)
            s = torch.as_tensor(np.where(valid, strand[offsets], 0), dtype=torch.long, device=device)
            outputs = model(x, s, v)
            if any(not torch.isfinite(outputs[k]).all() for k in ("object_logits", "mask_logits", "type_logits")):
                raise ValueError("Nonfinite model predictions")
            compact = compact_candidates(outputs, config.member_threshold, config.object_threshold)
            for bi, start in enumerate(starts):
                per_window = {}
                for _, anchor, bits, score, types in compact[compact[:, 0] == bi]:
                    members = members_from_bits(int(start + anchor), int(bits))
                    if members[-1] >= len(rows):
                        raise ValueError("Prediction outside native genome")
                    old = per_window.get(members, (0., 0))
                    per_window[members] = (max(old[0], float(score)), old[1] | int(types))
                for members, (score, types) in per_window.items():
                    previous = candidates.setdefault(members, [0, 0., 0])
                    previous[0] += 1
                    previous[1] = max(previous[1], score)
                    previous[2] |= types
                if len(candidates) > config.max_candidates_per_genome:
                    raise RuntimeError("Candidate budget exceeded; no silent truncation")
    selected = []
    for members, (support, score, types) in sorted(candidates.items()):
        eligible = covering_windows(len(rows), members, width)
        if not 0 < support <= eligible:
            raise ValueError("Inconsistent covering-window support")
        if support / eligible >= config.window_support:
            selected.append(dict(members=list(members), score=score, supporting_windows=support,
                                 covering_windows=eligible, window_support=support / eligible,
                                 auxiliary_type_bits=types))
    return selected, dict(windows=count, distinct_before_support=len(candidates), retained=len(selected))
