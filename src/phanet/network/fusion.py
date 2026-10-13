"""Continuous score fusion and validation-only multi-label threshold selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from phanet.common.constants import CATEGORIES, CATEGORY_THRESHOLDS
from phanet.common.io import finite_array, digest_object, write_json, read_json


def fuse(network, context, reliability) -> np.ndarray:
    network = finite_array(network, unit_interval=True)
    context = finite_array(context, shape=network.shape, unit_interval=True)
    if network.ndim != 2 or network.shape[1] != len(CATEGORIES):
        raise ValueError(f"Expected N x {len(CATEGORIES)} category scores")
    rho = finite_array(reliability, shape=(len(network),), unit_interval=True)
    return rho[:, None] * network + (1 - rho[:, None]) * context


def metrics(truth, scores, thresholds) -> dict:
    scores = finite_array(scores, unit_interval=True)
    truth = finite_array(truth, shape=scores.shape, unit_interval=True)
    if truth.ndim != 2 or not np.isin(truth, [0, 1]).all():
        raise ValueError("Multi-label truth must be a binary matrix")
    threshold = finite_array(thresholds, shape=(truth.shape[1],), unit_interval=True)
    if len(scores) == 0:
        return {"n": 0, "micro_f1": None, "macro_f1": None, "auprc_micro": None,
                "auprc_macro": None, "coverage": None, "precision_assigned": None, "summary_score": None}
    assigned = scores >= threshold
    coverage = float(assigned.any(axis=1).mean())
    precision = float((assigned * truth).sum() / assigned.sum()) if assigned.sum() else 0.
    auc = [float(average_precision_score(truth[:, c], scores[:, c]))
           for c in range(truth.shape[1]) if truth[:, c].sum() > 0]
    return {"n": len(scores), "micro_f1": float(f1_score(truth, assigned, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(truth, assigned, average="macro", zero_division=0)),
            "auprc_micro": float(average_precision_score(truth.ravel(), scores.ravel())) if truth.sum() else None,
            "auprc_macro": float(np.mean(auc)) if auc else None,
            "coverage": coverage, "precision_assigned": precision,
            "summary_score": 2 * coverage * precision / (coverage + precision) if coverage + precision else 0.}


@dataclass
class Calibration:
    thresholds: list[float]
    validation_ids_hash: str
    model_hash: str
    split_hash: str
    report: dict
    kind: str = "validation_selected"
    pipeline_hash: str = ""

    def save(self, path: str | Path):
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path, *, model_hash: str, split_hash: str, pipeline_hash: str | None = None):
        value = cls(**read_json(path))
        if value.model_hash != model_hash or value.split_hash != split_hash:
            raise ValueError("Calibration is not bound to this model and partition")
        if pipeline_hash is not None and value.pipeline_hash != pipeline_hash:
            raise ValueError("Calibration score pipeline differs (context, network or fusion evidence changed)")
        finite_array(value.thresholds, shape=(len(CATEGORIES),), unit_interval=True)
        return value


def calibrate(truth, scores, ids: list[str], *, partitions: list[str], model_hash: str,
              split_hash: str, max_passes=20) -> Calibration:
    """Select thresholds by coordinate ascent over observed validation scores.

    Optimize micro F1, macro F1, coverage, then proximity to the default.
    Categories with no validation positives retain the 0.5 threshold."""
    truth = finite_array(truth, unit_interval=True)
    scores = finite_array(scores, shape=truth.shape, unit_interval=True)
    if truth.ndim != 2 or truth.shape[1] != len(CATEGORIES) or len(ids) != len(scores) or len(partitions) != len(ids):
        raise ValueError("Calibration input shape mismatch")
    if not ids or len(ids) != len(set(ids)) or set(partitions) != {"val"}:
        raise ValueError("Calibration requires unique validation-only protein IDs")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("Calibration labels must be binary")
    if max_passes < 1:
        raise ValueError("Calibration needs at least one coordinate pass")
    scores = scores.astype(np.float64)
    thresholds = np.array(CATEGORY_THRESHOLDS)
    unavailable = [CATEGORIES[c] for c in range(len(CATEGORIES)) if not truth[:, c].any()]
    # Sort once per category. Prefix counts evaluate all exact breakpoints in
    # linear space, without repeatedly recomputing precision-recall curves.
    orders = [np.argsort(scores[:, c], kind="stable") for c in range(len(CATEGORIES))]
    def objectives(c, candidates):
        assigned = scores >= thresholds
        tp = (assigned * truth).sum(axis=0)
        predicted = assigned.sum(axis=0)
        positives = truth.sum(axis=0)
        other = np.arange(len(CATEGORIES)) != c
        order = orders[c]
        cut = np.searchsorted(scores[order, c], candidates, side="left")
        prefix = np.r_[0., np.cumsum(truth[order, c])]
        ctp = positives[c] - prefix[cut]
        cpred = len(scores) - cut
        numerator = 2 * (tp[other].sum() + ctp)
        denominator = predicted[other].sum() + cpred + positives.sum()
        micro = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
        denom = predicted[other] + positives[other]
        other_f1 = np.divide(2 * tp[other], denom, out=np.zeros_like(denom, dtype=float), where=denom > 0).sum()
        denom = cpred + positives[c]
        macro = (other_f1 + np.divide(2 * ctp, denom, out=np.zeros_like(ctp), where=denom > 0)) / len(CATEGORIES)
        already = assigned[:, other].any(axis=1)
        uncovered_prefix = np.r_[0, np.cumsum(~already[order])]
        coverage = (already.sum() + uncovered_prefix[-1] - uncovered_prefix[cut]) / len(scores)
        proximity = -np.abs(candidates - CATEGORY_THRESHOLDS[c])
        return list(zip(micro, macro, coverage, proximity))
    passes = 0
    for passes in range(1, max_passes + 1):
        changed = False
        for c in range(len(CATEGORIES)):
            if not truth[:, c].any():
                continue
            candidates = np.unique(np.r_[0., 1., scores[:, c], np.nextafter(scores[:, c], 1.), thresholds[c]])
            values = objectives(c, candidates)
            best_index = max(range(len(candidates)), key=lambda i: values[i])
            current_index = int(np.searchsorted(candidates, thresholds[c]))
            chosen = candidates[best_index] if values[best_index] > values[current_index] else thresholds[c]
            if chosen != thresholds[c]:
                changed = True
                thresholds[c] = chosen
        if not changed:
            break
    return Calibration(thresholds.tolist(), digest_object(sorted(ids)), model_hash, split_hash,
                       {"metrics": metrics(truth, scores, thresholds), "uncalibrated_categories": unavailable,
                        "coordinate_passes": passes, "converged": not changed})
