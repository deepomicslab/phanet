"""Strict identifiers and numeric inputs shared by evidence-only downstream tasks."""
from __future__ import annotations

import numpy as np


def identifier(value):
    if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value.strip()):
        raise ValueError("Identifiers must be nonempty strings without whitespace")
    return value.strip()


def identifiers(values):
    result = [identifier(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError("Identifiers must be unique")
    return result


def number(value, name, low=0., high=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if not np.isfinite(value) or value < low or (high is not None and value > high):
        raise ValueError(f"Invalid {name}: outside finite range")
    return value


def feature_matrix(values, ids):
    ids = identifiers(ids)
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) != len(ids) or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a finite matrix aligned to unique IDs")
    return x, ids
