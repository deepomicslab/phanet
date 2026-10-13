"""Validated, atomic, provenance-bearing stage artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np


def digest_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_object(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_table(path: str | Path, required: tuple[str, ...] = ()) -> Iterator[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = set(required) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for line, row in enumerate(reader, 2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"{path}:{line}: inconsistent column count")
            yield row


def write_table(path: str | Path, rows, fields: list[str] | tuple[str, ...]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def unique_index(ids: list[str]) -> dict[str, int]:
    result = {}
    for i, identifier in enumerate(ids):
        if not identifier or any(c.isspace() for c in identifier) or identifier in result:
            raise ValueError(f"Empty or duplicate ID: {identifier!r}")
        result[identifier] = i
    return result


def finite_array(value, *, shape=None, unit_interval=False) -> np.ndarray:
    array = np.asarray(value)
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Non-finite numeric input")
    if unit_interval and ((array < 0).any() or (array > 1).any()):
        raise ValueError("Expected values in [0, 1]")
    return array


def _stage_metadata(directory: Path, *, stage: str, inputs: dict[str, str | Path],
                    config: dict, smoke: bool = False) -> dict:
    return {
        "schema_version": 1, "stage": stage, "smoke": smoke,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "inputs": {k: {"path": str(Path(v).resolve()), "sha256": digest_file(v)}
                   for k, v in inputs.items()},
        "outputs": {str(p.relative_to(directory)): digest_file(p) for p in sorted(directory.rglob("*"))
                    if p.is_file() and p.name != "manifest.json"
                    and not any(part.startswith("._") for part in p.relative_to(directory).parts)},
        "status": "complete",
    }


def finalize_stage(directory: str | Path, *, stage: str, inputs: dict[str, str | Path],
                   config: dict, smoke: bool = False) -> None:
    """Write a completion manifest for a persistent, resumable stage."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    write_json(directory / "manifest.json", _stage_metadata(directory, stage=stage,
              inputs=inputs, config=config, smoke=smoke))


@contextmanager
def stage_output(destination: str | Path, *, stage: str, inputs: dict[str, str | Path],
                 config: dict, smoke: bool = False, preserve_on_error: bool = False):
    """Write a stage to a new directory and finalize its completion manifest."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output exists; choose a fresh directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name + ".partial-", dir=destination.parent))
    try:
        yield temporary
        write_json(temporary / "manifest.json", _stage_metadata(temporary, stage=stage,
                   inputs=inputs, config=config, smoke=smoke))
        os.replace(temporary, destination)
    except BaseException:
        if preserve_on_error and temporary.exists() and not destination.exists():
            # Retain incomplete output so training can resume from its last checkpoint.
            os.replace(temporary, destination)
        else:
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def validate_manifest(directory: str | Path) -> dict:
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError(f"Incomplete artifact: {directory}")
    for relative, expected in manifest["outputs"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or digest_file(path) != expected:
            raise ValueError(f"Artifact integrity failure: {relative}")
    return manifest
