"""Frozen ESM2 inference and sequence-addressed, sharded embedding storage."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from argparse import Namespace
from contextlib import nullcontext
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from phanet.common.io import finite_array, read_json, write_json, unique_index, digest_file
from phanet.datasets.schema import sequence_hash
from phanet.datasets.data import normalize_protein


@dataclass(frozen=True)
class ESMConfig:
    model: str = "esm2_t33_650M_UR50D"
    max_residues: int = 1024
    token_budget: int = 4096
    batch_size: int = 8
    device: str = "cpu"
    shard_size: int = 4096
    allow_download: bool = False
    full_length: bool = False
    checkpoint: str | None = None


def validate_full_length(data, store):
    """Check the ESM2 representation and native protein identities."""
    meta = store.metadata
    if (meta.get("smoke") is not False or store.dimension != 1280 or
            meta.get("model") != "esm2_t33_650M_UR50D" or
            meta.get("config", {}).get("full_length") is not True or meta.get("dtype") != "float16"):
        raise ValueError("Requires full-length ESM2 layer-33 mean embeddings")
    hashes = meta["sequence_hashes"]
    for p in data.proteins:
        if p.protein_id not in store.index or hashes[store.index[p.protein_id]] != p.sha256:
            raise ValueError(f"Missing or mismatched embedding; embedding belongs to another sequence: {p.protein_id}")


def _esm_input_sequence(sequence: str, max_residues: int) -> str:
    """Return the model-facing sequence without changing its stored identity."""
    return normalize_protein(sequence)[:max_residues].replace("J", "X")


class EmbeddingStore:
    def __init__(self, directory: str | Path, *, sequence_contract: str | Path | None = None):
        self.directory = Path(directory)
        self.metadata = read_json(self.directory / "embeddings.json")
        if self.metadata.get("schema_version") != 1:
            raise ValueError("Unknown embedding store schema")
        self.ids = list(self.metadata.get("ids", []))
        self.index = unique_index(self.ids)
        self.dimension = int(self.metadata["dimension"])
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        if len(self.metadata.get("sequence_hashes", [])) != len(self.ids):
            raise ValueError("Embedding IDs and sequence hashes disagree")
        self._shards = []
        self._ends = []
        count = 0
        seen_files = set()
        for shard in self.metadata.get("shards", []):
            try:
                relative = Path(shard["file"])
                rows = int(shard["rows"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Malformed embedding shard metadata") from exc
            if (relative.is_absolute() or ".." in relative.parts or str(relative) in seen_files or
                    rows < 1 or relative.name != str(relative)):
                raise ValueError("Invalid or duplicate embedding shard path")
            seen_files.add(str(relative))
            path = self.directory / relative
            if digest_file(path) != shard["sha256"]:
                raise ValueError("Embedding shard integrity failure")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (rows, self.dimension):
                raise ValueError("Embedding shard shape mismatch")
            self._shards.append(array)
            count += len(array)
            self._ends.append(count)
        if count != len(self.ids) or not self._shards:
            raise ValueError("Embedding row, ID and sequence counts disagree")
        self._sequence_overrides = {}
        if sequence_contract is not None:
            contract = read_json(sequence_contract)
            if (contract.get("schema_version") != 1 or contract.get("status") != "complete" or
                    contract.get("embedding_metadata_sha256") != digest_file(self.directory / "embeddings.json") or
                    contract.get("input_rule") != "normalize_protein(sequence)[:1024]"):
                raise ValueError("Invalid or mismatched embedding sequence contract")
            for row in contract["overrides"]:
                pid = row["protein_id"]
                original = row["original_sequence_sha256"]
                if (pid not in self.index or pid in self._sequence_overrides or
                        not isinstance(original, str) or len(original) != 64 or
                        any(c not in "0123456789abcdef" for c in original) or
                        row["input_sequence_sha256"] != self.metadata["sequence_hashes"][self.index[pid]]):
                    raise ValueError("Invalid sequence contract row")
                self._sequence_overrides[pid] = original
            # In-memory provenance participates in checkpoint identity, without rewriting the store.
            self.metadata = dict(self.metadata, sequence_contract_sha256=digest_file(sequence_contract))

    def get(self, ids: Sequence[str], *, sequences: Sequence[str] | None = None) -> np.ndarray:
        indices = [self.index[identifier] for identifier in ids]
        if sequences is not None:
            if len(sequences) != len(ids):
                raise ValueError("Sequence and ID counts disagree")
            for index, sequence in zip(indices, sequences):
                pid = self.ids[index]
                if pid in self._sequence_overrides:
                    if (sequence_hash(sequence) != self._sequence_overrides[pid] or
                            len(normalize_protein(sequence)) <= 1024 or
                            sequence_hash(normalize_protein(sequence)[:1024]) != self.metadata["sequence_hashes"][index]):
                        raise ValueError(f"Embedding full/input sequence contract mismatch: {pid}")
                    continue
                if self.metadata["sequence_hashes"][index] != sequence_hash(sequence):
                    raise ValueError(f"Embedding belongs to another sequence: {self.ids[index]}")
        result = np.empty((len(indices), self.dimension), dtype=np.float32)
        if not indices:
            return result
        # Gather vectors by shard while preserving requested order and duplicate IDs.
        absolute = np.asarray(indices, dtype=np.int64)
        shard_ids = np.searchsorted(np.asarray(self._ends, dtype=np.int64), absolute, side="right")
        starts = np.asarray([0, *self._ends[:-1]], dtype=np.int64)
        for shard, array in enumerate(self._shards):
            positions = np.flatnonzero(shard_ids == shard)
            if len(positions):
                offsets = absolute[positions] - starts[shard]
                result[positions] = np.asarray(array[offsets], dtype=np.float32)
        return finite_array(result)

    @classmethod
    def write(cls, directory: str | Path, ids: list[str], sequences: list[str], vectors,
              *, model: str, config: dict | None = None, smoke=False, shard_size=4096):
        unique_index(ids)
        if len(ids) != len(sequences) or shard_size < 1:
            raise ValueError("Invalid embedding write specification")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if (directory / "embeddings.json").exists() or any(directory.glob("vectors-*.npy")):
            raise FileExistsError("Embedding stores are immutable; use a fresh directory")
        shards, buffer, dimension, total = [], [], None, 0
        def flush():
            nonlocal buffer
            if not buffer:
                return
            array = finite_array(np.stack(buffer)).astype(np.float16)
            finite_array(array)
            filename = f"vectors-{len(shards):05d}.npy"
            np.save(directory / filename, array, allow_pickle=False)
            shards.append({"file": filename, "rows": len(array), "sha256": digest_file(directory / filename)})
            buffer = []
        for vector in vectors:
            vector = finite_array(vector)
            if vector.ndim != 1 or not len(vector):
                raise ValueError("Every protein embedding must be a nonempty vector")
            if dimension is None:
                dimension = len(vector)
            if len(vector) != dimension:
                raise ValueError("Inconsistent embedding dimension")
            # Iterators may reuse one mutable array for successive vectors.
            buffer.append(vector.copy())
            total += 1
            if len(buffer) == shard_size:
                flush()
        flush()
        if total != len(ids) or dimension is None:
            raise ValueError("Embedding vector count must match a nonempty ID list")
        write_json(directory / "embeddings.json", {
            "schema_version": 1, "ids": ids, "sequence_hashes": [sequence_hash(s) for s in sequences],
            "dimension": dimension, "model": model, "config": config or {}, "smoke": smoke,
            "shards": shards, "dtype": "float16",
        })
        return cls(directory)


def esm_vectors(sequences: list[str], config: ESMConfig):
    import torch
    import esm
    if config.max_residues < 1 or config.max_residues > 1024 or min(config.batch_size, config.token_budget) < 1:
        raise ValueError("Invalid ESM length or batching configuration")
    if not config.model.startswith("esm2_") or not hasattr(esm.pretrained, config.model):
        raise ValueError("Expected a named ESM2 checkpoint")
    local_checkpoint = config.checkpoint or os.environ.get("PHANET_ESM2_MODEL_PATH")
    # Allow ESM2 checkpoint configuration objects within the scoped loader context.
    safe_globals = getattr(torch.serialization, "safe_globals", None)
    with safe_globals([Namespace]) if safe_globals is not None else nullcontext():
        if local_checkpoint:
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(local_checkpoint)
        else:
            if not config.allow_download:
                raise FileNotFoundError(
                    "Set PHANET_ESM2_MODEL_PATH to a local checkpoint or explicitly "
                    "enable ESMConfig(allow_download=True)"
                )
            loader = getattr(esm.pretrained, config.model)
            model, alphabet = loader()
    model.eval().to(config.device)
    model.requires_grad_(False)
    converter = alphabet.get_batch_converter()
    start = 0
    with torch.inference_mode():
        while start < len(sequences):
            batch = []
            longest = 0
            for index in range(start, min(start + config.batch_size, len(sequences))):
                # ESM-2 has no token for the ambiguous I/L residue ``J``.
                # Keep the canonical sequence unchanged for IDs and hashes, but
                # present it to the model as the supported unknown token ``X``.
                sequence = _esm_input_sequence(sequences[index], len(sequences[index]) if config.full_length else config.max_residues)
                next_longest = max(longest, len(sequence) + 2)
                if next_longest * (len(batch) + 1) > config.token_budget:
                    if not batch:
                        if not config.full_length:
                            raise ValueError("Token budget is too small for one truncated protein")
                        # Process proteins exceeding the token budget individually at full length.
                        batch.append((str(index), sequence))
                    break
                batch.append((str(index), sequence))
                longest = next_longest
            _, _, tokens = converter(batch)
            output = model(tokens.to(config.device), repr_layers=[model.num_layers], return_contacts=False)
            states = output["representations"][model.num_layers]
            for row, (_, sequence) in enumerate(batch):
                # ESM2 places BOS at index 0. EOS and padding are excluded.
                yield states[row, 1:len(sequence) + 1].float().mean(dim=0).cpu().numpy()
            start += len(batch)


def embed_esm(ids: list[str], sequences: list[str], output: str | Path,
              config: ESMConfig | None = None, *, smoke=False) -> EmbeddingStore:
    config = config or ESMConfig()
    unique_index(ids)
    sequences = [normalize_protein(s) for s in sequences]
    return EmbeddingStore.write(output, ids, sequences, esm_vectors(sequences, config),
                                model=config.model, config=asdict(config), smoke=smoke,
                                shard_size=config.shard_size)
