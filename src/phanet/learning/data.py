"""Prepare and load GLM training data with genome-level partitions."""
from collections import Counter
from pathlib import Path
import os

from phanet.common.io import digest_file, read_json, write_json, read_table
from phanet.datasets.importing import load_input
from phanet.datasets.schema import Dataset
from phanet.datasets.splits import SplitManifest, MaskPolicy, dataset_identity, grouped_stratify
from phanet.learning.context import ContextDataset
from phanet.representations.embed import ESMConfig, EmbeddingStore, embed_esm, validate_full_length


def _validate_splits(data, splits):
    splits.validate(data)
    if splits.task != "annotation" or set(splits.assignments.values()) != {"train", "val", "test"}:
        raise ValueError("Training data requires nonempty train, val and test partitions")
    groups = {}
    for genome in data.genomes:
        partition = splits.assignments[genome.genome_id]
        if genome.cluster_id:
            groups.setdefault(("cluster", genome.cluster_id), set()).add(partition)
        if genome.sequence:
            from Bio.Seq import Seq
            sequence = min(genome.sequence, str(Seq(genome.sequence).reverse_complement()))
            groups.setdefault(("sequence", sequence), set()).add(partition)
    if any(len(parts) > 1 for parts in groups.values()):
        raise ValueError("Identical genomes or genome clusters cross training partitions")
    counts = Counter(splits.assignments[p.genome_id] for p in data.proteins)
    if any(not counts[p] for p in ("train", "val", "test")):
        raise ValueError("Each partition must contain proteins")


def prepare_training(output, *, embeddings=None, esm_checkpoint=None, splits=None,
                     ratios=(.8, .1, .1), unit="genome", seed=7, max_context=30,
                     device="cpu", batch_size=1, family_hits=None, profiles=None, threads=4,
                     progress=print, **inputs):
    if max_context < 8 or batch_size < 1 or threads < 1:
        raise ValueError("max_context must be at least 8 and batch_size must be positive")
    if (embeddings is None) == (esm_checkpoint is None):
        raise ValueError("Supply either an embedding store or an ESM2 checkpoint")
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    data = load_input(work=out, progress=progress, **inputs)
    if family_hits and profiles:
        raise ValueError("Choose a family-hit table or a profile library")
    if profiles:
        import pyhmmer
        from phanet.network.search import profile_search
        with pyhmmer.plan7.HMMFile(str(profiles)) as stream:
            count = sum(1 for _ in stream)
        if count < 1: raise ValueError('Profile library is empty')
        hits = profile_search({p.protein_id:p.sequence for p in data.proteins}, profiles,
            out / 'family-search', threads=threads, profile_count=count, profile_Z=None, progress=progress)
        data.provenance['family_profiles_sha256'] = digest_file(profiles)
    elif family_hits:
        hits = read_table(family_hits, ('protein_id','family_id','evalue','coverage'))
        data.provenance['family_hits_sha256'] = digest_file(family_hits)
    else:
        hits = None
    if hits is not None:
        import math
        found = {p.protein_id:set() for p in data.proteins}
        for hit in hits:
            pid, family = hit['protein_id'], hit['family_id']
            ev, cov = float(hit['evalue']), float(hit['coverage'])
            if pid not in found or not family or not math.isfinite(ev) or not math.isfinite(cov) or ev < 0 or not 0 <= cov <= 1:
                raise ValueError('Invalid protein family hit')
            if ev <= 1e-5 and cov >= .5:
                found[pid].add(family)
        for p in data.proteins:
            p.family_ids = tuple(sorted(found[p.protein_id]))
    if splits is None:
        if len(ratios) != 3:
            raise ValueError("Supply train, val and test ratios")
        assignments = grouped_stratify(data, dict(zip(("train", "val", "test"), ratios)), unit=unit, seed=seed)
        partition = SplitManifest("annotation", unit, assignments, seed, dataset_fingerprint=dataset_identity(data))
    else:
        partition = SplitManifest.load(splits)
    _validate_splits(data, partition)
    data.provenance = dict(data.provenance, context_windows={"max_context": max_context, "overlap": 0})
    data.save(out / "dataset")
    partition.save(out / "splits.json")
    if embeddings is None:
        progress("Encoding training proteins with ESM2")
        store = embed_esm([p.protein_id for p in data.proteins], [p.sequence for p in data.proteins],
                          out / "embeddings", ESMConfig(checkpoint=str(esm_checkpoint), device=device,
                                                       batch_size=batch_size, full_length=True))
    else:
        store = EmbeddingStore(embeddings)
    validate_full_length(data, store)
    for lo in range(0, len(data.proteins), 1024):
        rows = data.proteins[lo:lo+1024]
        store.get([p.protein_id for p in rows], sequences=[p.sequence for p in rows])
    windows = {part: len(ContextDataset(data, store, partition, MaskPolicy(), partition=part,
                                        mode="segments", max_context=max_context))
               for part in ("train", "val", "test")}
    files = [out / "splits.json", *sorted((out / "dataset").iterdir())]
    manifest = {"schema_version": 1, "dataset": "dataset", "splits": "splits.json",
                "embeddings": os.path.relpath(store.directory.resolve(), out),
                "embedding_metadata_sha256": digest_file(store.directory / "embeddings.json"),
                "files": {str(p.relative_to(out)): digest_file(p) for p in files},
                "max_context": max_context, "sampling_seed": seed, "embedding_dimension": store.dimension,
                "genomes": len(data.genomes), "proteins": len(data.proteins), "windows": windows}
    write_json(out / "training.json", manifest)
    return manifest


def load_training(directory, partition="train", *, mode="segments"):
    """Return a PyTorch dataset of context windows for train, val or test."""
    root = Path(directory).expanduser().resolve()
    meta = read_json(root / "training.json")
    if meta.get("schema_version") != 1 or partition not in ("train", "val", "test"):
        raise ValueError("Invalid training manifest or partition")
    if meta.get("dataset") != "dataset" or meta.get("splits") != "splits.json":
        raise ValueError("Invalid training data paths")
    expected = {"splits.json", *(str(p.relative_to(root)) for p in (root / "dataset").iterdir())}
    if set(meta["files"]) != expected:
        raise ValueError("Incomplete training data manifest")
    for name, checksum in meta["files"].items():
        if digest_file(root / name) != checksum:
            raise ValueError(f"Training data checksum mismatch: {name}")
    path = root / meta["embeddings"]
    if digest_file(path / "embeddings.json") != meta["embedding_metadata_sha256"]:
        raise ValueError("Training embedding metadata changed")
    store = EmbeddingStore(path)
    data = Dataset.load(root / "dataset")
    validate_full_length(data, store)
    splits = SplitManifest.load(root / "splits.json")
    _validate_splits(data, splits)
    return ContextDataset(data, store, splits, MaskPolicy(), partition=partition, mode=mode,
                          max_context=meta["max_context"], sampling_seed=meta["sampling_seed"])
