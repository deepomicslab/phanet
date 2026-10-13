"""Task-specific group separation and explicit information-access policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from phanet.common.constants import CATEGORIES, CATEGORY_INDEX
from phanet.common.io import digest_object, read_json, write_json
from phanet.datasets.schema import Dataset

UNRESOLVED, VISIBLE, MASKED, PADDING = 0, 1, 2, 3


@dataclass
class SplitManifest:
    task: str
    unit: str
    assignments: dict[str, str]
    seed: int
    fold: int | None = None
    dataset_fingerprint: str = ""

    @property
    def fingerprint(self):
        return digest_object(asdict(self))

    def validate(self, dataset: Dataset) -> None:
        if set(self.assignments) != {g.genome_id for g in dataset.genomes}:
            raise ValueError("Split manifest must assign every genome exactly once")
        if self.unit not in ("genome", "cluster"):
            raise ValueError("Split unit must be genome or cluster")
        allowed = {"annotation": {"train", "val", "test"}, "known_modules": {"train", "val", "test"},
                   "dark_modules": {"mining", "confirm"}}
        if self.task not in allowed or not set(self.assignments.values()) <= allowed[self.task]:
            raise ValueError("Unknown task or partition names in split manifest")
        if self.unit == "cluster":
            membership = {}
            for g in dataset.genomes:
                if not g.cluster_id:
                    raise ValueError(f"Cluster-level split requires a cluster ID: {g.genome_id}")
                membership.setdefault(g.cluster_id, set()).add(self.assignments[g.genome_id])
            if any(len(v) != 1 for v in membership.values()):
                raise ValueError("Genome-cluster leakage across partitions")
        actual = dataset_identity(dataset)
        if self.dataset_fingerprint and self.dataset_fingerprint != actual:
            raise ValueError("Split manifest was created for different genomic records")

    def save(self, path: str | Path):
        write_json(path, asdict(self))

    @classmethod
    def load(cls, path: str | Path):
        return cls(**read_json(path))


def dataset_identity(dataset: Dataset) -> str:
    return digest_object({"genomes": sorted((g.genome_id, g.cluster_id, g.length_nt) for g in dataset.genomes),
                          "proteins": sorted((p.protein_id, p.genome_id, p.sha256, p.start, p.end, p.order, p.strand)
                                             for p in dataset.proteins)})


def grouped_stratify(dataset: Dataset, ratios: dict[str, float], *, unit="genome", seed=7) -> dict[str, str]:
    """Assign indivisible groups by normalized protein-label and genome-count deficits."""
    if any(v <= 0 for v in ratios.values()) or not np.isclose(sum(ratios.values()), 1):
        raise ValueError("Positive split ratios must sum to one")
    if unit not in ("genome", "cluster"):
        raise ValueError("Unknown grouping unit")
    groups = {}
    for g in dataset.genomes:
        key = g.genome_id if unit == "genome" else g.cluster_id
        if not key:
            raise ValueError("Cluster IDs must be supplied before splitting")
        groups.setdefault(key, []).append(g.genome_id)
    if len(groups) < len(ratios):
        raise ValueError("Not enough independent groups for nonempty partitions")
    by_genome = dataset.by_genome()
    vectors = {}
    for key, genomes in groups.items():
        vector = np.zeros(len(CATEGORIES) + 1)
        vector[-1] = len(genomes)
        for gid in genomes:
            for p in by_genome[gid]:
                for label in p.labels:
                    vector[CATEGORY_INDEX[label]] += 1
        vectors[key] = vector
    names = list(ratios)
    fractions = np.array(list(ratios.values()))
    total = sum(vectors.values())
    desired = fractions[:, None] * total[None, :]
    scale = np.maximum(total, 1)
    used = np.zeros_like(desired)
    rng = np.random.default_rng(seed)
    tie = dict(zip(sorted(groups), rng.random(len(groups))))
    order = sorted(groups, key=lambda k: (-float((vectors[k] / scale).sum()), tie[k], k))
    assignments, counts = {}, np.zeros(len(names), dtype=int)
    for position, key in enumerate(order):
        remaining = len(order) - position
        empty = np.flatnonzero(counts == 0)
        candidates = empty if remaining == len(empty) else np.arange(len(names))
        costs = []
        for j in candidates:
            before = ((used[j] - desired[j]) / scale) ** 2
            after = ((used[j] + vectors[key] - desired[j]) / scale) ** 2
            costs.append((float((after - before).sum()), counts[j] / fractions[j], int(j)))
        chosen = min(costs)[2]
        used[chosen] += vectors[key]
        counts[chosen] += 1
        for genome in groups[key]:
            assignments[genome] = names[chosen]
    return assignments


def make_splits(dataset: Dataset, task: str, *, seed=7, fold=0) -> SplitManifest:
    if task == "annotation":
        if fold not in range(5):
            raise ValueError("Annotation fold must be between zero and four")
        outer = grouped_stratify(dataset, {str(i): .2 for i in range(5)}, seed=seed)
        development_genomes = [g for g in dataset.genomes if outer[g.genome_id] != str(fold)]
        ids = {g.genome_id for g in development_genomes}
        development = Dataset(development_genomes, [p for p in dataset.proteins if p.genome_id in ids])
        inner = grouped_stratify(development, {"train": .85, "val": .15}, seed=seed + fold + 1)
        assignments = {g.genome_id: inner.get(g.genome_id, "test") for g in dataset.genomes}
        result = SplitManifest(task, "genome", assignments, seed, fold, dataset_identity(dataset))
    elif task in ("known_modules", "dark_modules"):
        ratios = {"train": .70, "val": .15, "test": .15} if task == "known_modules" else {"mining": .70, "confirm": .30}
        # Dark-discovery splitting uses group identity.
        partition_data = dataset
        if task == "dark_modules":
            partition_data = Dataset(dataset.genomes, [])
        assignments = grouped_stratify(partition_data, ratios, unit="cluster", seed=seed)
        result = SplitManifest(task, "cluster", assignments, seed, None, dataset_identity(dataset))
    else:
        raise ValueError(f"Unsupported split task: {task}")
    result.validate(dataset)
    return result


@dataclass
class MaskPolicy:
    """Select annotations available to model inputs for each partition."""
    reference_partitions: tuple[str, ...] = ("train",)
    visible_partitions: tuple[str, ...] = ("train",)
    masked_proteins: frozenset[str] = field(default_factory=frozenset)
    masked_families: frozenset[str] = field(default_factory=frozenset)
    context_visible_proteins: frozenset[str] = field(default_factory=frozenset)
    label_blind: bool = False

    def apply(self, dataset: Dataset, splits: SplitManifest) -> dict[str, np.ndarray]:
        splits.validate(dataset)
        n = len(dataset.proteins)
        observed = np.zeros((n, len(CATEGORIES)), dtype=np.float32)
        states = np.full(n, UNRESOLVED, dtype=np.int64)
        references = np.zeros(n, dtype=bool)
        expansion = np.zeros(n, dtype=bool)
        targets = np.zeros(n, dtype=bool)
        # Apply the annotation-availability mask to training and evaluation labels.
        eligible_mask = np.zeros(n, dtype=bool)
        true_labels = dataset.label_matrix()
        for i, protein in enumerate(dataset.proteins):
            partition = splits.assignments[protein.genome_id]
            excluded = (protein.protein_id in self.masked_proteins or
                        bool(set(protein.family_ids) & self.masked_families))
            annotated = bool(protein.labels)
            visible = (partition in self.visible_partitions or
                       protein.protein_id in self.context_visible_proteins)
            if annotated:
                states[i] = VISIBLE if visible and not excluded and not self.label_blind else MASKED
                if states[i] == VISIBLE:
                    observed[i] = true_labels[i]
            references[i] = annotated and partition in self.reference_partitions and not excluded and not self.label_blind
            expansion[i] = partition in self.reference_partitions and not excluded and not self.label_blind
            targets[i] = annotated and partition == "train" and not excluded
            eligible_mask[i] = annotated and not excluded and not self.label_blind
        return {"annotation_input": observed, "state": states, "seed_mask": references,
                "expansion_mask": expansion, "loss_eligible": targets,
                "eligible_mask": eligible_mask}


def pretraining_subset(dataset: Dataset, splits: SplitManifest, partition="train") -> Dataset:
    """Physically exclude other records, including unlabeled held-out context."""
    splits.validate(dataset)
    genomes = [g for g in dataset.genomes if splits.assignments[g.genome_id] == partition]
    ids = {g.genome_id for g in genomes}
    return Dataset(genomes, [p for p in dataset.proteins if p.genome_id in ids], dataset.provenance)


def audit_development(dataset: Dataset, training: SplitManifest, *, excluded_genomes=(), excluded_hashes=()) -> dict:
    training.validate(dataset)
    forbidden = set(excluded_genomes)
    hashes = set(excluded_hashes)
    violations = [p.protein_id for p in dataset.proteins if training.assignments[p.genome_id] == "train"
                  and (p.genome_id in forbidden or p.sha256 in hashes)]
    if violations:
        raise ValueError(f"Development/evaluation overlap: {violations[:8]}")
    return {"status": "passed", "split_hash": training.fingerprint,
            "excluded_genomes": len(forbidden), "excluded_sequences": len(hashes)}
