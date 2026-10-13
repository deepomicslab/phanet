"""Genome and protein records with 0-based, half-open coordinates."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from phanet.common.constants import CATEGORIES, CATEGORY_INDEX
from phanet.common.io import read_table, unique_index, write_table, write_json, read_json


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@dataclass
class Genome:
    genome_id: str
    accession: str
    length_nt: int
    source: str
    complete: bool = False
    cluster_id: str = ""
    quality: str = "unknown"
    taxonomy: str = ""
    host: str = ""
    gc: float | None = None
    sequence: str = ""
    release_date: str = ""
    quality_flags: tuple[str, ...] = ()
    excluded_regions: tuple[tuple[int, int, str], ...] = ()
    protein_content_cluster: str = ""
    # Metadata for stratification; excluded from model inputs and targets.
    lifestyle: str = ""


@dataclass
class Protein:
    protein_id: str
    genome_id: str
    start: int
    end: int
    strand: int
    order: int
    sequence: str
    source_protein_id: str = ""
    product: str = ""
    labels: tuple[str, ...] = ()
    evidence_source: str = ""
    evidence_version: str = ""
    partial: bool = False
    quality_flags: tuple[str, ...] = ()
    blocks: tuple[tuple[int, int], ...] = ()
    family_ids: tuple[str, ...] = ()
    coding_score: float | None = None
    frame: int = 0

    @property
    def annotation_status(self) -> str:
        return "annotated" if self.labels else "unresolved"

    @property
    def sha256(self) -> str:
        return sequence_hash(self.sequence)


@dataclass
class Dataset:
    genomes: list[Genome]
    proteins: list[Protein]
    provenance: dict = field(default_factory=dict)

    def validate(self) -> None:
        unique_index([g.genome_id for g in self.genomes])
        unique_index([p.protein_id for p in self.proteins])
        genomes = {g.genome_id: g for g in self.genomes}
        orders: dict[str, set[int]] = {}
        for g in self.genomes:
            if g.length_nt < 1 or (g.sequence and len(g.sequence) != g.length_nt):
                raise ValueError(f"Invalid genome length: {g.genome_id}")
            if g.gc is not None and not 0 <= g.gc <= 1:
                raise ValueError(f"Invalid GC fraction: {g.genome_id}")
            if g.sequence and set(g.sequence) - set("ACGTURYSWKMBDHVN"):
                raise ValueError(f"Invalid nucleotide sequence: {g.genome_id}")
            for start, end, reason in g.excluded_regions:
                if not 0 <= start < end <= g.length_nt or not reason:
                    raise ValueError(f"Invalid quality exclusion interval: {g.genome_id}")
        for p in self.proteins:
            if p.genome_id not in genomes:
                raise ValueError(f"Unknown parent: {p.protein_id}")
            if not 0 <= p.start < p.end <= genomes[p.genome_id].length_nt:
                raise ValueError(f"Invalid coordinates: {p.protein_id}")
            if p.strand not in (-1, 1) or p.order < 0:
                raise ValueError(f"Invalid strand/order: {p.protein_id}")
            if p.frame not in (0, 1, 2) or p.coding_score is not None and not np.isfinite(p.coding_score):
                raise ValueError(f"Invalid coding score/frame: {p.protein_id}")
            if p.order in orders.setdefault(p.genome_id, set()):
                raise ValueError(f"Duplicate gene order in {p.genome_id}")
            orders[p.genome_id].add(p.order)
            if not p.sequence or set(p.sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
                raise ValueError(f"Invalid protein sequence: {p.protein_id}")
            if len(set(p.labels)) != len(p.labels) or set(p.labels) - set(CATEGORIES):
                raise ValueError(f"Invalid labels: {p.protein_id}")
            for start, end in p.blocks:
                if not p.start <= start < end <= p.end:
                    raise ValueError(f"Invalid CDS blocks: {p.protein_id}")
        for genome_id, proteins in self.by_genome().items():
            if [p.order for p in proteins] != list(range(len(proteins))):
                raise ValueError(f"Non-contiguous native order: {genome_id}")
            if [p.start for p in proteins] != sorted(p.start for p in proteins):
                raise ValueError(f"Order disagrees with genomic coordinates: {genome_id}")

    def by_genome(self) -> dict[str, list[Protein]]:
        groups = {g.genome_id: [] for g in self.genomes}
        for protein in self.proteins:
            groups[protein.genome_id].append(protein)
        return {k: sorted(v, key=lambda p: p.order) for k, v in groups.items()}

    def label_matrix(self) -> np.ndarray:
        y = np.zeros((len(self.proteins), len(CATEGORIES)), dtype=np.float32)
        for i, protein in enumerate(self.proteins):
            for label in protein.labels:
                y[i, CATEGORY_INDEX[label]] = 1
        return y

    def save(self, directory: str | Path) -> None:
        self.validate()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        genome_rows = []
        for genome in self.genomes:
            row = asdict(genome)
            row.pop("sequence")
            row["complete"] = int(genome.complete)
            row["gc"] = "" if genome.gc is None else genome.gc
            for key in ("quality_flags", "excluded_regions"):
                row[key] = json.dumps(row[key], separators=(",", ":"))
            genome_rows.append(row)
        genome_fields = [k for k in Genome.__dataclass_fields__ if k != "sequence"]
        write_table(directory / "genomes.tsv", genome_rows, genome_fields)
        fields = [k for k in Protein.__dataclass_fields__ if k not in
                  ("labels", "evidence_source", "evidence_version", "sequence", "product")]
        def protein_rows():
            for p in self.proteins:
                row = {k: getattr(p, k) for k in fields}
                for k in ("quality_flags", "blocks", "family_ids"):
                    row[k] = json.dumps(row[k], separators=(",", ":"))
                row["partial"] = int(p.partial)
                row["coding_score"] = "" if p.coding_score is None else p.coding_score
                row["sequence_sha256"] = p.sha256
                yield row
        write_table(directory / "proteins.tsv", protein_rows(), fields + ["sequence_sha256"])
        write_table(directory / "annotations.tsv", (
                    {"protein_id": p.protein_id, "labels": ";".join(p.labels),
                     "product": p.product, "source": p.evidence_source, "version": p.evidence_version}
                    for p in self.proteins),
                    ["protein_id", "labels", "product", "source", "version"])
        with (directory / "proteins.faa").open("w") as f:
            for p in self.proteins:
                f.write(f">{p.protein_id}\n{p.sequence}\n")
        with (directory / "genomes.fna").open("w") as f:
            for g in self.genomes:
                if g.sequence:
                    f.write(f">{g.genome_id}\n{g.sequence}\n")
        write_json(directory / "dataset.json", {"schema_version": 1, "coordinate_system": "0-based-half-open",
                   "categories": CATEGORIES, "provenance": self.provenance})

    @classmethod
    def load(cls, directory: str | Path) -> "Dataset":
        from Bio import SeqIO
        directory = Path(directory)
        metadata = read_json(directory / "dataset.json")
        if metadata.get("coordinate_system") != "0-based-half-open" or metadata.get("categories") != list(CATEGORIES):
            raise ValueError("Unsupported coordinate system or ontology")
        def read_fasta(path):
            sequences = {}
            for record in SeqIO.parse(path, "fasta"):
                if record.id in sequences:
                    raise ValueError(f"Duplicate FASTA ID: {record.id}")
                sequences[record.id] = str(record.seq)
            return sequences
        sequences = read_fasta(directory / "proteins.faa")
        nucleotides = read_fasta(directory / "genomes.fna")
        genomes = []
        for row in read_table(directory / "genomes.tsv", ("genome_id", "length_nt")):
            row["length_nt"] = int(row["length_nt"])
            row["complete"] = bool(int(row["complete"]))
            row["gc"] = float(row["gc"]) if row["gc"] else None
            row["sequence"] = nucleotides.get(row["genome_id"], "")
            row["quality_flags"] = tuple(json.loads(row.get("quality_flags", "[]")))
            row["excluded_regions"] = tuple(tuple(v) for v in json.loads(row.get("excluded_regions", "[]")))
            genomes.append(Genome(**row))
        annotations = {}
        for row in read_table(directory / "annotations.tsv", ("protein_id", "labels")):
            if row["protein_id"] in annotations:
                raise ValueError(f"Duplicate annotation ID: {row['protein_id']}")
            annotations[row["protein_id"]] = row
        proteins = []
        for row in read_table(directory / "proteins.tsv", ("protein_id", "sequence_sha256")):
            identifier = row["protein_id"]
            if identifier not in sequences:
                raise ValueError(f"Protein metadata references a missing FASTA sequence: {identifier}")
            sequence = sequences[identifier]
            if sequence_hash(sequence) != row.pop("sequence_sha256"):
                raise ValueError(f"Sequence hash mismatch: {identifier}")
            row["sequence"] = sequence
            for k in ("start", "end", "strand", "order"):
                row[k] = int(row[k])
            row["partial"] = bool(int(row["partial"]))
            row["coding_score"] = float(row["coding_score"]) if row.get("coding_score") else None
            row["frame"] = int(row.get("frame", 0))
            for k in ("quality_flags", "family_ids"):
                row[k] = tuple(json.loads(row[k]))
            row["blocks"] = tuple(tuple(v) for v in json.loads(row["blocks"]))
            if identifier not in annotations:
                raise ValueError(f"Protein metadata lacks annotation row: {identifier}")
            evidence = annotations[identifier]
            row.update(labels=tuple(filter(None, evidence["labels"].split(";"))), product=evidence["product"],
                       evidence_source=evidence["source"], evidence_version=evidence["version"])
            proteins.append(Protein(**row))
        if set(sequences) != {p.protein_id for p in proteins} or set(annotations) != set(sequences):
            raise ValueError("Protein, sequence and annotation IDs do not match")
        result = cls(genomes, proteins, metadata["provenance"])
        result.validate()
        return result
