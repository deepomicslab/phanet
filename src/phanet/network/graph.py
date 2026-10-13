"""Three-channel protein relationships with explicit evidence provenance."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import chain, combinations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import heapq
import hashlib
import os
import shutil
import struct
import tempfile

import numpy as np
from scipy import sparse
from Bio import SeqIO

from phanet.datasets.data import run_tool
from phanet.common.io import (finite_array, unique_index, write_json, read_json, read_table,
                               write_table, digest_object, digest_file)

CHANNELS = ("sequence", "profile", "structure")
MAX_NUMPY_PAIRWISE = 10_000_000
MMSEQS_FIELDS = ("query", "target", "fident", "alnlen", "qstart", "qend", "qlen",
                 "tstart", "tend", "tlen", "evalue", "bits", "qcov", "tcov")


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    channel: str
    score: float
    # Store channel evidence alongside each edge.
    raw_score: float | None = None
    threshold: float | None = None
    provenance: str = ""


class _DiskEdgeCollection:
    """Read graph evidence rows lazily from a table."""
    def __init__(self, path: str | Path, count: int | None = None):
        self.path = Path(path)
        self.count = count

    def __len__(self):
        if self.count is not None:
            return int(self.count)
        return sum(1 for _ in read_table(self.path))

    def __iter__(self):
        for row in read_table(self.path, ("source", "target", "channel", "score")):
            raw = _optional_edge_float(row.get("raw_score"), "raw_score")
            threshold = _optional_edge_float(row.get("threshold"), "threshold")
            yield Edge(row["source"], row["target"], row["channel"], float(row["score"]),
                       raw_score=raw, threshold=threshold,
                       provenance=row.get("provenance", ""))

    def __getitem__(self, item):
        """Index or slice evidence rows."""
        if isinstance(item, slice):
            # Slices are uncommon for graph consumers; iterate once and keep
            # only the requested bounded result instead of materializing E.
            start, stop, step = item.indices(len(self))
            if step == 0:  # pragma: no cover - ``slice.indices`` normally raises
                raise ValueError("slice step cannot be zero")
            if step < 0:
                # Reverse slices materialize the edge table.
                return list(self)[item]
            selected = []
            for index, edge in enumerate(self):
                if index < start:
                    continue
                if index >= stop:
                    break
                if (index - start) % step == 0:
                    selected.append(edge)
            return selected
        if not isinstance(item, (int, np.integer)):
            raise TypeError("edge indices must be integers or slices")
        index = int(item)
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError("edge index out of range")
        for current, edge in enumerate(self):
            if current == index:
                return edge
        raise IndexError("edge index out of range")


def _optional_edge_float(value, field):
    """Parse an optional numeric edge field, rejecting explicit NaN/inf."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Malformed graph edge {field}") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"Non-finite graph edge {field}")
    return parsed


def _partition_ids(query_ids, reference_ids) -> tuple[list[str], list[str]]:
    """Validate and normalize a disjoint query/reference partition."""
    queries, references = list(query_ids), list(reference_ids)
    unique_index(queries)
    unique_index(references)
    overlap = set(queries) & set(references)
    if overlap:
        raise ValueError(f"Query and reference IDs overlap: {sorted(overlap)[:5]}")
    return queries, references


def _fasta_ids(path: str | Path) -> list[str]:
    ids = [record.id for record in SeqIO.parse(str(path), "fasta")]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"FASTA must contain unique, non-empty IDs: {path}")
    return ids


def _partition_filter(edges, query_ids, reference_ids, *, include_reference_reference=True,
                      reject_query_query=True):
    """Select reference-reference and query-reference edges for a partitioned graph."""
    queries, references = _partition_ids(query_ids, reference_ids)
    qset, rset = set(queries), set(references)
    result = []
    for edge in edges:
        endpoints = {edge.source, edge.target}
        if not endpoints <= qset | rset:
            raise ValueError("Edge references an ID outside query/reference partitions")
        if edge.source in qset and edge.target in qset:
            if reject_query_query:
                raise ValueError("Query-query edges are forbidden in a fresh retrieval graph")
            continue
        if edge.source in rset and edge.target in rset:
            if include_reference_reference:
                result.append(edge)
            continue
        result.append(edge)
    return result


def sequence_edges(rows, *, evalue=1e-5, min_bits=50., minimum_score=.30,
                   query_ids=None, reference_ids=None, self_bits=None) -> list[Edge]:
    """Normalize sequence edges by self-alignment scores.

    With query_ids and reference_ids, emit query-reference edges. Self scores
    come from the alignment table or the supplied self_bits mapping."""
    if (not np.isfinite(evalue) or evalue <= 0 or not np.isfinite(min_bits)
            or min_bits < 0 or not 0 <= minimum_score <= 1):
        raise ValueError("Invalid sequence-channel thresholds")
    partitioned = query_ids is not None or reference_ids is not None
    if partitioned and (query_ids is None or reference_ids is None):
        raise ValueError("Both query_ids and reference_ids are required")
    if partitioned:
        queries, references = _partition_ids(query_ids, reference_ids)
        qset, rset = set(queries), set(references)
    else:
        qset = rset = None
    denominators = {identifier: float(value) for identifier, value in (self_bits or {}).items()}
    if partitioned and not set(denominators) <= qset | rset:
        raise ValueError("Self-alignment scores reference an unknown partition ID")
    for identifier, value in denominators.items():
        if not np.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid self-alignment bit score for {identifier}")
    # Keep only a compact tuple for candidate rows.  MMseqs2 can report up to
    # thousands of neighbors per protein; retaining the original dictionaries
    # doubles peak memory without adding information needed for normalization.
    candidates = []
    for row in rows:
        try:
            left, right = row["query"], row["target"]
            bits = float(row["bits"])
            significance = float(row["evalue"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed MMseqs2 alignment row") from exc
        if not np.isfinite(significance) or significance < 0:
            raise ValueError("Invalid alignment E-value")
        if not np.isfinite(bits) or bits < 0:
            raise ValueError("Invalid alignment bit score")
        if left == right:
            denominators[left] = max(bits, denominators.get(left, 0))
        if partitioned and not (left in qset and right in rset):
            continue
        if left == right or significance > evalue or bits < min_bits:
            continue
        candidates.append((left, right, bits))
    result = []
    for left, right, bits in candidates:
        if min(denominators.get(left, 0), denominators.get(right, 0)) <= 0:
            raise ValueError(f"Missing true self-alignment scores for {left}, {right}")
        score = bits / max(denominators[left], denominators[right])
        finite_array([score], unit_interval=True)
        if score >= minimum_score:
            result.append(Edge(left, right, "sequence", score, raw_score=bits,
                               threshold=minimum_score, provenance="mmseqs2"))
    return result


def sequence_query_reference_edges(rows, query_ids, reference_ids, *, self_bits=None,
                                   evalue=1e-5, min_bits=50., minimum_score=.30) -> list[Edge]:
    """Explicit query-to-reference sequence channel convenience wrapper."""
    return sequence_edges(rows, evalue=evalue, min_bits=min_bits,
                          minimum_score=minimum_score, query_ids=query_ids,
                          reference_ids=reference_ids, self_bits=self_bits)


def _run_mmseqs_search(query_fasta: str | Path, target_fasta: str | Path,
                       raw: Path, tmp: Path, *, threads: int, executable: str,
                       max_sequences: int, timeout: int | None, add_self: bool,
                       evalue: float = 1e-5):
    argv = [executable, "easy-search", str(Path(query_fasta).resolve()),
            str(Path(target_fasta).resolve()), str(raw), str(tmp), "--threads", str(threads),
            "-e", str(evalue), "-a", "1", "--max-seqs", str(max_sequences),
            "--format-output", ",".join(MMSEQS_FIELDS)]
    if add_self:
        insert_at = argv.index("-a")
        argv[insert_at:insert_at] = ["--add-self-matches", "1", "--alignment-mode", "3"]
    return run_tool(argv, cwd=raw.parent, timeout=timeout)


def _copy_mmseqs_rows(raw_paths: list[Path], result: Path) -> None:
    with result.open("w", encoding="utf-8") as target:
        target.write("\t".join(MMSEQS_FIELDS) + "\n")
        for raw in raw_paths:
            if not raw.exists():
                raise FileNotFoundError(f"MMseqs2 did not produce output: {raw}")
            with raw.open(encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        target.write(line)


def run_mmseqs(fasta: str | Path, output: str | Path, *positional,
               reference_fasta: str | Path | None = None, threads=2,
               executable="mmseqs", max_sequences=1000, timeout=None,
               evalue=1e-5) -> Path:
    """Run MMseqs2 and return alignment rows.

    The default searches all input proteins against each other. With
    reference_fasta, run query-reference alignment and separate self searches
    for normalization."""
    # Accept both query/reference and all-versus-all call signatures.
    if (threads < 1 or max_sequences < 1 or (timeout is not None and timeout <= 0)
            or not np.isfinite(evalue) or evalue <= 0):
        raise ValueError("Invalid MMseqs2 execution settings")
    if positional:
        if len(positional) != 1 or reference_fasta is not None:
            raise TypeError("Use run_mmseqs(fasta, output, reference_fasta=...) or the three-positional form")
        reference_fasta, output = output, positional[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reference = Path(reference_fasta) if reference_fasta is not None else Path(fasta)
    # Use PHANET_MMSEQS_TMPDIR or native Linux temporary storage for MMseqs2 files.
    work_root = Path(os.environ.get("PHANET_MMSEQS_TMPDIR", "/tmp")).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="phanet-mmseqs-", dir=work_root))
    try:
        if reference_fasta is None:
            raw = work / "alignments.raw.tsv"
            execution = _run_mmseqs_search(fasta, fasta, raw, work / "tmp", threads=threads,
                                            executable=executable, max_sequences=max_sequences,
                                            timeout=timeout, add_self=True, evalue=evalue)
            executions = {"all_vs_all": execution}
            raw_paths = [raw]
        else:
            queries, references = _fasta_ids(fasta), _fasta_ids(reference)
            _partition_ids(queries, references)
            write_json(output / "partitions.json", {"query_ids": queries, "reference_ids": references,
                                                     "query_query_edges": False})
            raw_paths = [work / "query_to_reference.raw.tsv",
                         work / "query_self.raw.tsv", work / "reference_self.raw.tsv"]
            executions = {
                "query_to_reference": _run_mmseqs_search(fasta, reference, raw_paths[0], work / "tmp_qr",
                                                           threads=threads, executable=executable,
                                                           max_sequences=max_sequences, timeout=timeout,
                                                           add_self=False, evalue=evalue),
                "query_self": _run_mmseqs_search(fasta, fasta, raw_paths[1], work / "tmp_q",
                                                  threads=threads, executable=executable,
                                                  max_sequences=max_sequences, timeout=timeout,
                                                  add_self=True, evalue=evalue),
                "reference_self": _run_mmseqs_search(reference, reference, raw_paths[2], work / "tmp_r",
                                                      threads=threads, executable=executable,
                                                      max_sequences=max_sequences, timeout=timeout,
                                                      add_self=True, evalue=evalue),
            }
        write_json(output / "execution.json", executions)
        result = output / "alignments.tsv"
        _copy_mmseqs_rows(raw_paths, result)
        if reference_fasta is not None:
            # Store query-reference rows separately; retain self alignments for normalization.
            _copy_mmseqs_rows([raw_paths[0]], output / "query_reference.tsv")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_mmseqs_query_reference(query_fasta: str | Path, reference_fasta: str | Path,
                               output: str | Path, **kwargs) -> Path:
    """Named query/reference entry point for deployment and workflow callers."""
    return run_mmseqs(query_fasta, output, reference_fasta=reference_fasta, **kwargs)


def parse_hmmer_domtbl(path: str | Path, *, mode="hmmsearch", evalue=1e-5,
                       minimum_coverage=.50) -> list[dict]:
    if mode not in ("hmmsearch", "hmmscan"):
        raise ValueError("HMMER mode must be explicit")
    if not np.isfinite(evalue) or evalue <= 0 or not 0 <= minimum_coverage <= 1:
        raise ValueError("Invalid HMMER thresholds")
    grouped = {}
    with Path(path).open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            columns = line.split(maxsplit=22)
            if len(columns) < 22:
                raise ValueError("Malformed HMMER domtblout row")
            # HMMER domtblout columns: full-sequence E-value/score (6/7),
            # conditional and independent domain E-values (11/12), domain
            # score (13), and mean posterior accuracy (21).
            numeric = (6, 7, 8, 11, 12, 13, 14, 21)
            if any(not np.isfinite(float(columns[i])) for i in numeric):
                raise ValueError("Nonfinite HMMER field")
            if any(float(columns[i]) < 0 for i in (6, 8, 11, 12, 14)):
                raise ValueError("Invalid HMMER E-value or bias")
            if not 0 <= float(columns[21]) <= 1:
                raise ValueError("Invalid HMMER posterior accuracy")
            if float(columns[6]) > evalue or float(columns[12]) > evalue:
                continue
            protein, family = (columns[0], columns[3]) if mode == "hmmsearch" else (columns[3], columns[0])
            length = int(columns[2] if mode == "hmmsearch" else columns[5])
            profile_length = int(columns[5] if mode == "hmmsearch" else columns[2])
            if length < 1 or profile_length < 1:
                raise ValueError("Invalid HMMER target/profile length")
            start, end = int(columns[17]) - 1, int(columns[18])
            if not 0 <= start < end <= length:
                raise ValueError("Invalid HMMER aligned sequence coordinates")
            env_start, env_end = int(columns[19]) - 1, int(columns[20])
            if not 0 <= env_start < env_end <= length:
                raise ValueError("Invalid HMMER envelope coordinates")
            entry = grouped.setdefault((protein, family), {
                "length": length, "profile_length": profile_length,
                "intervals": [], "envelopes": [],
                "full_evalue": float(columns[6]), "full_bits": float(columns[7]),
                "domain_evalue": float(columns[12]), "domain_bits": float(columns[13]),
                "conditional_domain_evalue": float(columns[11]),
                "num_domains": int(columns[9]), "posterior_accuracy": float(columns[21]),
            })
            if entry["length"] != length or entry["profile_length"] != profile_length:
                raise ValueError("Inconsistent HMMER target/profile length")
            entry["intervals"].append((start, end))
            entry["envelopes"].append((env_start, env_end))
            entry["full_evalue"] = min(entry["full_evalue"], float(columns[6]))
            entry["full_bits"] = max(entry["full_bits"], float(columns[7]))
            entry["domain_evalue"] = min(entry["domain_evalue"], float(columns[12]))
            entry["domain_bits"] = max(entry["domain_bits"], float(columns[13]))
            entry["conditional_domain_evalue"] = min(entry["conditional_domain_evalue"], float(columns[11]))
            entry["num_domains"] = max(entry["num_domains"], int(columns[9]))
            entry["posterior_accuracy"] = max(entry["posterior_accuracy"], float(columns[21]))
    hits = []
    for (protein, family), entry in sorted(grouped.items()):
        merged = []
        for start, end in sorted(entry["intervals"]):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(end, merged[-1][1])
            else:
                merged.append([start, end])
        coverage = sum(b - a for a, b in merged) / entry["length"]
        if coverage >= minimum_coverage:
            hits.append({"protein_id": protein, "family_id": family, "coverage": coverage,
                         # Retain aggregate and domain-level E-values.
                         "evalue": entry["full_evalue"],
                         "full_evalue": entry["full_evalue"],
                         "full_bits": entry["full_bits"],
                         "domain_evalue": entry["domain_evalue"],
                         "conditional_domain_evalue": entry["conditional_domain_evalue"],
                         "domain_bits": entry["domain_bits"],
                         "target_length": entry["length"],
                         "profile_length": entry["profile_length"],
                         "alignment_intervals": merged,
                         "num_domains": entry["num_domains"],
                         "posterior_accuracy": entry["posterior_accuracy"]})
    return hits


def profile_edges(hits, *, evalue=1e-5, minimum_coverage=.50,
                  query_ids=None, reference_ids=None,
                  include_reference_reference=True,
                  max_family_members=5000, top_k_per_family=50,
                  max_edges_per_family=100000, stats=None) -> list[Edge]:
    """Build protein relationships from shared HMMER profile hits and optional partitions."""
    if not np.isfinite(evalue) or evalue <= 0 or not 0 <= minimum_coverage <= 1:
        raise ValueError("Invalid profile-channel thresholds")
    if query_ids is not None or reference_ids is not None:
        if query_ids is None or reference_ids is None:
            raise ValueError("Both query_ids and reference_ids are required")
        queries, references = _partition_ids(query_ids, reference_ids)
        qset, rset = set(queries), set(references)
    else:
        qset = rset = None
    for name, value in (("max_family_members", max_family_members),
                        ("top_k_per_family", top_k_per_family),
                        ("max_edges_per_family", max_edges_per_family)):
        if value is not None and (not isinstance(value, (int, np.integer)) or value < 1):
            raise ValueError(f"{name} must be a positive integer or None")
    families = {}
    # Reuse the universe for every hit and for the optional clique branch.
    # Rebuilding it per hit costs O(hits * proteins) on partitioned corpora.
    universe = qset | rset if qset is not None else None
    for hit in hits:
        if universe is not None and hit.get("protein_id") not in universe:
            raise ValueError("Profile evidence references an ID outside query/reference partitions")
        coverage = float(hit["coverage"])
        finite_array([coverage], unit_interval=True)
        if not np.isfinite(float(hit["evalue"])) or float(hit["evalue"]) < 0:
            raise ValueError("Invalid profile E-value")
        if float(hit["evalue"]) > evalue or coverage < minimum_coverage:
            continue
        family = families.setdefault(hit["family_id"], {})
        family[hit["protein_id"]] = max(coverage, family.get(hit["protein_id"], 0))
    report = stats if stats is not None else {}
    report.update({"families": len(families), "max_family_size": max((len(v) for v in families.values()), default=0),
                   "families_truncated": 0, "families_member_truncated": 0,
                   "families_edge_capped": 0, "members_dropped": 0,
                   "candidate_edges": 0, "potential_edges": 0, "edges_retained": 0,
                   "max_family_members": max_family_members,
                   "top_k_per_family": top_k_per_family,
                   "max_edges_per_family": max_edges_per_family,
                   "strategy": "sparse-topk" if top_k_per_family is not None else "clique-capped"})
    edges = {}
    truncated_families = set()
    for family_id, members in sorted(families.items()):
        original_size = len(members)
        selected = members
        if max_family_members is not None and original_size > max_family_members:
            # In a fresh query/reference graph, retain all query members first
            # only when possible, while reserving at least one reference so a
            # giant family cannot be truncated into a query-only island.
            def priority(item):
                identifier, coverage = item
                query_priority = 0 if qset is not None and identifier in qset else 1
                return query_priority, -coverage, identifier
            if qset is not None:
                query_members = sorted(((i, c) for i, c in members.items() if i in qset),
                                       key=lambda v: (-v[1], v[0]))
                reference_members = sorted(((i, c) for i, c in members.items() if i in rset),
                                           key=lambda v: (-v[1], v[0]))
                if query_members and reference_members and max_family_members >= 2:
                    query_cap = min(len(query_members), max(1, int(round(
                        max_family_members * len(query_members) / original_size))))
                    query_cap = min(query_cap, max_family_members - 1)
                    reference_cap = min(len(reference_members), max_family_members - query_cap)
                    if reference_cap == 0:
                        reference_cap = 1
                        query_cap = max_family_members - 1
                    chosen = query_members[:query_cap] + reference_members[:reference_cap]
                else:
                    chosen = sorted(members.items(), key=priority)[:max_family_members]
            else:
                chosen = sorted(members.items(), key=priority)[:max_family_members]
            selected = dict(chosen)
            truncated_families.add(family_id)
            report["families_member_truncated"] += 1
            report["members_dropped"] += original_size - len(selected)
        candidates = []
        if qset is None:
            report["potential_edges"] += len(selected) * (len(selected) - 1) // 2
        else:
            query_count = sum(identifier in qset for identifier in selected)
            reference_count = len(selected) - query_count
            potential = query_count * reference_count
            if include_reference_reference:
                potential += reference_count * (reference_count - 1) // 2
            report["potential_edges"] += potential
        if top_k_per_family is not None:
            # Generate only the strongest k neighbors per member directly;
            # materializing a 5,000-member clique before trimming it would
            # otherwise create millions of temporary pairs.  The support is
            # ``min(coverage_left, coverage_right)``, so sorting members by
            # coverage gives the exact top-k order without an O(m^2) scan.
            ordered = sorted(selected, key=lambda identifier: (-selected[identifier], identifier))
            if qset is not None:
                query_order = [identifier for identifier in ordered if identifier in qset]
                reference_order = [identifier for identifier in ordered if identifier in rset]
            keep = set()
            for member in ordered:
                if qset is None:
                    neighbors = ordered[:top_k_per_family + 1]
                elif member in qset:
                    # Query nodes may only connect to references in a fresh
                    # retrieval graph; this avoids walking over a large block
                    # of forbidden query members.
                    neighbors = reference_order[:top_k_per_family]
                elif member in rset:
                    neighbors = (ordered[:top_k_per_family + 1]
                                 if include_reference_reference else query_order[:top_k_per_family])
                else:
                    raise ValueError("Profile evidence references an unknown partition ID")
                taken = 0
                for other in neighbors:
                    if member == other:
                        continue
                    keep.add(tuple(sorted((member, other))))
                    taken += 1
                    if taken == top_k_per_family:
                        break
            candidates = [(left, right, min(selected[left], selected[right]))
                          for left, right in sorted(keep)]
        else:
            for left, right in combinations(sorted(selected), 2):
                if qset is not None:
                    if left not in universe or right not in universe:
                        raise ValueError("Profile evidence references an unknown partition ID")
                    if left in qset and right in qset:
                        continue
                    if left in rset and right in rset and not include_reference_reference:
                        continue
                candidates.append((left, right, min(selected[left], selected[right])))
        report["candidate_edges"] += len(candidates)
        if max_edges_per_family is not None and len(candidates) > max_edges_per_family:
            candidates = sorted(candidates, key=lambda v: (-v[2], v[0], v[1]))[:max_edges_per_family]
            truncated_families.add(family_id)
            report["families_edge_capped"] += 1
        for left, right, score in candidates:
            edges[left, right] = max(edges.get((left, right), 0), score)
        report["edges_retained"] += len(candidates)
    report["families_truncated"] = len(truncated_families)
    provenance = "hmmer-profile;sparse-topk" if top_k_per_family is not None else "hmmer-profile"
    return [Edge(a, b, "profile", v, raw_score=v, threshold=minimum_coverage,
                 provenance=provenance) for (a, b), v in sorted(edges.items())]


def profile_query_reference_edges(query_hits, reference_hits=None, query_ids=None,
                                  reference_ids=None, *, evalue=1e-5,
                                  minimum_coverage=.50, **profile_kwargs) -> list[Edge]:
    """Create profile edges from independently scanned query/reference hits."""
    query_hits = list(query_hits)
    reference_hits = None if reference_hits is None else list(reference_hits)
    if query_ids is None:
        query_ids = sorted({h["protein_id"] for h in query_hits})
    if reference_ids is None:
        reference_ids = sorted({h["protein_id"] for h in (reference_hits or [])})
    queries, references = _partition_ids(query_ids, reference_ids)
    allowed = set(queries) | set(references)
    all_hits = [*query_hits, *(reference_hits or [])]
    if any(hit.get("protein_id") not in allowed for hit in all_hits):
        raise ValueError("Profile evidence references an ID outside query/reference partitions")
    if reference_hits is None:
        hits = query_hits
    else:
        hits = [*query_hits, *reference_hits]
    return profile_edges(hits, evalue=evalue, minimum_coverage=minimum_coverage,
                         query_ids=queries, reference_ids=references,
                         include_reference_reference=False, **profile_kwargs)


def run_hmmer(fasta: str | Path, profiles: str | Path, output: str | Path, *, threads=2,
              executable="hmmsearch", mode="hmmsearch", evalue=1e-5,
              minimum_coverage=.50, timeout=None) -> list[dict]:
    """Run HMMER against one protein universe and parse domain evidence."""
    if threads < 1:
        raise ValueError("HMMER threads must be positive")
    if not np.isfinite(evalue) or evalue <= 0 or not 0 <= minimum_coverage <= 1:
        raise ValueError("Invalid HMMER thresholds")
    if mode not in ("hmmsearch", "hmmscan"):
        raise ValueError("HMMER mode must be explicit")
    if mode == "hmmscan" and executable == "hmmsearch":
        executable = "hmmscan"
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    domtbl = output / "domains.tsv"
    # Align HMMER reporting thresholds with the parser filters.
    if timeout is not None and timeout <= 0:
        raise ValueError("HMMER timeout must be positive or None")
    execution = run_tool([executable, "--cpu", str(threads), "--noali", "-E", str(evalue),
                          "--domE", str(evalue), "--domtblout", str(domtbl),
                          str(Path(profiles).resolve()), str(Path(fasta).resolve())], cwd=output,
                         timeout=timeout)
    write_json(output / "execution.json", execution)
    return parse_hmmer_domtbl(domtbl, mode=mode, evalue=evalue,
                              minimum_coverage=minimum_coverage)


def run_hmmer_query_reference(query_fasta: str | Path, reference_fasta: str | Path,
                              profiles: str | Path, output: str | Path, *, threads=2,
                              executable="hmmsearch", mode="hmmsearch", evalue=1e-5,
                              minimum_coverage=.50, timeout=None) -> list[dict]:
    """Scan query and reference proteins independently for profile evidence."""
    if threads < 1:
        raise ValueError("HMMER threads must be positive")
    if not np.isfinite(evalue) or evalue <= 0 or not 0 <= minimum_coverage <= 1:
        raise ValueError("Invalid HMMER thresholds")
    if mode not in ("hmmsearch", "hmmscan"):
        raise ValueError("HMMER mode must be explicit")
    if mode == "hmmscan" and executable == "hmmsearch":
        executable = "hmmscan"
    if timeout is not None and timeout <= 0:
        raise ValueError("HMMER timeout must be positive or None")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    query_ids, reference_ids = _fasta_ids(query_fasta), _fasta_ids(reference_fasta)
    _partition_ids(query_ids, reference_ids)
    write_json(output / "partitions.json", {"query_ids": query_ids, "reference_ids": reference_ids,
                                             "query_query_edges": False})
    executions, hits = {}, []
    for name, fasta in (("query", query_fasta), ("reference", reference_fasta)):
        domtbl = output / f"{name}.domains.tsv"
        executions[name] = run_tool([executable, "--cpu", str(threads), "--noali", "-E", str(evalue),
                                     "--domE", str(evalue), "--domtblout", str(domtbl),
                                     str(Path(profiles).resolve()), str(Path(fasta).resolve())], cwd=output,
                                    timeout=timeout)
        hits.extend(parse_hmmer_domtbl(domtbl, mode=mode, evalue=evalue,
                                       minimum_coverage=minimum_coverage))
    write_json(output / "execution.json", executions)
    return hits


def _resolve_retrieval_backend(backend: str, pairwise: int) -> str:
    """Select a supported structure-search backend."""
    if backend == "auto":
        if pairwise <= MAX_NUMPY_PAIRWISE:
            return "numpy"
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Retrieval exceeds 10 million pairwise comparisons; install FAISS "
                "or pass a smaller graph"
            ) from exc
        return "faiss"
    if backend not in ("numpy", "faiss", "faiss-hnsw"):
        raise ValueError("Unknown nearest-neighbor backend")
    return backend


def nearest_neighbors(embeddings: np.ndarray, *, top_k=200, batch_size=128, backend="auto"):
    """Cosine retrieval with bounded query batches and exact retained-pair scores.

    numpy/FAISS Flat are exact references; faiss-hnsw is the scalable approximate
    option (M=32, efConstruction=200, efSearch=max(200, k+1), one build thread).
    """
    source = finite_array(embeddings)
    # Check float32 range before casting to prevent nonfinite cosine scores.
    if np.any(np.abs(source) > np.finfo(np.float32).max):
        raise ValueError("Embedding values overflow float32 representation")
    x = source.astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Embedding values overflow float32 representation")
    if (x.ndim != 2 or not isinstance(top_k, (int, np.integer)) or
            not isinstance(batch_size, (int, np.integer)) or top_k < 1 or batch_size < 1):
        raise ValueError("Expected a matrix and positive retrieval sizes")
    # Compute norms in float64 so large-but-valid float32 coordinates do not
    # overflow while squaring.  FAISS still receives compact float32 unit
    # vectors after this stable normalization.
    norms = np.linalg.norm(x.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise ValueError("Cannot retrieve from zero or non-finite norm vectors")
    # Keep normalized vectors and FAISS inputs in float32.
    x = (x / norms).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise ValueError("Embedding normalization produced non-finite values")
    n = len(x)
    backend = _resolve_retrieval_backend(backend, n * n)
    if backend == "numpy" and n * n > MAX_NUMPY_PAIRWISE:
        raise ValueError(
            "numpy structure retrieval is limited to 10 million pairwise comparisons; "
            "use backend='faiss' or 'faiss-hnsw' for large graphs"
        )
    index = None
    if backend in ("faiss", "faiss-hnsw"):
        import faiss
        if backend == "faiss-hnsw":
            faiss.omp_set_num_threads(1)
            index = faiss.IndexHNSWFlat(x.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = max(200, top_k + 1)
        else:
            index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
    for start in range(0, n, batch_size):
        if index is None:
            score_batch = x[start:start + batch_size] @ x.T
            candidates = np.broadcast_to(np.arange(n), score_batch.shape)
        else:
            # Extra tied neighbors may be omitted by FAISS; its native tie policy
            # is recorded by the calling stage. Sorting retained pairs is stable.
            score_batch, candidates = index.search(x[start:start + batch_size], min(top_k + 1, n))
        for offset, (scores, ids) in enumerate(zip(score_batch, candidates)):
            query = start + offset
            keep = (ids != query) & (ids >= 0)
            scores, ids = scores[keep], ids[keep]
            if index is not None:
                scores = x[ids] @ x[query]
            order = np.lexsort((ids, -scores))[:top_k]
            yield query, ids[order], np.clip(scores[order], -1, 1)


def nearest_neighbors_cross(query_embeddings: np.ndarray, reference_embeddings: np.ndarray, *,
                            top_k=200, batch_size=128, backend="auto"):
    """Search query vectors against a reference index and recompute retained cosine scores."""
    query_source = finite_array(query_embeddings)
    reference_source = finite_array(reference_embeddings)
    if (np.any(np.abs(query_source) > np.finfo(np.float32).max) or
            np.any(np.abs(reference_source) > np.finfo(np.float32).max)):
        raise ValueError("Embedding values overflow float32 representation")
    query = query_source.astype(np.float32)
    reference = reference_source.astype(np.float32)
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Embedding values overflow float32 representation")
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Query and reference embeddings must be compatible matrices")
    if (not isinstance(top_k, (int, np.integer)) or not isinstance(batch_size, (int, np.integer)) or
            top_k < 1 or batch_size < 1 or len(reference) == 0):
        raise ValueError("Reference retrieval requires positive sizes and at least one reference")
    qnorm = np.linalg.norm(query.astype(np.float64), axis=1, keepdims=True)
    rnorm = np.linalg.norm(reference.astype(np.float64), axis=1, keepdims=True)
    if (qnorm == 0).any() or (rnorm == 0).any() or not np.isfinite(qnorm).all() or not np.isfinite(rnorm).all():
        raise ValueError("Cannot retrieve from zero or non-finite norm vectors")
    query = (query / qnorm).astype(np.float32, copy=False)
    reference = (reference / rnorm).astype(np.float32, copy=False)
    if not np.isfinite(query).all() or not np.isfinite(reference).all():
        raise ValueError("Embedding normalization produced non-finite values")
    backend = _resolve_retrieval_backend(backend, len(query) * len(reference))
    if backend == "numpy" and len(query) * len(reference) > MAX_NUMPY_PAIRWISE:
        raise ValueError(
            "numpy cross retrieval is limited to 10 million pairwise comparisons; "
            "use backend='faiss' or 'faiss-hnsw' for large graphs"
        )
    index = None
    if backend in ("faiss", "faiss-hnsw"):
        import faiss
        if backend == "faiss-hnsw":
            faiss.omp_set_num_threads(1)
            index = faiss.IndexHNSWFlat(reference.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = max(200, top_k)
        else:
            index = faiss.IndexFlatIP(reference.shape[1])
        index.add(reference)
    k = min(top_k, len(reference))
    for start in range(0, len(query), batch_size):
        batch = query[start:start + batch_size]
        if index is None:
            score_batch = batch @ reference.T
            candidates = np.broadcast_to(np.arange(len(reference)), score_batch.shape)
        else:
            score_batch, candidates = index.search(batch, k)
        for offset, (scores, ids) in enumerate(zip(score_batch, candidates)):
            ids = ids[ids >= 0]
            if index is not None:
                scores = query[start + offset] @ reference[ids].T
            order = np.lexsort((ids, -scores))[:k]
            yield start + offset, ids[order], np.clip(scores[order], -1, 1)


def structure_edges(ids: list[str], embeddings: np.ndarray, *, minimum_score=.50,
                    top_k=200, batch_size=128, backend="numpy") -> list[Edge]:
    if not 0 <= minimum_score <= 1:
        raise ValueError("Invalid structure-channel threshold")
    unique_index(ids)
    if not ids:
        raise ValueError("Structure retrieval requires at least one protein")
    if len(ids) != len(embeddings):
        raise ValueError("Structure vector and ID counts differ")
    pairs = {}
    for query, neighbors, scores in nearest_neighbors(embeddings, top_k=top_k,
                                                      batch_size=batch_size, backend=backend):
        for neighbor, score in zip(neighbors, scores):
            if score >= minimum_score:
                pair = tuple(sorted((ids[query], ids[int(neighbor)])))
                pairs[pair] = max(pairs.get(pair, 0), float(score))
    return [Edge(a, b, "structure", score, raw_score=score, threshold=minimum_score,
                 provenance="tm-vec") for (a, b), score in sorted(pairs.items())]


def structure_query_reference_edges(query_ids: list[str], query_embeddings: np.ndarray,
                                    reference_ids: list[str], reference_embeddings: np.ndarray, *,
                                    minimum_score=.50, top_k=200, batch_size=128,
                                    backend="auto") -> list[Edge]:
    """Retrieve structural neighbors using a reference-only index."""
    if not 0 <= minimum_score <= 1:
        raise ValueError("Invalid structure-channel threshold")
    queries, references = _partition_ids(query_ids, reference_ids)
    if len(queries) != len(query_embeddings) or len(references) != len(reference_embeddings):
        raise ValueError("Structure vector and partition ID counts differ")
    edges = []
    for query, neighbors, scores in nearest_neighbors_cross(query_embeddings, reference_embeddings,
                                                              top_k=top_k, batch_size=batch_size,
                                                              backend=backend):
        for neighbor, score in zip(neighbors, scores):
            if score >= minimum_score:
                edges.append(Edge(queries[query], references[int(neighbor)], "structure", float(score),
                                  raw_score=float(score), threshold=minimum_score, provenance="tm-vec"))
    return edges


@dataclass
class GraphBundle:
    ids: list[str]
    adjacency: sparse.csr_matrix
    edges: list[Edge]
    channel_weights: dict[str, float]
    partitions: dict[str, str] | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def query_ids(self) -> list[str]:
        return [identifier for identifier in self.ids
                if self.partitions and self.partitions.get(identifier) == "query"]

    @property
    def reference_ids(self) -> list[str]:
        return [identifier for identifier in self.ids
                if self.partitions and self.partitions.get(identifier) == "reference"]

    @property
    def fingerprint(self):
        payload = {"ids": self.ids, "weights": self.channel_weights,
                   "partitions": self.partitions or {}}
        # Include optional metadata in graph fingerprints.
        if self.metadata:
            # Exclude derived query statistics from the evidence fingerprint.
            if self.metadata.get("streaming_edges"):
                payload["metadata"] = {key: value for key, value in self.metadata.items()
                                        if key != "streaming_statistics"}
            else:
                payload["metadata"] = self.metadata
        h = hashlib.sha256(digest_object(payload).encode())
        # Reuse the evidence digest computed during streaming output.
        if self.metadata.get("edge_fingerprint"):
            h.update(str(self.metadata["edge_fingerprint"]).encode())
            return h.hexdigest()
        # GraphBundle.build emits canonical pair/channel order already.  Avoid
        # an O(E log E) copy when fingerprinting a large generated graph, but
        # retain order-independent fingerprints for manually constructed bundles.
        keys = ((edge.source, edge.target, edge.channel) for edge in self.edges)
        canonical = True
        previous = None
        ordered_edges = self.edges
        for key in keys:
            if previous is not None and key < previous:
                canonical = False
                break
            previous = key
        if not canonical:
            ordered_edges = sorted(self.edges, key=lambda e: (e.source, e.target, e.channel))
        for edge in ordered_edges:
            h.update(digest_object(vars(edge)).encode())
        return h.hexdigest()

    @classmethod
    def build(cls, ids: list[str], edges: list[Edge], channel_weights=None, *, partitions=None,
              metadata=None):
        index = unique_index(ids)
        weights = dict(channel_weights or {c: 1 / 3 for c in CHANNELS})
        if set(weights) - set(CHANNELS) or any(not np.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
            raise ValueError("Invalid channel weights")
        if partitions is not None:
            if set(partitions) - set(ids) or set(partitions.values()) - {"query", "reference"}:
                raise ValueError("Graph partitions must map known IDs to query/reference")
            if set(partitions) != set(ids):
                raise ValueError("Graph partitions must cover every graph ID")
        pairs = {}
        for edge in edges:
            if edge.source not in index or edge.target not in index:
                raise ValueError("Edge references an unknown protein")
            if edge.channel not in CHANNELS:
                raise ValueError("Unknown edge channel")
            if partitions is not None and partitions[edge.source] == partitions[edge.target] == "query":
                raise ValueError("Query-query edges are forbidden in a partitioned graph")
            finite_array([edge.score], unit_interval=True)
            if edge.raw_score is not None and (not np.isfinite(edge.raw_score) or edge.raw_score < 0):
                raise ValueError("Non-finite raw edge score")
            if edge.threshold is not None and (not np.isfinite(edge.threshold) or not 0 <= edge.threshold <= 1):
                raise ValueError("Invalid edge threshold")
            if edge.source == edge.target or edge.score == 0:
                continue
            key = tuple(sorted((edge.source, edge.target)))
            pair = pairs.setdefault(key, {})
            current = pair.get(edge.channel)
            if current is None or edge.score > current.score:
                pair[edge.channel] = edge
        rows, columns, values, normalized = [], [], [], []
        for (left, right), scores in sorted(pairs.items()):
            active_scores = {channel: evidence for channel, evidence in scores.items()
                             if weights.get(channel, 0) > 0}
            denominator = sum(weights.get(c, 0) for c in active_scores)
            if not denominator:
                continue
            value = sum(weights.get(c, 0) * evidence.score for c, evidence in active_scores.items()) / denominator
            i, j = index[left], index[right]
            rows.extend([i, j]); columns.extend([j, i]); values.extend([value, value])
            normalized.extend(Edge(left, right, c, evidence.score,
                                   raw_score=evidence.raw_score, threshold=evidence.threshold,
                                   provenance=evidence.provenance or "")
                             for c, evidence in sorted(active_scores.items()))
        adjacency = sparse.csr_matrix((values, (rows, columns)), shape=(len(ids), len(ids)), dtype=np.float64)
        normalized_partitions = dict(partitions) if partitions is not None else None
        graph_metadata = dict(metadata or {})
        graph_metadata.setdefault("universe_fingerprint", digest_object({
            "ids": list(ids), "partitions": normalized_partitions or {}}))
        return cls(list(ids), adjacency, normalized, weights,
                   normalized_partitions, graph_metadata)

    @classmethod
    def build_query_reference(cls, query_ids, reference_ids, *args, **kwargs):
        """Class-oriented alias for :func:`build_query_reference_graph`."""
        return build_query_reference_graph(query_ids, reference_ids, *args, **kwargs)

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(path / "adjacency.npz", self.adjacency)
        write_json(path / "graph.json", {"ids": self.ids, "weights": self.channel_weights,
                                          "partitions": self.partitions,
                                          "metadata": self.metadata})
        universe_rows = ({"protein_id": identifier,
                          "partition": (self.partitions or {}).get(identifier, "")}
                         for identifier in self.ids)
        write_table(path / "network_universe.tsv", universe_rows, ["protein_id", "partition"])
        rows = ({"source": edge.source, "target": edge.target, "channel": edge.channel,
                 "score": edge.score, "raw_score": "" if edge.raw_score is None else edge.raw_score,
                 "threshold": "" if edge.threshold is None else edge.threshold,
                 "provenance": edge.provenance}
                for edge in self.edges)
        write_table(path / "edges.tsv", rows,
                    ["source", "target", "channel", "score", "raw_score", "threshold", "provenance"])

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        metadata = read_json(path / "graph.json")
        ids = list(metadata.get("ids", []))
        unique_index(ids)
        try:
            weights = {str(key): float(value) for key, value in dict(metadata.get("weights", {})).items()}
            valid_weights = (not set(weights) - set(CHANNELS) and
                             all(np.isfinite(value) and value >= 0 for value in weights.values()) and
                             any(weights.values()))
        except (TypeError, ValueError):
            valid_weights = False
            weights = {}
        if not valid_weights:
            raise ValueError("Invalid graph channel weights")
        partitions = metadata.get("partitions")
        if partitions is not None and (set(partitions) != set(ids) or
                                       set(partitions.values()) - {"query", "reference"}):
            raise ValueError("Graph partitions must cover every graph ID")
        graph_metadata = metadata.get("metadata") or {}
        if graph_metadata.get("streaming_edges"):
            # Validate stored sparse matrices by shape, finite values and content hash.
            stored = sparse.load_npz(path / "adjacency.npz").tocsr()
            if stored.shape != (len(ids), len(ids)):
                raise ValueError("Streaming graph adjacency shape disagrees with graph IDs")
            expected_adjacency = graph_metadata.get("adjacency_sha256")
            if expected_adjacency and digest_file(path / "adjacency.npz") != expected_adjacency:
                raise ValueError("Streaming graph adjacency integrity failure")
            if not np.isfinite(stored.data).all() or (stored.data < 0).any():
                raise ValueError("Streaming graph adjacency contains invalid values")
            result = cls(ids, stored,
                          _DiskEdgeCollection(path / "edges.tsv", graph_metadata.get("edge_count")),
                          weights, partitions=partitions,
                          metadata=graph_metadata)
            universe_path = path / "network_universe.tsv"
            if universe_path.is_file():
                expected_partitions = result.partitions or {}
                count = 0
                for row in read_table(universe_path, ("protein_id", "partition")):
                    if count >= len(result.ids) or row["protein_id"] != result.ids[count]:
                        raise ValueError("Graph universe disagrees with graph.json")
                    if row["partition"] != expected_partitions.get(result.ids[count], ""):
                        raise ValueError("Graph universe partition disagrees with graph.json")
                    count += 1
                if count != len(result.ids):
                    raise ValueError("Graph universe disagrees with graph.json")
            return result
        edges = []
        for r in read_table(path / "edges.tsv"):
            raw = _optional_edge_float(r.get("raw_score"), "raw_score")
            threshold = _optional_edge_float(r.get("threshold"), "threshold")
            edges.append(Edge(r["source"], r["target"], r["channel"], float(r["score"]),
                              raw_score=raw, threshold=threshold, provenance=r.get("provenance", "")))
        result = cls.build(ids, edges, weights, partitions=partitions,
                           metadata=metadata.get("metadata"))
        universe_path = path / "network_universe.tsv"
        if universe_path.is_file():
            expected_partitions = result.partitions or {}
            count = 0
            for row in read_table(universe_path, ("protein_id", "partition")):
                if count >= len(result.ids) or row["protein_id"] != result.ids[count]:
                    raise ValueError("Graph universe disagrees with graph.json")
                expected_partition = expected_partitions.get(result.ids[count], "")
                if row["partition"] != expected_partition:
                    raise ValueError("Graph universe partition disagrees with graph.json")
                count += 1
            if count != len(result.ids):
                raise ValueError("Graph universe disagrees with graph.json")
            expected = digest_object({"ids": result.ids, "partitions": result.partitions or {}})
            stored = result.metadata.get("universe_fingerprint")
            if stored is not None and stored != expected:
                raise ValueError("Graph universe fingerprint is invalid")
        stored = sparse.load_npz(path / "adjacency.npz")
        if stored.shape != result.adjacency.shape or (stored - result.adjacency).nnz:
            raise ValueError("Graph adjacency disagrees with its evidence table")
        return result

    def statistics(self, *, query_ids=None) -> dict:
        """Return graph topology and per-query evidence statistics."""
        if self.metadata.get("streaming_edges") and self.metadata.get("streaming_statistics") is not None and query_ids is None:
            return deepcopy(self.metadata["streaming_statistics"])
        ids = list(self.ids)
        id_index = {identifier: i for i, identifier in enumerate(ids)}
        edge_sets = {channel: {(min(e.source, e.target), max(e.source, e.target))
                               for e in self.edges if e.channel == channel}
                     for channel in CHANNELS}
        degrees = np.asarray(self.adjacency.getnnz(axis=1)).astype(int)
        from scipy.sparse.csgraph import connected_components
        component_count = int(connected_components(self.adjacency, directed=False,
                                                    return_labels=False))
        pair_union = set().union(*edge_sets.values()) if edge_sets else set()
        overlap = {}
        for left, right in combinations(CHANNELS, 2):
            union = edge_sets[left] | edge_sets[right]
            overlap[f"{left}+{right}"] = (len(edge_sets[left] & edge_sets[right]) / len(union)
                                          if union else 0.)
        # Accumulate endpoint, channel and query counts in one pass.
        channel_endpoint_counts = {channel: Counter() for channel in CHANNELS}
        channel_nodes = {channel: set() for channel in CHANNELS}
        query_evidence_counts = Counter()
        for edge in self.edges:
            channel_endpoint_counts[edge.channel][edge.source] += 1
            channel_endpoint_counts[edge.channel][edge.target] += 1
            channel_nodes[edge.channel].update((edge.source, edge.target))
            query_evidence_counts[edge.source] += 1
            query_evidence_counts[edge.target] += 1

        if len(ids) <= 20:
            top_indices = np.argsort(-degrees, kind="stable")
        else:
            candidates = np.argpartition(-degrees, 19)[:20]
            top_indices = candidates[np.lexsort((candidates, -degrees[candidates]))]
        report = {
            "nodes": len(ids),
            "edges": len(pair_union),
            "undirected_edges": int(self.adjacency.nnz // 2),
            "isolated_nodes": int((degrees == 0).sum()),
            "connected_components": component_count,
            "degree_distribution": self._degree_distribution(degrees),
            "channels": {},
            "channel_overlap_jaccard": overlap,
            "channel_overlap": overlap,
            "scores": self._score_statistics([e.score for e in self.edges]),
            "raw_scores": self._score_statistics([e.raw_score for e in self.edges if e.raw_score is not None]),
            "provenance": dict(sorted(Counter(e.provenance or "unspecified"
                                               for e in self.edges).items())),
            "top_hubs": [{"id": ids[i], "degree": int(degrees[i])}
                         for i in top_indices],
            "graph_checks": {
                "adjacency_symmetric": bool((self.adjacency - self.adjacency.T).nnz == 0),
                "score_finite_nonnegative": bool(all(np.isfinite(e.score) and e.score >= 0 for e in self.edges)),
                "raw_score_finite_nonnegative": bool(all(e.raw_score is None or
                                                         (np.isfinite(e.raw_score) and e.raw_score >= 0)
                                                         for e in self.edges)),
                "query_query_edges": 0,
            },
        }
        if self.partitions:
            qset = {identifier for identifier, part in self.partitions.items() if part == "query"}
            rset = set(self.ids) - qset
            report["partition_edges"] = {
                "query_reference": len({(min(e.source, e.target), max(e.source, e.target)) for e in self.edges
                                        if (e.source in qset) != (e.target in qset)}),
                "reference_reference": len({(min(e.source, e.target), max(e.source, e.target)) for e in self.edges
                                             if e.source in rset and e.target in rset}),
                "query_query": len({(min(e.source, e.target), max(e.source, e.target)) for e in self.edges
                                     if e.source in qset and e.target in qset}),
            }
            report["graph_checks"]["query_query_edges"] = report["partition_edges"]["query_query"]
        for channel in CHANNELS:
            channel_degrees = np.fromiter(
                (channel_endpoint_counts[channel].get(identifier, 0) for identifier in ids),
                dtype=np.int64, count=len(ids))
            report["channels"][channel] = {
                "edges": len(edge_sets[channel]),
                "nodes": len(channel_nodes[channel]),
                "missing_nodes": int(len(ids) - len(channel_nodes[channel])),
                "degree_mean": float(channel_degrees.mean()) if len(channel_degrees) else 0.,
                "degree_max": int(channel_degrees.max()) if len(channel_degrees) else 0,
                "degree_distribution": self._degree_distribution(channel_degrees),
            }
        if self.metadata.get("profile"):
            report["profile"] = dict(self.metadata["profile"])
        if query_ids is None and self.partitions:
            query_ids = [identifier for identifier, partition in self.partitions.items() if partition == "query"]
        if query_ids is not None:
            query_list = list(query_ids)
            unique_index(query_list)
            qset = set(query_list)
            if not qset <= set(ids):
                raise ValueError("Query statistics reference unknown IDs")
            report["queries"] = len(qset)
            report["query_evidence"] = {
                identifier: {
                    "edge_count": int(self.adjacency[id_index[identifier]].getnnz()),
                    "evidence_count": int(query_evidence_counts.get(identifier, 0)),
                    # There are only three channels; look up the already
                    # accumulated endpoint counters instead of scanning all
                    # edges once per query.
                    "channels": sorted(channel for channel in CHANNELS
                                        if channel_endpoint_counts[channel].get(identifier, 0))}
                for identifier in sorted(qset)
            }
        return report

    @staticmethod
    def _score_statistics(scores) -> dict:
        values = np.asarray([v for v in scores if v is not None], dtype=float)
        if not len(values):
            return {"count": 0, "min": 0., "p25": 0., "median": 0., "p75": 0., "max": 0.,
                    "mean": 0.}
        if not np.isfinite(values).all():
            raise ValueError("Graph contains non-finite scores")
        return {"count": int(len(values)), "min": float(values.min()),
                "p25": float(np.percentile(values, 25)), "median": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)), "max": float(values.max()),
                "mean": float(values.mean())}

    @staticmethod
    def _degree_distribution(degrees) -> dict:
        values = np.asarray(degrees, dtype=float)
        if not len(values):
            return {"min": 0, "p25": 0., "median": 0., "p75": 0., "max": 0}
        return {"min": int(values.min()), "p25": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)), "p75": float(np.percentile(values, 75)),
                "max": int(values.max())}


_STREAM_EDGE_DTYPE = np.dtype([
    ("left", "<u4"), ("right", "<u4"), ("channel", "u1"),
    ("score", "<f8"), ("raw", "<f8"), ("threshold", "<f8"),
    ("provenance", "<u4"), ("order", "<u8"),
])
_STREAM_FUSED_DTYPE = np.dtype([("left", "<u4"), ("right", "<u4"), ("score", "<f4")])


class _StreamingQuantiles:
    """Bounded-memory score summary for large edge tables."""
    _LIMIT = 200_000

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = np.inf
        self.maximum = -np.inf
        self.sample = np.empty(self._LIMIT, dtype=np.float64)
        self.sample_count = 0

    def add(self, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("Graph contains non-finite scores")
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if self.sample_count < self._LIMIT:
            self.sample[self.sample_count] = value
            self.sample_count += 1
        else:
            # Deterministic reservoir replacement; no random state is needed
            # for an audit summary and memory remains bounded.
            slot = ((self.count * 2654435761) & 0xFFFFFFFF) % self.count
            if slot < self._LIMIT:
                self.sample[slot] = value

    def finish(self):
        if not self.count:
            return {"count": 0, "min": 0., "p25": 0., "median": 0., "p75": 0., "max": 0., "mean": 0.}
        values = np.sort(self.sample[:self.sample_count])
        return {"count": int(self.count), "min": float(self.minimum),
                "p25": float(np.percentile(values, 25)),
                "median": float(np.percentile(values, 50)),
                "p75": float(np.percentile(values, 75)), "max": float(self.maximum),
                "mean": float(self.total / self.count)}


def _stream_chunk_files(edge_paths, index, weights, partitions, temporary, *, chunk_rows):
    """Convert TSV rows to sorted, compact binary runs on a native disk."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    channel_index = {name: i for i, name in enumerate(CHANNELS)}
    qset = {identifier for identifier, part in (partitions or {}).items() if part == "query"}
    rset = {identifier for identifier, part in (partitions or {}).items() if part == "reference"}
    provenance_index = {}
    chunk_paths = []
    buffer = np.empty(chunk_rows, dtype=_STREAM_EDGE_DTYPE)
    used = 0
    ordinal = 0

    def flush():
        nonlocal used
        if not used:
            return
        # Sort by pair/channel, descending score and input order; keep the first row per key.
        source = buffer[:used]
        order = np.lexsort((source["order"], -source["score"], source["channel"],
                            source["right"], source["left"]))
        values = source[order]
        del order
        if len(values) > 1:
            same_key = ((values["left"][1:] == values["left"][:-1]) &
                        (values["right"][1:] == values["right"][:-1]) &
                        (values["channel"][1:] == values["channel"][:-1]))
            keep = np.empty(len(values), dtype=bool)
            keep[0] = True
            keep[1:] = ~same_key
            values = values[keep]
        path = temporary / f"run-{len(chunk_paths):06d}.bin"
        values.tofile(path)
        chunk_paths.append(path)
        used = 0

    for edge_path in edge_paths:
        for row in read_table(edge_path, ("source", "target", "channel", "score")):
            source, target, channel = row["source"], row["target"], row["channel"]
            if source not in index or target not in index:
                raise ValueError("Edge references an unknown protein")
            if channel not in channel_index:
                raise ValueError("Unknown edge channel")
            if partitions and partitions[source] == partitions[target] == "query":
                raise ValueError("Query-query edges are forbidden in a partitioned graph")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Malformed graph edge row") from exc
            raw_value = _optional_edge_float(row.get("raw_score"), "raw_score")
            threshold_value = _optional_edge_float(row.get("threshold"), "threshold")
            raw = np.nan if raw_value is None else raw_value
            threshold = np.nan if threshold_value is None else threshold_value
            finite_array([score], unit_interval=True)
            if not np.isnan(raw) and raw < 0:
                raise ValueError("Non-finite raw edge score")
            if not np.isnan(threshold) and not 0 <= threshold <= 1:
                raise ValueError("Invalid edge threshold")
            if source == target or score == 0 or weights.get(channel, 0.) == 0:
                ordinal += 1
                continue
            left, right = index[source], index[target]
            if left > right:
                left, right = right, left
            provenance = row.get("provenance", "") or ""
            provenance_id = provenance_index.setdefault(provenance, len(provenance_index))
            buffer[used] = (left, right, channel_index[channel], score, raw, threshold,
                            provenance_id, ordinal)
            used += 1
            ordinal += 1
            if used == chunk_rows:
                flush()
    flush()
    return chunk_paths, provenance_index


def _iter_sorted_run(path):
    count = path.stat().st_size // _STREAM_EDGE_DTYPE.itemsize
    values = np.memmap(path, dtype=_STREAM_EDGE_DTYPE, mode="r", shape=(count,))
    try:
        for value in values:
            yield value
    finally:
        del values


def _stream_merge_records(chunk_paths):
    iterators = [_iter_sorted_run(path) for path in chunk_paths]
    heap = []
    for run, iterator in enumerate(iterators):
        try:
            value = next(iterator)
        except StopIteration:
            continue
        key = (int(value["left"]), int(value["right"]), int(value["channel"]), int(value["order"]), run)
        heapq.heappush(heap, (key, run, value))
    while heap:
        _, run, value = heapq.heappop(heap)
        group_key = (int(value["left"]), int(value["right"]), int(value["channel"]))
        best = value
        try:
            next_value = next(iterators[run])
            next_key = (int(next_value["left"]), int(next_value["right"]), int(next_value["channel"]), int(next_value["order"]), run)
            heapq.heappush(heap, (next_key, run, next_value))
        except StopIteration:
            pass
        while heap and (int(heap[0][2]["left"]), int(heap[0][2]["right"]), int(heap[0][2]["channel"])) == group_key:
            _, other_run, other = heapq.heappop(heap)
            if (float(other["score"]) > float(best["score"]) or
                    (float(other["score"]) == float(best["score"]) and int(other["order"]) < int(best["order"]))):
                best = other
            try:
                next_value = next(iterators[other_run])
                next_key = (int(next_value["left"]), int(next_value["right"]), int(next_value["channel"]), int(next_value["order"]), other_run)
                heapq.heappush(heap, (next_key, other_run, next_value))
            except StopIteration:
                pass
        yield best


def build_streaming_graph(edge_paths, ids, output, channel_weights=None, *, partitions=None,
                          metadata=None, chunk_rows=500_000, temporary_root=None):
    """Sort and merge compact edge batches into evidence tables and sparse adjacency."""
    ids = list(ids)
    index = unique_index(ids)
    if len(ids) >= 2 ** 32:
        raise ValueError("Streaming graph supports fewer than 2^32 nodes")
    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError("Streaming graph chunk_rows must be a positive integer")
    weights = dict(channel_weights or {c: 1 / 3 for c in CHANNELS})
    if set(weights) - set(CHANNELS) or any(not np.isfinite(v) or v < 0 for v in weights.values()) or not any(weights.values()):
        raise ValueError("Invalid channel weights")
    if partitions is not None:
        if set(partitions) != set(ids) or set(partitions.values()) - {"query", "reference"}:
            raise ValueError("Graph partitions must cover every graph ID")
    edge_paths = [Path(path) for path in edge_paths]
    if not edge_paths:
        raise ValueError("At least one edge table is required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(temporary_root).resolve() if temporary_root else Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="phanet-graph-merge-", dir=root))
    chunk_paths = []
    try:
        chunk_paths, provenance_index = _stream_chunk_files(edge_paths, index, weights, partitions, temporary,
                                                            chunk_rows=chunk_rows)
        edge_path = output / "edges.tsv"
        fused_path = temporary / "fused.bin"
        edge_handle = edge_path.open("w", encoding="utf-8", newline="")
        # ``csv.writer`` avoids constructing one Python dict per retained
        # evidence edge. At tens of millions of rows this reduces allocation
        # and formatting overhead while preserving the tabular schema.
        edge_fields = ["source", "target", "channel", "score", "raw_score", "threshold", "provenance"]
        writer = csv.writer(edge_handle, delimiter="\t", lineterminator="\n")
        writer.writerow(edge_fields)
        fused_handle = fused_path.open("wb")
        provenance_values = [value for value, _ in sorted(provenance_index.items(), key=lambda item: item[1])]
        channel_index = {name: i for i, name in enumerate(CHANNELS)}
        channel_counts = np.zeros(3, dtype=np.int64)
        channel_degrees = np.zeros((3, len(ids)), dtype=np.int64)
        channel_nodes = np.zeros((3, len(ids)), dtype=bool)
        degrees = np.zeros(len(ids), dtype=np.int64)
        query_evidence = np.zeros(len(ids), dtype=np.int64)
        query_channels = np.zeros((len(ids), 3), dtype=bool)
        pair_masks = np.zeros(8, dtype=np.int64)
        score_stats = _StreamingQuantiles(); raw_stats = _StreamingQuantiles()
        provenance_counts = Counter()
        edge_hash = hashlib.sha256()
        counts = np.zeros(len(ids), dtype=np.int64)
        pair_count = 0
        evidence_count = 0
        current_pair = None
        current = {}

        def emit(left, right, values):
            nonlocal pair_count, evidence_count
            if not values:
                return
            mask = sum(1 << channel for channel in values)
            pair_masks[mask] += 1
            denominator = sum(weights[CHANNELS[channel]] for channel in values)
            if denominator <= 0:
                return
            fused = sum(weights[CHANNELS[channel]] * float(value["score"])
                        for channel, value in values.items()) / denominator
            fused_handle.write(struct.pack("<IIf", left, right, float(fused)))
            pair_count += 1; counts[left] += 1; counts[right] += 1
            degrees[left] += 1; degrees[right] += 1
            for channel in sorted(values):
                value = values[channel]; name = CHANNELS[channel]
                raw = float(value["raw"]); threshold = float(value["threshold"])
                provenance = provenance_values[int(value["provenance"])]
                writer.writerow([ids[left], ids[right], name, float(value["score"]),
                                 "" if np.isnan(raw) else raw,
                                 "" if np.isnan(threshold) else threshold, provenance])
                packed = struct.pack("<IIBfffI", left, right, channel, float(value["score"]),
                                     raw if np.isfinite(raw) else np.nan,
                                     threshold if np.isfinite(threshold) else np.nan,
                                     int(value["provenance"]))
                edge_hash.update(packed)
                evidence_count += 1; channel_counts[channel] += 1
                channel_degrees[channel, left] += 1; channel_degrees[channel, right] += 1
                channel_nodes[channel, left] = True; channel_nodes[channel, right] = True
                score_stats.add(value["score"])
                if np.isfinite(raw): raw_stats.add(raw)
                provenance_counts[provenance or "unspecified"] += 1
                if partitions:
                    if partitions[ids[left]] == "query" or partitions[ids[right]] == "query":
                        for node in (left, right):
                            if partitions[ids[node]] == "query":
                                query_evidence[node] += 1; query_channels[node, channel] = True

        for value in _stream_merge_records(chunk_paths):
            pair = (int(value["left"]), int(value["right"]))
            channel = int(value["channel"])
            if current_pair is not None and pair != current_pair:
                emit(current_pair[0], current_pair[1], current)
                current = {}
            current_pair = pair
            current[channel] = value
        if current_pair is not None:
            emit(current_pair[0], current_pair[1], current)
        fused_handle.close(); edge_handle.close()

        fused_count = pair_count
        indptr = np.empty(len(ids) + 1, dtype=np.int64); indptr[0] = 0
        # ``counts`` already includes one endpoint for each side of every
        # fused pair, so its sum is the final CSR nnz.
        indptr[1:] = np.cumsum(counts, dtype=np.int64)
        nnz = int(indptr[-1])
        indices_array = np.empty(nnz, dtype=np.int32 if len(ids) < 2 ** 31 else np.int64)
        data_array = np.empty(nnz, dtype=np.float32)
        cursor = indptr[:-1].copy()

        def reserve_positions(nodes):
            """Assign distinct CSR offsets for each endpoint occurrence in a batch."""
            nodes = np.asarray(nodes, dtype=np.int64)
            if not len(nodes):
                return np.empty(0, dtype=np.int64)
            order = np.argsort(nodes, kind="stable")
            sorted_nodes = nodes[order]
            starts = np.empty(len(sorted_nodes), dtype=bool)
            starts[0] = True
            starts[1:] = sorted_nodes[1:] != sorted_nodes[:-1]
            group_starts = np.flatnonzero(starts)
            group_ends = np.r_[group_starts[1:], len(sorted_nodes)]
            offsets = np.arange(len(sorted_nodes), dtype=np.int64)
            offsets -= np.repeat(group_starts, group_ends - group_starts)
            sorted_positions = cursor[sorted_nodes] + offsets
            positions = np.empty(len(nodes), dtype=np.int64)
            positions[order] = sorted_positions
            cursor[sorted_nodes[group_starts]] += group_ends - group_starts
            return positions

        if fused_count:
            records = np.memmap(fused_path, dtype=_STREAM_FUSED_DTYPE, mode="r", shape=(fused_count,))
            try:
                for start in range(0, fused_count, max(1, chunk_rows)):
                    values = records[start:start + chunk_rows]
                    left = values["left"].astype(np.int64, copy=False)
                    right = values["right"].astype(np.int64, copy=False)
                    positions = reserve_positions(left)
                    indices_array[positions] = right
                    data_array[positions] = values["score"]
                    positions = reserve_positions(right)
                    indices_array[positions] = left
                    data_array[positions] = values["score"]
            finally:
                del records
        adjacency = sparse.csr_matrix((data_array, indices_array, indptr), shape=(len(ids), len(ids)), dtype=np.float32)
        sparse.save_npz(output / "adjacency.npz", adjacency)
        write_json(output / "graph.json", {"ids": ids, "weights": weights, "partitions": partitions,
                                            "metadata": {**(metadata or {}), "streaming_edges": True,
                                                         "edge_count": evidence_count,
                                                         "pair_count": pair_count,
                                                         "edge_fingerprint": edge_hash.hexdigest()}})
        universe_rows = ({"protein_id": identifier, "partition": (partitions or {}).get(identifier, "")} for identifier in ids)
        write_table(output / "network_universe.tsv", universe_rows, ["protein_id", "partition"])
        degrees = np.asarray(adjacency.getnnz(axis=1)).astype(int)
        from scipy.sparse.csgraph import connected_components
        components = int(connected_components(adjacency, directed=False, return_labels=False))
        union_pairs = int(pair_masks[1:].sum())
        overlap = {}
        for left_name, right_name in combinations(CHANNELS, 2):
            left_bit, right_bit = 1 << channel_index[left_name], 1 << channel_index[right_name]
            intersection = sum(pair_masks[mask] for mask in range(8) if mask & left_bit and mask & right_bit)
            union = sum(pair_masks[mask] for mask in range(8) if mask & left_bit or mask & right_bit)
            overlap[f"{left_name}+{right_name}"] = float(intersection / union) if union else 0.
        if len(ids) <= 20:
            top_indices = np.argsort(-degrees, kind="stable")
        else:
            candidates = np.argpartition(-degrees, 19)[:20]
            top_indices = candidates[np.lexsort((candidates, -degrees[candidates]))]
        statistics = {
            "nodes": len(ids), "edges": union_pairs, "undirected_edges": pair_count,
            "isolated_nodes": int((degrees == 0).sum()), "connected_components": components,
            "degree_distribution": GraphBundle._degree_distribution(degrees), "channels": {},
            "channel_overlap_jaccard": overlap, "channel_overlap": overlap,
            "scores": score_stats.finish(), "raw_scores": raw_stats.finish(),
            "provenance": dict(sorted(provenance_counts.items())),
            "top_hubs": [{"id": ids[i], "degree": int(degrees[i])} for i in top_indices],
            # Write each fused edge to both endpoint rows with the same score.
            "graph_checks": {"adjacency_symmetric": True,
                             "score_finite_nonnegative": True, "raw_score_finite_nonnegative": True,
                             "query_query_edges": 0},
        }
        for channel, name in enumerate(CHANNELS):
            statistics["channels"][name] = {"edges": int(channel_counts[channel]),
                "nodes": int(channel_nodes[channel].sum()), "missing_nodes": int(len(ids) - channel_nodes[channel].sum()),
                "degree_mean": float(channel_degrees[channel].mean()) if len(ids) else 0.,
                "degree_max": int(channel_degrees[channel].max()) if len(ids) else 0,
                "degree_distribution": GraphBundle._degree_distribution(channel_degrees[channel])}
        if partitions:
            qset = np.asarray([partitions[i] == "query" for i in ids], dtype=bool)
            # The evidence endpoint counters provide query evidence; derive
            # partition pair counts directly from the compact fused file.
            qr = rr = 0
            if fused_count:
                fused = np.memmap(fused_path, dtype=_STREAM_FUSED_DTYPE, mode="r", shape=(fused_count,))
                try:
                    for value in fused:
                        l, r = int(value["left"]), int(value["right"])
                        if qset[l] != qset[r]: qr += 1
                        elif not qset[l]: rr += 1
                finally:
                    del fused
            statistics["partition_edges"] = {"query_reference": qr, "reference_reference": rr, "query_query": 0}
            statistics["queries"] = int(qset.sum())
            statistics["query_evidence"] = {ids[i]: {"edge_count": int(degrees[i]),
                "evidence_count": int(query_evidence[i]),
                "channels": [CHANNELS[c] for c in range(3) if query_channels[i, c]]}
                for i in np.flatnonzero(qset)}
        graph_metadata = {**(metadata or {}), "streaming_edges": True, "edge_count": evidence_count,
                          "pair_count": pair_count, "edge_fingerprint": edge_hash.hexdigest(),
                          "streaming_statistics": statistics}
        adjacency_hash = digest_file(output / "adjacency.npz")
        graph_metadata["adjacency_sha256"] = adjacency_hash
        write_json(output / "graph.json", {"ids": ids, "weights": weights, "partitions": partitions,
                                            "metadata": graph_metadata})
        graph = GraphBundle(ids, adjacency, _DiskEdgeCollection(edge_path, evidence_count), weights,
                            dict(partitions) if partitions is not None else None, graph_metadata)
        return graph, statistics
    finally:
        for path in (edge_path if 'edge_path' in locals() else None,):
            if path is not None and 'edge_handle' in locals() and not edge_handle.closed:
                edge_handle.close()
        if 'fused_handle' in locals() and not fused_handle.closed:
            fused_handle.close()
        shutil.rmtree(temporary, ignore_errors=True)


def build_query_reference_graph(query_ids, reference_ids, sequence_hits=None,
                                profile_hits=None, structure_vectors=None,
                                reference_edges=None, channel_weights=None,
                                include_reference_reference=True,
                                sequence_self_bits=None, sequence_evalue=1e-5,
                                sequence_min_bits=50., sequence_minimum_score=.30,
                                profile_evalue=1e-5, profile_minimum_coverage=.50,
                                profile_max_family_members=5000,
                                profile_top_k_per_family=50,
                                profile_max_edges_per_family=100000,
                                structure_minimum_score=.50, structure_top_k=200,
                                structure_batch_size=128,
                                structure_backend="auto", partitions_order="reference-first") -> GraphBundle:
    """Build a graph from reference-reference and query-reference evidence.

    structure_vectors accepts a (query_vectors, reference_vectors) tuple or a
    mapping with query and reference keys. reference_edges supplies reference
    graph edges. Node order is reference-first by default; use
    partitions_order="query-first" to reverse it."""
    queries, references = _partition_ids(query_ids, reference_ids)
    if not queries or not references:
        raise ValueError("Fresh retrieval requires at least one query and one reference")
    if partitions_order not in ("reference-first", "query-first"):
        raise ValueError("partitions_order must be reference-first or query-first")
    edges = []
    metadata = {"reference_policy": "fresh-query-to-reference",
                "profile": {"families": 0, "families_truncated": 0,
                            "members_dropped": 0, "candidate_edges": 0,
                            "edges_retained": 0}}
    if reference_edges is not None:
        frozen_edges = list(reference_edges)
        reference_set = set(references)
        query_set = set(queries)
        if any(edge.source in query_set and edge.target in query_set for edge in frozen_edges):
            raise ValueError("Query-query edges are forbidden in a fresh retrieval graph")
        if any(edge.source not in reference_set or edge.target not in reference_set for edge in frozen_edges):
            raise ValueError("Frozen reference graph may contain reference nodes only")
        if include_reference_reference:
            edges.extend(frozen_edges)

    if sequence_hits is not None:
        # Consume MMseqs2 evidence as compact tuples for score normalization.
        sequence_iterator = iter(sequence_hits)
        first = next(sequence_iterator, None)
        if first is not None and isinstance(first, Edge):
            edge_items = [first]
            for item in sequence_iterator:
                if not isinstance(item, Edge):
                    raise ValueError("Sequence evidence must be uniformly MMseqs rows or Edge objects")
                edge_items.append(item)
            edges.extend(_partition_filter(edge_items, queries, references,
                                            include_reference_reference=include_reference_reference))
        elif first is not None:
            allowed_ids = set(queries) | set(references)
            query_set, reference_set = set(queries), set(references)
            cross_rows, reference_rows = [], []
            denominators = dict(sequence_self_bits or {})
            if not set(denominators) <= allowed_ids:
                raise ValueError("Self-alignment scores reference an unknown partition ID")
            for row in chain((first,), sequence_iterator):
                try:
                    left, right = row["query"], row["target"]
                    bits, significance = float(row["bits"]), float(row["evalue"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError("Malformed MMseqs2 alignment row") from exc
                if left not in allowed_ids or right not in allowed_ids:
                    raise ValueError("Sequence evidence references an ID outside query/reference partitions")
                if not np.isfinite(bits) or bits < 0 or not np.isfinite(significance) or significance < 0:
                    raise ValueError("Invalid MMseqs2 alignment score")
                if left == right:
                    denominators[left] = max(bits, denominators.get(left, 0.))
                elif left in query_set and right in reference_set:
                    cross_rows.append((left, right, bits, significance))
                elif include_reference_reference and left in reference_set and right in reference_set:
                    reference_rows.append((left, right, bits, significance))

            def compact_rows(items):
                return ({"query": left, "target": right, "bits": bits, "evalue": significance}
                        for left, right, bits, significance in items)

            edges.extend(sequence_edges(compact_rows(cross_rows), evalue=sequence_evalue,
                                         min_bits=sequence_min_bits,
                                         minimum_score=sequence_minimum_score,
                                         query_ids=queries, reference_ids=references,
                                         self_bits=denominators))
            if include_reference_reference:
                edges.extend(sequence_edges(compact_rows(reference_rows), evalue=sequence_evalue,
                                            min_bits=sequence_min_bits,
                                            minimum_score=sequence_minimum_score,
                                            self_bits=denominators))

    if profile_hits is not None:
        hits = list(profile_hits)
        allowed_ids = set(queries) | set(references)
        if any(hit.get("protein_id") not in allowed_ids for hit in hits):
            raise ValueError("Profile evidence references an ID outside query/reference partitions")
        profile_stats = {}
        profile_all = profile_edges(hits, evalue=profile_evalue,
                                    minimum_coverage=profile_minimum_coverage,
                                    query_ids=queries, reference_ids=references,
                                    include_reference_reference=include_reference_reference,
                                    max_family_members=profile_max_family_members,
                                    top_k_per_family=profile_top_k_per_family,
                                    max_edges_per_family=profile_max_edges_per_family,
                                    stats=profile_stats)
        metadata["profile"] = profile_stats
        edges.extend(_partition_filter(profile_all, queries, references,
                                       include_reference_reference=include_reference_reference,
                                       reject_query_query=True))

    if structure_vectors is not None:
        if isinstance(structure_vectors, dict):
            query_vectors = structure_vectors.get("query", structure_vectors.get("query_embeddings"))
            reference_vectors = structure_vectors.get("reference", structure_vectors.get("reference_embeddings"))
            supplied_query_ids = structure_vectors.get("query_ids")
            supplied_reference_ids = structure_vectors.get("reference_ids")
            if supplied_query_ids is not None and list(supplied_query_ids) != queries:
                raise ValueError("Structure query vector IDs differ from query_ids")
            if supplied_reference_ids is not None and list(supplied_reference_ids) != references:
                raise ValueError("Structure reference vector IDs differ from reference_ids")
        else:
            try:
                query_vectors, reference_vectors = structure_vectors
            except (TypeError, ValueError):
                raise ValueError("structure_vectors must be (query, reference) vectors")
        edges.extend(structure_query_reference_edges(queries, query_vectors, references, reference_vectors,
                                                     minimum_score=structure_minimum_score,
                                                     top_k=structure_top_k, batch_size=structure_batch_size,
                                                     backend=structure_backend))
        if include_reference_reference:
            edges.extend(structure_edges(references, reference_vectors,
                                         minimum_score=structure_minimum_score,
                                         top_k=structure_top_k, backend=structure_backend,
                                         batch_size=structure_batch_size))

    ids = [*references, *queries] if partitions_order == "reference-first" else [*queries, *references]
    partition_map = {identifier: "query" for identifier in queries}
    partition_map.update({identifier: "reference" for identifier in references})
    metadata["query_count"] = len(queries)
    metadata["reference_count"] = len(references)
    metadata["include_reference_reference"] = bool(include_reference_reference)
    metadata["partitions_order"] = partitions_order
    metadata["channel_parameters"] = {
        "sequence": {"evalue": sequence_evalue, "minimum_bits": sequence_min_bits,
                     "minimum_score": sequence_minimum_score},
        "profile": {"evalue": profile_evalue, "minimum_coverage": profile_minimum_coverage,
                    "max_family_members": profile_max_family_members,
                    "top_k_per_family": profile_top_k_per_family,
                    "max_edges_per_family": profile_max_edges_per_family},
        "structure": {"minimum_score": structure_minimum_score, "top_k": structure_top_k,
                      "backend": structure_backend},
    }
    return GraphBundle.build(ids, edges, channel_weights, partitions=partition_map,
                             metadata=metadata)


def retrieval_statistics(graph: GraphBundle, *, query_ids=None) -> dict:
    """Functional alias for callers that treat statistics as a stage output."""
    if not isinstance(graph, GraphBundle):
        raise TypeError("retrieval_statistics expects a GraphBundle")
    return graph.statistics(query_ids=query_ids)


def graph_quality_report(graph: GraphBundle, *, query_ids=None, strict=False) -> dict:
    """Check graph invariants and optionally raise on failures."""
    if not isinstance(graph, GraphBundle):
        raise TypeError("graph_quality_report expects a GraphBundle")
    report = graph.statistics(query_ids=query_ids)
    checks = report.get("graph_checks", {})
    failures = [name for name, value in checks.items()
                if name != "query_query_edges" and value is False]
    if checks.get("query_query_edges", 0) != 0:
        failures.append("query_query_edges")
    if strict and failures:
        raise ValueError(f"Graph QC failed: {', '.join(sorted(failures))}")
    report["qc_status"] = "failed" if failures else "passed"
    report["qc_failures"] = failures
    return report
