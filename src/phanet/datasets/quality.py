"""Explicit source quality metadata and conservative module eligibility."""
from __future__ import annotations
from copy import deepcopy

import numpy as np

from phanet.datasets.schema import Dataset
from phanet.datasets.data import qc_report
from phanet.common.io import digest_object


def apply_metadata(dataset, metadata):
    """Curated flags/intervals can be imported without changing labels or sequence."""
    if set(metadata) - {"source", "version", "genomes", "proteins"} or not metadata.get("source") or not metadata.get("version"):
        raise ValueError("Quality metadata requires source/version and explicit genome/protein mappings")
    result = deepcopy(dataset)
    for kind, records, allowed in (("genomes", result.genomes, {"quality", "complete", "quality_flags", "excluded_regions", "host", "taxonomy", "lifestyle", "protein_content_cluster"}),
                                    ("proteins", result.proteins, {"partial", "quality_flags", "coding_score"})):
        supplied = metadata.get(kind, {})
        if not isinstance(supplied, dict) or any(not isinstance(values, dict) for values in supplied.values()):
            raise ValueError(f"Quality metadata {kind} must map IDs to field mappings")
        lookup = {getattr(r, "genome_id" if kind == "genomes" else "protein_id"): r for r in records}
        if set(supplied) - set(lookup):
            raise ValueError("Quality metadata references unknown IDs")
        for identifier, values in supplied.items():
            if set(values) - allowed:
                raise ValueError("Quality metadata cannot overwrite sequence, coordinates or labels")
            for key, value in values.items():
                if key in ("partial", "complete") and not isinstance(value, bool):
                    raise ValueError("Completeness and partial flags must be JSON booleans")
                if key == "quality_flags":
                    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
                        raise ValueError("Quality flags must be a list of nonempty strings")
                    value = tuple(sorted(set(value)))
                if key == "excluded_regions":
                    if not isinstance(value, list):
                        raise ValueError("Excluded regions must be a list")
                    value = tuple(tuple(v) for v in value)
                setattr(lookup[identifier], key, value)
    result.provenance["quality_metadata"] = {"source": metadata["source"], "version": metadata["version"],
                                              "sha256": digest_object(metadata)}
    result.validate()
    return result


def filter_corpus(dataset, *, minimum_length=2000, coding_score_threshold=None, flag_nested=True):
    """Exclude failed records; keep short proteins and flag uncertain ORFs."""
    dataset.validate()
    if minimum_length < 1:
        raise ValueError("Minimum sequence length must be positive")
    if coding_score_threshold is not None:
        try:
            if not np.isfinite(coding_score_threshold):
                raise ValueError("Coding-score threshold must be finite")
        except (TypeError, ValueError) as exc:
            raise ValueError("Coding-score threshold must be finite") from exc
    report = qc_report(dataset, minimum_length)
    accepted = {r["genome_id"] for r in report if r["eligible"]}
    result = Dataset(deepcopy([g for g in dataset.genomes if g.genome_id in accepted]),
                     deepcopy([p for p in dataset.proteins if p.genome_id in accepted]), deepcopy(dataset.provenance))
    for proteins in result.by_genome().values():
        nested = set()
        if flag_nested:
            outer = {}
            for protein in sorted(proteins, key=lambda p: (p.start, -p.end, p.protein_id)):
                prior = outer.get(protein.strand)
                if prior is not None and prior.end >= protein.end and (prior.start, prior.end) != (protein.start, protein.end):
                    nested.add(protein.protein_id)
                if prior is None or protein.end > prior.end:
                    outer[protein.strand] = protein
        for protein in proteins:
            flags = set(protein.quality_flags)
            if coding_score_threshold is not None and protein.coding_score is not None and protein.coding_score < coding_score_threshold:
                flags.add("low_coding_score")
            if protein.protein_id in nested:
                flags.add("nested_same_strand_orf")
            protein.quality_flags = tuple(sorted(flags))
    result.provenance["qc"] = {"minimum_length": minimum_length, "coding_score_threshold": coding_score_threshold,
                                "nested_rule": "strict_same_strand_containment" if flag_nested else "disabled",
                                "record_report": report, "short_proteins_removed": False}
    result.validate()
    return result, report


def window_eligibility(genome, proteins, members):
    reasons = []
    if genome.quality not in ("complete", "high"):
        reasons.append("record_quality")
    if set(genome.quality_flags) & {"synthetic_construct", "cloning_vector", "non_phage", "severe_contamination", "severe_assembly_warning"}:
        reasons.append("record_quality_flags")
    if any(p.partial or p.quality_flags for p in members):
        reasons.append("member_quality")
    left, right = min(p.start for p in members), max(p.end for p in members)
    if any(start < right and end > left for start, end, _ in genome.excluded_regions):
        reasons.append("excluded_interval")
    if not genome.complete:
        first, last = members[0].order, members[-1].order
        if first == 0 or last == len(proteins) - 1:
            reasons.append("unresolved_contig_end")
        elif any(p.partial or p.quality_flags for p in (proteins[first - 1], proteins[last + 1])):
            reasons.append("incomplete_flanking_gene")
    return not reasons, reasons
