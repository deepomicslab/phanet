"""Reference matching and cross-genome grouping of predicted protein member sets."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from phanet.analysis.template_matching import Relationships
from phanet.common.io import digest_object, write_table


@dataclass(frozen=True)
class CatalogueConfig:
    minimum_clusters: int = 5
    minimum_dark_fraction: float = 2 / 3
    minimum_dark_positions: int = 2
    position_dark_fraction: float = 2 / 3
    max_pair_checks: int = 1000000
    filter_annotations: bool = True

    def __post_init__(self):
        if type(self.filter_annotations) is not bool:
            raise ValueError("filter_annotations must be boolean")
        if any(type(v) is not int or v < lower for v, lower in
               ((self.minimum_clusters, 2), (self.minimum_dark_positions, 2), (self.max_pair_checks, 1))):
            raise ValueError("Invalid catalogue counts/budget")
        if any(not np.isfinite(v) or not 0 < v <= 1 for v in
               (self.minimum_dark_fraction, self.position_dark_fraction)):
            raise ValueError("Invalid darkness thresholds")


def instance_id(genome, members):
    return "MPI-" + digest_object([genome, list(members)])[:24]


def complete_assignment(left, right, relationships, *, ambiguity=False):
    """One-to-one full-set match, retaining repeated protein copies as loci."""
    if len(left) != len(right) or len(left) < 2:
        return None
    edges = np.array([[a == b or relationships.score(a, b) is not None for b in right] for a in left])
    ii, jj = linear_sum_assignment(~edges)
    if not edges[ii, jj].all():
        return None
    assignment = jj.tolist()
    if not ambiguity:
        return assignment
    # An edge belongs to a perfect matching iff it is matched or lies on an
    # alternating cycle. SCCs avoid enumerating factorially many copy mappings.
    _, labels = connected_components(csr_matrix(edges[:, jj]), directed=True, connection="strong")
    inverse = np.argsort(jj)
    feasible = edges & (labels[:, None] == labels[inverse][None, :])
    return assignment, feasible


class SetIndex:
    """Intersect protein/profile postings before checking full member sets."""
    def __init__(self, sets, relationships):
        self.sets, self.relationships = sets, relationships
        self.protein = defaultdict(set)
        self.profile = defaultdict(set)
        self.sizes = defaultdict(set)
        self.sequence = defaultdict(set)
        self.sequence_targets = getattr(relationships.sequences, "targets_for", None)
        if self.sequence_targets is None:
            for (left, right), hit in relationships.sequences.items():
                if hit.eligible:
                    self.sequence[left].add(right)
        for i, proteins in enumerate(sets):
            self.sizes[len(proteins)].add(i)
            for protein in proteins:
                self.protein[protein].add(i)
                for family in relationships.profiles.get(protein, ()):
                    self.profile[family].add(i)

    def query(self, proteins):
        if len(proteins) not in self.sizes:
            return []
        postings = []
        for protein in proteins:
            hits = set(self.protein.get(protein, ()))
            for family in self.relationships.profiles.get(protein, ()):
                hits.update(self.profile.get(family, ()))
            targets = (self.sequence_targets(protein) if self.sequence_targets is not None
                       else self.sequence.get(protein, ()))
            for target in targets:
                hits.update(self.protein.get(target, ()))
            hits.intersection_update(self.sizes.get(len(proteins), ()))
            if not hits:
                return []
            postings.append(hits)
        return sorted(set.intersection(*sorted(postings, key=len)))


def read_references(path, relationships, *, format="native_members"):
    if format not in ("native_members", "literature_targets"):
        raise ValueError("Unknown known-reference format")
    references, identifiers = [], set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Reference rows must be JSON objects")
            if format == "literature_targets":
                references.extend(literature_references(row, relationships))
                continue
            if (not isinstance(row.get("reference_id"), str) or not row["reference_id"]
                    or row["reference_id"] in identifiers or not row.get("source")
                    or type(row.get("complete")) is not bool or not isinstance(row.get("members"), list)
                    or len(row["members"]) < 2):
                raise ValueError("Known references require unique IDs, source and explicit completeness")
            if "exclusion_eligible" in row and type(row["exclusion_eligible"]) is not bool:
                raise ValueError("Reference exclusion eligibility must be boolean")
            if (not row["complete"] and row.get("exclusion_eligible") and
                    row.get("reference_kind") != "observed_member_set"):
                raise ValueError("Only declared observed member sets may exclude incomplete references")
            if "reference_kind" in row and (not isinstance(row["reference_kind"], str) or not row["reference_kind"]):
                raise ValueError("Reference kind must be a nonempty string")
            identifiers.add(row["reference_id"])
            for member in row["members"]:
                if not isinstance(member, dict):
                    raise ValueError("Reference members must be objects")
                protein = member.get("protein_sha256")
                if not valid_sha(protein) or not isinstance(member.get("qualified_profiles"), list):
                    raise ValueError("Known members require protein identity and qualified profile lists")
                if any(not isinstance(p, str) or not p.strip() or p.upper() in {"UNKNOWN", "UNMAPPED"}
                       for p in member["qualified_profiles"]):
                    raise ValueError("Unknown placeholders cannot establish homology")
                profiles = frozenset(member["qualified_profiles"])
                if relationships.profiles.setdefault(protein, profiles) != profiles:
                    raise ValueError("Reference/native profile assignments disagree")
            references.append(row)
    if not references:
        raise ValueError("Empty known reference input; exclusion must not be silently skipped")
    if len({r["reference_id"] for r in references}) != len(references):
        raise ValueError("Duplicate reference IDs")
    return references


def literature_references(row, relationships):
    """Adapt stated locus sets; alternative products are choices, not extra genes."""
    if (not isinstance(row.get("module_id"), str) or not row["module_id"] or
            not isinstance(row.get("reference_members"), list) or
            not isinstance(row.get("evidence_citations"), list) or not row["evidence_citations"] or
            not row.get("member_set_scope")):
        raise ValueError("Literature references need identity, members, citations and scope")
    choices = []
    for member in row["reference_members"]:
        if (not isinstance(member, dict) or not isinstance(member.get("products"), list) or
                any(not isinstance(p, dict) or not valid_sha(p.get("aa_sha256")) for p in member["products"])):
            raise ValueError("Invalid literature member product identities")
        proteins = sorted({p["aa_sha256"] for p in member["products"]})
        if not proteins or any(not valid_sha(p) for p in proteins):
            raise ValueError("Literature member lacks full protein identities")
        choices.append(proteins)
    if len(choices) < 2:
        return []
    if np.prod([len(c) for c in choices], dtype=object) > 1024:
        raise ValueError("Literature alternative-product budget exceeded")
    eligible = row.get("reference_set_precision_recall_f1_enabled")
    if type(eligible) is not bool or row.get("membership_authority") != "published_member_statement":
        raise ValueError("Explicit published reference-set authority required")
    return [dict(reference_id=f"{row['module_id']}:products:{i}",
        source=dict(module_id=row["module_id"], citations=row["evidence_citations"],
                    scope=row["member_set_scope"], policy="stated_reference_set_not_whole_neighborhood"),
        complete=eligible, members=[dict(protein_sha256=p,
            qualified_profiles=sorted(relationships.profiles.get(p, ()))) for p in proteins])
        for i, proteins in enumerate(product(*choices))]


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def build_catalogue(candidates, loci, references, relationships: Relationships,
                    config=CatalogueConfig()):
    """Group unmatched sets around fixed representatives using direct one-to-one matches."""
    candidates = sorted(candidates, key=lambda c: (c["genome"], tuple(c["members"])))
    identities = [c["instance_id"] for c in candidates]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate predicted instances")
    protein_sets = [tuple(loci[uid]["protein_sha256"] for uid in c["members"]) for c in candidates]
    index = SetIndex(protein_sets, relationships)
    known = defaultdict(list)
    pair_checks = 0

    def match(left, right, ambiguity=False):
        nonlocal pair_checks
        pair_checks += 1
        if pair_checks > config.max_pair_checks:
            raise RuntimeError("Complete-match budget exceeded; results not silently truncated")
        return complete_assignment(left, right, relationships, ambiguity=ambiguity)

    for ref in references:
        if not ref.get("exclusion_eligible", ref["complete"]):
            continue
        proteins = tuple(m["protein_sha256"] for m in ref["members"])
        for i in index.query(proteins):
            if match(proteins, protein_sets[i]) is not None:
                known[i].append(ref["reference_id"])
    patterns, memberships, mappings = [], [], []
    represented = set()
    for seed, candidate in enumerate(candidates):
        if seed in known or seed in represented:
            continue
        pattern_id = "MPP-" + digest_object(candidate["instance_id"])[:24]
        hits = []
        for i in index.query(protein_sets[seed]):
            result = match(protein_sets[seed], protein_sets[i], True)
            if result is not None:
                hits.append((i, result))
        if not any(i == seed for i, _ in hits):
            raise ValueError("Pattern lost its source instance")
        represented.update(i for i, _ in hits)
        by_cluster = defaultdict(list)
        for i, (assignment, feasible) in hits:
            target = candidates[i]
            dark = np.array([loci[u]["baseline_unresolved"] for u in target["members"]], bool)
            lower = np.array([dark[columns].all() for columns in feasible], float)
            upper = np.array([dark[columns].any() for columns in feasible], float)
            by_cluster[target["cluster"]].append((float(dark.mean()), lower, upper))
            memberships.append(dict(pattern_id=pattern_id, instance_id=target["instance_id"],
                                    is_representative=i == seed))
            for pos, j in enumerate(assignment):
                possible = np.flatnonzero(feasible[pos]).tolist()
                mappings.append(dict(pattern_id=pattern_id, instance_id=target["instance_id"],
                    template_position=pos, source_locus=candidate["members"][pos],
                    target_locus=target["members"][j],
                    possible_target_loci=[target["members"][k] for k in possible],
                    unambiguous=len(possible) == 1))
        darkness = float(np.mean([np.mean([r[0] for r in rows]) for rows in by_cluster.values()]))
        lower = np.mean([np.mean([r[1] for r in rows], axis=0) for rows in by_cluster.values()], axis=0)
        upper = np.mean([np.mean([r[2] for r in rows], axis=0) for rows in by_cluster.values()], axis=0)
        support_failures, annotation_failures = [], []
        if len(by_cluster) < config.minimum_clusters:
            support_failures.append("insufficient_clusters")
        if darkness < config.minimum_dark_fraction:
            annotation_failures.append("insufficient_dark_fraction")
        if int((lower >= config.position_dark_fraction).sum()) < config.minimum_dark_positions:
            annotation_failures.append("insufficient_unambiguous_dark_positions")
        failures = support_failures + (annotation_failures if config.filter_annotations else [])
        patterns.append(dict(pattern_id=pattern_id, representative_instance=candidate["instance_id"],
            members=len(candidate["members"]), instances=len(hits), supporting_clusters=len(by_cluster),
            dark_fraction=darkness, position_dark_lower=lower.tolist(), position_dark_upper=upper.tolist(),
            candidate_dark=not (support_failures or annotation_failures),
            selected=not failures, failed_conditions=failures,
            annotation_failed_conditions=annotation_failures,
            known_target_instances=sum(i in known for i, _ in hits)))
    instances = []
    reference_kinds = {r["reference_id"]: r.get("reference_kind", "supplied_reference") for r in references}
    for i, c in enumerate(candidates):
        instances.append(dict(c, known_reference_ids=sorted(known[i]),
                              known_reference_kinds=sorted({reference_kinds[r] for r in known[i]}),
                              known=bool(known[i]), provenance="model_prediction",
                              dark_fraction=float(np.mean([loci[u]["baseline_unresolved"] for u in c["members"]]))))
    return dict(patterns=patterns, instances=instances, memberships=memberships, mappings=mappings,
                counts=dict(predicted_instances=len(instances), known_instances=sum(bool(v["known"]) for v in instances),
                    remaining_instances=sum(not v["known"] for v in instances), patterns=len(patterns),
                    dark_patterns=sum(p["candidate_dark"] for p in patterns),
                    selected_patterns=sum(p["selected"] for p in patterns),
                    annotation_filter_applied=config.filter_annotations,
                    cross_genome_scope="predicted_member_sets_only", complete_pair_checks=pair_checks,
                    complete_references=sum(r["complete"] for r in references),
                    partial_references=sum(not r["complete"] for r in references),
                    observed_references=sum(r.get("reference_kind") == "observed_member_set" for r in references)))


def export_catalogue(result, loci, out):
    out = Path(out)

    def table(name, rows, fields):
        write_table(out / name, ({k: json.dumps(r[k], sort_keys=True) if isinstance(r[k], (list, dict))
                                  else r[k] for k in fields} for r in rows), fields)

    pfields = ["pattern_id", "representative_instance", "members", "instances", "supporting_clusters",
               "dark_fraction", "position_dark_lower", "position_dark_upper", "candidate_dark",
               "failed_conditions", "known_target_instances", "selected", "annotation_failed_conditions"]
    table("patterns.tsv", result["patterns"], pfields)
    table("dark-patterns.tsv", [r for r in result["patterns"] if r["candidate_dark"]], pfields)
    table("selected-patterns.tsv", [r for r in result["patterns"] if r["selected"]], pfields)
    ifields = ["instance_id", "genome", "cluster", "members", "score", "supporting_windows",
               "covering_windows", "window_support", "auxiliary_type_bits", "known",
               "known_reference_ids", "known_reference_kinds", "provenance", "dark_fraction"]
    table("instances.tsv", result["instances"], ifields)
    table("known-instances.tsv", [r for r in result["instances"] if r["known"]], ifields)
    table("pattern-instances.tsv", result["memberships"], ["pattern_id", "instance_id", "is_representative"])
    member_fields = ["instance_id", "locus_uid", "genome", "cluster", "native_rank", "protein_id",
                     "protein_sha256", "start", "end", "strand", "qualified_profiles", "baseline_unresolved",
                     "baseline_reason", "baseline_rule_id", "product", "annotation_source", "annotation_version"]
    table("members.tsv", (dict(loci[uid], instance_id=c["instance_id"])
                          for c in result["instances"] for uid in c["members"]), member_fields)
    with (out / "member-mappings.jsonl").open("w") as stream:
        for row in result["mappings"]:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
