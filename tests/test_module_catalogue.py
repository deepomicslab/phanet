import hashlib
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import pytest

from phanet.analysis.prediction_pipeline import (
    CatalogueConfig, SetIndex, build_catalogue, complete_assignment, instance_id, read_references,
)
from phanet.analysis.template_matching import Relationships, SequenceHit


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def candidate(genome, proteins, dark, cluster=None):
    members = [f"{genome}:{i}" for i in range(len(proteins))]
    loci = {uid: dict(protein_sha256=p, baseline_unresolved=d) for uid, p, d in zip(members, proteins, dark)}
    row = dict(instance_id=instance_id(genome, members), genome=genome, cluster=cluster or genome,
               members=members, score=.9, supporting_windows=1, covering_windows=1, window_support=1.,
               auxiliary_type_bits=0)
    return row, loci


def test_full_matching_not_shared_subset_or_reused_copy():
    r = Relationships()
    r.profiles = {"a": {"A"}, "a2": {"A"}, "b": {"B"}, "c": {"C"}}
    assert complete_assignment(["a", "b"], ["a2", "b"], r) == [0, 1]
    assert complete_assignment(["a", "a2"], ["a", "b"], r) is None
    assert complete_assignment(["a", "b"], ["a", "b", "c"], r) is None
    assert complete_assignment(["unknown1", "b"], ["unknown2", "b"], r) is None


def test_ambiguity_matches_exhaustive_permutations():
    rng = np.random.default_rng(12)
    for n in range(2, 6):
        for _ in range(25):
            edges = rng.random((n, n)) > .5
            r = Relationships()
            r.score = lambda a, b: (1.,) if edges[int(a[1:]), int(b[1:])] else None
            left, right = [f"a{i}" for i in range(n)], [f"b{i}" for i in range(n)]
            expected = [p for p in permutations(range(n)) if all(edges[i, p[i]] for i in range(n))]
            found = complete_assignment(left, right, r, ambiguity=True)
            if not expected:
                assert found is None
            else:
                assignment, feasible = found
                assert tuple(assignment) in expected
                truth = np.zeros((n, n), bool)
                for p in expected:
                    truth[np.arange(n), p] = True
                np.testing.assert_array_equal(feasible, truth)


def test_index_agrees_with_directed_full_search():
    r = Relationships(sequences={("a", "b"): SequenceHit(1e-10, 80, 100, 100, 1., 1.)})
    r.profiles = {"x": {"X"}, "y": {"X"}}
    sets = [("a", "x"), ("b", "y"), ("b", "q"), ("a", "x", "x")]
    index = SetIndex(sets, r)
    for source in sets:
        indexed = {i for i in index.query(source) if complete_assignment(source, sets[i], r) is not None}
        exhaustive = {i for i, target in enumerate(sets) if complete_assignment(source, target, r) is not None}
        assert indexed == exhaustive
    assert 1 in index.query(sets[0]) and 0 not in index.query(sets[1])


def test_annotated_instances_remain_in_darkness_and_cluster_weighting():
    candidates, loci = [], {}
    for genome, dark, cluster in [("a", [True, True], "c1"), ("b", [True, True], "c1"),
                                  ("c", [False, False], "c2")]:
        row, local = candidate(genome, ["x", "y"], dark, cluster)
        candidates.append(row)
        loci.update(local)
    result = build_catalogue(candidates, loci, [], Relationships(), CatalogueConfig(minimum_clusters=2))
    pattern, = result["patterns"]
    assert pattern["supporting_clusters"] == 2
    assert pattern["dark_fraction"] == .5
    assert not pattern["candidate_dark"]
    assert len(result["instances"]) == 3


def test_complete_known_reference_excludes_whole_set_only():
    rows, loci = [], {}
    for genome, proteins in [("a", ["x", "y"]), ("b", ["x", "y", "z"])]:
        row, local = candidate(genome, proteins, [True] * len(proteins))
        rows.append(row)
        loci.update(local)
    refs = [dict(reference_id="known", complete=True,
                 members=[dict(protein_sha256="x"), dict(protein_sha256="y")])]
    result = build_catalogue(rows, loci, refs, Relationships())
    assert result["counts"]["known_instances"] == 1
    assert len(result["patterns"]) == 1 and result["patterns"][0]["members"] == 3
    refs[0]["complete"] = False
    assert build_catalogue(rows, loci, refs, Relationships())["counts"]["known_instances"] == 0


def test_representative_matching_does_not_chain():
    r = Relationships()
    r.profiles = {"a": {"A"}, "b": {"A", "B"}, "c": {"B"}, "x": {"X"}}
    rows, loci = [], {}
    for genome, p in [("1", "a"), ("2", "b"), ("3", "c")]:
        row, local = candidate(genome, [p, "x"], [True, True])
        rows.append(row)
        loci.update(local)
    result = build_catalogue(rows, loci, [], r)
    assert len(result["patterns"]) == 2
    assert all(p["instances"] == 2 for p in result["patterns"])


def test_same_family_dark_position_ambiguity_is_not_arbitrarily_resolved():
    row, loci = candidate("g", ["a", "b"], [True, False])
    r = Relationships()
    r.profiles = {"a": {"A"}, "b": {"A"}}
    p, = build_catalogue([row], loci, [], r)["patterns"]
    assert p["position_dark_lower"] == [0., 0.]
    assert p["position_dark_upper"] == [1., 1.]


def test_reference_reader_rejects_unknown_placeholders(tmp_path):
    path = tmp_path / "refs.jsonl"
    path.write_text(json.dumps(dict(reference_id="r", complete=True, source="fixture",
        members=[dict(protein_sha256=sha("x"), qualified_profiles=["UNMAPPED"])] * 2)) + "\n")
    with pytest.raises(ValueError, match="placeholders"):
        read_references(path, Relationships())
    path.write_text("")
    with pytest.raises(ValueError, match="Empty"):
        read_references(path, Relationships())


def test_empty_catalogue_and_budget():
    assert build_catalogue([], {}, [], Relationships())["counts"]["patterns"] == 0
    rows, loci = [], {}
    for g in ("a", "b"):
        row, local = candidate(g, ["x", "y"], [True, True])
        rows.append(row)
        loci.update(local)
    with pytest.raises(RuntimeError, match="budget"):
        build_catalogue(rows, loci, [], Relationships(), CatalogueConfig(max_pair_checks=1))


def test_known_targets_still_contribute_to_unknown_seed_darkness():
    relation = SequenceHit(1e-10, 80, 100, 100, 1., 1.)
    r = Relationships(sequences={("a", "b"): relation, ("ref", "b"): relation})
    source, l1 = candidate("1", ["a", "x"], [True, True])
    target, l2 = candidate("2", ["b", "x"], [False, False])
    refs = [dict(reference_id="r", complete=True,
                 members=[dict(protein_sha256="ref"), dict(protein_sha256="x")])]
    result = build_catalogue([source, target], l1 | l2, refs, r)
    pattern, = result["patterns"]
    assert pattern["known_target_instances"] == 1
    assert pattern["instances"] == 2 and pattern["dark_fraction"] == .5


def test_literature_alternative_products_are_not_extra_loci(tmp_path):
    row = dict(module_id="L1", reference_members=[
        dict(products=[dict(aa_sha256=sha("a")), dict(aa_sha256=sha("b"))]),
        dict(products=[dict(aa_sha256=sha("c"))])], reference_set_precision_recall_f1_enabled=True,
        membership_authority="published_member_statement", evidence_citations=[{"id": "fixture"}],
        member_set_scope="defined_fixture_reference")
    path = tmp_path / "targets.jsonl"
    path.write_text(json.dumps(row) + "\n")
    refs = read_references(path, Relationships(), format="literature_targets")
    assert len(refs) == 2 and all(len(r["members"]) == 2 for r in refs)
    assert {r["members"][0]["protein_sha256"] for r in refs} == {sha("a"), sha("b")}
