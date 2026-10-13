"""Minimal public inference contracts."""
import numpy as np
import pytest
from phanet.network.annotation import fuse, sequence_vote


def test_fusion_preserves_glm_without_assigned_network_categories():
    glm = np.full((2, 9), .7)
    np.testing.assert_array_equal(fuse(glm, np.full((2, 9), .4), np.ones((2, 9))), glm)


def test_fusion_rejects_old_ten_category_scores():
    with pytest.raises(ValueError):
        fuse(np.zeros((1, 10)), np.zeros((1, 10)), np.zeros((1, 10)))


def test_sequence_vote_reports_reference_evidence():
    hit = dict(reference_id="ref", bits=100, evalue=1e-10, identity=.8,
               query_coverage=.9, reference_coverage=1.)
    result = sequence_vote([hit], {"ref": [1]+[0]*8})
    assert result["references"] == ["ref"]
    assert result["scores"][0] == 1
    assert result["rho"][0] == pytest.approx(.72)
