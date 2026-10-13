import numpy as np
import pytest
import torch

from phanet.common.constants import CATEGORIES
from phanet.datasets.annotations import AnnotationMapper
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.learning.training import TrainConfig
from phanet.network.fusion import fuse


def test_public_ontology_is_native_table_s3_nine_axis_contract():
    assert CATEGORIES == (
        "Lysis", "Integration", "Replication", "tRNA-related", "Regulation",
        "Packaging", "Assembly", "Infection", "Immune",
    )
    mapper = AnnotationMapper()
    assert mapper.map("terminase large subunit") == ("Packaging",)
    assert mapper.map("hypothetical protein") == ()


def test_model_and_fusion_use_the_public_label_width():
    model = PhaNetModel(ModelConfig(input_dim=4, hidden_dim=8, num_layers=1, num_heads=2, max_context=8))
    assert model.annotation_head.out_features == len(CATEGORIES) == 9
    scores = fuse(np.zeros((2, 9)), np.ones((2, 9)), np.array([0.0, 1.0]))
    np.testing.assert_array_equal(scores, np.array([[1.0] * 9, [0.0] * 9]))


def test_ten_column_scores_are_rejected():
    with pytest.raises(ValueError, match="N x 9"):
        fuse(np.zeros((2, 10)), np.ones((2, 10)), np.array([0.0, 1.0]))


def test_ten_output_checkpoint_is_rejected_even_with_non_strict_loading():
    model = PhaNetModel(ModelConfig(input_dim=4, hidden_dim=8, num_layers=1, num_heads=2, max_context=8))
    state = model.state_dict()
    state["annotation_head.weight"] = torch.zeros(10, 8)
    state["annotation_head.bias"] = torch.zeros(10)
    state["annotation_projection.weight"] = torch.zeros(8, 10)
    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(state, strict=False)


@pytest.mark.parametrize("parameter", ["learning_rate", "weight_decay", "gradient_clip", "family_temperature"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_training_parameters_are_rejected(parameter, value):
    with pytest.raises(ValueError, match="finite"):
        TrainConfig(**{parameter: value})
