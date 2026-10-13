"""Boundary placement, annotation isolation and binary inference packaging."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from phanet.cli import main
from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json, write_json
from phanet.datasets.schema import Dataset, Genome, Protein
from phanet.datasets.splits import MASKED, PADDING
from phanet.downstream import bundles
from phanet.downstream.binary_context import context_batch, probability_batches
from phanet.downstream.bundles import BinaryContextModel
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.representations.embed import EmbeddingStore


def inputs(tmp_path, count=40):
    data = Dataset([Genome("g", "g", count*90, "test"), Genome("other", "other", 90, "test")],
        [Protein(f"p{i}", "g", i*90, i*90+60, 1 if i%2 else -1, i, "MAAAAA",
                 labels=(CATEGORIES[0],)) for i in range(count)] +
        [Protein("isolated", "other", 0, 60, 1, 0, "MCCCCC")])
    store = EmbeddingStore.write(tmp_path / "vectors", [p.protein_id for p in data.proteins],
        [p.sequence for p in data.proteins], [np.full(1280, i+1, np.float32) for i in range(len(data.proteins))],
        model="esm2_t33_650M_UR50D", config={"full_length": True})
    return data, store


def model_bundle(path):
    path.mkdir()
    model = BinaryContextModel(PhaNetModel(ModelConfig.tiny()))
    torch.save(model.state_dict(), path / "weights.pt")
    write_json(path / "model.json", dict(schema_version=1, task="vf", categories=list(CATEGORIES),
        embedding=bundles.EMBEDDING, model_config=asdict(model.encoder.config), weights="weights.pt",
        weights_sha256=digest_file(path / "weights.pt"), threshold=.6, threshold_source="validation",
        annotation_mode="all_masked", context_mode="fixed_center_padded", target_offset=14))


def test_native_boundaries_keep_target_at_14_and_do_not_cross_contigs(tmp_path):
    data, store = inputs(tmp_path)
    batch, offsets = context_batch(data, ["p0", "p20", "p39", "isolated"], store)
    x, annotation, state, strand, valid = batch
    assert offsets == [14]*4 and not annotation.any()
    assert valid.sum(1).tolist() == [16, 30, 15, 1]
    assert x[:, 14, 0].tolist() == [1, 21, 40, 41]
    assert x[0, 14:, 0].tolist() == list(range(1, 17))
    assert x[1, :, 0].tolist() == list(range(7, 37))
    assert x[2, :15, 0].tolist() == list(range(26, 41))
    assert not x[~valid].any() and not strand[~valid].any()
    assert (state[valid] == MASKED).all() and (state[~valid] == PADDING).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_predictions_ignore_all_functional_labels_and_batch_size(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data, store = inputs(tmp_path, count=3)
    model = BinaryContextModel(PhaNetModel(ModelConfig.tiny())).eval().to(device)
    first = list(probability_batches(data, store, model, batch_size=1, device=device))
    for p in data.proteins:
        p.labels = tuple(CATEGORIES)
    second = list(probability_batches(data, store, model, batch_size=4, device=device))
    np.testing.assert_allclose(np.concatenate([v for _, v in first]), second[0][1], atol=2e-6)
    assert [pid for ids, _ in first for pid in ids] == second[0][0]


@pytest.mark.parametrize("change", ["visible_annotation", "wrong_offset", "visible_state"])
def test_model_rejects_go_style_context(tmp_path, change):
    data, store = inputs(tmp_path, count=3)
    batch, offsets = context_batch(data, ["p1"], store)
    if change == "visible_annotation": batch[1][0, 13, 0] = 1
    if change == "wrong_offset": offsets[0] = 1
    if change == "visible_state": batch[2][0, 13] = 0
    model = BinaryContextModel(PhaNetModel(ModelConfig.tiny()))
    with pytest.raises(ValueError, match="fully hidden"):
        model(batch, offsets)


def test_cli_native_rows_threshold_and_source_unchanged(tmp_path):
    data, store = inputs(tmp_path, count=3)
    data.save(tmp_path / "data")
    model_bundle(tmp_path / "model")
    before = digest_file(tmp_path / "data/annotations.tsv")
    args = ["predict-vf", "--dataset", str(tmp_path / "data"), "--embeddings", str(store.directory),
            "--model", str(tmp_path / "model"), "--out", str(tmp_path / "out")]
    assert main(args) == 0
    result = read_json(tmp_path / "out/summary.json")
    assert result["proteins"] == 4 and result["target_offset"] == 14 and result["threshold"] == .6
    from phanet.common.io import read_table
    rows = list(read_table(tmp_path / "out/vf.tsv"))
    assert [r["protein_id"] for r in rows] == [p.protein_id for p in data.proteins]
    assert all(int(r["predicted_label"]) == int(float(r["probability"]) >= .6) for r in rows)
    assert digest_file(tmp_path / "data/annotations.tsv") == before
    assert main(args) == 1


@pytest.mark.parametrize("field,value", [("target_offset", 15), ("annotation_mode", "neighbors_target_masked"),
                                        ("threshold", float("nan")), ("threshold_source", "test")])
def test_bundle_rejects_mismatched_input_policy(tmp_path, field, value):
    model_bundle(tmp_path / "model")
    meta = read_json(tmp_path / "model/model.json")
    meta[field] = value
    # Deliberately malformed metadata is test input, not an exported model.
    import json
    (tmp_path / "model/model.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="policy"):
        bundles.load(tmp_path / "model", "vf")


def test_empty_dataset_exports_header(tmp_path):
    Dataset([Genome("empty", "empty", 90, "test")], []).save(tmp_path / "data")
    model_bundle(tmp_path / "model")
    assert main(["predict-vf", "--dataset", str(tmp_path / "data"), "--model", str(tmp_path / "model"),
                 "--out", str(tmp_path / "out")]) == 0
    assert len((tmp_path / "out/vf.tsv").read_text().splitlines()) == 1
