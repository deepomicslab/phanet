"""Downstream contracts: native context, exact members, input identity and failures."""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json, write_json
from phanet.datasets.schema import Dataset, Genome, Protein
from phanet.datasets.splits import MASKED, VISIBLE
from phanet.downstream import bundles
from phanet.downstream.bundles import GOModel
from phanet.downstream.module_decode import compact_candidates
from phanet.downstream.module_prediction import PredictionConfig, predict_genome
from phanet.downstream.predict import go_batches, module_rows, predict, validate_embeddings
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.representations.embed import EmbeddingStore


def native():
    return Dataset([Genome("g", "g", 900, "test"), Genome("empty", "empty", 100, "test")],
                   [Protein(f"p{i}", "g", i*90, i*90+60, 1 if i % 2 else -1, i,
                            "M" + "ACD"[i]*19, labels=(CATEGORIES[i],)) for i in range(3)])


def embedding_store(tmp_path, data):
    return EmbeddingStore.write(tmp_path / "vectors", [p.protein_id for p in data.proteins],
                               [p.sequence for p in data.proteins],
                               [np.full(1280, i/10, np.float32) for i in range(len(data.proteins))],
                               model="esm2_t33_650M_UR50D", config={"full_length": True})


def test_seventh_type_bit_and_member_gap():
    out = dict(object_logits=torch.full((1, 3, 1), -100.),
               mask_logits=torch.full((1, 3, 1, 3), -100.),
               type_logits=torch.full((1, 3, 1, 7), -100.), valid=torch.ones(1, 3, dtype=torch.bool))
    out["object_logits"][0, 0, 0] = 100
    out["mask_logits"][0, 0, 0, 2] = 100
    out["type_logits"][0, 0, 0, 6] = 100
    result = compact_candidates(out)
    assert result.shape == (1, 5)
    assert result[0, 2] == 5 and result[0, 4] == 64


def test_full_span_qc_and_empty_contigs(tmp_path, monkeypatch):
    data = native()
    store = embedding_store(tmp_path, data)
    data.proteins[1].partial = True
    import phanet.downstream.predict as api
    monkeypatch.setattr(api, "predict_genome", lambda *a, **kw: ([dict(members=[0, 2], score=.9,
        auxiliary_type_bits=64, supporting_windows=1, covering_windows=1, window_support=1)],
        dict(windows=1, distinct_before_support=1, retained=1)))
    rows = list(module_rows(data, store, None, PredictionConfig()))
    assert rows[0][1][0]["members"] == ["p0", "p2"]
    assert not rows[0][1][0]["span_qc_passed"]
    assert rows[0][2]["span_qc_rejected"] == 1
    assert rows[1][0] == "empty" and rows[1][2]["windows"] == 0


def test_embedding_identity_and_contract(tmp_path):
    data = native()
    store = embedding_store(tmp_path, data)
    validate_embeddings(data, store)
    data.proteins[0].sequence = "MAAAA"
    with pytest.raises(ValueError, match="mismatched"):
        validate_embeddings(data, store)
    store.metadata["config"]["full_length"] = False
    with pytest.raises(ValueError, match="full-length"):
        validate_embeddings(data, store)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_go_masking_and_batch_invariance(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(0)
    data = native()
    store = embedding_store(tmp_path, data)
    model = GOModel(PhaNetModel(ModelConfig.tiny()), 2).eval().to(device)
    batches = []
    handle = model.register_forward_pre_hook(lambda m, args: batches.append(args))
    first = list(go_batches(data, store, model, batch_size=1, device=device))
    handle.remove()
    for args in batches:
        (_, ann, state, _, valid), offsets = args
        target = offsets[0]
        assert state[0, target] == MASKED and not ann[0, target].any()
        assert int((state[0] == VISIBLE).sum()) == 2
        assert int(valid.sum()) == 3
    second = list(go_batches(data, store, model, batch_size=3, device=device))
    assert [pid for ids, _ in first for pid in ids] == second[0][0]
    np.testing.assert_allclose(np.concatenate([p for _, p in first]), second[0][1], atol=2e-6)


def go_bundle(path):
    path.mkdir(parents=True)
    model = GOModel(PhaNetModel(ModelConfig.tiny()), 2)
    torch.save(model.state_dict(), path / "weights.pt")
    write_json(path / "model.json", dict(schema_version=1, task="go", categories=list(CATEGORIES),
        embedding=bundles.EMBEDDING, model_config=asdict(model.encoder.config), weights="weights.pt",
        weights_sha256=digest_file(path / "weights.pt"), namespace="CC", terms=["GO:0000001", "GO:0000002"],
        threshold=.5, annotation_mode="neighbors_target_masked"))


def test_go_cli_export_and_no_overwrite(tmp_path):
    from phanet.cli import main
    data = native()
    data.save(tmp_path / "data")
    store = embedding_store(tmp_path, data)
    go_bundle(tmp_path / "models/CC")
    args = ["predict-go", "--dataset", str(tmp_path / "data"), "--embeddings", str(store.directory),
            "--model", str(tmp_path / "models"), "--namespace", "CC", "--out", str(tmp_path / "out")]
    assert main(args) == 0
    assert read_json(tmp_path / "out/status.json")["state"] == "complete"
    rows = (tmp_path / "out/go_CC.tsv").read_text().splitlines()
    assert len(rows) == 4 and rows[1].startswith("p0\t")
    before = digest_file(tmp_path / "out/go_CC.tsv")
    assert main(args) == 1
    assert digest_file(tmp_path / "out/go_CC.tsv") == before


def test_model_contract_rejects_wrong_vocabulary_and_weights(tmp_path):
    root = tmp_path / "CC"
    go_bundle(root)
    meta = read_json(root / "model.json")
    meta["terms"] = ["GO:0000001", "GO:0000001"]
    write_json(root / "model.json", meta)
    with pytest.raises(ValueError, match="vocabulary"):
        bundles.load(root, "go")
    meta["terms"] = ["GO:0000001", "GO:0000002"]
    write_json(root / "model.json", meta)
    with (root / "weights.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        bundles.load(root, "go")


def test_failed_run_retains_status(tmp_path, monkeypatch):
    data = native()
    data.save(tmp_path / "data")
    store = embedding_store(tmp_path, data)
    go_bundle(tmp_path / "models/CC")
    monkeypatch.setattr(bundles, "load", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("load failure")))
    with pytest.raises(RuntimeError, match="load failure"):
        predict("go", tmp_path / "out", tmp_path / "models", dataset=tmp_path / "data",
                embeddings=store.directory, namespaces=["CC"])
    assert read_json(tmp_path / "out/status.json")["state"] == "failed"


def test_invalid_genome_vectors_rejected():
    model = SimpleNamespace(head=SimpleNamespace(config=SimpleNamespace(width=30, input_dim=2)))
    with pytest.raises(ValueError, match="mapping"):
        predict_genome(model, np.zeros((2, 2)), [0, 2], [1, 1])


def test_nucleotide_preparation_then_shared_embeddings(tmp_path, monkeypatch):
    import phanet.downstream.predict as api
    data = native()
    store = embedding_store(tmp_path, data)
    go_bundle(tmp_path / "models/CC")
    checkpoint = tmp_path / "esm.pt"
    checkpoint.write_bytes(b"test checkpoint; embedding operation replaced")
    order = []
    def prepare(source, output, config, **kwargs):
        order.append("prepare")
        data.save(output / "prepared/dataset")
    def embed(ids, sequences, output, config):
        order.append("esm")
        assert ids == [p.protein_id for p in data.proteins] and config.full_length
        return store
    real_load = bundles.load
    def load(*args, **kwargs):
        assert order == ["prepare", "esm", "released"]
        return real_load(*args, **kwargs)
    monkeypatch.setattr(api, "prepare", prepare)
    monkeypatch.setattr(api, "embed_esm", embed)
    monkeypatch.setattr(api, "_release_stage_memory", lambda device: order.append("released"))
    monkeypatch.setattr(bundles, "load", load)
    result = predict("go", tmp_path / "out", tmp_path / "models", input_fasta=tmp_path / "input.fna",
                     config=object(), esm_checkpoint=checkpoint, namespaces=["CC"])
    assert result["proteins"] == 3
    assert read_json(tmp_path / "out/status.json")["state"] == "complete"


def test_empty_native_dataset_go_outputs_headers(tmp_path):
    data = Dataset([Genome("empty", "empty", 100, "test")], [])
    data.save(tmp_path / "data")
    go_bundle(tmp_path / "models/CC")
    result = predict("go", tmp_path / "out", tmp_path / "models", dataset=tmp_path / "data", namespaces=["CC"])
    assert result["genomes"] == 1 and result["proteins"] == 0
    assert len((tmp_path / "out/go_CC.tsv").read_text().splitlines()) == 1
