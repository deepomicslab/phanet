import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse
from torch.utils.data import DataLoader

from phanet.cli import main
from phanet.datasets.schema import Dataset, Genome, Protein
from phanet.datasets.splits import SplitManifest, dataset_identity
from phanet.learning.data import prepare_training, load_training
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.learning.training import TrainConfig, Trainer
from phanet.network.build import reference_records, profile_layer, structure_layer, indexed_structure_layer, merge_layers, reference_support
from phanet.representations.embed import EmbeddingStore


@pytest.fixture
def inputs(tmp_path):
    genomes = [Genome(f"g{i}", f"g{i}", 600, "test") for i in range(6)]
    proteins = [Protein(f"g{i}:p{j}", f"g{i}", j*60, j*60+60, 1, j,
                        "M" + "ACDEFG"[i]*10 + "HIKL"[j]*9,
                        labels=("Lysis",) if j % 2 else ()) for i in range(6) for j in range(4)]
    data = Dataset(genomes, proteins)
    data.save(tmp_path / "input")
    rng = np.random.default_rng(42)
    store = EmbeddingStore.write(tmp_path / "vectors", [p.protein_id for p in proteins],
                                 [p.sequence for p in proteins], rng.normal(size=(len(proteins), 1280)), model="esm2_t33_650M_UR50D", config={"full_length": True})
    return data, store, tmp_path


def test_training_cli_and_actual_update(inputs):
    data, store, root = inputs
    assert main(["prepare-training", "--dataset", str(root / "input"), "--embeddings", str(store.directory),
                 "--out", str(root / "training")]) == 0
    train = load_training(root / "training")
    val = load_training(root / "training", "val")
    assert set(train.splits.assignments.values()) == {"train", "val", "test"}
    assert not val[0]["annotation_input"].any()
    batch = next(iter(DataLoader(train, batch_size=2)))
    model = PhaNetModel(ModelConfig.tiny(input_dim=1280))
    config = TrainConfig(total_steps=2, warmup_steps=0, family_weight=0, mask_rate=1.)
    trainer = Trainer(model, config, data_hash=dataset_identity(data))
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    report = trainer.update(batch)
    assert trainer.step == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(before[k], p) for k, p in model.named_parameters())
    assert report
    trainer.checkpoint(root / "checkpoint.pt")
    restored = Trainer.restore(root / "checkpoint.pt", data_hash=dataset_identity(data))
    assert restored.step == 1
    restored.update(batch)
    assert restored.step == 2


def test_training_reuses_embeddings_and_rejects_changes(inputs):
    data, store, root = inputs
    out = root / "training"
    meta = prepare_training(out, dataset=root / "input", embeddings=store.directory)
    assert (out / meta["embeddings"]).resolve() == store.directory.resolve()
    assert not (out / "embeddings").exists()
    with pytest.raises(FileExistsError):
        prepare_training(out, dataset=root / "input", embeddings=store.directory)
    with (out / "dataset/annotations.tsv").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="checksum"):
        load_training(out)


def test_wrong_embedding_sequence_fails(inputs):
    data, store, root = inputs
    data.proteins[0].sequence = "M" + "Y"*19
    data.save(root / "input")
    with pytest.raises(ValueError, match="another sequence"):
        prepare_training(root / "bad", dataset=root / "input", embeddings=store.directory)
    assert not (root / "bad/training.json").exists()


def test_supplied_splits_and_cluster_leakage(inputs):
    data, store, root = inputs
    for g in data.genomes:
        g.cluster_id = "same"
    data.save(root / "input")
    splits = SplitManifest("annotation", "genome", {f"g{i}": ("train", "val", "test")[i % 3] for i in range(6)}, 7)
    splits.save(root / "splits.json")
    with pytest.raises(ValueError, match="clusters cross"):
        prepare_training(root / "bad", dataset=root / "input", embeddings=store.directory, splits=root / "splits.json")


def test_genbank_import_command(tmp_path):
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    record = SeqRecord(Seq("ATG"*25), id="phage", description="phage genome")
    record.annotations["molecule_type"] = "DNA"
    record.features = [SeqFeature(FeatureLocation(0, 60, strand=1), type="CDS",
                                  qualifiers={"translation": ["M"*20], "product": ["holin"]})]
    SeqIO.write(record, tmp_path / "input.gbk", "genbank")
    assert main(["import-data", "--genbank", str(tmp_path / "input.gbk"), "--out", str(tmp_path / "imported")]) == 0
    data = Dataset.load(tmp_path / "imported/dataset")
    assert data.proteins[0].start == 0 and data.proteins[0].labels == ("Lysis",)


def test_protein_import_requires_coordinates(tmp_path):
    assert main(["import-data", "--proteins", str(tmp_path / "input.faa"), "--out", str(tmp_path / "out")]) != 0


def test_deduplicate_references_and_conflicting_labels(inputs):
    data, _, _ = inputs
    original = data.proteins[0]
    duplicate = replace(original, protein_id="extra", genome_id="g1", order=4, start=240, end=300, labels=("Lysis",))
    data.proteins.append(duplicate)
    sequences, labels = reference_records(data)
    assert len(sequences) == len(data.proteins)-1 and labels[list(sequences).index(original.sha256)] == 1
    original.labels = ("Replication",)
    with pytest.raises(ValueError, match="conflicting"):
        reference_records(data)


def test_graph_merging_and_diffusion_match_dense_solution():
    a = sparse.csr_matrix([[0., .6, 0], [.6, 0, 0], [0, 0, 0]])
    b = sparse.csr_matrix([[0., .8, .4], [.8, 0, 0], [.4, 0, 0]])
    graph = merge_layers([a, b])
    np.testing.assert_allclose(graph.toarray(), [[0, .7, .4], [.7, 0, 0], [.4, 0, 0]])
    labels = np.array([1, 2, 0], dtype=np.uint16)
    seed = np.column_stack([((labels[:, None] >> np.arange(9)) & 1), labels != 0])
    transition = graph.toarray() / np.asarray(graph.sum(axis=1))
    expected = np.linalg.solve(np.eye(3)-.85*transition, .15*seed)
    np.testing.assert_allclose(reference_support(graph, labels), expected, atol=1e-6)
    isolated = reference_support(sparse.csr_matrix((1, 1)), np.array([1], np.uint16))
    np.testing.assert_array_equal(isolated[0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])


def test_profile_duplicates_max_coverage_and_no_self_edges():
    hits = [dict(protein_id=p, family_id="family", coverage=c) for p, c in (("a", .6), ("a", .9), ("b", .8))]
    graph, families, arrays = profile_layer(["a", "b", "c"], hits)
    np.testing.assert_allclose(graph.toarray(), [[0, .8, 0], [.8, 0, 0], [0, 0, 0]])
    assert families == ["family"] and len(arrays["profile_indices"]) == 2


def test_structure_exact_neighbors_and_symmetry():
    vectors = np.array([[1, 0], [.8, .6], [-1, 0]], dtype=np.float32)
    graph = structure_layer(["a", "b", "c"], vectors)
    np.testing.assert_allclose(graph.toarray(), [[0, .8, 0], [.8, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize("backend", ["faiss", "faiss-hnsw"])
def test_indexed_structure_small_graph_matches_exact(backend):
    pytest.importorskip("faiss")
    vectors = np.array([[1, 0], [.8, .6], [-1, 0]], dtype=np.float32)
    ids = ["a", "b", "c"]
    np.testing.assert_allclose(indexed_structure_layer(ids, vectors, backend).toarray(),
                               structure_layer(ids, vectors).toarray(), atol=1e-6)


@pytest.mark.parametrize("version", [1, 2])
def test_reference_bundle_versions_and_integrity(tmp_path, version):
    from phanet.common.io import digest_file, write_json
    from phanet.common.constants import CATEGORIES
    from phanet.datasets.schema import sequence_hash
    from phanet.network.resources import NetworkBundle
    from phanet.network.search import write_fasta
    sequences = dict(sorted({sequence_hash(s): s for s in ("MAA", "MCC")}.items()))
    ids = list(sequences)
    graph = sparse.csr_matrix([[0., .8], [.8, 0.]])
    vectors = np.zeros((2, 512), np.float32); vectors[:, 0] = 1
    labels = np.array([1, 0], np.uint16)
    arrays = dict(ids=np.array(ids, dtype="S64"), labels=labels, self_bits=np.array([80., 90.]),
                  vectors=vectors, support=reference_support(graph, labels), graph_data=graph.data,
                  graph_indices=graph.indices, graph_indptr=graph.indptr,
                  profile_indices=np.array([], np.int64), profile_indptr=np.array([0], np.int64),
                  profile_coverage=np.array([], np.float64))
    for name, value in arrays.items():
        np.save(tmp_path / (name + ".npy"), value)
    write_fasta(tmp_path / "references.faa", sequences)
    write_json(tmp_path / "families.json", [])
    meta = dict(schema_version=version, categories=list(CATEGORIES), method="three-channel-fixed-label-ppr-v1",
                embedding="prott5-tmvec:full_length:float16:l2", profile_Z=532910,
                profile_count=38880 if version == 1 else 1,
                files={p.name: dict(bytes=p.stat().st_size, sha256=digest_file(p)) for p in tmp_path.iterdir()})
    write_json(tmp_path / "network.json", meta)
    bundle = NetworkBundle(tmp_path, progress=lambda _: None)
    support, allowed, _ = bundle.masked_support({ids[0]}, progress=lambda _: None)
    assert not support.any() and not allowed[0]
    with (tmp_path / "references.faa").open("a") as stream:
        stream.write("changed\n")
    with pytest.raises(ValueError, match="integrity"):
        NetworkBundle(tmp_path, progress=lambda _: None)


def test_training_raw_protein_tables(inputs):
    import csv
    data, store, root = inputs
    genes = root / "genes.tsv"
    with genes.open("w") as stream:
        fields = ["protein_id", "genome_id", "start", "end", "strand", "order", "product"]
        writer = csv.DictWriter(stream, fields, delimiter="\t"); writer.writeheader()
        for p in data.proteins:
            writer.writerow({**{k: getattr(p, k) for k in fields}, "product": "holin" if p.labels else ""})
    assert main(["prepare-training", "--proteins", str(root / "input/proteins.faa"),
                 "--genes", str(genes), "--genomes", str(root / "input/genomes.tsv"),
                 "--embeddings", str(store.directory), "--out", str(root / "training")]) == 0
    loaded = load_training(root / "training")
    assert len(loaded.dataset.proteins) == len(data.proteins)
    assert all(p.start == q.start and p.sequence == q.sequence for p, q in zip(data.proteins, loaded.dataset.proteins))


@pytest.mark.parametrize("command", ["import-data", "build-network", "prepare-training"])
def test_cli_help(command):
    with pytest.raises(SystemExit) as exc:
        main([command, "--help"])
    assert exc.value.code == 0
