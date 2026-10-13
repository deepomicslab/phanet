from types import SimpleNamespace
import sys

import numpy as np
import pytest

from phanet.representations.embed import EmbeddingStore, ESMConfig, esm_vectors


def test_vectors_from_reused_buffer_keep_each_protein_value(tmp_path):
    def vectors():
        buffer = np.zeros(3, dtype=np.float32)
        for value in (1, 2, 3):
            buffer[:] = value
            yield buffer

    store = EmbeddingStore.write(tmp_path, ["a", "b", "c"], ["MA", "MC", "MD"],
                                 vectors(), model="example")
    np.testing.assert_array_equal(store.get(["c", "a", "b", "a"]),
                                  [[3] * 3, [1] * 3, [2] * 3, [1] * 3])


def test_incomplete_store_is_not_overwritten(tmp_path):
    path = tmp_path / "vectors-00000.npy"
    np.save(path, np.array([[42.0, 17.0]], dtype=np.float16))
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match="fresh directory"):
        EmbeddingStore.write(tmp_path, ["a"], ["MA"], [[1.0, 2.0]], model="example")
    assert path.read_bytes() == before
    assert not (tmp_path / "embeddings.json").exists()


def test_esm_does_not_download_without_explicit_opt_in(monkeypatch):
    def forbidden_download():
        pytest.fail("A download was attempted without opt-in")

    monkeypatch.delenv("PHANET_ESM2_MODEL_PATH", raising=False)
    fake = SimpleNamespace(pretrained=SimpleNamespace(esm2_t33_650M_UR50D=forbidden_download))
    monkeypatch.setitem(sys.modules, "esm", fake)
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        next(esm_vectors(["MA"], ESMConfig()))
