"""Resource setup interface tests."""
import json
from pathlib import Path

from phanet.cli import main


def test_setup_writes_reusable_config(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("phanet.resources.install_eggnog", lambda path, **kw: calls.append(path))
    config = tmp_path / "phanet.json"
    db = tmp_path / "database"
    assert main(["setup", "--eggnog-data", str(db), "--config", str(config)]) == 0
    value = json.loads(config.read_text())
    assert value["eggnog_data"] == str(db.resolve())
    value["threads"] = 7
    config.write_text(json.dumps(value))
    before = config.read_bytes()
    assert main(["setup", "--config", str(config)]) == 0
    assert config.read_bytes() == before
    assert calls == [str(db.resolve())] * 2


def test_setup_refuses_conflicting_config_before_download(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Must not download")
    monkeypatch.setattr("phanet.resources.install_eggnog", unexpected)
    config = tmp_path / "phanet.json"
    assert main(["init-config", "--eggnog-data", str(tmp_path / "a"), "--out", str(config)]) == 0
    before = config.read_bytes()
    assert main(["setup", "--eggnog-data", str(tmp_path / "b"), "--config", str(config)]) == 1
    assert config.read_bytes() == before
