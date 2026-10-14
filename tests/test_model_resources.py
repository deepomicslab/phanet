import hashlib
import zipfile

import pytest
from phanet import model_resources as m
from phanet.cli import main


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    archive_dir = tmp_path / 'archives'
    archive_dir.mkdir()
    path = archive_dir / 'tiny.zip'
    files = {'model.json': b'{"weights":"weights.pt"}', 'weights.pt': b'tensor bytes'}
    with zipfile.ZipFile(path, 'w') as z:
        for name, data in files.items():
            z.writestr('tiny/' + name, data)
    record = {'filename': 'tiny.zip', 'prefix': 'tiny', 'url': 'https://example.org/model',
              'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes()),
              'files': {n: {'bytes': len(b), 'sha256': sha(b)} for n, b in files.items()}}
    monkeypatch.setattr(m, 'MODELS', {'rbp': record})
    return archive_dir, record


def test_import_cli_reuses_and_refuses_changed_install(tmp_path, bundle):
    archives, _ = bundle
    out = tmp_path / 'models'
    args = ['setup-models', '--models', 'rbp', '--out', str(out), '--archives', str(archives)]
    assert main(args) == 0
    model = out / 'rbp/weights.pt'
    old = model.stat().st_mtime_ns
    assert main(args) == 0
    assert model.stat().st_mtime_ns == old
    model.write_bytes(b'damaged')
    assert main(args) == 1
    assert model.read_bytes() == b'damaged'
    assert not (out / '.setup.lock').exists()


def test_wrong_archive_never_installed(tmp_path, bundle):
    archives, _ = bundle
    (archives / 'tiny.zip').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='Integrity'):
        m.setup_models(tmp_path / 'models', models=['rbp'], archives=archives)
    assert not (tmp_path / 'models/rbp').exists()


@pytest.mark.parametrize('extra', ['../escape', 'tiny/extra', 'tiny/model.json'])
def test_unsafe_extra_duplicate_members_rejected(tmp_path, bundle, extra):
    archives, record = bundle
    p = archives / 'tiny.zip'
    with zipfile.ZipFile(p, 'a') as z:
        z.writestr(extra, b'x')
    record.update(bytes=p.stat().st_size, sha256=sha(p.read_bytes()))
    with pytest.raises(ValueError, match='members'):
        m.setup_models(tmp_path / 'models', models=['rbp'], archives=archives)
    assert not (tmp_path / 'models/rbp').exists()
    assert not (tmp_path / 'escape').exists()


def test_member_checksum_rejected(tmp_path, bundle):
    archives, record = bundle
    record['files']['weights.pt']['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='Integrity'):
        m.setup_models(tmp_path / 'models', models=['rbp'], archives=archives)
    assert not (tmp_path / 'models/rbp').exists()


def test_symlink_destination_refused(tmp_path, bundle):
    archives, _ = bundle
    root = tmp_path / 'models'; root.mkdir()
    (root / 'rbp').symlink_to(tmp_path / 'other', target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        m.setup_models(root, models=['rbp'], archives=archives)
    assert not (tmp_path / 'other').exists()


def test_download_then_offline_cache_and_installed_reuse(tmp_path, bundle, monkeypatch):
    archives, record = bundle
    calls = []
    def download(url, target, size, checksum, progress):
        calls.append(url)
        target.write_bytes((archives / 'tiny.zip').read_bytes())
    monkeypatch.setattr(m, '_download', download)
    root = tmp_path / 'models'
    m.setup_models(root, models=['rbp'])
    m.setup_models(root, models=['rbp'], offline=True)
    assert len(calls) == 1
    other = tmp_path / 'offline'; (other / '.downloads').mkdir(parents=True)
    (other / '.downloads/tiny.zip').write_bytes((archives / 'tiny.zip').read_bytes())
    m.setup_models(other, models=['rbp'], offline=True)
    assert len(calls) == 1


def test_missing_offline_cache_does_not_download(tmp_path, bundle, monkeypatch):
    monkeypatch.setattr(m, '_download', lambda *a: pytest.fail('offline download'))
    with pytest.raises(FileNotFoundError):
        m.setup_models(tmp_path / 'models', models=['rbp'], offline=True)


def test_busy_lock_refuses_writes(tmp_path, bundle):
    archives, _ = bundle
    root = tmp_path / 'models'; root.mkdir(); (root / '.setup.lock').write_text('123')
    with pytest.raises(RuntimeError, match='locked'):
        m.setup_models(root, models=['rbp'], archives=archives)
    assert (root / '.setup.lock').read_text() == '123'
