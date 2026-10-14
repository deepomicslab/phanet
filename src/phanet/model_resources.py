"""Install versioned PhaNet model bundles with archive and member checksums."""
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from phanet.model_catalogue import MODELS
from phanet.resource_setup import checked_path, directory
from phanet.resources import _download, _lock, _verify


def _check(root, record):
    for name, expected in record['files'].items():
        path = checked_path(root / name)
        _verify(path, expected['bytes'], expected['sha256'])


def _extract(archive, stage, record):
    expected = {record['prefix'] + '/' + name: name for name in record['files']}
    with zipfile.ZipFile(archive) as z:
        infos = z.infolist()
        if len(infos) != len(expected) or {i.filename for i in infos} != set(expected):
            raise ValueError('Unexpected, duplicate or missing model archive members')
        for item in infos:
            name = expected[item.filename]
            if (Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name
                    or stat.S_ISLNK(item.external_attr >> 16) or item.is_dir()
                    or item.file_size != record['files'][name]['bytes']):
                raise ValueError('Invalid model archive member: ' + item.filename)
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(item) as inp, target.open('xb') as out:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
    _check(stage, record)


def setup_models(output, *, models, archives=None, offline=False, progress=print):
    """Download or import selected bundles into output/<model>, without replacement."""
    names = list(dict.fromkeys(models))
    if not names or any(n not in MODELS for n in names):
        raise ValueError('Select models from: ' + ', '.join(MODELS))
    root = directory(output)
    local = checked_path(archives) if archives is not None else None
    if local is not None and not local.is_dir():
        raise ValueError('--archives must be an existing directory containing model ZIPs')
    with _lock(root):
        cache = directory(root / '.downloads') if local is None else local
        result = {}
        for name in names:
            record = MODELS[name]
            target = checked_path(root / name)
            if target.exists():
                if not target.is_dir():
                    raise ValueError('Model destination is not a directory: ' + str(target))
                _check(target, record)
                progress('Verified installed model: ' + name)
            else:
                archive = cache / record['filename']
                expanded = sum(f['bytes'] for f in record['files'].values())
                required = expanded + (record['bytes'] if local is None and not archive.exists() else 0)
                if shutil.disk_usage(root).free < required:
                    raise OSError('Insufficient disk space for model: ' + name)
                if local is not None or offline:
                    _verify(archive, record['bytes'], record['sha256'])
                else:
                    try:
                        _download(record['url'], archive, record['bytes'], record['sha256'], progress)
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise RuntimeError('Model download failed. Rerun to resume, or download '
                                           + record['filename'] + ' from the model folder and use --archives. '
                                           + str(exc)) from exc
                with tempfile.TemporaryDirectory(prefix='.model-', dir=root) as temp:
                    stage = Path(temp) / name
                    stage.mkdir()
                    _extract(archive, stage, record)
                    if target.exists() or target.is_symlink():
                        raise FileExistsError('Model destination appeared during installation: ' + str(target))
                    stage.rename(target)
                progress('Installed model: ' + str(target))
            result[name] = {'path': str(target), 'archive_sha256': record['sha256']}
        return result
