"""Native RBP inputs with protein-level supervision partitions."""
from pathlib import Path
from types import SimpleNamespace
import os
import shutil

from phanet.common.io import digest_file, read_json, read_table, write_json
from phanet.datasets.schema import Dataset
from phanet.representations.embed import EmbeddingStore, validate_full_length


def validate_labels(data, path):
    rows = list(read_table(path, ('protein_id', 'label', 'partition', 'sequence_sha256')))
    proteins = {p.protein_id: p for p in data.proteins}
    seen, hashes, classes = set(), {}, {part: set() for part in ('train', 'val', 'test')}
    for row in rows:
        pid, part = row['protein_id'], row['partition']
        if (pid in seen or pid not in proteins or part not in classes
                or row['label'] not in ('0', '1') or proteins[pid].sha256 != row['sequence_sha256']):
            raise ValueError('Invalid RBP supervision identity, label or partition')
        seen.add(pid)
        old = hashes.setdefault(row['sequence_sha256'], (part, row['label']))
        if old != (part, row['label']):
            raise ValueError('Identical RBP targets cross partitions or have conflicting labels')
        classes[part].add(row['label'])
    if any(not values for values in classes.values()) or any(classes[p] != {'0', '1'} for p in ('train', 'val')):
        raise ValueError('RBP requires all three partitions and both train/val classes')
    return rows


def prepare(output, *, dataset, embeddings, labels):
    root = Path(output).resolve()
    data_path, embedding_path = Path(dataset).resolve(), Path(embeddings).resolve()
    data, store = Dataset.load(data_path), EmbeddingStore(embedding_path)
    rows = validate_labels(data, labels)
    validate_full_length(data, store)
    for start in range(0, len(data.proteins), 1024):
        ps = data.proteins[start:start + 1024]
        store.get([p.protein_id for p in ps], sequences=[p.sequence for p in ps])
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(labels, root / 'labels.tsv')
    meta = dict(schema_version=1, task='rbp', split_unit='protein',
                dataset=os.path.relpath(data_path, root), embeddings=os.path.relpath(embedding_path, root),
                labels='labels.tsv', labels_sha256=digest_file(root / 'labels.tsv'),
                dataset_files={p.name: digest_file(p) for p in sorted(data_path.iterdir()) if p.is_file()},
                embedding_metadata_sha256=digest_file(embedding_path / 'embeddings.json'),
                partitions={p: sum(r['partition'] == p for r in rows) for p in ('train', 'val', 'test')})
    write_json(root / 'training.json', meta)
    return meta


def load(directory, labels_path):
    root = Path(directory).resolve()
    meta = read_json(root / 'training.json')
    if (meta.get('schema_version') != 1 or meta.get('task') != 'rbp'
            or meta.get('split_unit') != 'protein' or meta.get('labels') != 'labels.tsv'):
        raise ValueError('Invalid RBP training manifest')
    if (digest_file(root / 'labels.tsv') != meta['labels_sha256']
            or digest_file(labels_path) != meta['labels_sha256']):
        raise ValueError('RBP supervision changed')
    data_path, embedding_path = root / meta['dataset'], root / meta['embeddings']
    files = {p.name for p in data_path.iterdir() if p.is_file()}
    if files != set(meta['dataset_files']) or any(digest_file(data_path / p) != sha for p, sha in meta['dataset_files'].items()):
        raise ValueError('RBP native dataset changed')
    if digest_file(embedding_path / 'embeddings.json') != meta['embedding_metadata_sha256']:
        raise ValueError('RBP embedding metadata changed')
    data, store = Dataset.load(data_path), EmbeddingStore(embedding_path)
    rows = validate_labels(data, root / 'labels.tsv')
    counts = {p: sum(r['partition'] == p for r in rows) for p in ('train', 'val', 'test')}
    if counts != meta['partitions']:
        raise ValueError('RBP partition counts changed')
    validate_full_length(data, store)
    return SimpleNamespace(dataset=data, embeddings=store)
