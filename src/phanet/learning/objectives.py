"""Training-only family prototypes and native reciprocal-exchange samples."""
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from phanet.learning.context import ContextDataset
from phanet.datasets.splits import MaskPolicy


def family_bank(data, store, splits):
    sums, counts = {}, {}
    for lo in range(0, len(data.proteins), 1024):
        proteins = [p for p in data.proteins[lo:lo+1024]
                    if splits.assignments[p.genome_id] == 'train' and p.family_ids]
        vectors = store.get([p.protein_id for p in proteins], sequences=[p.sequence for p in proteins])
        for p, v in zip(proteins, vectors):
            for f in set(p.family_ids):
                if f not in sums:
                    sums[f] = np.zeros(store.dimension, np.float64); counts[f] = 0
                sums[f] += v; counts[f] += 1
    names = sorted(sums)
    if len(names) < 2:
        raise ValueError('Family training requires at least two training-supported families')
    vectors = np.stack([sums[f]/counts[f] for f in names])
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise ValueError('Invalid family prototype')
    index = {f: i for i, f in enumerate(names)}
    targets = np.array([min((index[f] for f in p.family_ids if f in index), default=-1)
                        for p in data.proteins], np.int64)
    return names, torch.from_numpy((vectors/norms).astype(np.float32)), targets


class ObjectiveDataset(Dataset):
    def __init__(self, contexts, *, families=None, composition=False, sample_rate=.25):
        if not 0 < sample_rate <= 1:
            raise ValueError('Composition sample rate must lie in (0,1]')
        self.contexts, self.families = contexts, families
        self.composition, self.sample_rate = composition, sample_rate
        self.epoch = 0
        self.by_genome = contexts.dataset.by_genome()
        self.partition = contexts.splits.assignments

    def __len__(self):
        return len(self.contexts)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('Epoch must be a nonnegative integer')
        self.epoch = epoch
        self.contexts.set_epoch(epoch)

    def __getitem__(self, index):
        c = self.contexts
        w = c.window(index)
        epoch = self.epoch if self.partition[w.genome_id] == 'train' else 0
        if c.epoch != epoch:
            c.set_epoch(epoch); w = c.window(index)
        batch = c[index]
        if self.families is not None:
            indices = batch['protein_indices'].numpy()
            values = np.full(indices.shape, -1, np.int64)
            valid = indices >= 0; values[valid] = self.families[indices[valid]]
            batch['family_index'] = torch.from_numpy(values)
        if not self.composition:
            return batch
        arrays = dict(composition_eligible=np.array(False),
            composition_embeddings=np.zeros((2, 8, c.embeddings.dimension), np.float32),
            composition_valid=np.zeros((2, 8), bool), composition_strand=np.ones((2, 8), np.int64),
            composition_swap=np.zeros(8, bool), composition_indices=np.full((2, 8), -1, np.int64),
            composition_bounds=np.array([0, len(self.by_genome[w.genome_id])], np.int64))
        proteins = self.by_genome[w.genome_id]
        focal = [c.dataset.proteins[w.indices[j]].order for j in w.members]
        n = len(focal)
        key = f'same_genome_internal_exchange_v1:{c.sampling_seed}:{epoch}:{w.genome_id}:{c.mode}:{index}'
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:16], 'big'))
        if 4 <= n <= 8 and rng.random() < self.sample_rate:
            left = max(0, focal[0]-n)
            right = focal[-1]+2
            right_count = max(0, len(proteins)-n-right+1)
            if left+right_count:
                a = [proteins[j] for j in focal]
                va = c.embeddings.get([p.protein_id for p in a], sequences=[p.sequence for p in a])
                for _ in range(32):
                    offset = int(rng.integers(left+right_count))
                    start = offset if offset < left else right+offset-left
                    b = proteins[start:start+n]
                    if {p.sha256 for p in a} & {p.sha256 for p in b}:
                        continue
                    width = int(rng.integers(1, min(2, n-2)+1))
                    slot = int(rng.integers(1, n-width))
                    if [p.strand for p in a[slot:slot+width]] != [p.strand for p in b[slot:slot+width]]:
                        continue
                    vb = c.embeddings.get([p.protein_id for p in b], sequences=[p.sequence for p in b])
                    if (va[:, None] == vb[None]).all(-1).any():
                        continue
                    arrays['composition_eligible'][...] = True
                    arrays['composition_embeddings'][:, :n] = np.stack([va, vb])
                    arrays['composition_valid'][:, :n] = True
                    arrays['composition_strand'][:, :n] = [[p.strand for p in a], [p.strand for p in b]]
                    arrays['composition_indices'][:, :n] = [focal, list(range(start, start+n))]
                    arrays['composition_swap'][slot:slot+width] = True
                    break
        batch.update({k: torch.from_numpy(v) for k, v in arrays.items()})
        return batch


def training_views(base, partition, *, families=None, composition=False):
    children = []
    for mode in ('segments', 'sampled_modules'):
        contexts = ContextDataset(base.dataset, base.embeddings, base.splits, MaskPolicy(),
            partition=partition, mode=mode, max_context=base.max_context, sampling_seed=base.sampling_seed)
        children.append(ObjectiveDataset(contexts, families=families, composition=composition))
    return ConcatDataset(children)
