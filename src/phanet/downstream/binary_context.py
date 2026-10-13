"""Fixed-center native contexts for binary protein classification."""
import numpy as np
import torch

from phanet.datasets.splits import MASKED, PADDING


def context_batch(data, targets, embeddings, *, context_index=None):
    """Keep the target at index 14 and pad boundaries without shifting neighbors."""
    if context_index is None:
        groups = data.by_genome()
        lookup = {p.protein_id: (gid, i) for gid, ps in groups.items() for i, p in enumerate(ps)}
    else:
        groups, lookup = context_index
    size = len(targets)
    x = np.zeros((size, 30, embeddings.dimension), np.float32)
    annotation = np.zeros((size, 30, 9), np.float32)
    state = np.full((size, 30), PADDING, np.int64)
    strand = np.zeros((size, 30), np.int64)
    valid = np.zeros((size, 30), bool)
    selected, placements = [], []
    for b, pid in enumerate(targets):
        gid, center = lookup[pid]
        proteins = groups[gid]
        left, right = max(0, center - 14), min(len(proteins), center + 16)
        for native_index in range(left, right):
            position = native_index - center + 14
            selected.append(proteins[native_index])
            placements.append((b, position))
            valid[b, position] = True
            state[b, position] = MASKED
            strand[b, position] = proteins[native_index].strand
    if selected:
        vectors = embeddings.get([p.protein_id for p in selected], sequences=[p.sequence for p in selected])
        rows, columns = zip(*placements)
        x[rows, columns] = vectors
    return [torch.from_numpy(v) for v in (x, annotation, state, strand, valid)], [14] * size


def probability_batches(data, embeddings, model, *, batch_size=8, device="cpu"):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data.validate()
    groups = data.by_genome()
    index = (groups, {p.protein_id: (gid, i) for gid, ps in groups.items() for i, p in enumerate(ps)})
    ids = [p.protein_id for p in data.proteins]
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(ids), batch_size):
            selected = ids[first:first + batch_size]
            batch, offsets = context_batch(data, selected, embeddings, context_index=index)
            result = model([x.to(device) for x in batch], offsets).sigmoid()
            if result.shape != (len(selected), 1) or not torch.isfinite(result).all():
                raise ValueError("Invalid binary probabilities")
            yield selected, result[:, 0].cpu().numpy()
