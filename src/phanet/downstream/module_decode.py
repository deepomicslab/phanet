"""Decode module candidates into member masks, scores and auxiliary types."""
from __future__ import annotations

import numpy as np
import torch


def compact_candidates(outputs, member_threshold=.5, object_floor=.25):
    """Return batch, anchor, member bits, score and types in float64 to preserve 30-bit masks."""
    if not 0 < member_threshold < 1 or not 0 < object_floor < 1:
        raise ValueError('Invalid threshold')
    obj = outputs['object_logits'].detach().float().sigmoid()
    valid = outputs['valid']
    indices = torch.nonzero((obj >= object_floor) & valid[:, :, None])
    if not len(indices):
        return np.empty((0, 5), np.float64)
    bi, anchor, slot = indices.unbind(1)
    length = valid.shape[1]
    pos = torch.arange(length, device=obj.device)[None]
    delta = pos - anchor[:, None]
    selected = (outputs['mask_logits'][bi, anchor, slot].detach().float().sigmoid()
                >= member_threshold) & valid[bi] & (delta > 0)
    bits = (selected.long() * (1 << delta.clamp_min(0))).sum(1) | 1
    type_bits = ((outputs['type_logits'][bi, anchor, slot].detach().float().sigmoid() >= .5).long()
                 * (1 << torch.arange(7, device=obj.device))).sum(1)
    keep = bits > 1
    return torch.stack((bi[keep].double(), anchor[keep].double(), bits[keep].double(),
                        obj[bi[keep], anchor[keep], slot[keep]].double(),
                        type_bits[keep].double()), 1).cpu().numpy()


def members_from_bits(anchor, bits):
    return tuple(int(anchor) + i for i in range(30) if int(bits) & (1 << i))
