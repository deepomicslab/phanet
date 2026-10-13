"""Rank native protein contexts against reciprocal span exchanges."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from phanet.datasets.splits import PADDING, UNRESOLVED
from phanet.learning.model import forward_batch


@contextmanager
def deterministic_dropout(model):
    """Retain autograd while removing independent-view dropout shortcuts/noise."""
    layers = [(m, m.training) for m in model.modules() if isinstance(m, nn.Dropout)]
    try:
        for layer, _ in layers:
            layer.train(False)
        yield
    finally:
        for layer, training in layers:
            layer.train(training)


def exchange_ranking_loss(scores):
    """scores[condition, pair, side]; condition 0=native, 1=exchanged."""
    if scores.ndim != 3 or scores.shape[0] != 2 or scores.shape[2] != 2 or not scores.shape[1]:
        raise ValueError("Ranking requires native/exchanged scores for both sides of each pair")
    scores = scores.float()
    if not torch.isfinite(scores).all():
        raise FloatingPointError("Nonfinite composition scores")
    difference = (scores[0] - scores[1]).sum(dim=-1)
    return F.softplus(-difference).mean(), difference


def composition_loss(model, batch, zero):
    required = ("composition_eligible", "composition_embeddings", "composition_valid",
                "composition_strand", "composition_swap", "composition_indices", "composition_bounds")
    if any(k not in batch for k in required):
        raise ValueError("Composition training requires explicitly sampled same-genome pairs")
    active = batch["composition_eligible"].bool()
    report = dict(composition_pairs=0, composition_margin=0., composition_accuracy=0.,
                  composition_score_rms=0., composition_module_variance=0.)
    if active.ndim != 1 or len(active) != len(batch["valid"]):
        raise ValueError("Invalid composition eligibility")
    if not active.any():
        return zero, report
    embeddings = batch["composition_embeddings"][active]
    valid = batch["composition_valid"][active].bool()
    strand = batch["composition_strand"][active]
    swap = batch["composition_swap"][active].bool()
    indices = batch["composition_indices"][active]
    bounds = batch["composition_bounds"][active]
    if embeddings.ndim != 4 or embeddings.shape[1] != 2 or valid.shape != embeddings.shape[:3]:
        raise ValueError("Composition pairs must have shape pairs x 2 x length x features")
    n, _, length, features = embeddings.shape
    if strand.shape != valid.shape or indices.shape != valid.shape or swap.shape != (n, length):
        raise ValueError("Composition metadata dimensions disagree")
    if bounds.shape != (n, 2) or indices.dtype != torch.int64 or bounds.dtype != torch.int64:
        raise ValueError("Composition needs integer source-genome bounds")
    if (((indices < bounds[:, 0, None, None]) | (indices >= bounds[:, 1, None, None])) & valid).any():
        raise ValueError("Composition indices fall outside their common source genome")
    if ((indices[:, :, 1:] - indices[:, :, :-1] != 1) & valid[:, :, 1:]).any():
        raise ValueError("Composition source windows must be consecutive")
    counts = valid[:, 0].sum(-1)
    prefix = torch.arange(length, device=valid.device)[None] < counts[:, None]
    if (not torch.equal(valid[:, 0], valid[:, 1]) or not torch.equal(valid[:, 0], prefix)
            or (counts < 4).any() or (swap & ~prefix).any() or (swap.sum(-1) < 1).any()
            or (swap.sum(-1) > 2).any() or swap[:, 0].any()
            or swap.gather(1, (counts - 1)[:, None]).any()):
        raise ValueError("Exchange needs equal-length windows and unchanged endpoint anchors")
    # Only contiguous internal spans may be exchanged; ranks/directions stay fixed.
    span_edges = swap[:, 1:].long() - swap[:, :-1].long()
    if ((span_edges == 1).sum(-1) != 1).any() or ((span_edges == -1).sum(-1) != 1).any():
        raise ValueError("Exchange span must be contiguous")
    if ((strand[:, 0] != strand[:, 1]) & swap).any():
        raise ValueError("Exchanged slots must have matching strand orientation")
    overlap = indices[:, 0, :, None] == indices[:, 1, None, :]
    if (overlap & valid[:, 0, :, None] & valid[:, 1, None, :]).any():
        raise ValueError("Composition source windows overlap")
    equal_tokens = (embeddings[:, 0] == embeddings[:, 1]).all(-1)
    if ((~equal_tokens & swap).sum(-1) == 0).any():
        raise ValueError("No-op exchange cannot supply a ranking target")
    exchanged = torch.where(swap[:, None, :, None], embeddings.flip(1), embeddings)
    # Exchanging indistinguishable remainders can merely swap the two whole inputs.
    cross_equal = ((exchanged == embeddings.flip(1)) | ~valid[:, :, :, None]).flatten(1).all(-1)
    if cross_equal.any():
        raise ValueError("Whole-pair permutation cannot supply a ranking target")
    inputs = torch.cat((embeddings, exchanged), dim=0).reshape(4 * n, length, features)
    validity = valid.repeat(2, 1, 1).reshape(4 * n, length)
    view = dict(embeddings=inputs, valid=validity, member_mask=validity,
                strand=strand.repeat(2, 1, 1).reshape(4 * n, length),
                annotation_input=inputs.new_zeros((4 * n, length, model.annotation_head.out_features)),
                state=torch.where(validity, UNRESOLVED, PADDING))
    with deterministic_dropout(model):
        output = forward_batch(model, view)
    loss, difference = exchange_ranking_loss(output["composition"].reshape(2, n, 2))
    with torch.no_grad():
        z = F.normalize(output["module"][:2 * n].float(), dim=-1)
        report.update(composition_pairs=n, composition_margin=float(difference.mean()),
                      composition_accuracy=float(((difference > 0).float() + .5 * (difference == 0)).mean()),
                      composition_score_rms=float(output["composition"].square().mean().sqrt()),
                      composition_module_variance=float(z.var(dim=0, unbiased=False).sum()))
    return loss, report
