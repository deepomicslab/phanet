"""Fixed-label sequence voting, query PPR and evidence-weighted nine-class fusion."""
from __future__ import annotations

import numpy as np
from scipy import sparse


def _labels(values):
    x = np.asarray(values)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isin(x, [0, 1]).all():
        raise ValueError("Expected a binary nine-category label matrix")
    return x.astype(np.float64)


def _scores(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 9 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("Expected finite nine-category scores in [0, 1]")
    return x


def _edges(values):
    x = sparse.csr_matrix(values, dtype=np.float64, copy=True)
    if not x.has_canonical_format or not np.isfinite(x.data).all() or np.any((x.data <= 0) | (x.data > 1)):
        raise ValueError("Expected canonical positive edge weights in (0, 1]")
    return x


def sequence_vote(hits, reference_labels):
    """Top five labeled, distinct references plus all bit-score ties; proportions in [0,1]."""
    chosen, seen = [], set()
    for hit in hits:
        rid = hit["reference_id"]
        if rid in seen:
            raise ValueError("Sequence voting requires one best alignment per distinct reference")
        seen.add(rid)
        bits, ev = float(hit["bits"]), float(hit["evalue"])
        quality = np.asarray([hit["identity"], hit["query_coverage"], hit["reference_coverage"]], dtype=float)
        if not np.isfinite([bits, ev]).all() or bits < 0 or ev < 0 or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
            raise ValueError("Invalid sequence alignment scores or proportions")
        if rid not in reference_labels:
            raise ValueError(f"Unknown reference identity: {rid}")
        labels = _labels([reference_labels[rid]])[0]
        if ev <= 1e-5 and bits >= 50 and labels.any():
            chosen.append((bits, rid, labels, quality[0] * min(quality[1:])))
    chosen.sort(key=lambda row: (-row[0], row[1]))
    if len(chosen) > 5:
        chosen = [r for r in chosen if r[0] >= chosen[4][0]]
    scores, rho = np.zeros(9), np.zeros(9)
    if chosen:
        bits = np.array([r[0] for r in chosen]); labels = np.array([r[2] for r in chosen])
        support = (labels * bits[:, None]).sum(0)
        scores = support / bits.sum()
        quality = np.array([r[3] for r in chosen])
        rho = np.divide((labels * (bits * quality)[:, None]).sum(0), support,
                        out=np.zeros(9), where=support > 0)
        rho[scores < .5] = 0
    return {"scores": scores, "rho": rho, "assigned": scores >= .5,
            "references": [r[1] for r in chosen]}


def reference_support(graph, labels, *, continuation=.85, tolerance=1e-7, max_iterations=200):
    """Ten numeric columns: nine category masses and one labeled-reference mass."""
    graph, labels = _edges(graph), _labels(labels)
    if graph.shape != (len(labels), len(labels)) or not len(labels) or graph.diagonal().any() or (graph != graph.T).nnz:
        raise ValueError("Expected a nonempty symmetric reference graph without self edges")
    if not np.isfinite([continuation, tolerance]).all() or not 0 < continuation < 1 or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Invalid PPR convergence settings")
    seed = np.column_stack([labels, labels.any(axis=1)]).astype(np.float64)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    transition = sparse.diags(inverse) @ graph + sparse.diags((degree == 0).astype(float))
    old = seed.copy()
    for iteration in range(1, max_iterations+1):
        new = (1-continuation)*seed + continuation*(transition @ old)
        residual = np.max(np.abs(new-old))
        old = new
        if residual < tolerance:
            return old, {"iterations":iteration, "residual":float(residual)}
    raise RuntimeError("PPR did not converge; incomplete support must not be used")


def ppr_queries(global_edges, support, *, local_edges=None, local_labels=None, allowed_local=None):
    """Attach each query to global references and its own local terminal sources."""
    edges = _edges(global_edges)
    support = np.asarray(support, dtype=np.float64)
    if (support.shape != (edges.shape[1], 10) or not np.isfinite(support).all() or
            np.any(support < 0) or np.any(support > 1+1e-12) or np.any(support[:,:9] > support[:,9,None]+1e-12)):
        raise ValueError("Invalid reference category/existence masses")
    edges.data[edges.data <= .70] = 0; edges.eliminate_zeros()
    mass = edges @ support
    maximum = edges.max(axis=1).toarray().ravel() if edges.shape[1] else np.zeros(edges.shape[0])
    if local_edges is not None:
        local = _edges(local_edges)
        labels = _labels(local_labels)
        if local.shape != (edges.shape[0], len(labels)) or allowed_local is None:
            raise ValueError("Local sources require explicit per-query permission")
        allowed = sparse.csr_matrix(allowed_local, dtype=bool)
        if allowed.shape != local.shape or local.multiply(allowed).nnz != local.nnz:
            raise ValueError("Query uses local sources outside its own contig")
        local.data[local.data <= .70] = 0; local.eliminate_zeros()
        mass += local @ np.column_stack([labels, labels.any(axis=1)])
        if local.shape[1]: maximum = np.maximum(maximum, local.max(axis=1).toarray().ravel())
    elif local_labels is not None or allowed_local is not None:
        raise ValueError("Local labels or permissions supplied without edges")
    scores = np.divide(mass[:,:9], mass[:,9,None], out=np.zeros_like(mass[:,:9]), where=mass[:,9,None]>0)
    scores = np.clip(scores, 0, 1)
    rho = maximum[:,None] * scores
    rho[scores < .5] = 0
    return {"scores":scores, "rho":rho, "assigned":scores>=.5, "max_direct_edge":maximum}


def choose_network(sequence_scores, sequence_rho, ppr_scores, ppr_rho):
    """Sequence output takes precedence; PPR is used only for unassigned queries."""
    arrays = [_scores(x) for x in (sequence_scores, sequence_rho, ppr_scores, ppr_rho)]
    if len({x.shape for x in arrays}) != 1:
        raise ValueError("Network route shapes differ")
    seq, srho, ppr, prho = arrays
    use_sequence = (seq >= .5).any(axis=1)
    scores = np.where(use_sequence[:,None], seq, ppr)
    rho = np.where(use_sequence[:,None], srho, prho)
    rho[scores < .5] = 0
    route = np.where(use_sequence, "sequence", np.where((ppr>=.5).any(axis=1), "ppr", "unassigned"))
    return scores, rho, route


def fuse(glm_scores, network_scores, network_rho):
    glm, network, rho = [_scores(x) for x in (glm_scores, network_scores, network_rho)]
    if glm.shape != network.shape or glm.shape != rho.shape:
        raise ValueError("Fusion inputs must have identical query and category order")
    rho = np.where(network >= .5, rho, 0.)
    return rho*network + (1-rho)*glm
