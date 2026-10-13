"""Mass-conserving PPR and synchronous, provenance-aware label expansion."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from phanet.common.io import finite_array


@dataclass(frozen=True)
class PropagationConfig:
    alpha: float = .15
    tolerance: float = 1e-6
    max_iterations: int = 100
    relevance_threshold: float = .34
    admission_threshold: float = .52
    category_support_threshold: float = .5
    effective_support_threshold: float = 0.
    inherited_reliability_threshold: float = .52
    min_independent_sources: int = 2
    max_depth: int = 5
    expansion_rounds: int = 5
    batch_size: int = 16
    # Use node IDs as source units when no genome/cluster metadata is supplied.
    independence_unit: str = "genome"

    def validate(self):
        try:
            finite_scalars = np.isfinite(self.alpha) and np.isfinite(self.tolerance)
            valid_convergence = finite_scalars and 0 < self.alpha <= 1 and self.tolerance > 0
        except (TypeError, ValueError):
            valid_convergence = False
        if not valid_convergence:
            raise ValueError("Invalid PPR convergence parameters")
        for name in ("max_iterations", "batch_size", "min_independent_sources",
                     "max_depth", "expansion_rounds"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
        if min(self.max_iterations, self.batch_size, self.min_independent_sources) < 1 or min(self.max_depth, self.expansion_rounds) < 0:
            raise ValueError("Invalid iteration/batch/depth setting")
        for key in ("relevance_threshold", "admission_threshold", "category_support_threshold",
                    "effective_support_threshold", "inherited_reliability_threshold"):
            try:
                valid_threshold = np.isfinite(getattr(self, key)) and 0 <= getattr(self, key) <= 1
            except (TypeError, ValueError):
                valid_threshold = False
            if not valid_threshold:
                raise ValueError(f"Invalid threshold: {key}")
        if self.independence_unit not in ("node", "protein", "genome", "cluster"):
            raise ValueError("independence_unit must be node, protein, genome or cluster")


def _prepare_ppr_operator(adjacency: sparse.csr_matrix,
                          transit_mask: np.ndarray | None = None):
    """Prepare transition matrices for a graph component and active-reference mask."""
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    if transit_mask is None:
        degree = np.asarray(a.sum(axis=1)).ravel()
        transition = sparse.diags(np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)) @ a
        return transition.tocsr(), degree, None, None
    mask = np.asarray(transit_mask, dtype=bool)
    permitted = sparse.diags(mask.astype(np.float64))
    a_transit = (permitted @ a @ permitted).tocsr()
    degree = np.asarray(a_transit.sum(axis=1)).ravel()
    transition = sparse.diags(np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)) @ a_transit
    # A forbidden query may inspect one-hop evidence but cannot relay through
    # another forbidden node.  This row-normalized operator supplies exactly
    # that one-hop transition and is also used to preserve probability mass.
    source_adjacency = (a @ permitted).tocsr()
    source_degree = np.asarray(source_adjacency.sum(axis=1)).ravel()
    source_transition = sparse.diags(
        np.divide(1., source_degree, out=np.zeros_like(source_degree), where=source_degree > 0)
    ) @ source_adjacency
    return transition.tocsr(), degree, source_transition.tocsr(), source_degree


def _forbidden_query_ppr(prepared, queries: np.ndarray, transit_mask: np.ndarray,
                         config: PropagationConfig, *, compressed=None, source_rows=None):
    """Solve held-out-query PPR over the active reference subgraph."""
    transition, degree, source_transition, source_degree = prepared
    if compressed is None:
        active_positions = np.flatnonzero(transit_mask)
        active_transition = transition[active_positions][:, active_positions].tocsr()
        active_degree = degree[active_positions]
    else:
        active_transition, active_degree, source_transition, source_degree, active_positions = compressed
    if source_rows is None:
        source_rows = source_transition[queries][:, active_positions].tocsr()
    ranks = np.zeros((len(active_positions), len(queries)), dtype=np.float64)
    query_mass = np.ones(len(queries), dtype=np.float64)
    query_dangling = source_degree[queries] == 0
    if np.all(query_dangling):
        # This is the exact first iterate: no reference receives any mass.
        return ranks, query_mass, active_positions, {
            "iterations": 1, "converged": True, "max_residual": 0.,
            "state_nodes": int(len(active_positions)),
        }
    residual = np.zeros(len(queries), dtype=np.float64)
    iterations = 0
    beta = 1. - config.alpha
    for iterations in range(1, config.max_iterations + 1):
        propagated = active_transition.T @ ranks
        # Each sparse source row belongs to exactly one PPR column.  Updating
        # those entries directly avoids materializing an active_nodes x batch
        # dense one-hop correction at every iteration.
        for column in range(len(queries)):
            start, end = source_rows.indptr[column:column + 2]
            if start != end:
                rows = source_rows.indices[start:end]
                propagated[rows, column] += source_rows.data[start:end] * query_mass[column]
        active_dangling = ranks[active_degree == 0].sum(axis=0)
        updated_query_mass = config.alpha + beta * (
            active_dangling + query_mass * query_dangling)
        updated = beta * propagated
        residual = (np.abs(updated - ranks).sum(axis=0)
                    + np.abs(updated_query_mass - query_mass))
        ranks, query_mass = updated, updated_query_mass
        if np.all(residual < config.tolerance):
            break
    diagnostics = {
        "iterations": iterations,
        "converged": bool(np.all(residual < config.tolerance)),
        "max_residual": float(residual.max()) if len(residual) else 0.,
        "state_nodes": int(len(active_positions)),
    }
    return ranks, query_mass, active_positions, diagnostics


def _iter_reference_batches(prepared, queries, transit_mask, config, compressed, workers):
    """Process read-only query batches in parallel and yield results in query order."""
    sources = prepared[2][:, compressed[4]].tocsr()
    batches = (queries[start:start + config.batch_size]
               for start in range(0, len(queries), config.batch_size))

    def solve(batch):
        return batch, _forbidden_query_ppr(
            prepared, batch, transit_mask, config, compressed=compressed,
            source_rows=sources[batch])

    if workers == 1:
        for batch in batches:
            yield solve(batch)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="phanet-ppr") as pool:
        pending = deque()
        try:
            for _ in range(2 * workers):
                batch = next(batches, None)
                if batch is None:
                    break
                pending.append(pool.submit(solve, batch))
            while pending:
                yield pending.popleft().result()
                batch = next(batches, None)
                if batch is not None:
                    pending.append(pool.submit(solve, batch))
        finally:
            for future in pending:
                future.cancel()


def personalized_pagerank(adjacency: sparse.csr_matrix, queries: np.ndarray,
                         config: PropagationConfig,
                         transit_mask: np.ndarray | None = None,
                         _prepared_operator=None) -> tuple[np.ndarray, dict]:
    """Compute batched personalized PageRank.

    transit_mask selects nodes allowed to relay mass. Masked queries contribute
    their initial one-hop evidence. With no mask, all nodes can relay mass."""
    config.validate()
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError("PPR requires a finite nonnegative square adjacency")
    queries = np.asarray(queries, dtype=int)
    if ((queries < 0) | (queries >= a.shape[0])).any():
        raise ValueError("Query index out of bounds")
    if not len(queries):
        return np.zeros((a.shape[0], 0), dtype=np.float64), {
            "iterations": 0, "converged": True, "max_residual": 0.,
            "state_nodes": int(a.shape[0]),
        }
    if transit_mask is not None:
        transit_mask = np.asarray(transit_mask, dtype=bool)
        if transit_mask.shape != (a.shape[0],):
            raise ValueError("Transit mask shape mismatch")
        if _prepared_operator is None:
            prepared = _prepare_ppr_operator(a, transit_mask)
        else:
            prepared = _prepared_operator
        transition, degree, source_transition, source_degree = prepared
    else:
        if _prepared_operator is None:
            prepared = _prepare_ppr_operator(a)
        else:
            prepared = _prepared_operator
        transition, degree, source_transition, source_degree = prepared
    if transit_mask is not None and np.all(~transit_mask[queries]):
        active_ranks, query_mass, active_positions, diagnostics = _forbidden_query_ppr(
            prepared, queries, transit_mask, config)
        ranks = np.zeros((a.shape[0], len(queries)), dtype=np.float64)
        ranks[active_positions] = active_ranks
        ranks[queries, np.arange(len(queries))] = query_mass
        if not np.allclose(ranks.sum(axis=0), 1, atol=1e-10):
            raise ArithmeticError("PageRank probability mass was not conserved")
        return ranks, diagnostics
    restart = np.zeros((a.shape[0], len(queries)), dtype=np.float64)
    restart[queries, np.arange(len(queries))] = 1
    ranks = restart.copy()
    residual = np.zeros(len(queries))
    iterations = 0
    for iterations in range(1, config.max_iterations + 1):
        propagated = transition.T @ ranks
        if transit_mask is not None:
            forbidden = ~transit_mask[queries]
            correction_columns = np.flatnonzero(forbidden & (source_degree[queries] > 0))
            if len(correction_columns):
                # Each forbidden query may emit its one-hop row only into
                # its own PPR column.  A plain ``source_rows.T @ ranks``
                # would mix all query columns and silently create
                # cross-query probability flow when a batch contains more
                # than one forbidden query.
                source_rows = source_transition[queries[correction_columns]]
                columns = correction_columns
                diagonal_mass = ranks[queries[columns], columns]
                propagated[:, columns] += source_rows.multiply(diagonal_mass[:, None]).T.toarray()
            dangling_mask = degree == 0
            dangling = ranks[dangling_mask].sum(axis=0)
            if len(correction_columns):
                dangling[correction_columns] -= ranks[queries[correction_columns], correction_columns]
        else:
            dangling = ranks[degree == 0].sum(axis=0)
        updated = config.alpha * restart + (1 - config.alpha) * (propagated + restart * dangling)
        residual = np.abs(updated - ranks).sum(axis=0)
        ranks = updated
        if np.all(residual < config.tolerance):
            break
    if len(queries) and not np.allclose(ranks.sum(axis=0), 1, atol=1e-10):
        raise ArithmeticError("PageRank probability mass was not conserved")
    return ranks, {"iterations": iterations, "converged": bool(np.all(residual < config.tolerance)),
                   "max_residual": float(residual.max()) if len(residual) else 0.,
                   "state_nodes": int(a.shape[0])}


@dataclass
class PropagationResult:
    scores: np.ndarray
    reliability: np.ndarray
    support: np.ndarray
    references: np.ndarray
    depths: np.ndarray
    origins: list[set[int]]
    ancestry: list[set[int]]
    round_log: list[dict]
    contributor_log: list[dict]
    # Track genome/cluster provenance separately from node ancestry.
    origin_units: list[set] = field(default_factory=list)
    ancestry_units: list[set] = field(default_factory=list)
    independence_unit: str = "node"


def independent_contributors(indices, origins, ancestry, required: int,
                             source_units=None) -> list[int]:
    """Find disjoint reference witnesses, optionally projected to genome or cluster IDs."""
    indices = sorted(indices)
    if required < 1:
        return []
    if source_units is not None:
        units = list(source_units)
        def project(values):
            projected = set()
            for value in values:
                # Only genuine integer node indices are projected.  Numeric
                # strings are valid genome/cluster IDs and must not be
                # accidentally interpreted as positions in ``units``.
                if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(units):
                    projected.add(units[int(value)])
                else:
                    # Already-projected values are valid when callers pass
                    # unit-level provenance rather than node-level sets.
                    projected.add(value)
            return projected
        origins = [project(v) for v in origins]
        ancestry = [project(v) for v in ancestry]

    def search(start, chosen, used_origins, used_ancestry):
        if len(chosen) == required:
            return chosen
        for position in range(start, len(indices)):
            i = indices[position]
            if not origins[i] or origins[i] & used_origins or ancestry[i] & used_ancestry:
                continue
            found = search(position + 1, chosen + [i], used_origins | origins[i], used_ancestry | ancestry[i])
            if found:
                return found
        return []
    return search(0, [], set(), set())


def propagate(adjacency: sparse.csr_matrix, labels: np.ndarray, seed_mask: np.ndarray,
              expansion_mask: np.ndarray, config: PropagationConfig | None = None,
              source_units=None, *, workers: int = 1) -> PropagationResult:
    """Transfer seed labels with up to five synchronous reference-expansion rounds.

    seed_mask selects initial labels. Active references relay probability mass;
    held-out queries contribute one-hop evidence. Recompute predictions after
    the final admission round."""
    config = config or PropagationConfig()
    config.validate()
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers < 1:
        raise ValueError("workers must be a positive integer")
    a = sparse.csr_matrix(adjacency, dtype=np.float64)
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError("Propagation requires a finite nonnegative square adjacency")
    n = a.shape[0]
    labels = finite_array(labels, unit_interval=True)
    if labels.ndim != 2 or len(labels) != n:
        raise ValueError("Graph and category score shapes differ")
    seeds = np.asarray(seed_mask, dtype=bool)
    permitted = np.asarray(expansion_mask, dtype=bool)
    if seeds.shape != (n,) or permitted.shape != (n,):
        raise ValueError("Reference mask shape mismatch")
    if (seeds & ~permitted).any() or (seeds & (labels.sum(axis=1) == 0)).any():
        raise ValueError("Every initial reference needs labels and permission")
    if source_units is None:
        # Default source units to node IDs; callers may supply genome or cluster IDs.
        units = np.arange(n, dtype=object)
        effective_unit = "node" if config.independence_unit in ("genome", "cluster") else config.independence_unit
    else:
        units = np.asarray(list(source_units), dtype=object)
        if units.shape != (n,) or any(value is None or str(value) == "" for value in units):
            raise ValueError("source_units must contain one non-empty value per graph node")
        effective_unit = config.independence_unit
    active = seeds.copy()
    y = np.zeros_like(labels, dtype=np.float64)
    y[seeds] = labels[seeds]
    confidence = seeds.astype(np.float64)
    depths = np.zeros(n, dtype=int)
    origins = [{i} if seeds[i] else set() for i in range(n)]
    ancestry = [{i} if seeds[i] else set() for i in range(n)]
    origin_units = [{units[i]} if seeds[i] else set() for i in range(n)]
    ancestry_units = [{units[i]} if seeds[i] else set() for i in range(n)]
    scores = np.zeros_like(y)
    support = np.zeros_like(y)
    rho = np.zeros(n)
    # Initial references are already known labels and are not queries.  Keep
    # their exact vectors/reliability in the output; re-predicting them would
    # overwrite true seed labels with neighbor-induced scores.
    scores[seeds] = labels[seeds]
    support[seeds] = labels[seeds]
    rho[seeds] = 1.
    rounds, contributors = [], []
    _, component_ids = connected_components(a, directed=False)
    components = [np.flatnonzero(component_ids == c) for c in np.unique(component_ids)]
    for round_number in range(config.expansion_rounds + 1):
        pending = []
        round_contributors = []
        nonconverged = 0
        ppr_iterations_max = 0
        ppr_max_residual = 0.0
        ppr_state_nodes_max = 0
        ppr_converged = True
        for nodes in components:
            if not active[nodes].any():
                round_contributors.extend({"round": round_number, "query": int(i),
                                           "references": [], "weights": [], "reason": "no_reference",
                                           "reliability": 0.,
                                           "support": [0.] * labels.shape[1],
                                           "scores": [0.] * labels.shape[1]}
                                          for i in nodes)
                continue
            # Active references are never PPR queries.  They already have
            # labels and only non-active nodes can receive a new prediction or
            # be admitted in this synchronous round.
            query_nodes = nodes[~active[nodes]]
            round_contributors.extend({"round": round_number, "query": int(i),
                                       "references": [], "weights": [], "reason": "already_reference",
                                       "reliability": float(rho[i]),
                                       "support": support[i].tolist(),
                                       "scores": scores[i].tolist()}
                                      for i in nodes[active[nodes]])
            if not len(query_nodes):
                continue
            subgraph = a[nodes][:, nodes]
            prepared = _prepare_ppr_operator(subgraph, active[nodes])
            active_positions = np.flatnonzero(active[nodes])
            # Store P as CSC so the repeated P.T @ ranks uses CSR row gathers
            # instead of CSC scatter writes; conversion is once per round.
            compressed = (prepared[0][active_positions][:, active_positions].tocsc(),
                          prepared[1][active_positions], prepared[2], prepared[3], active_positions)
            local_query_positions = np.flatnonzero(~active[nodes])
            for local_queries, batch_result in _iter_reference_batches(
                    prepared, local_query_positions, active[nodes], config, compressed, workers):
                ranks, _, rank_positions, diagnostics = batch_result
                nonconverged += int(not diagnostics["converged"])
                ppr_iterations_max = max(ppr_iterations_max, int(diagnostics["iterations"]))
                ppr_max_residual = max(ppr_max_residual, float(diagnostics["max_residual"]))
                ppr_state_nodes_max = max(ppr_state_nodes_max,
                                          int(diagnostics.get("state_nodes", subgraph.shape[0])))
                ppr_converged = ppr_converged and bool(diagnostics["converged"])
                for column, local_query in enumerate(local_queries):
                    i = int(nodes[local_query])
                    accepted_rank_rows = np.flatnonzero(
                        ranks[:, column] > config.relevance_threshold)
                    accepted_positions = rank_positions[accepted_rank_rows]
                    accepted = nodes[accepted_positions]
                    # Avoid scanning every node in the component for every
                    # query; ancestry checks are performed only for accepted
                    # high-relevance references.
                    if len(accepted):
                        keep = np.asarray([i not in ancestry[int(reference)] for reference in accepted], dtype=bool)
                        accepted_rank_rows = accepted_rank_rows[keep]
                        accepted_positions = accepted_positions[keep]
                        accepted = accepted[keep]
                    weights = ranks[accepted_rank_rows, column] * confidence[accepted]
                    support[i] = weights @ y[accepted]
                    scores[i] = support[i] / weights.sum() if weights.sum() > 0 else 0
                    rho[i] = float(support[i].max())
                    if not len(accepted):
                        reason = "no_reference"
                    elif not permitted[i]:
                        reason = "reference_partition_forbidden"
                    elif active[i]:
                        reason = "already_reference"
                    elif rho[i] <= config.admission_threshold:
                        reason = "low_reliability"
                    else:
                        category = int(np.argmax(support[i]))
                        eligible = [int(j) for j, w in zip(accepted, weights)
                                    if y[j, category] >= config.category_support_threshold
                                    and w > config.effective_support_threshold
                                    and confidence[j] > config.inherited_reliability_threshold
                                    and depths[j] < config.max_depth
                                    and i not in ancestry[j]]
                        witness = independent_contributors(eligible, origin_units, ancestry_units,
                                                           config.min_independent_sources)
                        if witness and scores[i, category] >= config.category_support_threshold:
                            new_origins = set().union(*(origins[j] for j in accepted))
                            new_ancestry = {i} | set().union(*(ancestry[j] for j in accepted))
                            new_origin_units = set().union(*(origin_units[j] for j in accepted))
                            new_ancestry_units = {units[i]} | set().union(*(ancestry_units[j] for j in accepted))
                            new_depth = 1 + max(depths[j] for j in accepted)
                            if new_depth <= config.max_depth:
                                pending.append((i, scores[i].copy(), rho[i], new_depth,
                                                new_origins, new_ancestry,
                                                new_origin_units, new_ancestry_units))
                                reason = "eligible_for_next_round"
                            else:
                                reason = "depth_limit"
                        else:
                            reason = "insufficient_independent_support"
                    round_contributors.append({"round": round_number, "query": i,
                                               "references": accepted.tolist(),
                                               "weights": weights.tolist(), "reason": reason,
                                               "reliability": float(rho[i]),
                                               "support": support[i].tolist(),
                                               "scores": scores[i].tolist()})
        contributors.extend(round_contributors)
        active_before = int(active.sum())
        added = []
        if round_number == config.expansion_rounds or not pending:
            active_after = active_before
        else:
            # Deduplicate candidate admissions within each synchronous round.
            seen = set()
            for i, score, reliability, depth, roots, ancestors, roots_units, ancestors_units in pending:
                if i in seen or active[i]:
                    continue
                seen.add(i)
                added.append(int(i))
                active[i] = True
                y[i], confidence[i], depths[i] = score, reliability, depth
                origins[i], ancestry[i] = roots, ancestors
                origin_units[i], ancestry_units[i] = roots_units, ancestors_units
            active_after = int(active.sum())
        reason_counts = Counter(item["reason"] for item in round_contributors)
        rounds.append({"round": round_number,
                       "active_references_before": active_before,
                       "active_references_after": active_after,
                       # Alias for the active count before this round's admissions.
                       "reference_count": active_before,
                       "eligible_admissions": len(pending),
                       "new_references": sorted(added),
                       "nonconverged_batches": nonconverged,
                       "ppr_iterations_max": ppr_iterations_max,
                       "ppr_max_residual": ppr_max_residual,
                       "ppr_state_nodes_max": ppr_state_nodes_max,
                       "ppr_converged": ppr_converged,
                       "rejection_reasons": dict(sorted(reason_counts.items()))})
        if round_number == config.expansion_rounds or not pending:
            break
    finite_array(scores, unit_interval=True)
    finite_array(rho, unit_interval=True)
    return PropagationResult(scores, rho, support, active, depths, origins, ancestry,
                             rounds, contributors, origin_units, ancestry_units,
                             effective_unit)
