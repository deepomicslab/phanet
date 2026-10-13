"""RBP score screening, similarity retrieval and domain summaries."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

from .evidence import feature_matrix, identifier, identifiers, number


@dataclass(frozen=True)
class RBPScreenConfig:
    go_threshold: float = .50
    tmvec_threshold: float = .50
    classifier_threshold: float = .51
    plddt_threshold: float = 70.
    iptm_threshold: float = .60
    buried_surface_area: float = 800.
    tm_score_threshold: float = .50
    infection_threshold: float | None = None

    def __post_init__(self):
        for name in ('go_threshold', 'tmvec_threshold', 'classifier_threshold',
                     'iptm_threshold', 'tm_score_threshold', 'infection_threshold'):
            value = getattr(self, name)
            if name == 'infection_threshold' and value is None:
                continue
            object.__setattr__(self, name, number(value, name, high=1.))
        object.__setattr__(self, 'plddt_threshold', number(self.plddt_threshold, 'plddt_threshold', high=100.))
        object.__setattr__(self, 'buried_surface_area', number(self.buried_surface_area, 'buried_surface_area'))


def tmvec_retrieve(query_ids, query_vectors, reference_ids, reference_vectors, *, top_k=50):
    """Retrieve TM-Vec neighbors by cosine similarity, preserving reference order for ties."""
    q, query_ids = feature_matrix(query_vectors, query_ids)
    r, reference_ids = feature_matrix(reference_vectors, reference_ids)
    if q.shape[1] != r.shape[1] or not len(r):
        raise ValueError('TM-Vec requires nonempty references with matching dimensions')
    if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError('TM-Vec top_k must be a positive integer')
    # Rescale before normalization to avoid overflow for finite, large vectors.
    for matrix in (q, r):
        scales = np.max(np.abs(matrix), axis=1)
        if (scales == 0).any():
            raise ValueError('TM-Vec vectors cannot have zero norm')
    q = q / np.max(np.abs(q), axis=1)[:, None]
    r = r / np.max(np.abs(r), axis=1)[:, None]
    q = q / np.linalg.norm(q, axis=1)[:, None]
    r = r / np.linalg.norm(r, axis=1)[:, None]
    k = min(int(top_k), len(r))
    pairs, best = [], np.empty(len(q), dtype=np.float64)
    for index, query in enumerate(query_ids):
        scores = np.clip(r @ q[index], -1., 1.)
        order = np.argsort(-scores, kind='stable')[:k]
        for rank, ref in enumerate(order, 1):
            pairs.append(dict(protein_id=query, reference_id=reference_ids[ref],
                              cosine_score=float(scores[ref]), rank=rank))
        best[index] = scores[order[0]]
    return dict(best_scores=best, pairs=pairs, top_k=k, metric='cosine_recomputed')


def _scores(values, n, name, low=0.):
    if values is None:
        return np.full(n, np.nan)
    scores = np.asarray(values, dtype=np.float64)
    if scores.shape != (n,) or np.isinf(scores).any() or ((scores < low) | (scores > 1)).any():
        raise ValueError(f'{name} requires one score per ID, in [{low}, 1], or NaN for missing')
    return scores


def _structural_rows(rows, ids, fields):
    result = {}
    for row in rows if rows is not None else ():
        key = identifier(row.get('protein_id'))
        if key not in ids or key in result:
            raise ValueError(f'Unknown or duplicate RBP evidence ID: {key}')
        result[key] = {name: number(row.get(name), name, high=high) for name, high in fields}
    return result


def _stage_status(values, supplied):
    if not supplied:
        return 'pending'
    return 'partial' if any(value is None for value in values) else 'complete'


def screen_rbp(ids, *, go_scores=None, go_terms=(), tmvec_scores=None, classifier_scores=None,
               monomer_rows=None, complex_rows=None, literature_ids=None, infection_scores=None,
               config=None):
    """Filter supplied RBP scores and structural evidence; retain missing values as pending."""
    config = config or RBPScreenConfig()
    ids, go_terms = identifiers(ids), identifiers(go_terms)
    n = len(ids)
    if (infection_scores is None) != (config.infection_threshold is None):
        raise ValueError('Infection scores and an explicit infection_threshold must be supplied together')
    if go_scores is None and go_terms:
        raise ValueError('GO terms require a GO score matrix')
    if go_scores is not None:
        go = np.asarray(go_scores, dtype=np.float64)
        if (go.shape != (n, len(go_terms)) or not go_terms or np.isinf(go).any()
                or ((go < 0) | (go > 1)).any()):
            raise ValueError('GO scores must align with IDs/terms and lie in [0, 1] or be NaN')
    else:
        go = np.empty((n, 0))
    columns = [i for i, term in enumerate(go_terms) if term in {'GO:0019062', 'GO:0046813', 'GO:0046718'}]
    tm = _scores(tmvec_scores, n, 'TM-Vec', -1.)
    clf = _scores(classifier_scores, n, 'Classifier')
    infection = _scores(infection_scores, n, 'Infection')
    monomer = _structural_rows(monomer_rows, set(ids), [('mean_plddt', 100.)])
    complexes = _structural_rows(complex_rows, set(ids), [('iptm', 1.), ('buried_surface_area', None), ('tm_score', 1.)])
    literature = set(identifiers(literature_ids if literature_ids is not None else []))

    def passed(score, threshold):
        return None if np.isnan(score) else bool(score >= threshold)

    def both(previous, current):
        return False if previous is False else None if previous is None else current

    def either(values):
        return True if True in values else None if None in values else False

    candidates = []
    for i, pid in enumerate(ids):
        go_hit = either([passed(go[i, j], config.go_threshold) for j in columns]) if columns else None
        tm_hit = passed(tm[i], config.tmvec_threshold)
        available = ([go_hit] if columns else []) + ([tm_hit] if tmvec_scores is not None else [])
        preliminary = (passed(infection[i], config.infection_threshold) if infection_scores is not None
                       else either(available) if available else None)
        classifier_pass = both(preliminary, passed(clf[i], config.classifier_threshold))
        mono = monomer.get(pid)
        mono_pass = both(classifier_pass, None if mono is None else mono['mean_plddt'] >= config.plddt_threshold)
        comp = complexes.get(pid)
        complex_pass = both(mono_pass, None if comp is None else
                            comp['iptm'] >= config.iptm_threshold and
                            comp['buried_surface_area'] >= config.buried_surface_area and
                            comp['tm_score'] >= config.tm_score_threshold)
        candidates.append(dict(protein_id=pid, go_host_recognition=go_hit, tmvec_supported=tm_hit,
            infection_score=None if np.isnan(infection[i]) else float(infection[i]), preliminary=preliminary,
            classifier_score=None if np.isnan(clf[i]) else float(clf[i]), classifier_pass=classifier_pass,
            monomer_pass=mono_pass, complex_pass=complex_pass, literature_supported=pid in literature))
    fields = dict(go='go_host_recognition', tmvec='tmvec_supported', preliminary='preliminary',
                  classifier='classifier_pass', monomer='monomer_pass', final='complex_pass')
    supplied = dict(go=bool(columns), tmvec=tmvec_scores is not None,
                    preliminary=infection_scores is not None or bool(columns) or tmvec_scores is not None,
                    classifier=classifier_scores is not None, monomer=monomer_rows is not None,
                    final=complex_rows is not None)
    stages = {}
    for stage, field in fields.items():
        values = [row[field] for row in candidates]
        status = _stage_status(values, supplied[stage])
        stages[stage] = None if status == 'pending' else sum(value is True for value in values)
        stages[stage + '_status'] = status
        stages[stage + '_pending'] = sum(value is None for value in values)
    stages['complex_status'] = stages['final_status']
    selected = {stage: [row['protein_id'] for row in candidates if row[field] is True]
                for stage, field in fields.items()}
    return dict(config=asdict(config), library_size=n, stages=stages, candidates=candidates,
        preliminary_rule='infection' if infection_scores is not None else 'legacy_go_or_tmvec',
        classifier_ids=selected['classifier'],
        monomer_ids=None if stages['monomer_status'] == 'pending' else selected['monomer'],
        final_ids=None if stages['final_status'] == 'pending' else selected['final'],
        literature_matches=sorted(literature & set(selected['classifier'])),
        evidence_separation='Score-based screening with separate literature matches.')


def domain_architecture(rows, *, protein_lengths=None):
    """Summarize 0-based, half-open integer domain intervals; overlaps are allowed."""
    lengths = {}
    for key, value in (protein_lengths or {}).items():
        key = identifier(key)
        if key in lengths:
            raise ValueError(f'Duplicate protein length: {key}')
        length = number(value, 'protein_length', low=1.)
        if not length.is_integer():
            raise ValueError('Protein length must be an integer')
        lengths[key] = length
    groups, observations, seen = defaultdict(list), [], set()
    for row in rows:
        pid = identifier(row.get('protein_id'))
        name = identifier(row.get('domain') or row.get('domain_id'))
        start, end = number(row.get('start'), 'start'), number(row.get('end'), 'end')
        if not start.is_integer() or not end.is_integer() or end <= start:
            raise ValueError(f'Invalid domain interval for {pid}')
        supplied = row.get('protein_length')
        length = lengths.get(pid)
        if supplied not in (None, ''):
            supplied = number(supplied, 'protein_length', low=1.)
            if not supplied.is_integer() or (length is not None and length != supplied):
                raise ValueError(f'Conflicting or invalid protein length for {pid}')
            length = supplied
        if length is None or end > length:
            raise ValueError(f'Missing protein length or domain exceeds protein length for {pid}')
        lengths[pid] = length
        key = (pid, name, start, end)
        if key in seen:
            raise ValueError(f'Duplicate domain interval for {pid}')
        seen.add(key)
        item = dict(protein_id=pid, domain=name, start=int(start), end=int(end), normalized_center=(start + end) / 2 / length)
        groups[name].append(item)
        observations.append(item)
    summary = [dict(domain=name, occurrences=len(values),
                    median_normalized_center=float(np.median([v['normalized_center'] for v in values])),
                    proteins=sorted({v['protein_id'] for v in values})) for name, values in sorted(groups.items())]
    return dict(domains=len(summary), instances=len(observations), summary=summary,
                observations=observations, coordinate_system='0-based-half-open',
                unit='protein/domain-instance (not genome or module counts)')
