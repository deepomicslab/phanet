"""Genome taxonomy transfer and explicit, training-only contextual fallbacks."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from .evidence import feature_matrix, identifier, identifiers, number

RANKS = ('order', 'family', 'subfamily', 'genus', 'species')
PARENT_RANK = dict(zip(RANKS, (None, *RANKS[:-1])))


def _ranks(values):
    values = tuple(values)
    if not values or values != tuple(rank for rank in RANKS if rank in values):
        raise ValueError('Taxonomy ranks must be a unique ordered subset of supported ranks')
    return values


def _label(row, rank):
    value = row.get(rank)
    if value is None:
        return ''
    if not isinstance(value, str):
        raise ValueError('Taxonomy labels must be strings')
    return value.strip()


def _taxonomy_label_map(labels, *, ranks=RANKS):
    result = {}
    for row in labels if labels is not None else ():
        genome = identifier(row.get('genome_id'))
        if row.get('partition', '') not in ('', 'train', 'val', 'test'):
            raise ValueError('Unknown taxonomy partition')
        for rank in ranks:
            label = _label(row, rank)
            if label:
                key = (genome, rank)
                if key in result and result[key] != label:
                    raise ValueError(f'Conflicting taxonomy labels for {genome} at {rank}')
                result[key] = label
    return result


def _ancestry(labels, ranks):
    mapping = _taxonomy_label_map(labels, ranks=ranks)
    result = {}
    for (gid, rank), label in mapping.items():
        parents = result.setdefault((rank, label), defaultdict(set))
        for parent in ranks[:ranks.index(rank)]:
            if (gid, parent) in mapping:
                parents[parent].add(mapping[gid, parent])
    return {key: {rank: sorted(values) for rank, values in parents.items()} for key, parents in result.items()}


def aggregate_taxonomy_hits(hits, *, ranks=RANKS):
    """Retain one informative hit per protein and normalize by rank coverage."""
    from .taxonomy_native import best_hits
    ranks = _ranks(ranks)
    grouped = best_hits(hits, ranks=ranks)
    result = {}
    for gid, rows in grouped.items():
        result[gid] = {}
        for rank in ranks:
            best = [r for r in rows if r.get(rank)]
            groups = defaultdict(list)
            for row in best: groups[row[rank]].append(row)
            summary = []
            for label, values in groups.items():
                ancestors = {parent: sorted({v.get(parent, '') for v in values if v.get(parent)})
                             for parent in ranks[:ranks.index(rank)]}
                summary.append(dict(label=label, support_count=len(values), support_proportion=len(values)/len(best),
                    confidence=sum(v['weight'] for v in values)/len(best),
                    query_proteins=sorted(v['query_protein_id'] for v in values),
                    reference_ids=sorted({v.get('reference_id','') for v in values if v.get('reference_id')}),
                    ancestors=ancestors,parent_labels=ancestors.get(PARENT_RANK[rank],[])))
            summary.sort(key=lambda r:(-r['confidence'],r['label']))
            item=dict(status='ambiguous' if summary else 'no_support',candidates=summary)
            if summary:
                item.update(top=summary[0],query_count=len(best),margin=summary[0]['confidence']-(summary[1]['confidence'] if len(summary)>1 else 0.))
            result[gid][rank]=item
    return result


@dataclass(frozen=True)
class TaxonomyThresholds:
    min_support: int = 2
    min_proportion: float = .5
    min_confidence: float = .5
    min_margin: float = .1
    rank_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, (int, np.integer)) or self.min_support < 1:
            raise ValueError('Taxonomy min_support must be a positive integer')
        object.__setattr__(self, 'min_support', int(self.min_support))
        for key in ('min_proportion', 'min_confidence', 'min_margin'):
            object.__setattr__(self, key, number(getattr(self, key), key, high=1.))
        if not isinstance(self.rank_overrides, dict) or set(self.rank_overrides) - set(RANKS):
            raise ValueError('Taxonomy rank overrides must map supported ranks')
        overrides = {}
        for rank, values in self.rank_overrides.items():
            if not isinstance(values, dict) or set(values) - {'min_support', 'min_proportion', 'min_confidence', 'min_margin'}:
                raise ValueError('Unknown taxonomy threshold override')
            clean = {}
            for key, value in values.items():
                if key == 'min_support':
                    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                        raise ValueError('Taxonomy min_support override must be a positive integer')
                    clean[key] = int(value)
                else:
                    clean[key] = number(value, key, high=1.)
            overrides[rank] = clean
        object.__setattr__(self, 'rank_overrides', overrides)

    def for_rank(self, rank):
        return dict(min_support=self.min_support, min_proportion=self.min_proportion,
                    min_confidence=self.min_confidence, min_margin=self.min_margin) | self.rank_overrides.get(rank, {})


def _training(features, feature_ids, labels, train_ids, ranks):
    features, feature_ids = feature_matrix(features, feature_ids)
    train_ids = set(identifiers(train_ids))
    mapping = _taxonomy_label_map(labels, ranks=ranks)
    if not train_ids or not train_ids <= set(feature_ids):
        raise ValueError('Taxonomy requires nonempty training IDs covered by features')
    for row in labels:
        if identifier(row.get('genome_id')) in train_ids and row.get('partition', '') not in ('', 'train'):
            raise ValueError('Validation/test taxonomy labels cannot be used for fitting')
    if not train_ids <= {gid for gid, _ in mapping}:
        raise ValueError('Every training genome requires at least one taxonomy label')
    train_labels = [row for row in labels if identifier(row.get('genome_id')) in train_ids]
    return features, feature_ids, train_ids, mapping, _ancestry(train_labels, ranks)


def fit_taxonomy_classifiers(features, feature_ids, labels, train_ids, *, ranks=RANKS):
    """Fit balanced logistic classifiers for each rank from training genome features."""
    ranks = _ranks(ranks)
    x, ids, train_ids, mapping, ancestry = _training(features, feature_ids, labels, train_ids, ranks)
    index = {pid: i for i, pid in enumerate(ids)}
    models = {}
    for rank in ranks:
        selected = sorted(pid for pid in train_ids if (pid, rank) in mapping)
        if not selected:
            continue
        y = np.asarray([mapping[pid, rank] for pid in selected])
        classes = np.unique(y)
        estimator = None
        if len(classes) > 1:
            estimator = LogisticRegression(C=1., solver='lbfgs', class_weight='balanced', max_iter=500, random_state=7)
            estimator.fit(x[[index[pid] for pid in selected]], y)
        models[rank] = dict(estimator=estimator, classes=classes.tolist(),
            constant=str(classes[0]) if estimator is None else None, feature_dimension=x.shape[1], train_ids=selected,
            ancestors={label: ancestry.get((rank, label), {}) for label in classes})
    return models


def predict_taxonomy_classifiers(models, features, feature_ids, *, ranks=RANKS):
    ranks = _ranks(ranks)
    x, ids = feature_matrix(features, feature_ids)
    result = {pid: {} for pid in ids}
    for rank in ranks:
        model = models.get(rank)
        if model is None:
            continue
        if x.shape[1] != model['feature_dimension']:
            raise ValueError('Taxonomy classifier feature dimension mismatch')
        if set(ids) & set(model['train_ids']):
            raise ValueError('Training and query taxonomy IDs overlap')
        if not ids:
            continue
        classes, estimator = model['classes'], model['estimator']
        scores = np.ones((len(x), 1)) if estimator is None else estimator.predict_proba(x)
        for i, pid in enumerate(ids):
            order = np.argsort(-scores[i], kind='stable')
            first = int(order[0]); confidence = float(scores[i, first])
            margin = confidence - (float(scores[i, order[1]]) if len(order) > 1 else 0.)
            label = classes[first]
            result[pid][rank] = dict(label=label, confidence=confidence, margin=margin, method='context_glm',
                ambiguous=confidence < .5 or margin < .1, ancestors=model['ancestors'].get(label, {}),
                constant=estimator is None)
    return result


def _centroid_fallback(features, query_ids, labels, train_ids, ranks, *, feature_ids=None):
    x, ids, train_ids, mapping, ancestry = _training(features, query_ids if feature_ids is None else feature_ids,
                                                   labels, train_ids, ranks)
    if set(query_ids) & train_ids or not set(query_ids) <= set(ids):
        raise ValueError('Query features must be complete and disjoint from taxonomy training IDs')
    index = {pid: i for i, pid in enumerate(ids)}
    result = {pid: {} for pid in query_ids}
    for rank in ranks:
        groups = defaultdict(list)
        for pid in sorted(train_ids):
            if (pid, rank) in mapping:
                groups[mapping[pid, rank]].append(index[pid])
        names = sorted(groups)
        if not names:
            continue
        centers = np.vstack([x[groups[name]].mean(axis=0) for name in names])
        for pid in query_ids:
            distance = np.sum((centers - x[index[pid]]) ** 2, axis=1)
            if not np.isfinite(distance).all():
                raise ValueError('Nonfinite taxonomy centroid distances')
            order = np.argsort(distance, kind='stable')
            scale = max(float(distance.max()), 1e-12)
            confidence = float(np.exp(-distance[order[0]] / scale))
            margin = float((distance[order[1]] - distance[order[0]]) / scale) if len(names) > 1 else 1.
            label = names[order[0]]
            result[pid][rank] = dict(label=label, confidence=confidence, margin=margin,
                ambiguous=confidence < .5 or margin < .1, method='context_centroid_fallback',
                ancestors=ancestry.get((rank, label), {}))
    return result


def classify_taxonomy(query_ids, hits, *, labels=None, features=None, train_ids=None, feature_ids=None,
                      classifier_predictions=None, thresholds=None, ranks=RANKS):
    """Transfer taxonomy with abstention and consistent available ancestry.

    Unavailable ranks may be skipped. Evidence at an unresolved ancestor blocks
    a child; accepted ancestors must agree with the child's retained evidence.
    """
    if features is None and not classifier_predictions:
        from .taxonomy_native import transfer
        return transfer(query_ids, hits, thresholds, ranks=ranks)
    ranks, query_ids = _ranks(ranks), identifiers(query_ids)
    if not query_ids:
        raise ValueError('Taxonomy requires at least one query genome')
    thresholds = thresholds or TaxonomyThresholds()
    support = aggregate_taxonomy_hits(hits, ranks=ranks)
    if set(support) - set(query_ids):
        raise ValueError('Taxonomy hits contain unknown query genomes')
    ancestry = _ancestry(labels, ranks) if labels is not None else {}
    fallback = {}
    if features is not None:
        if labels is None or train_ids is None:
            raise ValueError('Taxonomy fallback requires labeled training IDs')
        fallback = _centroid_fallback(features, query_ids, labels, train_ids, ranks, feature_ids=feature_ids)
    predictions = classifier_predictions or {}
    if set(predictions) - set(query_ids):
        raise ValueError('Classifier predictions contain unknown taxonomy query IDs')
    for pid, values in predictions.items():
        if set(values) - set(ranks):
            raise ValueError('Classifier predictions contain unknown ranks')
        for value in values.values():
            if not _label(value, 'label') or type(value.get('ambiguous')) is not bool:
                raise ValueError('Classifier predictions require labels and boolean ambiguity')
            number(value.get('confidence'), 'classifier confidence', high=1.)
            number(value.get('margin'), 'classifier margin', high=1.)
    output = []
    for gid in query_ids:
        row = dict(genome_id=gid, ranks={}, method='ambiguous')
        evidence_ranks = {rank for rank in ranks if support.get(gid, {}).get(rank, {}).get('top')
                          or rank in predictions.get(gid, {}) or rank in fallback.get(gid, {})}
        assigned = {}
        for rank in ranks:
            item = support.get(gid, {}).get(rank, {})
            top, limits = item.get('top'), thresholds.for_rank(rank)
            options = []
            if (top and top['support_count'] >= limits['min_support'] and
                    top['support_proportion'] >= limits['min_proportion'] and
                    top['confidence'] >= limits['min_confidence'] and item['margin'] >= limits['min_margin']):
                options.append(dict(top, margin=item['margin'], method='similarity_aggregate'))
            for table in (predictions, fallback):
                candidate = table.get(gid, {}).get(rank)
                if (candidate and not candidate['ambiguous'] and candidate['confidence'] >= limits['min_confidence']
                        and candidate['margin'] >= limits['min_margin']):
                    options.append(candidate.copy())
            chosen = None
            for candidate in options:
                parents = candidate.get('ancestors', ancestry.get((rank, candidate['label']), {}))
                if not isinstance(parents, dict) or any(not isinstance(v, list) for v in parents.values()):
                    raise ValueError('Taxonomy ancestry must map ranks to label lists')
                consistent = True
                for parent in ranks[:ranks.index(rank)]:
                    if parent in evidence_ranks and parent not in assigned:
                        consistent = False
                    if parent in assigned and set(parents.get(parent, [])) != {assigned[parent]['label']}:
                        consistent = False
                if consistent:
                    chosen = dict(candidate, ancestors=parents)
                    break
            if chosen is None:
                chosen = dict(label='', confidence=None, method='unresolved')
            else:
                assigned[rank] = chosen
            row['ranks'][rank] = chosen
        methods = {value['method'] for value in assigned.values()}
        row['method'] = next(iter(methods)) if len(methods) == 1 else 'mixed' if methods else 'ambiguous'
        output.append(row)
    return output
