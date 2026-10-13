"""RBP and taxonomy input, output and algorithm tests."""
import json

import numpy as np
import pytest

from phanet.cli import main
from phanet.common.io import read_table, validate_manifest, write_table
from phanet.downstream.rbp import RBPScreenConfig, domain_architecture, screen_rbp, tmvec_retrieve
from phanet.downstream.taxonomy import (TaxonomyThresholds, aggregate_taxonomy_hits, classify_taxonomy,
                                        fit_taxonomy_classifiers, predict_taxonomy_classifiers)


def hit(pid, family='F', genus='G', weight=.9, order=''):
    return dict(genome_id='q', query_protein_id=pid, weight=weight, family=family, genus=genus, order=order)


def test_taxonomy_skips_missing_ranks_and_deduplicates_protein_votes():
    rows = [hit('p1'), hit('p2'), hit('p2', family='Other', weight=.8)]
    result = classify_taxonomy(['q', 'no-hit'], rows)
    assert result[0]['ranks']['family']['support_count'] == 2
    assert result[0]['ranks']['genus']['label'] == 'G'
    assert result[0]['ranks']['subfamily']['method'] == 'unresolved'
    assert result[1]['method'] == 'ambiguous'


def test_taxonomy_optional_subfamily_does_not_bypass_conflicting_family():
    rows = [hit('p1', 'F1', 'G1'), hit('p2', 'F1', 'G2'), hit('p3', 'F1', 'G3'),
            hit('p4', 'F2', 'Wrong'), hit('p5', 'F2', 'Wrong')]
    result = classify_taxonomy(['q'], rows, thresholds=TaxonomyThresholds(min_proportion=.3, min_margin=0.))
    assert result[0]['ranks']['family']['label'] == 'F1'
    assert result[0]['ranks']['genus']['label'] == ''


def test_taxonomy_ambiguous_parent_blocks_confident_child():
    rows = [hit('p1', 'F1', 'G'), hit('p2', 'F1', 'G'), hit('p3', 'F2', 'G'), hit('p4', 'F2', 'G')]
    assert classify_taxonomy(['q'], rows)[0]['ranks']['genus']['label'] == ''


def test_taxonomy_weak_unselected_hit_cannot_poison_ancestry():
    rows = [hit('p1'), hit('p2'), hit('p2', 'Other', 'G', weight=.1)]
    assert classify_taxonomy(['q'], rows)[0]['ranks']['genus']['label'] == 'G'


def test_taxonomy_equal_weight_hits_are_permutation_invariant():
    rows = [hit('p1', 'F1'), hit('p1', 'F2'), hit('p2', 'F1')]
    assert aggregate_taxonomy_hits(rows) == aggregate_taxonomy_hits(list(reversed(rows)))


@pytest.mark.parametrize('value', [None, '', -1, 1.1, float('inf'), float('nan'), True])
def test_taxonomy_rejects_invalid_weights(value):
    with pytest.raises(ValueError):
        classify_taxonomy(['q'], [hit('p1', weight=value)])


def test_zero_weight_hits_do_not_support_labels():
    assert classify_taxonomy(['q'], [hit('p1', weight=0), hit('p2', weight=0)])[0]['method'] == 'ambiguous'


def test_taxonomy_rejects_foreign_queries_ids_and_ranks():
    with pytest.raises(ValueError, match='unknown query'):
        classify_taxonomy(['other'], [hit('p')])
    with pytest.raises(ValueError):
        classify_taxonomy([None], [])
    with pytest.raises(ValueError):
        classify_taxonomy(['q'], [], ranks=('family', 'order'))
    with pytest.raises(ValueError, match='multiple genomes'):
        aggregate_taxonomy_hits([hit('p'), dict(hit('p'), genome_id='other')])


@pytest.mark.parametrize('kwargs', [dict(min_support=True), dict(min_confidence=float('nan')),
    dict(rank_overrides={'family': {'min_support': float('inf')}}),
    dict(rank_overrides={'family': {'min_margin': True}}), dict(rank_overrides={'bad': {}})])
def test_taxonomy_threshold_validation(kwargs):
    with pytest.raises(ValueError):
        TaxonomyThresholds(**kwargs)


def training_data():
    x = np.array([[4., 0.], [3.9, .1], [0., 4.], [.1, 3.9]])
    ids = ['t1', 't2', 't3', 't4']
    labels = [dict(genome_id=pid, partition='train', family='F1' if i < 2 else 'F2',
                   genus='G1' if i < 2 else 'G2') for i, pid in enumerate(ids)]
    return x, ids, labels


def test_taxonomy_contextual_fit_predict_and_centroid():
    x, ids, labels = training_data()
    models = fit_taxonomy_classifiers(x, ids, labels, set(ids))
    predictions = predict_taxonomy_classifiers(models, x[:1], ['q'])
    result = classify_taxonomy(['q'], [], classifier_predictions=predictions)[0]
    assert result['ranks']['family']['label'] == 'F1'
    assert result['ranks']['genus']['label'] == 'G1'
    assert result['method'] == 'context_glm'
    result = classify_taxonomy(['q'], [], features=np.vstack([x, x[:1]]), feature_ids=ids + ['q'], labels=labels, train_ids=set(ids))[0]
    assert result['ranks']['genus']['label'] == 'G1'
    assert result['method'] == 'context_centroid_fallback'


def test_ambiguous_glm_can_use_centroid_without_selecting_ambiguous_glm_again():
    x, ids, labels = training_data()
    pred = {'q': {'family': dict(label='F2', confidence=.5, margin=0., ambiguous=True, method='context_glm')}}
    result = classify_taxonomy(['q'], [], classifier_predictions=pred, features=np.vstack([x, x[:1]]),
                               feature_ids=ids + ['q'], labels=labels, train_ids=set(ids))[0]
    assert result['ranks']['family']['label'] == 'F1'
    assert result['ranks']['family']['method'] == 'context_centroid_fallback'


def test_classifier_unknown_parent_does_not_crash_or_invent_lineage():
    pred = {'q': {'family': dict(label='F', confidence=.99, margin=.98, ambiguous=False, method='context_glm')}}
    result = classify_taxonomy(['q'], [hit('p1', family='', genus='G'), hit('p2', family='', genus='G')], classifier_predictions=pred)[0]
    assert result['ranks']['family']['label'] == 'F'
    assert result['ranks']['genus']['label'] == ''


def test_tied_centroids_abstain():
    labels = [dict(genome_id='t1', family='F1'), dict(genome_id='t2', family='F2')]
    result = classify_taxonomy(['q'], [], labels=labels, train_ids={'t1', 't2'},
                               features=np.ones((3, 2)), feature_ids=['t1', 't2', 'q'])[0]
    assert result['ranks']['family']['label'] == ''


def test_training_leakage_missing_labels_nonfinite_and_conflicts_rejected():
    x, ids, labels = training_data()
    with pytest.raises(ValueError, match='Validation/test'):
        fit_taxonomy_classifiers(x, ids, [dict(r, partition='test') for r in labels], set(ids))
    with pytest.raises(ValueError, match='Conflicting'):
        fit_taxonomy_classifiers(x, ids, labels + [dict(labels[0], family='Other')], set(ids))
    with pytest.raises(ValueError, match='at least one'):
        fit_taxonomy_classifiers(x, ids, labels[:2], set(ids))
    with pytest.raises(ValueError, match='finite'):
        fit_taxonomy_classifiers(x * np.nan, ids, labels, set(ids))
    models = fit_taxonomy_classifiers(x, ids, labels, set(ids))
    with pytest.raises(ValueError, match='overlap'):
        predict_taxonomy_classifiers(models, x[:1], ids[:1])
    with pytest.raises(ValueError, match='dimension'):
        predict_taxonomy_classifiers(models, np.ones((1, 3)), ['q'])


def test_rbp_legacy_path_and_literature_separation():
    result = screen_rbp(['p1'], go_scores=[[.8]], go_terms=['GO:0019062'], tmvec_scores=[.7], classifier_scores=[.9],
        monomer_rows=[dict(protein_id='p1', mean_plddt=80)],
        complex_rows=[dict(protein_id='p1', iptm=.7, buried_surface_area=900, tm_score=.6)])
    assert result['final_ids'] == ['p1']
    assert result['stages']['final_status'] == 'complete'
    assert not result['candidates'][0]['literature_supported']


def test_rbp_missing_and_partial_evidence_stays_pending():
    result = screen_rbp(['p1', 'p2'], tmvec_scores=[.8, .8], classifier_scores=[.9, .9],
                        monomer_rows=[dict(protein_id='p1', mean_plddt=90)], complex_rows=[])
    assert result['stages']['monomer_status'] == 'partial'
    assert result['candidates'][1]['monomer_pass'] is None
    assert result['stages']['complex_status'] == 'partial'
    assert all(row['complex_pass'] is None for row in result['candidates'])
    result = screen_rbp(['p1'], tmvec_scores=[.8])
    assert result['stages']['classifier_status'] == 'pending'
    assert result['candidates'][0]['classifier_pass'] is None


def test_rbp_tm_only_negative_is_false_not_missing():
    result = screen_rbp(['p'], tmvec_scores=[.1], classifier_scores=[.9])
    assert result['candidates'][0]['preliminary'] is False
    assert result['candidates'][0]['classifier_pass'] is False


def test_rbp_infection_rule_is_explicit_and_does_not_or_in_other_scores():
    with pytest.raises(ValueError, match='together'):
        screen_rbp(['p'], infection_scores=[.8])
    result = screen_rbp(['p1', 'p2'], infection_scores=[.2, .8], tmvec_scores=[.9, .1], classifier_scores=[.9, .9],
                        config=RBPScreenConfig(infection_threshold=.5))
    assert result['classifier_ids'] == ['p2']
    assert result['preliminary_rule'] == 'infection'


@pytest.mark.parametrize('config', [dict(plddt_threshold=101), dict(plddt_threshold=float('nan')),
    dict(buried_surface_area=float('inf')), dict(classifier_threshold=True)])
def test_rbp_invalid_thresholds(config):
    with pytest.raises(ValueError):
        RBPScreenConfig(**config)


@pytest.mark.parametrize('rows', [[dict(protein_id='other', mean_plddt=80)],
    [dict(protein_id='p', mean_plddt=float('nan'))], [dict(protein_id='p', mean_plddt=110)],
    [dict(protein_id='p', mean_plddt=80)] * 2])
def test_rbp_validates_all_evidence_even_non_candidates(rows):
    with pytest.raises(ValueError):
        screen_rbp(['p'], tmvec_scores=[.1], monomer_rows=rows)


def test_rbp_rejects_misaligned_scores_and_ids():
    with pytest.raises(ValueError):
        screen_rbp(['p', 'p'], tmvec_scores=[.8, .8])
    with pytest.raises(ValueError):
        screen_rbp(['p'], classifier_scores=[[.8]])
    with pytest.raises(ValueError):
        screen_rbp(['p'], tmvec_scores=[2.])


def test_tmvec_cosine_ties_and_large_vectors():
    result = tmvec_retrieve(['q'], [[1e300, 0]], ['r1', 'r2', 'r3'], [[1e300, 0], [1., 0], [0, 1]], top_k=2)
    assert result['best_scores'].tolist() == [1.]
    assert [p['reference_id'] for p in result['pairs']] == ['r1', 'r2']
    assert 'cosine_score' in result['pairs'][0] and 'tm_score' not in result['pairs'][0]
    with pytest.raises(ValueError):
        tmvec_retrieve(['q'], [[1, 0]], [], np.empty((0, 2)))
    with pytest.raises(ValueError):
        tmvec_retrieve(['q'], [[0, 0]], ['r'], [[1, 0]])


def test_domain_intervals_counts_and_conflicting_lengths():
    row = dict(protein_id='p', domain='D', start=0, end=10, protein_length=20)
    result = domain_architecture([row])
    assert result['summary'][0]['median_normalized_center'] == .25
    for rows in ([row, row], [dict(row, start=.5)], [dict(row, end=21)], [dict(row, protein_id=None)],
                 [row, dict(row, domain='E', protein_length=30)]):
        with pytest.raises(ValueError):
            domain_architecture(rows)
    with pytest.raises(ValueError):
        domain_architecture([row], protein_lengths={'p': 30})


def test_taxonomy_cli_provenance_zero_hits_and_immutable_outputs(tmp_path):
    queries = tmp_path / 'ids.txt'; queries.write_text('q\nno-hit\n')
    hits = tmp_path / 'hits.tsv'
    write_table(hits, [hit('p1'), hit('p2')], list(hit('p1')))
    out = tmp_path / 'out'
    args = ['taxonomy', '--queries', str(queries), '--hits', str(hits), '--reference-version', 'synthetic-v1', '--out', str(out)]
    assert main(args) == 0
    validate_manifest(out)
    assert len(list(read_table(out / 'predictions.tsv'))) == 8
    assert json.loads((out / 'status.json').read_text())['state'] == 'complete'
    before = (out / 'manifest.json').read_bytes()
    assert main(args) == 1
    assert (out / 'manifest.json').read_bytes() == before


def test_taxonomy_cli_fits_explicit_training_partition(tmp_path):
    x, ids, labels = training_data()
    np.savez(tmp_path / 'train.npz', ids=ids, features=x)
    np.savez(tmp_path / 'query.npz', ids=['q'], features=x[:1])
    (tmp_path / 'ids.txt').write_text('q\n')
    write_table(tmp_path / 'labels.tsv', labels, list(labels[0]))
    out = tmp_path / 'out'
    args = ['taxonomy', '--queries', str(tmp_path / 'ids.txt'), '--features', str(tmp_path / 'query.npz'),
        '--train-features', str(tmp_path / 'train.npz'), '--labels', str(tmp_path / 'labels.tsv'),
        '--feature-version', 'synthetic-2d', '--reference-version', 'synthetic-v1', '--out', str(out)]
    assert main(args) == 0
    validate_manifest(out)
    assert json.loads((out / 'predictions.json').read_text())[0]['ranks']['family']['label'] == 'F1'
    heads = json.loads((out / 'fitted-heads.json').read_text())
    assert heads['family']['coefficients'] and heads['family']['train_ids'] == ids


def test_cli_rbp_reorders_npz_ids_and_records_partial_stages(tmp_path):
    scores = tmp_path / 'scores.tsv'
    write_table(scores, [dict(protein_id='p1', classifier_score=.9), dict(protein_id='p2', classifier_score=.9)], ['protein_id', 'classifier_score'])
    np.savez(tmp_path / 'q.npz', ids=np.array([b'p2', b'p1']), features=[[0, 1], [1, 0]])
    np.savez(tmp_path / 'r.npz', ids=['r'], features=[[1, 0]])
    out = tmp_path / 'out'
    assert main(['rbp', 'screen', '--scores', str(scores), '--evidence-version', 'synthetic-v1',
        '--tmvec-query-vectors', str(tmp_path / 'q.npz'), '--tmvec-reference-vectors', str(tmp_path / 'r.npz'),
        '--tmvec-reference-version', 'synthetic-vectors-v1', '--out', str(out)]) == 0
    validate_manifest(out)
    report = json.loads((out / 'report.json').read_text())
    assert report['classifier_ids'] == ['p1']
    assert report['stages']['monomer_status'] == 'pending'
    assert list(read_table(out / 'tmvec_retrieval.tsv'))[0]['protein_id'] == 'p1'


def test_cli_invalid_input_retains_failure_without_complete_manifest(tmp_path):
    scores = tmp_path / 'scores.tsv'; scores.write_text('protein_id\ttmvec_score\np\t2\n')
    out = tmp_path / 'bad'
    assert main(['rbp', 'screen', '--scores', str(scores), '--evidence-version', 'synthetic-v1', '--out', str(out)]) == 1
    assert json.loads((out / 'status.json').read_text())['state'] == 'failed'
    assert not (out / 'manifest.json').exists()
    assert (out / 'provenance.json').exists()


def test_cli_domains_and_infection_config(tmp_path):
    domains = tmp_path / 'domains.tsv'
    write_table(domains, [dict(protein_id='p', domain='D', start=0, end=10, protein_length=20)],
                ['protein_id', 'domain', 'start', 'end', 'protein_length'])
    out = tmp_path / 'domains'
    assert main(['rbp', 'domains', '--domains', str(domains), '--evidence-version', 'synthetic-v1', '--out', str(out)]) == 0
    validate_manifest(out)
    scores = tmp_path / 'scores.tsv'; scores.write_text('protein_id\tinfection_score\tclassifier_score\np\t0.8\t0.9\n')
    config = tmp_path / 'config.json'; config.write_text('{"infection_threshold": 0.7}')
    out = tmp_path / 'screen'
    assert main(['rbp', 'screen', '--scores', str(scores), '--evidence-version', 'synthetic-v1', '--config', str(config), '--out', str(out)]) == 0
    assert json.loads((out / 'report.json').read_text())['classifier_ids'] == ['p']


def test_feature_archive_schema_and_duplicate_headers_fail_cleanly(tmp_path):
    from phanet.downstream.evidence_cli import _features, _table
    np.savez(tmp_path / 'missing.npz', features=np.ones((1, 2)))
    with pytest.raises(ValueError, match='requires ids'):
        _features(tmp_path / 'missing.npz')
    np.save(tmp_path / 'plain.npy', np.ones((1, 2)))
    with pytest.raises(ValueError, match='NPZ'):
        _features(tmp_path / 'plain.npy')
    path = tmp_path / 'bad.tsv'; path.write_text('protein_id\ttmvec_score\ttmvec_score\np\t.2\t.8\n')
    with pytest.raises(ValueError, match='unique'):
        _table(path, ('protein_id',))


def test_rbp_accepts_existing_go_output_table(tmp_path):
    path = tmp_path / 'go_BP.tsv'
    path.write_text('protein_id\tpredicted_terms\tGO:0019062\np\tGO:0019062\t0.8\n')
    out = tmp_path / 'rbp'
    assert main(['rbp', 'screen', '--scores', str(path), '--evidence-version', 'synthetic-go-v1', '--out', str(out)]) == 0
    report = json.loads((out / 'report.json').read_text())
    assert report['stages']['preliminary'] == 1
    assert report['stages']['classifier_status'] == 'pending'


def test_constant_taxonomy_head_and_empty_queries():
    labels = [dict(genome_id='t', family='F')]
    models = fit_taxonomy_classifiers([[1, 0]], ['t'], labels, {'t'})
    result = predict_taxonomy_classifiers(models, [[1, 0]], ['q'])
    assert result['q']['family']['constant'] is True
    assert predict_taxonomy_classifiers(models, np.empty((0, 2)), []) == {}
    assert tmvec_retrieve([], np.empty((0, 2)), ['r'], [[1, 0]])['pairs'] == []


def test_output_symlink_is_not_overwritten(tmp_path):
    scores = tmp_path / 'scores.tsv'; scores.write_text('protein_id\ttmvec_score\np\t0.8\n')
    target = tmp_path / 'missing'
    out = tmp_path / 'link'; out.symlink_to(target, target_is_directory=True)
    assert main(['rbp', 'screen', '--scores', str(scores), '--evidence-version', 'synthetic-v1', '--out', str(out)]) == 1
    assert out.is_symlink() and not target.exists()


@pytest.mark.parametrize('ranks', [('family',), ('genus',), ('family', 'genus'), ('species',)])
def test_taxonomy_requested_ranks_and_species(ranks):
    rows = [dict(hit(pid), species='Species') for pid in ('p1', 'p2')]
    result = classify_taxonomy(['q'], rows, ranks=ranks)[0]
    assert tuple(result['ranks']) == ranks
    assert all(row['label'] for row in result['ranks'].values())
    if 'species' in ranks:
        assert result['ranks']['species']['label'] == 'Species'
    empty = classify_taxonomy(['empty'], [], ranks=ranks)[0]
    assert tuple(empty['ranks']) == ranks
    assert empty['method'] == 'ambiguous'


def test_taxonomy_species_rejects_conflicting_parent():
    rows = [dict(hit(pid), species='Good') for pid in ('p1', 'p2')]
    rows.append(dict(hit('p3', family='Other', genus='Other'), species='Wrong', weight=.9))
    result = classify_taxonomy(['q'], rows)[0]['ranks']
    assert result['species']['label'] == 'Good'
