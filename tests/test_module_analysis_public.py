"""Public postprocessing inputs, evidence, failure handling and sequential CLI."""
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from phanet.analysis.public import analyze, discover, init_config, load_config, load_relationships
from phanet.analysis.prediction_pipeline import CatalogueConfig, read_references
from phanet.analysis.template_matching import Relationships
from phanet.common.io import digest_file, digest_object, read_json, write_json, write_table
from phanet.datasets.schema import Dataset, Genome, Protein
from phanet.cli import main


def seal(predictions, dataset):
    write_json(predictions / 'manifest.json', dict(task='module', models={'module': {'network_mode': 'off'}},
        dataset={p.name: digest_file(p) for p in dataset.iterdir() if p.is_file()}))
    write_json(predictions / 'status.json', dict(state='complete', task='module',
        files={p.name: digest_file(p) for p in predictions.iterdir() if p.is_file() and p.name != 'status.json'}))


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def inputs(tmp_path):
    data = Dataset([Genome(f'g{i}', f'g{i}', 1200, 'synthetic-test', cluster_id=f'c{i}') for i in range(6)],
        [Protein(f'g{i}:p{j}', f'g{i}', 100*j, 100*j+60, 1, j, seq, product='hypothetical protein' if j > 1 else 'test annotation')
         for i in range(6) for j, seq in enumerate(('MAAAA', 'MCCCC', 'MDDDD', 'MEEEE'))])
    native = tmp_path / 'dataset'
    data.save(native)
    pred = tmp_path / 'prediction'
    pred.mkdir()
    rows, genomes = [], []
    for gid, proteins in data.by_genome().items():
        for ranks in ([0, 1], [2, 3]):
            members = [proteins[i].protein_id for i in ranks]
            rows.append(dict(module_id=digest_object([gid, members]), genome_id=gid, members=members,
                member_ranks=ranks, score=.9, supporting_windows=1, covering_windows=1, window_support=1.,
                auxiliary_type_bits=64, span_qc_passed=True, start=proteins[ranks[0]].start, end=proteins[ranks[-1]].end))
        genomes.append(dict(genome_id=gid, proteins=4, windows=1, retained=2, accepted=2, span_qc_rejected=0))
    write_rows(pred / 'modules.jsonl', rows)
    write_rows(pred / 'genomes.jsonl', genomes)
    write_json(pred / 'summary.json', dict(genomes=6, proteins=24, candidates=12, accepted=12, span_qc_rejected=0))
    write_table(pred / 'module_members.tsv', [], ['module_id', 'protein_id'])
    seal(pred, native)
    references = tmp_path / 'references.jsonl'
    write_rows(references, [dict(reference_id='test-known', complete=True, source='synthetic-test',
        members=[dict(protein_sha256=p.sha256, qualified_profiles=[]) for p in data.proteins[:2]])])
    baseline = tmp_path / 'baseline.json'
    write_json(baseline, dict(schema=1, rule_id='test-baseline-v1', scope='development_pilot', parent_rule='pilot-product-text-v1',
        generic=['', 'hypothetical protein'], substrings=['uncharacterized'], prefixes=['duf'], reviewed_descriptions=[]))
    config = tmp_path / 'analysis.json'
    init_config(config, references=references, baseline_policy=baseline,
                reference_scope='synthetic test reference', cluster_scope='synthetic six clusters')
    return data, native, pred, config


def test_complete_analysis_and_cli(tmp_path):
    data, native, pred, cfg = inputs(tmp_path)
    assert main(['analyze-modules', '--dataset', str(native), '--predictions', str(pred), '--config', str(cfg), '--out', str(tmp_path/'out')]) == 0
    result = read_json(tmp_path/'out/report.json')
    assert (result['known_instances'], result['patterns'], result['dark_patterns']) == (6, 1, 1)
    assert result['inference_executed'] is False
    assert (tmp_path/'out/predictions.jsonl').read_bytes() == (pred/'modules.jsonl').read_bytes()
    status = read_json(tmp_path/'out/status.json')
    assert status['state'] == 'complete'
    for name, sha in status['files'].items():
        assert digest_file(tmp_path/'out'/name) == sha
    assert len((tmp_path/'out/instances.tsv').read_text().splitlines()) == 13
    with pytest.raises(FileExistsError):
        analyze(native, pred, cfg, tmp_path/'out')


@pytest.mark.parametrize('damage', ['prediction_bytes', 'dataset', 'coverage', 'ranks', 'id', 'support', 'qc', 'duplicate', 'cluster', 'unfinished'])
def test_invalid_inputs_never_complete(tmp_path, damage):
    data, native, pred, cfg = inputs(tmp_path)
    if damage == 'prediction_bytes':
        with (pred/'modules.jsonl').open('a') as stream:
            stream.write('{}\n')
    elif damage == 'dataset':
        data.proteins[0].product = 'changed'
        data.save(native)
    elif damage == 'cluster':
        data.genomes[0].cluster_id = ''
        data.save(native)
        seal(pred, native)
    elif damage == 'unfinished':
        write_json(pred/'status.json', {'state': 'failed', 'task': 'module'})
    elif damage == 'coverage':
        (pred/'genomes.jsonl').write_text('')
        seal(pred, native)
    else:
        rows = [json.loads(line) for line in (pred/'modules.jsonl').read_text().splitlines()]
        if damage == 'ranks': rows[0]['member_ranks'] = [0, 2]
        if damage == 'id': rows[0]['module_id'] = 'wrong'
        if damage == 'support': rows[0]['covering_windows'] = 2
        if damage == 'qc': rows[0]['span_qc_passed'] = False
        if damage == 'duplicate': rows.append(rows[0])
        write_rows(pred/'modules.jsonl', rows)
        seal(pred, native)
    with pytest.raises(ValueError):
        analyze(native, pred, cfg, tmp_path/'out')
    assert read_json(tmp_path/'out/status.json')['state'] == 'failed'


def test_resource_tamper_rejected_before_inference(tmp_path, monkeypatch):
    _, native, _, cfg = inputs(tmp_path)
    (tmp_path/'references.jsonl').write_text('')
    import phanet.downstream.predict as prediction
    monkeypatch.setattr(prediction, 'predict', lambda *a, **k: pytest.fail('must not run inference'))
    with pytest.raises(ValueError, match='SHA256'):
        discover(tmp_path/'out', tmp_path/'unused-model', cfg, dataset=native)
    assert not (tmp_path/'out').exists()


def test_sequential_pipeline_preserves_prediction_on_analysis_failure(tmp_path, monkeypatch):
    import shutil
    _, native, pred, cfg = inputs(tmp_path)
    import phanet.downstream.predict as prediction
    calls = []
    def saved_prediction(task, output, model, **kwargs):
        calls.append(task)
        shutil.copytree(pred, output)
        return read_json(pred/'summary.json')
    monkeypatch.setattr(prediction, 'predict', saved_prediction)
    summary = discover(tmp_path/'out', tmp_path/'unused-model', cfg, dataset=native)
    assert calls == ['module'] and summary['analysis']['dark_patterns'] == 1
    assert read_json(tmp_path/'out/status.json')['state'] == 'complete'
    settings = read_json(cfg)
    settings['catalogue']['max_pair_checks'] = 1
    write_json(cfg, settings)
    with pytest.raises(RuntimeError, match='budget'):
        discover(tmp_path/'failed', tmp_path/'unused-model', cfg, dataset=native)
    assert read_json(tmp_path/'failed/status.json')['state'] == 'failed'
    assert read_json(tmp_path/'failed/prediction/status.json')['state'] == 'complete'
    assert read_json(tmp_path/'failed/analysis/status.json')['state'] == 'failed'


def test_empty_predictions_keep_genomes_and_references(tmp_path):
    _, native, pred, cfg = inputs(tmp_path)
    (pred/'modules.jsonl').write_text('')
    rows = [json.loads(line) for line in (pred/'genomes.jsonl').read_text().splitlines()]
    for row in rows: row.update(retained=0, accepted=0)
    write_rows(pred/'genomes.jsonl', rows)
    write_json(pred/'summary.json', dict(genomes=6, proteins=24, candidates=0, accepted=0, span_qc_rejected=0))
    seal(pred, native)
    result = analyze(native, pred, cfg, tmp_path/'out')
    assert result['genomes'] == 6 and result['complete_references'] == 1 and result['patterns'] == 0
    assert len((tmp_path/'out/genomes.tsv').read_text().splitlines()) == 7


def test_partial_and_observed_reference_semantics(tmp_path):
    _, native, pred, cfg = inputs(tmp_path)
    reference = json.loads((tmp_path/'references.jsonl').read_text())
    reference.update(complete=False, exclusion_eligible='false')
    write_rows(tmp_path/'references.jsonl', [reference])
    with pytest.raises(ValueError, match='boolean'):
        read_references(tmp_path/'references.jsonl', Relationships())
    reference.update(exclusion_eligible=True)
    write_rows(tmp_path/'references.jsonl', [reference])
    with pytest.raises(ValueError, match='observed'):
        read_references(tmp_path/'references.jsonl', Relationships())
    reference.update(reference_kind='observed_member_set')
    write_rows(tmp_path/'references.jsonl', [reference])
    settings=read_json(cfg)
    settings['resources']['references']['sha256'] = digest_file(tmp_path/'references.jsonl')
    write_json(cfg, settings)
    result=analyze(native, pred, cfg, tmp_path/'out')
    assert result['observed_references'] == 1 and result['known_instances'] == 6


def test_sequence_evidence_direction_and_self_scores(tmp_path):
    data, native, pred, cfg = inputs(tmp_path)
    a, b = [p.sha256 for p in data.proteins[:2]]
    universe = tmp_path/'universe.json'
    write_json(universe, [a,b])
    alignments=tmp_path/'hits.tsv'
    alignments.write_text(f'{a}\t{a}\t0\t100\t1\t1\n{b}\t{b}\t0\t100\t1\t1\n{a}\t{b}\t1e-9\t80\t1\t1\n')
    settings, paths, _ = load_config(cfg)
    paths.update(sequence_alignments=alignments, sequence_proteins=universe)
    out=tmp_path/'relations'; out.mkdir()
    rel, _, counts = load_relationships(paths, settings, out, {a,b})
    assert rel.score(a,b) is not None and rel.score(b,a) is None
    assert counts['sequence_rows'] == 3
    alignments.write_text(f'{a}\t{b}\t1e-9\t80\t1\t1\n')
    other=tmp_path/'missing-self'; other.mkdir()
    with pytest.raises(ValueError, match='self scores'):
        load_relationships(paths, settings, other, {a,b})


def test_explicit_baseline_overlay_preserves_decisions(tmp_path):
    data, native, pred, cfg = inputs(tmp_path)
    overlay=tmp_path/'decisions.jsonl'
    write_rows(overlay, [dict(protein_id=p.protein_id, protein_sha256=p.sha256, unresolved=False,
        rule_id='test-baseline-v1', reason='synthetic reviewed decision') for p in data.proteins])
    settings=read_json(cfg)
    settings['resources']['baseline_decisions'] = dict(path=overlay.name, sha256=digest_file(overlay))
    settings['catalogue']['filter_annotations'] = False
    write_json(cfg, settings)
    result=analyze(native, pred, cfg, tmp_path/'out')
    assert result['dark_patterns'] == 0 and result['selected_patterns'] == 1
    assert result['annotation_filter_applied'] is False


def test_init_cli_binds_paths_and_refuses_overwrite(tmp_path):
    _, _, _, _ = inputs(tmp_path)
    command=['init-module-analysis', '--references', str(tmp_path/'references.jsonl'),
        '--baseline-policy', str(tmp_path/'baseline.json'), '--reference-scope', 'synthetic test',
        '--cluster-scope', 'synthetic clusters', '--out', str(tmp_path/'new.json')]
    assert main(command) == 0 and main(command) == 1
    assert load_config(tmp_path/'new.json')[0]['schema'] == 'phanet-module-analysis-v1'


def test_profile_qualification_and_conflicts(tmp_path):
    data, _, _, cfg = inputs(tmp_path)
    settings, paths, _ = load_config(cfg)
    settings['profile_scope'] = 'synthetic versioned profiles'
    scored = tmp_path/'profiles.tsv'
    write_table(scored, [dict(protein_sha256=data.proteins[0].sha256, profile='F1', evalue=1e-3, coverage=1),
                        dict(protein_sha256=data.proteins[2].sha256, profile='F2', evalue=1e-8, coverage=.8)],
                ['protein_sha256','profile','evalue','coverage'])
    paths['profiles'] = scored
    rel, _, _ = load_relationships(paths, settings, tmp_path, {p.sha256 for p in data.proteins})
    assert not rel.profiles[data.proteins[0].sha256]
    assert rel.profiles[data.proteins[2].sha256] == {'F2'}
    write_table(scored, [dict(protein_sha256=data.proteins[0].sha256, profile='F1', evalue=1e-9, coverage=1)],
                ['protein_sha256','profile','evalue','coverage'])
    with pytest.raises(ValueError, match='disagree'):
        load_relationships(paths, settings, tmp_path, set())


def test_qualified_profiles_preserved_and_require_scope(tmp_path):
    data, native, pred, cfg = inputs(tmp_path)
    profiles=tmp_path/'qualified.tsv'
    write_table(profiles, [dict(protein_sha256=data.proteins[2].sha256, qualified_profiles='["family-7"]')],
                ['protein_sha256','qualified_profiles'])
    settings=read_json(cfg)
    settings['resources']['qualified_profiles']=dict(path=profiles.name,sha256=digest_file(profiles))
    write_json(cfg,settings)
    with pytest.raises(ValueError,match='profile_scope'):
        load_config(cfg)
    settings['profile_scope']='synthetic qualified assignments'
    write_json(cfg,settings)
    report=analyze(native,pred,cfg,tmp_path/'out')
    assert report['evidence']['profile_rows']==1
    assert 'family-7' in (tmp_path/'out/members.tsv').read_text()


def test_qc_rejected_predictions_are_retained_only_in_raw(tmp_path):
    data,native,pred,cfg=inputs(tmp_path)
    data.proteins[0].partial=True
    data.save(native)
    rows=[json.loads(line) for line in (pred/'modules.jsonl').read_text().splitlines()]
    rows[0]['span_qc_passed']=False
    write_rows(pred/'modules.jsonl',rows)
    coverage=[json.loads(line) for line in (pred/'genomes.jsonl').read_text().splitlines()]
    coverage[0].update(accepted=1,span_qc_rejected=1)
    write_rows(pred/'genomes.jsonl',coverage)
    summary=read_json(pred/'summary.json');summary.update(accepted=11,span_qc_rejected=1)
    write_json(pred/'summary.json',summary);seal(pred,native)
    report=analyze(native,pred,cfg,tmp_path/'out')
    assert report['predicted_instances']==11 and report['span_qc_rejected']==1
    assert len((tmp_path/'out/predictions.jsonl').read_text().splitlines())==12


def test_default_policy_is_installed_and_hides_no_labels(tmp_path):
    _,_,_,_=inputs(tmp_path)
    init_config(tmp_path/'default.json',references=tmp_path/'references.jsonl',
                reference_scope='synthetic reference',cluster_scope='synthetic clusters')
    _,paths,_=load_config(tmp_path/'default.json')
    assert read_json(paths['baseline_policy'])['rule_id']=='public-product-text-v3-missing-markers'
