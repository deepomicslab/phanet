"""Regression tests for public input mistakes and preparation-stage ordering."""
from pathlib import Path
import json

import pytest

from phanet.analysis.annotation_baseline import ProductTextBaseline
from phanet.analysis.public import discover, init_config, load_config
from phanet.common.io import read_json, write_json, write_table
from test_module_analysis_public import inputs


@pytest.mark.parametrize('product', ['-', 'NA', 'N/A', '  na  '])
def test_default_baseline_treats_missing_markers_as_unresolved(tmp_path, product):
    inputs(tmp_path)
    init_config(tmp_path/'default.json', references=tmp_path/'references.jsonl',
                reference_scope='synthetic reference', cluster_scope='synthetic clusters')
    _, paths, _ = load_config(tmp_path/'default.json')
    policy = ProductTextBaseline(read_json(paths['baseline_policy']))
    assert policy.assess(product).unresolved
    assert not policy.assess('DNA polymerase').unresolved
    old = Path(__file__).parents[1]/'src/phanet/analysis/policies/product-text-v2.json'
    assert ProductTextBaseline(read_json(old)).assess('-').unresolved is False


@pytest.mark.parametrize('field', ['generic', 'substrings', 'prefixes', 'reviewed_descriptions'])
def test_malformed_policy_lists_are_rejected(tmp_path, field):
    inputs(tmp_path)
    policy = read_json(tmp_path/'baseline.json')
    policy[field] = 'unknown'
    with pytest.raises(ValueError):
        ProductTextBaseline(policy)


def test_invalid_initializer_does_not_leave_config(tmp_path):
    inputs(tmp_path)
    destination=tmp_path/'new.json'
    with pytest.raises(ValueError):
        init_config(destination, references=tmp_path/'references.jsonl',
                    reference_scope='', cluster_scope='synthetic clusters')
    assert not destination.exists()
    init_config(destination, references=tmp_path/'references.jsonl',
                reference_scope='synthetic reference', cluster_scope='synthetic clusters')
    assert load_config(destination)


def test_initializer_resolves_symlinked_parent(tmp_path):
    inputs(tmp_path)
    actual=tmp_path/'different/depth/configs';actual.mkdir(parents=True)
    alias=tmp_path/'alias';alias.symlink_to(actual, target_is_directory=True)
    init_config(alias/'analysis.json', references=tmp_path/'references.jsonl',
                reference_scope='synthetic reference', cluster_scope='synthetic clusters')
    _, paths, _ = load_config(actual/'analysis.json')
    assert paths['references']==(tmp_path/'references.jsonl').resolve()


def test_fasta_missing_clusters_stops_before_model_inference(tmp_path, monkeypatch):
    data, _, _, cfg=inputs(tmp_path)
    for g in data.genomes:g.cluster_id=''
    import phanet.downstream.predict as prediction
    calls=[]
    def prepare(source, output, config, **kwargs):
        calls.append('prepare')
        data.save(Path(output)/'prepared/dataset')
    def forbidden(*args, **kwargs):
        pytest.fail('Embedding/model inference must not run before cluster validation')
    monkeypatch.setattr(prediction, 'prepare', prepare)
    monkeypatch.setattr(prediction.bundles, 'metadata', lambda *a: (None, {}))
    monkeypatch.setattr(prediction.bundles, 'load', forbidden)
    monkeypatch.setattr(prediction, 'embed_esm', forbidden)
    checkpoint=tmp_path/'esm.pt';checkpoint.write_bytes(b'synthetic path-only preflight stub')
    with pytest.raises(ValueError, match='cluster'):
        discover(tmp_path/'out', tmp_path/'unused-model', cfg, input_fasta=tmp_path/'input.fna',
                 config=object(), esm_checkpoint=checkpoint)
    assert calls==['prepare']
    assert read_json(tmp_path/'out/status.json')['state']=='failed'


def test_fasta_prepares_once_and_validates_before_predicting(tmp_path, monkeypatch):
    data, _, pred, cfg=inputs(tmp_path)
    for g in data.genomes:g.cluster_id=''
    clusters=tmp_path/'clusters.tsv'
    write_table(clusters, [dict(genome_id=g.genome_id,cluster_id='shared-cluster') for g in data.genomes],['genome_id','cluster_id'])
    init_config(tmp_path/'bound.json',references=tmp_path/'references.jsonl',baseline_policy=tmp_path/'baseline.json',
                reference_scope='synthetic',cluster_scope='synthetic',clusters=clusters)
    import phanet.downstream.predict as prediction
    calls=[]
    def prepare(source,output,config,**kwargs):
        calls.append('prepare');data.save(Path(output)/'prepared/dataset')
    from phanet.analysis import public
    original_validator=public.load_loci
    def validate(*args, **kwargs):
        calls.append('validate')
        return original_validator(*args, **kwargs)
    monkeypatch.setattr(public,'load_loci',validate)
    def embed(*args, **kwargs):
        assert calls==['prepare','validate']
        calls.append('embed')
        from test_downstream_contract import embedding_store
        return embedding_store(tmp_path, data)
    def load(*args, **kwargs):
        calls.append('load-model')
        return None, {'prediction': {}}
    rows=[json.loads(line) for line in (pred/'modules.jsonl').read_text().splitlines()]
    coverage={r['genome_id']: r for r in map(json.loads, (pred/'genomes.jsonl').read_text().splitlines())}
    def module_rows(*args, **kwargs):
        for gid in coverage:
            yield gid, [r for r in rows if r['genome_id']==gid], {k:v for k,v in coverage[gid].items() if k!='genome_id'}
    monkeypatch.setattr(prediction,'prepare',prepare)
    monkeypatch.setattr(prediction.bundles,'metadata',lambda *a:(None, {}))
    monkeypatch.setattr(prediction.bundles,'load',load)
    monkeypatch.setattr(prediction,'embed_esm',embed)
    monkeypatch.setattr(prediction,'module_rows',module_rows)
    checkpoint=tmp_path/'esm.pt';checkpoint.write_bytes(b'synthetic path-only preflight stub')
    summary=discover(tmp_path/'out',tmp_path/'unused-model',tmp_path/'bound.json',
                     input_fasta=tmp_path/'input.fna',config=object(),esm_checkpoint=checkpoint)
    assert calls==['prepare','validate','embed','load-model','validate']
    assert summary['analysis']['genome_clusters']==1
    assert read_json(tmp_path/'out/status.json')['state']=='complete'
