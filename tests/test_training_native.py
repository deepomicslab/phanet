"""Training continuation, objective isolation and native downstream decisions."""
from dataclasses import replace,asdict
import json
import numpy as np
import pytest
import torch
from phanet.common.io import write_table,write_json,digest_file,read_json,read_table
from phanet.datasets.schema import Dataset,Genome,Protein
from phanet.datasets.splits import SplitManifest,dataset_identity
from phanet.representations.embed import EmbeddingStore
from phanet.learning.data import prepare_training,load_training
from phanet.learning.objectives import family_bank,ObjectiveDataset
from phanet.learning.model import ModelConfig,PhaNetModel
from phanet.learning.training import TrainConfig,joint_loss
from phanet.learning.run import fit,evaluate
from phanet.downstream import native
from phanet.downstream.bundles import GOModel
from phanet.downstream.taxonomy_native import transfer,taxonomy_tree,TaxonomyModel
from phanet.downstream.taxonomy import TaxonomyThresholds
from phanet.downstream.native_train import select_binary_threshold,FineTuneConfig,fit as fit_task
from phanet.downstream.native_predict import predict


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    genomes=[Genome(f'g{i}',f'g{i}',12000,'synthetic') for i in range(6)]
    proteins=[Protein(f'g{i}:p{j}',f'g{i}',j*300,j*300+270,1,j,
        'M'+'ACDEFG'[i]*20+'HIKLMNPQRSTVWY'[j%14]*(j+2),labels=('Lysis',),
        family_ids=(f'f{j%2}',)) for i in range(6) for j in range(32)]
    data=Dataset(genomes,proteins);data.save(tmp_path/'input')
    store=EmbeddingStore.write(tmp_path/'embeddings',[p.protein_id for p in proteins],[p.sequence for p in proteins],
        np.random.default_rng(7).normal(size=(len(proteins),1280)),model='esm2_t33_650M_UR50D',config={'full_length':True})
    splits=SplitManifest('annotation','genome',{f'g{i}':('train' if i<2 else 'val' if i<4 else 'test') for i in range(6)},7,dataset_fingerprint=dataset_identity(data))
    splits.save(tmp_path/'splits.json')
    prepare_training(tmp_path/'training',dataset=tmp_path/'input',embeddings=store.directory,splits=tmp_path/'splits.json')
    return tmp_path,data,store,splits


def test_reject_truncated_store(prepared):
    root,data,store,splits=prepared
    path=store.directory/'embeddings.json';meta=json.loads(path.read_text());meta['config']['full_length']=False;write_json(path,meta)
    with pytest.raises(ValueError,match='full-length'):
        prepare_training(root/'bad',dataset=root/'input',embeddings=store.directory,splits=root/'splits.json')


def test_rbp_protein_partitions_share_native_genome(prepared):
    from phanet.downstream import rbp_training
    root,data,store,_=prepared
    rows=[dict(protein_id=p.protein_id,sequence_sha256=p.sha256,label=str(i%2),
               partition=('train','train','val','val','test')[i]) for i,p in enumerate(data.proteins[:5])]
    write_table(root/'protein-labels.tsv',rows,list(rows[0]))
    meta=rbp_training.prepare(root/'rbp-input',dataset=root/'input',embeddings=store.directory,
                              labels=root/'protein-labels.tsv')
    assert meta['partitions']=={'train':2,'val':2,'test':1}
    loaded=rbp_training.load(root/'rbp-input',root/'protein-labels.tsv')
    assert len(loaded.dataset.genomes)==6
    assert {p.genome_id for p in loaded.dataset.proteins[:5]}=={'g0'}
    rows[0]['partition']='test'
    write_table(root/'protein-labels.tsv',rows,list(rows[0]))
    with pytest.raises(ValueError,match='supervision changed'):
        rbp_training.load(root/'rbp-input',root/'protein-labels.tsv')


def test_rbp_partition_alias_and_identity_rejection(prepared):
    from phanet.downstream import rbp_training
    root,data,store,_=prepared
    ps=data.proteins[:5]
    rows=[dict(protein_id=p.protein_id,sequence_sha256=p.sha256,label=str(i%2),
               partition=('train','train','val','val','test')[i]) for i,p in enumerate(ps)]
    data.proteins[2]=replace(ps[2],sequence=ps[0].sequence)
    rows[2]['sequence_sha256']=ps[0].sha256
    write_table(root/'alias-labels.tsv',rows,list(rows[0]))
    with pytest.raises(ValueError,match='cross partitions'):
        rbp_training.validate_labels(data,root/'alias-labels.tsv')
    rows[2]['sequence_sha256']='0'*64
    write_table(root/'alias-labels.tsv',rows,list(rows[0]))
    with pytest.raises(ValueError,match='identity'):
        rbp_training.validate_labels(data,root/'alias-labels.tsv')


def test_family_prototypes_and_composition(prepared):
    root,data,store,splits=prepared
    base=load_training(root/'training')
    names,bank,targets=family_bank(data,store,splits)
    rows=[p for p in data.proteins if splits.assignments[p.genome_id]=='train' and 'f0' in p.family_ids]
    expected=store.get([p.protein_id for p in rows]).astype(np.float64).mean(0);expected/=np.linalg.norm(expected)
    np.testing.assert_allclose(bank[0],expected,atol=1e-7)
    view=ObjectiveDataset(base,families=targets,composition=True,sample_rate=1.)
    items=[view[i] for i in range(len(view))];active=[x for x in items if x['composition_eligible']]
    assert active
    batch=torch.utils.data.default_collate(active)
    model=PhaNetModel(replace(ModelConfig.tiny(),module_objective='composition'))
    loss,report=joint_loss(model,batch,TrainConfig(consistency_weight=0,composition_weight=.1),family_bank=bank)
    loss.backward();assert report['composition_pairs'] and report['family_examples']
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    validation=ObjectiveDataset(load_training(root/'training','val'),families=targets,composition=True,sample_rate=1.)
    a=validation[0];validation.set_epoch(5);b=validation[0]
    for k in a:assert torch.equal(a[k],b[k])


def test_fit_resume_export_and_evaluate(prepared):
    root,data,store,splits=prepared
    cfg=TrainConfig(max_epochs=2,warmup_steps=0,consistency_weight=0,family_weight=0,order_weight=0,checkpoint_interval=1)
    model=ModelConfig.tiny()
    full=fit(root/'training',root/'full',model_config=model,train_config=cfg,batch_size=2)
    partial=fit(root/'training',root/'partial',model_config=model,train_config=cfg,batch_size=2,max_updates=1)
    assert partial['state']=='paused'
    continued=fit(root/'training',root/'continued',batch_size=2,resume=root/'partial/last.pt')
    a=torch.load(root/'full/last.pt',weights_only=False);b=torch.load(root/'continued/last.pt',weights_only=False)
    for k in a['model']:assert torch.equal(a['model'][k],b['model'][k]),k
    from phanet.inference import load_model,glm_probabilities
    loaded=load_model(root/'continued/model')
    scores=glm_probabilities(data,[data.proteins[0].protein_id],store,loaded,batch_size=1)
    assert scores.shape==(1,9)
    assert evaluate(root/'training',root/'continued/best.pt')['annotated_proteins']==64


def hit(pid,family,genus,w,subfamily=''):
    return dict(genome_id='q',query_protein_id=pid,family=family,genus=genus,weight=w,subfamily=subfamily)


def test_taxonomy_paper_weight_denominator_and_global_hit():
    rows=[hit('a','F','G',.9),hit('b','F','G',.9),hit('c','B','X',.1)]
    result=transfer(['q'],rows,TaxonomyThresholds(min_support=1,min_margin=0))
    assert result[0]['ranks']['family']['confidence']==pytest.approx(.6)
    # The best hit has no genus; a weaker hit must not supply an extra genus vote.
    rows=[hit('a','F','',.9),hit('a','F','G',.8),hit('b','F','G',.9)]
    result=transfer(['q'],rows)
    assert result[0]['ranks']['family']['label']=='F'
    assert result[0]['ranks']['genus']['label']==''


def test_taxonomy_parent_mask_and_optional_subfamily():
    labels=[dict(genome_id=str(i),parent_id=str(i),partition='train',family=f,genus=g,subfamily='',order='O')
        for i,(f,g) in enumerate([('F','G'),('B','X')])]
    tree=taxonomy_tree(labels,1);model=TaxonomyModel(PhaNetModel(ModelConfig.tiny()),tree)
    with torch.no_grad():
        for h in model.heads.values():h.weight.zero_();h.bias.zero_()
        model.heads['family'].bias[model.classes['family'].index('F')]=5
        model.heads['genus'].bias[model.classes['genus'].index('X')]=10
    rows=model.classify(torch.zeros(64),{r:.5 for r in ('family','subfamily','genus')})
    assert rows['family']['label']=='F' and rows['genus']['label']=='G' and rows['order']['label']=='O'
    duplicate=[dict(r,genome_id=r['genome_id']+'fragment') for r in labels]
    with pytest.raises(ValueError,match='training-supported'):taxonomy_tree(labels+duplicate,2)


def test_rbp_gate_threshold_and_native_context(prepared):
    root,data,store,splits=prepared
    model=GOModel(PhaNetModel(ModelConfig.tiny()),1)
    with torch.no_grad():model.head.weight.zero_();model.head.bias.fill_(2.)
    native.save_model(model,root/'rbp-model',task='rbp',thresholds={'head':.6,'infection':.5},provenance={})
    rows=[dict(protein_id=p.protein_id,protein_sha256=p.sha256,infection_score=.8 if i==0 else .2) for i,p in enumerate(data.proteins)]
    write_table(root/'infection.tsv',rows,list(rows[0]))
    result=predict('rbp',root/'prediction',root/'rbp-model',dataset=root/'input',embeddings=store.directory,infection_scores=root/'infection.tsv')
    assert result['candidates']==1
    from phanet.common.io import read_table
    output=list(read_table(root/'prediction/rbp.tsv'));assert float(output[1]['pipeline_score'])==0
    batch,offset=native.target_batch(data,store,[data.proteins[0].protein_id],'cpu')
    assert offset==[0] and batch[4].sum()==30 and batch[1][0,0].sum()==0 and batch[1][0,1].sum()>0
    threshold,f1=select_binary_threshold([1,0,1],[.8,.8,.1],[1,1,0])
    assert threshold==.8 and f1==pytest.approx(.5)


@pytest.mark.parametrize('protein_manifest', [False, True])
@pytest.mark.parametrize('gated', [False, True])
def test_rbp_finetune_and_reload(prepared, protein_manifest, gated):
    root,data,store,splits=prepared
    cfg=TrainConfig(max_epochs=1,warmup_steps=0,consistency_weight=0,family_weight=0,order_weight=0)
    fit(root/'training',root/'glm',model_config=ModelConfig.tiny(),train_config=cfg,batch_size=8)
    rows=[dict(protein_id=p.protein_id,sequence_sha256=p.sha256,label=i%2,partition=splits.assignments[p.genome_id]) for i,p in enumerate(data.proteins)]
    write_table(root/'labels.tsv',rows,list(rows[0]))
    scores=[dict(protein_id=p.protein_id,protein_sha256=p.sha256,infection_score=.8) for p in data.proteins]
    write_table(root/'infection.tsv',scores,list(scores[0]))
    directory=root/'training'
    if protein_manifest:
        from phanet.downstream.rbp_training import prepare
        directory=root/'protein-training'
        prepare(directory,dataset=root/'input',embeddings=store.directory,labels=root/'labels.tsv')
    fit_task('rbp',directory,root/'labels.tsv',root/'glm/model',root/'rbp',
        config=FineTuneConfig(epochs=1,batch_size=64),infection_scores=root/'infection.tsv' if gated else None)
    _,meta=native.load_model(root/'rbp/model','rbp')
    assert ('infection' in meta['thresholds'])==gated
    history=read_json(root/'rbp/history.json')
    assert ('pipeline_f1' in history[0])==gated
    if not gated:
        result=predict('rbp',root/'standalone',root/'rbp/model',dataset=root/'input',embeddings=store.directory)
        output=list(read_table(root/'standalone/rbp.tsv'))
        assert len(output)==len(data.proteins)
        assert all(row['infection_score']=='' and row['head_score']!='' for row in output)
        assert result['rbp_mode']=='standalone' and result['infection_assigned'] is None
        assert result['candidates']==sum(float(row['head_score'])>=meta['thresholds']['head'] for row in output)
        with pytest.raises(ValueError,match='do not use Infection'):
            predict('rbp',root/'wrong-mode',root/'rbp/model',dataset=root/'input',
                embeddings=store.directory,infection_scores=root/'infection.tsv')


def test_family_loss_requires_bank(prepared):
    from phanet.learning.training import Trainer
    with pytest.raises(ValueError,match='family bank'):
        Trainer(PhaNetModel(ModelConfig.tiny()),TrainConfig(total_steps=1),data_hash='test')
    root,data,store,splits=prepared
    data.proteins[-1].family_ids=('heldout-only',)
    names,bank,targets=family_bank(data,store,splits)
    assert 'heldout-only' not in names and targets[-1]==-1


def test_taxonomy_fit_reload_calibration_and_empty(prepared):
    root,data,store,splits=prepared
    cfg=TrainConfig(max_epochs=1,warmup_steps=0,consistency_weight=0,family_weight=0,order_weight=0)
    fit(root/'training',root/'glm',model_config=ModelConfig.tiny(),train_config=cfg,batch_size=8)
    labels=[dict(genome_id=g.genome_id,partition=splits.assignments[g.genome_id],family='F',genus='G',order='O') for g in data.genomes]
    write_table(root/'taxonomy.tsv',labels,list(labels[0]))
    fit_task('taxonomy',root/'training',root/'taxonomy.tsv',root/'glm/model',root/'tax',
        config=FineTuneConfig(epochs=1,batch_size=32),minimum_parents=1)
    model,meta=native.load_model(root/'tax/model','taxonomy')
    assert (root/'tax/reference-labels.tsv').is_file()
    result=predict('taxonomy',root/'classified',root/'tax/model',method='glm',dataset=root/'input',embeddings=store.directory)
    assert result['assigned']==6
    empty=Dataset([Genome('empty','empty',100,'synthetic')],[]);empty.save(root/'empty')
    result=predict('taxonomy',root/'empty-result',root/'tax/model',method='glm',dataset=root/'empty')
    assert result['genomes']==1 and result['assigned']==0
    from phanet.downstream.native_train import calibrate_taxonomy,_taxonomy_objective
    hits=[dict(genome_id='g2',query_protein_id=f'g2:p{i}',family='F',genus='G',order='O',weight=.9) for i in range(3)]
    truth=[r for r in labels if r['partition']=='val'];vectors=[torch.zeros(64) for r in truth]
    score,thresholds,limits=calibrate_taxonomy(model,vectors,truth,hits)
    predictions=transfer([r['genome_id'] for r in truth],hits,TaxonomyThresholds(**limits))
    for i,row in enumerate(predictions):
        if not row['ranks']['family']['label']:row['ranks']=model.classify(vectors[i],thresholds)
    assert score==_taxonomy_objective(truth,[r['ranks'] for r in predictions],model.tree)


def test_score_infection(prepared,monkeypatch):
    from phanet.downstream.native_predict import score_infection
    root,data,store,splits=prepared
    monkeypatch.setattr(native,'infection_scores',lambda data,*a,**kw:{p.protein_id:.8 for p in data.proteins})
    write_json(root/'model.json',{})
    from dataclasses import dataclass
    @dataclass
    class Config:
        reference:str='test-reference'
    result=score_infection(root/'scored',root,Config(),dataset=root/'input',embeddings=store.directory)
    assert result['proteins']==len(data.proteins)
    scores=native.infection_table(root/'scored/infection.tsv',data,required=[data.proteins[0].protein_id])
    assert len(scores)==len(data.proteins)


@pytest.mark.parametrize('epochs', [1, 2])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_resume_terminal_update_validates_and_selects(prepared, epochs, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    root, _, _, _ = prepared
    cfg = TrainConfig(max_epochs=epochs, warmup_steps=0, consistency_weight=0,
                      family_weight=0, order_weight=0, checkpoint_interval=1)
    kwargs = dict(model_config=ModelConfig.tiny(), train_config=cfg, batch_size=2, device=device)
    fit(root/'training', root/'full', **kwargs)
    fit(root/'training', root/'partial', max_updates=2*epochs, **kwargs)
    paused = torch.load(root/'partial/last.pt', map_location='cpu', weights_only=False)
    assert paused['step'] == paused['train_config']['total_steps']
    result = fit(root/'training', root/'resumed', batch_size=2, device=device, resume=root/'partial/last.pt')
    assert result['state'] == 'complete'
    assert [r['epoch'] for r in result['epochs']] == [epochs]
    for filename in ('last.pt', 'best.pt'):
        expected = torch.load(root/'full'/filename, map_location='cpu', weights_only=False)
        actual = torch.load(root/'resumed'/filename, map_location='cpu', weights_only=False)
        assert actual['sampler_state'] == expected['sampler_state']
        assert actual['best_validation'] == expected['best_validation']
        for key in expected['model']:
            assert torch.equal(actual['model'][key], expected['model'][key]), key


def test_mixed_subfamily_branch_training_prediction_reload(tmp_path):
    labels = [dict(genome_id='direct', partition='train', family='F', subfamily='', genus='Direct', order='O'),
              dict(genome_id='sub', partition='train', family='F', subfamily='S', genus='InS', order='O')]
    model = TaxonomyModel(PhaNetModel(ModelConfig.tiny()), taxonomy_tree(labels, 1))
    thresholds = {r: .6 for r in ('family', 'subfamily', 'genus')}
    vector = torch.zeros(64)
    with torch.no_grad():
        for head in model.heads.values():
            head.weight.zero_(); head.bias.zero_()
        model.heads['genus'].bias[model.classes['genus'].index('Direct')] = 20
        model.heads['genus'].bias[model.classes['genus'].index('InS')] = -20
    assert model.classify(vector, thresholds)['genus']['label'] == ''
    model.loss(vector, labels[0]).backward()
    direct = model.direct_branches['F']; sub = model.classes['subfamily'].index('S')
    assert model.heads['subfamily'].bias.grad[direct] < 0
    assert model.heads['subfamily'].bias.grad[sub] > 0
    with torch.no_grad():
        model.heads['subfamily'].bias[direct] = 20
    result = model.classify(vector, thresholds)
    assert result['genus']['label'] == 'Direct' and result['subfamily']['label'] == ''
    native.save_model(model, tmp_path/'model', task='taxonomy', thresholds=thresholds, provenance={}, tree=model.tree)
    loaded, _ = native.load_model(tmp_path/'model', 'taxonomy')
    assert loaded.classify(vector, thresholds) == result
    with torch.no_grad():
        loaded.heads['subfamily'].bias[direct] = -20
        loaded.heads['subfamily'].bias[sub] = 20
    assert loaded.classify(vector, thresholds)['genus']['label'] == 'InS'
    loaded.zero_grad()
    with torch.no_grad(): loaded.heads['subfamily'].bias.zero_()
    loaded.loss(vector, labels[1]).backward()
    assert loaded.heads['subfamily'].bias.grad[sub] < 0
    assert loaded.heads['subfamily'].bias.grad[direct] > 0


def test_mixed_subfamily_transfer_direct_genus():
    rows = [hit('a', 'F', 'Direct', .9), hit('b', 'F', 'Direct', .9), hit('c', 'F', 'InS', .1, 'S')]
    result = transfer(['q'], rows)[0]['ranks']
    assert result['genus']['label'] == 'Direct'
    assert result['subfamily']['label'] == ''


def test_optional_subfamily_calibration_matches_prediction():
    from phanet.downstream.native_train import calibrate_taxonomy, _taxonomy_objective
    truth = [dict(genome_id='a', partition='train', family='F', subfamily='S', genus='InS', order='O'),
             dict(genome_id='b', partition='train', family='F', subfamily='', genus='Direct', order='O')]
    model = TaxonomyModel(PhaNetModel(ModelConfig.tiny()), taxonomy_tree(truth, 1))
    with torch.no_grad():
        for head in model.heads.values(): head.weight.zero_(); head.bias.zero_()
        direct = model.direct_branches['F']
        model.heads['subfamily'].bias[direct] = .2
        model.heads['subfamily'].weight[direct, 0] = 5
    vectors = [torch.zeros(64), torch.zeros(64)]; vectors[1][0] = 1
    score, thresholds, _ = calibrate_taxonomy(model, vectors, truth)
    assert thresholds['subfamily'] > .5
    rows = [model.classify(v, thresholds) for v in vectors]
    assert not rows[0]['genus']['label'] and rows[1]['genus']['label'] == 'Direct'
    assert _taxonomy_objective(truth, rows, model.tree) == score


def test_partial_taxonomy_merges_only_known_training_ancestry():
    from copy import deepcopy
    labels=[dict(genome_id='a',partition='train',genus='G',family=None),
            dict(genome_id='b',partition='train',genus='G',subfamily='S',family='F'),
            dict(genome_id='c',partition='train',subfamily='S',family='',order='O'),
            dict(genome_id='v',partition='val',genus='G',family='Other')]
    original=deepcopy(labels)
    tree=taxonomy_tree(labels,1)
    assert tree['genus']['G']==dict(order='O',family='F',subfamily='S')
    assert labels==original
    assert taxonomy_tree(list(reversed(labels)),1)==tree
    with pytest.raises(ValueError,match='Conflicting'):
        taxonomy_tree(labels+[dict(genome_id='bad',partition='train',genus='G',family='Other')],1)
    rare=taxonomy_tree(labels,2)
    assert 'G' in rare['genus'] and not rare['family']
    model=TaxonomyModel(PhaNetModel(ModelConfig.tiny()),tree)
    assert model.loss(torch.zeros(64),labels[0]) is not None


@pytest.mark.parametrize('device',['cpu',pytest.param('cuda',marks=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable'))])
def test_orphan_genus_routing_gradient_calibration_reload(tmp_path,device):
    from phanet.downstream.native_train import calibrate_taxonomy,_taxonomy_objective
    labels=[dict(genome_id='f',partition='train',family='F',genus='InF',order='O'),
            dict(genome_id='x',partition='train',family='',genus='Orphan',order=''),
            dict(genome_id='y',partition='train',family='',genus='Other',order='')]
    model=TaxonomyModel(PhaNetModel(ModelConfig.tiny()),taxonomy_tree(labels,1)).to(device)
    vector=torch.zeros(64,device=device,requires_grad=True)
    with torch.no_grad():
        for head in model.heads.values():head.weight.zero_();head.bias.zero_()
    loss=model.loss(vector,labels[1]);assert torch.isfinite(loss)
    loss.backward()
    assert model.heads['family'].bias.grad[model.family_direct]<0
    assert model.heads['genus'].bias.grad[model.classes['genus'].index('Orphan')]<0
    assert model.heads['genus'].bias.grad[model.classes['genus'].index('InF')]==0
    before=model.heads['genus'].bias.detach().clone()
    torch.optim.SGD(model.parameters(),lr=.1).step()
    assert not torch.equal(before,model.heads['genus'].bias)
    with torch.no_grad():
        model.heads['family'].bias[model.family_direct]=10
        model.heads['genus'].bias[model.classes['genus'].index('Orphan')]=10
    threshold={r:.5 for r in ('family','subfamily','genus')}
    out=model.classify(vector,threshold)
    assert out['genus']['label']=='Orphan' and out['family']['label']==out['order']['label']==''
    truth=[labels[1]];score,threshold,_=calibrate_taxonomy(model,[vector],truth)
    assert score==_taxonomy_objective(truth,[model.classify(vector,threshold)],model.tree)
    native.save_model(model,tmp_path/'model',task='taxonomy',thresholds=threshold,provenance={},tree=model.tree)
    loaded,_=native.load_model(tmp_path/'model','taxonomy',device=device)
    assert loaded.classify(vector,threshold)==model.classify(vector,threshold)
    with torch.no_grad():
        loaded.heads['family'].bias.zero_()
    assert loaded.classify(vector,{r:.75 for r in threshold})['genus']['label']==''


def test_missing_family_with_mixed_subfamily_paths():
    labels=[dict(genome_id='a',partition='train',genus='Direct',family='',subfamily=''),
            dict(genome_id='b',partition='train',genus='InS',family='',subfamily='S')]
    model=TaxonomyModel(PhaNetModel(ModelConfig.tiny()),taxonomy_tree(labels,1))
    assert not model.tree['family']
    with torch.no_grad():
        for head in model.heads.values():head.weight.zero_();head.bias.zero_()
        model.heads['subfamily'].bias[model.direct_branches['']]=10
    out=model.classify(torch.zeros(64),{r:.5 for r in ('family','subfamily','genus')})
    assert out['genus']['label']=='Direct' and not out['family']['label'] and not out['subfamily']['label']
    assert model.loss(torch.zeros(64),labels[1]) is not None


def test_orphan_reference_transfer_and_calibration():
    from phanet.downstream.native_predict import reference_taxonomy
    from phanet.downstream.native_train import calibrate_taxonomy,_taxonomy_objective
    labels=[dict(genome_id='q',partition='train',family='',genus='G',order='O')]
    tree=taxonomy_tree(labels,1)
    assert reference_taxonomy(labels[0],tree)['family']==''
    with pytest.raises(ValueError,match='Conflicting'):
        reference_taxonomy(dict(labels[0],order='Other'),tree)
    hits=[hit('p1','','G',.9),hit('p2','','G',.9)]
    result=transfer(['q'],hits)[0]['ranks']
    assert result['genus']['label']=='G' and not result['family']['label']
    model=TaxonomyModel(PhaNetModel(ModelConfig.tiny()),tree)
    score,thresholds,limits=calibrate_taxonomy(model,[torch.zeros(64)],labels,hits)
    rows=[r['ranks'] for r in transfer(['q'],hits,TaxonomyThresholds(**limits))]
    rows=[row if any(v['label'] for v in row.values()) else model.classify(torch.zeros(64),thresholds) for row in rows]
    assert score==_taxonomy_objective(labels,rows,tree)


def test_genus_only_native_fit_predict(prepared):
    root,data,store,splits=prepared
    cfg=TrainConfig(max_epochs=1,warmup_steps=0,consistency_weight=0,family_weight=0,order_weight=0)
    fit(root/'training',root/'glm',model_config=ModelConfig.tiny(),train_config=cfg,batch_size=8)
    labels=[dict(genome_id=g.genome_id,partition=splits.assignments[g.genome_id],genus='G') for g in data.genomes]
    write_table(root/'labels.tsv',labels,list(labels[0]))
    result=fit_task('taxonomy',root/'training',root/'labels.tsv',root/'glm/model',root/'tax',
                    config=FineTuneConfig(epochs=1,batch_size=32),minimum_parents=1)
    assert result['state']=='complete'
    model,_=native.load_model(root/'tax/model','taxonomy')
    assert not model.tree['family'] and 'G' in model.tree['genus']
    result=predict('taxonomy',root/'predicted',root/'tax/model',method='glm',dataset=root/'input',embeddings=store.directory)
    assert result['assigned']==6
    rows=json.loads((root/'predicted/predictions.json').read_text())
    assert all(r['method']=='context_glm' and r['ranks']['genus']['label']=='G' and r['ranks']['family']['label']=='' for r in rows)


def test_taxonomy_hits_cli_accepts_genus_only(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from phanet.downstream import native_cli,native_predict
    from phanet.network.inference import NetworkConfig
    write_table(tmp_path/'labels.tsv',[dict(genome_id='g',partition='train',genus='G')],['genome_id','partition','genus'])
    monkeypatch.setattr(Dataset,'load',lambda path:None)
    monkeypatch.setattr(NetworkConfig,'load',lambda path:None)
    trees=[]
    def search(*args,**kwargs):
        trees.append(kwargs['tree']);return []
    monkeypatch.setattr(native_predict,'taxonomy_hits',search)
    result=native_cli.run(SimpleNamespace(command='taxonomy-hits',labels=tmp_path/'labels.tsv',dataset='data',network_config='network',
        reference_labels='refs',out=tmp_path/'out',minimum_parents=1,device='cpu',threads=1))
    assert result=={'hits':0} and 'G' in trees[0]['genus'] and not trees[0]['family']
