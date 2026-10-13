"""Fine-tune native RBP and hierarchical taxonomy heads."""
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
import copy
import math
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from phanet.common.io import digest_file, read_table, write_json, write_table
from phanet.inference import load_model
from phanet.learning.data import load_training
from phanet.learning.training import seed_all
from phanet.downstream.bundles import GOModel
from phanet.downstream import native
from phanet.downstream.taxonomy_native import TaxonomyModel,taxonomy_tree,RANKS,transfer
from phanet.downstream.taxonomy import TaxonomyThresholds


@dataclass(frozen=True)
class FineTuneConfig:
    epochs:int=50
    batch_size:int=64
    learning_rate:float=5e-5
    weight_decay:float=.01
    patience:int=8
    seed:int=7
    train_last_layers:int=4
    def __post_init__(self):
        if min(self.epochs,self.batch_size,self.patience,self.train_last_layers)<1 or not math.isfinite(self.learning_rate) or self.learning_rate<=0 or not math.isfinite(self.weight_decay) or self.weight_decay<0:
            raise ValueError('Invalid fine-tuning configuration')


def select_binary_threshold(truth,scores,gate):
    truth=np.asarray(truth,bool); scores=np.asarray(scores,float);gate=np.asarray(gate,bool)
    if truth.shape!=scores.shape or gate.shape!=scores.shape or not len(truth) or not np.isfinite(scores).all():
        raise ValueError('Invalid validation predictions')
    # Include a no-positive operating point without serializing infinity.
    order=np.argsort(-scores[gate],kind='stable'); p=scores[gate][order]; y=truth[gate][order]
    if not len(p): return 1.000001,0.
    ends=np.r_[np.flatnonzero(p[:-1]!=p[1:]),len(p)-1]
    tp=np.cumsum(y)[ends]; predicted=ends+1
    f1=2*tp/np.maximum(predicted+truth.sum(),1)
    best=float(f1.max())
    if best==0: return 1.000001,0.
    i=int(np.flatnonzero(f1==best)[0])
    return float(p[ends[i]]),best


def _taxonomy_objective(truth,predictions,tree):
    scores=[];coverage=[];depth=0
    for rank in ('order','family','genus'):
        classes=sorted({p.get('order','') for nodes in tree.values() for p in nodes.values()}-{''}) if rank=='order' else sorted(tree[rank])
        chosen=[i for i,t in enumerate(truth) if t.get(rank,'') in classes]
        if not chosen: continue
        y=[truth[i][rank] for i in chosen]; p=[predictions[i].get(rank,{}).get('label','') for i in chosen]
        evaluated=sorted(set(y))
        scores.append(float(f1_score(y,p,labels=evaluated,average='macro',zero_division=0)))
        coverage.append(sum(bool(x) for x in p)/len(p));depth+=sum(bool(x) for x in p)
    return (float(np.mean(scores)) if scores else 0.,float(np.mean(coverage)) if coverage else 0.,depth)


def calibrate_taxonomy(model,vectors,truth,hits=None):
    grid=[round(.5+.05*i,2) for i in range(10)]
    settings=list(product(grid,repeat=3))
    with torch.inference_mode():
        paths=[model.classify(v,{r:0. for r in RANKS}) for v in vectors]
    evaluated=('order','family','genus')
    classes={r:sorted({p.get('order','') for nodes in model.tree.values() for p in nodes.values()}-{''})
             if r=='order' else sorted(model.tree[r]) for r in evaluated}
    indices={r:{label:i for i,label in enumerate(classes[r])} for r in evaluated}
    expected=np.array([[indices[r].get(t.get(r,''),-1) for t in truth] for r in evaluated])
    predictions=np.full((len(settings),3,len(truth)),-1,np.int64)
    for j,values in enumerate(settings):
        thresholds=dict(zip(RANKS,values))
        for i,path in enumerate(paths):
            for rank in RANKS:
                node=path[rank]
                if node['confidence'] is None: continue
                if node['confidence']<thresholds[rank]: break
                name=node['label']
                if not name: continue
                if rank in evaluated: predictions[j,evaluated.index(rank),i]=indices[rank][name]
                order=model.tree[rank][name].get('order','')
                if order and predictions[j,0,i]<0: predictions[j,0,i]=indices['order'].get(order,-1)
    options=[TaxonomyThresholds()]
    if hits is not None:
        options=[TaxonomyThresholds(*values) for values in product((2,3,5),(.5,.6,.7),(.5,.6,.7),(.05,.1,.2))]
    best=None;query=[t['genome_id'] for t in truth]
    def candidates():
        yield from options
        if hits is None:return
        # Refine rank-specific transfer limits while retaining the complete classifier grid.
        while True:
            previous=best[0]
            for rank in RANKS:
                base=best[2]
                for option in options:
                    values=asdict(option)
                    overrides=dict(base.get('rank_overrides',{}))
                    overrides[rank]={k:values[k] for k in ('min_support','min_proportion','min_confidence','min_margin')}
                    yield TaxonomyThresholds(**(base|{'rank_overrides':overrides}))
            if best[0]<=previous:return
    for limits in candidates():
        network=transfer(query,hits or [],limits)
        mask=np.array([any(v['label'] for v in r['ranks'].values()) for r in network])
        network_ids=np.array([[indices[r].get(row['ranks'][r]['label'],-1) for row in network] for r in evaluated])
        combined=np.where(mask[None,None,:],network_ids[None],predictions)
        metrics=[];coverages=[]
        for rank in range(3):
            eligible=expected[rank]>=0
            if not eligible.any():continue
            y=expected[rank,eligible];p=combined[:,rank,eligible]
            f=[]
            for label in np.unique(y):
                positive=y==label
                tp=((p==label)&positive).sum(1)
                denominator=(p==label).sum(1)+positive.sum()
                f.append(2*tp/np.maximum(denominator,1))
            metrics.append(np.mean(f,axis=0));coverages.append((p>=0).mean(1))
        metric=np.mean(metrics,axis=0) if metrics else np.zeros(len(settings))
        coverage=np.mean(coverages,axis=0) if coverages else np.zeros(len(settings))
        depth=((combined>=0)&(expected[None]>=0)).sum((1,2))
        chosen=max(range(len(settings)),key=lambda i:(metric[i],coverage[i],depth[i]))
        score=(float(metric[chosen]),float(coverage[chosen]),int(depth[chosen]))
        if best is None or score>best[0]:best=(score,dict(zip(RANKS,settings[chosen])),asdict(limits))
    return best


def fit(task,directory,labels_path,initialize,output,*,config=None,device='cpu',infection_scores=None,
        validation_hits=None,minimum_parents=5,progress=print):
    if task not in ('rbp','taxonomy'):raise ValueError('Unknown fine-tuning task')
    config=config or FineTuneConfig();seed_all(config.seed)
    from phanet.common.io import read_json
    manifest=read_json(Path(directory)/'training.json')
    if task=='rbp' and manifest.get('task')=='rbp':
        from phanet.downstream.rbp_training import load as load_rbp_training
        base=load_rbp_training(directory,labels_path)
    else:
        base=load_training(directory)
    data,store=base.dataset,base.embeddings
    labels=list(read_table(labels_path,('protein_id','label','partition') if task=='rbp' else ('genome_id','partition')))
    key='protein_id' if task=='rbp' else 'genome_id'
    native_groups=data.by_genome()
    mapping={p.protein_id:p.genome_id for p in data.proteins} if task=='rbp' else {g.genome_id:g.genome_id for g in data.genomes}
    if len({r[key] for r in labels})!=len(labels):raise ValueError('Duplicate supervised identifier')
    for row in labels:
        if row[key] not in mapping or row['partition'] not in ('train','val','test'):
            raise ValueError('Supervision requires native identifiers and train/val/test partitions')
        if task=='taxonomy' and row['partition']!=base.splits.assignments[mapping[row[key]]]:
            raise ValueError('Taxonomy supervision must match prepared genome partitions')
        if task=='rbp':
            if row['label'] not in ('0','1'):raise ValueError('RBP labels must be 0 or 1')
        elif not native_groups[row[key]]:raise ValueError('Labeled taxonomy genome has no proteins')
    train=[r for r in labels if r['partition']=='train'];val=[r for r in labels if r['partition']=='val']
    if not train or not val:raise ValueError('Nonempty training and validation supervision required')
    if task=='rbp' and (len({r['label'] for r in train})!=2 or len({r['label'] for r in val})!=2):
        raise ValueError('RBP train and validation require both classes')
    groups={}
    proteins={p.protein_id:p for p in data.proteins}
    for row in labels:
        group=proteins[row[key]].sha256 if task=='rbp' else row.get('parent_id') or row[key]
        groups.setdefault(group,set()).add(row['partition'])
    if any(len(parts)>1 for parts in groups.values()):
        raise ValueError('Identical target sequences or taxonomy parent genomes cross partitions')
    hits=list(read_table(validation_hits,('genome_id','query_protein_id','weight'))) if validation_hits else None
    if hits is not None:
        allowed={r[key] for r in val}
        if any(r['genome_id'] not in allowed or r['query_protein_id'] not in proteins or proteins[r['query_protein_id']].genome_id!=r['genome_id'] for r in hits):
            raise ValueError('Calibration hits must belong to validation native proteins')
    encoder=load_model(initialize)
    if task=='rbp':
        model=GOModel(encoder,1)
        gate_scores=(native.infection_table(infection_scores,data,required=[r[key] for r in labels])
                     if infection_scores is not None else None)
    else:model=TaxonomyModel(encoder,taxonomy_tree(labels,minimum_parents))
    for p in model.encoder.parameters():p.requires_grad_(False)
    for layer in model.encoder.layers[-config.train_last_layers:]:
        for p in layer.parameters():p.requires_grad_(True)
    model.to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate,weight_decay=config.weight_decay)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    index=native.index_context(data);best=-float('inf');best_selection=None;bad=0;best_weights=None;history=[]
    try:
        for epoch in range(config.epochs):
            model.train();order=np.random.default_rng(config.seed+epoch).permutation(len(train))
            losses=[]
            for start in range(0,len(order),config.batch_size):
                rows=[train[i] for i in order[start:start+config.batch_size]]
                optimizer.zero_grad(set_to_none=True)
                if task=='rbp':
                    batch,offsets=native.target_batch(data,store,[r[key] for r in rows],device,index)
                    logits=model(batch,offsets).flatten()
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([float(r['label']) for r in rows],device=device))
                    loss.backward();losses.append(float(loss.detach()))
                else:
                    valid=0; loss_total=0.
                    for row in rows:
                        vector=native.genome_vector(model,data,store,row[key],device=device,batch_size=config.batch_size,index=index)
                        loss=model.loss(vector,row)
                        if loss is not None:
                            loss.backward();valid+=1;loss_total+=float(loss.detach())
                    if not valid:continue
                    for p in model.parameters():
                        if p.grad is not None:p.grad.div_(valid)
                    losses.append(loss_total/valid)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                if start==0 or (start//config.batch_size+1)%100==0:
                    progress(f'{task} epoch {epoch+1}: {min(start+config.batch_size,len(order))}/{len(order)} training records; loss {losses[-1]:.6f}')
            if not losses:raise ValueError('No eligible task training supervision')
            model.eval()
            with torch.inference_mode():
                if task=='rbp':
                    probabilities=[]
                    for start in range(0,len(val),config.batch_size):
                        rows=val[start:start+config.batch_size];batch,offsets=native.target_batch(data,store,[r[key] for r in rows],device,index)
                        probabilities.extend(model(batch,offsets).sigmoid().flatten().cpu().tolist())
                    truth=np.array([int(r['label']) for r in val])
                    gate=(np.array([gate_scores[r[key]]>=.5 for r in val]) if gate_scores is not None else np.ones(len(val),bool))
                    metric=float(average_precision_score(truth,probabilities))
                    threshold,f1=select_binary_threshold(truth,probabilities,gate)
                    detail=dict(validation_ap=metric,threshold=threshold)
                    detail['pipeline_f1' if gate_scores is not None else 'validation_f1']=f1
                else:
                    vectors=[native.genome_vector(model,data,store,r[key],device=device,batch_size=config.batch_size,index=index).detach() for r in val]
                    score,thresholds,limits=calibrate_taxonomy(model,vectors,val,hits)
                    metric=score[0];detail=dict(validation_macro_f1=metric,selection_score=list(score),thresholds=thresholds,transfer_thresholds=limits)
            selection=(metric,) if task=='rbp' else tuple(detail['selection_score'])
            improved=best_selection is None or selection>best_selection
            if improved: best_selection=selection
            if improved:
                best=metric;bad=0;best_weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_detail=detail
                torch.save(best_weights,out/'best-weights.pt')
            else:bad+=1
            history.append(dict(epoch=epoch+1,training_loss=float(np.mean(losses)),improved=improved,**detail))
            write_json(out/'history.json',history)
            progress(f'{task} epoch {epoch+1}: validation {metric:.6f}')
            if bad>=config.patience:break
        model.load_state_dict(best_weights);model.eval()
        selected_epoch=max(h['epoch'] for h in history if h['improved'])
        provenance=dict(config=asdict(config),training_manifest_sha256=digest_file(Path(directory)/'training.json'),
            supervision_sha256=digest_file(labels_path),initialization_sha256=digest_file(Path(initialize)/'model.json'),
            selected_epoch=selected_epoch)
        if task=='rbp':
            thresholds=dict(head=best_detail['threshold'])
            provenance['rbp_mode']='standalone' if gate_scores is None else 'infection_gated'
            if gate_scores is not None:
                thresholds['infection']=.5
                provenance['infection_scores_sha256']=digest_file(infection_scores)
        else:
            thresholds=best_detail['thresholds'];limits=best_detail['transfer_thresholds']
            provenance.update(transfer_thresholds=limits,calibration_macro_f1=best,minimum_training_parents=minimum_parents)
            if validation_hits:provenance['validation_hits_sha256']=digest_file(validation_hits)
            train_labels={r['genome_id']:r for r in train}
            reference_rows=[dict(protein_sha256=p.sha256,partition='train',
                **{r:train_labels[p.genome_id].get(r,'') for r in ('order',*RANKS)})
                for p in data.proteins if p.genome_id in train_labels]
            write_table(out/'reference-labels.tsv',reference_rows,['protein_sha256','partition','order',*RANKS])
        native.save_model(model,out/'model',task=task,thresholds=thresholds,provenance=provenance,
            tree=model.tree if task=='taxonomy' else None)
        result=dict(state='complete',task=task,epochs=len(history),model='model',thresholds=thresholds)
        write_json(out/'status.json',result);return result
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)));raise
