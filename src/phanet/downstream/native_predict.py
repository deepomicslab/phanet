"""RBP scoring and genome taxonomy from native records or nucleotide FASTA."""
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from phanet.common.io import digest_file, read_json, read_table, write_json, write_table
from phanet.downstream import native
from phanet.downstream.taxonomy_native import transfer, RANKS, _label, _merge_ancestry
from phanet.downstream.taxonomy import TaxonomyThresholds


def reference_taxonomy(row, tree):
    labels={r:_label(row,r) for r in ('order',*RANKS)}
    for rank in RANKS:
        if labels[rank] in tree[rank]:
            _merge_ancestry(dict(labels),tree[rank][labels[rank]])
    return {r:value if r=='order' or value in tree[r] else '' for r,value in labels.items()}


def taxonomy_hits(data, network_config, reference_labels, output, *, tree, device='cpu', threads=4, progress=print):
    from phanet.network.inference import preflight, mean_edges
    from phanet.network.search import sequence_search, profile_search, profile_edges, structure_edges
    from phanet.representations.tmvec import full_length_vectors
    bundle,_=preflight(network_config,progress=progress)
    references={}
    for row in read_table(reference_labels,('protein_sha256','partition')):
        h=row['protein_sha256']
        if row['partition']!='train': raise ValueError('Taxonomy references must be training records')
        labels=reference_taxonomy(row,tree)
        if not any(labels[r] for r in RANKS): continue
        if h in references:
            _merge_ancestry(labels,references[h])
            labels=reference_taxonomy(labels,tree)
        references[h]=labels
    if not references: raise ValueError('No taxonomy references in the retained class set')
    if (bundle.positions(list(references))<0).any(): raise ValueError('Taxonomy reference absent from network bundle')
    sequences={p.sha256:p.sequence for p in data.proteins}
    all_refs={bytes(h).decode() for h in bundle.ids}
    hidden=(all_refs-set(references))|set(sequences)
    groups={gid:sorted({p.sha256 for p in ps}) for gid,ps in data.by_genome().items() if ps}
    local={gid:{} for gid in groups}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    seq=sequence_search(bundle,groups,sequences,local,out/'sequence',mmseqs=network_config.mmseqs,threads=threads,hidden=hidden)
    hits=profile_search(sequences,network_config.profiles,out/'profile',threads=threads,
        profile_count=bundle.meta['profile_count'],profile_Z=bundle.meta['profile_Z'],progress=progress)
    allowed=np.array([bytes(h).decode() not in hidden for h in bundle.ids])
    ids=sorted(sequences); pos=bundle.positions(ids)
    vectors={h:np.asarray(bundle.vectors[j]) for h,j in zip(ids,pos) if j>=0}
    missing=[h for h,j in zip(ids,pos) if j<0]
    if missing:
        vectors.update(zip(missing,full_length_vectors([sequences[h] for h in missing],network_config.tmvec_model,network_config.prott5,device=device,progress=progress)))
    prof=profile_edges(bundle,hits,[],allowed,set())
    struct=structure_edges(bundle,vectors,{},allowed)
    rows=[]
    for p in data.proteins:
        integrated=mean_edges(seq[p.genome_id][p.sha256]['edges'],prof.get(p.sha256,{}),struct[p.sha256])
        for h,value in integrated.items():
            if h not in references or h in hidden: raise ValueError('Taxonomy retrieved an excluded reference')
            rows.append(dict(genome_id=p.genome_id,query_protein_id=p.protein_id,reference_id=h,weight=value,**references[h]))
    write_table(out/'hits.tsv',rows,['genome_id','query_protein_id','reference_id','weight','order','family','subfamily','genus'])
    return rows


def predict(task, output, model_dir, *, infection_scores=None, annotation_model=None, network_config=None,
            reference_labels=None, method='combined', batch_size=64, device='cpu', threads=4, **sources):
    if task not in ('rbp','taxonomy') or batch_size<1: raise ValueError('Invalid task or batch size')
    model,meta=native.load_model(model_dir,task,device='cpu')
    gated=task=='rbp' and 'infection' in meta['thresholds']
    if gated and not infection_scores and (not annotation_model or network_config is None):
        raise ValueError('RBP requires infection scores or the combined annotation model and network')
    if task=='rbp' and not gated and (infection_scores is not None or annotation_model is not None or network_config is not None):
        raise ValueError('Standalone RBP models do not use Infection inputs')
    if task=='taxonomy' and (method not in ('glm','combined') or method=='combined' and (not reference_labels or network_config is None)):
        raise ValueError('Combined taxonomy requires network configuration and training reference labels')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    try:
        data,store=native.inputs(out,batch_size=batch_size,device=device,**sources)
        data.save(out/'dataset')
        identity=dict(task=task,model_sha256=digest_file(Path(model_dir)/'model.json'),
            dataset={p.name:digest_file(p) for p in (out/'dataset').iterdir()},
            embedding_sha256=digest_file(store.directory/'embeddings.json') if store is not None else None)
        if task=='rbp':
            scores=((native.infection_table(infection_scores,data) if infection_scores else
                native.infection_scores(data,store,out/'infection',annotation_model,network_config,device=device,batch_size=batch_size,threads=threads))
                if gated else {})
            identity['rbp_mode']='infection_gated' if gated else 'standalone'
            if gated:
                identity['infection_input']=digest_file(infection_scores) if infection_scores else dict(annotation_model=digest_file(Path(annotation_model)/'model.json'),network=asdict(network_config))
            selected=[p.protein_id for p in data.proteins if not gated or scores[p.protein_id]>=meta['thresholds']['infection']]
            values={};model.to(device); index=native.index_context(data)
            with torch.inference_mode():
                for first in range(0,len(selected),batch_size):
                    ids=selected[first:first+batch_size];batch,offsets=native.target_batch(data,store,ids,device,index)
                    probs=model(batch,offsets).sigmoid().flatten().cpu().tolist()
                    values.update(zip(ids,probs))
            rows=[dict(protein_id=p.protein_id,genome_id=p.genome_id,protein_sha256=p.sha256,
                infection_score=scores.get(p.protein_id,''),head_score=values.get(p.protein_id,''),
                pipeline_score=values.get(p.protein_id,0.),rbp= int(p.protein_id in values and values[p.protein_id]>=meta['thresholds']['head'])) for p in data.proteins]
            write_table(out/'rbp.tsv',rows,['protein_id','genome_id','protein_sha256','infection_score','head_score','pipeline_score','rbp'])
            summary=dict(task=task,proteins=len(rows),rbp_mode=identity['rbp_mode'],
                         infection_assigned=len(selected) if gated else None,candidates=sum(r['rbp'] for r in rows))
        else:
            hits=taxonomy_hits(data,network_config,reference_labels,out/'network',tree=meta['tree'],device=device,threads=threads) if method=='combined' and data.proteins else []
            if reference_labels: identity['reference_sha256']=digest_file(reference_labels)
            limits=TaxonomyThresholds(**meta.get('provenance',{}).get('transfer_thresholds',{}))
            results=transfer([g.genome_id for g in data.genomes],hits,limits)
            model.to(device);index=native.index_context(data)
            with torch.inference_mode():
                for row in results:
                    if any(v['label'] for v in row['ranks'].values()): continue
                    vector=native.genome_vector(model,data,store,row['genome_id'],batch_size=batch_size,device=device,index=index)
                    if vector is None: continue
                    row['ranks']=model.classify(vector,meta['thresholds'])
                    row['method']='context_glm' if any(v['label'] for v in row['ranks'].values()) else 'ambiguous'
            write_json(out/'predictions.json',results)
            rows=[dict(genome_id=row['genome_id'],rank=rank,**{k:value[k] for k in ('label','confidence','method')}) for row in results for rank,value in row['ranks'].items()]
            write_table(out/'predictions.tsv',rows,['genome_id','rank','label','confidence','method'])
            summary=dict(task=task,genomes=len(results),assigned=sum(any(v['label'] for v in r['ranks'].values()) for r in results))
        write_json(out/'manifest.json',identity);write_json(out/'summary.json',summary)
        write_json(out/'status.json',dict(state='complete',**summary))
        return summary
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)));raise


def score_infection(output, annotation_model, network_config, *, device='cpu', batch_size=1, threads=4, progress=print, **sources):
    if batch_size<1 or threads<1: raise ValueError('Batch size and threads must be positive')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    try:
        data,store=native.inputs(out,device=device,batch_size=batch_size,**sources)
        data.save(out/'dataset')
        scores=native.infection_scores(data,store,out/'network',annotation_model,network_config,
            device=device,batch_size=batch_size,threads=threads,progress=progress)
        write_table(out/'infection.tsv',
            [dict(protein_id=p.protein_id,protein_sha256=p.sha256,infection_score=scores[p.protein_id]) for p in data.proteins],
            ['protein_id','protein_sha256','infection_score'])
        write_json(out/'manifest.json',dict(annotation_model_sha256=digest_file(Path(annotation_model)/'model.json'),
            network=asdict(network_config),dataset={p.name:digest_file(p) for p in (out/'dataset').iterdir()},
            embedding_sha256=digest_file(store.directory/'embeddings.json') if store is not None else None))
        result=dict(state='complete',proteins=len(scores));write_json(out/'status.json',result);return result
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',error=str(exc)));raise
