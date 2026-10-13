"""Native-context encoding and versioned RBP/taxonomy model bundles."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
from phanet.common.io import digest_file, read_json, write_json, read_table
from phanet.common.constants import CATEGORIES
from phanet.datasets.importing import load_input
from phanet.inference import context_batch
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.representations.embed import EmbeddingStore, ESMConfig, embed_esm, validate_full_length
from phanet.downstream.bundles import GOModel, EMBEDDING
from phanet.downstream.taxonomy_native import TaxonomyModel, RANKS


def inputs(output, *, embeddings=None, esm_checkpoint=None, device='cpu', batch_size=1, **sources):
    data = load_input(work=output, allow_empty=True, **sources)
    if embeddings is not None and esm_checkpoint is not None: raise ValueError('Choose embeddings or ESM checkpoint')
    if not data.proteins: return data, None
    if embeddings is None:
        if not esm_checkpoint: raise ValueError('Supply embeddings or an ESM checkpoint')
        store = embed_esm([p.protein_id for p in data.proteins], [p.sequence for p in data.proteins],
            Path(output)/'embeddings', ESMConfig(checkpoint=str(esm_checkpoint),full_length=True,device=device,batch_size=batch_size))
    else: store = EmbeddingStore(embeddings)
    validate_full_length(data,store)
    return data,store


def index_context(data):
    groups = data.by_genome()
    return groups,{p.protein_id:(gid,i) for gid,ps in groups.items() for i,p in enumerate(ps)}


def target_batch(data, store, ids, device, index=None):
    batch, offsets = context_batch(data,ids,store,hidden_hashes=set(),context_index=index)
    return [x.to(device) for x in batch], offsets


def genome_vector(model, data, store, gid, *, batch_size=64, device='cpu', index=None):
    index = index or index_context(data)
    ids = [p.protein_id for p in index[0][gid]]
    if not ids: return None
    total = None
    for start in range(0,len(ids),batch_size):
        batch, offsets = target_batch(data,store,ids[start:start+batch_size],device,index)
        states = model.encoder(*batch,member_mask=None)['states']
        rows = torch.arange(len(offsets),device=device)
        value = states[rows,torch.tensor(offsets,device=device)].sum(0)
        total = value if total is None else total+value
    return total/len(ids)


def save_model(model, output, *, task, thresholds, provenance, tree=None):
    out = Path(output); out.mkdir(parents=True,exist_ok=False)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},out/'weights.pt')
    meta = dict(schema_version=1,task=task,model_config=asdict(model.encoder.config),categories=list(CATEGORIES),
        embedding=EMBEDDING,context='shifted_native_target_masked',weights='weights.pt',
        weights_sha256=digest_file(out/'weights.pt'),thresholds=thresholds,provenance=provenance)
    if tree is not None: meta['tree']=tree
    write_json(out/'model.json',meta)
    return meta


def load_model(directory, task, device='cpu'):
    root=Path(directory); meta=read_json(root/'model.json')
    if (meta.get('schema_version')!=1 or meta.get('task')!=task or meta.get('categories')!=list(CATEGORIES)
            or meta.get('embedding')!=EMBEDDING or meta.get('context')!='shifted_native_target_masked'
            or meta.get('weights')!='weights.pt' or digest_file(root/'weights.pt')!=meta.get('weights_sha256')):
        raise ValueError('Incompatible or damaged native task bundle')
    config=ModelConfig(**meta['model_config']); config.validate()
    if config.input_dim!=1280 or config.max_context!=30: raise ValueError('Native tasks require ESM2 and 30 positions')
    values=meta['thresholds']
    if task=='rbp':
        if not (set(values)=={'head'} and meta.get('provenance',{}).get('rbp_mode')=='standalone'
                or set(values)=={'head','infection'} and values['infection']==.5):
            raise ValueError('Invalid RBP thresholds')
    elif task=='taxonomy':
        if set(values)!=set(RANKS): raise ValueError('Invalid taxonomy thresholds')
    else: raise ValueError('Unknown native task')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1.000001 for v in values.values()):
        raise ValueError('Invalid model decision threshold')
    encoder=PhaNetModel(config)
    model=GOModel(encoder,1) if task=='rbp' else TaxonomyModel(encoder,meta['tree'])
    model.load_state_dict(torch.load(root/'weights.pt',map_location='cpu',weights_only=True),strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError('Nonfinite task model')
    return model.to(device).eval(),meta


def infection_table(path, data, *, required=None):
    known={p.protein_id:p for p in data.proteins}; scores={}
    for row in read_table(path,('protein_id','protein_sha256','infection_score')):
        pid=row['protein_id']
        value=float(row['infection_score'])
        if pid not in known or pid in scores or row['protein_sha256']!=known[pid].sha256 or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid infection-score identity or value')
        scores[pid]=value
    if not (set(known) if required is None else set(required)) <= set(scores):
        raise ValueError('Infection scores must cover the requested native proteins')
    return scores


def infection_scores(data,store,output,annotation_model,network_config,*,device='cpu',batch_size=1,threads=4,progress=print):
    from phanet.network.inference import preflight, network_probabilities
    from phanet.network.annotation import fuse
    from phanet.inference import load_model as load_glm, glm_probabilities, _release_stage_memory
    ids=[p.protein_id for p in data.proteins]
    if not ids: return {}
    bundle,_=preflight(network_config,progress=progress)
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    scores,rho,_=network_probabilities(data,ids,bundle,network_config,out,device=device,threads=threads,progress=progress)
    del bundle;_release_stage_memory(device)
    model=load_glm(annotation_model,device)
    glm=glm_probabilities(data,ids,store,model,batch_size=batch_size,device=device,progress=progress)
    values=fuse(glm,scores,rho)[:,CATEGORIES.index('Infection')]
    del model;_release_stage_memory(device)
    return dict(zip(ids,map(float,values)))
