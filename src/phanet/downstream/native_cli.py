"""Commands for native RBP and taxonomy prediction and fine-tuning."""
from pathlib import Path
import os
from phanet.common.io import read_json

COMMANDS={'predict-rbp','predict-taxonomy','train-rbp','train-taxonomy','score-infection','taxonomy-hits'}


def add_parsers(sub):
    score=sub.add_parser('score-infection',help='Generate combined Infection scores for native proteins')
    inputs=score.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--dataset',type=Path)
    inputs.add_argument('--input',type=Path)
    inputs.add_argument('--genbank',nargs='+',type=Path)
    score.add_argument('--config',type=Path)
    score.add_argument('--annotation-model',type=Path,required=True)
    score.add_argument('--network-config',type=Path,required=True)
    embeddings=score.add_mutually_exclusive_group()
    embeddings.add_argument('--embeddings',type=Path)
    embeddings.add_argument('--esm-model',type=Path)
    score.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    score.add_argument('--batch-size',type=int,default=1)
    score.add_argument('--threads',type=int,default=4)
    score.add_argument('--out',type=Path,required=True)
    hits=sub.add_parser('taxonomy-hits',help='Search training taxonomy references for native query proteins')
    hits.add_argument('--dataset',type=Path,required=True)
    hits.add_argument('--labels',type=Path,required=True,help='Genome taxonomy supervision TSV')
    hits.add_argument('--reference-labels',type=Path,required=True)
    hits.add_argument('--network-config',type=Path,required=True)
    hits.add_argument('--minimum-parents',type=int,default=5)
    hits.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    hits.add_argument('--threads',type=int,default=4)
    hits.add_argument('--out',type=Path,required=True)
    for task in ('rbp','taxonomy'):

        parser=sub.add_parser('predict-'+task,help='Predict '+task+' from native genomic inputs')
        inputs=parser.add_mutually_exclusive_group(required=True)
        inputs.add_argument('--dataset',type=Path)
        inputs.add_argument('--input',type=Path,help='Nucleotide FASTA')
        inputs.add_argument('--genbank',nargs='+',type=Path)
        parser.add_argument('--config',type=Path)
        parser.add_argument('--model',type=Path,required=True)
        embeddings=parser.add_mutually_exclusive_group()
        embeddings.add_argument('--embeddings',type=Path)
        embeddings.add_argument('--esm-model',type=Path)
        parser.add_argument('--network-config',type=Path)
        parser.add_argument('--batch-size',type=int,default=1)
        parser.add_argument('--threads',type=int,default=4)
        parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        parser.add_argument('--out',type=Path,required=True)
        if task=='rbp':
            parser.add_argument('--infection-scores',type=Path)
            parser.add_argument('--annotation-model',type=Path)
        else:
            parser.add_argument('--reference-labels',type=Path)
            parser.add_argument('--method',choices=('combined','glm'),default='combined')
        train=sub.add_parser('train-'+task,help='Fine-tune native '+task+' prediction')
        train.add_argument('--data',type=Path,required=True)
        train.add_argument('--labels',type=Path,required=True)
        train.add_argument('--initialize',type=Path,required=True)
        train.add_argument('--train-config',type=Path)
        train.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        train.add_argument('--out',type=Path,required=True)
        if task=='rbp':train.add_argument('--infection-scores',type=Path,
            help='Optional combined Infection scores for two-stage validation; omit for standalone RBP training')
        else:
            train.add_argument('--validation-hits',type=Path)
            train.add_argument('--minimum-parents',type=int,default=5)


def run(args,progress=print):
    if args.command=='taxonomy-hits':
        from .native_predict import taxonomy_hits
        from .taxonomy_native import taxonomy_tree
        from phanet.datasets.schema import Dataset
        from phanet.network.inference import NetworkConfig
        from phanet.common.io import read_table
        labels=list(read_table(args.labels,('genome_id','partition')))
        return dict(hits=len(taxonomy_hits(Dataset.load(args.dataset),NetworkConfig.load(args.network_config),
            args.reference_labels,args.out,tree=taxonomy_tree(labels,args.minimum_parents),
            device=args.device,threads=args.threads,progress=progress)))
    if args.command=='score-infection':
        from .native_predict import score_infection
        from phanet.network.inference import NetworkConfig
        return score_infection(args.out,args.annotation_model,NetworkConfig.load(args.network_config),
            dataset=args.dataset,input_fasta=args.input,genbank=args.genbank,
            config=args.config or os.environ.get('PHANET_CONFIG'),embeddings=args.embeddings,esm_checkpoint=args.esm_model,
            device=args.device,batch_size=args.batch_size,threads=args.threads,progress=progress)
    task=args.command.split('-')[1]
    if args.command.startswith('train-'):
        from .native_train import fit,FineTuneConfig
        config=FineTuneConfig(**read_json(args.train_config)) if args.train_config else None
        kw=dict(infection_scores=args.infection_scores) if task=='rbp' else dict(validation_hits=args.validation_hits,minimum_parents=args.minimum_parents)
        return fit(task,args.data,args.labels,args.initialize,args.out,config=config,device=args.device,progress=progress,**kw)
    from .native_predict import predict
    from phanet.network.inference import NetworkConfig
    config=NetworkConfig.load(args.network_config) if args.network_config else None
    kw=dict(infection_scores=args.infection_scores,annotation_model=args.annotation_model) if task=='rbp' else dict(reference_labels=args.reference_labels,method=args.method)
    return predict(task,args.out,args.model,network_config=config,batch_size=args.batch_size,device=args.device,
        threads=args.threads,dataset=args.dataset,input_fasta=args.input,genbank=args.genbank,
        config=args.config or os.environ.get('PHANET_CONFIG'),embeddings=args.embeddings,esm_checkpoint=args.esm_model,**kw)
