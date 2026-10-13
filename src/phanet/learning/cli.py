"""Training, evaluation and model export commands."""
from dataclasses import asdict
from pathlib import Path
from phanet.common.io import read_json, write_json
from phanet.learning.model import ModelConfig
from phanet.learning.training import TrainConfig

COMMANDS = {'train', 'evaluate', 'export-model'}


def add_parsers(sub):
    train = sub.add_parser('train', help='Train or fine-tune a GLM')
    train.add_argument('--data', type=Path, required=True)
    train.add_argument('--out', type=Path, required=True)
    train.add_argument('--mode', choices=('annotation','pretrain'), default='annotation')
    train.add_argument('--model-config', type=Path)
    train.add_argument('--train-config', type=Path)
    initialize = train.add_mutually_exclusive_group()
    initialize.add_argument('--initialize', type=Path, help='Inference bundle to fine-tune')
    initialize.add_argument('--resume', type=Path, help='Training checkpoint to continue in a new output directory')
    train.add_argument('--batch-size', type=int, default=64)
    train.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    evaluate = sub.add_parser('evaluate', help='Evaluate a GLM checkpoint')
    evaluate.add_argument('--data', type=Path, required=True)
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--partition', choices=('val','test'), default='test')
    evaluate.add_argument('--batch-size', type=int, default=64)
    evaluate.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    evaluate.add_argument('--out', type=Path, required=True)
    export = sub.add_parser('export-model', help='Export a GLM inference bundle')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--out', type=Path, required=True)


def run(args, progress=print):
    from phanet.learning.run import fit, evaluate, export_model
    if args.command == 'export-model':
        return export_model(args.checkpoint, args.out)
    if args.command == 'evaluate':
        args.out.mkdir(parents=True, exist_ok=False)
        result = evaluate(args.data, args.checkpoint, partition=args.partition, batch_size=args.batch_size, device=args.device)
        write_json(args.out/'metrics.json', result)
        return result
    model = ModelConfig(**read_json(args.model_config)) if args.model_config else None
    config = None
    if args.train_config:
        defaults = TrainConfig(consistency_weight=0, composition_weight=.1) if args.mode=='pretrain' else TrainConfig(consistency_weight=0, family_weight=0, order_weight=0)
        config = TrainConfig(**(asdict(defaults) | read_json(args.train_config)))
    return fit(args.data, args.out, mode=args.mode, model_config=model, train_config=config,
               initialize=args.initialize, resume=args.resume, batch_size=args.batch_size,
               device=args.device, progress=progress)
