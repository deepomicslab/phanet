"""GLM fitting, checkpoint evaluation and inference-bundle export."""
from dataclasses import asdict, replace
from pathlib import Path
import math
import shutil
import numpy as np
import torch
from torch.utils.data import default_collate
from phanet.common.io import digest_file, digest_object, read_json, write_json
from phanet.common.constants import CATEGORIES
from phanet.learning.data import load_training
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.learning.training import TrainConfig, Trainer, seed_all
from phanet.learning.objectives import family_bank, training_views


def export_model(checkpoint, output):
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**payload['model_config'])
    if cfg.input_dim != 1280 or cfg.max_context != 30:
        raise ValueError('Annotation export requires 1280-dimensional ESM2 and max_context=30')
    model = PhaNetModel(cfg)
    model.load_state_dict(payload['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite checkpoint parameters')
    if payload.get('provenance', {}).get('embedding') != 'esm2_t33_650M_UR50D:layer33:full_length:mean:float16':
        raise ValueError('Checkpoint lacks full-length ESM2 training provenance')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), out/'weights.pt')
    meta = dict(schema_version=1, model_config=asdict(cfg), categories=list(CATEGORIES),
        embedding=payload['provenance']['embedding'], threshold=.5, weights='weights.pt',
        weights_sha256=digest_file(out/'weights.pt'), training_checkpoint_sha256=digest_file(checkpoint))
    write_json(out/'model.json', meta)
    return meta


def _batches(dataset, batch_size, order=None, start=0):
    order = range(len(dataset)) if order is None else order
    for offset in range(start*batch_size, len(order), batch_size):
        yield default_collate([dataset[int(i)] for i in order[offset:offset+batch_size]])


def fit(directory, output, *, model_config=None, train_config=None, mode='annotation',
        batch_size=64, device='cpu', initialize=None, resume=None, max_updates=None, progress=print):
    if mode not in ('annotation', 'pretrain') or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Choose annotation/pretrain and a positive batch size')
    if initialize and resume:
        raise ValueError('Choose initialization or checkpoint continuation')
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        model_config = model_config or ModelConfig(**saved['model_config'])
        train_config = train_config or TrainConfig(**saved['train_config'])
        del saved
    base = load_training(directory)
    seed_all((train_config or TrainConfig()).seed)
    if initialize:
        from phanet.inference import load_model
        model = load_model(initialize)
        if model_config is not None and model.config != model_config:
            raise ValueError('Initialization and model configuration differ')
        model_config = model.config
    else:
        model_config = model_config or ModelConfig(module_objective='composition' if mode == 'pretrain' else 'consistency')
        model = PhaNetModel(model_config)
    if model_config.input_dim != base.embeddings.dimension or model_config.max_context != base.max_context:
        raise ValueError('Model and prepared data dimensions differ')
    config = train_config or (TrainConfig(consistency_weight=0, composition_weight=.1, window_sampling='bounded_v1')
                              if mode == 'pretrain' else TrainConfig(consistency_weight=0, family_weight=0, order_weight=0))
    if mode == 'annotation' and any((config.consistency_weight, config.composition_weight, config.family_weight, config.order_weight)):
        raise ValueError('Annotation mode requires zero auxiliary loss weights')
    if config.generic_pretraining or model_config.centered_context or config.window_sampling not in ('exhaustive','bounded_v1'):
        raise ValueError('This runner uses annotation targets and segment/sampled-window contexts')
    if bool(config.composition_weight) and model_config.module_objective != 'composition':
        raise ValueError('Composition training requires a composition encoder')
    names, bank, targets = family_bank(base.dataset, base.embeddings, base.splits) if config.family_weight else ([], None, None)
    training = training_views(base, 'train', families=targets, composition=bool(config.composition_weight))
    validation = training_views(base, 'val', families=targets, composition=bool(config.composition_weight))
    if mode == 'annotation':
        training, validation = training.datasets[0], validation.datasets[0]
        for part, view in [('train', training), ('val', validation)]:
            if not any(p.labels for p in base.dataset.proteins if base.splits.assignments[p.genome_id] == part):
                raise ValueError(f'{part} has no functional supervision')
    steps = math.ceil(math.ceil(len(training)/batch_size)/config.gradient_accumulation)*config.max_epochs
    config = replace(config, total_steps=config.total_steps or steps, window_sampling='bounded_v1')
    source_root = Path(__file__).parents[1]
    source_hash = digest_object({str(p.relative_to(source_root)): digest_file(p) for p in source_root.rglob('*.py')})
    identity = digest_object(dict(training_manifest=digest_file(Path(directory)/'training.json'),
        model=asdict(model_config), config=asdict(config), batch_size=batch_size, mode=mode, source=source_hash,
        torch=torch.__version__, device=device))
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    write_json(out/'run.json', dict(data_hash=identity, mode=mode, model=asdict(model_config), config=asdict(config),
        training_directory=str(Path(directory).resolve()), batch_size=batch_size, device=device))
    if resume:
        trainer = Trainer.restore(resume, data_hash=identity, device=device)
        state = trainer.sampler_state
        old_best = Path(resume).parent/'best.pt'
        if trainer.best_validation < float('inf'):
            if not old_best.is_file():
                raise ValueError('Resume requires the validation-best checkpoint beside the source checkpoint')
            old = torch.load(old_best, map_location='cpu', weights_only=False)
            if old.get('data_hash') != identity:
                raise ValueError('Best checkpoint belongs to different training data')
            shutil.copyfile(old_best, out/'best.pt')
    else:
        trainer = Trainer(model, config, device=device, data_hash=identity, family_bank=bank)
        state = dict(epoch=0, next_batch=0)
    del model
    trainer.provenance = dict(embedding='esm2_t33_650M_UR50D:layer33:full_length:mean:float16',
        training_manifest_sha256=digest_file(Path(directory)/'training.json'), family_ids=names)
    if names:
        write_json(out/'families.json', names)
    initial_step = trainer.step
    history = []
    try:
        for epoch in range(state['epoch'], config.max_epochs):
            pending_validation = epoch == state['epoch'] and state['next_batch'] > 0
            if ((trainer.step >= config.total_steps and not pending_validation)
                    or trainer.bad_epochs >= config.patience):
                break
            for child in getattr(training, 'datasets', [training]):
                child.set_epoch(epoch)
            order = torch.randperm(len(training), generator=torch.Generator().manual_seed(config.seed+epoch)).tolist()
            start = state['next_batch'] if epoch == state['epoch'] else 0
            for b, batch in enumerate(_batches(training, batch_size, order, start), start):
                if trainer.step >= config.total_steps:
                    break
                report = trainer.update(batch)
                cursor = dict(epoch=epoch, next_batch=b+1)
                if report['optimizer_updated'] and trainer.step % config.checkpoint_interval == 0:
                    trainer.checkpoint(out/'last.pt', sampler_state=cursor)
                if max_updates is not None and trainer.step-initial_step >= max_updates and not trainer.pending_micro_steps:
                    trainer.checkpoint(out/'last.pt', sampler_state=cursor)
                    write_json(out/'status.json', dict(state='paused', step=trainer.step))
                    return dict(state='paused', step=trainer.step)
            trainer.finish_epoch()
            value = trainer.validate(_batches(validation, batch_size))
            improved = value < trainer.best_validation
            trainer.best_validation = min(value, trainer.best_validation)
            trainer.bad_epochs = 0 if improved else trainer.bad_epochs+1
            state = dict(epoch=epoch+1, next_batch=0)
            trainer.checkpoint(out/'last.pt', sampler_state=state)
            if improved:
                trainer.checkpoint(out/'best.pt', sampler_state=state)
            history.append(dict(epoch=epoch+1, step=trainer.step, validation=trainer.last_validation_report, improved=improved))
            write_json(out/'history.json', history)
            progress(f'Epoch {epoch+1}: validation loss {value:.6f}, step {trainer.step}')
        if not (out/'best.pt').is_file():
            raise ValueError('No validated checkpoint; schedule is exhausted or validation is empty')
        export_model(out/'best.pt', out/'model')
        result = dict(state='complete', step=trainer.step, epochs=history, model='model')
        write_json(out/'status.json', result)
        return result
    except BaseException as exc:
        write_json(out/'status.json', dict(state='failed', error=str(exc)))
        raise


def evaluate(directory, checkpoint, *, partition='test', batch_size=64, device='cpu'):
    if partition not in ('val', 'test') or batch_size < 1:
        raise ValueError('Evaluation requires val/test and positive batch size')
    base = load_training(directory, partition)
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload.get('provenance', {}).get('training_manifest_sha256') != digest_file(Path(directory)/'training.json'):
        raise ValueError('Checkpoint and training manifest differ')
    trainer = Trainer.restore(checkpoint, data_hash=payload['data_hash'], device=device)
    names = trainer.provenance.get('family_ids', [])
    index = {f:i for i,f in enumerate(names)}
    targets = np.array([min((index[f] for f in p.family_ids if f in index), default=-1) for p in base.dataset.proteins]) if names else None
    views = training_views(base, partition, families=targets, composition=bool(trainer.config.composition_weight))
    if not any((trainer.config.family_weight, trainer.config.composition_weight, trainer.config.order_weight, trainer.config.consistency_weight)):
        views = views.datasets[0]
    scores, truths = {}, {}
    def observe(batch, report, diagnostic):
        probabilities = diagnostic['native']['logits'].sigmoid().cpu().numpy()
        eligible = batch['loss_eligible'].cpu().numpy()
        ids = batch['protein_indices'].cpu().numpy()
        targets = batch['targets'].cpu().numpy()
        for row, col in zip(*np.where(eligible)):
            i = int(ids[row,col]); scores.setdefault(i, []).append(probabilities[row,col]); truths[i] = targets[row,col]
    trainer.validate(_batches(views, batch_size), observer=observe)
    result = dict(trainer.last_validation_report, partition=partition, annotated_proteins=len(scores))
    if scores:
        from sklearn.metrics import f1_score, average_precision_score
        y = np.stack([truths[i] for i in sorted(scores)])
        p = np.stack([np.mean(scores[i], axis=0) for i in sorted(scores)])
        result.update(micro_f1=float(f1_score(y,p>=.5,average='micro',zero_division=0)),
            micro_ap=float(average_precision_score(y.ravel(),p.ravel())))
    return result
