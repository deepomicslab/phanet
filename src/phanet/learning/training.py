"""Joint objectives, bounded CPU training and exact stochastic continuation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist

from phanet.learning.model import ModelConfig, PhaNetModel, forward_batch
from phanet.learning.composition import composition_loss
from phanet.datasets.splits import MASKED, UNRESOLVED
from phanet.common.io import digest_object, write_json


@dataclass(frozen=True)
class TrainConfig:
    learning_rate: float = 1e-4
    weight_decay: float = .01
    warmup_steps: int = 2000
    total_steps: int | None = None
    mask_rate: float = .15
    consistency_weight: float = .10
    composition_weight: float = 0.
    family_weight: float = .05
    order_weight: float = .05
    family_temperature: float = .1
    negative_families: int = 32
    gradient_accumulation: int = 1
    gradient_clip: float = 1.
    patience: int = 10
    max_epochs: int = 100
    amp: bool = False
    seed: int = 7
    generic_pretraining: bool = False
    checkpoint_interval: int = 100
    window_sampling: str = "exhaustive"
    annotation_pos_weight: tuple[float, ...] | None = None

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (
                self.learning_rate, self.weight_decay, self.gradient_clip,
                self.family_temperature)):
            raise ValueError("Training parameters must be finite")
        if self.annotation_pos_weight is not None:
            weights = tuple(float(v) for v in self.annotation_pos_weight)
            if not weights or any(not math.isfinite(v) or v <= 0 for v in weights):
                raise ValueError("Annotation positive weights must be finite and positive")
            object.__setattr__(self, "annotation_pos_weight", weights)
        if self.window_sampling not in ("exhaustive", "bounded_v1"):
            raise ValueError("Unknown window sampling policy")
        if ((self.total_steps is not None and self.total_steps < 1) or self.gradient_accumulation < 1 or self.max_epochs < 1 or
                self.patience < 1 or self.checkpoint_interval < 1 or self.warmup_steps < 0 or
                self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0 or
                self.family_temperature <= 0 or self.negative_families < 1 or
                not 0 <= self.mask_rate <= 1 or min(self.consistency_weight, self.composition_weight, self.family_weight, self.order_weight) < 0):
            raise ValueError("Invalid training configuration")
        if (not all(math.isfinite(v) for v in (self.consistency_weight, self.composition_weight,
                                             self.family_weight, self.order_weight))
                or self.consistency_weight and self.composition_weight):
            raise ValueError("Composition replaces consistency; objective weights must be finite")


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def copy_batch(batch):
    return {k: v.clone() for k, v in batch.items()}


def sample_annotation_mask(eligible, rate=.15):
    if not 0 <= rate <= 1:
        raise ValueError("Mask probability must be in [0, 1]")
    eligible = eligible.bool()
    selected = (torch.rand(eligible.shape, device=eligible.device) < rate) & eligible
    # Mask at least one annotated token in each nonempty segment.
    if rate:
        for row in range(eligible.shape[0]):
            if eligible[row].any() and not selected[row].any():
                candidates = eligible[row].nonzero(as_tuple=False).flatten()
                selected[row, candidates[torch.randint(len(candidates), ())]] = True
    return selected


def reverse_complement(batch):
    result = copy_batch(batch)
    for row in range(len(batch["valid"])):
        positions = batch["valid"][row].nonzero().flatten()
        reversed_positions = positions.flip(0)
        for key, value in batch.items():
            if value.ndim >= 2 and value.shape[:2] == batch["valid"].shape:
                result[key][row, positions] = value[row, reversed_positions]
        result["strand"][row, positions] *= -1
    return result


def shuffle_members(batch):
    result = copy_batch(batch)
    eligible = torch.zeros(len(batch["valid"]), dtype=torch.bool, device=batch["valid"].device)
    for row in range(len(batch["valid"])):
        positions = (batch["member_mask"][row] & batch["valid"][row]).nonzero().flatten()
        if len(positions) < 2:
            continue
        permutation = torch.randperm(len(positions), device=positions.device)
        if torch.equal(permutation, torch.arange(len(positions), device=positions.device)):
            permutation = permutation.roll(1)
        shuffled = positions[permutation]
        for key, value in batch.items():
            if value.ndim >= 2 and value.shape[:2] == batch["valid"].shape:
                result[key][row, positions] = value[row, shuffled]
        # Identical token contents cannot supply a valid order discrimination target.
        eligible[row] = not (torch.equal(result["embeddings"][row], batch["embeddings"][row]) and
                             torch.equal(result["strand"][row], batch["strand"][row]))
    return result, eligible


def joint_loss(model: PhaNetModel, batch, config: TrainConfig, *, family_bank=None,
               annotation_mask=None, validation=False, diagnostics=None):
    if config.family_weight and family_bank is None:
        raise ValueError("Family supervision requires a training family bank")
    if diagnostics is not None and not validation:
        raise ValueError("Detailed diagnostics are only available during validation")
    if ((config.composition_weight and model.config.module_objective != "composition")
            or (config.consistency_weight and model.config.module_objective != "consistency")
            or (model.config.protein_only and config.composition_weight)):
        raise ValueError("Module head and configured objective disagree")
    # Keep composition-pair arrays separate from single-window augmentation.
    visible = copy_batch({k: v for k, v in batch.items() if not k.startswith("composition_")})
    eligible = batch["loss_eligible"].bool() & batch["valid"].bool()
    masked = sample_annotation_mask(eligible, config.mask_rate) if annotation_mask is None else annotation_mask & eligible
    if config.generic_pretraining:
        visible["annotation_input"].zero_()
        visible["state"][visible["valid"]] = UNRESOLVED
        masked = torch.zeros_like(masked)
    else:
        visible["state"][masked] = MASKED
        visible["annotation_input"][masked] = 0
    result = forward_batch(model, visible)
    if diagnostics is not None:
        diagnostics.update(native={k: v.detach() for k, v in result.items()},
                           annotation_mask=masked.detach())
    zero = result["logits"].sum() * 0
    pos_weight = None
    if config.annotation_pos_weight is not None:
        if len(config.annotation_pos_weight) != result["logits"].shape[-1]:
            raise ValueError("Annotation positive weights do not match output categories")
        pos_weight = torch.as_tensor(config.annotation_pos_weight, device=result["logits"].device,
                                     dtype=torch.float32)
    annotation = F.binary_cross_entropy_with_logits(
        result["logits"][masked], batch["targets"][masked], pos_weight=pos_weight
    ) if masked.any() else zero

    consistency = zero
    if model.config.module_objective == "consistency":
        # Preserve RNG order across checkpoint resume.
        positive = reverse_complement(visible)
        if not validation:
            positive["embeddings"] = F.dropout(positive["embeddings"], p=.05, training=True)
        available = positive["valid"] & (positive["state"] != UNRESOLVED)
        positive["state"][available] = MASKED
        positive["annotation_input"][available] = 0
        positive_result = forward_batch(model, positive)
        consistency = (1 - (result["projection"] * positive_result["projection"]).sum(dim=-1)).mean()

    negative, order_eligible = shuffle_members(visible)
    negative_result = forward_batch(model, negative)
    if diagnostics is not None:
        diagnostics.update(order_eligible=order_eligible.detach(),
                           negative_order=negative_result["order"].detach())
    if order_eligible.any():
        order = .5 * (F.binary_cross_entropy_with_logits(result["order"][order_eligible], torch.ones_like(result["order"][order_eligible])) +
                      F.binary_cross_entropy_with_logits(negative_result["order"][order_eligible], torch.zeros_like(negative_result["order"][order_eligible])))
    else:
        order = zero

    family = zero
    family_examples = 0
    if family_bank is not None:
        if "family_index" not in batch or family_bank.ndim != 2 or family_bank.shape[1] != model.config.input_dim:
            raise ValueError("Family prediction requires compatible training-frozen family targets")
        corrupted = copy_batch(visible)
        rows, targets = [], []
        for row in range(len(batch["valid"])):
            positions = ((batch["family_index"][row] >= 0) & batch["member_mask"][row] & batch["valid"][row]).nonzero().flatten()
            if not len(positions):
                continue
            position = positions[torch.randint(len(positions), ())]
            rows.append(row); targets.append(int(batch["family_index"][row, position]))
            corrupted["embeddings"][row, position] = model.protein_mask
            corrupted["state"][row, position] = MASKED
            corrupted["annotation_input"][row, position] = 0
        if rows and len(family_bank) > 1:
            predictions = forward_batch(model, corrupted)["family"][rows]
            target_tensor = torch.tensor(targets, device=predictions.device)
            positive_ids = set(targets)
            all_negative = [i for i in range(len(family_bank)) if i not in positive_ids]
            order_ids = torch.randperm(len(all_negative))[:config.negative_families].tolist()
            candidate_ids = sorted(positive_ids) + [all_negative[i] for i in order_ids]
            candidates = torch.tensor(candidate_ids, device=predictions.device)
            logits = predictions @ F.normalize(family_bank[candidates].detach(), dim=-1).T / config.family_temperature
            target_positions = torch.tensor([candidate_ids.index(t) for t in targets], device=predictions.device)
            family = F.cross_entropy(logits, target_positions)
            family_examples = len(rows)
            if diagnostics is not None:
                diagnostics["family"] = dict(logits=logits.detach(), targets=target_positions.detach(),
                                             predictions=predictions.detach(), rows=rows,
                                             candidate_ids=candidate_ids)
    composition, composition_report = zero, {}
    if config.composition_weight:
        composition, composition_report = composition_loss(model, batch, zero)
    total = (annotation + config.consistency_weight * consistency + config.family_weight * family
             + config.order_weight * order + config.composition_weight * composition)
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite joint loss")
    components = {"annotation": annotation, "consistency": consistency, "family": family,
                  "order": order, "total": total}
    report = {k: float(v.detach()) for k, v in components.items()}
    report.update(masked_annotations=int(masked.sum()), family_examples=family_examples,
                  order_examples=int(order_eligible.sum()))
    if model.config.module_objective == "composition":
        report.update(composition=float(composition.detach()), **composition_report)
    return total, report


class JointObjective(nn.Module):
    """One DDP forward encloses all augmented views and the complete objective."""
    def __init__(self, model, config, family_bank):
        super().__init__()
        self.encoder, self.config = model, config
        self.register_buffer("family_bank", family_bank, persistent=False)

    def forward(self, batch):
        return joint_loss(self.encoder, batch, self.config, family_bank=self.family_bank)


class Trainer:
    def __init__(self, model: PhaNetModel, config: TrainConfig, *, device="cpu", data_hash: str,
                 family_bank=None):
        if config.family_weight and family_bank is None:
            raise ValueError("Family supervision requires a training family bank")
        if config.total_steps is None:
            raise ValueError("Resolve the schedule from dataset size before creating a Trainer")
        self.model = model.to(device)
        self.config, self.device, self.data_hash = config, torch.device(device), data_hash
        self.family_bank = family_bank.to(device) if family_bank is not None else None
        objective = JointObjective(self.model, config, self.family_bank)
        self.distributed = dist.is_available() and dist.is_initialized()
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel
            self.objective = DistributedDataParallel(objective,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False, find_unused_parameters=True)
        else:
            self.objective = objective
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.amp and self.device.type == "cuda")
        self.step = 0
        self.micro_step = 0
        self.pending_micro_steps = 0
        self.amp_skipped_steps = 0
        self.amp_consecutive_skips = 0
        self.provenance = {}
        self.history = []
        self.best_validation = float("inf")
        self.bad_epochs = 0
        self.optimizer.zero_grad(set_to_none=True)

    def learning_rate(self):
        step = self.step
        warmup = self.config.warmup_steps
        if warmup and step < warmup:
            return self.config.learning_rate * (step + 1) / warmup
        progress = min(1., max(0., (step - warmup) / max(1, self.config.total_steps - warmup)))
        return self.config.learning_rate * .5 * (1 + math.cos(math.pi * progress))

    def update(self, batch):
        if self.step >= self.config.total_steps:
            raise RuntimeError("Configured optimizer-step budget is exhausted")
        self.model.train()
        batch = {k: v.to(self.device) for k, v in batch.items()}
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate()
        with torch.autocast(device_type=self.device.type, enabled=self.config.amp and self.device.type == "cuda"):
            loss, report = self.objective(batch)
        self.scaler.scale(loss / self.config.gradient_accumulation).backward()
        self.micro_step += 1
        self.pending_micro_steps += 1
        boundary = self.pending_micro_steps == self.config.gradient_accumulation
        previous_step = self.step
        if boundary:
            report["gradient_norm"] = self._optimizer_step()
            report.update(amp_overflow=self.step == previous_step, amp_scale=self.scaler.get_scale(),
                          amp_skipped_steps=self.amp_skipped_steps)
        report.update(step=self.step, optimizer_updated=self.step != previous_step,
                      learning_rate=self.optimizer.param_groups[0]["lr"])
        self.history.append(report)
        return report

    def _optimizer_step(self):
        self.scaler.unscale_(self.optimizer)
        if self.scaler.is_enabled():
            # GradScaler records nonfinite gradients and skips the accumulated update.
            checks = [torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None]
            if not checks:
                raise RuntimeError("No gradients at optimizer boundary")
            nonfinite = not bool(torch.stack(checks).all())
            if self.distributed:
                votes = torch.tensor(int(nonfinite), device=self.device)
                dist.all_reduce(votes)
                if int(votes) not in (0, dist.get_world_size()):
                    raise FloatingPointError("AMP overflow differs across ranks; refuse inconsistent updates")
            if nonfinite:
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)  # recorded Inf/NaN => no optimizer update
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.pending_micro_steps = 0
                self.amp_skipped_steps += 1
                self.amp_consecutive_skips += 1
                new_scale = self.scaler.get_scale()
                if (not math.isfinite(new_scale) or not 0 < new_scale < old_scale or
                        self.amp_consecutive_skips >= 16):
                    raise FloatingPointError("Persistent AMP overflow or invalid loss scale; stop for diagnosis")
                return None  # Missing norm after a skipped update
        # An epoch's final accumulation group can be shorter than configured.
        # Average its actual microbatches rather than dropping or underweighting it.
        correction = self.config.gradient_accumulation / self.pending_micro_steps
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(correction)
        norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip, error_if_nonfinite=True)
        self.scaler.step(self.optimizer); self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_micro_steps = 0
        self.amp_consecutive_skips = 0
        self.step += 1
        return float(norm)

    def finish_epoch(self):
        if self.pending_micro_steps:
            previous_step = self.step
            norm = self._optimizer_step()
            self.history[-1].update(step=self.step, optimizer_updated=self.step != previous_step,
                                    gradient_norm=norm, partial_accumulation=True,
                                    amp_overflow=self.step == previous_step, amp_scale=self.scaler.get_scale(),
                                    amp_skipped_steps=self.amp_skipped_steps)

    @torch.no_grad()
    def validate(self, batches, *, observer=None):
        self.model.eval()
        # Fixed RNG scope makes checkpoint selection independent of training RNG.
        values = []
        composition_keys = ("composition", "composition_margin", "composition_accuracy",
                            "composition_score_rms", "composition_module_variance")
        composition_sums = [0.] * (len(composition_keys) + 1)
        task_counts = {"annotation": "masked_annotations", "family": "family_examples", "order": "order_examples"}
        task_sums = {key: [0., 0., 0.] for key in task_counts}
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config.seed + 104729)
            for batch in batches:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                diagnostics = {} if observer is not None else None
                loss, report = joint_loss(self.model, batch, self.config, family_bank=self.family_bank,
                                     annotation_mask=batch["loss_eligible"], validation=True, diagnostics=diagnostics)
                values.append((float(loss), len(batch["valid"])))
                for key, count_key in task_counts.items():
                    count = report[count_key]
                    task_sums[key][0] += report[key] * len(batch["valid"])
                    task_sums[key][1] += report[key] * count
                    task_sums[key][2] += count
                if observer is not None:
                    observer(batch, report, diagnostics)
                if self.config.composition_weight:
                    count = report["composition_pairs"]
                    for i, key in enumerate(composition_keys):
                        composition_sums[i] += report[key] * count
                    composition_sums[-1] += count
        totals = torch.tensor([sum(v * n for v, n in values), sum(n for _, n in values)], dtype=torch.float64, device=self.device)
        if self.distributed:
            dist.all_reduce(totals)
        if not totals[1]:
            raise ValueError("Validation partition is empty")
        self.last_validation_report = {"total": float(totals[0] / totals[1]), "windows": int(totals[1])}
        for key, count_key in task_counts.items():
            sums = torch.tensor(task_sums[key], dtype=torch.float64, device=self.device)
            if self.distributed:
                dist.all_reduce(sums)
            self.last_validation_report[key] = float(sums[0] / totals[1])
            self.last_validation_report[key + "_target_mean"] = float(sums[1] / sums[2]) if sums[2] else None
            self.last_validation_report[count_key] = int(sums[2])
        if self.config.composition_weight:
            sums = torch.tensor(composition_sums, dtype=torch.float64, device=self.device)
            if self.distributed:
                dist.all_reduce(sums)
            if not sums[-1]:
                raise ValueError("Validation has no eligible composition pairs")
            self.last_validation_report.update({key: float(sums[i] / sums[-1])
                                                for i, key in enumerate(composition_keys)})
            self.last_validation_report["composition_pairs"] = int(sums[-1])
        return float(totals[0] / totals[1])

    def checkpoint(self, path: str | Path, *, sampler_state: dict | None = None):
        if self.pending_micro_steps:
            raise RuntimeError("Checkpoint only at an optimizer boundary; do not drop accumulated gradients")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "model_config": asdict(self.model.config), "train_config": asdict(self.config),
                   "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(),
                   "step": self.step, "micro_step": self.micro_step, "data_hash": self.data_hash,
                   "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                   "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                   "family_bank": self.family_bank, "history": self.history, "best_validation": self.best_validation,
                   "bad_epochs": self.bad_epochs, "sampler_state": sampler_state or {}}
        payload["amp_overflow_state"] = {"skipped_steps": self.amp_skipped_steps,
                                         "consecutive_skips": self.amp_consecutive_skips}
        payload["provenance"] = self.provenance
        payload["distributed"] = {"rank": dist.get_rank() if self.distributed else 0,
                                  "world_size": dist.get_world_size() if self.distributed else 1}
        temporary = path.with_suffix(path.suffix + ".partial")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    @classmethod
    def restore(cls, path: str | Path, *, data_hash: str, device="cpu"):
        # Training checkpoints contain Python optimizer and RNG state.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        distributed = dist.is_available() and dist.is_initialized()
        current = {"rank": dist.get_rank() if distributed else 0, "world_size": dist.get_world_size() if distributed else 1}
        if payload.get("distributed", {"rank": 0, "world_size": 1}) != current:
            raise ValueError("Exact resume requires the same rank and world size")
        if payload["data_hash"] != data_hash or payload["schema_version"] != 1:
            raise ValueError("Checkpoint belongs to a different dataset/schema")
        model = PhaNetModel(ModelConfig(**payload["model_config"]))
        trainer = cls(model, TrainConfig(**payload["train_config"]), device=device,
                      data_hash=data_hash, family_bank=payload["family_bank"])
        model.load_state_dict(payload["model"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        # Move restored optimizer tensors to the training device.
        for state in trainer.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(trainer.device)
        trainer.scaler.load_state_dict(payload["scaler"])
        trainer.step, trainer.micro_step = payload["step"], payload["micro_step"]
        overflow = payload.get("amp_overflow_state", {})
        trainer.amp_skipped_steps = overflow.get("skipped_steps", 0)
        trainer.amp_consecutive_skips = overflow.get("consecutive_skips", 0)
        trainer.history, trainer.best_validation, trainer.bad_epochs = payload["history"], payload["best_validation"], payload["bad_epochs"]
        trainer.sampler_state = payload["sampler_state"]
        trainer.provenance = payload.get("provenance", {})
        torch.set_rng_state(payload["torch_rng"])
        np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"])
        if payload["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return trainer
