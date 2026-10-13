"""Genomic context windows and batch tensor construction."""
from __future__ import annotations
from dataclasses import dataclass
import bisect
import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from phanet.datasets.schema import Dataset
from phanet.datasets.splits import MaskPolicy, SplitManifest, PADDING
from phanet.representations.embed import EmbeddingStore
from phanet.datasets.preparation import segment_starts
from phanet.common.constants import CATEGORIES


@dataclass(frozen=True)
class Window:
    genome_id: str
    indices: tuple[int, ...]
    members: tuple[int, ...]
    target: int | None = None


def context_bounds(n: int, start: int, width: int, length=30) -> tuple[int, int]:
    if not 0 <= start < start + width <= n or width > length:
        raise ValueError("Invalid module/context window")
    left = max(0, min(start - (length - width) // 2, n - length))
    return left, min(n, left + length)


class ContextDataset(TorchDataset):
    """O(genomes) window indexing; vectors are loaded only for requested examples."""
    def __init__(self, dataset: Dataset, embeddings: EmbeddingStore, splits: SplitManifest,
                 policy: MaskPolicy, *, partition: str | None, max_context=30, mode="segments", genome_ids=None,
                 sampling_seed=7):
        dataset.validate(); splits.validate(dataset)
        if mode not in ("segments", "modules", "centered", "sampled_modules") or max_context < 8:
            raise ValueError("Unsupported context construction")
        self.dataset, self.embeddings, self.splits = dataset, embeddings, splits
        if partition is not None and partition not in set(splits.assignments.values()):
            raise ValueError(f"Unknown context partition: {partition}")
        if genome_ids is not None:
            requested = list(genome_ids)
            known = {g.genome_id for g in dataset.genomes}
            if len(requested) != len(set(requested)) or not set(requested) <= known:
                raise ValueError("Context genome IDs must be unique and belong to the dataset")
            genome_ids = set(requested)
        self.view = policy.apply(dataset, splits)
        self.truth = dataset.label_matrix()
        self.max_context, self.mode = max_context, mode
        self.sampling_seed, self.epoch = sampling_seed, 0
        segmented = mode in ("segments", "sampled_modules")
        window_config = dataset.provenance.get("context_windows", {})
        configured_overlap = window_config.get("overlap", 0) if segmented else 0
        if (isinstance(configured_overlap, bool) or
                not isinstance(configured_overlap, (int, np.integer))):
            raise ValueError("Context-window overlap must be an integer")
        if not 0 <= int(configured_overlap) < max_context:
            raise ValueError("Context-window overlap must lie in [0, max_context)")
        self.overlap = int(configured_overlap)
        if segmented and window_config and window_config.get("max_context") != max_context:
            raise ValueError("Model max_context differs from the prepared window contract")
        segment_starts(0, max_context, self.overlap)
        self.groups, self.ends = [], []
        self.segment_offsets = []
        protein_index = {p.protein_id: i for i, p in enumerate(dataset.proteins)}
        total = 0
        for gid, proteins in dataset.by_genome().items():
            if genome_ids is not None and gid not in genome_ids:
                continue
            if (partition is not None and splits.assignments[gid] != partition) or not proteins:
                continue
            indices = tuple(protein_index[p.protein_id] for p in proteins)
            starts = segment_starts(len(indices), max_context, self.overlap) if segmented else ()
            if mode == "sampled_modules":
                starts = [s for s in starts if len(indices) - s >= 2]
            count = len(starts) if segmented else len(indices) if mode == "centered" else sum(max(0, len(indices) - w + 1) for w in range(2, 9))
            if count:
                self.groups.append((gid, indices))
                self.segment_offsets.append(starts)
                total += count
                self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Invalid sampling epoch")
        self.epoch = epoch

    def window(self, index: int) -> Window:
        if not 0 <= index < len(self):
            raise IndexError(index)
        group = bisect.bisect_right(self.ends, index)
        local = index - (self.ends[group - 1] if group else 0)
        gid, indices = self.groups[group]
        if self.mode == "sampled_modules":
            anchor = self.segment_offsets[group][local]
            size = min(self.max_context, len(indices) - anchor)
            # Stateless selection: neither batching nor resume consumes training RNG.
            payload = f"bounded_v1:{self.sampling_seed}:{self.epoch}:{gid}:{anchor}".encode()
            bits = hashlib.sha256(payload).digest()
            width = 2 + int.from_bytes(bits[:8], "big") % (min(8, size) - 1)
            start = anchor + int.from_bytes(bits[8:16], "big") % (size - width + 1)
            left, right = context_bounds(len(indices), start, width, self.max_context)
            return Window(gid, indices[left:right], tuple(range(start-left, start-left+width)))
        if self.mode == "centered":
            left, right = context_bounds(len(indices), local, 1, self.max_context)
            selected = indices[left:right]
            start = max(0, min(local - left - 3, len(selected) - 8))
            return Window(gid, selected, tuple(range(start, min(start + 8, len(selected)))), target=indices[local])
        if self.mode == "segments":
            start = self.segment_offsets[group][local]
            selected = indices[start:start + self.max_context]
            # Joint objectives use a native <=8-gene window. All 2--8 windows
            # are covered separately through mode='modules'.
            members = tuple(range(min(8, len(selected))))
        else:
            for width in range(2, 9):
                count = max(0, len(indices) - width + 1)
                if local < count:
                    start = local
                    break
                local -= count
            left, right = context_bounds(len(indices), start, width, self.max_context)
            selected = indices[left:right]
            members = tuple(range(start - left, start - left + width))
        return Window(gid, selected, members)

    def __getitem__(self, index):
        return self.collate([index])

    def collate(self, indices):
        """Materialize a batch of windows with one embedding-store lookup."""
        windows = [self.window(int(index)) for index in indices]
        batch_size, length = len(windows), self.max_context
        if not windows:
            raise ValueError("Cannot collate an empty context batch")
        all_positions = []
        for window in windows:
            all_positions.extend(window.indices)
        proteins = [self.dataset.proteins[i] for i in all_positions]
        vectors = self.embeddings.get([p.protein_id for p in proteins],
                                      sequences=[p.sequence for p in proteins])
        cursor = 0
        x = np.zeros((batch_size, length, self.embeddings.dimension), dtype=np.float32)
        observed = np.zeros((batch_size, length, len(CATEGORIES)), dtype=np.float32)
        state = np.full((batch_size, length), PADDING, dtype=np.int64)
        strand = np.ones((batch_size, length), dtype=np.int64)
        valid = np.zeros((batch_size, length), dtype=bool)
        target = np.zeros((batch_size, length, len(CATEGORIES)), dtype=np.float32)
        eligible = np.zeros((batch_size, length), dtype=bool)
        members = np.zeros((batch_size, length), dtype=bool)
        protein_indices = np.full((batch_size, length), -1, dtype=np.int64)
        prediction_mask = np.zeros((batch_size, length), dtype=bool)
        for row, window in enumerate(windows):
            positions = list(window.indices)
            n = len(positions)
            if n:
                x[row, :n] = vectors[cursor:cursor + n]
                cursor += n
                protein_rows = np.asarray(positions, dtype=np.int64)
                observed[row, :n] = self.view["annotation_input"][protein_rows]
                state[row, :n] = self.view["state"][protein_rows]
                strand[row, :n] = [self.dataset.proteins[i].strand for i in positions]
                valid[row, :n] = True
                target[row, :n] = self.truth[protein_rows]
                if "eligible_mask" in self.view:
                    eligible[row, :n] = self.view["eligible_mask"][protein_rows]
                else:  # compatibility with pre-existing serialized policies
                    eligible[row, :n] = self.truth[protein_rows].any(axis=1)
                protein_indices[row, :n] = protein_rows
                prediction_mask[row, :n] = True
            if window.target is not None:
                prediction_mask[row] = protein_indices[row] == window.target
            member_positions = np.asarray(window.members, dtype=np.int64)
            if len(member_positions):
                members[row, member_positions] = True
        # Training loss is restricted to the policy's train targets; validation
        # and test windows score their own labels, while still respecting any
        # explicitly masked proteins.
        for row, window in enumerate(windows):
            if self.splits.assignments[window.genome_id] == "train":
                positions = np.asarray(window.indices, dtype=np.int64)
                eligible[row, :len(positions)] = self.view["loss_eligible"][positions]
        values = {"embeddings": x, "annotation_input": observed, "state": state, "strand": strand,
                  "valid": valid, "targets": target, "loss_eligible": eligible,
                  "member_mask": members, "protein_indices": protein_indices, "prediction_mask": prediction_mask}
        result = {key: torch.from_numpy(value) for key, value in values.items()}
        return result if len(windows) > 1 else {key: value[0] for key, value in result.items()}
