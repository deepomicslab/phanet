"""Automatic three-channel network annotation for prepared nucleotide inputs."""
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib.metadata
import shutil
import subprocess

import numpy as np
from scipy import sparse

from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, read_json, write_json
from phanet.network.annotation import sequence_vote, ppr_queries, choose_network
from phanet.network.resources import NetworkBundle
from phanet.network.search import sequence_search, profile_search, profile_edges, structure_edges
from phanet.representations.tmvec import model_identity, full_length_vectors


@dataclass(frozen=True)
class NetworkConfig:
    reference: str
    profiles: str
    tmvec_model: str
    prott5: str
    mmseqs: str = "mmseqs"

    @classmethod
    def load(cls, path):
        path = Path(path).expanduser().resolve()
        raw = read_json(path)
        if raw.pop("schema_version", None) != 1:
            raise ValueError("Unknown network configuration version")
        for name in ("reference", "profiles", "tmvec_model", "prott5", "mmseqs"):
            if name in raw and (name != "mmseqs" or "/" in raw[name]):
                p = Path(raw[name]).expanduser()
                raw[name] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
        return cls(**raw)


def preflight(config, *, progress=print):
    try:
        import pyhmmer
        import transformers
        import sentencepiece
    except ImportError as exc:
        raise RuntimeError("Install network dependencies: pip install 'phanet-core[network]'") from exc
    if pyhmmer.__version__ != "0.11.0":
        raise ValueError("Network profile inference requires PyHMMER 0.11.0")
    binary = shutil.which(config.mmseqs)
    if not binary:
        raise FileNotFoundError("MMseqs2 executable not found; configure mmseqs in the network JSON")
    version = subprocess.run([binary, "version"], capture_output=True, text=True, check=True).stdout.strip()
    bundle = NetworkBundle(config.reference, progress=progress)
    if version != bundle.meta["mmseqs_version"]:
        raise ValueError("MMseqs2 version differs from the reference self-score contract")
    if digest_file(config.profiles) != bundle.meta["phrog_sha256"]:
        raise ValueError("PHROG library checksum mismatch")
    models = model_identity(config.tmvec_model, config.prott5)
    if models != bundle.meta["models"]:
        raise ValueError("TM-Vec/ProtT5 resources differ from the reference representation model")
    return bundle, {"config": asdict(config), "bundle_sha256": digest_file(bundle.root / "network.json"),
                    "models": models, "mmseqs_sha256": digest_file(binary), "mmseqs_version": version,
                    "phrog_sha256": bundle.meta["phrog_sha256"], "pyhmmer": pyhmmer.__version__,
                    "transformers": transformers.__version__, "sentencepiece": sentencepiece.__version__}


def source_policy(data, targets, bundle):
    by_id = {p.protein_id: p for p in data.proteins}
    hidden = {by_id[pid].sha256 for pid in targets}
    target_groups = {}
    for pid in targets:
        p = by_id[pid]
        target_groups.setdefault(p.genome_id, set()).add(p.sha256)
    visible = {}
    for contig in target_groups:
        seen = {}
        for p in data.by_genome()[contig]:
            if p.sha256 not in hidden and p.labels:
                seen.setdefault(p.sha256, set()).add(tuple(sorted(p.labels)))
        # Exclude local sources with conflicting annotations.
        visible[contig] = {h: np.array([c in next(iter(labels)) for c in CATEGORIES], float)
                           for h, labels in seen.items() if len(labels) == 1}
    return hidden, {g: sorted(hs) for g, hs in target_groups.items()}, visible


def mean_edges(*layers):
    sums, counts = {}, {}
    for layer in layers:
        for h, value in layer.items():
            sums[h] = sums.get(h, 0.) + value
            counts[h] = counts.get(h, 0) + 1
    return {h: value/counts[h] for h, value in sums.items()}


def network_probabilities(data, targets, bundle, config, output, *, device="cpu", threads=4, progress=print):
    """Output one row per native target; no predicted label becomes a new source."""
    out = Path(output)
    hidden, groups, visible = source_policy(data, targets, bundle)
    sequences = {p.sha256: p.sequence for p in data.proteins}
    needed = sorted(hidden | {h for rows in visible.values() for h in rows})
    global_pos = dict(zip(needed, bundle.positions(needed)))
    local = {g: {h: labels for h, labels in rows.items() if global_pos[h] < 0} for g, rows in visible.items()}
    allowed = np.ones(len(bundle.ids), bool)
    for h in hidden:
        if global_pos[h] >= 0:
            allowed[global_pos[h]] = False
    progress("Searching sequence relationships")
    seq = sequence_search(bundle, groups, sequences, local, out / "sequence", mmseqs=config.mmseqs,
                          threads=threads, hidden=hidden)
    progress("Searching PHROG family relationships")
    # Fresh full-length profile searches include every target and local-only source.
    hits = profile_search({h: sequences[h] for h in needed}, config.profiles, out / "profile", threads=threads,
                          progress=progress, profile_count=bundle.meta["profile_count"],
                          profile_Z=bundle.meta["profile_Z"])
    missing = [h for h in needed if global_pos[h] < 0]
    progress(f"Preparing full-length structure representations ({len(missing)} new proteins)")
    vectors = {h: np.asarray(bundle.vectors[j]) for h, j in global_pos.items() if j >= 0}
    if missing:
        new = full_length_vectors([sequences[h] for h in missing], config.tmvec_model, config.prott5, device=device, progress=progress)
        vectors.update(zip(missing, new))
    np.save(out / "vectors.npy", np.stack([vectors[h] for h in needed]), allow_pickle=False)
    write_json(out / "vector_ids.json", needed)
    labels = bundle.label_matrix()
    # Query/reference alignment rows are small; avoid a Python dictionary of all references' label arrays.
    vote_labels = {h: labels[j] for h, j in zip(needed, bundle.positions(needed)) if j >= 0}
    all_hit_ids = sorted({hit["reference_id"] for gs in seq.values() for row in gs.values() for hit in row["hits"]})
    vote_labels.update((h, labels[j]) for h, j in zip(all_hit_ids, bundle.positions(all_hit_ids)) if j >= 0)
    by_id = {p.protein_id: p for p in data.proteins}
    row_index = {pid: i for i, pid in enumerate(targets)}
    sequence_scores = np.zeros((len(targets), 9)); sequence_rho = np.zeros_like(sequence_scores)
    global_rows, global_cols, global_values = [], [], []
    local_rows, local_cols, local_values, local_labels = [], [], [], []
    evidence = []
    for contig, queries in groups.items():
        prof = profile_edges(bundle, [h for h in hits if h["protein_id"] in queries], hits, allowed, set(local[contig]))
        struct = structure_edges(bundle, {h: vectors[h] for h in queries}, {h: vectors[h] for h in local[contig]}, allowed)
        contig_labels = {**vote_labels, **local[contig]}
        for pid in targets:
            p = by_id[pid]
            if p.genome_id != contig:
                continue
            i, h = row_index[pid], p.sha256
            vote = sequence_vote(seq[contig][h]["hits"], contig_labels)
            sequence_scores[i], sequence_rho[i] = vote["scores"], vote["rho"]
            integrated = mean_edges(seq[contig][h]["edges"], prof.get(h, {}), struct[h])
            ordered = sorted(integrated); positions = bundle.positions(ordered)
            for r, j in zip(ordered, positions):
                weight = integrated[r]
                if j >= 0:
                    if not allowed[j]:
                        raise ValueError("Hidden target used as network reference")
                    global_rows.append(i); global_cols.append(j); global_values.append(weight)
                else:
                    if r not in local[contig]:
                        raise ValueError("Local source outside target contig")
                    local_rows.append(i); local_cols.append(len(local_labels)); local_values.append(weight)
                    local_labels.append(local[contig][r])
            evidence.append({"protein_id": pid, "sequence_references": vote["references"],
                             "channels": {"sequence": seq[contig][h]["edges"], "profile": prof.get(h, {}), "structure": struct[h]},
                             "integrated": integrated})
    shape = (len(targets), len(bundle.ids))
    ge = sparse.csr_matrix((global_values, (global_rows, global_cols)), shape=shape)
    le = sparse.csr_matrix((local_values, (local_rows, local_cols)), shape=(len(targets), len(local_labels)))
    needs_ppr = ~(sequence_scores >= .5).any(axis=1)
    if needs_ppr.any():
        support, _, convergence = bundle.masked_support(hidden, progress=progress)
        ppr = ppr_queries(ge, support, local_edges=le, local_labels=np.asarray(local_labels).reshape(-1, 9), allowed_local=le.astype(bool))
    else:
        convergence = {"skipped": "all targets have sequence assignments"}
        ppr = {"scores": np.zeros_like(sequence_scores), "rho": np.zeros_like(sequence_scores)}
    scores, rho, route = choose_network(sequence_scores, sequence_rho, ppr["scores"], ppr["rho"])
    np.savez(out / "predictions.npz", scores=scores, rho=rho, route=route,
             sequence_scores=sequence_scores, sequence_rho=sequence_rho, ppr_scores=ppr["scores"], ppr_rho=ppr["rho"])
    write_json(out / "targets.json", targets)
    write_json(out / "evidence.json", evidence)
    write_json(out / "summary.json", {"channels": ["sequence", "profile", "structure"], "targets": len(targets),
                                     "hidden_references": int((~allowed).sum()), "convergence": convergence})
    return scores, rho, route
