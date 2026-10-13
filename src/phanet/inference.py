"""User-facing GLM inference with native contig context and fixed nine-class outputs."""
from __future__ import annotations

from dataclasses import asdict
import gc
import os
from pathlib import Path
import shutil
from urllib.parse import quote

import numpy as np
import torch

from phanet.annotation import PrepareConfig, prepare, _stage
from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, digest_object, read_json, read_table, write_json, write_table
from phanet.datasets.schema import Dataset
from phanet.datasets.splits import MASKED, VISIBLE, UNRESOLVED, PADDING
from phanet.learning.context import context_bounds
from phanet.learning.model import ModelConfig, PhaNetModel
from phanet.representations.embed import ESMConfig, EmbeddingStore, embed_esm


def model_metadata(directory):
    root = Path(directory).expanduser().resolve()
    meta = read_json(root / "model.json")
    if meta.get("schema_version") != 1 or meta.get("categories") != list(CATEGORIES):
        raise ValueError("Expected a versioned nine-category inference model bundle")
    if meta.get("embedding") != "esm2_t33_650M_UR50D:layer33:full_length:mean:float16":
        raise ValueError("Unsupported model embedding contract")
    if meta.get("threshold") != 0.5 or meta.get("weights") != "weights.pt":
        raise ValueError("Unsupported model decision rule or weights path")
    if digest_file(root / "weights.pt") != meta.get("weights_sha256"):
        raise ValueError("Model weights checksum mismatch")
    if not isinstance(meta.get("model_config"), dict):
        raise ValueError("Missing model architecture configuration")
    config = ModelConfig(**meta["model_config"])
    config.validate()
    if config.input_dim != 1280 or config.max_context != 30:
        raise ValueError("Expected 1280-dimensional proteins and 30-position context")
    return root, meta


def load_model(directory, device="cpu"):
    root, meta = model_metadata(directory)
    # Load the inference tensor dictionary.
    weights = torch.load(root / "weights.pt", map_location="cpu", weights_only=True)
    model = PhaNetModel(ModelConfig(**meta["model_config"]))
    model.load_state_dict(weights, strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("Nonfinite model weights")
    return model.eval().to(device)


def context_batch(data, targets, embeddings, *, hidden_hashes, length=30, context_index=None):
    """One centered native window per target; all target-sequence aliases are hidden."""
    if context_index is None:
        groups = data.by_genome()
        lookup = {p.protein_id: (gid, i) for gid, ps in groups.items() for i, p in enumerate(ps)}
    else:
        groups, lookup = context_index
    selected, offsets = [], []
    for pid in targets:
        gid, i = lookup[pid]
        left, right = context_bounds(len(groups[gid]), i, 1, length)
        selected.append(groups[gid][left:right]); offsets.append(i-left)
    batch = len(targets)
    x = np.zeros((batch, length, embeddings.dimension), np.float32)
    ann = np.zeros((batch, length, 9), np.float32)
    state = np.full((batch, length), PADDING, np.int64)
    strand = np.ones((batch, length), np.int64)
    valid = np.zeros((batch, length), bool)
    proteins = [p for window in selected for p in window]
    vectors = embeddings.get([p.protein_id for p in proteins], sequences=[p.sequence for p in proteins])
    cursor = 0
    for b, window in enumerate(selected):
        for j, p in enumerate(window):
            x[b, j] = vectors[cursor]; cursor += 1
            strand[b, j] = p.strand; valid[b, j] = True
            if p.sha256 in hidden_hashes or p.protein_id == targets[b]:
                state[b, j] = MASKED
            else:
                state[b, j] = VISIBLE if p.labels else UNRESOLVED
                for label in p.labels:
                    ann[b, j, CATEGORIES.index(label)] = 1
    return [torch.from_numpy(a) for a in (x, ann, state, strand, valid)], offsets


def glm_probabilities(data, targets, embeddings, model, *, batch_size=8, device="cpu", progress=print):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data.validate()
    by_id = {p.protein_id: p for p in data.proteins}
    if len(targets) != len(set(targets)) or not set(targets) <= set(by_id):
        raise ValueError("Targets must be unique protein IDs in the dataset")
    if embeddings.dimension != model.config.input_dim:
        raise ValueError("Embedding and model dimensions differ")
    hidden = {by_id[p].sha256 for p in targets}
    result = np.empty((len(targets), 9), np.float32)
    groups = data.by_genome()
    context_index = (groups, {p.protein_id: (gid, i) for gid, ps in groups.items() for i, p in enumerate(ps)})
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(targets), batch_size):
            selected = targets[start:start+batch_size]
            batch, offsets = context_batch(data, selected, embeddings, hidden_hashes=hidden,
                                           length=model.config.max_context, context_index=context_index)
            logits = model(*[x.to(device) for x in batch])["logits"]
            probs = logits[torch.arange(len(selected), device=device), torch.tensor(offsets, device=device)].sigmoid()
            if not torch.isfinite(probs).all():
                raise ValueError("Nonfinite GLM predictions")
            result[start:start+len(selected)] = probs.cpu().numpy()
            progress(f"GLM: {start+len(selected)}/{len(targets)} proteins")
    return result


def _release_stage_memory(device):
    """Collect unreachable stage objects and return unused CUDA allocator blocks."""
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def annotate_glm(input_fasta, output, config: PrepareConfig, model_dir=None, esm_checkpoint=None, *,
                 device="cpu", batch_size=8, resume=False, progress=print, network_config=None,
                 method=None):
    """Traditional annotation plus GLM, network, or both."""
    if method is None:
        method = "combined" if network_config is not None else "glm"
    if method not in ("glm", "network", "combined"):
        raise ValueError("method must be glm, network, or combined")
    if method in ("network", "combined") and network_config is None:
        raise ValueError("Network inference requires network_config")
    if method in ("glm", "combined") and (model_dir is None or esm_checkpoint is None):
        raise ValueError("GLM inference requires model_dir and esm_checkpoint")
    bundle = None
    network_identity = None
    if method in ("network", "combined"):
        from phanet.network.inference import preflight as network_preflight
        bundle, network_identity = network_preflight(network_config, progress=progress)
    if device not in ("cpu", "cuda") or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Select cpu or an available cuda device")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    root = meta = esm = None
    resources = {}
    if method in ("glm", "combined"):
        root, meta = model_metadata(model_dir)
        esm = Path(esm_checkpoint).expanduser().resolve()
        if esm.name != "esm2_t33_650M_UR50D.pt" or not esm.is_file():
            raise FileNotFoundError("Provide the local esm2_t33_650M_UR50D.pt checkpoint")
        import importlib.util
        if importlib.util.find_spec("esm") is None:
            raise RuntimeError("Install the ESM dependency with pip install 'phanet-core[esm]'")
        resources = {"model_json_sha256": digest_file(root / "model.json"),
                     "weights_sha256": meta["weights_sha256"], "esm_sha256": digest_file(esm)}
        regression = esm.with_name("esm2_t33_650M_UR50D-contact-regression.pt")
        if not regression.is_file():
            raise FileNotFoundError(f"Required ESM companion checkpoint: {regression}")
        resources["esm_contact_sha256"] = digest_file(regression)
    source, out = Path(input_fasta).resolve(), Path(output).absolute()
    if out.is_symlink() or source.is_relative_to(out.resolve()):
        raise ValueError("Input must be outside a nonsymlink output directory")
    out = out.resolve()
    identity = {"input_sha256": digest_file(source), "config": asdict(config), "resources": resources,
                "method": method, "network": network_identity, "categories": list(CATEGORIES), "device": device, "batch_size": batch_size,
                "torch_version": torch.__version__, "code": {str(p.relative_to(Path(__file__).parent)): digest_file(p)
                for p in sorted(Path(__file__).parent.rglob("*.py"))}}
    fingerprint = digest_object(identity)
    if out.exists():
        if not resume:
            raise FileExistsError(f"Output exists: {out}; use --resume or a new directory")
        if read_json(out / "RUN.json").get("fingerprint") != fingerprint:
            raise ValueError("Resume input, model, settings or code changed")
    else:
        out.mkdir(parents=True)
    lock = out / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
    try:
        if not (out / "RUN.json").exists():
            write_json(out / "RUN.json", {"fingerprint": fingerprint, **identity})
        write_json(out / "status.json", {"state": "running", "method": method})
        prepare(source, out / "traditional", config, resume=resume, progress=progress)
        prepared = out / "traditional/prepared"
        data = Dataset.load(prepared / "dataset")
        targets = [row["protein_id"] for row in read_table(prepared / "targets.tsv")]
        net_scores = np.empty((0, 9)); net_rho = np.empty((0, 9)); net_routes = []
        network_used = bool(targets and bundle is not None)
        if network_used:
            from phanet.network.inference import network_probabilities
            def infer_network(path):
                network_probabilities(data, targets, bundle, network_config, path,
                                      device=device, threads=config.threads, progress=progress)
            try:
                _stage(out / "network", fingerprint, infer_network)
            finally:
                # The graph and reference mmap arrays are only needed here.
                # Release them before loading ESM2 or the GLM model.
                bundle = None
                _release_stage_memory(device)
            with np.load(out / "network/predictions.npz", allow_pickle=False) as saved:
                net_scores, net_rho, net_routes = saved["scores"], saved["rho"], saved["route"].tolist()
            if read_json(out / "network/targets.json") != targets:
                raise ValueError("Network and target orders differ")
            if (net_scores.shape != (len(targets), 9) or net_rho.shape != net_scores.shape
                    or not np.isfinite(net_scores).all() or not np.isfinite(net_rho).all()
                    or np.any((net_scores < 0) | (net_scores > 1))
                    or np.any((net_rho < 0) | (net_rho > 1))
                    or len(net_routes) != len(targets)
                    or any(route not in ("sequence", "ppr", "unassigned") for route in net_routes)):
                raise ValueError("Invalid saved network predictions")
        elif bundle is not None:
            bundle = None
            _release_stage_memory(device)
        probs = np.empty((0, 9), np.float32)
        if targets and method in ("glm", "combined"):
            def vectors(path):
                progress("Computing full-length ESM2 protein representations")
                embed_esm([p.protein_id for p in data.proteins], [p.sequence for p in data.proteins], path,
                          ESMConfig(device=device, batch_size=batch_size, full_length=True, checkpoint=str(esm)))
            try:
                _stage(out / "embeddings", fingerprint, vectors)
            finally:
                _release_stage_memory(device)
            def infer(path):
                model = load_model(root, device)
                store = EmbeddingStore(out / "embeddings")
                values = glm_probabilities(data, targets, store, model, batch_size=batch_size,
                                           device=device, progress=progress)
                np.save(path / "probabilities.npy", values, allow_pickle=False)
                write_json(path / "targets.json", targets)
            try:
                _stage(out / "glm", fingerprint, infer)
            finally:
                _release_stage_memory(device)
            probs = np.load(out / "glm/probabilities.npy", allow_pickle=False)
            if probs.shape != (len(targets), 9) or not np.isfinite(probs).all() or np.any((probs<0)|(probs>1)):
                raise ValueError("Invalid saved GLM probabilities")
        glm_probs = probs.copy()
        if network_used:
            from phanet.network.annotation import fuse
            probs = fuse(glm_probs, net_scores, net_rho) if method == "combined" else net_scores
        def export(path):
            predicted = dict(zip(targets, probs))
            target_index = {pid: i for i, pid in enumerate(targets)}
            rows = list(read_table(prepared / "annotations.tsv"))
            assigned = 0
            for row in rows:
                scores = predicted.get(row["protein_id"])
                labels = [] if scores is None else [c for c, value in zip(CATEGORIES, scores) if value >= .5]
                assigned += bool(labels)
                row.update(phanet_labels=";".join(labels), phanet_method=method if scores is not None else "",
                           final_labels=row["labels"] if scores is None else ";".join(labels),
                           final_source="eggNOG-mapper" if row["traditional_status"] == "annotated" else ("PhaNet" if method == "combined" else "PhaNet-Network" if method == "network" else "PhaNet-GLM") if labels else "unresolved")
                for i, label in enumerate(CATEGORIES):
                    row["score_"+label] = "" if scores is None else float(scores[i])
                if method in ("combined", "network"):
                    j = target_index.get(row["protein_id"])
                    row["network_route"] = "" if j is None else net_routes[j]
                    for k, label in enumerate(CATEGORIES):
                        if method == "combined":
                            row["glm_"+label] = "" if j is None else float(glm_probs[j, k])
                        row["network_"+label] = "" if j is None else float(net_scores[j, k])
                        row["network_weight_"+label] = "" if j is None else float(net_rho[j, k])
            fields = list(rows[0]) if rows else ["protein_id", "contig_id", "product", "labels", "phanet_labels", "phanet_method", "final_labels", "final_source", *["score_"+c for c in CATEGORIES]]
            if method in ("combined", "network") and not rows:
                prefixes = ("glm_", "network_", "network_weight_") if method == "combined" else ("network_", "network_weight_")
                fields += ["network_route", *[prefix+c for c in CATEGORIES for prefix in prefixes]]
            write_table(path / "annotations.tsv", rows, fields)
            with (path / "annotations.gff3").open("w") as stream:
                stream.write("##gff-version 3\n")
                for g in data.genomes:
                    stream.write(f"##sequence-region {quote(g.genome_id, safe='')} 1 {g.length_nt}\n")
                for p, row in zip(data.proteins, rows):
                    attrs={"ID":p.protein_id,"product":p.product or "hypothetical protein", "annotation_source":row["final_source"], "phanet_labels":row["phanet_labels"], "phanet_method":row["phanet_method"]}
                    encoded=";".join(f"{k}={quote(v,safe='')}" for k,v in attrs.items())
                    stream.write(f"{quote(p.genome_id,safe='')}\tProdigal\tCDS\t{p.start+1}\t{p.end}\t.\t{'+' if p.strand==1 else '-'}\t{p.frame}\t{encoded}\n")
            shutil.copyfile(prepared / "proteins.faa", path / "proteins.faa")
            write_json(path / "summary.json", {"state":"complete", "method":method, "network_used":network_used,
                       "proteins":len(rows),"traditional_annotated":len(rows)-len(targets),"phanet_targets":len(targets),
                       "phanet_assigned":assigned,"remaining_unresolved":len(targets)-assigned,"categories":list(CATEGORIES)})
        _stage(out / "results", fingerprint, export)
        summary = read_json(out / "results/summary.json")
        write_json(out / "status.json", summary)
        return summary
    except BaseException as exc:
        write_json(out / "status.json", {"state":"failed", "method":method, "error":str(exc)})
        raise
    finally:
        lock.unlink()
