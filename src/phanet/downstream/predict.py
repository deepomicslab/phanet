"""Downstream prediction from complete native datasets, with shared ESM2 input."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from phanet.annotation import prepare
from phanet.common.io import digest_file, digest_object, write_json, write_table
from phanet.datasets.schema import Dataset
from phanet.inference import context_batch, _release_stage_memory
from phanet.representations.embed import ESMConfig, EmbeddingStore, embed_esm
from . import bundles
from .module_prediction import PredictionConfig, predict_genome

FATAL = {"invalid_coordinate", "frameshift", "severe_assembly_warning", "unsupported_gene_call"}


def validate_embeddings(data, store):
    from phanet.representations.embed import validate_full_length
    validate_full_length(data, store)


def module_rows(data, store, model, config, *, device="cpu"):
    """Yield candidates and coverage for every genome, including zero-CDS contigs."""
    data.validate()
    for gid, proteins in data.by_genome().items():
        if not proteins:
            yield gid, [], dict(proteins=0, windows=0, distinct_before_support=0,
                                retained=0, span_qc_rejected=0, accepted=0)
            continue
        vectors = store.get([p.protein_id for p in proteins], sequences=[p.sequence for p in proteins])
        raw, counts = predict_genome(model, vectors, np.arange(len(proteins)),
                                    [p.strand for p in proteins], config, device=device)
        counts.update(proteins=len(proteins), span_qc_rejected=0, accepted=0)
        rows = []
        for prediction in raw:
            ranks = prediction["members"]
            span = proteins[ranks[0]:ranks[-1] + 1]
            passed = all(not p.partial and not (set(p.quality_flags) & FATAL) for p in span)
            members = [proteins[i].protein_id for i in ranks]
            rows.append(dict(prediction, member_ranks=ranks, members=members, genome_id=gid,
                             module_id=digest_object([gid, members]), span_qc_passed=passed,
                             start=min(p.start for p in span), end=max(p.end for p in span)))
            counts["accepted" if passed else "span_qc_rejected"] += 1
        yield gid, rows, counts


def go_batches(data, store, model, *, batch_size=8, device="cpu"):
    """Predict every CDS, masking each target while retaining neighbor annotations."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    data.validate()
    groups = data.by_genome()
    index = (groups, {p.protein_id: (gid, i) for gid, ps in groups.items() for i, p in enumerate(ps)})
    ids = [p.protein_id for p in data.proteins]
    model.eval()
    with torch.inference_mode():
        for first in range(0, len(ids), batch_size):
            selected = ids[first:first + batch_size]
            batch, offsets = context_batch(data, selected, store, hidden_hashes=set(),
                                           length=model.encoder.config.max_context, context_index=index)
            values = model([x.to(device) for x in batch], offsets).sigmoid()
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite GO probabilities")
            yield selected, values.cpu().numpy()


def predict(task, output, model_dir, *, dataset=None, input_fasta=None, config=None,
            embeddings=None, esm_checkpoint=None, batch_size=8, device="cpu",
            namespaces=("BP", "CC", "MF"), progress=print, dataset_validator=None):
    """Run into a new directory; optional dataset validation precedes embedding/inference."""
    if task not in ("module", "go", "vf") or (dataset is None) == (input_fasta is None):
        raise ValueError("Choose module/go/vf and exactly one nucleotide input or native dataset")
    if type(batch_size) is not int or batch_size < 1 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid batch size or device")
    if input_fasta is not None and config is None:
        raise ValueError("Nucleotide input requires traditional annotation configuration")
    if embeddings is not None and esm_checkpoint is not None:
        raise ValueError("Choose existing embeddings or an ESM checkpoint")
    if not namespaces or len(set(namespaces)) != len(namespaces) or set(namespaces) - {"BP", "CC", "MF"}:
        raise ValueError("Choose unique GO namespaces BP, CC and/or MF")
    paths = {ns: Path(model_dir) / ns for ns in namespaces} if task == "go" else {task: Path(model_dir)}
    models = {}
    for name, path in paths.items():
        _, meta = bundles.metadata(path, task)
        if task == "go" and meta["namespace"] != name:
            raise ValueError("GO namespace directory disagrees with vocabulary")
        models[name] = meta
    out = Path(output).expanduser().absolute()
    if out.is_symlink() or out.exists():
        raise FileExistsError("Choose a new output directory; existing results are retained")
    data = Dataset.load(dataset) if dataset is not None else None
    store = EmbeddingStore(embeddings) if embeddings is not None else None
    if store is not None and data is not None:
        validate_embeddings(data, store)
    if store is None and (data is None or data.proteins):
        if esm_checkpoint is None or not Path(esm_checkpoint).is_file():
            raise ValueError("Provide --embeddings or a local --esm-model checkpoint")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", {"state": "running", "task": task})
    try:
        if data is None:
            prepare(input_fasta, out / "annotation", config, progress=progress)
            dataset = out / "annotation/prepared/dataset"
            data = Dataset.load(dataset)
        if dataset_validator is not None:
            dataset_validator(data)
        if data.proteins and store is None:
            progress("Computing full-length ESM2 for all native CDS")
            store = embed_esm([p.protein_id for p in data.proteins], [p.sequence for p in data.proteins],
                              out / "embeddings", ESMConfig(device=device, batch_size=batch_size,
                              full_length=True, checkpoint=str(esm_checkpoint)))
            _release_stage_memory(device)
        if store is not None:
            validate_embeddings(data, store)
        identity = {"task": task, "models": models, "device": device, "batch_size": batch_size,
                    "dataset": {p.name: digest_file(p) for p in sorted(Path(dataset).iterdir()) if p.is_file()},
                    "embedding_metadata_sha256": digest_file(store.directory / "embeddings.json") if store else None,
                    "code": {str(p.relative_to(Path(__file__).parents[1])): digest_file(p)
                             for p in sorted(Path(__file__).parents[1].rglob("*.py"))}}
        write_json(out / "manifest.json", identity)
        summary = dict(task=task, genomes=len(data.genomes), proteins=len(data.proteins), models=list(models))
        if task == "module":
            model, meta = bundles.load(paths["module"], task, device=device)
            pc = replace(PredictionConfig(**meta["prediction"]), batch_size=batch_size)
            totals = dict(candidates=0, accepted=0, span_qc_rejected=0)
            with (out / "modules.jsonl").open("x") as raw, (out / "genomes.jsonl").open("x") as coverage:
                def members():
                    for gid, rows, counts in module_rows(data, store, model, pc, device=device):
                        coverage.write(json.dumps(dict(genome_id=gid, **counts)) + "\n")
                        for row in rows:
                            raw.write(json.dumps(row, allow_nan=False) + "\n")
                            totals["candidates"] += 1
                            totals["accepted" if row["span_qc_passed"] else "span_qc_rejected"] += 1
                            if row["span_qc_passed"]:
                                for pid in row["members"]:
                                    yield {k: row[k] for k in ("module_id", "genome_id", "score", "auxiliary_type_bits")} | {"protein_id": pid}
                        progress(f"Module: {gid}, {counts['accepted']} accepted")
                write_table(out / "module_members.tsv", members(),
                            ["module_id", "genome_id", "protein_id", "score", "auxiliary_type_bits"])
            summary.update(totals, network_mode="off")
            del model
            _release_stage_memory(device)
        elif task == "vf":
            from .binary_context import probability_batches
            progress("Loading binary context model")
            model, meta = bundles.load(paths["vf"], task, device=device)
            by_id = {p.protein_id: p for p in data.proteins}
            positive = 0
            def scores():
                nonlocal positive
                for ids, values in probability_batches(data, store, model, batch_size=batch_size, device=device):
                    for pid, score in zip(ids, values):
                        assigned = bool(score >= meta["threshold"])
                        positive += int(assigned)
                        yield dict(protein_id=pid, genome_id=by_id[pid].genome_id,
                                   probability=float(score), predicted_label=int(assigned),
                                   threshold=meta["threshold"])
            write_table(out / "vf.tsv", scores(),
                        ["protein_id", "genome_id", "probability", "predicted_label", "threshold"])
            summary.update(predicted_positive=positive, threshold=meta["threshold"],
                           target_offset=14, annotation_mode="all_masked")
            del model
            _release_stage_memory(device)
        else:
            summary["namespaces"] = {}
            for ns, path in paths.items():
                progress(f"GO: loading {ns}")
                model, meta = bundles.load(path, task, device=device)
                terms = meta["terms"]
                count = 0
                def scores():
                    nonlocal count
                    for ids, values in go_batches(data, store, model, batch_size=batch_size, device=device):
                        for pid, values_row in zip(ids, values):
                            count += 1
                            yield {"protein_id": pid, "predicted_terms": ";".join(t for t, v in zip(terms, values_row) if v >= meta["threshold"]),
                                   **dict(zip(terms, map(float, values_row)))}
                write_table(out / f"go_{ns}.tsv", scores(), ["protein_id", "predicted_terms", *terms])
                summary["namespaces"][ns] = dict(proteins=count, terms=len(terms), threshold=meta["threshold"])
                del model
                _release_stage_memory(device)
        write_json(out / "summary.json", summary)
        files = {p.name: digest_file(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != "status.json"}
        write_json(out / "status.json", {"state": "complete", "task": task, "files": files})
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        write_json(out / "status.json", {"state": "failed", "task": task, "error": str(exc)})
        raise
