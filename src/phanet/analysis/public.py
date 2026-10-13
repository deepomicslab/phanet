"""Input adapters and orchestration for module prediction and catalogue analysis."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil

from phanet.common.io import digest_file, digest_object, read_json, read_table, write_json, write_table
from phanet.datasets.schema import Dataset
from .annotation_baseline import ProductTextBaseline
from .prediction_pipeline import CatalogueConfig, build_catalogue, export_catalogue, read_references, valid_sha
from .sequence_evidence import DiskAlignments
from .template_matching import ProfileHit, Relationships

SCHEMA = "phanet-module-analysis-v1"
RESOURCES = {"references", "baseline_policy", "clusters", "profiles", "sequence_alignments",
             "sequence_proteins", "baseline_decisions", "qualified_profiles"}
DATASET_FILES = {"dataset.json", "genomes.tsv", "genomes.fna", "proteins.tsv", "proteins.faa", "annotations.tsv"}
FATAL = {"invalid_coordinate", "frameshift", "severe_assembly_warning", "unsupported_gene_call"}


def load_config(path):
    path = Path(path).expanduser().resolve()
    return _validate_config(read_json(path), path.parent)


def _validate_config(cfg, base):
    """Validate before publishing a new configuration, using its real parent."""
    allowed = {"schema", "resources", "reference_scope", "cluster_scope", "profile_scope", "sequence_scope",
               "reference_format", "catalogue", "max_candidates"}
    if not isinstance(cfg, dict) or set(cfg) - allowed or cfg.get("schema") != SCHEMA:
        raise ValueError("Invalid module analysis configuration/schema")
    for key in ("reference_scope", "cluster_scope", *[k for k in ("profile_scope", "sequence_scope") if k in cfg]):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"Explicit {key} is required")
    resources = cfg.get("resources", {})
    if not isinstance(resources, dict) or set(resources) - RESOURCES or not {"references", "baseline_policy"} <= resources.keys():
        raise ValueError("Bind reference and annotation baseline resources")
    paths = {}
    for key, binding in resources.items():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"} or not valid_sha(binding["sha256"]):
            raise ValueError(f"Invalid resource binding: {key}")
        if not isinstance(binding["path"], str) or not binding["path"]:
            raise ValueError(f"Invalid resource path: {key}")
        resolved = (base / Path(binding["path"]).expanduser()).resolve()
        if digest_file(resolved) != binding["sha256"]:
            raise ValueError(f"Resource SHA256 mismatch: {key}")
        paths[key] = resolved
    if not paths["references"].stat().st_size:
        raise ValueError("Empty known reference input")
    ProductTextBaseline(read_json(paths["baseline_policy"]))
    if "profiles" in paths and "qualified_profiles" in paths:
        raise ValueError("Choose scored profiles or prequalified profile assignments")
    if ("sequence_alignments" in paths) != ("sequence_proteins" in paths):
        raise ValueError("Sequence alignments and their protein universe must be supplied together")
    for resource, scope in (("profiles", "profile_scope"), ("qualified_profiles", "profile_scope"), ("sequence_alignments", "sequence_scope")):
        if resource in paths and (not isinstance(cfg.get(scope), str) or not cfg[scope].strip()):
            raise ValueError(f"Explicit {scope} is required")
    if cfg.get("reference_format", "native_members") not in ("native_members", "literature_targets"):
        raise ValueError("Use portable native_members or literature_targets references")
    budget = cfg.get("max_candidates", 1000000)
    if type(budget) is not int or budget < 1:
        raise ValueError("Positive candidate budget required")
    catalogue = CatalogueConfig(**cfg.get("catalogue", {}))
    return cfg, paths, catalogue


def init_config(output, *, references, reference_scope, cluster_scope, baseline_policy=None,
                reference_format="native_members", clusters=None, profiles=None, qualified_profiles=None, profile_scope=None,
                sequence_alignments=None, sequence_proteins=None, sequence_scope=None, baseline_decisions=None):
    """Write a configuration binding local reference and annotation resources."""
    out = Path(output).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Choose a new analysis configuration")
    out = out.parent.resolve() / out.name
    if baseline_policy is None:
        baseline_policy = Path(__file__).parent / "policies/product-text-v3.json"
    values = locals()
    bindings = {}
    for name in sorted(RESOURCES):
        if values.get(name) is not None:
            path = Path(values[name]).expanduser().resolve()
            bindings[name] = dict(path=os.path.relpath(path, out.parent), sha256=digest_file(path))
    cfg = dict(schema=SCHEMA, resources=bindings, reference_scope=reference_scope,
               cluster_scope=cluster_scope, reference_format=reference_format,
               catalogue=asdict(CatalogueConfig()), max_candidates=1000000)
    if profile_scope is not None:
        cfg["profile_scope"] = profile_scope
    if sequence_alignments is not None:
        cfg["sequence_scope"] = sequence_scope
    _validate_config(cfg, out.parent)
    payload = json.dumps(cfg, indent=2, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        stream.write(payload)
    return cfg


def load_loci(data, cfg, paths):
    policy = ProductTextBaseline(read_json(paths["baseline_policy"]))
    clusters = {g.genome_id: g.cluster_id for g in data.genomes}
    if "clusters" in paths:
        assigned = {}
        for row in read_table(paths["clusters"], ("genome_id", "cluster_id")):
            gid, cluster = row["genome_id"], row["cluster_id"]
            if gid in assigned or gid not in clusters or not cluster.strip():
                raise ValueError("Invalid/duplicate genome cluster assignment")
            if clusters[gid] and clusters[gid] != cluster:
                raise ValueError("Dataset and supplied cluster assignments disagree")
            assigned[gid] = cluster
        if assigned.keys() != clusters.keys():
            raise ValueError("Cluster table must cover exactly all input genomes")
        clusters = assigned
    if any(not isinstance(c, str) or not c.strip() for c in clusters.values()):
        raise ValueError("Every genome needs a cluster_id; do not count unclustered genomes as independent support")
    decisions = {}
    if "baseline_decisions" in paths:
        for row in json_rows(paths["baseline_decisions"]):
            pid = row.get("protein_id")
            if not isinstance(pid, str) or pid in decisions or type(row.get("unresolved")) is not bool:
                raise ValueError("Invalid/duplicate baseline decision")
            if not row.get("reason") or row.get("rule_id") != policy.rule_id or not valid_sha(row.get("protein_sha256")):
                raise ValueError("Baseline decisions require bound identities, reasons and policy rule_id")
            decisions[pid] = row
        if decisions.keys() != {p.protein_id for p in data.proteins}:
            raise ValueError("Baseline decisions must cover exactly all native proteins")
    loci = {}
    for p in data.proteins:
        assessed = policy.assess(p.product)
        if decisions:
            row = decisions[p.protein_id]
            if row["protein_sha256"] != p.sha256:
                raise ValueError("Baseline decision protein identity mismatch")
            unresolved, reason = row["unresolved"], row["reason"]
        else:
            unresolved, reason = assessed.unresolved, assessed.reason
        loci[p.protein_id] = dict(locus_uid=p.protein_id, genome=p.genome_id,
            cluster=clusters[p.genome_id], native_rank=p.order, protein_id=p.protein_id,
            protein_sha256=p.sha256, start=p.start, end=p.end, strand=p.strand,
            qualified_profiles=[], baseline_unresolved=unresolved, baseline_reason=reason,
            baseline_rule_id=policy.rule_id, product=p.product,
            annotation_source=p.evidence_source, annotation_version=p.evidence_version)
    return loci, clusters


def json_rows(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object at line {number}")
            yield row


def verified_prediction(predictions, dataset):
    root = Path(predictions)
    status = read_json(root / "status.json")
    required = {"manifest.json", "summary.json", "modules.jsonl", "genomes.jsonl", "module_members.tsv"}
    if not isinstance(status, dict):
        raise ValueError("Invalid prediction completion receipt")
    files = status.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Invalid prediction file manifest")
    if status.get("state") != "complete" or status.get("task") != "module" or not required <= files.keys():
        raise ValueError("A completed predict-modules result with its integrity receipt is required")
    for name, sha in files.items():
        if Path(name).name != name or not valid_sha(sha) or digest_file(root / name) != sha:
            raise ValueError("Prediction output integrity failure")
    manifest = read_json(root / "manifest.json")
    actual = {p.name: digest_file(p) for p in Path(dataset).iterdir() if p.is_file()}
    if not DATASET_FILES <= actual.keys() or manifest.get("dataset") != actual or manifest.get("task") != "module":
        raise ValueError("Prediction and native dataset identity mismatch")
    if manifest.get("models", {}).get("module", {}).get("network_mode") not in (None, "off"):
        raise ValueError("Only the accepted network-off module model is supported")
    return {"status_sha256": digest_file(root / "status.json"), "dataset": actual,
            "prediction_files": files}


def load_candidates(root, data, loci, clusters, budget):
    groups = data.by_genome()
    seen, candidates = set(), []
    counts = {gid: {"candidates": 0, "accepted": 0, "span_qc_rejected": 0} for gid in groups}
    for row in json_rows(Path(root) / "modules.jsonl"):
        gid, members, ranks = row.get("genome_id"), row.get("members"), row.get("member_ranks")
        if gid not in groups or not isinstance(members, list) or not 2 <= len(members) <= 30:
            raise ValueError("Invalid module genome/member set")
        if (not isinstance(ranks, list) or len(ranks) != len(members) or
                any(type(i) is not int or not 0 <= i < len(groups[gid]) for i in ranks) or
                ranks != sorted(set(ranks)) or ranks[-1] - ranks[0] >= 30):
            raise ValueError("Invalid native module ranks/span")
        if members != [groups[gid][i].protein_id for i in ranks]:
            raise ValueError("Module members do not match native ranks")
        identifier = digest_object([gid, members])
        if row.get("module_id") != identifier or identifier in seen:
            raise ValueError("Invalid or duplicate module identity")
        seen.add(identifier)
        if len(seen) > budget:
            raise RuntimeError("Candidate budget exceeded; no truncation")
        span = groups[gid][ranks[0]:ranks[-1] + 1]
        passed = all(not p.partial and not set(p.quality_flags) & FATAL for p in span)
        if (type(row.get("span_qc_passed")) is not bool or row["span_qc_passed"] != passed or
                row.get("start") != min(p.start for p in span) or row.get("end") != max(p.end for p in span)):
            raise ValueError("Module span QC/coordinates disagree with native input")
        covering = max(0, min(ranks[0], max(0, len(groups[gid]) - 30)) - max(0, ranks[-1] - 29) + 1)
        support, score, fraction, types = (row.get(k) for k in
            ("supporting_windows", "score", "window_support", "auxiliary_type_bits"))
        if (type(support) is not int or not 0 < support <= covering or
                type(row.get("covering_windows")) is not int or row["covering_windows"] != covering or
                type(types) is not int or not 0 <= types <= 127 or
                not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 or
                not isinstance(fraction, (int, float)) or not math.isclose(fraction, support / covering, abs_tol=1e-12)):
            raise ValueError("Invalid module score/window support/type bits")
        counts[gid]["candidates"] += 1
        counts[gid]["accepted" if passed else "span_qc_rejected"] += 1
        if passed:
            candidates.append(dict(instance_id=identifier, genome=gid, cluster=clusters[gid], members=members,
                **{k: row[k] for k in ("score", "supporting_windows", "covering_windows", "window_support", "auxiliary_type_bits")}))
    coverage = {}
    for row in json_rows(Path(root) / "genomes.jsonl"):
        gid = row.get("genome_id")
        if gid not in groups or gid in coverage:
            raise ValueError("Unknown or duplicate genome coverage record")
        if (row.get("proteins") != len(groups[gid]) or row.get("windows") != (max(1, len(groups[gid])-29) if groups[gid] else 0) or
                any(row.get(k) != counts[gid][k] for k in ("accepted", "span_qc_rejected")) or
                row.get("retained") != counts[gid]["candidates"]):
            raise ValueError("Genome prediction coverage mismatch")
        coverage[gid] = row
    if coverage.keys() != groups.keys():
        raise ValueError("Missing input genomes in prediction coverage")
    summary = read_json(Path(root) / "summary.json")
    if (summary.get("genomes") != len(groups) or summary.get("proteins") != len(data.proteins) or
            any(summary.get(k) != sum(v[k] for v in counts.values()) for k in ("candidates", "accepted", "span_qc_rejected"))):
        raise ValueError("Prediction summary count mismatch")
    return candidates, counts


def load_relationships(paths, cfg, out, target_proteins):
    relationships = Relationships()
    profile_rows = 0
    if "profiles" in paths:
        seen = set()
        for row in read_table(paths["profiles"], ("protein_sha256", "profile", "evalue", "coverage")):
            protein, profile = row["protein_sha256"], row["profile"]
            if not valid_sha(protein) or not profile.strip() or profile.upper() in {"UNKNOWN", "UNMAPPED"} or (protein, profile) in seen:
                raise ValueError("Invalid or duplicate scored profile identity")
            seen.add((protein, profile))
            hit = ProfileHit(profile, float(row["evalue"]), float(row["coverage"]))
            current = relationships.profiles.setdefault(protein, set())
            if hit.eligible:
                current.add(profile)
            profile_rows += 1
    if "qualified_profiles" in paths:
        for row in read_table(paths["qualified_profiles"], ("protein_sha256", "qualified_profiles")):
            protein = row["protein_sha256"]
            profiles = json.loads(row["qualified_profiles"])
            if (not valid_sha(protein) or protein in relationships.profiles or not isinstance(profiles, list) or
                    any(not isinstance(p, str) or not p.strip() or p.upper() in {"UNKNOWN", "UNMAPPED"} for p in profiles)):
                raise ValueError("Invalid or duplicate prequalified profile assignment")
            relationships.profiles[protein] = frozenset(profiles)
            profile_rows += 1
    sequence_rows, universe = 0, []
    if "sequence_alignments" in paths:
        universe = read_json(paths["sequence_proteins"])
        if (not isinstance(universe, list) or any(not valid_sha(p) for p in universe) or
                len(set(universe)) != len(universe) or not universe):
            raise ValueError("Invalid sequence protein universe")
        with DiskAlignments(paths["sequence_alignments"], universe, out / "alignments.sqlite") as alignments:
            sequence_rows = alignments.count
            # Validate the entire evidence, then retain only useful target edges.
            # All source proteins remain eligible, including reference-only ones.
            for (q, t), hit in alignments.items():
                if t in target_proteins and hit.eligible:
                    relationships.sequences[q, t] = hit
    references = read_references(paths["references"], relationships,
                                  format=cfg.get("reference_format", "native_members"))
    if any(m["qualified_profiles"] for r in references for m in r["members"]) and not cfg.get("profile_scope"):
        raise ValueError("Qualified reference profiles require an explicit profile_scope")
    return relationships, references, dict(profile_rows=profile_rows, sequence_rows=sequence_rows,
        sequence_proteins=len(universe), retained_directed_edges=len(relationships.sequences),
        predicted_proteins_in_sequence_universe=len(target_proteins & set(universe)),
        reference_proteins_in_sequence_universe=len({m["protein_sha256"] for r in references for m in r["members"]} & set(universe)))


def analyze(dataset, predictions, config, output, *, progress=print):
    """Match and group predicted member sets on CPU."""
    cfg, paths, catalogue = load_config(config)
    out = Path(output).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Choose a new analysis output directory")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", dict(state="running", task="module-analysis"))
    try:
        input_identity = verified_prediction(predictions, dataset)
        data = Dataset.load(dataset)
        loci, clusters = load_loci(data, cfg, paths)
        candidates, counts = load_candidates(predictions, data, loci, clusters, cfg.get("max_candidates", 1000000))
        progress(f"Module analysis: {len(candidates)} QC-passed predictions, loading reference evidence")
        target_proteins = {loci[pid]["protein_sha256"] for c in candidates for pid in c["members"]}
        relationships, references, evidence_counts = load_relationships(paths, cfg, out, target_proteins)
        for local in loci.values():
            local["qualified_profiles"] = sorted(relationships.profiles.get(local["protein_sha256"], ()))
        source = Path(__file__).parents[1]
        source_files = sorted(source.rglob("*.py"))
        code = {str(p.relative_to(source)): digest_file(p) for p in source_files}
        for p in source_files:
            dest = out / "source/phanet" / p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest)
        write_json(out / "manifest.json", dict(schema=SCHEMA, inputs=input_identity,
            config_sha256=digest_file(config), resources=cfg["resources"],
            resolved_resources={k: str(v) for k, v in paths.items()}, code=code))
        write_json(out / "config.json", cfg)
        write_json(out / "baseline-policy.json", read_json(paths["baseline_policy"]))
        shutil.copyfile(Path(predictions) / "modules.jsonl", out / "predictions.jsonl")
        progress("Module analysis: matching complete sets and computing cluster-weighted annotation reports")
        result = build_catalogue(candidates, loci, references, relationships, catalogue)
        export_catalogue(result, loci, out)
        with (out / "reference-view.jsonl").open("x") as stream:
            for row in references:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        write_table(out / "genomes.tsv", (dict(genome_id=gid, cluster_id=clusters[gid], **value)
                    for gid, value in counts.items()), ["genome_id", "cluster_id", "candidates", "accepted", "span_qc_rejected"])
        report = dict(result["counts"], schema=SCHEMA, inference_executed=False,
            genomes=len(data.genomes), proteins=len(data.proteins), genome_clusters=len(set(clusters.values())),
            reference_scope=cfg["reference_scope"], cluster_scope=cfg["cluster_scope"],
            profile_scope=cfg.get("profile_scope"), sequence_scope=cfg.get("sequence_scope"),
            evidence=evidence_counts, span_qc_rejected=sum(v["span_qc_rejected"] for v in counts.values()),
            interpretation="Annotation gaps in predicted member sets relative to the supplied references.")
        write_json(out / "report.json", report)
        # Verify inputs again before writing the completion manifest.
        if verified_prediction(predictions, dataset) != input_identity or load_config(config)[0] != cfg:
            raise ValueError("Inputs changed during module analysis")
        files = {str(p.relative_to(out)): digest_file(p) for p in sorted(out.rglob("*"))
                 if p.is_file() and p != out / "status.json"}
        write_json(out / "status.json", dict(state="complete", task="module-analysis", files=files))
        return report
    except (Exception, KeyboardInterrupt) as exc:
        write_json(out / "status.json", dict(state="failed", task="module-analysis", error=str(exc)))
        raise


def discover(output, model_dir, analysis_config, *, dataset=None, input_fasta=None,
             config=None, embeddings=None, esm_checkpoint=None, batch_size=8, device="cpu", progress=print):
    """Run prediction, release its model, then run the CPU catalogue stage."""
    from phanet.downstream.predict import predict
    cfg, paths, _ = load_config(analysis_config)
    if dataset is not None:
        load_loci(Dataset.load(dataset), cfg, paths)
    out = Path(output).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError("Choose a new pipeline output directory")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "status.json", dict(state="running", task="discover-modules", stage="prediction"))
    try:
        prediction = predict("module", out / "prediction", model_dir, dataset=dataset,
            input_fasta=input_fasta, config=config, embeddings=embeddings,
            esm_checkpoint=esm_checkpoint, batch_size=batch_size, device=device, progress=progress,
            dataset_validator=lambda data: load_loci(data, cfg, paths))
        # predict() has already released the model and embedding objects.
        write_json(out / "status.json", dict(state="running", task="discover-modules", stage="analysis"))
        native = dataset if dataset is not None else out / "prediction/annotation/prepared/dataset"
        report = analyze(native, out / "prediction", analysis_config, out / "analysis", progress=progress)
        summary = dict(prediction=prediction, analysis=report)
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(state="complete", task="discover-modules", files={
            name: digest_file(out / name) for name in ("summary.json", "prediction/status.json", "analysis/status.json")}))
        return summary
    except (Exception, KeyboardInterrupt) as exc:
        write_json(out / "status.json", dict(state="failed", task="discover-modules", error=str(exc)))
        raise
