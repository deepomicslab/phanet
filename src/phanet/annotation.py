"""Prepare native CDS context using Prodigal and eggNOG-mapper."""
from __future__ import annotations

import gzip
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import quote

from Bio import SeqIO

from phanet import __version__
from phanet.common.constants import CATEGORIES
from phanet.common.io import digest_file, digest_object, read_json, write_json, write_table
from phanet.datasets.annotations import AnnotationMapper
from phanet.datasets.data import call_genes
from phanet.datasets.schema import Dataset


@dataclass(frozen=True)
class PrepareConfig:
    eggnog_data: str
    prodigal: str = "prodigal"
    emapper: tuple[str, ...] = ("emapper.py",)
    threads: int = 2
    diamond_block_size: float = 0.4
    max_evalue: float = 0.001

    def __post_init__(self):
        if not isinstance(self.eggnog_data, str) or not self.eggnog_data.strip():
            raise ValueError("Set eggnog_data to an installed eggNOG database directory")
        if not isinstance(self.prodigal, str) or not self.prodigal.strip():
            raise ValueError("prodigal must be an executable name or path")
        if (not isinstance(self.emapper, (tuple, list)) or not self.emapper or
                any(not isinstance(x, str) or not x for x in self.emapper)):
            raise ValueError("emapper must be a nonempty command argument list")
        object.__setattr__(self, "emapper", tuple(self.emapper))
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("threads must be a positive integer")
        for name in ("diamond_block_size", "max_evalue"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = read_json(path)
        if not isinstance(values, dict) or values.pop("schema_version", None) != 1:
            raise ValueError("Expected preparation configuration schema_version 1")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown preparation settings: {sorted(unknown)}")
        def resolve(value):
            p = Path(value).expanduser()
            return str(p if p.is_absolute() else path.parent / p)
        if values.get("eggnog_data"):
            values["eggnog_data"] = resolve(values["eggnog_data"])
        if isinstance(values.get("prodigal"), str) and "/" in values["prodigal"]:
            values["prodigal"] = resolve(values["prodigal"])
        if isinstance(values.get("emapper"), list):
            values["emapper"] = [resolve(x) if isinstance(x, str) and "/" in x else x
                                 for x in values["emapper"]]
        return cls(**values)


def _executable(name):
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(f"Required executable not found: {name}")
    return str(Path(found).resolve())


def _tool_env(argv):
    env = os.environ.copy()
    executable = Path(argv[0])
    if executable.is_absolute():
        env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    return env


def _probe(argv):
    try:
        # Some tools derive a version suffix from the caller's Git directory.
        # Probe outside any repository so moving the caller does not change identity.
        with tempfile.TemporaryDirectory(prefix="phanet-tool-version-") as cwd:
            result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=60, env=_tool_env(argv))
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Tool check timed out: {argv[0]}") from exc
    text = (result.stdout + result.stderr).strip()
    if result.returncode:
        raise RuntimeError(f"Tool check failed: {argv[0]}\n{text[-2000:]}")
    return text


def preflight(config: PrepareConfig):
    """Check configured tools and database files."""
    prodigal = _executable(config.prodigal)
    emapper = [_executable(config.emapper[0]), *config.emapper[1:]]
    root = Path(config.eggnog_data).expanduser().resolve()
    resources = {}
    for name in ("eggnog.db", "eggnog.taxa.db", "eggnog_proteins.dmnd"):
        p = root / name
        if not p.is_file() or p.stat().st_size == 0:
            raise FileNotFoundError(f"Required eggNOG database file missing or empty: {p}")
        resources[name] = {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
    # Track an optional installation manifest without repeatedly hashing a large database.
    manifest = root / "manifest.json"
    identity = {"directory": str(root), "files": resources,
                "manifest_sha256": digest_file(manifest) if manifest.is_file() else None,
                "setup_receipt_sha256": digest_file(root / "phanet-eggnog.json")
                if (root / "phanet-eggnog.json").is_file() else None}
    ev = _probe([*emapper, "--data_dir", str(root), "--version"])
    if "emapper-2.1." not in ev:
        raise ValueError("This adapter requires eggNOG-mapper 2.1.x")
    diamond = re.search(r"Diamond version found:\s*(.+)", ev, re.I)
    if not diamond or "diamond version" not in diamond.group(1).lower():
        raise RuntimeError("eggNOG-mapper could not locate DIAMOND")
    if "Installed eggNOG DB version: 5.0.2" not in ev:
        raise ValueError("This adapter currently supports eggNOG database 5.0.2")
    tools = {"prodigal": {"command": [prodigal], "version": _probe([prodigal, "-v"])},
             "emapper": {"command": emapper, "version": ev}}
    for info in tools.values():
        info["command_files"] = {str(Path(a).resolve()): digest_file(a) for a in info["command"] if Path(a).is_file()}
    return {"tools": tools, "database": identity}


def normalize_fasta(source, output):
    """Assign tool-compatible contig IDs and retain the input ID mapping."""
    source, output = Path(source), Path(output)
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    with opener(source, "rt") as stream:
        first = next((line for line in stream if line.strip()), "")
        if not first.startswith(">"):
            raise ValueError("Input must be a nonempty nucleotide FASTA")
    rows, seen = [], set()
    with opener(source, "rt") as stream, output.open("w") as out:
        for i, record in enumerate(SeqIO.parse(stream, "fasta"), 1):
            original = record.id
            seq = str(record.seq).upper()
            if not original or original in seen:
                raise ValueError(f"Empty or duplicate contig ID: {original}")
            if not seq or set(seq) - set("ACGTRYSWKMBDHVN"):
                raise ValueError(f"Invalid DNA sequence: {original}")
            seen.add(original)
            tool_id = f"contig{i:06d}"
            out.write(f">{tool_id}\n{seq}\n")
            rows.append({"tool_id": tool_id, "contig_id": original,
                         "description": record.description, "length_nt": len(seq)})
    if not rows:
        raise ValueError("Input must contain at least one contig")
    return rows


_UNKNOWN = re.compile(r"\b(?:hypothetical|uncharacteri[sz]ed|unknown|unassigned|unsorted)\b|\bDUF\d+\b", re.I)


def informative_product(text):
    """Conservative description qualification, separate from nine-class mapping."""
    text = text.strip()
    if not text or text in ("-", "NA", "N/A") or _UNKNOWN.search(text):
        return False
    normalized = re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()
    generic = {"protein", "phage protein", "putative protein", "predicted protein",
               "conserved protein", "conserved phage protein", "domain containing protein"}
    return normalized not in generic


def read_eggnog(path, protein_ids):
    """Read the documented header, rejecting truncated, duplicate and foreign rows."""
    allowed, result, fields = set(protein_ids), {}, None
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith("#query\t"):
                if fields is not None:
                    raise ValueError("Repeated eggNOG annotation header")
                fields = line.rstrip("\r\n").lstrip("#").split("\t")
                required = {"query", "seed_ortholog", "evalue", "score", "Description"}
                if len(set(fields)) != len(fields) or not required <= set(fields):
                    raise ValueError("Unsupported eggNOG annotation columns")
                continue
            if line.startswith("#") or not line.strip():
                continue
            values = line.rstrip("\r\n").split("\t")
            if fields is None or len(values) != len(fields):
                raise ValueError("Missing header or truncated eggNOG annotation row")
            row = dict(zip(fields, values))
            pid = row["query"]
            if pid not in allowed or pid in result:
                raise ValueError(f"Unknown or repeated eggNOG query: {pid}")
            for key in ("evalue", "score"):
                value = float(row[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid eggNOG {key}: {pid}")
            result[pid] = row
    if fields is None:
        raise ValueError("Missing eggNOG annotation header; output may be incomplete")
    return result


def _file_manifest(root):
    return {str(p.relative_to(root)): digest_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != "COMPLETE.json"}


def _verify_stage(root, fingerprint):
    receipt = read_json(root / "COMPLETE.json")
    if receipt.get("fingerprint") != fingerprint or receipt.get("files") != _file_manifest(root):
        raise ValueError(f"Stage inputs or outputs changed: {root}")


def _stage(root, fingerprint, action):
    if root.exists():
        if (root / "COMPLETE.json").is_file():
            _verify_stage(root, fingerprint)
            return
        raise FileExistsError(f"Incomplete stage retained at {root}; use a new output directory")
    root.mkdir()
    action(root)
    write_json(root / "COMPLETE.json", {"fingerprint": fingerprint, "files": _file_manifest(root)})


def prepare(input_fasta, output, config: PrepareConfig, *, resume=False, progress=None):
    """Run the default traditional frontend and export intact inference context."""
    source, out = Path(input_fasta).resolve(), Path(output).absolute()
    if out.is_symlink():
        raise ValueError("Output must not be a symlink")
    out = out.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_relative_to(out):
        raise ValueError("Input must be outside the output directory")
    emit = progress or (lambda message: None)
    emit("Checking Prodigal, eggNOG-mapper and database")
    environment = preflight(config)
    # Version and code identity prevent resume across incompatible implementations.
    code = {str(p.relative_to(Path(__file__).parent)): digest_file(p)
            for p in sorted(Path(__file__).parent.rglob("*.py"))}
    inputs = {"input_sha256": digest_file(source), "config": asdict(config),
              "environment": environment, "version": __version__, "code": code,
              "categories": list(CATEGORIES), "mapper": AnnotationMapper().fingerprint}
    fingerprint = digest_object(inputs)
    if out.exists():
        if not resume:
            raise FileExistsError(f"Output exists; use --resume or a new directory: {out}")
        if not (out / "RUN.json").is_file() or read_json(out / "RUN.json").get("fingerprint") != fingerprint:
            raise ValueError("Resume input/configuration/tools/database/code differ from the saved run")
    else:
        out.mkdir(parents=True)
    lock = out / ".lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        if not (out / "RUN.json").exists():
            write_json(out / "RUN.json", {"fingerprint": fingerprint, **inputs})
        write_json(out / "status.json", {"state": "running", "stage": "genes"})
        def genes(path):
            contigs = normalize_fasta(source, path / "input.fna")
            # Abort if an input changed between preflight and parsing.
            if digest_file(source) != inputs["input_sha256"]:
                raise ValueError("Input changed while reading")
            data = call_genes(path / "input.fna", path / "prodigal", executable=environment["tools"]["prodigal"]["command"][0])
            names = {row["tool_id"]: row["contig_id"] for row in contigs}
            for genome in data.genomes:
                genome.genome_id = names[genome.genome_id]
                genome.accession = genome.genome_id
                genome.source = "input.fasta"
            for protein in data.proteins:
                protein.genome_id = names[protein.genome_id]
                protein.source_protein_id = protein.protein_id
            data.provenance = {"gene_caller": "Prodigal", "mode": "meta", "input_sha256": inputs["input_sha256"]}
            data.save(path / "dataset")
            write_table(path / "contigs.tsv", contigs, ["tool_id", "contig_id", "description", "length_nt"])
        emit("Calling genes with Prodigal (meta)")
        _stage(out / "genes", fingerprint, genes)
        data = Dataset.load(out / "genes/dataset")
        def eggnog(path):
            if not data.proteins:
                write_json(path / "SKIPPED.json", {"reason": "no_coding_genes"})
                return
            argv = [*environment["tools"]["emapper"]["command"], "-i", str(out / "genes/dataset/proteins.faa"),
                    "--itype", "proteins", "-m", "diamond", "--cpu", str(config.threads),
                    "--data_dir", environment["database"]["directory"], "--output", "proteins", "--output_dir", str(path),
                    "--pfam_realign", "none", "--block_size", str(config.diamond_block_size),
                    "--evalue", str(config.max_evalue), "--seed_ortholog_evalue", str(config.max_evalue)]
            write_json(path / "command.json", {"argv": argv})
            with (path / "tool.log").open("w") as log:
                process = subprocess.run(argv, cwd=path, stdout=log, stderr=subprocess.STDOUT, env=_tool_env(argv))
            write_json(path / "exit.json", {"returncode": process.returncode})
            if process.returncode:
                raise RuntimeError(f"eggNOG-mapper failed ({process.returncode}); see {path / 'tool.log'}")
            read_eggnog(path / "proteins.emapper.annotations", [p.protein_id for p in data.proteins])
        write_json(out / "status.json", {"state": "running", "stage": "eggnog"})
        emit("Annotating proteins with eggNOG-mapper (DIAMOND)")
        _stage(out / "eggnog", fingerprint, eggnog)
        def export(path):
            annotations = read_eggnog(out / "eggnog/proteins.emapper.annotations", [p.protein_id for p in data.proteins]) if data.proteins else {}
            mapper = AnnotationMapper()
            rows, targets = [], []
            for p in data.proteins:
                hit = annotations.get(p.protein_id)
                product = hit["Description"].strip() if hit else ""
                qualified = bool(hit and hit["seed_ortholog"] not in ("", "-") and
                                 float(hit["evalue"]) <= config.max_evalue and informative_product(product))
                p.product = product
                p.labels = mapper.map(product) if qualified else ()
                p.evidence_source = "eggNOG-mapper" if hit else ""
                p.evidence_version = environment["tools"]["emapper"]["version"] if hit else ""
                reason = "" if qualified else "no_hit" if not hit else "weak_hit" if float(hit["evalue"]) > config.max_evalue else "unresolved_description"
                row = {"protein_id": p.protein_id, "contig_id": p.genome_id,
                       "start": p.start + 1, "end": p.end, "strand": "+" if p.strand == 1 else "-",
                       "partial": int(p.partial), "product": product, "source": p.evidence_source,
                       "traditional_status": "annotated" if qualified else "unresolved",
                       "labels": ";".join(p.labels), "mapping_status": "mapped" if p.labels else "unmapped" if qualified else "unresolved",
                       "needs_phanet": int(not qualified), "reason": reason,
                       "seed_ortholog": hit["seed_ortholog"] if hit else "", "evalue": hit["evalue"] if hit else "",
                       "sequence_sha256": p.sha256}
                rows.append(row)
                if not qualified:
                    targets.append({k: row[k] for k in ("protein_id", "contig_id", "sequence_sha256", "reason")})
            fields = ["protein_id", "contig_id", "start", "end", "strand", "partial", "product", "source", "traditional_status",
                      "labels", "mapping_status", "needs_phanet", "reason", "seed_ortholog", "evalue", "sequence_sha256"]
            write_table(path / "annotations.tsv", rows, fields)
            write_table(path / "targets.tsv", targets, ["protein_id", "contig_id", "sequence_sha256", "reason"])
            data.provenance.update(frontend="prodigal-eggnog-v1", mapper=mapper.fingerprint, categories=list(CATEGORIES))
            data.save(path / "dataset")
            shutil.copyfile(path / "dataset/proteins.faa", path / "proteins.faa")
            with (path / "annotations.gff3").open("w") as gff:
                gff.write("##gff-version 3\n")
                for g in data.genomes:
                    gff.write(f"##sequence-region {quote(g.genome_id, safe='')} 1 {g.length_nt}\n")
                for p, row in zip(data.proteins, rows):
                    attrs = {"ID": p.protein_id, "product": p.product or "hypothetical protein", "annotation_source": p.evidence_source or "none",
                             "phanet_target": str(bool(row['needs_phanet'])).lower(), "partial": str(p.partial).lower()}
                    encoded = ";".join(f"{k}={quote(v, safe='')}" for k, v in attrs.items())
                    gff.write(f"{quote(p.genome_id, safe='')}\tProdigal\tCDS\t{p.start + 1}\t{p.end}\t.\t{row['strand']}\t{p.frame}\t{encoded}\n")
            write_json(path / "summary.json", {"state": "prepared", "contigs": len(data.genomes), "proteins": len(data.proteins),
                       "traditional_annotated": len(rows) - len(targets), "pending_phanet": len(targets),
                       "no_coding_genes": sum(not v for v in data.by_genome().values()), "phanet_predictions_generated": False})
            write_json(path / "handoff.json", {"schema_version": 1, "state": "prepared", "categories": list(CATEGORIES),
                       "dataset": "dataset", "targets": "targets.tsv", "target_selection": "no_confident_product_v1",
                       "context": "all_proteins_within_each_original_contig", "table_coordinates": "1-based-inclusive",
                       "dataset_coordinates": "0-based-half-open", "reference_labels": "qualified_traditional_annotations_only",
                       "input_fingerprint": fingerprint})
        emit("Exporting traditional annotations, full context and PhaNet targets")
        write_json(out / "status.json", {"state": "running", "stage": "export"})
        _stage(out / "prepared", fingerprint, export)
        summary = read_json(out / "prepared/summary.json")
        write_json(out / "status.json", {"state": "prepared", "stage": "complete", "fingerprint": fingerprint,
                                         "phanet_predictions_generated": False})
        return summary
    except BaseException as exc:
        write_json(out / "status.json", {"state": "failed", "error": str(exc), "fingerprint": fingerprint})
        raise
    finally:
        lock.unlink()
