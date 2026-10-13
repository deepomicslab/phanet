"""FASTA import and Prodigal gene prediction."""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from Bio import SeqIO
from Bio.SeqFeature import ExactPosition

from phanet.datasets.annotations import AnnotationMapper
from phanet.datasets.schema import Dataset, Genome, Protein, sequence_hash
from phanet.common.io import digest_file, read_table, unique_index, write_json


def normalized_accession(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^(?:ref|gb|emb|dbj)\|", "", value, flags=re.I).strip("|")
    return re.sub(r"\.\d+$", "", value)


def normalize_protein(sequence: str) -> str:
    sequence = "".join(sequence.split()).upper().rstrip("*")
    if not sequence or set(sequence) - set("ACDEFGHIKLMNPQRSTVWYBXZJUO"):
        raise ValueError("Empty/invalid protein or internal stop codon")
    return sequence


def import_genbank(paths: list[str | Path], *, max_records: int | None = None,
                   mapper: AnnotationMapper | None = None) -> Dataset:
    mapper = mapper or AnnotationMapper()
    genomes, proteins, audit = [], [], []
    seen_records = {}
    for path in paths:
        path = Path(path)
        for record in SeqIO.parse(path, "genbank"):
            sequence = str(record.seq).upper()
            if not sequence or set(sequence) - set("ACGTURYSWKMBDHVN"):
                raise ValueError(f"Invalid nucleotide sequence: {record.id}")
            hash_value = sequence_hash(sequence)
            if record.id in seen_records:
                if seen_records[record.id] != hash_value:
                    raise ValueError(f"Conflicting sequence for accession {record.id}")
                audit.append({"record": record.id, "reason": "duplicate_accession_sequence", "source": str(path)})
                continue
            seen_records[record.id] = hash_value
            genome_id = record.id
            complete = "complete genome" in record.description.casefold()
            genome = Genome(genome_id, record.id, len(sequence), str(path.resolve()), complete=complete,
                            quality="complete" if complete else "unknown",
                            taxonomy=";".join(record.annotations.get("taxonomy", [])),
                            release_date=record.annotations.get("date", ""),
                            gc=(sequence.count("G") + sequence.count("C")) / len(sequence), sequence=sequence)
            for feature in record.features:
                if feature.type == "source":
                    genome.host = feature.qualifiers.get("host", [""])[0]
            local = []
            for feature_index, feature in enumerate(record.features):
                if feature.type != "CDS" or feature.location is None:
                    continue
                if "pseudo" in feature.qualifiers or "pseudogene" in feature.qualifiers:
                    audit.append({"record": record.id, "feature": feature_index, "reason": "pseudogene"})
                    continue
                try:
                    if "translation" in feature.qualifiers:
                        aa = normalize_protein(feature.qualifiers["translation"][0])
                    else:
                        offset = int(feature.qualifiers.get("codon_start", [1])[0]) - 1
                        table = int(feature.qualifiers.get("transl_table", [11])[0])
                        cds = feature.extract(record.seq)[offset:]
                        cds = cds[:len(cds) // 3 * 3]
                        aa = normalize_protein(str(cds.translate(table=table)))
                except (ValueError, KeyError) as exc:
                    audit.append({"record": record.id, "feature": feature_index, "reason": str(exc)})
                    continue
                strand = feature.location.strand
                if strand not in (-1, 1):
                    audit.append({"record": record.id, "feature": feature_index, "reason": "unknown_strand"})
                    continue
                parts = tuple((int(p.start), int(p.end)) for p in feature.location.parts)
                partial = not all(isinstance(p.start, ExactPosition) and isinstance(p.end, ExactPosition)
                                  for p in feature.location.parts)
                product = feature.qualifiers.get("product", [""])[0]
                protein = Protein(f"{genome_id}:cds:{feature_index}", genome_id,
                                  int(feature.location.start), int(feature.location.end), strand, 0, aa,
                                  source_protein_id=feature.qualifiers.get("protein_id", [""])[0],
                                  product=product, labels=mapper.map(product), evidence_source="GenBank product mapping",
                                  evidence_version=mapper.fingerprint, partial=partial, blocks=parts,
                                  frame=int(feature.qualifiers.get("codon_start", [1])[0]) - 1,
                                  quality_flags=("compound_cds",) if len(parts) > 1 else ())
                local.append(protein)
            local.sort(key=lambda p: (p.start, p.end, p.protein_id))
            for i, protein in enumerate(local):
                protein.order = i
            genomes.append(genome)
            proteins.extend(local)
            if max_records is not None and len(genomes) >= max_records:
                break
        if max_records is not None and len(genomes) >= max_records:
            break
    result = Dataset(genomes, proteins, {"import": "genbank", "annotation_mapper": mapper.fingerprint,
                                      "audit": audit, "sources": [str(Path(p).resolve()) for p in paths]})
    result.validate()
    return result


def import_proteins(fasta: str | Path, genes_tsv: str | Path, genomes_tsv: str | Path,
                    mapper: AnnotationMapper | None = None) -> Dataset:
    """Read protein FASTA with explicit genome and coordinate metadata."""
    mapper = mapper or AnnotationMapper()
    seqs = list(SeqIO.parse(fasta, "fasta"))
    unique_index([r.id for r in seqs])
    sequences = {r.id: normalize_protein(str(r.seq)) for r in seqs}
    genomes = []
    for row in read_table(genomes_tsv, ("genome_id", "length_nt")):
        genomes.append(Genome(row["genome_id"], row.get("accession", row["genome_id"]),
                              int(row["length_nt"]), row.get("source", str(genomes_tsv)),
                              complete=row.get("complete") == "1", cluster_id=row.get("cluster_id", ""),
                              quality=row.get("quality", "unknown"), taxonomy=row.get("taxonomy", ""),
                              host=row.get("host", ""), gc=float(row["gc"]) if row.get("gc") else None,
                              lifestyle=row.get("lifestyle", "")))
    proteins = []
    for row in read_table(genes_tsv, ("protein_id", "genome_id", "start", "end", "strand", "order")):
        identifier = row["protein_id"]
        if identifier not in sequences:
            raise ValueError(f"Gene metadata references a protein absent from FASTA: {identifier}")
        product = row.get("product", "")
        strand = {"+": 1, "-": -1, "1": 1, "-1": -1}.get(row["strand"])
        if strand is None:
            raise ValueError(f"Unknown strand: {identifier}")
        labels = tuple(filter(None, row.get("labels", "").split(";"))) if "labels" in row else mapper.map(product)
        proteins.append(Protein(identifier, row["genome_id"], int(row["start"]), int(row["end"]),
                                strand, int(row["order"]), sequences[identifier], product=product,
                                labels=labels, evidence_source=row.get("evidence_source", "imported annotations"),
                                evidence_version=row.get("evidence_version", mapper.fingerprint),
                                family_ids=tuple(filter(None, row.get("family_ids", "").split(";")))))
    if set(sequences) != {p.protein_id for p in proteins}:
        raise ValueError("FASTA IDs and gene metadata IDs must match exactly")
    result = Dataset(genomes, proteins, {"import": "protein_fasta_with_coordinates"})
    result.validate()
    return result


def run_tool(argv: list[str], *, cwd: str | Path, timeout: int | None = None) -> dict:
    """Run a configured scientific tool with an argument list."""
    if timeout is not None and timeout <= 0:
        raise ValueError("External-tool timeout must be positive or None")
    executable = shutil.which(argv[0])
    if executable is None:
        raise FileNotFoundError(f"Required external tool is unavailable: {argv[0]}")
    result = subprocess.run([executable, *map(str, argv[1:])], cwd=cwd, capture_output=True,
                            text=True, timeout=timeout)
    record = {"argv": [executable, *map(str, argv[1:])], "returncode": result.returncode,
              "stdout": result.stdout, "stderr": result.stderr}
    if result.returncode:
        raise RuntimeError(f"{argv[0]} failed ({result.returncode}): {result.stderr[-3000:]}")
    return record


def call_genes(fasta: str | Path, output: str | Path, *, mode: str = "meta", executable="prodigal") -> Dataset:
    if mode not in ("single", "meta"):
        raise ValueError("Prodigal mode must be single or meta")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(fasta).resolve()
    records = list(SeqIO.parse(source, "fasta"))
    unique_index([r.id for r in records])
    for record in records:
        if not record.seq or set(str(record.seq).upper()) - set("ACGTURYSWKMBDHVN"):
            raise ValueError(f"Invalid nucleotide sequence: {record.id}")
    execution = run_tool([executable, "-i", str(source), "-a", str(output / "orfs.faa"),
                          "-d", str(output / "orfs.fna"), "-o", str(output / "orfs.gff"),
                          "-f", "gff", "-p", mode, "-q"], cwd=output)
    write_json(output / "prodigal_execution.json", execution)
    genomes = [Genome(r.id, r.id, len(r.seq), str(source), complete=False,
                      quality="unassessed", sequence=str(r.seq).upper()) for r in records]
    genes_by_internal_id = {}
    with (output / "orfs.gff").open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            values = line.rstrip("\n").split("\t")
            if len(values) != 9 or values[2] != "CDS":
                continue
            attr = dict(item.split("=", 1) for item in values[8].split(";") if "=" in item)
            genes_by_internal_id[attr["ID"]] = (values, attr)
    proteins = []
    for aa in SeqIO.parse(output / "orfs.faa", "fasta"):
        match = re.search(r"(?:^|[; ])ID=([^; ]+)", aa.description)
        if not match or match.group(1) not in genes_by_internal_id:
            raise ValueError(f"Cannot align Prodigal GFF and FASTA: {aa.id}")
        values, attr = genes_by_internal_id[match.group(1)]
        proteins.append(Protein(aa.id, values[0], int(values[3]) - 1, int(values[4]),
                                1 if values[6] == "+" else -1, 0, normalize_protein(str(aa.seq)),
                                evidence_source="Prodigal", partial=attr.get("partial", "00") != "00",
                                coding_score=float(attr["cscore"]) if "cscore" in attr else None,
                                frame=int(values[7]) if values[7] != "." else 0))
    grouped = {g.genome_id: [] for g in genomes}
    for p in proteins:
        grouped[p.genome_id].append(p)
    for group in grouped.values():
        for order, p in enumerate(sorted(group, key=lambda p: (p.start, p.end, p.protein_id))):
            p.order = order
    result = Dataset(genomes, proteins, {"import": "prodigal", "mode": mode, "source_hash": digest_file(source)})
    result.validate()
    return result


def qc_report(dataset: Dataset, minimum_length: int = 2000) -> list[dict]:
    report = []
    by_genome = dataset.by_genome()
    for genome in dataset.genomes:
        proteins = by_genome[genome.genome_id]
        reasons = []
        if genome.length_nt < minimum_length:
            reasons.append("below_minimum_length")
        if not proteins:
            reasons.append("no_coding_genes")
        if genome.quality in ("contaminated", "low"):
            reasons.append("low_quality")
        reasons.extend(sorted(set(genome.quality_flags) & {"synthetic_construct", "cloning_vector", "non_phage",
                                                           "severe_contamination", "severe_assembly_warning"}))
        report.append({"genome_id": genome.genome_id, "eligible": not reasons, "reasons": reasons})
    return report
