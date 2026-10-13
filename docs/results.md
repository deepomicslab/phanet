# Output files

After the Quick start, results are in `lambda-analysis/annotation/results/`.
Run the examples below from the PhaNet source directory.

| File | Contents |
|---|---|
| `results/annotations.tsv` | Protein annotations and scores |
| `results/annotations.gff3` | CDS coordinates and annotations for genome browsers |
| `results/proteins.faa` | All predicted protein sequences |
| `results/summary.json` | Counts, selected method and run state |
| `traditional/prepared/targets.tsv` | Proteins selected for supplementation |
| `traditional/genes/contigs.tsv` | Input contig names and corresponding tool IDs |
| `traditional/eggnog/proteins.emapper.annotations` | Raw eggNOG-mapper annotations |
| `network/evidence.json` | Reference hits and network evidence in network/combined mode |

## View the annotations

```bash
python - <<'PYCODE'
import csv
with open("lambda-analysis/annotation/results/annotations.tsv") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
for row in rows[:5]:
    print(row["protein_id"], row["product"], row["final_labels"], sep="\t")
PYCODE
```

The source of each assignment is in `final_source`; PhaNet's additional labels
are in `phanet_labels`. The original product text remains in `product`.

A selection of rows from the Lambda example (`annotations.tsv`):

| `protein_id` | `product` | `final_labels` | `final_source` |
|---|---|---|---|
| `contig000001_1` | `DNA packaging` | `Packaging` | `eggNOG-mapper` |
| `contig000001_26` | `-` | `Packaging` | `PhaNet-GLM` |
| `contig000001_32` | `Protein of unknown function (DUF1382)` | `Infection` | `PhaNet-GLM` |
| `contig000001_25` | `-` | — | `unresolved` |

## Annotation table

| Column | Meaning |
|---|---|
| `protein_id`, `contig_id` | Protein ID and input contig ID |
| `start`, `end`, `strand` | 1-based inclusive coordinates and strand |
| `product` | Traditional product description |
| `traditional_status` | `annotated` or `unresolved` |
| `labels`, `mapping_status` | Functional labels mapped from the description |
| `needs_phanet`, `reason` | Whether supplementation was requested and its reason |
| `phanet_method`, `phanet_labels` | Prediction method and assigned labels |
| `final_labels`, `final_source` | Final labels and annotation source |
| `score_*` | Scores from the selected method |
| `glm_*`, `network_*` | Branch scores in combined output |
| `network_weight_*`, `network_route` | Network evidence weights and route |

Multiple labels use semicolons. Blank scores indicate proteins that were not
submitted for prediction. Coordinates refer to the submitted contig or fragment.
Prepared dataset coordinates use 0-based, half-open intervals. Protein IDs such
as `contig000001_1` are generated during gene prediction; `contig_id` retains the
input FASTA ID. Use `protein_id` to join the annotation, module, GO and VF tables.

## Summary

| Field | Meaning |
|---|---|
| `proteins` | Number of predicted proteins |
| `traditional_annotated` | Proteins resolved by traditional annotation |
| `phanet_targets` | Proteins submitted for supplementation |
| `phanet_assigned` | Targets assigned at least one label |
| `remaining_unresolved` | Targets with no assigned label |
| `network_used` | Whether the run used network inference |

## Intermediate files

GLM output is stored in `glm/probabilities.npy` with row IDs in `glm/targets.json`.
Network output is stored in `network/predictions.npz` with IDs in
`network/targets.json`. `embeddings/` stores reusable protein vectors.
Keep these files when using `--resume` or the [downstream commands](downstream.md).

## Export proteins with additional annotations

This writes a TSV and a protein FASTA to `lambda-analysis/exports/`:

```bash
python - <<'PYCODE'
import csv
from pathlib import Path
from Bio import SeqIO

root = Path("lambda-analysis")
out = root / "exports"
out.mkdir(exist_ok=True)
with (root / "annotation/results/annotations.tsv").open() as stream:
    reader = csv.DictReader(stream, delimiter="\t")
    fields = reader.fieldnames
    selected = [row for row in reader if row["phanet_labels"]]
with (out / "supplemented.tsv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
    writer.writeheader()
    writer.writerows(selected)
ids = {row["protein_id"] for row in selected}
proteins = SeqIO.parse(root / "annotation/results/proteins.faa", "fasta")
count = SeqIO.write((p for p in proteins if p.id in ids), out / "supplemented.faa", "fasta")
print("Exported proteins:", count)
PYCODE
```

To select all model targets, including unassigned ones, filter on a nonempty
`phanet_method` instead. To select one predicted function, test its membership
in `row["phanet_labels"].split(";")`.

## Export module members

After [module prediction](downstream.md#1-predict-modules), export its member
proteins with the module ID included in each FASTA header:

```bash
python - <<'PYCODE'
import csv
from pathlib import Path
from Bio import SeqIO

root = Path("lambda-analysis")
out = root / "exports"
out.mkdir(exist_ok=True)
proteins = SeqIO.to_dict(SeqIO.parse(root / "annotation/results/proteins.faa", "fasta"))
with (root / "modules/module_members.tsv").open() as source, (out / "module-members.faa").open("w") as dest:
    for row in csv.DictReader(source, delimiter="\t"):
        pid = row["protein_id"]
        dest.write(f">{row['module_id']}|{pid}\n{proteins[pid].seq}\n")
print(out / "module-members.faa")
PYCODE
```

A protein belonging to multiple modules appears once per module. To select one
module, keep only rows with its `module_id` inside the loop.

## Read and export assigned GO terms

Each namespace table has one row per protein. This converts the assigned BP
terms to a three-column table:

```bash
python - <<'PYCODE'
import csv
from pathlib import Path

root = Path("lambda-analysis")
out = root / "exports"
out.mkdir(exist_ok=True)
with (root / "go/go_BP.tsv").open() as source, (out / "go-BP-assigned.tsv").open("w", newline="") as dest:
    writer = csv.writer(dest, delimiter="\t")
    writer.writerow(["protein_id", "go_term", "score"])
    for row in csv.DictReader(source, delimiter="\t"):
        for term in filter(None, row["predicted_terms"].split(";")):
            writer.writerow([row["protein_id"], term, row[term]])
print(out / "go-BP-assigned.tsv")
PYCODE
```

Use `go_CC.tsv` or `go_MF.tsv` and a different output filename for the other
namespaces. Proteins without assigned terms contribute no rows to this export.

## Read VF predictions

Export rows whose `predicted_label` is `1`:

```bash
python - <<'PYCODE'
import csv
from pathlib import Path

root = Path("lambda-analysis")
out = root / "exports"
out.mkdir(exist_ok=True)
with (root / "vf/vf.tsv").open() as source, (out / "vf-positive.tsv").open("w", newline="") as dest:
    reader = csv.DictReader(source, delimiter="\t")
    writer = csv.DictWriter(dest, fieldnames=reader.fieldnames, delimiter="\t")
    writer.writeheader()
    selected = [row for row in reader if row["predicted_label"] == "1"]
    writer.writerows(selected)
print("VF-positive proteins:", len(selected))
PYCODE
```

`probability` is the model score; `predicted_label` is its assigned binary label.
Use `protein_id` to join VF and GO results back to `annotations.tsv`.
