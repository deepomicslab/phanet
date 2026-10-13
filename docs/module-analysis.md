# Module analysis

Compare predicted modules with a reference collection, group them across genomes
and export member annotations.

## Analyze existing predictions

Continue from the [module prediction workflow](downstream.md#1-predict-modules).
Supply your reference collection and a TSV assigning each input genome to its
cluster (formats below). These files are inputs to analysis:

```bash
phanet init-module-analysis \
  --references resources/known-modules.jsonl \
  --reference-scope 'My reference collection, version 1' \
  --clusters resources/genome-clusters.tsv \
  --cluster-scope 'My genome clusters, version 1' \
  --out lambda-analysis/module-analysis.json

phanet analyze-modules \
  --dataset lambda-analysis/annotation/traditional/prepared/dataset \
  --predictions lambda-analysis/modules \
  --config lambda-analysis/module-analysis.json --out lambda-analysis/module-analysis
```

`--predictions` takes a completed `predict-modules` output directory. Use a new
output directory for analysis. If resource files move or change, regenerate the
configuration with `init-module-analysis`.

## Predict and analyze

```bash
phanet discover-modules \
  --dataset prepared/dataset --embeddings complete-esm2-store \
  --model models/module --analysis-config lambda-analysis/module-analysis.json \
  --device cuda --out discovery-output
```

For nucleotide FASTA, replace `--dataset` and `--embeddings` with
`--input genomes.fna --config traditional.json --esm-model esm2.pt`.
See [downstream prediction](downstream.md) for model and embedding setup.

The output contains `prediction/`, `analysis/`, `summary.json` and `status.json`.
To repeat analysis, pass the completed `prediction/` directory to
`analyze-modules` with a new output directory.

## Input files

The reference JSONL contains one member set per line:

```json
{"reference_id":"reference-1","source":"Collection/version","complete":true,"members":[{"protein_sha256":"<64 lowercase hex characters>","qualified_profiles":[]},{"protein_sha256":"<64 lowercase hex characters>","qualified_profiles":[]}]}
```

Use full protein-sequence SHA256 values from
`lambda-analysis/annotation/traditional/prepared/dataset/proteins.tsv`. Each reference needs
at least two members; repeated copies have separate entries. Set `complete` to
`false` for partial references. Only eligible references exclude matched predictions
from new-pattern selection.

The cluster TSV has `genome_id` and `cluster_id` columns and covers exactly the
input genomes. IDs must match the prepared dataset. Omit `--clusters` if the
dataset already has all cluster assignments; a supplied table must agree with them.

Optional inputs to `init-module-analysis`:

| Option | Input |
|---|---|
| `--profiles` | TSV: `protein_sha256`, `profile`, `evalue`, `coverage` |
| `--qualified-profiles` | Alternative TSV: `protein_sha256`, `qualified_profiles` (JSON array of profile IDs) |
| `--sequence-alignments` | Headerless TSV: query SHA, target SHA, E-value, bits, query coverage, target coverage |
| `--sequence-proteins` | JSON array of full protein SHA256 values used in the sequence search |
| `--baseline-policy` | Custom product-description policy JSON; defaults to the bundled policy |
| `--baseline-decisions` | JSONL for all dataset proteins: `protein_id`, `protein_sha256`, `unresolved`, `reason`, `rule_id` |

Profile inputs require `--profile-scope` to identify the source/version. Profile
assignments repeated in references must agree. Sequence evidence requires both
sequence files and `--sequence-scope`; include measured self-alignments with positive
bits and full query/target coverage for every listed protein.

## Outputs

Files below are in the analysis directory.

| File | Contents |
|---|---|
| `patterns.tsv`, `selected-patterns.tsv` | All grouped patterns and the configured selection |
| `dark-patterns.tsv` | Patterns selected for annotation gaps |
| `instances.tsv`, `known-instances.tsv` | Predicted member sets and reference-matched sets |
| `members.tsv` | Protein IDs, coordinates and annotations |
| `pattern-instances.tsv`, `member-mappings.jsonl` | Pattern membership and protein correspondences |
| `genomes.tsv` | Genome clusters and prediction counts |
| `report.json` | Analysis summary |
| `config.json`, `baseline-policy.json` | Settings and annotation policy used |

Run `phanet init-module-analysis --help` for configuration options. Python entry
points and a complete call example are in the [Python API](api.md#analyze-predicted-modules).
