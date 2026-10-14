# Continue from annotation to downstream prediction

This workflow reuses the dataset and embeddings produced by the
[Quick start](../README.md#quick-start). Run commands from the PhaNet source directory.
Activate the same environment and set `PHANET_DATA` to your resource location:

```bash
conda activate phanet
source "$PHANET_DATA/config-glm/activate.sh"
```

Install the task models once (see [model installation](models.md)):

```bash
phanet setup-models --models module go vf --out "$PHANET_DATA/models"
```

The examples use these model locations:

| Command | Purpose | Model path |
|---|---|---|
| `predict-modules` | Predict protein member sets | `$PHANET_DATA/models/module/` |
| `predict-go` | Predict GO terms | `$PHANET_DATA/models/go/`, with `BP/`, `CC/`, `MF/` |
| `predict-vf` | Predict VF labels | `$PHANET_DATA/models/vf/` |

Each task/namespace directory contains `model.json` and `weights.pt`. Change
`--model` if your bundles are stored elsewhere.

## 1. Predict modules

After the Quick start, reuse its output directly:

```bash
phanet predict-modules \
  --dataset lambda-analysis/annotation/traditional/prepared/dataset \
  --embeddings lambda-analysis/annotation/embeddings \
  --model "$PHANET_DATA/models/module" \
  --out lambda-analysis/modules --device cpu --batch-size 1
```

Read `lambda-analysis/modules/module_members.tsv` for module membership and
`lambda-analysis/modules/summary.json` for counts.

## 2. Predict GO terms

```bash
phanet predict-go \
  --dataset lambda-analysis/annotation/traditional/prepared/dataset \
  --embeddings lambda-analysis/annotation/embeddings \
  --model "$PHANET_DATA/models/go" \
  --out lambda-analysis/go --device cpu --batch-size 1
```

This writes `go_BP.tsv`, `go_CC.tsv` and `go_MF.tsv`. Each row contains a protein
ID, assigned terms and per-term scores. Add `--namespace BP` to run only BP,
or `--namespace BP MF` to select two namespaces.

## 3. Predict VF labels

```bash
phanet predict-vf \
  --dataset lambda-analysis/annotation/traditional/prepared/dataset \
  --embeddings lambda-analysis/annotation/embeddings \
  --model "$PHANET_DATA/models/vf" \
  --out lambda-analysis/vf --device cpu --batch-size 1
```

Read `lambda-analysis/vf/vf.tsv` for protein scores and assigned labels.
The commands above run independently; GO and VF do not require module predictions.
Use `--device cuda` for GPU execution.

## Outputs

```text
lambda-analysis/
  annotation/
    traditional/prepared/dataset/
    embeddings/
    results/
  modules/
    modules.jsonl
    module_members.tsv
    genomes.jsonl
  go/
    go_BP.tsv
    go_CC.tsv
    go_MF.tsv
  vf/
    vf.tsv
```

| File | Contents |
|---|---|
| `modules.jsonl` | Predicted member sets, coordinates, scores and quality status |
| `module_members.tsv` | One row per member of a module passing quality checks |
| `genomes.jsonl` | Per-contig prediction counts |
| `go_*.tsv` | Protein IDs, assigned GO terms and term scores |
| `vf.tsv` | Protein/genome IDs, probability and binary label |
| `summary.json`, `status.json`, `manifest.json` | Counts, run state and input/model information in each task directory |

Module coordinates are 0-based, half-open. An empty member table means no modules
passed selection for that input. See [reading and exporting results](results.md)
for commands to export members, assigned GO terms and VF-positive proteins.

## Start with your own FASTA

Use a new project directory. The first task can prepare the input and generate
embeddings; later tasks reuse them:

```bash
phanet predict-modules --input my-genomes.fna --config "$PHANET_CONFIG" \
  --esm-model "$PHANET_ESM_MODEL" --model "$PHANET_DATA/models/module" \
  --out my-analysis/modules --device cpu --batch-size 1

phanet predict-go \
  --dataset my-analysis/modules/annotation/prepared/dataset \
  --embeddings my-analysis/modules/embeddings \
  --model "$PHANET_DATA/models/go" \
  --out my-analysis/go --device cpu --batch-size 1
```

The same input options work for VF. For a dataset from `phanet prepare`, pass
`--dataset <prepare-output>/prepared/dataset` and `--esm-model "$PHANET_ESM_MODEL"`
to the first prediction command.

If an annotation run has no unresolved proteins, it does not generate embeddings.
In that case, replace `--embeddings` with `--esm-model "$PHANET_ESM_MODEL"` for the
first downstream task; subsequent tasks can reuse that task's `embeddings/`.

## Continue analysis

- [Module analysis](module-analysis.md) starts from `lambda-analysis/modules/`.
- [RBP screening](evidence-downstream.md#reuse-go-output) reads `lambda-analysis/go/go_BP.tsv`.
- [Python API](api.md) provides the equivalent function calls.

Each prediction task requires a new output directory. If a task fails, inspect
its `status.json` and rerun with a new `--out`; `--resume` is available for
`annotate` and `prepare`, not the downstream prediction commands.
