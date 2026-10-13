# Custom networks and GLM training data

## Import your data

For annotated GenBank files, run:

```bash
phanet import-data --genbank references/*.gbk --out imported
```

Replace `references/*.gbk` with your files. The result is a PhaNet dataset at
`imported/dataset`, shared by the network and training commands below.

For nucleotide FASTA, use your traditional annotation configuration:

```bash
phanet import-data --input genomes.fna \
  --config "$PHANET_DATA/traditional.json" --out imported
```

This runs gene prediction and eggNOG annotation. To reuse a previous annotation
run, pass its `traditional/prepared/dataset` directory directly to `--dataset`
in either command below.

For protein FASTA, supply gene positions and genome metadata:

```bash
phanet import-data --proteins proteins.faa \
  --genes genes.tsv --genomes genomes.tsv --out imported
```

The tables are tab-separated and include a header:

| Table | Required columns | Optional columns |
|---|---|---|
| `genes.tsv` | `protein_id`, `genome_id`, `start`, `end`, `strand`, `order` | `product`, `labels`, `family_ids`, `evidence_source`, `evidence_version` |
| `genomes.tsv` | `genome_id`, `length_nt` | `accession`, `source`, `cluster_id`, `complete`, `quality`, `taxonomy`, `host`, `gc`, `lifestyle` |

Protein IDs must match the FASTA headers exactly. Coordinates are 0-based,
half-open; `strand` is `+` or `-`. Within each genome, `order` starts at zero
and follows increasing genomic coordinates. `complete` uses `1` or `0`.
Multiple `labels` or `family_ids` are separated by semicolons. If the `labels`
column is omitted, labels are derived from `product`.

## Build a reference network

Install the [network dependencies and resources](resources.md), including
MMseqs2, a HMMER-format PHROG library, TM-Vec and ProtT5. Using the standard
resource locations:

```bash
phanet build-network --dataset imported/dataset \
  --profiles "$PHANET_DATA/phanet/profiles/phrog_v4.hmm" \
  --tmvec-model "$PHANET_DATA/external/tmvec" \
  --prott5 "$PHANET_DATA/external/prott5" \
  --mmseqs "$PHANET_DATA/external/mmseqs/bin/mmseqs" \
  --threads 4 --device cpu --out custom-network
```

If MMseqs2 is on `PATH`, omit `--mmseqs`. Use `--device cuda` with CUDA PyTorch
to compute TM-Vec vectors on a GPU. An alternative protein HMM library can be
passed with `--profiles`; subsequent queries use the same library and models.

For larger reference collections, run `python -m pip install '.[faiss]'` from
the source directory and add
`--structure-backend faiss-hnsw` for approximate indexed structure search.
`--structure-backend faiss` uses exact indexed search; the default `exact`
uses blockwise comparisons without FAISS.

The command computes sequence, profile and structure relationships, merges
them, and writes the reference package and its configuration. At least one
reference protein needs a functional label. Repeated sequences are combined;
conflicting labels for the same sequence must be resolved in the input.

Annotate new genomes with the resulting network:

```bash
phanet annotate --input query-genomes.fna --out custom-annotation \
  --method network --config "$PHANET_DATA/traditional.json" \
  --network-config custom-network/network-config.json --device cpu --threads 4
```

| Output | Contents |
|---|---|
| `custom-network/reference/` | Reference sequences, vectors, graph, propagation values and checksums |
| `custom-network/network-config.json` | Configuration accepted by `annotate` and `doctor` |
| `custom-network/summary.json` | Reference and edge counts, input checksums |
| `custom-network/build/` | Alignment databases, hits and tool logs |

Keep `reference/` and its configuration together. If you move the external
models, HMM library or MMseqs2, update their paths in `network-config.json`.
Construction uses more memory and temporary disk than querying an existing
network; requirements depend on protein count and the number of retained edges.
Use a new output directory for each build.

## Prepare GLM training data

Install `python -m pip install '.[esm]'` from the source directory and configure a local ESM2 checkpoint as in the
[installation guide](resources.md). From an imported dataset:

```bash
source "$PHANET_DATA/config-glm/activate.sh"
phanet prepare-training --dataset imported/dataset \
  --esm-checkpoint "$PHANET_ESM_MODEL" --out glm-training
```

This generates ESM2 vectors, train/validation/test partitions and the files
used by the training loader. Supply at least three independent genomes with
proteins. Use `--device cuda` to compute ESM2 on a GPU; the default batch size
is one. GenBank, nucleotide FASTA and protein/table inputs are also accepted
directly through the same options as `import-data`.

To reuse existing ESM2 vectors, replace `--esm-checkpoint` with
`--embeddings path/to/embeddings`. The store is checked against protein IDs and
sequences and referenced without copying; keep it available at that location.

Optional settings:

| Option | Use |
|---|---|
| `--ratios 0.8 0.1 0.1` | Train, validation and test proportions; genomes stay together |
| `--unit cluster` | Keep genomes with the same `cluster_id` together; supply at least three clusters |
| `--splits splits.json` | Reuse an existing `SplitManifest`; overrides automatic partitioning |
| `--seed 7` | Reproduce partitioning and window sampling |
| `--max-context 30` | Maximum proteins per context window |
| `--batch-size 1` | ESM2 sequences per batch |

For related genomes, assign `cluster_id` before partitioning and use
`--unit cluster`. Identical genome sequences and supplied clusters cannot cross
partitions. Genomes with no functional annotations remain available as context.

`glm-training/` contains `dataset/`, `splits.json`, `training.json`, and
`embeddings/` when vectors are newly generated. `training.json` records the
paths, checksums and number of windows in each partition.

Load batches through the Python interface:

```python
from torch.utils.data import DataLoader
from phanet.learning.data import load_training

train = load_training("glm-training", "train")
validation = load_training("glm-training", "val")
loader = DataLoader(train, batch_size=8, shuffle=True)
batch = next(iter(loader))
print(batch["embeddings"].shape)
```

Each batch includes protein vectors, strand, visible annotations, prediction
targets and masks. Validation and test annotations are hidden from model inputs.
Windows are generated when read and stay within their source genome. For
sampled module windows, use `mode="sampled_modules"` and call
`train.set_epoch(epoch)` at the start of each training epoch.

The loader returns a PyTorch dataset accepted by `phanet.learning.training.Trainer`.
Set `ModelConfig.input_dim` to `train.embeddings.dimension` and
`ModelConfig.max_context` to `train.max_context` when constructing the model.
For training, validation, continuation and model export, use the
[training commands](training.md). The plain loader supplies annotation windows;
`phanet.learning.objectives.training_views` adds auxiliary pretraining targets.
