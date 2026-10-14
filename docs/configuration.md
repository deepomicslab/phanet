# Configuration and data paths

## Directory layout

Choose a writable directory on your data disk and set `PHANET_DATA` once
in your shell. Change `/data/phanet-data` below to your location:

```bash
export PHANET_DATA="/data/phanet-data"
```

The commands in the guides reuse this shell variable. It selects the resource
location for those commands; PhaNet reads the resulting paths from its
configuration files. The Quick start and recommended installation use
`config-glm/` for GLM; the optional network/combined setup uses `config/`.
The directory name is selected by `setup-resources --out`.

```text
$PHANET_DATA/
  traditional.json          # Tool and eggNOG paths from phanet setup
  eggnog/5.0.2/             # Traditional annotation database
  phanet/                   # Downloaded PhaNet resource folder
    glm/
    network/
    profiles/
  external/                # Downloaded external models and built search tool
    esm/
    prott5/
    tmvec/
    mmseqs-build/
  config-glm/              # Recommended GLM configuration
    phanet.json
    activate.sh
    RESOURCES.json
  config/                  # Optional network/combined configuration
    phanet.json
    network.json
    activate.sh
    RESOURCES.json
  models/                  # Separate downstream model bundles, if used
    module/
    go/BP/
    go/CC/
    go/MF/
    vf/
    rbp/
    taxonomy/
```

Install the task bundles with [setup-models](models.md).

The Python environment and annotation-tool environment are separate from these
data directories. Keep input FASTA files outside output directories. For example:

```text
lambda-analysis/
  annotation/              # Created by phanet annotate
  annotation.log
  modules/
  go/
  vf/
  exports/
```

Create the project directory for the log; let PhaNet create `annotation/`:

```bash
mkdir -p lambda-analysis
export PYTHONUNBUFFERED=1
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1 > lambda-analysis/annotation.log 2>&1
```

## Configuration files

| File | Purpose |
|---|---|
| `traditional.json` | Tool commands, eggNOG path and preparation settings written by `setup` |
| `config-glm/phanet.json` | Preparation configuration copied by `setup-resources`, with absolute paths |
| `config/network.json` | Network reference, PHROG, ProtT5, TM-Vec and MMseqs2 paths |
| `config-glm/activate.sh` | Shell exports for the configuration and annotation models |
| `config-glm/RESOURCES.json` | Resource URLs and checksums |

Use the generated files after installation. If you edit preparation settings for
annotation, edit the active `config-glm/phanet.json`; later annotation commands read
that file, not the original `traditional.json`.

Preparation configuration fields:

| Field | Value |
|---|---|
| `schema_version` | `1` |
| `eggnog_data` | eggNOG 5.0.2 database directory |
| `prodigal` | Prodigal executable name or full path |
| `emapper` | Command list, such as `["emapper.py"]` or `["/path/to/python", "-I", "/path/to/emapper.py"]` |
| `threads` | CPU thread count; can also be set with `--threads` |
| `diamond_block_size` | DIAMOND block size; default `0.4` |
| `max_evalue` | eggNOG search E-value limit; default `0.001` |

The [preparation template](../configs/default.json) and
[network template](../configs/network.json) show the JSON format. Replace their
example paths before use. Relative paths in these configuration files resolve
from the configuration file's directory; command-line input/output paths resolve
from your current working directory. Use absolute paths when sharing a config
between working directories.

## Environment variables and explicit arguments

Set `PHANET_DATA` to your chosen directory again when opening a new shell.

```bash
conda activate phanet
source "$PHANET_DATA/config-glm/activate.sh"
```

| Variable | Equivalent `annotate` option | Points to |
|---|---|---|
| `PHANET_CONFIG` | `--config` | `config-glm/phanet.json` |
| `PHANET_NETWORK_CONFIG` | `--network-config` | `config/network.json` for network/combined setup |
| `PHANET_MODEL` | `--model` | Bundle `glm/` directory |
| `PHANET_ESM_MODEL` | `--esm-model` | `external/esm/esm2_t33_650M_UR50D.pt` |

Explicit options override environment variables. For GLM annotation without
sourcing `activate.sh`, pass the paths directly:

```bash
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --config "$PHANET_DATA/config-glm/phanet.json" \
  --model "$PHANET_DATA/phanet/glm" \
  --esm-model "$PHANET_DATA/external/esm/esm2_t33_650M_UR50D.pt" \
  --method glm --device cpu
```

The downstream commands require their own `--model` and either `--embeddings` or
`--esm-model`; they do not use `PHANET_MODEL` or `PHANET_ESM_MODEL` as defaults.
See [downstream prediction](downstream.md) for examples.

## Existing tools and database

If Prodigal, eggNOG-mapper and DIAMOND are already on PATH and eggNOG 5.0.2 is
installed:

```bash
phanet init-config --eggnog-data /data/eggnog/5.0.2 --out traditional.json
phanet doctor --config traditional.json
```

For a separate tool environment:

```bash
phanet init-config --eggnog-data /data/eggnog/5.0.2 \
  --prodigal /opt/annotation-tools/bin/prodigal \
  --emapper /opt/annotation-tools/bin/python /opt/annotation-tools/bin/emapper.py \
  --out traditional.json
phanet doctor --config traditional.json
```

Replace `/data/eggnog/5.0.2` and `/opt/annotation-tools` with your paths. Keep
DIAMOND available in the tool environment. `init-config` writes configuration for
these installed resources. Continue with steps 3–5 of the
[installation guide](resources.md), passing this file as `--config`.

## Reusing prepared data

| Original command | Dataset path | Embedding path |
|---|---|---|
| `prepare --out prepared-run` | `prepared-run/prepared/dataset` | Generated by a later prediction command |
| `annotate --out annotation` | `annotation/traditional/prepared/dataset` | `annotation/embeddings` when GLM runs |
| `predict-modules --input ... --out modules` | `modules/annotation/prepared/dataset` | `modules/embeddings` |

GO and VF accept the same dataset and embedding options as module prediction.
Keep the dataset and its matching embeddings together. If the original command
did not generate embeddings, supply `--esm-model` to generate them.
