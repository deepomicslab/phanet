# Installation

For the first run, follow the [Quick start](../README.md#quick-start).
This guide covers dependency choices, full network setup and existing installations.

## Environment and dependencies

Use Linux with Python 3.10 or later. The commands below use a separate
Conda environment with Python 3.11. Install
[Miniconda](https://docs.conda.io/en/latest/miniconda.html) or
[Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html)
first if neither is available.

| Component | Requirement | Installation |
|---|---|---|
| Python | 3.10+ | Conda environment below |
| PyTorch | 2.1–2.x, CPU or CUDA build | Select before installing PhaNet |
| NumPy / SciPy | NumPy ≥1.24,<2; SciPy ≥1.11,<2 | Installed by pip with PhaNet |
| scikit-learn / Biopython | scikit-learn ≥1.3,<2; Biopython ≥1.83,<2 | Installed by pip with PhaNet |
| ESM2 Python support | fair-esm 2.0.0 | PhaNet `esm` extra |
| Network Python support | pyhmmer 0.11.0, transformers 4.38.2, sentencepiece 0.2.0 | PhaNet `network` extra |
| Traditional annotation tools | Prodigal 2.6.3, eggNOG-mapper 2.1.13, DIAMOND 2.0.11 | `phanet setup --install-tools` |
| Traditional annotation database | eggNOG 5.0.2 | `phanet setup` |
| Network search | MMseqs2 matching the reference bundle | `phanet setup-resources --build-mmseqs` |
| Build tools for MMseqs2 | C/C++ compiler, CMake, make, zlib headers | System package manager |

## Recommended hardware

Start with these allocations for **one process on a single genome or a small
set of genomes**, using `--batch-size 1`:

| Workflow | CPU cores | System RAM | GPU for acceleration |
|---|---|---|---|
| Gene prediction and traditional annotation (`prepare`) | 4 | 16 GB | Not used |
| GLM annotation (Quick start) | 4–8 | 32 GB | Optional: one NVIDIA GPU with 12 GB VRAM or more |
| Network or combined annotation with the supplied references | 8 | 128 GB | Recommended: one NVIDIA GPU with 16 GB VRAM or more |
| Module/GO/VF prediction, reusing existing embeddings | 4 | 16 GB | Optional: one NVIDIA GPU with 8 GB VRAM or more |

All prediction modes support CPU execution with `--device cpu`. GPU execution
requires an NVIDIA driver and a compatible CUDA build of PyTorch; follow
[GPU installation](#gpu-installation). The PyTorch wheel supplies
the CUDA runtime, so these commands do not require a separate CUDA toolkit.

When module/GO/VF prediction generates ESM2 embeddings from FASTA, use the GLM
resource allocation. Longer proteins and larger batches increase memory use;
keep `--batch-size 1` initially. Network reference data also occupies system RAM
when prediction runs on a GPU. For large collections, split input into smaller
FASTA files while keeping each genome or contig intact, and use separate outputs.

## Disk space

Use an SSD-backed data directory and reserve **150 GB free for the Quick start**,
**200 GB for network/combined setup**, or **80 GB for traditional annotation only**.
These installation budgets include downloads, extraction, environments and a
small example run:

| Resource | Space to allow |
|---|---|
| eggNOG installation, retained downloads and extraction | About 65 GB |
| Extracted PhaNet resource bundle | About 26 GB; downloaded ZIPs need additional space |
| External models | About 2.6 GB for GLM; 11.7 GB for network; 14.3 GB for combined |
| Module/GO/VF/RBP/taxonomy model bundles, when used | About 2.3 GB extracted, plus 2.1 GB cached ZIPs |

Add space for your input files, protein embeddings and results as the dataset
grows. Choose the data location with `PHANET_DATA`; configurations and model
paths are explained in [configuration and data paths](configuration.md).

## 1. Install PhaNet

```bash
conda create -n phanet python=3.11 pip -y
conda activate phanet
```

Install the CPU build of PyTorch for the first run:

```bash
python -m pip install 'torch>=2.1,<3' --index-url https://download.pytorch.org/whl/cpu
```

From the extracted PhaNet source directory (the one containing `pyproject.toml`):

```bash
python -m pip install '.[esm]'
python -m pip check
phanet --help
```

This installs the dependencies for the Quick start and module/GO/VF prediction.
For network or combined annotation, install `'.[esm,network]'` instead.
For traditional annotation, RBP tables or taxonomy tables only, install `'.'`.

## 2. Install traditional tools and the database

Choose a writable directory on your data disk. Change `/data/phanet-data`
below to your location; subsequent commands reuse `PHANET_DATA`. See
[configuration and paths](configuration.md) for the directory layout.

```bash
export PHANET_DATA="/data/phanet-data"
mkdir -p "$PHANET_DATA"
phanet setup --install-tools \
  --tools-prefix "$HOME/.local/share/phanet/tools" \
  --eggnog-data "$PHANET_DATA/eggnog/5.0.2" \
  --config "$PHANET_DATA/traditional.json"
```

This creates a separate environment for Prodigal, eggNOG-mapper and DIAMOND,
downloads eggNOG 5.0.2 and writes their paths to `traditional.json`. Keep the
`phanet` environment active; the configuration locates the separate tools.
Use `--manager mamba` to select Mamba. Installation logs are under
`$HOME/.local/share/phanet/tools.phanet-install/`.

For tools and databases already installed, use the
[existing installation instructions](configuration.md#existing-tools-and-database)
instead of this step. RBP and taxonomy table commands do not need eggNOG or these
tools; continue with their [usage guide](evidence-downstream.md).

## 3. Download the PhaNet bundle

Open the [PhaNet resource folder on Google
Drive](https://drive.google.com/drive/folders/1Qb4Uq5YNzlKOk29ipgy4FL08IzdXA8Wa?usp=drive_link).
Download the `phanet` folder. Google Drive may split it into several ZIP archives;
extract all parts into one destination, preserving the relative paths.

Extract the downloaded `phanet` folder under `$PHANET_DATA`, keeping its name.
The file `$PHANET_DATA/phanet/RESOURCE_MANIFEST.json` should be present:

```text
$PHANET_DATA/phanet/
  RESOURCE_MANIFEST.json
  SHA256SUMS
  README.md
  RESOURCE_NOTICES.md
  LICENSE
  glm/model.json
  glm/weights.pt
  network/network.json
  network/...
  profiles/phrog_v4.hmm
```

Check the extracted files (the subshell keeps your current directory unchanged):

```bash
(cd "$PHANET_DATA/phanet" && sha256sum -c SHA256SUMS)
```

Every file should report `OK`. If files are missing, check that all ZIP parts
were extracted. Keep the directory intact. The bundle provides GLM weights,
network references and PHROG profiles; see the [file guide](resource-files.md).
ESM2, ProtT5 and TM-Vec are installed in the next step.

## 4. Install external models and generate configuration

For the recommended GLM setup, run:

```bash
phanet setup-resources --method glm \
  --bundle "$PHANET_DATA/phanet" \
  --external "$PHANET_DATA/external" \
  --out "$PHANET_DATA/config-glm" \
  --config "$PHANET_DATA/traditional.json"
source "$PHANET_DATA/config-glm/activate.sh"
```

Continue to step 5. For network or combined annotation, install the additional
Python dependencies and build tools. From the source directory on Debian/Ubuntu:

```bash
python -m pip install '.[esm,network]'
sudo apt-get install build-essential cmake zlib1g-dev
```

Then configure combined annotation:

```bash
phanet setup-resources \
  --bundle "$PHANET_DATA/phanet" \
  --external "$PHANET_DATA/external" \
  --out "$PHANET_DATA/config" \
  --config "$PHANET_DATA/traditional.json" \
  --build-mmseqs --threads 4
source "$PHANET_DATA/config/activate.sh"
```

This downloads external models, builds MMseqs2 and writes `phanet.json`,
`network.json`, `activate.sh` and `RESOURCES.json` under `config/`.
`activate.sh` sets the resource paths for later commands.

| Annotation mode | Setup options | External model downloads |
|---|---|---|
| Combined (default) | Command above | ESM2, ProtT5, TM-Vec: about 14.3 GB |
| GLM | Add `--method glm`; omit `--build-mmseqs` | ESM2: about 2.6 GB; no compiler needed |
| Network | Add `--method network` | ProtT5, TM-Vec: about 11.7 GB |

The MMseqs2 version must match the reference bundle. If it is already installed,
replace `--build-mmseqs` with `--mmseqs /path/to/matching/mmseqs`. The installer
checks its version. Build on the machine where it will run; rebuild after moving
to a CPU with a different instruction set.

## 5. Check the installation and run

In each new shell, set `PHANET_DATA` to the directory chosen in step 2, then
activate the Python environment and resource paths:

```bash
export PHANET_DATA="/data/phanet-data"
conda activate phanet
source "$PHANET_DATA/config-glm/activate.sh"
python -m pip check
phanet doctor --config "$PHANET_CONFIG"
```

For network/combined setup, source `config/activate.sh` instead and add
`--network-config "$PHANET_NETWORK_CONFIG"` to the doctor command.

Run with a nucleotide FASTA file and a new output directory:

```bash
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1
```

Results are under `lambda-analysis/annotation/results/`.
For network/combined setup select `--method network` or `--method combined`.
See the [worked examples](../examples/README.md),
[run guide](workflow.md) and [output fields](results.md).

## Task models

Install Module, GO, VF, RBP and taxonomy models with
[the model installation command](models.md). It checks each download and writes
model directories that can be passed directly to prediction commands.

## GPU installation

Activate the same PhaNet environment, run `nvidia-smi`, and select a CUDA wheel
compatible with your driver using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/).
For example, for a driver supporting CUDA 12.8:

```bash
python -m pip install --upgrade --force-reinstall 'torch>=2.1,<3' \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip check
python -c 'import torch; print(torch.ones(1, device="cuda"))'
```

Use `--device cuda` in prediction commands. Start with `--batch-size 1` and use
a new output directory when changing an existing run's device or batch size.

## Interrupted downloads and offline use

Rerun the same `setup` or `setup-resources` command to resume interrupted
downloads. Completed files are verified and reused. If tool-environment creation
failed, read `tools.phanet-install/install.log` and retry with a new
`--tools-prefix` and `--config`. A failed MMseqs2 build keeps logs under
`external/mmseqs-build/`; use a new `--external` directory for another build.

For offline use, copy the completed bundle, eggNOG database and `external/`
resources from an online machine, and install the required Python packages and
tools locally. Generate `traditional.json` with your local tool/database paths.
Run `setup-resources` with those paths, `--offline`, a new `--out` directory and
`--mmseqs /path/to/matching/mmseqs` instead of `--build-mmseqs`. The command checks
existing files without downloading them.

If directories move, regenerate configurations with the new paths. Bundle,
external-model and configuration directories must be separate, with no one
nested inside another. Use real directories, not symlink or hardlink aliases.
After an interrupted process, remove `.setup.lock` only after its writer has stopped.

For specific error messages and diagnostic commands, see [troubleshooting](troubleshooting.md).
