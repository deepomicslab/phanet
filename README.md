# PhaNet

PhaNet annotates phage genomes and provides protein function prediction,
module analysis, RBP screening and genome classification.

## Requirements

- Linux.
- Python 3.10 or later (3.11 recommended).
- [Conda](https://docs.conda.io/en/latest/miniconda.html) or [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

For the Quick start, we recommend **4–8 CPU cores, 32 GB RAM and 150 GB free disk
space**. An NVIDIA GPU is optional; GPU use requires a working NVIDIA driver.

See the [installation guide](docs/resources.md) for dependencies, GPU setup and
resource recommendations for other workflows.

## Quick start

Annotate the λ (Lambda) phage genome included at `examples/lambda-genome.fna`.
This example runs gene prediction, traditional annotation and PhaNet prediction
on CPU. The input is the 48,502 bp genome from [ENA J02459.1](https://www.ebi.ac.uk/ena/browser/view/J02459.1).

The commands below install and configure GLM mode from the extracted PhaNet source.
Choose a writable directory on your data disk;
change the path in the first line below to your location:

```bash
export PHANET_DATA="/data/phanet-data"
mkdir -p "$PHANET_DATA"
```

Download the `phanet` resource folder from
[Google Drive](https://drive.google.com/drive/folders/1Qb4Uq5YNzlKOk29ipgy4FL08IzdXA8Wa?usp=drive_link)
and extract all ZIP parts into that directory, keeping the downloaded folder
name. The bundle should be at `$PHANET_DATA/phanet/`, with
`RESOURCE_MANIFEST.json` directly inside it.

### 1. Install and configure once

From the source directory containing `pyproject.toml` and `examples/`, copy:

```bash
conda create -n phanet python=3.11 pip -y
conda activate phanet
python -m pip install 'torch>=2.1,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[esm]'

phanet setup --install-tools \
  --eggnog-data "$PHANET_DATA/eggnog/5.0.2" \
  --config "$PHANET_DATA/traditional.json"
phanet setup-resources --method glm \
  --bundle "$PHANET_DATA/phanet" \
  --external "$PHANET_DATA/external" \
  --out "$PHANET_DATA/config-glm" \
  --config "$PHANET_DATA/traditional.json"
```

The setup commands install the annotation tools, download eggNOG and ESM2,
and generate configuration files. For other installation options, see the
[installation guide](docs/resources.md).

### 2. Run the example

In the same source directory:

```bash
source "$PHANET_DATA/config-glm/activate.sh"
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1
cat lambda-analysis/annotation/results/summary.json
```

With the tools and database installed above, the example produces 62 proteins:
53 have traditional annotations, 7 receive additional PhaNet labels and 2 remain
unassigned.
Open `lambda-analysis/annotation/results/annotations.tsv` for the annotations.
The same directory contains `proteins.faa`, `annotations.gff3` and `summary.json`.

For your own data, replace `examples/lambda-genome.fna` with your nucleotide
FASTA or FASTA.gz file and `lambda-analysis/annotation` with a new output directory.
In a new shell, set `PHANET_DATA` to the same location, run `conda activate phanet`
and source `"$PHANET_DATA/config-glm/activate.sh"` before annotation.
To continue the same run, add `--resume`.

## Optional workflows

| Task | How to use it |
|---|---|
| Run on GPU | Install CUDA PyTorch as described in [installation](docs/resources.md), then use `--device cuda` |
| Add network annotation or use combined mode | Install the network dependencies and resources using [installation steps 1 and 4](docs/resources.md); then select `--method network` or `--method combined` |
| Use existing tools, change data locations or pass explicit paths | [Configuration and data paths](docs/configuration.md) |
| Run gene prediction and traditional annotation only | [Preparation](docs/workflow.md#preparation) |
| Build a network from your own references or prepare GLM training data | [Custom data](docs/custom-data.md) |
| Train, fine-tune, evaluate or export a model | [Training](docs/training.md) |
| Predict modules, GO terms or VF labels | [Downstream prediction](docs/downstream.md) |
| Compare and group predicted modules | [Module analysis](docs/module-analysis.md) |
| Predict RBP, classify genomes or summarize domains | [RBP and classification](docs/evidence-downstream.md) |

See [reading and exporting results](docs/results.md),
[running and recovery](docs/workflow.md), [troubleshooting](docs/troubleshooting.md)
and [more examples](examples/README.md) for common tasks. Run `phanet --help`
or `phanet <command> --help` for command options.

## Python interfaces and code structure

Use the [Python API](docs/api.md) for function calls, return values and output
paths. The [architecture overview](docs/architecture.md) shows the main modules
and processing flow.

## License

PhaNet uses the [MIT License](LICENSE). Database and model sources are listed
in [resource sources](docs/resource-sources.md) and
[third-party notices](THIRD_PARTY_NOTICES.txt).
