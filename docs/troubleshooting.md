# Troubleshooting

Run these checks in the environment used for PhaNet. Set `PHANET_DATA` to your
resource directory. The examples below use the Quick start's GLM configuration.

## Command or Python package not found

```bash
conda activate phanet
command -v python
command -v phanet
python -m pip check
python -c 'import phanet; print(phanet.__file__)'
```

If `phanet` is missing, run `python -m pip install '.[esm]'` from the source
directory containing `pyproject.toml`. Use that environment's Python in notebooks
and scripts too.

## Configuration or resource file not found

```bash
source "$PHANET_DATA/config-glm/activate.sh"
python - <<'PY'
import os
from pathlib import Path
for name in ("PHANET_CONFIG", "PHANET_MODEL", "PHANET_ESM_MODEL"):
    value = os.environ.get(name)
    print(name, value, "exists:", bool(value) and Path(value).exists())
PY
phanet doctor --config "$PHANET_CONFIG"
```

The three paths should point to the preparation JSON, GLM directory and ESM2
checkpoint respectively. For network mode, also check:

```bash
phanet doctor --config "$PHANET_CONFIG" --network-config "$PHANET_NETWORK_CONFIG"
```

If you moved the resource directories, rerun `setup-resources` with their new
locations and a new `--out` configuration directory, then source its `activate.sh`.
The [configuration guide](configuration.md) lists all path fields.

## eggNOG-mapper, Prodigal or DIAMOND is missing

Run `phanet doctor --config "$PHANET_CONFIG"` and inspect the named executable.
The configuration points to the annotation-tool environment; keep that environment
installed even when the main `phanet` environment is active. For an existing
installation, regenerate the JSON with the correct executable paths using
[`init-config`](configuration.md#existing-tools-and-database).

If `setup --install-tools` failed, read the tool installation log. With the default
prefix it is `$HOME/.local/share/phanet/tools.phanet-install/install.log`.
Resolve the Conda error and retry using a new `--tools-prefix` and `--config`.

## CUDA is unavailable or runs out of memory

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

If `nvidia-smi` fails, fix the NVIDIA driver. If it works but PyTorch reports
`False`, install a compatible CUDA PyTorch wheel in the active environment using
[GPU setup](resources.md#gpu-installation). For a memory error, use `--batch-size 1`.
If an individual protein still exceeds GPU memory, use `--device cpu`.
Use a new output directory after changing the device or batch size.

## Output already exists or a previous run stopped

For the same annotation input and settings, repeat the original command with
`--resume`. For example:

```bash
phanet annotate --input examples/lambda-genome.fna \
  --out lambda-analysis/annotation --method glm --device cpu \
  --threads 4 --batch-size 1 --resume
```

If the error is `Incomplete stage retained`, inspect the tool log in that stage,
fix the cause and use a new output directory. `--resume` reuses completed stages.
For `predict-modules`, `predict-go`, `predict-vf` and analysis commands, rerun with
a new `--out` directory.

If a lock remains after an interrupted process, first confirm that the process
has stopped. Then remove only the lock file named in the error and retry.

## Find the failing stage

```bash
cat lambda-analysis/annotation/status.json
cat lambda-analysis/annotation/traditional/status.json
```

Traditional-tool records are at:

- `lambda-analysis/annotation/traditional/genes/prodigal/prodigal_execution.json`
- `lambda-analysis/annotation/traditional/eggnog/tool.log`

If you redirected the annotation log, read its final lines with
`tail -n 40 lambda-analysis/annotation.log`. Downloads and database searches
can take much longer than the small example's model prediction.

## Missing embeddings or empty results

If `summary.json` reports `phanet_targets: 0`, traditional annotation resolved
all proteins and no GLM embeddings were generated. For downstream prediction,
use `--esm-model "$PHANET_ESM_MODEL"` to generate embeddings, then reuse that
command's `embeddings/` directory. See [downstream input options](downstream.md#start-with-your-own-fasta).

An empty module table or empty `predicted_terms` cell is a valid result. Check
`status.json` and `summary.json` for completion and counts. Blank annotation
scores indicate proteins not submitted to model prediction.

## Interrupted download or disk space error

```bash
df -h "$PHANET_DATA"
```

Rerun the same setup command to resume downloads. For a checksum mismatch, use
the resource version required by your configuration and a fresh destination.
See [offline and interrupted installation](resources.md#interrupted-downloads-and-offline-use)
for the complete steps.
