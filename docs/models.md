# Install prediction models

Use the same PhaNet environment as the [Quick start](../README.md#quick-start).
Set `PHANET_DATA` to a writable directory on your data disk.

```bash
export PHANET_DATA="/data/phanet-data"
phanet setup-models --models module go vf rbp taxonomy \
  --out "$PHANET_DATA/models"
```

Select only the models you need. The command downloads ZIPs from the
[model folder](https://drive.google.com/drive/folders/1RajAiCH51hHKPcjBYXlXDyw3FjhAHxqq),
checks their pinned SHA256 values and every extracted file, then installs them
under `--out`. ZIPs are retained in `models/.downloads`. Repeating the command
verifies and reuses installed models; interrupted downloads resume on rerun.
Allow about 4.4 GB for all five task bundles and their cached ZIPs.

| Model | Archive | Prediction command | `--model` directory |
|---|---|---|---|
| GLM | `glm-model.zip` | `annotate --method glm` | `$PHANET_DATA/models/glm` |
| Module | `module-model.zip` | `predict-modules` | `$PHANET_DATA/models/module` |
| GO BP/CC/MF | `go-models.zip` | `predict-go` | `$PHANET_DATA/models/go` |
| VF | `vf-model.zip` | `predict-vf` | `$PHANET_DATA/models/vf` |
| RBP | `rbp-model.zip` | `predict-rbp` | `$PHANET_DATA/models/rbp` |
| Taxonomy | `taxonomy-model.zip` | `predict-taxonomy --method glm` | `$PHANET_DATA/models/taxonomy` |

GO installs `BP/`, `CC/` and `MF/` under one model directory. Each model or
namespace contains `model.json` and `weights.pt`, including its label mappings
and prediction thresholds. GLM is already included in the annotation resource
bundle used by the Quick start; select `--models glm` to install it separately.
A separate GLM installation is used with `annotate --model` or `PHANET_MODEL`.

Model installation does not download ESM2, annotation tools or network resources.
Prepare these using the [installation guide](resources.md), or reuse a prepared
dataset and complete ESM2 embeddings. Prediction examples are in
[Module/GO/VF](downstream.md) and [RBP/taxonomy](evidence-downstream.md).

## Manual download and offline installation

Download the required ZIPs from the model folder into one directory. Pass the
ZIP directory directly; no manual extraction or separate checksum file is needed:

```bash
phanet setup-models --models rbp taxonomy \
  --archives /data/model-downloads --out "$PHANET_DATA/models"
```

`--archives` uses local files only. The installer checks them against the same
pinned checksums as online downloads. If Google Drive limits automatic downloads,
use this method. Do not rename the ZIP files.

To verify installed models or install from an existing `.downloads` cache without
network access:

```bash
phanet setup-models --models rbp taxonomy --out "$PHANET_DATA/models" --offline
```

Existing model directories are never replaced. For a damaged or different
installation, choose a new `--out` and use that model path for prediction.
