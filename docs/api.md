# Python API

Install PhaNet and configure resources as in the [Quick start](../README.md#quick-start).
The examples below run from the source directory and use the same paths as the
[command-line workflow](downstream.md). Set `PHANET_DATA` and source your
`config-glm/activate.sh` before starting Python.

## Annotate a genome

```python
import os
from pathlib import Path
import torch
from phanet.annotation import PrepareConfig
from phanet.inference import annotate_glm

work = Path("lambda-python")
config = PrepareConfig.load(os.environ["PHANET_CONFIG"])
torch.set_num_threads(config.threads)
summary = annotate_glm(
    "examples/lambda-genome.fna", work / "annotation", config,
    model_dir=os.environ["PHANET_MODEL"],
    esm_checkpoint=os.environ["PHANET_ESM_MODEL"],
    method="glm", device="cpu", batch_size=1,
)
print(summary["proteins"], summary["phanet_assigned"])
```

Returns a summary `dict` and writes annotations to
`lambda-python/annotation/results/`. `annotate_glm` also accepts `method="network"`
or `"combined"` with `network_config=NetworkConfig.load(path)` from
`phanet.network.inference`.

| Argument | Use |
|---|---|
| `input_fasta`, `output` | Nucleotide FASTA path and new run directory |
| `config` | A `PrepareConfig` object, loaded from JSON above |
| `model_dir`, `esm_checkpoint` | GLM bundle directory and local ESM2 checkpoint |
| `device` | `"cpu"` or `"cuda"` |
| `batch_size` | Number of sequences/windows per batch; start with `1` |
| `resume` | `True` to reuse completed stages with the same inputs/settings |
| `progress` | Callback receiving a message string, e.g. `print` |

## Prepare and read genome data

For GenBank/protein imports, custom reference networks and GLM training batches,
see [custom data](custom-data.md).

To run only gene prediction and traditional annotation:

```python
import os
from phanet.annotation import PrepareConfig, prepare
from phanet.datasets.schema import Dataset

config = PrepareConfig.load(os.environ["PHANET_CONFIG"])
summary = prepare("examples/lambda-genome.fna", "lambda-prepared", config, progress=print)
data = Dataset.load("lambda-prepared/prepared/dataset")
print(summary["proteins"])
for protein in data.proteins[:3]:
    print(protein.protein_id, protein.product, protein.start, protein.end)
```

`prepare` returns a summary `dict`. `Dataset.load(directory)` returns a `Dataset`
with `genomes` and `proteins` lists. Protein records provide `protein_id`,
`genome_id`, `sequence`, `product`, `labels`, `start`, `end` and `strand`.
Dataset coordinates are 0-based, half-open; `strand` is `1` or `-1`.

## Predict modules, GO and VF

Continue from the annotation above, with task bundles already under
`$PHANET_DATA/models/`:

```python
from phanet.downstream.predict import predict

models = Path(os.environ["PHANET_DATA"]) / "models"
dataset = work / "annotation/traditional/prepared/dataset"
embeddings = work / "annotation/embeddings"

module_summary = predict(
    "module", work / "modules", models / "module",
    dataset=dataset, embeddings=embeddings, device="cpu", batch_size=1,
)
go_summary = predict(
    "go", work / "go", models / "go",
    dataset=dataset, embeddings=embeddings, namespaces=("BP", "CC", "MF"),
    device="cpu", batch_size=1,
)
vf_summary = predict(
    "vf", work / "vf", models / "vf",
    dataset=dataset, embeddings=embeddings, device="cpu", batch_size=1,
)
print(module_summary["accepted"], vf_summary["predicted_positive"])
```

`predict(task, output, model_dir, ...)` returns a summary `dict` and writes the
[task output files](downstream.md#outputs). Use `"module"`, `"go"` or `"vf"` for
`task`. `dataset` is a directory path, not a `Dataset` object.

To start from FASTA, replace `dataset=...` with `input_fasta=...` and
`config=config`. To generate embeddings, replace `embeddings=...` with
`esm_checkpoint=os.environ["PHANET_ESM_MODEL"]`. Those embeddings are written to
`output/embeddings/` and can be reused by the next call.

## Analyze predicted modules

Prepare the reference/cluster files described in [module analysis](module-analysis.md),
then create a configuration and analyze the module output:

```python
from phanet.analysis.public import init_config, analyze

analysis_config = work / "module-analysis.json"
settings = init_config(
    analysis_config,
    references="resources/known-modules.jsonl",
    reference_scope="My reference collection, version 1",
    clusters="resources/genome-clusters.tsv",
    cluster_scope="My genome clusters, version 1",
)
report = analyze(dataset, work / "modules", analysis_config, work / "module-analysis")
print(report["genomes"], report["proteins"])
```

`init_config` writes JSON and returns its contents as a `dict`. `analyze` returns
a report `dict` and writes the module tables. Both accept paths.
`discover(output, model_dir, analysis_config, ...)` combines prediction and analysis;
it returns `{"prediction": ..., "analysis": ...}` and writes `prediction/` and
`analysis/` under `output`.

## Screen RBP scores in memory

Read the BP output from the GO step:

```python
import csv
import numpy as np
from phanet.downstream.rbp import screen_rbp

with (work / "go/go_BP.tsv").open() as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
ids = [row["protein_id"] for row in rows]
terms = [key for key in rows[0] if key.startswith("GO:")]
scores = np.array([[float(row[t]) for t in terms] for row in rows])
report = screen_rbp(ids, go_scores=scores, go_terms=terms)
print(report["stages"])
```

`screen_rbp` returns a `dict` containing `candidates`, `stages`, `classifier_ids`
and `final_ids`. Additional score vectors must follow the same protein order.
Pass `classifier_scores=...`, `tmvec_scores=...` or structural rows when available.
The function does not write files; `phanet rbp screen` writes TSV/JSON results.
See [RBP inputs](evidence-downstream.md#rbp-screening) for the table formats.

## Summarize protein domains

Read the domain table described in [domain inputs](evidence-downstream.md#domain-summaries):

```python
import csv
from phanet.downstream.rbp import domain_architecture

with open("resources/domains.tsv") as stream:
    domains = list(csv.DictReader(stream, delimiter="\t"))
report = domain_architecture(domains)
print(report["domains"], report["instances"])
```

Returns a `dict` with per-domain `summary` rows and per-interval `observations`.
The table supplies `protein_length`; alternatively pass a
`protein_lengths={protein_id: length, ...}` dictionary. Coordinates are 0-based,
half-open. The function does not write files.

## Classify genomes from reference hits

```python
import csv
from phanet.downstream.taxonomy import classify_taxonomy

with open("resources/query-genomes.txt") as stream:
    query_ids = [line.strip() for line in stream if line.strip()]
with open("resources/taxonomy-hits.tsv") as stream:
    hits = list(csv.DictReader(stream, delimiter="\t"))
predictions = classify_taxonomy(query_ids, hits)
for genome in predictions:
    print(genome["genome_id"], genome["ranks"])
```

Returns a list of genome dictionaries. Each `ranks` entry contains `label`,
`confidence` and `method`. The function does not write files. Use
[`phanet taxonomy`](evidence-downstream.md#genome-classification) for TSV/JSON export.
Pass `ranks=("family", "genus")` to select output ranks. The API supports
`order`, `family`, `subfamily`, `genus` and `species`; supply the available
ancestor columns when requesting a finer rank.


## Train and reload native models

```python
from phanet.learning.run import fit, evaluate, export_model
from phanet.downstream.native_predict import predict

fit("glm-training", "glm-fit", initialize="resources/glm", batch_size=8)
metrics = evaluate("glm-training", "glm-fit/best.pt", partition="test")
predict("taxonomy", "taxonomy-results", "taxonomy-fit/model",
        dataset="prepared/dataset", embeddings="protein-embeddings", method="glm")
```

`fit` writes checkpoints and an inference `model/` directory and returns a run
summary. `evaluate` returns metrics. `export_model(checkpoint, output)` writes
an inference bundle. `phanet.downstream.native_train.fit(task, directory,
labels_path, initialize, output, **options)` fits `rbp` or `taxonomy` heads;
see [training inputs and options](training.md#rbp-and-taxonomy-models).
`predict` writes task tables and returns counts. Use `device="cuda"` to select
a GPU, and a new output path for each run.
