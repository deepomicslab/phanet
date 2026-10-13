# Run and manage an annotation

## Choose the input and method

`phanet annotate` accepts nucleotide FASTA or FASTA.gz with unique contig IDs and
nonempty DNA sequences. Each contig's boundaries and CDS order are retained.

| Task | Entry point |
|---|---|
| Gene prediction and traditional annotation | `phanet prepare` |
| GLM supplementation | `phanet annotate --method glm` |
| Network supplementation | `phanet annotate --method network` |
| Combined supplementation | `phanet annotate --method combined` (default) |

## Configure once and run

After [resource installation](resources.md), load the configuration in each shell.
Set `PHANET_DATA` to the data directory chosen during installation. See
[configuration and data paths](configuration.md) for fields and path settings:

```bash
source "$PHANET_DATA/config-glm/activate.sh"
mkdir -p lambda-analysis
export PYTHONUNBUFFERED=1
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1 \
  > lambda-analysis/annotation.log 2>&1
```

Use a new output directory for each run. Place the log outside it so the program can
create the directory. `--device cpu` selects CPU execution, `--threads` controls CPU
parallelism and `--batch-size` controls ESM2/GLM batching. Memory use depends on the reference size, sequence lengths and batch size.

## Progress and results

```bash
tail -f lambda-analysis/annotation.log
```

The log reports resource verification, searches and model progress. `status.json`
records the run state, and `traditional/status.json` records preparation stages. Use the
process exit status and `results/summary.json` to check completion. See [result
fields](results.md) for output files and column definitions.

## Resume completed stages

Repeat the same command with `--resume`:

```bash
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1 --resume \
  > lambda-analysis/annotation-resume.log 2>&1
```

Resume reuses completed stages after checking inputs, code, resource versions, settings
and output checksums. Changes to these require a new output directory. A partially
written stage also requires a new output directory after fixing the error reported in
its log.

After a hard interruption, a stale `.lock` or `.setup.lock` may remain. Remove it after
the writing process has stopped. Resource downloads resume through the setup commands
described in the [installation guide](resources.md).

## Common issues

See [troubleshooting](troubleshooting.md) for configuration checks, missing tools,
CUDA errors, interrupted runs and absent embeddings.

## Preparation

To run gene prediction and traditional annotation only:

```bash
phanet prepare --input examples/lambda-genome.fna --out lambda-prepared --config "$PHANET_CONFIG"
```

`lambda-prepared/prepared/` contains `proteins.faa`, annotation TSV/GFF3 files,
`dataset/`, `targets.tsv` and summary files. The dataset can be passed to the
[downstream commands](downstream.md).

For an existing eggNOG database and tools on PATH:

```bash
phanet init-config --eggnog-data /path/to/eggnog/5.0.2 --out phanet.json
phanet doctor --config phanet.json
```

Set `PHANET_CONFIG` to reuse this configuration in subsequent commands.
