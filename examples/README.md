# Example inputs

The [Quick start](../README.md#quick-start) uses the complete λ (Lambda) phage
genome, [ENA J02459.1](https://www.ebi.ac.uk/ena/browser/view/J02459.1).
Its 48,502 bp sequence is included in `lambda-genome.fna`; source and checksum
are in `lambda-genome-provenance.json`.

## Lambda genome annotation

After Quick start installation, run from the PhaNet source directory:

```bash
source "$PHANET_DATA/config-glm/activate.sh"
phanet annotate --input examples/lambda-genome.fna --out lambda-analysis/annotation \
  --method glm --device cpu --threads 4 --batch-size 1
cat lambda-analysis/annotation/results/summary.json
python examples/check_result.py lambda-analysis/annotation \
  --expected examples/expected-lambda-glm.json
```

Expected output: 62 proteins, 53 traditional annotations, 7 proteins with
additional PhaNet labels and 2 remaining unassigned. The check command compares
counts and annotations with `expected-lambda-glm.json`.

Read `lambda-analysis/annotation/results/annotations.tsv` for the annotations.
Continue with the [downstream workflow](../docs/downstream.md) to reuse the
prepared data, or [export selected results](../docs/results.md).

## Network and combined annotation on a short fragment

Follow [the installation tutorial](../docs/resources.md) first, then source the
generated configuration in the current shell. Set `PHANET_DATA` to the data
directory chosen during installation:

```bash
source "$PHANET_DATA/config/activate.sh"
```

`rggu10-contig.fna` is the 1,700 bp fragment at positions 7,351–9,050 of [Ruminococcus
phage RgGU10, LC889025.1](https://www.ebi.ac.uk/ena/browser/view/LC889025.1). Source
coordinates, provenance and SHA-256 are in `rggu10-provenance.json`. Use the resource
bundle and pinned traditional tools from the installation guide.

```bash
phanet annotate --input examples/rggu10-contig.fna \
  --out rggu10-combined --method combined --device cuda --threads 4 --batch-size 1
python examples/check_result.py rggu10-combined \
  --expected examples/expected-rggu10-combined.json
```

Expected output: 3 proteins, 2 traditional annotations and 1 supplemented protein.
To run a single prediction branch:

```bash
phanet annotate --input examples/rggu10-contig.fna \
  --out rggu10-glm --method glm --device cuda --threads 4 --batch-size 1
python examples/check_result.py rggu10-glm \
  --expected examples/expected-rggu10-glm.json

phanet annotate --input examples/rggu10-contig.fna \
  --out rggu10-network --method network --device cuda --threads 4 --batch-size 1
python examples/check_result.py rggu10-network \
  --expected examples/expected-rggu10-network.json
```

Choose the mode you need; running all three is optional. The checker compares counts,
coordinates, retained descriptions, label decisions and source fields. See [result
fields](../docs/results.md) and [running and resuming](../docs/workflow.md).

## Traditional annotation on a short Lambda fragment

Both Lambda fragments below come from
[J02459.1](https://www.ebi.ac.uk/ena/browser/view/J02459.1), Escherichia phage Lambda.
Their source coordinates and SHA-256 are in `provenance.json` and `nin-provenance.json`.

`lambda-contig.fna` contains positions 43,501–47,000 (3,500 bp).

```bash
phanet prepare --input examples/lambda-contig.fna --out example-traditional
python examples/check_result.py example-traditional \
  --expected examples/expected-traditional.json
```

With Prodigal 2.6.3, eggNOG-mapper 2.1.13, DIAMOND 2.0.11 and eggNOG 5.0.2, the expected
output has **1 contig, 6 proteins, 6 traditionally annotated proteins and 0 unresolved
targets**. The checker compares the counts, native coordinates, traditional descriptions
and functional labels to `expected-traditional.json`.

The additional `lambda-nin-contig.fna` input contains positions 40,601–43,890
(3,290 bp), covering the Lambda Nin region. It produces 5 traditionally annotated
proteins with the same tools and database; substitute it in the preparation
command above to use this region.
