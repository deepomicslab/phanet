# Training

Install PhaNet and prepare `glm-training/` using [custom data](custom-data.md#prepare-glm-training-data).
Keep its referenced embedding store in place. Use a new output directory for
each command. Training uses the installed PyTorch build; select `--device cuda`
for an NVIDIA GPU. Reduce `--batch-size` if GPU memory is insufficient.

## Quick start: fine-tune a GLM

After the [resource setup](resources.md):

```bash
phanet train --data glm-training --initialize "$PHANET_DATA/phanet/glm" \
  --batch-size 8 --device cuda --out glm-fit
phanet evaluate --data glm-training --checkpoint glm-fit/best.pt \
  --partition test --device cuda --out glm-evaluation
phanet annotate --input query-genomes.fna --method glm \
  --model glm-fit/model --esm-model "$PHANET_ESM_MODEL" \
  --config "$PHANET_DATA/traditional.json" --out query-annotation
```

`glm-fit/model/` is the validation-selected model accepted by `annotate`.
`best.pt` and `last.pt` include training state. `history.json` contains epoch
metrics; `glm-evaluation/metrics.json` contains held-out evaluation results.

## Configuration and continuation

To change optimization settings, create a JSON file and pass `--train-config`:

```json
{
  "learning_rate": 0.00005,
  "max_epochs": 50,
  "patience": 8,
  "warmup_steps": 100,
  "gradient_accumulation": 4,
  "checkpoint_interval": 100
}
```

`--batch-size` counts windows per update input; `gradient_accumulation` combines
that many inputs before an optimizer update. The learning-rate schedule is
calculated from dataset size unless `total_steps` is supplied.

Continue in a new directory, keeping the original mode, batch size, device,
source version and prepared data:

```bash
phanet train --data glm-training --resume glm-fit/last.pt \
  --batch-size 8 --device cuda --out glm-continued
```

Model and optimization settings are read from the checkpoint. Keep `best.pt`
next to `last.pt` to retain the earlier validation-selected model.
To export another checkpoint explicitly:

```bash
phanet export-model --checkpoint glm-fit/best.pt --out exported-glm
```

## Pretrain from custom data

Prepare functional labels and protein-family assignments before pretraining:

```bash
phanet prepare-training --dataset imported/dataset \
  --embeddings protein-embeddings --profiles "$PHANET_DATA/phanet/profiles/phrog_v4.hmm" \
  --out pretraining-data
phanet train --data pretraining-data --mode pretrain \
  --batch-size 8 --device cuda --out pretraining-fit
```

Profile search requires the `network` installation extra. Alternatively, use
`--family-hits families.tsv` with `protein_id`, `family_id`, `evalue` and
`coverage` columns. Supply qualified HMMER family hits, with coverage measured
over the union of accepted domain intervals. Existing dataset `family_ids`
are retained when neither option is supplied. At least two families must have
training members.

Pretraining adds sampled module windows, reciprocal composition pairs, family
prototypes and order supervision. Prototypes are fitted from training proteins.
The exported `pretraining-fit/model/` can initialize annotation fine-tuning.
For a custom architecture, pass `--model-config model.json`; its keys correspond
to `phanet.learning.model.ModelConfig`. Annotation models use 1280-dimensional
ESM2 inputs and a maximum context of 30 proteins.

## RBP and taxonomy models

Both commands initialize from a GLM bundle and write a task bundle to `model/`.
`--train-config` accepts `epochs`, `batch_size`, `learning_rate`, `weight_decay`,
`patience`, `seed` and `train_last_layers`. For example:

```json
{"epochs": 50, "batch_size": 64, "learning_rate": 0.00005, "patience": 8}
```

### RBP

Prepare a TSV with `protein_id`, binary `label` and `partition`
(`train`, `val`, `test`). IDs must occur in the prepared native dataset.
RBP partitions apply to targets independently of the GLM genome partitions;
identical target sequences must remain in one partition. Include both classes
in training and validation.

Train a standalone RBP classifier directly:

```bash
phanet train-rbp --data glm-training --labels rbp-labels.tsv \
  --initialize "$PHANET_DATA/phanet/glm" --device cuda --out rbp-fit
```

For two-stage Infection/RBP screening, generate the Infection input and supply it during training:

```bash
phanet score-infection --dataset glm-training/dataset \
  --embeddings protein-embeddings --annotation-model "$PHANET_DATA/phanet/glm" \
  --network-config "$PHANET_NETWORK_CONFIG" --device cuda --out infection-input
phanet train-rbp --data glm-training --labels rbp-labels.tsv \
  --infection-scores infection-input/infection.tsv \
  --initialize "$PHANET_DATA/phanet/glm" --device cuda --out rbp-fit
```

The score table binds each `protein_id` to `protein_sha256` and
`infection_score`. It must cover every supervised protein. The saved bundle
includes the validation-selected decision threshold.

### Taxonomy

Prepare a TSV with `genome_id`, `partition` and available `order`, `family`,
`subfamily`, `genus` labels. Leave missing labels blank or omit their columns.
Genus labels can be used without family labels; predictions leave unresolved
levels blank. For fragments, include
`parent_id` identifying the source genome. Partitions must match the prepared
data, and fragments of a parent stay together. By default, each retained class
requires five independent training parents.

For GLM classification:

```bash
phanet train-taxonomy --data glm-training --labels taxonomy-labels.tsv \
  --initialize "$PHANET_DATA/phanet/glm" --device cuda --out taxonomy-fit
```

The run also exports `reference-labels.tsv` with training protein hashes and
lineages. To train with combined network/GLM validation, first obtain network
hits for the validation dataset, using these training references:

```bash
phanet taxonomy-hits --dataset validation-native/dataset \
  --labels taxonomy-labels.tsv --reference-labels taxonomy-fit/reference-labels.tsv \
  --network-config "$PHANET_NETWORK_CONFIG" --out validation-hits
phanet train-taxonomy --data glm-training --labels taxonomy-labels.tsv \
  --validation-hits validation-hits/hits.tsv \
  --initialize "$PHANET_DATA/phanet/glm" --device cuda --out taxonomy-combined-fit
```

`validation-native/dataset` contains the validation genomes from the same
prepared data. Network references must include the training hashes in
`reference-labels.tsv`; [build a custom network](custom-data.md#build-a-reference-network)
when using your own reference corpus. An existing reference TSV with the same
columns can be used directly, without a preliminary GLM fit.

Use the saved models with the [prediction commands](evidence-downstream.md#sequence-prediction).
