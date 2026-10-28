# Stack LoRA adaptation on TF/time data

This study adapts the released **Stack-Large-Aligned** checkpoint. It tests
parameter-update constraints while retaining native teacher alignment. It does
not retrain the original foundation model or replace its initial alignment stage.

The completed parent benchmark lives on Modal volume `gentonik-data-20260925`,
prefix `followup/stack_tf_timing_20260929`. Its admitted counts, gene vocabulary,
splits, fixtures, and original results are mounted read-only. New artifacts live
on the separate v2 volume `stack-lora-tf-time-20261028`.

## Model changes

| Group | Stack attributes | Large-model dimensions |
| --- | --- | --- |
| Attention across cells | `layers.*.gene_attn.{qkv,proj}` | Width 1,600 |
| Attention within a cell | `layers.*.cell_attn.{qkv,proj}` | Width 16 |
| Token MLP | `layers.*.mlp.{0,3}` | 16 → 64 → 16 |
| Input projection | `gene_reduction.0` | 15,012 → 1,600 |
| Decoder | `output_mlp.{0,3}` | 1,600 → 3,200 → 30,024 |

LoRA starts with zero effective updates. The query embedding and confidence
classifier remain trainable. Other backbone parameters are frozen. The teacher
stays dense and averages the **effective** student weights every 500 updates,
with decay 0.95. Averaging adapter factors independently would produce a different
teacher. The objective remains native expression/embedding distributional
alignment, reconstruction, sliced-Wasserstein regularization, and classification.

## Frozen protocol

`study_config.json` specifies the protocol. The corpus has 2,244,047 cells,
including 1,419,454 training cells, in the published 15,012-gene vocabulary.
The mixture is 20% Lin2025, 5% Lin2021, 25% general replay, and 50% prior neural
data. Each update uses four microbatches of four sets of 512 cells, including
128 prompt cells per set. The AdamW schedule has 50 warmup updates and a
4,000-update cosine horizon. AdamW weight decay is 0.003; gradient clipping is 1.
Training uses BF16 autocast and the parent's numerically checked SDPA attention.
Inference uses exported dense checkpoints and the parent's eager, five-step
bounded-MDM procedure, with unchanged prompts and metrics.

The pilot grid has 27 arms:

- Across-cell attention, both attention axes, and across-cell attention plus
  input/decoder projections; each at ranks 8/32/128 and learning rates 2e-5/1e-4.
  Within-cell attention uses rank 2 when enabled.
- Full fine-tuning at learning rates 5e-6/2e-5.
- Within-cell-only, input/decoder-only, and all-group controls at two learning
  rates; the all-group control uses rank 16 on wide layers and rank 2 on small layers.
- A query-embedding/classifier-only control.

All pilots run 500 updates from the same initialization with seed 0. Adapter
initialization restores the main RNG state, preserving matched mask/time draws.
The rank range was widened before pilot-result inspection because the full-FT
update diagnostic showed substantial energy above rank 16 in the wide layers.

Selection uses all 15 Lin2025 intervention and two NGN2-time validation tasks,
plus four tasks per retained source. The primary score is 80% intervention / 20%
time relative MSE. Retention requires old-neural MSE and replay MSE no worse than
1.10× the released model, and replay correlation at least 0.90× its baseline.
The two best distinct LoRA placements and best full-FT learning rate advance to
4,000 updates with seeds 0/1/2. A failing-retention promotion is explicitly
exploratory. Each arm selects one common checkpoint step across training seeds
using validation alone. Both selected and final-budget checkpoints are tested.

The external panels contain 186 tasks per endpoint model, including held-out
recipes, combinations, time, selected TF-KO responders, line transfer, older
neural tasks, chemicals, and transplant/culture-age tasks. Unchanged and source-mean
baselines remain. Selected models receive wrong-prompt controls and three RNA
generation seeds on the fixed 77-task repeat panel. All three frozen GenTonik
readouts and three generation seeds are retained. Phenotypes and test results
are never used for training, model choice, or readout choice.

Lin2025 primarily measures fate patterning within one cell line. NGN2 collection
time has sample/media confounding. Author-selected TF-KO responders do not measure
unbiased knockout efficacy. These limits carry over from the parent benchmark.

## CLI execution

The source benchmark package and its configuration must be available locally.
Their exact code hashes and source bundles are retained with each launch.

```bash
export STACK_BENCHMARK_SOURCE=/path/to/completed/benchmark/workspace
export STACK_LORA_RUN_DIR=/path/to/local/run-ledger

python3 experiments/tf_time_lora/plan.py
python3 experiments/tf_time_lora/freeze_sources.py
modal run --detach experiments/tf_time_lora/modal_app.py --specs /path/to/specs.json
modal run experiments/tf_time_lora/modal_app.py --action collect
```

`main` records function-call IDs before returning. Each worker writes a separate
success/failure receipt. A local CLI timeout does not imply a failed worker.
`kind=metadata` bundles JSON/CSV artifacts without downloading model weights.
After extracting that bundle to `$STACK_LORA_RUN_DIR/metadata`, use:

```bash
python3 experiments/tf_time_lora/review.py pilot
# Submit the generated long_specs.json, then collect/extract its metadata.
python3 experiments/tf_time_lora/review.py long
python3 experiments/tf_time_lora/review.py tests
```

Each training run retains ordinary merged `.ckpt` inference files, compact
adapter `.pt` files, and a full `resume.pt` with student, dense EMA teacher,
optimizer, RNGs, and sampler state. Source/input hashes must match on resume.
SDPA continuation is not claimed to be bit-exact. `checkpoint_audit` reconstructs
exports, frozen parameters, optimizer steps, and teacher updates independently;
`metric_audit` recalculates metrics from retained generated counts and fixtures.

The pilots used seven concurrent workers. Long training and inference use up to
nine concurrent workers, with one separate diagnostic worker, within a ten-GPU
ceiling. Training and external-test phases run sequentially. Inference uses
64 GiB of host RAM; corpus-loading training retains the parent's 192 GiB.
No unrelated Modal apps or data are changed.
