# Supervised initial-state tuning on scene boundaries

This experiment learns the initial attention matrix state of frozen RWKV-7
0.4B, with and without one-hop semantic retrieval. At the matched 500-update
budget, neither state-only nor joint state/memory training improves on the
earlier head-only, local-adapter, or semantic one-hop controls.

The combined method uses the modified semantic ROSA memory, not the original
suffix-matching ROSA input adapter. Initial-state tuning combined with original
ROSA has not been tested.

## Method and validation

`InitialStateTuner` learns an FP32 tensor with axes `[layer, head, value, key]`.
For this backbone its shape is `[24, 16, 64, 64]`: 1,572,864 parameters. Every
example starts with this same learned tensor, expanded across the batch;
processing does not modify it or carry a previous example's final state.
The time-mix and channel-mix previous-token vectors still start at zero.

The CUDA recurrence now accepts the initial state and returns its gradient.
The existing zero-state kernel path and model checkpoint loading remain
available. Trainable parameters are the initial matrix state, the common
133,377-parameter classifier, and optionally the 272,577-parameter semantic
memory. All pretrained RWKV weights remain frozen. State-only has more
trainable parameters than the local and semantic controls; this is a matched
data/optimizer budget comparison, not a parameter-matched comparison.

The corrected kernel passes comparisons against a straightforward PyTorch
recurrence for outputs and gradients of all six sequence inputs and the
initial state at lengths 16, 48, and 256. Further tests verify exact zero-state
equivalence, batch gradient accumulation, nonzero-state behavior, example
isolation, padding, checkpointing, frozen backbone gradients, and state reload.
The full RWKV-0.4B model also passed exact zero-state equivalence and finite
state-gradient checks on a scene example.

Earlier temporary `/tmp/scene-state-*` smoke runs used a faulty experimental
backward implementation and are invalid for assessing state tuning. That
implementation and its NaN-replacement workaround were removed. The results
below use the verified kernel, no gradient sanitization, unit state scaling,
and ordinary global gradient clipping at norm 1.0.

## Protocol

This uses the same dataset revision, model hash, tokenizer hash, and prepared
training/development split as `SCENE_CONTROLS.md`: 1,437 train rows, 367
development rows, and the publisher's 149 test rows. The test contains 1,994
paragraph labels and 155 positive boundaries. Exact prompt disjointness is
checked; source novel IDs are unavailable.

Both new methods use seeds 0 and 1, 500 updates, accumulation 2, AdamW learning
rate 1e-4, weight decay 0.01, and maximum length 4,096. Sampling is with
replacement, giving 1,000 example presentations per run. The classifier
initialization is shared across methods for each seed. The combination uses
one-hop semantic memory before block 6 with memory dimension 64 and chunk size
32. No state-specific learning-rate search or RL was performed.

The threshold is selected per run on development F1 from 0.1 through 0.9 in
increments of 0.1, then written to `selection.json` before test scoring. The
test evaluation has no threshold sweep. All four saved checkpoints reload
successfully; every one of the 2,000 training updates has finite logged loss
and gradient norm. The final state tensors are finite and nonzero.

## Test results

| Independently trained method | Seed 0 F1 | Seed 1 F1 | Mean F1 | Development cutoffs (0 / 1) |
|---|---:|---:|---:|---|
| Head only (prior control) | 0.2469 | 0.2391 | 0.2430 | 0.6 / 0.5 |
| Local adapter (prior control) | 0.2664 | 0.2642 | 0.2653 | 0.5 / 0.5 |
| Semantic one-hop (prior control) | 0.2535 | 0.2813 | 0.2674 | 0.6 / 0.6 |
| Initial state | 0.2072 | 0.2279 | 0.2175 | 0.5 / 0.5 |
| Initial state + semantic one-hop | 0.2593 | 0.1888 | 0.2240 | 0.7 / 0.5 |

Development F1 for state-only was 0.2471 / 0.2369; for state plus memory,
0.3307 / 0.2247. The joint method varies substantially between the two seeds.
There is no evidence here that this state-tuning setup improves test F1.
This does not rule out state tuning with different optimization or tasks:
the budget is short, the learning rate was held fixed, and only two seeds
were evaluated. The test set was inspected during earlier experiments, so
these remain exploratory results rather than a fresh final benchmark.

`state_disabled` and `memory_disabled` reports are inference interventions on
the jointly trained model, not independent baselines. In the joint model,
disabling memory retains the state, and disabling the state retains memory.
Use the independently trained rows above for the main comparison.

## Reproduction and artifacts

Run from the repository root:

```bash
python scripts/train_scene_head.py train \
  --adapter-kind state --seed 0 \
  --train ../data/scene-controls-20260926/train.jsonl \
  --validation ../data/scene-controls-20260926/validation.jsonl \
  --output /path/to/new-state-seed0 --model 0.4b --device cuda:0 \
  --steps 500 --accum 2 --lr 1e-4 --max-length 4096 \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

Use `--adapter-kind state_semantic --hops 1` for the joint method and repeat
with seed 1. Feed the resulting run directories into
`scripts/evaluate_scene_controls.py` with the official test path to select
development cutoffs and reload checkpoints for test scoring.

The four completed runs and their test selections/reports are under
`results/runs/scene-state-controls-20260927/`. Previous controls are under
`results/runs/scene-controls-20260926/`. The folder names are run identifiers.

Validation: `PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 pytest -q` passes
31 tests on CUDA, including six kernel/model state tests. CUDA state tests
skip when no CUDA device is available.

The next useful step is a development-only supervised learning-rate and
training-duration study, retaining the head-only and local controls. RL is
not justified by this result alone: the dataset already supplies direct
supervision, and these experiments have not established a state-tuning gain.
