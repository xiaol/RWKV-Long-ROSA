# Frozen RWKV scene controls: 2026-09-26

Training semantic memory is working, but this experiment does not establish a
retrieval-specific improvement over a similarly sized local residual adapter.
The two-hop version does not beat one hop on the evaluated test rows.

## Protocol

- Frozen RWKV-7 0.4B; train the selected adapter and paragraph classifier only.
- Dataset: `mikuhhn1239/novel-agent-sft-dataset`, revision
  `5d3040d21f51b3ce90b9396b058e552c47f43cd5`, scene-boundary task.
- Official train rows partitioned into 1,437 training and 367 development rows
  with `prepare_novel_agent.py`, seed 0, validation fraction 0.2. The split
  manifest and JSONL files are in `../../data/scene-controls-20260926/`.
- Official test: 149 rows, 1,994 paragraph labels, 155 positive boundaries.
- Seeds 0 and 1; 500 updates, accumulation 2, AdamW at 1e-4, weight decay 0.01,
  gradient clipping 1.0, maximum length 4,096. Sampling is with replacement:
  1,000 examples per run, fewer than one traversal of the training partition.
- Adapter injection before block 6; semantic memory dimension 64, chunk size
  32, query block 128; classifier hidden width 128.
- The head is independently seeded across methods. All methods have identical
  initial training loss at the same seed because their residual starts at zero.
- Cutoffs selected separately per method and seed by maximum development F1
  over 0.1 through 0.9 in increments of 0.1. Ties prefer proximity to 0.5, then
  the lower cutoff. Selection is written before test evaluation.

## Results

These are independently trained methods, not inference-time removals of a
jointly trained adapter. Values are boundary micro F1 per test run; the mean
is the arithmetic mean across the two seeds.

| Method | Adapter parameters | Seed 0 F1 | Seed 1 F1 | Mean F1 | Cutoffs (0 / 1) |
|---|---:|---:|---:|---:|---|
| Head only | 0 | 0.2469 | 0.2391 | 0.2430 | 0.6 / 0.5 |
| Local residual | 273,540 | 0.2664 | 0.2642 | 0.2653 | 0.5 / 0.5 |
| Semantic one-hop | 272,577 | 0.2535 | 0.2813 | 0.2674 | 0.6 / 0.6 |
| Semantic two-hop | 272,577 | 0.2500 | 0.2524 | 0.2512 | 0.6 / 0.6 |
| Original ROSA input | 1,052,321 | 0.2016 | 0.2283 | 0.2149 | 0.5 / 0.4 |

The classifier adds 133,377 trainable parameters to every method. The ROSA
control uses `RosaInputAdapter` with four suffix candidates, not the original
vocabulary pointer-generator. It has more parameters and is evaluated at the
same layer as the semantic adapter; this is not a search for its best layer.

## Interpretation

The one-hop/local difference is only 0.0021 mean F1, and their ordering reverses
between seeds. Two seeds cannot establish a reliable advantage. Both local
and semantic adaptation can improve on the trained head-only control under
this short training budget, but semantic retrieval has not demonstrated a
distinct benefit. More hops alone did not help.

Earlier comparisons against `memory_disabled` overstated the evidence:
switching off an adapter after joint training changes the features the head
expects. A separately trained head-only model is the relevant baseline.
Earlier threshold sweeps on the test split also do not count as independent
calibration. This experiment uses training-only development rows for cutoff
selection, but the official test set was already inspected in previous pilots.

No source novel IDs are available in the prepared examples. Exact prompt
disjointness is checked, but common novels and overlapping windows may still
cross splits. These results are exploratory task measurements, not a claim of
novel-level generalization, long-context improvement, or multi-hop reasoning.

The next memory-training experiment should supply evidence supervision or
controlled examples requiring two linked facts, with irrelevant-memory and
fact-removal controls. Keep the local adapter baseline to test whether better
retrieval, rather than additional trainable capacity, explains any improvement.

## Artifacts and reproduction

All ten checkpoints, training logs, configurations, and development reports are
in `../results/runs/scene-controls-20260926/`. Each `test-*` directory contains
the frozen selections, checkpoint reload evaluations, and `comparison.json`.

Run from the repository root, using the prepared training and development files:

```bash
python scripts/train_scene_head.py train \
  --adapter-kind semantic --hops 1 --seed 0 \
  --train ../data/scene-controls-20260926/train.jsonl \
  --validation ../data/scene-controls-20260926/validation.jsonl \
  --output /path/to/new-one-hop-seed0 \
  --model 0.4b --device cuda:0 --layer 6 \
  --steps 500 --accum 2 --lr 1e-4 --max-length 4096 \
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9
```

Repeat with seed 1 and the documented `none`, `local`, `semantic --hops 2`,
and `rosa` controls. Use `evaluate_scene_controls.py` to select development
cutoffs and test the saved runs, as described in `SEMANTIC_MEMORY.md`.

Validation: 22 tests pass, including probability-threshold regression,
control variants, adapter disable behavior, and stable cutoff selection.
All ten trained checkpoints successfully reload for test evaluation.
