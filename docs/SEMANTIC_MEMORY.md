# Semantic multi-hop memory on a frozen RWKV

This experimental adapter replaces token-suffix matching with learned retrieval
over contextual hidden states. It uses the existing pre-block residual hook and
keeps every original RWKV parameter frozen. It is a separate selectable method:
the original ROSA automaton, pointer head, and their checkpoints remain available
as baselines. No benchmark improvement is established by this implementation.

The [matched scene-control experiment](SCENE_CONTROLS.md) compares a separately
trained head-only baseline, local adaptation, semantic memory, and original
ROSA input adaptation across two seeds. Semantic one-hop and local adaptation
perform similarly; additional hops have not established an improvement.
The [supervised state-tuning comparison](STATE_TUNING.md) adds learned initial
RWKV states, alone and with semantic memory; neither improves the matched
test result at the evaluated training budget.

## Mechanism

At an intermediate layer, normalize the incoming contextual hidden states and
pool them into fixed-size chunks. Train separate query, key, and value
projections. Each query attends to strictly earlier completed chunks using
cosine similarity with temperature 0.1. After each read, a learned update
combines the current query and retrieved value to form the next query. The
last read passes through a learned sigmoid gate and a projection back into the
RWKV residual stream. Subsequent frozen layers compute the final prediction.

The default is two hops, 64-dimensional memory, 32-token chunks, and injection
before block 6 (zero-based). The output projection starts at zero, so the
initial model is exactly the frozen baseline. The first optimization step
updates that projection; later steps also train the retrieval projections,
query update, normalization, and gate. Gradients flow through the frozen later
layers without updating their parameters.

The SFT runner permits only prompt tokens to write memory. Prior assistant
tokens still participate in normal causal RWKV computation and in the current
query, but cannot become stored retrieval values. This avoids counting repeated
answer JSON as successful retrieval from the input. The adapter receives no
labels or future-token information. Both training and generation use the same
completed-chunk visibility rule; a partial final prompt chunk becomes available
only after its nominal chunk end has passed.

This is soft attention over a compressed contextual memory bank, not exact
ROSA, learned binary ROSA-QKV, or a proven semantic reasoning system. Semantic
addressing must be learned from the supervised task. Multiple hops provide
the capacity to compose evidence; they do not guarantee reasoning improvements.

## Train on the novel-agent dataset

Use explicit JSONL train and validation files with system/user/assistant
messages, matching the dataset's schema. Obtain their split assignments from
the publisher or a documented book/document-level partition. The four cached
HF blobs alone do not establish the official split assignment. Do not combine
train and evaluation blobs or randomly split overlapping windows from the
same novel. The runner rejects identical user prompts across supplied splits;
it cannot detect shared books or all near-duplicates without source metadata.

When only JSONL blobs are available, `scripts/prepare_novel_agent.py` can make a
deterministic task-balanced pilot split, remove exact duplicate prompts, and
write a source-hash manifest:

~~~bash
python scripts/prepare_novel_agent.py \\
  --input /path/to/blob_a /path/to/blob_b \\
  --output /tmp/novel-agent-pilot --seed 0 --validation-fraction 0.2
~~~

This is a pilot split, not the publisher's official split. Without source or
document IDs, it cannot prevent examples from the same novel crossing partitions.

Run from the repository root:

~~~bash
python scripts/train_semantic.py train \
  --train /path/to/novel_train.jsonl \
  --validation /path/to/novel_validation.jsonl \
  --output results/runs/semantic_novel_2hop \
  --model 0.4b --layer 6 --hops 2 --memory-dim 64 \
  --chunk-size 32 --query-block 128 \
  --steps 200 --accum 4 --lr 1e-4 --max-length 4096
~~~

The model argument also accepts a checkpoint path. Use --vocab to override the
RWKV World vocabulary path. CUDA, the RWKV tokenizer, PyTorch, and the existing
RWKV CUDA extension build requirements are needed. The new semantic adapter
does not require compiling the ROSA CPU extension.

Only assistant tokens, including the answer terminator, contribute to loss.
Rows are sampled uniformly; one example is processed per microbatch and both
states are reset between examples. Long rows fail explicitly rather than
silently losing their prompt or answer. Start with separate task runs because
the dataset has far more scene examples than attribution or unit examples.

Outputs include an adapter-only checkpoint, training log, evaluation report,
and a manifest containing arguments, row counts, and SHA-256 hashes of the
backbone, vocabulary, and supplied data. Checkpoints support evaluation reload,
not optimizer resume. Output directories must be new or empty.

## Scene-boundary classifier

For the scene-boundary subset, `scripts/train_scene_head.py` trains a small
paragraph-level classifier together with the semantic adapter. The RWKV
backbone remains frozen. This measures the task directly instead of asking a
non-instruction-tuned checkpoint to generate valid JSON:

~~~bash
python scripts/train_scene_head.py train \
  --train /path/to/scene_train.jsonl \
  --validation /path/to/scene_validation.jsonl \
  --output results/runs/scene_head_2hop \
  --model 0.4b --layer 6 --hops 2 --memory-dim 64 \
  --chunk-size 32 --query-block 128 --steps 500 --accum 2 --lr 1e-4
~~~

The report compares `memory_disabled`, `one_hop`, and `multi_hop` using
boundary precision, recall, F1, exact match, and binary loss. `--threshold`
is applied to sigmoid probabilities, not raw logits. An optional development
sweep is enabled by `--thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9`.
Do not select a cutoff from test results. Earlier local pilot reports applied
0.5 to logits, equivalent to a probability cutoff of approximately 0.622459.

### Matched training controls

The scene runner supports the following independently trained controls:

| Arguments | Trainable components |
|---|---|
| `--adapter-kind none` | Scene head only on frozen RWKV features |
| `--adapter-kind local` | Per-token residual MLP and scene head |
| `--adapter-kind state` | Learned initial RWKV recurrent state and scene head |
| `--adapter-kind state_semantic --hops 1` | Initial recurrent state, one-hop memory, and scene head |
| `--adapter-kind semantic --hops 1` | One-hop memory and scene head |
| `--adapter-kind semantic --hops 2` | Two-hop memory and scene head |
| `--adapter-kind rosa` | Original suffix-based input adapter and scene head |

All original RWKV weights remain frozen. The head is independently seeded so
its initialization is identical for each method at the same seed. The default
local bottleneck is 132 units: 273,540 adapter parameters versus 272,577 for
semantic memory on RWKV-0.4B. The ROSA input control has about 1.05M parameters;
it is not parameter matched. It uses the original ROSA candidate embedding
adapter, not the vocabulary pointer-generator, whose outputs are token
probabilities rather than paragraph boundary scores.

State tuning uses the RWKV recurrence's actual initial state, with one learned
matrix per layer and attention head. The backbone weights remain frozen; the
CUDA recurrence receives the learned state and returns gradients to it. A zero
state is exactly the original model. Gradients are checked against a PyTorch
reference recurrence; the runner uses ordinary global gradient clipping. This is a
global state shared across examples, not a separate state inferred from each
prompt; compare it against a separately trained head-only control.

Only the attention matrix states are learned; time-mix and channel-mix previous
token vectors start at zero. RWKV-0.4B uses 24 layers, 16 heads, and a 64 by 64
matrix per head: 1,572,864 state parameters, plus the common 133,377-parameter
classifier. `state_semantic` adds the semantic memory parameters as well.
Every example starts from the same learned state; processing never mutates it
or carries the final state between examples. The state is stored in FP32 with
axes `[layer, head, value, key]` and saved with its dimensions in the checkpoint.

Use `--adapter-kind state` for state-only adaptation and
`--adapter-kind state_semantic --hops 1` for joint state and memory training.
Both use supervised weighted binary cross-entropy and keep all RWKV weights
frozen. In state-plus-memory reports, `memory_disabled` retains the learned
state, and `state_disabled` retains the learned memory. These are interventions
on a jointly trained model, not independently trained controls. Select cutoffs
on development reports using the same evaluator as the other methods.

`memory_disabled` is an inference intervention on a jointly trained model.
Its head was trained with the adapter enabled, so it is not a separately
trained head-only baseline. Likewise, `one_hop` in a two-hop report is an
intervention, not an independently optimized one-hop run. Gains against those
interventions alone do not establish that retrieval beats ordinary adaptation.

Reserve development rows from the official training split, then train all
controls using identical rows, seeds, steps, accumulation, and optimizer
settings. If source novel IDs are unavailable, a prompt-disjoint pilot split
does not establish novel-level generalization. The existing pilot splitter can
be applied to the official training file alone:

~~~bash
python scripts/prepare_novel_agent.py \
  --input /path/to/official_scene_train.jsonl \
  --output /path/to/scene_development --seed 0 --validation-fraction 0.2
~~~

Supply the resulting `train.jsonl` and `validation.jsonl` to the scene runner
and enable the development threshold sweep. After training, run this from the
repository root to freeze thresholds before evaluating the official test file:

~~~bash
python scripts/evaluate_scene_controls.py \
  --runs results/runs/scene_controls/head-only-seed0 \
         results/runs/scene_controls/one-hop-seed0 \
         results/runs/scene_controls/two-hop-seed0 \
  --test /path/to/official_scene_test.jsonl \
  --output results/runs/scene_controls/test --device cuda:0
~~~

The evaluator verifies source hashes and exact prompt disjointness, writes
`selection.json` before any test scoring, and records test metrics in
`comparison.json`. Ties in development F1 prefer the cutoff closest to 0.5,
then the lower cutoff. These checks cannot identify shared novels. The official
test split was already inspected in earlier pilots, so new results on it remain
exploratory even with development-only cutoff selection.

## Evaluate

~~~bash
python scripts/train_semantic.py evaluate \
  --checkpoint results/runs/semantic_novel_2hop/adapter.pt \
  --validation /path/to/novel_validation.jsonl \
  --output results/runs/semantic_novel_2hop_generation \
  --generate --max-new-tokens 1200
~~~

Every evaluation compares the same rows under three conditions:

| Condition | Interpretation |
|---|---|
| memory_disabled | Exactly zero residual; the frozen backbone baseline |
| one_hop | Same trained adapter with only its first read |
| multi_hop | The configured number of reads |

The one-hop intervention tests reliance on later reads, but is not a separately
optimized one-hop baseline. Train an additional run with --hops 1 for that
comparison, holding data, seeds, steps, and adapter dimensions fixed. A local
residual-adapter baseline would further distinguish adaptation from retrieval.

Without --generate, reports contain teacher-forced assistant NLL only. With
--generate, generation begins from the prompt alone and writes per-row
predictions. Metrics include strict JSON and schema validity, exact match,
scene-boundary micro precision/recall/F1, unit-label accuracy, candidate
accuracy, and uncertainty accuracy. Malformed/schema-invalid outputs fail exact
match; unit IDs are matched by their supplied values, not renumbered.
Boundary IDs are evaluated literally against the supplied labels; the runner
does not infer whether a boundary denotes the start or end of a paragraph.

Use held-out document groups and multiple seeds before claiming gains. Lower
teacher-forced JSON loss alone does not establish better semantic decisions.
For multi-hop claims, add tasks with two independently necessary facts and
intervene on those facts. For long-context claims, evaluate length and evidence
distance separately from the native short-context dataset.

## Cost and current limits

For T tokens, chunk size S, memory dimension D, and H hops, there are
M = ceil(T/S) memory entries. Retrieval costs O(H*T*M*D): it does not retain
ROSA's claimed constant amortized token cost. Stored projected keys/values
cost O(M*D). Query blocking bounds each score tensor to
batch * query_block * M; training checkpoints each query block to avoid
retaining every hop's score tensor. Backbone activations and vocabulary logits
can still dominate memory.

Generation currently recomputes the full prefix, as in the repository's
existing evaluation scripts. There is no integrated recurrent RWKV/semantic
memory serving cache yet. The memory is local to each forward pass and is
rebuilt deterministically, so there is no cross-example state leakage.

Run focused correctness tests with:

~~~bash
PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_semantic.py
~~~

Tests cover zero-init identity, causal prefix equivalence, future-token
gradients, empty and prompt-only memory, controlled two-hop address changes,
gradient flow through a frozen downstream layer, checkpoint state reload,
batch isolation, query blocking, assistant loss masks, and structured metrics.
