# Results

All numbers: frozen RWKV-7 World checkpoints (`rwkv7-g1d-0.4b`, `rwkv7-g1j-1.5b`), held-out Project
Gutenberg novels (val = book id % 5 == 0) and held-out Python source docs, first 32k tokens of each.
Adapters were trained for 1000 steps × 4 sequences × 8192 tokens (≈33M tokens, ~25–45 min on one A100).
Raw logs are in `results/`; tables in `results/eval/*.json`, `results/niah/*.json`, `results/copy/*.json`.

## 1. Zero-training diagnostic: where does the RNN lose information that the context still has?

`scripts/diag_headroom.py` / `scripts/diag_report.py`. ROSA(k=0) = BlinkDL's prediction (successor of the
longest previously-seen suffix). "cover" = a previous occurrence exists; "acc" = ROSA's token is the target.

**Novels, 0.4B, 24 val books (786k tokens):**

| position | RWKV NLL | RWKV top-1 | ROSA cover | ROSA acc | ROSA acc given cover | mean match len |
|---|---|---|---|---|---|---|
| 0–1k | 2.712 | 0.480 | 0.53 | 0.106 | 0.200 | 0.9 |
| 1k–2k | 2.848 | 0.437 | 0.67 | 0.085 | 0.127 | 1.0 |
| 2k–4k | 2.770 | 0.437 | 0.75 | 0.091 | 0.122 | 1.1 |
| 4k–8k | 2.791 | 0.434 | 0.81 | 0.101 | 0.126 | 1.3 |
| 8k–16k | 2.815 | 0.431 | 0.86 | 0.111 | 0.130 | 1.5 |
| 16k–32k | 2.783 | 0.432 | 0.90 | 0.126 | 0.140 | 1.7 |

The RNN's loss is flat with position. ROSA's top candidate is right only 13 % of the time on prose, and
when it *is* a long match RWKV already knows it (match ≥ 9 tokens: ROSA 60 % right, RWKV top-1 70 %).
A training-free per-bucket mixture gains 0.001 nats. **On prose there is almost no average-loss headroom.**

**Code, 0.4B, 24 held-out docs:**

| position | RWKV NLL | ROSA cover | ROSA acc | ROSA acc given cover | mean match len |
|---|---|---|---|---|---|
| 0–1k | 1.595 | 0.69 | 0.32 | 0.46 | 3.8 |
| 4k–8k | 1.152 | 0.91 | 0.44 | 0.48 | 7.5 |
| 8k–16k | 1.096 | 0.94 | 0.52 | 0.55 | 30 |
| 16k–32k | 1.094 | 0.96 | 0.53 | 0.55 | 29 |

| ROSA match len | frac of tokens | ROSA acc | RWKV top-1 | RWKV NLL | RWKV p(ROSA token) | NLL if ROSA hits were free |
|---|---|---|---|---|---|---|
| 1 | 0.22 | 0.23 | 0.61 | 1.74 | 0.23 | 1.66 |
| 4–5 | 0.13 | 0.62 | 0.80 | 0.84 | 0.59 | 0.69 |
| 9–16 | 0.08 | 0.81 | 0.87 | 0.54 | 0.76 | 0.36 |
| ≥17 | 0.12 | **0.95** | 0.91 | 0.36 | 0.86 | 0.11 |

Twelve percent of code tokens sit on a ≥17-token exact repeat that ROSA resolves with 95 % accuracy while
the RNN assigns it only 0.86 probability. Training-free mixture: −0.03/−0.04 nats at 8k–16k/16k–32k.

**Exact retrieval (needle in a haystack), 0.4B.** A sentence "The secret code for NAME is DDDDDD." is
planted at a depth in a haystack of held-out novel text; the model is asked to complete "The secret code
for NAME is". Exact match of the 6-digit code, greedy, n = 20 per cell. `gated` = the parameter-free rule
"emit ROSA's token if its match length ≥ 3, else RWKV's argmax".

| L | depth | RWKV-7 0.4B | RWKV-7 1.5B | ROSA alone | gated rule |
|---|---|---|---|---|---|
| 4k | .1 / .5 / .9 | .95 / 1 / 1 | – | 1 / 1 / 1 | 1 / 1 / 1 |
| 8k | .1 / .5 / .9 | .70 / .65 / 1 | .95 / .90 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| 16k | .1 / .5 / .9 | **.00 / .05 / .95** | .45 / .55 / .90 | 1 / 1 / 1 | 1 / 1 / 1 |
| 32k | .1 / .5 / .9 | **.00 / .00 / .35** | **.00 / .00 / .35** | 1 / 1 / 1 | 1 / 1 / 1 |
| 32k, 4 distractor codes | .1 / .5 / .9 | .00 / .00 / .10 | – | 1 / 1 / 1 | 1 / 1 / 1 |

Both RNNs lose the fact once it is ~10k tokens back. ROSA never does, by construction. **This is what
"long-context dependence" means for RWKV: not average loss, but exact retrieval.**

## 2. Trained adapters on the frozen 0.4B backbone

| run | what | trainable params | data |
|---|---|---|---|
| `pointer_k4` | pointer-generator head, K=4 candidates | 0.115M | novels |
| `pointer_mix` | same | 0.115M | novels + code |
| `pointer_aug` | same, **+ repeat augmentation** (50 % of samples get 1–3 earlier 64–512-token spans pasted later) | 0.117M | novels + code |
| `pointer_aug_nolpc` | `pointer_aug` but the gate cannot see RWKV's own probability of the candidate | 0.117M | novels + code |
| `input_k4_l2` | BlinkDL's Emb(ROSA(x)) added to the residual before block 2 | 1.05M | novels |
| `both_k4_l2` | `input_k4_l2` + pointer head | 1.17M | novels |
| `engram_l1` | Engram-lite: hashed 2/3-gram tables (4 heads × 65536 rows × 32 d per order) + context gate + conv before block 1 | 17.3M | novels |

### 2a. Needle in a haystack, greedy decoding through the mixture (n = 20)

| L | depth | RWKV | `pointer_k4` | `pointer_mix` | `both_k4_l2` | **`pointer_aug`** |
|---|---|---|---|---|---|---|
| 8k | .1 / .5 / .9 | .70 / .65 / 1 | .85 / .80 / 1 | 1 / 1 / 1 | 1 / 1 / 1 | 1 / 1 / 1 |
| 16k | .1 / .5 / .9 | .00 / .05 / .95 | .05 / .05 / .70 | .05 / .05 / .75 | .00 / .00 / .55 | **.90 / .95 / 1** |
| 32k | .1 / .5 / .9 | .00 / .00 / .35 | .00 / .00 / .25 | .00 / .00 / .30 | .00 / .00 / .10 | **.95 / .85 / 1** |
| 32k, 4 distractors | .1 / .5 / .9 | .00 / .00 / .10 | – | – | – | **.95 / 1 / 1** |

Heads trained on natural text alone learn a gate that is too timid for a once-seen 8-token match at 16k+
(natural repeats that long are rare in novels: 0.03 % of tokens). Repeat augmentation supplies them and
the gate learns to trust a long, once-seen match. The demo (`scripts/demo.py`, `results/demo_*.txt`)
shows the gate opening to 0.6–0.98 on the six code tokens and closing to 0.01 right after.

### 2b. Passage copy: NLL on a 256-token passage repeated `dist` tokens later (val novels, 12 cases each)

| dist | RWKV | `pointer_k4` | `both_k4_l2` | `engram_l1` | **`pointer_aug`** | neighbouring text (RWKV) |
|---|---|---|---|---|---|---|
| 512 | 1.107 | 0.825 | 0.768 | 1.161 | **0.144** | 2.77 |
| 2k | 2.222 | 1.496 | 1.354 | 2.276 | **0.147** | 3.03 |
| 8k | 2.854 | 1.968 | 1.877 | 2.888 | **0.153** | 2.82 |
| 16k | 2.952 | 1.962 | 1.805 | 2.969 | **0.170** | 2.86 |
| 30k | 3.148 | 2.031 | 1.883 | 3.157 | **0.164** | 3.09 |

RWKV-7 gets ~1 nat of benefit from a repeat 512 tokens back and none at 8k+. With the augmented pointer
head the copy costs 0.15 nats/token at any distance, i.e. the passage is essentially free. Engram-lite,
being parametric, does nothing here (as expected).

### 2c. Language-model loss by position, held-out (12 docs × 32k each)

Code (NLL, Δ vs frozen backbone):

| position | RWKV | `pointer_k4` | `pointer_mix` | `pointer_aug` | `pointer_aug_nolpc` | `input_k4_l2` | `both_k4_l2` | `engram_l1` |
|---|---|---|---|---|---|---|---|---|
| 0–1k | 1.450 | −0.000 | −0.000 | +0.001 | +0.002 | +0.119 | +0.111 | +0.051 |
| 2k–4k | 1.284 | −0.003 | −0.004 | −0.003 | −0.001 | +0.183 | +0.151 | +0.031 |
| 4k–8k | 1.127 | −0.006 | −0.008 | −0.009 | −0.007 | +0.195 | +0.157 | +0.026 |
| 8k–16k | 1.012 | −0.020 | −0.025 | **−0.040** | −0.036 | +0.190 | +0.126 | +0.025 |
| 16k–32k | 1.012 | −0.028 | −0.033 | **−0.051** | −0.047 | +0.205 | +0.129 | +0.026 |
| overall ppl | 2.91 | 2.85 | 2.84 | **2.81** | 2.82 | 3.53 | 3.32 | 2.99 |

By match length (code, `pointer_aug`): ≥17-token matches 0.293 → 0.146; 9–16: 0.494 → 0.435; 1: unchanged.

Novels (NLL): frozen 2.8104 (ppl 16.62) → `pointer_aug` 2.8085 (16.59), `pointer_k4` 2.8077 (16.57);
`input_k4_l2` 2.8519 (17.32), `both_k4_l2` 2.8490 (17.27), `engram_l1` 2.8279 (16.91). The pointer heads
never hurt at any position; the gains are in the 16k–32k bucket (−0.003/−0.004) and on long matches
(≥6 tokens: −0.03). 

### 2d. Ablations

* **Repeat augmentation** is what turns the head from "helps on code" into "restores retrieval":
  NIAH 32k .00/.00/.30 → .95/.85/1, copy@30k 2.03 → 0.16, code 16k–32k −0.033 → −0.051. It costs nothing
  on natural text (novels 2.8078 vs 2.8085; within noise).
* **RWKV's own probability of the candidate as a gate input** (`nolpc` ablation) is worth ~10 % of the
  code gain and half of the (small) novel gain: the gate needs to know whether the backbone already
  predicts the copy.
* **Input-side injection on a frozen backbone hurts at this budget.** Emb(ROSA(x)) at layer 0 (before the
  input LayerNorm), layer 2, and Engram-lite at layer 1 all raise loss everywhere (+0.02–0.05 nats on
  novels, +0.03–0.2 on code) after 33M tokens, even with zero-initialised output projections and a
  10× lower learning rate; their eval loss was still improving at step 1000, so this is a budget/
  optimisation limit, not a proof they cannot work. They do learn *something* (both_k4_l2 halves the
  copy penalty at 30k; NIAH 8k → 1.0), but the pointer head does the same job with 1/10 the parameters
  and no risk to the backbone. With a trainable backbone (BlinkDL's setting) the picture may differ.
* **Engram-lite** confirms the parametric/contextual split: it is the only adapter with zero effect on the
  copy task at any distance and on NIAH.

## 3. 1.5B backbone

RWKV-7 1.5B alone also fails NIAH at 32k (.00/.00/.35). With its own `pointer_aug_1.5b` head (0.13M params, same recipe):

| L | depth | RWKV 1.5B | `pointer_aug_1.5b` |
|---|---|---|---|
| 16k | .1 / .5 / .9 | .45 / .55 / .90 | 1 / .95 / 1 |
| 32k | .1 / .5 / .9 | .00 / .00 / .35 | 1 / .95 / 1 |

Passage copy (NLL on the repeated 256 tokens): 512: 0.513 → 0.134; 2k: 1.179 → 0.150; 8k: 2.264 → 0.210;
16k: 2.547 → 0.252; 30k: 2.804 → 0.276. Code NLL: overall 0.699 → 0.684 (ppl 2.01 → 1.98), 16k–32k
0.638 → 0.618 (−0.020). Novels: unchanged (2.4840 → 2.4837, ppl 11.99). The stronger backbone already exploits repeats up to ~2k
tokens back (copy@512 costs 0.51 vs 1.11 for 0.4B) but is just as blind past 8k; the same tiny head fixes it.

**64k tokens (0.4B, `pointer_aug`, n = 10 per cell):** NIAH 1.0 / 0.9 / 1.0 at depths .1/.5/.9; RWKV alone 0.0.
The adapter was trained at 8k context; ROSA has no length to extrapolate.

## 4. Cost

ROSA runs on CPU: 8 × 16k tokens in 14–29 ms batched, 12 µs/token streaming (`RosaStream`, 83k automaton
states for 64k tokens). The pointer head is one 128-wide MLP per candidate on top of the RWKV hidden state,
plus one gather from the logits. Memory is O(context) integers, no KV cache, no float attention.

## 5. What this does and does not show

* Does: an exact, unbounded-order suffix memory bolted onto a frozen RNN through a 0.1M-parameter
  output-side gate restores verbatim long-range retrieval (NIAH, multi-key NIAH, passage copy) at 16k–64k
  where the RNN alone fails, and gives a real perplexity gain on repetitive domains (code −5 % ppl,
  −0.05 nats past 16k). ROSA is the right tool for this; a parametric n-gram memory (Engram) is not.
* Does not: improve average loss on prose (RWKV-7 already models what it can there; ROSA's prose candidates
  are 87 % wrong), and does not help semantic (non-verbatim) long-range dependence — a paraphrased question
  gets no ROSA candidate. That would need BlinkDL's learned ROSA-QKV (matching learned bit-strings rather
  than token ids), which needs a trainable backbone and a gradient method the official repo does not ship.
