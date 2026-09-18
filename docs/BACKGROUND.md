# Background: ROSA vs Engram, and what "long context" means for an RNN

## ROSA (RWKV-8 "Heron", BlinkDL, Oct 2025 – Feb 2026)

**ROSA = Rapid Online Suffix Automaton.** It is *not* an n-gram model. Given a token sequence
`x_0 … x_{n-1}`, ROSA computes, for every position `i`,

```
y_i = x_{j+1}   where j < i maximises m such that x_{j-m:j} == x_{i-m:i}   (ties: largest j)
y_i = -1        if no earlier occurrence of even the single token x_i exists
```

i.e. *the token that followed the longest earlier occurrence of the current suffix*. The match order is
unbounded (whatever the longest repeated suffix is), the memory is the suffix automaton of the context
itself (O(n) states, amortised O(1) per token), and the payload is a real token copied from the
context. There is no hash table, no learned table and no floating point in the core mechanism.

Sources (all in `BlinkDL/RWKV-LM`): `RWKV-8-ROSA.png` (the Oct-11-2025 one-pager, transcribed above),
`RWKV-8-ROSA-260120.png` (variants), `RWKV-v8/*.py` (toy demos), `RWKV-v8/README.md` (community repos).
BlinkDL's own characterisation: *"a neurosymbolic infinite-range lossless information propagator to
replace attention"*; *"Naïve ROSA obtains 100% MQAR and 100% NIAH by design, regardless of ctxlen.
But it's useless for most real tasks, for obvious reasons."*

How BlinkDL wires ROSA into a network (three designs, in chronological order):

1. **Emb(ROSA(x))** – add an embedding of the ROSA-predicted token to early layers
   (`251014_rosa_onlyemb_train.py`). This is the token-level form and the one this repo builds on.
2. **ROSA-1bit inner-monologue layers** – binarise every hidden channel (`x>0`), run ROSA on each
   0/1 sequence, emit `±e_c`. Gradients by bit-flip finite differences.
3. **ROSA-QKV-1bit** (current "best overall design") – trainable Q/K/V projections, binarised;
   match Q's suffix in K, emit V's bit at the successor position as `±e_c`. The official repo ships
   *no backward* for this op; several community repos exist only to provide one
   (`wjie98/rosa_soft`, `johanwind/wind_rosa`, `xiaoiecc/qkv-rosa-fast-exact-backward`, `zyaaa-ux/ROSA-Tuning`).

Closest prior work to what we do here: **ROSA-Tuning** (arXiv 2602.02499, Jan 2026) adds token-level
ROSA as a CPU retrieval side channel to a *windowed-attention* Qwen3 and recovers long-context
perplexity (PG-19 16k: 74.5 windowed → 17.6 with ROSA) and NIAH-32k (6 → 100). Nobody had done this
for RWKV-7, whose fixed-size state makes it the natural host.

## Engram (DeepSeek, arXiv 2601.07372, Jan 2026)

"Conditional Memory via Scalable Lookup". Engram *is* n-gram based, but **parametric**: the last
2–3 (tokenizer-normalised) token ids are multi-head hashed (multiplicative-XOR, prime moduli, 8
heads) into large **learned** embedding tables (2.3M–7.2M rows × 1280 d per layer at 27B–40B), the
retrieved vector is gated against the hidden state
(`alpha = sigmoid(RMSNorm(h)·RMSNorm(W_K e)/sqrt(d))`), passed through a short depthwise conv and
added to the residual stream at layers 2 and 15. It stores *pretrained knowledge*, not the current
context: ablating it at inference removes 56–71 % of factual-QA accuracy but only 7–19 % of
reading-comprehension accuracy. Its long-context gains (RULER MQ-NIAH 84 → 97) are indirect: local
n-gram reconstruction is offloaded, freeing attention for global context.

**So: ROSA = context memory (exact, unbounded order, copies from history). Engram = parametric memory
(hashed fixed-order n-gram, learned tables).** The ROSA-Tuning authors make the same distinction and
note the two are complementary. For an RNN that *cannot* look back into its context, only the
former restores long-range dependence; this repo tests exactly that and includes an Engram-lite
adapter as the parametric control.

## What "long context dependence" means for RWKV-7 (measured here)

RWKV-7 0.4B's average NLL on held-out novels is *flat* from 1k to 32k tokens (2.71 → 2.78); it does
not "forget how to model language". What it loses with distance is **exact retrieval**: a
needle-in-a-haystack fact inserted at 10 % depth of a 16k context is retrieved 0 % of the time
(95 % at 90 % depth), and at 32k it is 0 % / 0 % / 35 % by depth. That is precisely the
capability ROSA has *by construction*, and the one our adapters restore.
