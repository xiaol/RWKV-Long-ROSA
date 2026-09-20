# RWKV-Long-ROSA

**Giving a frozen RWKV-7 long-context dependence with ROSA (Rapid Online Suffix Automaton).**

RWKV-7 is an RNN with a fixed-size state. It models prose as well at 32k tokens as at 1k, but it
cannot *retrieve*: a fact planted 16k tokens back is recovered 0 % of the time, and a passage it has
already seen is predicted no better than new text once it is more than a few thousand tokens away.
ROSA, BlinkDL's RWKV-8 mechanism, is an exact, parameter-free suffix-match memory over the context.
This repo implements ROSA as a fast CPU op, bolts it onto frozen RWKV-7 through small trained
adapters, and measures what it fixes and what it does not.

See `docs/BACKGROUND.md` for what ROSA is, how it differs from DeepSeek's Engram (parametric hashed
n-gram tables), and the sources. `docs/RESULTS.md` has the full tables and ablations.

## Headline results (frozen RWKV-7, 0.1M-parameter ROSA pointer head trained on 33M tokens)

Needle in a haystack, exact match of a 6-digit code planted at depth 10 % / 50 % / 90 % (n = 20):

| context | RWKV-7 0.4B | + ROSA head | RWKV-7 1.5B | + ROSA head |
|---|---|---|---|---|
| 16k | .00 / .05 / .95 | .90 / .95 / 1 | .45 / .55 / .90 | 1 / .95 / 1 |
| 32k | .00 / .00 / .35 | .95 / .85 / 1 | .00 / .00 / .35 | 1 / .95 / 1 |
| 32k, 4 distractor codes | .00 / .00 / .10 | .95 / 1 / 1 | | |
| 64k (n = 10) | 0 / 0 / 0 | 1 / .9 / 1 | | |

Passage copy, NLL per token on a 256-token passage repeated 30k tokens later: 3.15 → 0.16 (0.4B),
2.80 → 0.28 (1.5B). Code (held-out Python source), NLL past 16k: 1.012 → 0.961 (0.4B, −5 % ppl overall).
Prose: unchanged (the RNN's loss is flat with position; ROSA's prose candidates are 87 % wrong and the
gate learns to ignore them). Engram-lite (parametric n-gram tables, 17M params) does nothing on any
retrieval test, as expected for a memory that does not read the context.

```
$ python scripts/demo.py --run pointer_aug --L 32768 --depth 0.5 --seed 1
book 'Siddhartha', context 32789 tokens, fact planted at token 16384 (50% depth)
ROSA candidates for the next token (longest earlier-seen suffix first):
  k=0: next=' 73'   matched suffix len   8  seen 1x  source pos 16392
  k=1: next=' out'  matched suffix len   1  seen 154x
RWKV-7 (frozen):            The secret code for Heron is':\n\n“What does the word,'
RWKV-7 + ROSA pointer head: The secret code for Heron is' 734912.\nas a sort'
  copy gate per step: 0.31 0.67 0.69 0.90 0.89 0.10 0.01 0.01
```

## How it works

1. **ROSA** (`rosa/csrc`): a suffix automaton over the token ids seen so far. For each position it returns
   the K longest suffixes that occurred earlier, each with the token that followed, the match length,
   how often it was seen, and where. Exact, parameter-free, CPU, 12 µs/token streaming.
2. **Frozen RWKV-7** produces its usual next-token distribution `p_rwkv` and hidden state `h`.
3. **Pointer head**: `p = (1-g)·p_rwkv + g·Σ_k a_k·onehot(candidate_k)`. `g` and `a_k` are a small MLP
   over `LayerNorm(h)`, `log(match len)`, `log(count)`, and `log p_rwkv(candidate_k)`. Trained with the LM
   loss of the mixture on the target token only (no full-vocab softmax needed beyond the backbone's).
4. **Repeat augmentation**: half of the training sequences get one to three earlier spans pasted later.
   Novels contain too few long verbatim repeats for the gate to learn to trust a once-seen 8-token match at
   30k tokens; this supplies them and turns a modest code-perplexity gain into full NIAH recovery.

What did *not* work on a frozen backbone at this budget: injecting `Emb(ROSA(x))` into the residual stream
(BlinkDL's input-side design) and the Engram-lite adapter both raised loss; see `docs/RESULTS.md` §2d.

## What is here

| path | what |
|---|---|
| `rosa/csrc/rosa_cpu.cpp` | C++/OpenMP suffix automaton. `rosa_tokens`: BlinkDL's ROSA over token ids, returning the K longest previously-seen suffixes per position with successor token, match length, occurrence count, source position. `rosa_qkv`: BlinkDL's ROSA-QKV (match Q's suffix in K, read V) for small alphabets. `RosaStream`: incremental version for decoding, 12 µs/token. All verified against BlinkDL's reference code and a brute-force oracle (`tests/`). |
| `rosa/rwkv7.py` | RWKV-7 in trainable GPT mode with the official `wind_backstepping` CUDA kernel and per-layer adapter hooks. Loads the released `.pth` checkpoints. |
| `rosa/adapters.py` | `RosaPointerHead`: pointer-generator mixture `p = (1-g)·p_rwkv + g·Σ_k a_k·onehot(ROSA_k)` with `g`, `a_k` from a 0.1M-parameter MLP over (hidden state, match length, occurrence count, RWKV's own probability of the candidate). `RosaInputAdapter`: BlinkDL's `Emb(ROSA(x))` added to the residual stream. `EngramLiteAdapter`: DeepSeek Engram (hashed 2/3-gram tables + context gate) as the parametric control. |
| `scripts/` | corpus prep (Project Gutenberg, Python site-packages), zero-training headroom diagnostic, needle-in-a-haystack, passage-copy, adapter training and per-position evaluation. |

## Setup

```bash
pip install -r requirements.txt          # torch, numpy, rwkv (tokenizer), ninja
# RWKV-7 World checkpoints + vocab: edit MODELS / VOCAB in rosa/rwkv7.py
python tests/test_rosa_ops.py            # builds the C++ op, ~1 min
python scripts/prep_gutenberg.py         # expects data/gutenberg/pg*.txt
python scripts/diag_headroom.py --model 0.4b && python scripts/diag_report.py
python scripts/niah_eval.py --model 0.4b
python scripts/train_adapter.py --name pointer --mode pointer --data tok,tok_code --repeat_aug 0.5
python scripts/eval_adapter.py pointer --data tok_code; python scripts/copy_eval.py pointer
python scripts/niah_eval.py --runs pointer --skip_rule
```

Everything ran on one A100-40GB (shared) plus CPU; ROSA itself needs no GPU.
