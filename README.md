# RWKV-Long-ROSA

**Giving a frozen RWKV-7 long-context dependence with ROSA (Rapid Online Suffix Automaton).**

RWKV-7 is an RNN with a fixed-size state. It models prose as well at 32k tokens as at 1k, but it
cannot *retrieve*: a fact planted 16k tokens back is recovered 0 % of the time, and a passage it has
already seen is predicted no better than new text once it is more than a few thousand tokens away.
ROSA, BlinkDL's RWKV-8 mechanism, is an exact, parameter-free suffix-match memory over the context.
This repo implements ROSA as a fast CPU op, bolts it onto frozen RWKV-7 through small trained
adapters, and measures what it fixes and what it does not.

See `docs/BACKGROUND.md` for what ROSA is, how it differs from DeepSeek's Engram (parametric hashed
n-gram tables), and the sources. `docs/RESULTS.md` has the full tables.

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
