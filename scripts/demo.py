"""Qualitative demo: plant a fact deep in a long book excerpt, then ask for it.

Prints greedy continuations from (a) frozen RWKV-7, (b) RWKV-7 + trained ROSA pointer head, and the
ROSA candidate chain at the question position (what the suffix automaton found, and how long the match was).

    python scripts/demo.py --run pointer_aug --L 16384 --depth 0.2
"""
import os, sys, json, random, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS, get_tokenizer
from rosa.adapters import RosaFeatures, RosaPointerHead
from rosa.ops import RosaStream
from rosa.gpu import pick_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="pointer_aug"); ap.add_argument("--model", default="0.4b"); ap.add_argument("--device", default="auto")
ap.add_argument("--L", type=int, default=16384); ap.add_argument("--depth", type=float, default=0.2); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fact", default="The secret code for Heron is 734912."); ap.add_argument("--question", default="The secret code for Heron is")
ap.add_argument("--n_new", type=int, default=8)
args = ap.parse_args()
if args.device == "auto": args.device = pick_gpu(8000)
dev = args.device; root = os.path.join(os.path.dirname(__file__), "..")
tok = get_tokenizer(); idx = json.load(open(f"{root}/data/tok/index.json"))
rng = random.Random(args.seed)
book = rng.choice([b for b, v in idx.items() if v["split"] == "val" and v["tokens"] > args.L + 1000])
a = np.load(f"{root}/data/tok/{book}.npy").astype(np.int64); s = rng.randrange(0, len(a) - args.L); body = a[s: s + args.L].tolist()
cut = int(args.depth * args.L); ctx = body[:cut] + tok.encode("\n" + args.fact + "\n") + body[cut:] + tok.encode("\n" + args.question)
print(f"book {idx[book]['title']!r}, context {len(ctx)} tokens, fact planted at token {cut} ({args.depth:.0%} depth)\n")

ck = torch.load(f"{root}/results/runs/{args.run}/ckpt.pt", map_location="cpu", weights_only=False); ca = ck["args"]
model = load_rwkv7(MODELS[ca["model"]], device=dev); C = model.args.n_embd
head = RosaPointerHead(C, K=ca["K"], use_h=not ca["no_h"], use_lpc=not ca.get("no_lpc", False)).to(dev); head.load_state_dict(ck["head"]); head.eval()

# what ROSA sees at the question position (streaming API, as a decoder would use it)
st = RosaStream(len(ctx) + args.n_new + 8, K=ca["K"]); pred, mlen, src, cnt = st.extend(ctx)
print("ROSA candidates for the next token (longest earlier-seen suffix first):")
for k in range(ca["K"]):
    if pred[k] >= 0:
        print(f"  k={k}: next={tok.decode([int(pred[k])])!r:12s} matched suffix len {int(mlen[k]):3d}  seen {int(cnt[k])}x  source pos {int(src[k])}")
print()

@torch.no_grad()
def gen(use_head):
    ids = list(ctx); out = []
    for _ in range(args.n_new):
        inp = torch.tensor([ids]); aux = RosaFeatures(ca["K"])(inp); aux = {k: v.to(dev) for k, v in aux.items()}; aux["idx"] = inp.to(dev)
        logits, h = model(inp.to(dev), aux=aux, return_hidden=True); lg = logits[:, -1:]
        if use_head:
            al = {k: v[:, -1:] for k, v in aux.items()}
            lpc = F.log_softmax(lg.float(), -1).gather(-1, al["pred"].clamp_min(0))
            log_g, log_1mg, log_a = head(h[:, -1:], lpc, al); lg = RosaPointerHead.mixture_logits(lg, log_g, log_1mg, log_a, al["pred"])
            g = float(log_g.exp())
        else:
            g = 0.0
        t = int(lg[0, -1].argmax()); out.append((t, g)); ids.append(t)
    return out

for name, use in [("RWKV-7 (frozen)", False), (f"RWKV-7 + ROSA pointer head ({args.run})", True)]:
    o = gen(use); txt = tok.decode([t for t, _ in o])
    gates = " ".join(f"{g:.2f}" for _, g in o) if use else ""
    print(f"{name}:\n  {args.question}{txt!r}" + (f"\n  copy gate per step: {gates}" if use else "") + "\n")
