"""Experiment 1: zero-training diagnostic on long documents.

For every token of each held-out book (up to --max_t tokens), record
  * RWKV-7 NLL and top-1 hit, and RWKV-7's probability of the ROSA candidate
  * ROSA (k=0..K-1) candidate token, matched suffix length, occurrence count, hit
so we can answer: where does the frozen RNN lose information that exact suffix matching over the
context still has?  Saves results/diag/<model>/<book>.npz.
"""
import os, sys, json, math, time, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS
from rosa.ops import rosa_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="0.4b"); ap.add_argument("--device", default="cuda:1")
ap.add_argument("--max_t", type=int, default=32768); ap.add_argument("--split", default="val"); ap.add_argument("--K", type=int, default=4)
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--data", default="tok")
args = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), ".."); idx = json.load(open(f"{root}/data/{args.data}/index.json"))
out = f"{root}/results/diag/{args.model}" + ("" if args.data == "tok" else "_" + args.data); os.makedirs(out, exist_ok=True)
model = load_rwkv7(MODELS[args.model], device=args.device)
books = [b for b, v in sorted(idx.items(), key=lambda kv: str(kv[0]).zfill(8)) if v["split"] == args.split]
if args.limit: books = books[:args.limit]
for bid in books:
    dst = f"{out}/{bid}.npz"
    if os.path.exists(dst): continue
    ids = np.load(f"{root}/data/{args.data}/{bid}.npy")[: args.max_t].astype(np.int64)
    x = torch.from_numpy(ids)[None]
    t0 = time.time(); pred, mlen, src, cnt = rosa_tokens(x, K=args.K); t_rosa = time.time() - t0
    xg = x.to(args.device)
    with torch.no_grad():
        t0 = time.time(); logits = model(xg)[0]; torch.cuda.synchronize(); t_rwkv = time.time() - t0
        T = len(ids) - 1
        nll = torch.empty(T); top1 = torch.empty(T, dtype=torch.bool); p_rosa = torch.empty(T, args.K)
        tgt = xg[0, 1:]
        for s in range(0, T, 4096):
            e = min(T, s + 4096)
            lp = F.log_softmax(logits[s:e].float(), -1)
            nll[s:e] = -lp.gather(1, tgt[s:e, None])[:, 0].cpu()
            top1[s:e] = (lp.argmax(-1) == tgt[s:e]).cpu()
            pr = pred[0, s:e].to(args.device).clamp_min(0)
            p_rosa[s:e] = lp.gather(1, pr).exp().cpu() * (pred[0, s:e] >= 0)
    np.savez(dst, nll=nll.numpy(), top1=top1.numpy(), p_rosa=p_rosa.numpy(), tgt=ids[1:],
             pred=pred[0, :T].numpy(), mlen=mlen[0, :T].numpy(), cnt=cnt[0, :T].numpy(), src=src[0, :T].numpy())
    hit = (pred[0, :T, 0].numpy() == ids[1:]); cov = (mlen[0, :T, 0].numpy() > 0)
    print(f"{bid:>6} {idx[bid]['title'][:40]:40s} T={T:6d} rwkv nll {nll.mean():.3f} top1 {top1.float().mean():.3f} | "
          f"rosa cover {cov.mean():.3f} acc {hit.mean():.3f} acc|cover {hit[cov].mean():.3f} | rosa {t_rosa*1e3:.0f}ms rwkv {t_rwkv*1e3:.0f}ms", flush=True)
