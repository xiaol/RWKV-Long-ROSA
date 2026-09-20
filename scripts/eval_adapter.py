"""Per-position long-context evaluation of trained adapters on held-out books.

For each val book (first --max_t tokens) computes per-token NLL for the frozen backbone and for the
adapter-augmented model; reports NLL by position bucket and by ROSA match length.  Multiple runs can be
passed; results saved to results/eval/<run>.json and a combined table printed.
"""
import os, sys, json, math, argparse, glob
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS
from rosa.adapters import RosaFeatures, RosaInputAdapter, RosaPointerHead, EngramLiteAdapter
from rosa.gpu import pick_gpu

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+"); ap.add_argument("--device", default="auto"); ap.add_argument("--max_t", type=int, default=32768)
ap.add_argument("--books", type=int, default=24); ap.add_argument("--model", default="0.4b"); ap.add_argument("--data", default="tok"); ap.add_argument("--tag", default="")
args = ap.parse_args()
if args.device == "auto": args.device = pick_gpu(12000)
dev = args.device
root = os.path.join(os.path.dirname(__file__), ".."); idx = json.load(open(f"{root}/data/{args.data}/index.json"))
val_books = [b for b, v in sorted(idx.items(), key=lambda kv: str(kv[0]).zfill(8)) if v["split"] == "val"][: args.books]
POS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768)]
MB = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 16), (17, 10**9)]
os.makedirs(f"{root}/results/eval", exist_ok=True)

def build(run):
    ck = torch.load(f"{root}/results/runs/{run}/ckpt.pt", map_location="cpu", weights_only=False)
    a = ck["args"]; model = load_rwkv7(MODELS[a["model"]], device=dev); C = model.args.n_embd
    if a["mode"] in ("input", "both"):
        model.add_adapter(a["in_layer"], RosaInputAdapter(model.emb, K=a["K"]).to(dev))
    if "engram" in a["mode"]:
        model.add_adapter(a["in_layer"], EngramLiteAdapter(C, rows=a["engram_rows"], seed=a["seed"]).to(dev))
    model.adapters.load_state_dict(ck["adapters"])
    head = None
    if ck["head"] is not None:
        head = RosaPointerHead(C, K=a["K"], use_h=not a["no_h"], use_lpc=not a.get("no_lpc", False)).to(dev); head.load_state_dict(ck["head"]); head.eval()
    return model, head, a

@torch.no_grad()
def per_token(model, head, K, ids):
    x = torch.from_numpy(ids)[None]; inp, tgt = x[:, :-1], x[:, 1:]
    aux = RosaFeatures(K)(inp); aux = {k: v.to(dev) for k, v in aux.items()}; aux["idx"] = inp.to(dev)
    tgt_g = tgt.to(dev)
    logits, h = model(inp.to(dev), aux=aux, return_hidden=True)
    T = logits.shape[1]; lp_tgt = []; lp_cand = []
    for s in range(0, T, 4096):
        lp = F.log_softmax(logits[:, s: s + 4096].float(), -1)
        lp_tgt.append(lp.gather(-1, tgt_g[:, s: s + 4096, None]).squeeze(-1)); lp_cand.append(lp.gather(-1, aux["pred"][:, s: s + 4096].clamp_min(0)))
    lp_tgt = torch.cat(lp_tgt, 1); lp_cand = torch.cat(lp_cand, 1)
    if head is not None:
        log_g, log_1mg, log_a = head(h, lp_cand, aux)
        lp_tgt = RosaPointerHead.mixture_logp_target(lp_tgt, log_g, log_1mg, log_a, aux["pred"], tgt_g)
        g = log_g.exp()[0].cpu().numpy()
    else:
        g = np.zeros(T)
    return -lp_tgt[0].cpu().numpy(), aux["mlen"][0, :, 0].cpu().numpy(), g

results = {}
# baseline first
base_model = load_rwkv7(MODELS[args.model], device=dev)
data = {}
for b in val_books:
    ids = np.load(f"{root}/data/{args.data}/{b}.npy")[: args.max_t + 1].astype(np.int64)
    nll, mlen, _ = per_token(base_model, None, 1, ids); data[b] = dict(base=nll, mlen=mlen)
    print(f"baseline {b} T={len(nll)} nll {nll.mean():.4f}", flush=True)
del base_model; torch.cuda.empty_cache()
for run in args.runs:
    model, head, a = build(run)
    for b in val_books:
        ids = np.load(f"{root}/data/{args.data}/{b}.npy")[: args.max_t + 1].astype(np.int64)
        nll, _, g = per_token(model, head, a["K"], ids); data[b][run] = nll; data[b][run + "_g"] = g
        print(f"{run} {b} nll {nll.mean():.4f} (base {data[b]['base'].mean():.4f}) mean gate {g.mean():.3f}", flush=True)
    del model, head; torch.cuda.empty_cache()

def table(key_fn, buckets, label):
    print(f"\n### NLL by {label} (val books={len(val_books)}, max_t={args.max_t})")
    cols = ["base"] + args.runs
    print(f"{'bucket':>14} {'ntok':>9} " + " ".join(f"{c:>14}" for c in cols))
    summary = {}
    for bk in buckets:
        sel_all = {c: [] for c in cols}; n = 0
        for b in val_books:
            sel = key_fn(data[b], bk)
            if sel is None or sel.sum() == 0: continue
            for c in cols: sel_all[c].append(data[b][c][sel])
        if not sel_all["base"]: continue
        vals = {c: np.concatenate(sel_all[c]) for c in cols}
        n = len(vals["base"])
        print(f"{str(bk):>14} {n:>9} " + " ".join(f"{vals[c].mean():8.4f}({vals[c].mean()-vals['base'].mean():+.3f})" for c in cols))
        summary[str(bk)] = {c: float(vals[c].mean()) for c in cols}
    return summary

def pos_sel(d, bk):
    T = len(d["base"]); lo, hi = bk
    if T <= lo: return None
    s = np.zeros(T, bool); s[lo:min(hi, T)] = True; return s
def mlen_sel(d, bk):
    lo, hi = bk; return (d["mlen"] >= lo) & (d["mlen"] <= hi)
out = dict(by_position=table(pos_sel, POS, "position"), by_mlen=table(mlen_sel, MB, "ROSA match length"))
allv = {c: float(np.concatenate([data[b][c] for b in val_books]).mean()) for c in ["base"] + args.runs}
print("\noverall:", {k: f"{v:.4f} (ppl {math.exp(v):.2f})" for k, v in allv.items()})
out["overall"] = allv; out["books"] = val_books
json.dump(out, open(f"{root}/results/eval/{args.data}{args.tag}__{'__'.join(args.runs)}.json", "w"), indent=1)
