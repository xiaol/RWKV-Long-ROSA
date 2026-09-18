"""Passage-copy evaluation: how well does the model exploit verbatim repetition at long distance?

For each held-out book: take a P-token passage at position p, place it again after `dist` tokens of
intervening text (so the copy starts at p+P+dist), and measure NLL on the copy under (a) the frozen
backbone and (b) each adapter run.  ROSA gets the copy right by design; the question is whether the
learned gate exploits it.  Reports NLL on the copied passage by distance, and on the surrounding text.
"""
import os, sys, json, math, argparse, random
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS
from rosa.adapters import RosaFeatures, RosaInputAdapter, RosaPointerHead, EngramLiteAdapter
from rosa.gpu import pick_gpu

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="*"); ap.add_argument("--device", default="auto"); ap.add_argument("--model", default="0.4b")
ap.add_argument("--P", type=int, default=256); ap.add_argument("--dists", default="512,2048,8192,16384,30000"); ap.add_argument("--n", type=int, default=12)
args = ap.parse_args()
if args.device == "auto": args.device = pick_gpu(10000)
dev = args.device
root = os.path.join(os.path.dirname(__file__), ".."); idx = json.load(open(f"{root}/data/tok/index.json"))
val_books = [b for b, v in sorted(idx.items(), key=lambda kv: int(kv[0])) if v["split"] == "val" and v["tokens"] > 40000]
rng = random.Random(0)

def build(run):
    ck = torch.load(f"{root}/results/runs/{run}/ckpt.pt", map_location="cpu", weights_only=False)
    a = ck["args"]; model = load_rwkv7(MODELS[a["model"]], device=dev); C = model.args.n_embd
    if a["mode"] in ("input", "both"): model.add_adapter(a["in_layer"], RosaInputAdapter(model.emb, K=a["K"]).to(dev))
    if "engram" in a["mode"]: model.add_adapter(a["in_layer"], EngramLiteAdapter(C, rows=a["engram_rows"], seed=a["seed"]).to(dev))
    model.adapters.load_state_dict(ck["adapters"]); head = None
    if ck["head"] is not None:
        head = RosaPointerHead(C, K=a["K"], use_h=not a["no_h"]).to(dev); head.load_state_dict(ck["head"]); head.eval()
    return model, head, a["K"]

@torch.no_grad()
def nll_seq(model, head, K, ids):
    x = torch.from_numpy(ids)[None]; inp, tgt = x[:, :-1], x[:, 1:]
    aux = RosaFeatures(K)(inp); aux = {k: v.to(dev) for k, v in aux.items()}; aux["idx"] = inp.to(dev); tgt_g = tgt.to(dev)
    logits, h = model(inp.to(dev), aux=aux, return_hidden=True)
    T = logits.shape[1]; lp_tgt = []; lp_cand = []
    for s_ in range(0, T, 4096):
        lp = F.log_softmax(logits[:, s_: s_ + 4096].float(), -1)
        lp_tgt.append(lp.gather(-1, tgt_g[:, s_: s_ + 4096, None]).squeeze(-1)); lp_cand.append(lp.gather(-1, aux["pred"][:, s_: s_ + 4096].clamp_min(0)))
    lp_tgt = torch.cat(lp_tgt, 1); lp_cand = torch.cat(lp_cand, 1)
    if head is not None:
        log_g, log_1mg, log_a = head(h, lp_cand, aux); lp_tgt = RosaPointerHead.mixture_logp_target(lp_tgt, log_g, log_1mg, log_a, aux["pred"], tgt_g)
    return -lp_tgt[0].cpu().numpy()

cases = []
for dist in [int(d) for d in args.dists.split(",")]:
    for i in range(args.n):
        b = rng.choice(val_books); a = np.load(f"{root}/data/tok/{b}.npy").astype(np.int64)
        need = 1024 + args.P + dist + args.P + 64
        if len(a) < need + 10: continue
        s = rng.randrange(0, len(a) - need)
        seq = a[s: s + need].copy()
        p = 1024; c = p + args.P + dist
        seq[c: c + args.P] = seq[p: p + args.P]  # paste the passage again
        cases.append(dict(dist=dist, seq=seq, p=p, c=c))
print(f"{len(cases)} cases", flush=True)

res = {}
def run_model(name, model, head, K):
    out = {}
    for cs in cases:
        nll = nll_seq(model, head, K, cs["seq"]); c = cs["c"] - 1; P = args.P  # nll index t predicts token t+1
        copy = nll[c: c + P].mean(); around = np.concatenate([nll[c - 512: c], nll[c + P: c + P + 64]]).mean()
        out.setdefault(cs["dist"], []).append((copy, around))
    res[name] = {d: (float(np.mean([v[0] for v in vs])), float(np.mean([v[1] for v in vs]))) for d, vs in out.items()}
    print(name, {d: f"copy {v[0]:.3f} / around {v[1]:.3f}" for d, v in res[name].items()}, flush=True)

m = load_rwkv7(MODELS[args.model], device=dev); run_model("base", m, None, 1); del m; torch.cuda.empty_cache()
for run in args.runs:
    m, h, K = build(run); run_model(run, m, h, K); del m, h; torch.cuda.empty_cache()
os.makedirs(f"{root}/results/copy", exist_ok=True)
json.dump(res, open(f"{root}/results/copy/{args.model}__{'__'.join(args.runs) or 'base'}.json", "w"), indent=1)
print(f"\n### NLL on a {args.P}-token passage repeated after `dist` tokens (val books; 'around' = neighbouring un-repeated text)")
print(f"{'dist':>7} " + " ".join(f"{n:>22}" for n in res))
for d in sorted(res["base"]):
    print(f"{d:>7} " + " ".join(f"{res[n][d][0]:9.3f} / {res[n][d][1]:8.3f}" for n in res))
