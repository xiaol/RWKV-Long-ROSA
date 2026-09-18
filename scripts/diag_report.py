"""Aggregate results/diag/<model>/*.npz into tables: loss vs position, ROSA accuracy vs match length,
and a training-free ROSA/RWKV mixture (lambda per match-length bucket fit on train books, eval on val)."""
import os, sys, json, glob, math, argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--model", default="0.4b"); ap.add_argument("--fit_split", default="train"); ap.add_argument("--data", default="tok")
args = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), ".."); idx = json.load(open(f"{root}/data/{args.data}/index.json"))
files = sorted(glob.glob(f"{root}/results/diag/{args.model}" + ("" if args.data == "tok" else "_" + args.data) + "/*.npz"))
D = {os.path.basename(f)[:-4]: dict(np.load(f)) for f in files}
print(f"{len(D)} books")
POS = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, 65536)]
MB = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, 16), (17, 10**9)]

def mbucket(m):
    b = np.zeros_like(m)
    for i, (lo, hi) in enumerate(MB): b[(m >= lo) & (m <= hi)] = i
    return b

def cat(key, split):
    return np.concatenate([D[b][key] for b in D if idx[b]["split"] == split])

for split in ["val", "train"]:
    bs = [b for b in D if idx[b]["split"] == split]
    if not bs: continue
    print(f"\n=== split {split}: {len(bs)} books ===")
    print("RWKV-7 NLL / top1 and ROSA(k=0) coverage / accuracy by position:")
    print(f"{'pos':>14} {'ntok':>8} {'rwkv_nll':>9} {'rwkv_top1':>9} {'rosa_cov':>9} {'rosa_acc':>9} {'acc|cov':>8} {'rwkv_top1|cov':>13} {'mean_mlen':>9}")
    for lo, hi in POS:
        nll = []; top1 = []; cov = []; hit = []; ml = []
        for b in bs:
            d = D[b]; T = len(d["nll"]); 
            if T <= lo: continue
            sl = slice(lo, min(hi, T))
            nll.append(d["nll"][sl]); top1.append(d["top1"][sl]); m = d["mlen"][sl, 0]; cov.append(m > 0); hit.append(d["pred"][sl, 0] == d["tgt"][sl]); ml.append(m)
        if not nll: continue
        nll = np.concatenate(nll); top1 = np.concatenate(top1); cov = np.concatenate(cov); hit = np.concatenate(hit); ml = np.concatenate(ml)
        print(f"{lo:>6}-{hi:<7} {len(nll):>8} {nll.mean():9.3f} {top1.mean():9.3f} {cov.mean():9.3f} {hit.mean():9.3f} {hit[cov].mean():8.3f} {top1[cov].mean():13.3f} {ml.mean():9.1f}")
    print("\nROSA(k=0) accuracy vs matched suffix length, and RWKV's own prob on the ROSA candidate:")
    m = cat("mlen", split)[:, 0]; hit = cat("pred", split)[:, 0] == cat("tgt", split); pr = cat("p_rosa", split)[:, 0]; nll = cat("nll", split); top1 = cat("top1", split)
    mb = mbucket(m)
    print(f"{'mlen':>8} {'frac':>7} {'rosa_acc':>9} {'rwkv_top1':>9} {'rwkv_nll':>9} {'rwkv_p(rosa)':>12} {'oracle_nll':>10}")
    for i, (lo, hi) in enumerate(MB):
        sel = mb == i
        if sel.sum() == 0: continue
        oracle = np.where(hit[sel], 0.0, nll[sel]).mean()
        print(f"{lo:>3}-{min(hi,99):<4} {sel.mean():7.3f} {hit[sel].mean():9.3f} {top1[sel].mean():9.3f} {nll[sel].mean():9.3f} {pr[sel].mean():12.3f} {oracle:10.3f}")

# ---- training-free mixture: p = (1-l) p_rwkv + l * onehot(rosa), l per (mlen bucket x cnt bucket)
def feats(split):
    m = cat("mlen", split)[:, 0]; c = cat("cnt", split)[:, 0]
    cb = np.minimum(np.log2(np.maximum(c, 1)).astype(int), 4)
    return mbucket(m) * 5 + cb
def mix_nll(l, p_true, hit):
    return -np.log(np.clip((1 - l) * p_true + l * hit, 1e-9, None))
fit_split = args.fit_split if any(idx[b]["split"] == args.fit_split for b in D) else "val"
f_fit = feats(fit_split); p_fit = np.exp(-cat("nll", fit_split)); h_fit = (cat("pred", fit_split)[:, 0] == cat("tgt", fit_split)).astype(float)
grid = np.concatenate([[0.0], np.geomspace(1e-3, 0.99, 60)])
lam = np.zeros(len(MB) * 5)
for f in range(len(lam)):
    sel = f_fit == f
    if sel.sum() < 50: continue
    losses = [mix_nll(l, p_fit[sel], h_fit[sel]).mean() for l in grid]
    lam[f] = grid[int(np.argmin(losses))]
print(f"\n=== training-free mixture (lambda fit on {fit_split}) ===")
print("lambda table rows=mlen bucket, cols=log2(cnt) bucket 0..4:")
for i, (lo, hi) in enumerate(MB): print(f"  mlen {lo:>3}-{min(hi,99):<3} " + " ".join(f"{lam[i*5+j]:6.3f}" for j in range(5)))
for split in ["val", "train"]:
    bs = [b for b in D if idx[b]["split"] == split]
    if not bs: continue
    f = feats(split); p = np.exp(-cat("nll", split)); h = (cat("pred", split)[:, 0] == cat("tgt", split)).astype(float); base = cat("nll", split)
    mixed = mix_nll(lam[f], p, h)
    print(f"{split}: RWKV nll {base.mean():.4f} (ppl {math.exp(base.mean()):.2f}) -> mixture nll {mixed.mean():.4f} (ppl {math.exp(mixed.mean()):.2f})  delta {mixed.mean()-base.mean():+.4f}")
    # by position
    off = 0; pos_base = {k: [] for k in POS}; pos_mix = {k: [] for k in POS}
    for b in bs:
        T = len(D[b]["nll"])
        for lo, hi in POS:
            if T > lo:
                pos_base[(lo, hi)].append(base[off + lo: off + min(hi, T)]); pos_mix[(lo, hi)].append(mixed[off + lo: off + min(hi, T)])
        off += T
    for k in POS:
        if pos_base[k]:
            a = np.concatenate(pos_base[k]).mean(); m_ = np.concatenate(pos_mix[k]).mean()
            print(f"   pos {k[0]:>6}-{k[1]:<6}: nll {a:.4f} -> {m_:.4f}  ({(m_-a):+.4f}, ppl {math.exp(a):.2f} -> {math.exp(m_):.2f})")
