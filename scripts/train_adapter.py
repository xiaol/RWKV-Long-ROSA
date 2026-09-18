"""Train memory adapters on a frozen RWKV-7 backbone with long-document LM loss.

--mode pointer          : ROSA pointer-generator head only (output side; cheap, no backprop through backbone)
--mode input            : ROSA Emb(ROSA(x)) input adapter at layer --in_layer (backprop through frozen backbone)
--mode both             : input adapter + pointer head
--mode engram           : Engram-lite hashed n-gram adapter at layer --in_layer (parametric baseline)
--mode engram+pointer   : Engram-lite + ROSA pointer head
Saves checkpoints + train log to results/runs/<name>/.
"""
import os, sys, json, math, time, argparse, random
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS
from rosa.adapters import RosaFeatures, RosaInputAdapter, RosaPointerHead, EngramLiteAdapter
from rosa.gpu import pick_gpu

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--mode", default="both")
ap.add_argument("--model", default="0.4b"); ap.add_argument("--device", default="auto")
ap.add_argument("--ctx", type=int, default=8192); ap.add_argument("--steps", type=int, default=1200); ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--lr_in", type=float, default=1e-4, help="lr for input-side adapters (ROSA input / Engram)"); ap.add_argument("--lr_f", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--K", type=int, default=4); ap.add_argument("--in_layer", type=int, default=0); ap.add_argument("--no_h", action="store_true")
ap.add_argument("--engram_rows", type=int, default=65536); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="tok", help="comma-separated data dirs under data/ (sampled proportionally to tokens)"); ap.add_argument("--eval_every", type=int, default=200); ap.add_argument("--eval_books", type=int, default=3); ap.add_argument("--eval_t", type=int, default=16384)
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
if args.device == "auto": args.device = pick_gpu(24000)
dev = args.device
root = os.path.join(os.path.dirname(__file__), "..")
idx = {}
for d in args.data.split(","):
    for b, v in json.load(open(f"{root}/data/{d}/index.json")).items():
        idx[f"{d}/{b}"] = v
run = f"{root}/results/runs/{args.name}"; os.makedirs(run, exist_ok=True)
json.dump(vars(args), open(f"{run}/args.json", "w"), indent=1)
log = open(f"{run}/train.log", "a")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
P("device", dev, "args", vars(args))

model = load_rwkv7(MODELS[args.model], device=dev)
C = model.args.n_embd
feats = RosaFeatures(args.K)
use_pointer = "pointer" in args.mode or args.mode == "both"
use_input = args.mode in ("input", "both")
use_engram = "engram" in args.mode
params = []; params_in = []
if use_input:
    ad = RosaInputAdapter(model.emb, K=args.K).to(dev); model.add_adapter(args.in_layer, ad); params_in += list(ad.parameters())
if use_engram:
    eg = EngramLiteAdapter(C, rows=args.engram_rows, seed=args.seed).to(dev); model.add_adapter(args.in_layer, eg); params_in += list(eg.parameters())
head = None
if use_pointer:
    head = RosaPointerHead(C, K=args.K, use_h=not args.no_h).to(dev); params += list(head.parameters())
model.grad_ckpt = use_input or use_engram
n_params = sum(p.numel() for p in params + params_in); P(f"trainable params: {n_params/1e6:.3f}M (input-side {sum(p.numel() for p in params_in)/1e6:.3f}M @ lr {args.lr_in})")
groups = ([{"params": params, "lr": args.lr_f}] if params else []) + ([{"params": params_in, "lr": args.lr_in}] if params_in else [])
opt = torch.optim.AdamW(groups, lr=args.lr_f, weight_decay=args.wd, betas=(0.9, 0.99))
params = params + params_in
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps)) * 0.9 + 0.1))

train_books = [b for b, v in idx.items() if v["split"] == "train"]
val_books = [b for b, v in idx.items() if v["split"] == "val" and b.startswith("tok/")][: args.eval_books] + [b for b, v in idx.items() if v["split"] == "val" and not b.startswith("tok/")][: 2]
train_tok = {b: np.load(f"{root}/data/{b}.npy") for b in train_books}
weights = np.array([max(0, len(train_tok[b]) - args.ctx - 1) for b in train_books], dtype=float); weights /= weights.sum()
P(f"train books {len(train_books)} tokens {sum(len(v) for v in train_tok.values())/1e6:.2f}M; val books {val_books}")

def sample_batch():
    b = np.random.choice(train_books, p=weights); a = train_tok[b]
    s = random.randrange(0, len(a) - args.ctx - 1)
    return torch.from_numpy(a[s: s + args.ctx + 1].astype(np.int64))[None]

def compute_loss(x, need_grad=True):
    """x: [1, T+1] int64 on CPU. returns (loss_mixture, loss_backbone) as means over T."""
    inp, tgt = x[:, :-1], x[:, 1:]
    aux = feats(inp); aux = {k: v.to(dev) for k, v in aux.items()}; aux["idx"] = inp.to(dev)
    inp_g, tgt_g = inp.to(dev), tgt.to(dev)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        logits, h = model(inp_g, aux=aux, return_hidden=True)
        T = logits.shape[1]
        lp_tgt = []; lp_cand = []
        for s in range(0, T, 4096):
            lp = F.log_softmax(logits[:, s: s + 4096].float(), -1)
            lp_tgt.append(lp.gather(-1, tgt_g[:, s: s + 4096, None]).squeeze(-1))
            lp_cand.append(lp.gather(-1, aux["pred"][:, s: s + 4096].clamp_min(0)))
            del lp
        lp_tgt = torch.cat(lp_tgt, 1); lp_cand = torch.cat(lp_cand, 1)
        base = -lp_tgt
        if head is not None:
            log_g, log_1mg, log_a = head(h, lp_cand.detach(), aux)
            lp_mix = RosaPointerHead.mixture_logp_target(lp_tgt, log_g, log_1mg, log_a, aux["pred"], tgt_g)
            loss = -lp_mix
        else:
            loss = base
    return loss, base.detach()

@torch.no_grad()
def evaluate():
    model.eval(); tot = 0; n = 0; pos = {}
    for b in val_books:
        a = np.load(f"{root}/data/{b}.npy")[: args.eval_t + 1].astype(np.int64)
        loss, base = compute_loss(torch.from_numpy(a)[None], need_grad=False)
        loss = loss[0].cpu().numpy(); base = base[0].cpu().numpy()
        tot += loss.sum(); n += len(loss)
        for lo, hi in [(0, 2048), (2048, 8192), (8192, 16384)]:
            if len(loss) > lo:
                d = pos.setdefault((lo, hi), [0, 0, 0]); d[0] += loss[lo:hi].sum(); d[1] += base[lo:hi].sum(); d[2] += len(loss[lo:hi])
    s = f"val nll {tot/n:.4f} | " + " ".join(f"[{lo}-{hi}] {d[0]/d[2]:.4f} (base {d[1]/d[2]:.4f})" for (lo, hi), d in sorted(pos.items()))
    return tot / n, s

t0 = time.time(); ema = None
for step in range(1, args.steps + 1):
    opt.zero_grad(set_to_none=True)
    acc_loss = 0; acc_base = 0
    for _ in range(args.accum):
        x = sample_batch()
        loss, base = compute_loss(x)
        (loss.mean() / args.accum).backward()
        acc_loss += loss.mean().item() / args.accum; acc_base += base.mean().item() / args.accum
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    ema = acc_loss if ema is None else 0.98 * ema + 0.02 * acc_loss
    if step % 10 == 0 or step == 1:
        P(f"step {step:5d} loss {acc_loss:.4f} ema {ema:.4f} base {acc_base:.4f} gain {acc_base-acc_loss:+.4f} lr {sched.get_last_lr()[0]:.2e} {(time.time()-t0)/step:.2f}s/step")
    if step % args.eval_every == 0 or step == args.steps:
        v, s = evaluate(); P(f"EVAL step {step}: {s}")
        torch.save({"adapters": model.adapters.state_dict(), "head": head.state_dict() if head is not None else None, "args": vars(args), "step": step, "val": v}, f"{run}/ckpt.pt")
P("done")
