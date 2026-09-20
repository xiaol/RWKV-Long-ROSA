"""Experiment 1b: synthetic long-range retrieval on the frozen RNN vs ROSA.

Needle-in-a-haystack: a random 'The secret code for <name> is <5 tokens>' sentence is inserted at
depth d (fraction of the haystack); the question 'The secret code for <name> is' is appended at the
end of a haystack of length L taken from a held-out book.  We score exact match of the 5 answer tokens
under greedy decoding for (a) RWKV-7 alone, (b) ROSA alone (parameter-free), (c) a simple gate:
use the ROSA candidate when its matched suffix length >= tau, else RWKV.
Also: passage repeat — a 64-token passage from position p is repeated at the end; loss on the copy.
"""
import os, sys, json, math, random, argparse
import numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, MODELS, get_tokenizer
from rosa.ops import rosa_tokens

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="0.4b"); ap.add_argument("--device", default="auto")
ap.add_argument("--lengths", default="1024,2048,4096,8192,16384,32768"); ap.add_argument("--n", type=int, default=20)
ap.add_argument("--tau", type=int, default=3); ap.add_argument("--distractors", type=int, default=0); ap.add_argument("--tag", default="")
ap.add_argument("--runs", default="", help="comma-separated trained adapter runs to decode with (mixture greedy)")
ap.add_argument("--skip_rule", action="store_true", help="skip rwkv/rosa/gated rule baselines (only run adapters)")
args = ap.parse_args()
root = os.path.join(os.path.dirname(__file__), ".."); idx = json.load(open(f"{root}/data/tok/index.json"))
from rosa.gpu import pick_gpu
if args.device == 'auto': args.device = pick_gpu(12000)
print('device', args.device)
tok = get_tokenizer(); model = load_rwkv7(MODELS[args.model], device=args.device)
val = [b for b, v in idx.items() if v["split"] == "val"]
hay = np.concatenate([np.load(f"{root}/data/tok/{b}.npy") for b in val]).astype(np.int64)
rng = random.Random(0)
NAMES = ["Alice", "Bob", "Carol", "Dave", "Erin", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Walter"]

def greedy(prefix_ids, n_new):
    ids = list(prefix_ids)
    outs = []
    with torch.no_grad():
        for _ in range(n_new):
            lg = model(torch.tensor([ids], device=args.device), last_only=True)[0, -1]
            t = int(lg.argmax()); outs.append(t); ids.append(t)
    return outs

def rosa_greedy(prefix_ids, n_new):
    ids = list(prefix_ids); outs = []
    for _ in range(n_new):
        pred, mlen, _, _ = rosa_tokens(torch.tensor([ids]), K=1)
        t = int(pred[0, -1, 0]); outs.append(t); ids.append(t if t >= 0 else 0)
    return outs

def gated_greedy(prefix_ids, n_new, tau):
    ids = list(prefix_ids); outs = []
    with torch.no_grad():
        for _ in range(n_new):
            pred, mlen, _, _ = rosa_tokens(torch.tensor([ids]), K=1)
            if int(mlen[0, -1, 0]) >= tau:
                t = int(pred[0, -1, 0])
            else:
                t = int(model(torch.tensor([ids], device=args.device), last_only=True)[0, -1].argmax())
            outs.append(t); ids.append(t)
    return outs

from rosa.adapters import RosaFeatures, RosaInputAdapter, RosaPointerHead, EngramLiteAdapter
adapters = {}
for run in [r for r in args.runs.split(",") if r]:
    ck = torch.load(f"{root}/results/runs/{run}/ckpt.pt", map_location="cpu", weights_only=False); a = ck["args"]
    m = load_rwkv7(MODELS[a["model"]], device=args.device); C = m.args.n_embd
    if a["mode"] in ("input", "both"): m.add_adapter(a["in_layer"], RosaInputAdapter(m.emb, K=a["K"]).to(args.device))
    if "engram" in a["mode"]: m.add_adapter(a["in_layer"], EngramLiteAdapter(C, rows=a["engram_rows"], seed=a["seed"]).to(args.device))
    m.adapters.load_state_dict(ck["adapters"]); head = None
    if ck["head"] is not None:
        head = RosaPointerHead(C, K=a["K"], use_h=not a["no_h"], use_lpc=not a.get("no_lpc", False)).to(args.device); head.load_state_dict(ck["head"]); head.eval()
    adapters[run] = (m, head, a["K"])

def adapter_greedy(run, prefix_ids, n_new):
    m, head, K = adapters[run]; ids = list(prefix_ids); outs = []
    with torch.no_grad():
        for _ in range(n_new):
            inp = torch.tensor([ids]); aux = RosaFeatures(K)(inp); aux = {k: v.to(args.device) for k, v in aux.items()}; aux["idx"] = inp.to(args.device)
            logits, h = m(inp.to(args.device), aux=aux, return_hidden=True)
            lg = logits[:, -1:]
            if head is not None:
                aux_last = {k: v[:, -1:] for k, v in aux.items()}
                lp_cand = F.log_softmax(lg.float(), -1).gather(-1, aux_last["pred"].clamp_min(0))
                log_g, log_1mg, log_a = head(h[:, -1:], lp_cand, aux_last)
                lg = RosaPointerHead.mixture_logits(lg, log_g, log_1mg, log_a, aux_last["pred"])
            t = int(lg[0, -1].argmax()); outs.append(t); ids.append(t)
    return outs

results = []
for L in [int(x) for x in args.lengths.split(",")]:
    for depth in [0.1, 0.5, 0.9]:
        acc = {"rwkv": 0, "rosa": 0, "gated": 0, **{r: 0 for r in adapters}}
        for i in range(args.n):
            name = rng.choice(NAMES); code = "".join(rng.choice("0123456789") for _ in range(6))
            needle = tok.encode(f"\nThe secret code for {name} is {code}.\n")
            q = tok.encode(f"\nThe secret code for {name} is")
            ans = tok.encode(f" {code}")
            ans = tok.encode(f"\nThe secret code for {name} is {code}")[len(q):]  # answer tokens as they appear after the question
            start = rng.randrange(0, len(hay) - L - 10)
            body = hay[start:start + L].tolist()
            cut = int(depth * L)
            # multi-key variant: other people's codes are scattered through the haystack (must retrieve the right one)
            inserts = [(cut, needle)]
            for other in rng.sample([n_ for n_ in NAMES if n_ != name], args.distractors):
                ocode = "".join(rng.choice("0123456789") for _ in range(6))
                inserts.append((rng.randrange(0, L), tok.encode(f"\nThe secret code for {other} is {ocode}.\n")))
            ctx = []; last = 0
            for pos_, ins in sorted(inserts, key=lambda t: t[0]):
                ctx += body[last:pos_] + ins; last = pos_
            ctx += body[last:] + q
            n_new = len(ans)
            if not args.skip_rule:
                r1 = greedy(ctx, n_new); r2 = rosa_greedy(ctx, n_new); r3 = gated_greedy(ctx, n_new, args.tau)
                acc["rwkv"] += r1 == ans; acc["rosa"] += r2 == ans; acc["gated"] += r3 == ans
            for r in adapters:
                acc[r] += adapter_greedy(r, ctx, n_new) == ans
        row = dict(L=L, depth=depth, n=args.n, **{k: v / args.n for k, v in acc.items()})
        results.append(row); print(row, flush=True)
os.makedirs(f"{root}/results/niah", exist_ok=True)
json.dump(results, open(f"{root}/results/niah/{args.model}_niah{args.tag}.json", "w"), indent=1)
