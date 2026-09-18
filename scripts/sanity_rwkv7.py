import sys, os, time, torch, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import load_rwkv7, get_tokenizer, MODELS
dev = "cuda:1"
tok = get_tokenizer()
m = load_rwkv7(MODELS["0.4b"], device=dev)
txt = open("data/gutenberg/pg1342.txt").read()
ids = tok.encode(txt[20000:60000])[:4096]
x = torch.tensor([ids], device=dev)
with torch.no_grad():
    t = time.time(); logits = m(x); torch.cuda.synchronize(); dt = time.time() - t
    loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), x[0, 1:])
print(f"T={len(ids)} fwd {dt*1000:.0f} ms  loss {loss.item():.3f}  ppl {math.exp(loss.item()):.2f}")
prompt = "The capital of France is"
ids = tok.encode(prompt)
with torch.no_grad():
    for _ in range(12):
        lg = m(torch.tensor([ids], device=dev))[0, -1]
        ids.append(int(lg.argmax()))
print(repr(tok.decode(ids)))
