"""Tokenize Project Gutenberg books with the RWKV World tokenizer into per-book int32 arrays.

Strips the Gutenberg header/footer, keeps English books >= min_tokens, writes data/tok/<id>.npy and
data/tok/index.json with a deterministic train/val split by book id (val = every 5th book, sorted).
"""
import os, re, json, sys, glob
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import get_tokenizer

MIN_TOKENS = 20000
root = os.path.join(os.path.dirname(__file__), "..", "data")
out = os.path.join(root, "tok"); os.makedirs(out, exist_ok=True)
tok = get_tokenizer()
index = {}
for path in sorted(glob.glob(os.path.join(root, "gutenberg", "pg*.txt"))):
    bid = int(re.search(r"pg(\d+)\.txt", path).group(1))
    dst = os.path.join(out, f"{bid}.npy")
    try:
        raw = open(path, encoding="utf-8", errors="replace").read()
    except FileNotFoundError:
        continue  # download loop may remove incomplete files concurrently
    if not re.search(r"^Language:\s*English", raw[:5000], re.M):
        continue
    m1 = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", raw)
    m2 = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", raw)
    if not (m1 and m2):
        continue
    body = raw[m1.end():m2.start()].strip("\r\n")
    body = body.replace("\r\n", "\n")
    if os.path.exists(dst):
        ids = np.load(dst)
    else:
        ids = np.array(tok.encode(body), dtype=np.int32)
        np.save(dst, ids)
    if len(ids) < MIN_TOKENS:
        continue
    title = re.search(r"^Title:\s*(.*)$", raw[:5000], re.M)
    index[bid] = dict(tokens=int(len(ids)), chars=len(body), title=title.group(1).strip() if title else "")
ids_sorted = sorted(index)
for i, bid in enumerate(ids_sorted):
    index[bid]["split"] = "val" if bid % 5 == 0 else "train"  # stable under corpus growth
json.dump(index, open(os.path.join(out, "index.json"), "w"), indent=1)
tr = sum(v["tokens"] for v in index.values() if v["split"] == "train"); va = sum(v["tokens"] for v in index.values() if v["split"] == "val")
print(f"books {len(index)}  train tokens {tr/1e6:.2f}M  val tokens {va/1e6:.2f}M")
print("longest:", sorted(((v["tokens"], v["title"]) for v in index.values()), reverse=True)[:5])
