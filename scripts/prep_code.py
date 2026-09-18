"""Build long code documents (one per package subtree) from locally installed Python sources.
Writes data/tok_code/<name>.npy + index.json (alternating docs are 'train'/'val')."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rosa.rwkv7 import get_tokenizer
import torch, numpy, transformers
tok = get_tokenizer()
SP = os.path.dirname(os.path.dirname(torch.__file__))
GROUPS = {
    "torch_inductor": f"{SP}/torch/_inductor", "torch_dynamo": f"{SP}/torch/_dynamo", "torch_distributed": f"{SP}/torch/distributed",
    "torch_nn": f"{SP}/torch/nn", "torch_onnx": f"{SP}/torch/onnx", "torch_fx": f"{SP}/torch/fx", "torch_export": f"{SP}/torch/export",
    "numpy_core": f"{SP}/numpy/_core", "numpy_lib": f"{SP}/numpy/lib", "transformers_models_a": f"{SP}/transformers/models/[a-c]*",
    "transformers_models_l": f"{SP}/transformers/models/[l-m]*", "transformers_generation": f"{SP}/transformers/generation",
    "sympy_core": f"{SP}/sympy/core", "sympy_polys": f"{SP}/sympy/polys", "pip_vendor": f"{SP}/pip/_vendor/rich", "setuptools": f"{SP}/setuptools",
    "networkx_algorithms": f"{SP}/networkx/algorithms", "scipy_stats": f"{SP}/scipy/stats", "pandas_core": f"{SP}/pandas/core",
    "matplotlib": f"{SP}/matplotlib", "sklearn": f"{SP}/sklearn", "jinja2": f"{SP}/jinja2", "requests": f"{SP}/requests", "yaml": f"{SP}/yaml",
}
root = os.path.join(os.path.dirname(__file__), "..", "data", "tok_code"); os.makedirs(root, exist_ok=True)
index = {}
for name, pat in GROUPS.items():
    files = sorted(f for d in glob.glob(pat) for f in glob.glob(os.path.join(d, "**", "*.py"), recursive=True))
    if not files: continue
    parts = []
    for f in files:
        try: parts.append(f"# file: {os.path.relpath(f, SP)}\n" + open(f, encoding="utf-8", errors="replace").read())
        except Exception: pass
    text = "\n\n".join(parts)
    ids = np.array(tok.encode(text[:1_200_000]), dtype=np.int32)  # cap ~1.2M chars (~400k tokens)
    if len(ids) < 40000: continue
    np.save(f"{root}/{name}.npy", ids); index[name] = dict(tokens=int(len(ids)), files=len(files), split="val" if len(index) % 2 == 0 else "train", title=name)  # alternate docs train/val
    print(name, len(files), "files", len(ids), "tokens", flush=True)
json.dump(index, open(f"{root}/index.json", "w"), indent=1)
print("total tokens", sum(v["tokens"] for v in index.values()) / 1e6, "M in", len(index), "docs")
