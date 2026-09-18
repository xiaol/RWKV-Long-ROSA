"""Pick a GPU with enough free memory on a shared box."""
import subprocess, torch

def free_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, u, t = [int(v) for v in line.split(",")]
        res[i] = t - u
    return res

def pick_gpu(min_free_mib=20000, exclude=()):
    f = free_mib()
    cands = sorted(((v, k) for k, v in f.items() if k not in exclude and v >= min_free_mib), reverse=True)
    if not cands:
        raise RuntimeError(f"no GPU with >= {min_free_mib} MiB free: {f}")
    return f"cuda:{cands[0][1]}"
