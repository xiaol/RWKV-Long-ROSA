"""Fast ROSA ops (C++/OpenMP, JIT-compiled with torch.utils.cpp_extension)."""
from __future__ import annotations
import os
import torch

_ext = None


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(os.path.abspath(__file__))
        _ext = load(
            name="rosa_cpu_ext",
            sources=[os.path.join(here, "csrc", "rosa_cpu.cpp")],
            extra_cflags=["-O3", "-fopenmp", "-std=c++17"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    return _ext


def rosa_tokens(x: torch.Tensor, K: int = 1):
    """ROSA over token ids, with a K-deep suffix-link chain.

    Args:  x: int64 [B,T] (any device; computed on CPU).  K: number of candidates per position.
    Returns (pred, mlen, src, cnt), each int64 [B,T,K] on x.device.  Index k=0 is BlinkDL's ROSA:
        pred[i,0] = token that followed the longest earlier occurrence of the suffix ending at i (-1 if none)
        mlen[i,0] = length of that matched suffix (0 if none)
        src[i,0]  = position of pred[i,0] in x (-1 if none)
        cnt[i,0]  = number of earlier occurrences of that suffix
    k>0 are the next-shorter suffixes with earlier occurrences (distinct automaton states), padded with -1/0.
    """
    ext = _load()
    dev = x.device
    if x.dim() == 1:
        x = x[None]
    pred, mlen, src, cnt = ext.rosa_tokens(x.detach().to("cpu", torch.int64), int(K))
    return pred.to(dev), mlen.to(dev), src.to(dev), cnt.to(dev)


def rosa_qkv_symbols(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, alphabet: int):
    """ROSA-QKV over small-alphabet symbol rows (BlinkDL's samx_qkv semantics).

    q,k,v: uint8 [R,T] with values < alphabet.  Returns (out int16 [R,T], mlen int32, src int32);
    out=-1 where no match.  For position i: largest w with q[i+1-w:i+1]==k[j:j+w], j+w<=i, largest j; out=v[j+w].
    """
    ext = _load()
    dev = q.device
    out, mlen, src = ext.rosa_qkv(q.detach().to("cpu", torch.uint8), k.detach().to("cpu", torch.uint8), v.detach().to("cpu", torch.uint8), int(alphabet))
    return out.to(dev), mlen.to(dev), src.to(dev)


def bits_to_symbols(bits: torch.Tensor, nbits: int) -> torch.Tensor:
    """bits: bool/uint8 [B,T,C] -> uint8 symbols [B*G, T] with G=C//nbits (channel-major rows)."""
    B, T, C = bits.shape
    assert C % nbits == 0
    G = C // nbits
    b = bits.to(torch.int32).view(B, T, G, nbits)
    weights = (1 << torch.arange(nbits, device=bits.device, dtype=torch.int32))
    sym = (b * weights).sum(-1)  # [B,T,G]
    return sym.permute(0, 2, 1).reshape(B * G, T).to(torch.uint8)


def symbols_to_bits(sym: torch.Tensor, nbits: int, B: int, T: int) -> torch.Tensor:
    """inverse of bits_to_symbols; sym int [B*G, T] (negative = unmatched -> all bits 0 flagged separately)."""
    G = sym.shape[0] // B
    s = sym.view(B, G, T).permute(0, 2, 1).to(torch.int32)  # [B,T,G]
    shifts = torch.arange(nbits, device=sym.device, dtype=torch.int32)
    bits = (s.unsqueeze(-1) >> shifts) & 1  # [B,T,G,nbits]
    return bits.reshape(B, T, G * nbits)


class RosaStream:
    """Incremental ROSA for decoding.  ``push(token)`` returns (pred, mlen, src, cnt) int64 [K] for the
    suffix ending at the pushed token; amortised O(1) per token, memory O(context)."""

    def __init__(self, capacity: int, K: int = 4):
        self._s = _load().RosaStream(int(capacity), int(K)); self.K = K

    def push(self, token: int):
        self._s.push(int(token))
        return self._s.candidates()

    def extend(self, tokens):
        out = None
        for t in tokens:
            out = self.push(t)
        return out

    def __len__(self):
        return self._s.size()

    @property
    def states(self):
        return self._s.states()
