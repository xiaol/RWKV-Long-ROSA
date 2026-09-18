"""Memory adapters for a frozen RWKV-7 backbone.

* RosaFeatures      – computes ROSA (suffix-automaton) candidates from token ids on CPU.
* RosaInputAdapter  – BlinkDL's "Emb(ROSA(x))": adds a gated embedding of the retrieved candidates to
                      the residual stream at an early layer (input-side, lets all layers see the retrieval).
* RosaPointerHead   – pointer-generator mixture at the output: p = (1-g) p_rwkv + g * sum_k a_k onehot(pred_k),
                      g and a_k from tiny MLPs over (hidden state, match length, occurrence count, RWKV's own
                      log-prob of the candidate).  Loss needs only p_rwkv(target) and the candidate hits.
* EngramLiteAdapter – DeepSeek Engram-style *parametric* memory: hashed 2/3-gram embedding tables with the
                      context-aware gate of arXiv:2601.07372 (Eq. 3-5), for comparison.  Retrieves from
                      learned tables, not from the context.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops import rosa_tokens


class RosaFeatures:
    def __init__(self, K: int = 4):
        self.K = K

    def __call__(self, idx: torch.Tensor) -> dict:
        pred, mlen, src, cnt = rosa_tokens(idx, K=self.K)  # [B,T,K] on idx.device
        return dict(pred=pred, mlen=mlen, cnt=cnt, src=src)


def _cand_feats(aux, K):
    """[B,T,K,3] float features: log1p(mlen), log1p(cnt), valid."""
    mlen = aux["mlen"][..., :K].float(); cnt = aux["cnt"][..., :K].float(); valid = (aux["pred"][..., :K] >= 0).float()
    return torch.stack([torch.log1p(mlen), torch.log1p(cnt), valid], -1)


class RosaInputAdapter(nn.Module):
    """delta_i = W( sum_k g_k * E[pred_{i,k}] ),  g_k = sigmoid(MLP(feats_k, k)),  W zero-init."""

    def __init__(self, emb: nn.Embedding, K: int = 4, hidden: int = 64):
        super().__init__()
        object.__setattr__(self, "emb_weight", emb.weight)  # frozen backbone embedding; plain reference, not a parameter of this module
        self.K = K; C = emb.weight.shape[1]
        self.k_emb = nn.Parameter(torch.zeros(K, 8))
        self.gate = nn.Sequential(nn.Linear(3 + 8, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.proj = nn.Linear(C, C, bias=False); nn.init.zeros_(self.proj.weight)
        self.ln = nn.LayerNorm(C)

    def forward(self, x, aux):
        pred = aux["pred"][..., :self.K]
        f = _cand_feats(aux, self.K)  # [B,T,K,3]
        f = torch.cat([f, self.k_emb.expand(f.shape[0], f.shape[1], -1, -1)], -1)
        g = torch.sigmoid(self.gate(f)) * f[..., 2:3]  # zero for invalid
        with torch.no_grad():
            e = F.embedding(pred.clamp_min(0), self.emb_weight).detach()  # [B,T,K,C]
        mixed = (g.to(e.dtype) * e).sum(2)
        return self.proj(self.ln(mixed.float())).to(x.dtype)


class RosaPointerHead(nn.Module):
    def __init__(self, C: int, K: int = 4, hidden: int = 128, use_h: bool = True, hproj: int = 64):
        super().__init__()
        self.K = K; self.use_h = use_h
        self.hn = nn.LayerNorm(C) if use_h else None  # backbone hidden has entries up to ~400; normalise first
        self.hp = nn.Linear(C, hproj) if use_h else None
        din = 3 + 1 + 8 + (hproj if use_h else 0)  # feats, logp_rwkv(cand), k-emb, h
        self.k_emb = nn.Parameter(torch.randn(K, 8) * 0.1)
        self.score = nn.Sequential(nn.Linear(din, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.gate = nn.Sequential(nn.Linear(din * K, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.gate[-1].bias, -3.0)  # start with g ~ 0.05

    def features(self, h, logp_cand, aux):
        f = _cand_feats(aux, self.K)  # [B,T,K,3]
        parts = [f, logp_cand.unsqueeze(-1).float(), self.k_emb.expand(f.shape[0], f.shape[1], -1, -1)]
        if self.use_h:
            hp = self.hp(self.hn(h.float())).unsqueeze(2).expand(-1, -1, self.K, -1)
            parts.append(hp)
        return torch.cat(parts, -1)  # [B,T,K,din]

    def forward(self, h, logp_cand, aux):
        """Returns (log_g [B,T], log_a [B,T,K]) with log_a = -inf for invalid candidates."""
        z = self.features(h, logp_cand, aux)
        valid = aux["pred"][..., :self.K] >= 0
        s = self.score(z).squeeze(-1).masked_fill(~valid, -1e4)
        log_a = F.log_softmax(s, -1)
        g_logit = self.gate(z.flatten(2)).squeeze(-1)
        g_logit = g_logit.masked_fill(~valid.any(-1), -30.0)  # no candidates -> generator only
        return F.logsigmoid(g_logit), F.logsigmoid(-g_logit), log_a

    @staticmethod
    def mixture_logp_target(logp_rwkv_tgt, log_g, log_1mg, log_a, pred, tgt):
        """log p(target) under the mixture, per token."""
        hit = (pred == tgt.unsqueeze(-1))  # [B,T,K]
        log_copy = torch.logsumexp(log_a.masked_fill(~hit, -1e4), -1)  # log sum_k a_k [pred_k==tgt]
        return torch.logaddexp(log_1mg + logp_rwkv_tgt, log_g + log_copy)

    @staticmethod
    def mixture_logits(logits, log_g, log_1mg, log_a, pred):
        """Full-vocab mixture log-probs (for greedy decoding / analysis).  logits: [B,T,V]."""
        logp = F.log_softmax(logits.float(), -1) + log_1mg.unsqueeze(-1)
        copy = torch.full_like(logp, -1e4)
        copy = copy.scatter_reduce(-1, pred.clamp_min(0), (log_g.unsqueeze(-1) + log_a), reduce="amax", include_self=True)
        return torch.logaddexp(logp, copy)


class EngramLiteAdapter(nn.Module):
    """Hashed n-gram (n=2,3) multi-head embedding memory with Engram's context-aware gate.

    e_t = ||_{n,k} E_{n,k}[hash_{n,k}(x_{t-n+1..t})];  k_t = W_K e_t; v_t = W_V e_t
    alpha_t = sigmoid(RMSNorm(h_t) . RMSNorm(k_t) / sqrt(d));  y = alpha_t v_t + SiLU(DWConv(RMSNorm(alpha v)))
    """

    def __init__(self, C: int, orders=(2, 3), heads: int = 4, rows: int = 65536, dim: int = 32, seed: int = 0, vocab: int = 65536):
        super().__init__()
        self.orders = orders; self.heads = heads; self.rows = rows
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("mult", (torch.randint(1, 2**31 - 1, (max(orders),), generator=g) * 2 + 1))
        # distinct primes just below `rows` for each (order, head)
        primes = []; p = rows
        while len(primes) < len(orders) * heads:
            p -= 1
            if all(p % q for q in range(2, int(p ** 0.5) + 1)): primes.append(p)
        self.register_buffer("mods", torch.tensor(primes).view(len(orders), heads))
        self.tables = nn.Embedding(len(orders) * heads * rows, dim)
        nn.init.normal_(self.tables.weight, std=0.02)
        d_mem = len(orders) * heads * dim
        self.k_proj = nn.Linear(d_mem, C, bias=False); self.v_proj = nn.Linear(d_mem, C, bias=False)
        self.n1 = nn.RMSNorm(C); self.n2 = nn.RMSNorm(C); self.n3 = nn.RMSNorm(C)
        self.conv = nn.Conv1d(C, C, kernel_size=4, groups=C, dilation=max(orders), padding=(4 - 1) * max(orders))
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)
        self.C = C

    def hashes(self, idx):
        B, T = idx.shape
        shifts = [F.pad(idx, (k, 0), value=0)[:, :T] for k in range(max(self.orders))]
        out = []
        for oi, n in enumerate(self.orders):
            mix = shifts[0] * self.mult[0]
            for k in range(1, n):
                mix = torch.bitwise_xor(mix, shifts[k] * self.mult[k])
            for h in range(self.heads):
                out.append((mix % self.mods[oi, h]) + (oi * self.heads + h) * self.rows)
        return torch.stack(out, -1)  # [B,T,orders*heads]

    def forward(self, x, aux):
        idx = aux["idx"]
        e = self.tables(self.hashes(idx)).flatten(2)  # [B,T,d_mem] fp32
        k = self.k_proj(e); v = self.v_proj(e)
        gate = (self.n1(x.float()) * self.n2(k)).sum(-1) / math.sqrt(self.C)
        alpha = torch.sigmoid(gate).unsqueeze(-1)
        val = alpha * v
        y = self.conv(self.n3(val).transpose(1, 2))[:, :, : val.shape[1]].transpose(1, 2)
        return (val + F.silu(y)).to(x.dtype)
