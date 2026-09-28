"""RWKV-7 "Goose" (x070) in GPT/parallel mode with the official CUDA kernel (fwd + bwd), plus
per-layer *adapter hooks* so ROSA / Engram style memory modules can be injected into the residual
stream before any block.

Model code follows BlinkDL/RWKV-LM RWKV-v7/rwkv_v7_demo.py (Apache-2.0); the kernel is
RWKV-v7/train_temp/cuda/wkv7_cuda.cu ("wind_backstepping", bf16 in/out, fp32 state).
"""
from __future__ import annotations
import os, math, types
import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

HEAD_SIZE = 64
CHUNK_LEN = 16
_kernel_loaded = False


def load_kernel():
    global _kernel_loaded
    if _kernel_loaded:
        return
    from torch.utils.cpp_extension import load
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")
    flags = ['-res-usage', f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization"]
    load(name="wind_backstepping", sources=[os.path.join(here, 'wkv7_cuda.cu'), os.path.join(here, 'wkv7_op.cpp')],
         is_python_module=False, verbose=False, extra_cuda_cflags=flags)
    _kernel_loaded = True


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b):
        B, T, H, C = w.shape
        assert T % CHUNK_LEN == 0
        assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
        if any(t.requires_grad for t in (w, q, k, v, z, b)):
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
        return y

    @staticmethod
    def backward(ctx, dy):
        assert dy.dtype == torch.bfloat16
        dy = dy.contiguous()
        w, q, k, v, z, b, s, sa = ctx.saved_tensors
        dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]
        with torch.cuda.device(w.device):
            torch.ops.wind_backstepping.backward(w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db)
        return dw, dq, dk, dv, dz, db


class WindBacksteppingState(torch.autograd.Function):
    @staticmethod
    def forward(ctx, decay, query, key, value, erase, write, initial):
        batch, length, heads, width = decay.shape
        inputs = (decay, query, key, value, erase, write)
        assert length % CHUNK_LEN == 0
        assert initial.shape == (batch, heads, width, width) and initial.dtype == torch.float32
        assert all(tensor.dtype == torch.bfloat16 for tensor in inputs)
        assert all(tensor.is_contiguous() for tensor in (*inputs, initial))
        output = torch.empty_like(value)
        states = torch.empty(batch, heads, length // CHUNK_LEN, width, width,
                             dtype=torch.float32, device=decay.device)
        projections = torch.empty_like(decay, dtype=torch.float32)
        torch.ops.wind_backstepping.forward_state(*inputs, initial, output, states, projections)
        if any(tensor.requires_grad for tensor in (*inputs, initial)):
            ctx.save_for_backward(*inputs, states, projections)
        return output

    @staticmethod
    def backward(ctx, output_gradient):
        assert output_gradient.dtype == torch.bfloat16
        *inputs, states, projections = ctx.saved_tensors
        gradients = [torch.empty_like(tensor) for tensor in inputs]
        batch, _, heads, width = inputs[0].shape
        initial_gradient = torch.empty(batch, heads, width, width,
                                       dtype=torch.float32, device=inputs[0].device)
        with torch.cuda.device(inputs[0].device):
            torch.ops.wind_backstepping.backward_state(
                *inputs, output_gradient.contiguous(), states, projections, *gradients, initial_gradient)
        return (*gradients, initial_gradient)


def RWKV7_OP(r, w, k, v, a, b, initial_state=None):
    B, T, HC = r.shape
    r, w, k, v, a, b = [i.view(B, T, HC // HEAD_SIZE, HEAD_SIZE).contiguous() for i in [r, w, k, v, a, b]]
    if initial_state is not None:
        if (initial_state.shape != (B, HC // HEAD_SIZE, HEAD_SIZE, HEAD_SIZE)
                or initial_state.dtype != torch.float32 or initial_state.device != r.device):
            raise ValueError("Initial state must be float32 [batch, heads, value, key] on the input device")
        initial_state = initial_state.contiguous()
    with torch.cuda.device(r.device):  # TORCH_LIBRARY op launches on the *current* device
        output = (WindBacksteppingState.apply(w, r, k, v, a, b, initial_state)
                  if initial_state is not None else WindBackstepping.apply(w, r, k, v, a, b))
        return output.view(B, T, HC)


class TimeMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        C = args.n_embd; H = C // HEAD_SIZE; N = HEAD_SIZE
        self.layer_id = layer_id; self.n_head = H
        for n in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "w0", "a0", "v0", "k_k", "k_a"]:
            setattr(self, n, nn.Parameter(torch.empty(1, 1, C)))
        self.w1 = nn.Parameter(torch.empty(C, args.d_decay)); self.w2 = nn.Parameter(torch.empty(args.d_decay, C))
        self.a1 = nn.Parameter(torch.empty(C, args.d_aaa)); self.a2 = nn.Parameter(torch.empty(args.d_aaa, C))
        self.v1 = nn.Parameter(torch.empty(C, args.d_mv)); self.v2 = nn.Parameter(torch.empty(args.d_mv, C))
        self.g1 = nn.Parameter(torch.empty(C, args.d_gate)); self.g2 = nn.Parameter(torch.empty(args.d_gate, C))
        self.r_k = nn.Parameter(torch.empty(H, N))
        self.receptance = nn.Linear(C, C, bias=False); self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False); self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first, initial_state=None):
        B, T, C = x.size(); H = self.n_head
        xx = F.pad(x, (0, 0, 1, -1)) - x
        xr = x + xx * self.x_r; xw = x + xx * self.x_w; xk = x + xx * self.x_k
        xv = x + xx * self.x_v; xa = x + xx * self.x_a; xg = x + xx * self.x_g
        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk); v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        x = RWKV7_OP(r, w, k, v, -kk, kk * a, initial_state)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        return self.output(x * g), v_first


class ChannelMix(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.x_k = nn.Parameter(torch.empty(1, 1, args.n_embd))
        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

    def forward(self, x):
        xx = F.pad(x, (0, 0, 1, -1)) - x
        k = x + xx * self.x_k
        return self.value(torch.relu(self.key(k)) ** 2)


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
        self.ln1 = nn.LayerNorm(args.n_embd); self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = TimeMix(args, layer_id); self.ffn = ChannelMix(args, layer_id)

    def forward(self, x, v_first, initial_state=None):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first, initial_state)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RWKV7(nn.Module):
    """RWKV-7 with optional adapters.  ``adapters`` maps layer index -> module with signature
    ``adapter(x, aux) -> delta`` added to the residual stream *before* block ``i``.  ``aux`` is an
    arbitrary dict passed from ``forward`` (e.g. precomputed ROSA features)."""

    def __init__(self, n_layer, n_embd, vocab_size=65536, d_decay=64, d_aaa=64, d_mv=32, d_gate=128):
        super().__init__()
        args = types.SimpleNamespace(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab_size, d_decay=d_decay, d_aaa=d_aaa, d_mv=d_mv, d_gate=d_gate)
        self.args = args
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(n_layer)])
        self.ln_out = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.adapters = nn.ModuleDict()
        self.grad_ckpt = False
        self.state_adapter = None

    def add_adapter(self, layer: int, module: nn.Module):
        self.adapters[str(layer)] = module

    def add_state_adapter(self, module: nn.Module):
        expected = dict(layers=self.args.n_layer, heads=self.args.n_embd // HEAD_SIZE, head_size=HEAD_SIZE)
        if module.config != expected:
            raise ValueError("Initial state dimensions do not match the RWKV backbone")
        self.state_adapter = module

    def forward(self, idx, aux=None, return_hidden=False, last_only=False):
        B, T = idx.shape
        pad = (-T) % CHUNK_LEN
        if pad:
            idx = F.pad(idx, (0, pad), value=0)
            if aux is not None and len(self.adapters):  # keep adapter features aligned with the padded sequence
                fill = {"pred": -1, "src": -1}
                aux = {k: (F.pad(v, (0, 0) * (v.dim() - 2) + (0, pad), value=fill.get(k, 0)) if torch.is_tensor(v) and v.dim() >= 2 and v.shape[1] == T else v) for k, v in aux.items()}
        x = self.emb(idx)
        v_first = None
        initial_states = None
        if self.state_adapter is not None and not (aux is not None and aux.get("disable_state", False)):
            initial_states = self.state_adapter.initial_states(B, device=x.device)
        for i, block in enumerate(self.blocks):
            ad = self.adapters[str(i)] if str(i) in self.adapters else None
            if ad is not None:
                x = x + ad(x, aux)
            state = None if initial_states is None else initial_states[i]
            if self.grad_ckpt and torch.is_grad_enabled():
                x, v_first = torch.utils.checkpoint.checkpoint(block, x, v_first, state, use_reentrant=False)
            else:
                x, v_first = block(x, v_first, state)
        x = self.ln_out(x)
        if pad:
            x = x[:, :T]
        if last_only:
            x = x[:, -1:]
        logits = self.head(x)
        return (logits, x) if return_hidden else logits


def infer_config(sd):
    n_layer = 1 + max(int(k.split('.')[1]) for k in sd if k.startswith('blocks.'))
    n_embd = sd['emb.weight'].shape[1]; vocab = sd['emb.weight'].shape[0]
    return dict(n_layer=n_layer, n_embd=n_embd, vocab_size=vocab,
                d_decay=sd['blocks.0.att.w1'].shape[1], d_aaa=sd['blocks.0.att.a1'].shape[1],
                d_mv=sd['blocks.1.att.v1'].shape[1], d_gate=sd['blocks.0.att.g1'].shape[1])


def load_rwkv7(path: str, device="cuda", dtype=torch.bfloat16) -> RWKV7:
    load_kernel()
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    cfg = infer_config(sd)
    model = RWKV7(**cfg)
    # layer 0 has no v0/v1/v2 in the checkpoint? (it does exist in g1 series but is unused) -> strict=False
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if not m.startswith('adapters.')]
    assert not unexpected, unexpected
    assert all(('blocks.0.att.v' in m) for m in missing), missing
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


MODELS = {
    "0.4b": "/root/x/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth",
    "1.5b": "/root/x/models/rwkv7/rwkv7-g1j-1.5b-20260831-ctx16384.pth",
}
VOCAB = "/root/x/models/rwkv7/rwkv_vocab_v20230424.txt"


def get_tokenizer():
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    return TRIE_TOKENIZER(VOCAB)
