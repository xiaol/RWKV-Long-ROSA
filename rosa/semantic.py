"""Causal multi-hop retrieval from contextual hidden states of a frozen backbone."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class SemanticMemoryAdapter(nn.Module):
    """Pool completed chunks, retrieve by learned similarity, and inject a gated residual.

    This is a learned memory alternative to token-suffix ROSA, not an exact suffix
    automaton. Queries are processed in blocks to bound the temporary score tensor.
    ``aux['memory_mask']`` optionally restricts which tokens may enter memory.
    """

    def __init__(self, width: int, memory_dim: int = 64, hops: int = 2,
                 chunk_size: int = 32, query_block: int = 128,
                 temperature: float = 0.1):
        super().__init__()
        if min(width, memory_dim, hops, chunk_size, query_block) < 1 or temperature <= 0:
            raise ValueError("Dimensions, hops, block sizes, and temperature must be positive")
        self.config = dict(width=width, memory_dim=memory_dim, hops=hops,
                           chunk_size=chunk_size, query_block=query_block,
                           temperature=temperature)
        self.hops = hops
        self.chunk_size = chunk_size
        self.query_block = query_block
        self.temperature = temperature
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, memory_dim, bias=False)
        self.key = nn.Linear(width, memory_dim, bias=False)
        self.value = nn.Linear(width, memory_dim, bias=False)
        self.update = nn.Linear(2 * memory_dim, memory_dim)
        self.gate = nn.Linear(2 * memory_dim, 1)
        self.output = nn.Linear(memory_dim, width, bias=False)
        nn.init.constant_(self.gate.bias, -3.0)
        nn.init.zeros_(self.output.weight)

    def retrieve(self, query, keys, values, allowed, return_trace=False):
        """Read the same memory repeatedly, conditioning each hop on the previous read."""
        current = F.normalize(query, dim=-1)
        trace = []
        for hop in range(self.hops):
            scores = current @ keys.transpose(-1, -2) / self.temperature
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            weights = scores.softmax(-1) * allowed.to(scores.dtype)
            weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
            read = weights @ values
            if return_trace:
                trace.append(weights)
            if hop + 1 < self.hops:
                current = F.normalize(current + torch.tanh(
                    self.update(torch.cat([current, read], -1))), dim=-1)
        if return_trace:
            return read, torch.stack(trace, dim=1)
        return read

    def _read_block(self, query, keys, values, allowed):
        read = self.retrieve(query, keys, values, allowed)
        gate = torch.sigmoid(self.gate(torch.cat([query, read], -1)))
        return self.output(read * gate)

    def forward(self, hidden, aux=None):
        batch, length, width = hidden.shape
        if length == 0:
            return torch.zeros_like(hidden)
        normalized = self.norm(hidden.to(self.norm.weight.dtype))
        memory_mask = None if aux is None else aux.get("memory_mask")
        if memory_mask is None:
            memory_mask = torch.ones(batch, length, device=hidden.device, dtype=torch.bool)
        if memory_mask.shape != (batch, length):
            raise ValueError("memory_mask must match hidden [batch, length]")
        memory_mask = memory_mask.to(device=hidden.device, dtype=torch.bool)
        padding = (-length) % self.chunk_size
        masked = normalized.masked_fill(~memory_mask.unsqueeze(-1), 0)
        chunks = F.pad(masked, (0, 0, 0, padding)).reshape(
            batch, -1, self.chunk_size, width)
        counts = F.pad(memory_mask, (0, padding)).reshape(
            batch, -1, self.chunk_size).sum(-1)
        pooled = chunks.sum(2) / counts.clamp_min(1).unsqueeze(-1)
        keys = F.normalize(self.key(pooled), dim=-1)
        values = self.value(pooled)
        queries = self.query(normalized)
        chunk_ends = (torch.arange(pooled.shape[1], device=hidden.device) + 1) * self.chunk_size - 1
        outputs = []
        for start in range(0, length, self.query_block):
            stop = min(start + self.query_block, length)
            positions = torch.arange(start, stop, device=hidden.device)
            allowed = (chunk_ends[None, None, :] < positions[None, :, None]) & (counts[:, None, :] > 0)
            arguments = (queries[:, start:stop], keys, values, allowed)
            if self.training and torch.is_grad_enabled():
                delta = checkpoint(self._read_block, *arguments, use_reentrant=False)
            else:
                delta = self._read_block(*arguments)
            outputs.append(delta)
        return torch.cat(outputs, dim=1).to(hidden.dtype)
