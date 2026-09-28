"""Small task heads for frozen-backbone semantic-memory experiments."""
from __future__ import annotations

import torch
from torch import nn


class SceneBoundaryHead(nn.Module):
    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.config = dict(width=width, hidden=hidden)
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                 nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, hidden, positions):
        if positions.ndim != 2 or positions.shape[0] != hidden.shape[0]:
            raise ValueError("positions must have shape [batch, paragraphs]")
        gathered = hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        return self.net(gathered.float()).squeeze(-1)


class LocalResidualAdapter(nn.Module):
    """Per-token trainable residual control with no access to earlier memory."""

    def __init__(self, width: int, hidden: int = 128):
        super().__init__()
        self.config = dict(width=width, hidden=hidden)
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                 nn.GELU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, aux=None):
        if aux is not None and aux.get("disable_adapter", False):
            return torch.zeros_like(hidden)
        return self.net(hidden.float()).to(hidden.dtype)


class InitialStateTuner(nn.Module):
    """Learn one RWKV recurrent state per layer, shared across examples."""

    def __init__(self, layers: int, heads: int, head_size: int = 64):
        super().__init__()
        if min(layers, heads, head_size) < 1:
            raise ValueError("State dimensions must be positive")
        self.config = dict(layers=layers, heads=heads, head_size=head_size)
        self.state = nn.Parameter(torch.zeros(layers, heads, head_size, head_size))

    def initial_states(self, batch: int, device=None):
        if batch < 1:
            raise ValueError("Batch size must be positive")
        state = self.state if device is None else self.state.to(device)
        return state.unsqueeze(0).expand(batch, -1, -1, -1, -1).unbind(1)

    def forward(self, batch: int, device=None):
        return self.initial_states(batch, device)
