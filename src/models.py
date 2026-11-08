"""Models built on the Mamba stack."""
from __future__ import annotations

import torch.nn as nn

from .mamba import MambaStack


class MambaForecaster(nn.Module):
    """Linear(in_ch -> d_model) -> Mamba stack -> Linear(d_model -> out_ch).

    Trained for next-step prediction: output at time t is the prediction of the
    signals at time t+1. The model is causal, so this never peeks at the future.
    """

    def __init__(self, in_channels: int, out_channels: int, d_model: int = 32,
                 n_layers: int = 2, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.config = dict(in_channels=in_channels, out_channels=out_channels, d_model=d_model,
                           n_layers=n_layers, d_state=d_state, d_conv=d_conv, expand=expand)
        self.embed = nn.Linear(in_channels, d_model)
        self.backbone = MambaStack(d_model, n_layers, d_state=d_state, d_conv=d_conv,
                                   expand=expand)
        self.head = nn.Linear(d_model, out_channels)

    def forward(self, x):
        """x (B, L, in_ch) -> (B, L, out_ch)"""
        return self.head(self.backbone(self.embed(x)))

    def init_state(self, batch, device=None, dtype=None):
        return self.backbone.init_state(batch, device, dtype)

    def step(self, x_t, states):
        """Streaming: x_t (B, in_ch) -> prediction for t+1 (B, out_ch), new states."""
        h, states = self.backbone.step(self.embed(x_t), states)
        return self.head(h), states
