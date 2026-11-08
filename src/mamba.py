"""Mamba (selective state-space model) from scratch in pure PyTorch.

Follows Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces",
but with an explicit, readable for-loop selective scan instead of the fused CUDA kernel.

Shapes (B = batch, L = sequence length, D = d_model, E = d_inner = expand * D,
        N = d_state, R = dt_rank, K = d_conv)

    x (B, L, D) -> in_proj -> [x (B, L, E) | z (B, L, E)]
    x -> causal depthwise conv1d (kernel K) -> SiLU
    x -> x_proj -> [dt_low (B, L, R) | B (B, L, N) | C (B, L, N)]
    dt = softplus(dt_proj(dt_low))            (B, L, E)    input-dependent step size
    A  = -exp(A_log)                          (E, N)       negative real -> stable
    selective scan:  h_t = exp(dt_t A) * h_{t-1} + dt_t B_t x_t     (zero-order hold on A,
                     y_t = C_t . h_t + D * x_t                       Euler on B)
    y * SiLU(z) -> out_proj -> (B, L, D)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(u, delta, A, B, C, D):
    """Explicit sequential selective scan.

    u:     (B, L, E)  input after conv + SiLU
    delta: (B, L, E)  positive step sizes
    A:     (E, N)     negative state matrix (diagonal, one row per channel)
    B, C:  (B, L, N)  input-dependent input/output projections (shared across channels)
    D:     (E,)       skip connection
    returns y: (B, L, E)
    """
    b, L, E = u.shape
    N = A.shape[1]
    # Discretise once for all timesteps (vectorised), then recur in a plain loop.
    dA = torch.exp(delta.unsqueeze(-1) * A)                              # (B, L, E, N)
    dBu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)         # (B, L, E, N)

    # unbind(1) splits along time ONCE. (Indexing dA[:, t] inside the loop would make
    # autograd allocate a full (B, L, E, N) gradient for every t -> O(L^2) backward.)
    dA_t, dBu_t, C_t = dA.unbind(1), dBu.unbind(1), C.unsqueeze(2).unbind(1)

    h = u.new_zeros(b, E, N)
    ys = []
    for t in range(L):
        h = dA_t[t] * h + dBu_t[t]                                       # (B, E, N)
        y_t = (h * C_t[t]).sum(-1)                                       # (B, E)
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                                           # (B, L, E)
    return y + u * D


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_rank: int | str = "auto", dt_min: float = 1e-3, dt_max: float = 1e-1,
                 bias: bool = False, conv_bias: bool = True):
        super().__init__()
        self.d_model, self.d_state, self.d_conv = d_model, d_state, d_conv
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        E, N, R = self.d_inner, d_state, self.dt_rank

        self.in_proj = nn.Linear(d_model, 2 * E, bias=bias)
        # depthwise (groups=E) causal conv: pad K-1 on both sides, keep the first L outputs
        self.conv1d = nn.Conv1d(E, E, kernel_size=d_conv, groups=E,
                                padding=d_conv - 1, bias=conv_bias)
        self.x_proj = nn.Linear(E, R + 2 * N, bias=False)
        self.dt_proj = nn.Linear(R, E, bias=True)

        # dt bias init so that softplus(bias) ~ LogUniform[dt_min, dt_max] (as in the paper)
        dt = torch.exp(torch.rand(E) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))   # inverse softplus
        nn.init.uniform_(self.dt_proj.weight, -R ** -0.5, R ** -0.5)

        # S4D-real init: A[:, n] = -(n + 1)
        A = torch.arange(1, N + 1, dtype=torch.float32).repeat(E, 1)     # (E, N)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(E))
        self.out_proj = nn.Linear(E, d_model, bias=bias)

    # ---- helpers ---------------------------------------------------------------------
    def _ssm_params(self, x):
        """x (..., E) -> delta (..., E), B (..., N), C (..., N)."""
        dt_low, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], -1)
        delta = F.softplus(self.dt_proj(dt_low))
        return delta, B, C

    # ---- full-sequence forward (training) -------------------------------------------
    def forward(self, x):
        """x: (B, L, D) -> (B, L, D)"""
        b, L, _ = x.shape
        xz = self.in_proj(x)                                  # (B, L, 2E)
        x, z = xz.chunk(2, dim=-1)                            # (B, L, E) each
        x = self.conv1d(x.transpose(1, 2))[..., :L]           # causal: drop right padding
        x = F.silu(x.transpose(1, 2))                         # (B, L, E)
        delta, B, C = self._ssm_params(x)
        A = -torch.exp(self.A_log)                            # (E, N)
        y = selective_scan(x, delta, A, B, C, self.D)         # (B, L, E)
        y = y * F.silu(z)
        return self.out_proj(y)

    # ---- streaming inference --------------------------------------------------------
    def init_state(self, batch: int, device=None, dtype=None):
        p = self.A_log
        conv_state = torch.zeros(batch, self.d_inner, self.d_conv, device=device or p.device,
                                 dtype=dtype or p.dtype)
        h = torch.zeros(batch, self.d_inner, self.d_state, device=device or p.device,
                        dtype=dtype or p.dtype)
        return conv_state, h

    def step(self, x_t, conv_state, h):
        """One timestep. x_t (B, D); conv_state (B, E, K) last K conv inputs; h (B, E, N).

        Returns (y_t (B, D), new conv_state, new h). Pure function: inputs are not mutated.
        """
        xz = self.in_proj(x_t)
        x, z = xz.chunk(2, dim=-1)                            # (B, E)
        conv_state = torch.cat([conv_state[:, :, 1:], x.unsqueeze(-1)], dim=-1)
        x = (conv_state * self.conv1d.weight.squeeze(1)).sum(-1)    # weight (E, 1, K)
        if self.conv1d.bias is not None:
            x = x + self.conv1d.bias
        x = F.silu(x)
        delta, B, C = self._ssm_params(x)                     # (B, E), (B, N), (B, N)
        A = -torch.exp(self.A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A)               # (B, E, N)
        h = dA * h + delta.unsqueeze(-1) * B.unsqueeze(1) * x.unsqueeze(-1)
        y = (h * C.unsqueeze(1)).sum(-1) + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y), conv_state, h


class ResidualMambaLayer(nn.Module):
    """Pre-norm residual: x + Mamba(LayerNorm(x))."""

    def __init__(self, d_model, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = MambaBlock(d_model, **kw)

    def forward(self, x):
        return x + self.mixer(self.norm(x))

    def step(self, x_t, state):
        y, conv_state, h = self.mixer.step(self.norm(x_t), *state)
        return x_t + y, (conv_state, h)


class MambaStack(nn.Module):
    """N residual Mamba layers + final LayerNorm."""

    def __init__(self, d_model: int, n_layers: int = 2, **kw):
        super().__init__()
        self.layers = nn.ModuleList([ResidualMambaLayer(d_model, **kw) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)

    def init_state(self, batch, device=None, dtype=None):
        return [l.mixer.init_state(batch, device, dtype) for l in self.layers]

    def step(self, x_t, states):
        new_states = []
        for layer, st in zip(self.layers, states):
            x_t, st = layer.step(x_t, st)
            new_states.append(st)
        return self.norm_f(x_t), new_states


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
