"""Student world model.

Students may replace this residual MLP with a GRU or another dynamics model,
but the public interface must stay the same.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        self.obs_dim = int(obs_dim)
        self.q_dim = self.obs_dim // 2
        self.v_dim = self.obs_dim - self.q_dim

        self.input = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
        )
        self.encoder = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(int(num_layers))])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.post_norm = nn.LayerNorm(hidden_dim)

        # MoSim-style split: learn a velocity-rate term, integrate it into a
        # structured delta, then let a residual corrector handle model mismatch.
        self.accel_head = nn.Linear(hidden_dim, self.v_dim)
        self.corrector_head = nn.Linear(hidden_dim, obs_dim)
        self.gate_head = nn.Linear(hidden_dim, obs_dim)
        self.q_rate_scale_raw = nn.Parameter(torch.full((self.q_dim,), -3.0))
        self.v_rate_scale_raw = nn.Parameter(torch.full((self.v_dim,), -1.0))

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        feat = self.input(torch.cat([obs_norm, act_norm], dim=-1))
        feat = self.encoder(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        feat = self.post_norm(feat)

        q = obs_norm[:, : self.q_dim]
        v = obs_norm[:, self.q_dim :]
        raw_accel = self.accel_head(feat)
        v_scale = F.softplus(self.v_rate_scale_raw).unsqueeze(0)
        v_delta = v_scale * torch.tanh(raw_accel)

        q_scale = F.softplus(self.q_rate_scale_raw).unsqueeze(0)
        q_delta = q_scale * (v + 0.5 * v_delta)
        structured_delta = torch.cat([q_delta, v_delta], dim=-1)

        correction = self.corrector_head(feat)
        gate = torch.sigmoid(self.gate_head(feat))
        raw_delta = structured_delta + gate * correction
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
