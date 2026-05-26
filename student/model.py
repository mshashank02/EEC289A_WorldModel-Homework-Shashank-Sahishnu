
"""Physics-informed student world model.

This keeps the simple GRU interface, but adds an explicit learnable
linearized inverted-pendulum prior. The residual network only needs to learn
what the structured prior misses.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


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
        self.act_dim = int(act_dim)
        self.hidden_dim = int(hidden_dim)

        in_dim = obs_dim + act_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        blocks: list[nn.Module] = []
        for _ in range(max(0, int(num_layers) - 1)):
            blocks.append(ResidualBlock(hidden_dim))
        self.backbone = nn.Sequential(*blocks)

        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None

        # Residual correction on top of the physics prior.
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # A small gate lets the model decide how much residual correction to use.
        self.gate_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Learnable normalized-time integration coefficients.
        # State order assumed for InvertedPendulum-v5:
        # [cart position, pole angle, cart velocity, pole angular velocity].
        self.dt_pos = nn.Parameter(torch.tensor([0.04, 0.04], dtype=torch.float32))

        # Linearized acceleration prior from [x, theta, x_dot, theta_dot, action].
        # Initialized small because inputs/targets are normalized.
        self.accel_prior = nn.Linear(in_dim, 2, bias=True)
        nn.init.zeros_(self.accel_prior.weight)
        nn.init.zeros_(self.accel_prior.bias)

        with torch.no_grad():
            if obs_dim >= 4 and act_dim >= 1:
                # cart acceleration prior: theta, x_dot damping, action
                self.accel_prior.weight[0, 1] = 0.02
                self.accel_prior.weight[0, 2] = -0.01
                self.accel_prior.weight[0, obs_dim] = 0.02

                # pole angular acceleration prior: unstable theta term, damping, action
                self.accel_prior.weight[1, 1] = 0.05
                self.accel_prior.weight[1, 3] = -0.01
                self.accel_prior.weight[1, obs_dim] = 0.02

        # Residual starts modest so early training respects the prior, but it can
        # grow if data says the prior is wrong.
        self.residual_log_scale = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def _physics_prior(self, obs_norm: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        prior = obs_norm.new_zeros(obs_norm.shape)

        if obs_norm.shape[-1] >= 4:
            # Kinematic prior: position-like dimensions integrate velocity-like dimensions.
            dt = F.softplus(self.dt_pos)
            prior[:, 0] = dt[0] * obs_norm[:, 2]
            prior[:, 1] = dt[1] * obs_norm[:, 3]

            # Dynamic prior: velocity deltas from a learned linearized model.
            accel = self.accel_prior(torch.cat([obs_norm, act_norm], dim=-1))
            prior[:, 2] = accel[:, 0]
            prior[:, 3] = accel[:, 1]

        return prior

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.encoder(x)
        feat = self.backbone(feat)

        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden

        prior_delta = self._physics_prior(obs_norm, act_norm)

        residual = self.residual_head(feat)
        residual_scale = F.softplus(self.residual_log_scale)
        gate = torch.sigmoid(self.gate_head(feat))

        raw_delta = prior_delta + gate * residual_scale * residual
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)

        return delta, hidden
