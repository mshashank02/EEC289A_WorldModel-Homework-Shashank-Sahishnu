
"""Student one-step plus VPT-aware rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer):
    """Train one-step prediction while preserving recurrent hidden state."""
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    losses = []
    kl_terms = []

    for t in range(actions.shape[1]):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        target_delta = states[:, t + 1] - states[:, t]
        target_norm = normalizer.normalize_delta(target_delta)

        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        losses.append(F.mse_loss(pred_norm, target_norm))

        kl = getattr(model, "_last_kl", None)
        if kl is not None:
            kl_terms.append(kl)

    one = torch.stack(losses).mean()
    kl_loss = torch.stack(kl_terms).mean() if kl_terms else states.new_zeros(())
    return one, kl_loss


def _sample_subsequences(
    states: torch.Tensor,
    actions: torch.Tensor,
    needed_states: int,
    needed_actions: int,
):
    batch_size = states.shape[0]
    max_start = states.shape[1] - needed_states
    if max_start <= 0:
        return states[:, :needed_states], actions[:, :needed_actions]

    starts = torch.randint(0, max_start + 1, (batch_size,), device=states.device)
    sub_states = torch.stack([states[b, s : s + needed_states] for b, s in enumerate(starts)], dim=0)
    sub_actions = torch.stack([actions[b, s : s + needed_actions] for b, s in enumerate(starts)], dim=0)
    return sub_states, sub_actions


def _curriculum_value(step: int, default_value, schedule):
    value = default_value
    for threshold, scheduled_value in schedule:
        if step >= int(threshold):
            value = scheduled_value
    return value


def _loss_step(model) -> int:
    step = int(getattr(model, "_student_loss_step", 0)) + 1
    model._student_loss_step = step
    return step


def _component_weights(loss_cfg: dict, device, dtype, obs_dim: int):
    weights = torch.ones(obs_dim, device=device, dtype=dtype)

    # Assumes InvertedPendulum obs order is roughly:
    # [cart position, pole angle, cart velocity, pole angular velocity].
    if obs_dim >= 4:
        weights[0] = float(loss_cfg.get("x_weight", 0.75))
        weights[1] = float(loss_cfg.get("theta_weight", 2.0))
        weights[2] = float(loss_cfg.get("x_dot_weight", 1.0))
        weights[3] = float(loss_cfg.get("theta_dot_weight", 1.5))

    # Keep average scale comparable to ordinary MSE.
    weights = weights / weights.mean().clamp_min(1e-8)
    return weights


def _vpt_band_rollout_mse(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    loss_cfg: dict,
):
    """Rollout loss that emphasizes the VPT threshold band.

    Official nMSE/VPT is based on rollout-average normalized MSE, so the
    band is computed from cumulative mean error, not only per-step error.
    """
    error_by_dim = (pred_norm - target_norm).pow(2)
    comp_w = _component_weights(loss_cfg, pred_norm.device, pred_norm.dtype, pred_norm.shape[-1])
    step_error = (error_by_dim * comp_w.view(1, 1, -1)).mean(dim=-1)

    steps = torch.arange(1, step_error.shape[1] + 1, device=step_error.device, dtype=step_error.dtype)
    rollout_avg_error = torch.cumsum(step_error, dim=1) / steps.view(1, -1)

    gamma = float(loss_cfg.get("rollout_discount_gamma", 1.0))
    time_w = gamma ** torch.arange(step_error.shape[1], device=step_error.device, dtype=step_error.dtype)
    time_w = time_w / time_w.mean().clamp_min(1e-8)

    threshold = float(loss_cfg.get("vpt_threshold", 0.25))
    band_width = float(loss_cfg.get("vpt_band_width", 0.20))
    temp = float(loss_cfg.get("vpt_band_temperature", 0.04))
    band_weight = float(loss_cfg.get("vpt_band_weight", 0.0))
    survival_weight = float(loss_cfg.get("vpt_survival_weight", 0.0))

    lower = threshold - band_width
    upper = threshold + band_width

    # High when rollout-average error is near the threshold band.
    band = torch.sigmoid((rollout_avg_error - lower) / temp) * torch.sigmoid((upper - rollout_avg_error) / temp)
    weighted_step_error = step_error * (1.0 + band_weight * band)

    base_loss = (weighted_step_error * time_w.view(1, -1)).mean()

    if survival_weight <= 0.0:
        return base_loss

    # Smooth penalty for being above the VPT threshold.
    survival_penalty = F.softplus((rollout_avg_error - threshold) / temp) * temp
    survival_loss = (survival_penalty * time_w.view(1, -1)).mean()
    return base_loss + survival_weight * survival_loss


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    loss_cfg: dict,
):
    needed_states = int(warmup_steps) + int(horizon) + 1
    needed_actions = int(warmup_steps) + int(horizon)

    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}, "
            f"but got {states.shape[1] - 1}."
        )

    sub_states, sub_actions = _sample_subsequences(states, actions, needed_states, needed_actions)
    preds = open_loop_rollout(
        model,
        sub_states,
        sub_actions,
        normalizer,
        warmup_steps=warmup_steps,
        horizon=horizon,
    )
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]

    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)

    if float(loss_cfg.get("vpt_band_weight", 0.0)) > 0.0 or float(loss_cfg.get("vpt_survival_weight", 0.0)) > 0.0:
        return _vpt_band_rollout_mse(pred_norm, target_norm, loss_cfg)

    gamma = float(loss_cfg.get("rollout_discount_gamma", 1.0))
    step_error = (pred_norm - target_norm).pow(2).mean(dim=-1)
    if gamma >= 1.0:
        return step_error.mean()

    time_w = gamma ** torch.arange(step_error.shape[1], device=step_error.device, dtype=step_error.dtype)
    time_w = time_w / time_w.mean().clamp_min(1e-8)
    return (step_error * time_w.view(1, -1)).mean()


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]

    step = _loss_step(model)
    one, kl = one_step_delta_loss(model, states, actions, normalizer)

    default_horizon = loss_cfg.get("rollout_train_horizon", 5)
    horizon = int(_curriculum_value(step, default_horizon, loss_cfg.get("rollout_horizon_schedule", [])))

    # Optional multi-horizon loss. Example: rollout_train_horizons: [20, 40, 60]
    horizons = loss_cfg.get("rollout_train_horizons", None)
    if horizons is None:
        horizons = [horizon]
    horizons = [int(h) for h in horizons]

    warmup = int(cfg["eval"].get("warmup_steps", 5))
    rollout_weight = float(
        _curriculum_value(
            step,
            loss_cfg.get("rollout_weight", 0.3),
            loss_cfg.get("rollout_weight_schedule", []),
        )
    )

    if rollout_weight > 0.0:
        roll_terms = [
            rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=h, loss_cfg=loss_cfg)
            for h in horizons
        ]
        roll = torch.stack(roll_terms).mean()
    else:
        roll = one.new_zeros(())

    kl_weight = float(loss_cfg.get("kl_weight", 0.0))
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + rollout_weight * roll + kl_weight * kl

    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/kl": float(kl.detach().cpu()),
        "loss/kl_weight": float(kl_weight),
        "loss/rollout_horizon": float(max(horizons)),
        "loss/rollout_weight": float(rollout_weight),
    }
