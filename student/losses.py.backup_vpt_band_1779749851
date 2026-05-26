"""Student one-step plus rollout loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .rollout import open_loop_rollout


def one_step_delta_loss(model, states: torch.Tensor, actions: torch.Tensor, normalizer) -> torch.Tensor:
    """Train one-step prediction while preserving recurrent hidden state."""
    batch_size = states.shape[0]
    hidden = model.initial_hidden(batch_size, states.device)
    losses = []
    for t in range(actions.shape[1]):
        obs_norm = normalizer.normalize_obs(states[:, t])
        act_norm = normalizer.normalize_act(actions[:, t])
        target_delta = states[:, t + 1] - states[:, t]
        target_norm = normalizer.normalize_delta(target_delta)
        pred_norm, hidden = model(obs_norm, act_norm, hidden)
        losses.append(F.mse_loss(pred_norm, target_norm))
    return torch.stack(losses).mean()


def _sample_subsequences(
    states: torch.Tensor,
    actions: torch.Tensor,
    needed_states: int,
    needed_actions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = states.shape[0]
    max_start = states.shape[1] - needed_states
    if max_start <= 0:
        return states[:, :needed_states], actions[:, :needed_actions]
    starts = torch.randint(0, max_start + 1, (batch_size,), device=states.device)
    sub_states = torch.stack([states[b, s : s + needed_states] for b, s in enumerate(starts)], dim=0)
    sub_actions = torch.stack([actions[b, s : s + needed_actions] for b, s in enumerate(starts)], dim=0)
    return sub_states, sub_actions


def _discounted_mse(pred_norm: torch.Tensor, target_norm: torch.Tensor, gamma: float) -> torch.Tensor:
    error = (pred_norm - target_norm).pow(2).mean(dim=-1)
    weights = gamma ** torch.arange(error.shape[1], device=error.device, dtype=error.dtype)
    weights = weights / weights.mean()
    return (error * weights.unsqueeze(0)).mean()


def rollout_loss(
    model,
    states: torch.Tensor,
    actions: torch.Tensor,
    normalizer,
    warmup_steps: int,
    horizon: int,
    gamma: float = 1.0,
) -> torch.Tensor:
    # Train local open-loop stability at random positions, not only at the
    # beginning of each stored window.
    needed_states = int(warmup_steps) + int(horizon) + 1
    needed_actions = int(warmup_steps) + int(horizon)
    if states.shape[1] < needed_states:
        raise ValueError(
            "training.train_sequence_length is too short for rollout loss: "
            f"need at least {needed_states - 1} actions for warmup={warmup_steps}, horizon={horizon}."
        )
    sub_states, sub_actions = _sample_subsequences(states, actions, needed_states, needed_actions)
    preds = open_loop_rollout(model, sub_states, sub_actions, normalizer, warmup_steps=warmup_steps, horizon=horizon)
    targets = sub_states[:, warmup_steps + 1 : warmup_steps + 1 + horizon]
    pred_norm = normalizer.normalize_obs(preds)
    target_norm = normalizer.normalize_obs(targets)
    if gamma >= 1.0:
        return F.mse_loss(pred_norm, target_norm)
    return _discounted_mse(pred_norm, target_norm, gamma=gamma)


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


def compute_loss(model, batch: dict[str, torch.Tensor], normalizer, cfg: dict):
    loss_cfg = cfg["loss"]
    states = batch["states"]
    actions = batch["actions"]
    step = _loss_step(model)
    one = one_step_delta_loss(model, states, actions, normalizer)
    horizon = int(
        _curriculum_value(
            step,
            loss_cfg.get("rollout_train_horizon", 5),
            loss_cfg.get("rollout_horizon_schedule", []),
        )
    )
    warmup = int(cfg["eval"].get("warmup_steps", 5))
    gamma = float(loss_cfg.get("rollout_discount_gamma", 1.0))
    rollout_weight = float(
        _curriculum_value(
            step,
            loss_cfg.get("rollout_weight", 0.3),
            loss_cfg.get("rollout_weight_schedule", []),
        )
    )
    if rollout_weight > 0.0:
        roll = rollout_loss(model, states, actions, normalizer, warmup_steps=warmup, horizon=horizon, gamma=gamma)
    else:
        roll = one.new_zeros(())
    total = float(loss_cfg.get("one_step_weight", 1.0)) * one + rollout_weight * roll
    return total, {
        "loss/total": float(total.detach().cpu()),
        "loss/one_step": float(one.detach().cpu()),
        "loss/rollout": float(roll.detach().cpu()),
        "loss/rollout_horizon": float(horizon),
        "loss/rollout_weight": float(rollout_weight),
    }
