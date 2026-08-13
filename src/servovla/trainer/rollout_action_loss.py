from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def rollout_action_loss_weight(cfg: Any, step: int) -> float:
    if cfg is None or not bool(getattr(cfg, "enabled", False)):
        return 0.0

    max_weight = float(getattr(cfg, "max_weight", 0.0))
    if max_weight <= 0.0:
        return 0.0

    start_step = int(getattr(cfg, "start_step", 0))
    end_step = int(getattr(cfg, "end_step", start_step))
    if int(step) <= start_step:
        return 0.0
    if end_step <= start_step:
        return max_weight

    progress = (float(step) - float(start_step)) / float(end_step - start_step)
    progress = min(max(progress, 0.0), 1.0)

    schedule = str(getattr(cfg, "schedule", "cosine")).lower()
    if schedule == "linear":
        ramp = progress
    elif schedule == "cosine":
        ramp = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        raise ValueError(f"Unsupported rollout_action_loss schedule: {schedule}")

    return float(max_weight * ramp)


def differentiable_euler_rollout(
    *,
    policy_head: torch.nn.Module,
    x_0: torch.Tensor,
    f_vision: torch.Tensor,
    c_sem: torch.Tensor,
    c_sem_mask: torch.Tensor,
    frame_delay: torch.Tensor,
    q_current: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    steps = max(int(num_inference_steps), 1)
    dt = 1.0 / float(steps)
    x_t = x_0
    batch_size = int(x_t.shape[0])

    for step_idx in range(steps):
        t = torch.full(
            (batch_size,),
            step_idx * dt,
            device=x_t.device,
            dtype=x_t.dtype,
        )
        v_t = policy_head(
            x_t=x_t,
            t=t,
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )
        x_t = x_t + dt * v_t

    return x_t


def masked_action_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_type: str,
    beta: float,
) -> torch.Tensor:
    loss_type = str(loss_type).lower()
    if loss_type == "smooth_l1":
        per_dim = F.smooth_l1_loss(pred.float(), target.float(), reduction="none", beta=float(beta))
    elif loss_type == "l1":
        per_dim = F.l1_loss(pred.float(), target.float(), reduction="none")
    elif loss_type == "mse":
        per_dim = F.mse_loss(pred.float(), target.float(), reduction="none")
    else:
        raise ValueError(f"Unsupported rollout_action_loss loss_type: {loss_type}")

    per_step = per_dim.mean(dim=-1)
    mask = loss_mask.float()
    return (per_step * mask).sum() / mask.sum().clamp(min=1.0)


def compute_rollout_action_loss(
    *,
    policy_head: torch.nn.Module,
    x_1: torch.Tensor,
    x_0: torch.Tensor,
    loss_mask: torch.Tensor,
    f_vision: torch.Tensor,
    c_sem: torch.Tensor,
    c_sem_mask: torch.Tensor,
    frame_delay: torch.Tensor,
    q_current: torch.Tensor,
    num_inference_steps: int,
    loss_type: str,
    beta: float,
) -> torch.Tensor:
    pred = differentiable_euler_rollout(
        policy_head=policy_head,
        x_0=x_0,
        f_vision=f_vision,
        c_sem=c_sem,
        c_sem_mask=c_sem_mask,
        frame_delay=frame_delay,
        q_current=q_current,
        num_inference_steps=num_inference_steps,
    )
    return masked_action_loss(
        pred,
        x_1,
        loss_mask,
        loss_type=loss_type,
        beta=beta,
    )
