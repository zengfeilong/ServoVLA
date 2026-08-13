from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from servovla.config.action_mode import normalize_action_delta_state_indices

TensorDict = dict[str, torch.Tensor]


def restore_absolute_actions(
    action_targets: torch.Tensor,
    q_current: torch.Tensor,
    *,
    action_is_delta: bool,
    action_delta_state_indices: list[int | None] | tuple[int | None, ...] | None = None,
) -> torch.Tensor:
    action_targets = action_targets.detach().cpu().float()
    q_current = q_current.detach().cpu().float()

    if action_targets.ndim == 2:
        action_targets = action_targets.unsqueeze(1)
    if q_current.ndim == 1:
        q_current = q_current.unsqueeze(0)

    if not action_is_delta:
        return action_targets

    action_dim = action_targets.shape[-1]
    state_indices = normalize_action_delta_state_indices(
        action_delta_state_indices,
        action_dim=action_dim,
    )
    basis = torch.zeros(
        (q_current.shape[0], action_dim), dtype=q_current.dtype, device=q_current.device
    )
    for action_idx, state_idx in enumerate(state_indices):
        if state_idx is None:
            continue
        if int(state_idx) >= q_current.shape[-1]:
            raise ValueError(
                f"Action delta state index {state_idx} for action dim {action_idx} exceeds "
                f"q_current dim {q_current.shape[-1]}"
            )
        basis[:, action_idx] = q_current[:, int(state_idx)]
    return action_targets + basis[:, None, :]


def _init_accumulator(action_dim: int) -> dict[str, Any]:
    return {
        "abs_sum": torch.zeros(action_dim, dtype=torch.float64),
        "sq_sum": torch.zeros(action_dim, dtype=torch.float64),
        "signed_sum": torch.zeros(action_dim, dtype=torch.float64),
        "count": 0,
    }


def _dataset_slugs_for_batch(batch: TensorDict, batch_size: int) -> list[str]:
    dataset_slug = batch.get("dataset_slug", "unknown")
    if isinstance(dataset_slug, tuple):
        dataset_slug = list(dataset_slug)
    if isinstance(dataset_slug, list):
        if len(dataset_slug) != batch_size:
            raise ValueError(
                f"Batch dataset_slug list length {len(dataset_slug)} does not match batch_size={batch_size}."
            )
        return [str(item) for item in dataset_slug]
    return [str(dataset_slug)] * batch_size


def _accumulate_metrics(
    accumulator: dict[str, Any],
    pred_abs: torch.Tensor,
    target_abs: torch.Tensor,
    loss_mask: torch.Tensor,
) -> None:
    pred_abs = pred_abs.detach().cpu().float()
    target_abs = target_abs.detach().cpu().float()
    loss_mask = loss_mask.detach().cpu().float()

    if pred_abs.ndim == 2:
        pred_abs = pred_abs.unsqueeze(1)
    if target_abs.ndim == 2:
        target_abs = target_abs.unsqueeze(1)
    if loss_mask.ndim == 1:
        loss_mask = loss_mask.unsqueeze(0)

    horizon = min(pred_abs.shape[1], target_abs.shape[1], loss_mask.shape[1])
    pred_abs = pred_abs[:, :horizon]
    target_abs = target_abs[:, :horizon]
    valid = loss_mask[:, :horizon] > 0

    if not valid.any():
        return

    err = pred_abs - target_abs
    mask = valid.unsqueeze(-1)
    accumulator["abs_sum"] += err.abs().masked_fill(~mask, 0).sum(dim=(0, 1), dtype=torch.float64)
    accumulator["sq_sum"] += err.square().masked_fill(~mask, 0).sum(dim=(0, 1), dtype=torch.float64)
    accumulator["signed_sum"] += err.masked_fill(~mask, 0).sum(dim=(0, 1), dtype=torch.float64)
    accumulator["count"] += int(valid.sum().item())


def _finalize_metrics(accumulator: dict[str, Any]) -> dict[str, Any]:
    count = int(accumulator["count"])
    action_dim = int(accumulator["abs_sum"].numel())
    if count == 0:
        zeros = [0.0] * action_dim
        return {
            "count": 0,
            "mae_per_joint": zeros,
            "rmse_per_joint": zeros,
            "signed_mean_per_joint": zeros,
            "mae_mean": 0.0,
            "rmse_mean": 0.0,
            "signed_mean": 0.0,
        }

    mae_per_joint = (accumulator["abs_sum"] / count).tolist()
    rmse_per_joint = torch.sqrt(accumulator["sq_sum"] / count).tolist()
    signed_mean_per_joint = (accumulator["signed_sum"] / count).tolist()
    return {
        "count": count,
        "mae_per_joint": [float(x) for x in mae_per_joint],
        "rmse_per_joint": [float(x) for x in rmse_per_joint],
        "signed_mean_per_joint": [float(x) for x in signed_mean_per_joint],
        "mae_mean": float(sum(mae_per_joint) / action_dim),
        "rmse_mean": float(sum(rmse_per_joint) / action_dim),
        "signed_mean": float(sum(signed_mean_per_joint) / action_dim),
    }


def evaluate_action_prediction_batches(
    batches: Iterable[TensorDict],
    predict_absolute_chunk_fn: Callable[[TensorDict], torch.Tensor],
    *,
    action_is_delta: bool,
    action_delta_state_indices: list[int | None] | tuple[int | None, ...] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    total_batches = 0
    all_steps_acc = None
    first_step_acc = None

    for batch_idx, batch in enumerate(batches):
        if max_batches is not None and batch_idx >= max_batches:
            break

        target_abs = restore_absolute_actions(
            batch["action"],
            batch["q_current"],
            action_is_delta=action_is_delta,
            action_delta_state_indices=action_delta_state_indices,
        )
        pred_abs = predict_absolute_chunk_fn(batch)
        if pred_abs.ndim == 2:
            pred_abs = pred_abs.unsqueeze(1)
        pred_abs = pred_abs.detach().cpu().float()[..., : target_abs.shape[-1]]

        if all_steps_acc is None:
            action_dim = int(target_abs.shape[-1])
            all_steps_acc = _init_accumulator(action_dim)
            first_step_acc = _init_accumulator(action_dim)

        _accumulate_metrics(all_steps_acc, pred_abs, target_abs, batch["loss_mask"])
        _accumulate_metrics(
            first_step_acc, pred_abs[:, :1], target_abs[:, :1], batch["loss_mask"][:, :1]
        )
        total_batches += 1

    if all_steps_acc is None or first_step_acc is None:
        raise ValueError("No batches were evaluated.")

    summary = {
        "num_batches": total_batches,
        "num_valid_steps": _finalize_metrics(all_steps_acc)["count"],
        "num_valid_first_steps": _finalize_metrics(first_step_acc)["count"],
        "all_steps": _finalize_metrics(all_steps_acc),
        "first_step": _finalize_metrics(first_step_acc),
    }
    return summary
