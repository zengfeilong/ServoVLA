from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from servovla.architectures.servo_vla import ServoVLA
from servovla.evaluation.action_metrics import (
    _accumulate_metrics,
    _finalize_metrics,
    _init_accumulator,
    evaluate_action_prediction_batches,
    restore_absolute_actions,
)

TensorDict = dict[str, Any]


def _autocast_ctx(device: torch.device, dtype: torch.dtype | None):
    if dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


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


def evaluate_model_on_batches(
    batches: Iterable[TensorDict],
    predict_absolute_chunk_fn: Callable[[TensorDict], torch.Tensor],
    *,
    action_is_delta: bool,
    action_delta_state_indices: list[int | None] | tuple[int | None, ...] | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    return evaluate_action_prediction_batches(
        batches,
        predict_absolute_chunk_fn,
        action_is_delta=action_is_delta,
        action_delta_state_indices=action_delta_state_indices,
        max_batches=max_batches,
    )


def _predict_vector_field(
    model: torch.nn.Module,
    *,
    x_t: torch.Tensor,
    t: torch.Tensor,
    pixel_values: torch.Tensor,
    vlm_inputs: dict[str, torch.Tensor],
    c_sem_mask: torch.Tensor,
    frame_delay: torch.Tensor,
    q_current: torch.Tensor,
) -> torch.Tensor:
    if isinstance(model, ServoVLA):
        return model(
            x_t=x_t,
            t=t,
            pixel_values=pixel_values,
            vlm_inputs=vlm_inputs,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )
    return model(x_t, t, pixel_values, vlm_inputs, c_sem_mask, frame_delay, q_current)


def _normalize_action_targets(model: torch.nn.Module, action: torch.Tensor) -> torch.Tensor:
    action_normalizer = getattr(model, "action_normalizer", None)
    if action_normalizer is None or not bool(getattr(action_normalizer, "enabled", False)):
        return action
    return action_normalizer.normalize(action)


def _policy_param(model: torch.nn.Module) -> torch.nn.Parameter:
    policy_head = getattr(model, "policy_head", None)
    if policy_head is not None:
        return next(policy_head.parameters())
    return next(model.parameters())


def evaluate_loss_on_batches(
    model: torch.nn.Module,
    batches: Iterable[TensorDict],
    *,
    device: str,
    max_batches: int | None = None,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    eval_device = torch.device(device)
    model_param = _policy_param(model)
    model_dtype = model_param.dtype
    noise_device = eval_device if eval_device.type != "cpu" else torch.device("cpu")
    generator = torch.Generator(device=noise_device).manual_seed(int(seed))

    total_fm_loss_sum = 0.0
    total_weight = 0.0
    dataset_fm_loss_sum: dict[str, float] = {}
    dataset_weight_sum: dict[str, float] = {}
    num_batches = 0

    for batch_idx, batch in enumerate(batches):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_1 = batch["action"].to(eval_device, dtype=model_dtype)
        x_1 = _normalize_action_targets(model, x_1)
        loss_mask = batch["loss_mask"].to(eval_device)
        pixel_values = batch["pixel_values"].to(eval_device)
        vlm_inputs = {
            key: value.to(eval_device) if torch.is_tensor(value) else value
            for key, value in batch["vlm_inputs"].items()
        }
        c_sem_mask = batch["c_sem_mask"].to(eval_device)
        frame_delay = batch["frame_delay"].to(eval_device)
        q_current = batch["q_current"].to(eval_device, dtype=model_dtype)

        batch_size, horizon, action_dim = x_1.shape
        x_0 = torch.randn(
            batch_size,
            horizon,
            action_dim,
            generator=generator,
            device=eval_device,
            dtype=model_dtype,
        )
        t = torch.rand(batch_size, generator=generator, device=eval_device, dtype=model_dtype)
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)

        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_true = x_1 - x_0
        with _autocast_ctx(eval_device, amp_dtype):
            v_pred = _predict_vector_field(
                model,
                x_t=x_t,
                t=t,
                pixel_values=pixel_values,
                vlm_inputs=vlm_inputs,
                c_sem_mask=c_sem_mask,
                frame_delay=frame_delay,
                q_current=q_current,
            )

        mse_loss = F.mse_loss(v_pred.float(), v_true.float(), reduction="none").mean(dim=-1)
        weighted_fm_loss_sum = float((mse_loss * loss_mask).sum().item())
        weight = float(loss_mask.sum().clamp(min=1.0).item())
        dataset_slugs = _dataset_slugs_for_batch(batch, batch_size)

        total_fm_loss_sum += weighted_fm_loss_sum
        total_weight += weight
        sample_fm_loss_sum = (mse_loss * loss_mask).sum(dim=1).detach().cpu()
        sample_weight = loss_mask.sum(dim=1).detach().cpu()
        for sample_idx, slug in enumerate(dataset_slugs):
            dataset_fm_loss_sum[slug] = dataset_fm_loss_sum.get(slug, 0.0) + float(
                sample_fm_loss_sum[sample_idx].item()
            )
            dataset_weight_sum[slug] = dataset_weight_sum.get(slug, 0.0) + float(
                sample_weight[sample_idx].item()
            )

        num_batches += 1

    if num_batches == 0:
        raise ValueError("No batches were evaluated.")

    summary: dict[str, Any] = {
        "num_batches": num_batches,
        "fm_loss_mean": total_fm_loss_sum / max(total_weight, 1.0),
        "fm_loss_by_dataset": {
            slug: dataset_fm_loss_sum[slug] / max(dataset_weight_sum[slug], 1.0)
            for slug in sorted(dataset_fm_loss_sum)
        },
    }
    return summary


def _finalize_action_summary(
    all_steps_acc: dict[str, Any] | None,
    first_step_acc: dict[str, Any] | None,
    *,
    num_batches: int,
) -> dict[str, Any]:
    if all_steps_acc is None or first_step_acc is None:
        raise ValueError("No batches were evaluated.")

    all_steps = _finalize_metrics(all_steps_acc)
    first_step = _finalize_metrics(first_step_acc)
    return {
        "num_batches": int(num_batches),
        "num_valid_steps": all_steps["count"],
        "num_valid_first_steps": first_step["count"],
        "all_steps": all_steps,
        "first_step": first_step,
        "action_loss_mean": float(all_steps["mae_mean"]),
    }


def _finalize_fm_summary(
    dataset_fm_loss_sum: dict[str, float],
    dataset_weight_sum: dict[str, float],
    *,
    total_fm_loss_sum: float,
    total_weight: float,
    num_batches: int,
) -> dict[str, Any]:
    if num_batches == 0:
        raise ValueError("No batches were evaluated.")
    return {
        "num_loss_batches": int(num_batches),
        "fm_loss_mean": float(total_fm_loss_sum / max(total_weight, 1.0)),
        "fm_loss_by_dataset": {
            slug: float(dataset_fm_loss_sum[slug] / max(dataset_weight_sum[slug], 1.0))
            for slug in sorted(dataset_fm_loss_sum)
        },
    }


def evaluate_action_and_loss_on_batches(
    model: torch.nn.Module,
    batches: Iterable[TensorDict],
    predict_absolute_chunk_fn: Callable[[TensorDict], torch.Tensor],
    *,
    device: str,
    action_is_delta: bool,
    action_delta_state_indices: list[int | None] | tuple[int | None, ...] | None = None,
    max_batches: int | None = None,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    eval_device = torch.device(device)
    model_param = _policy_param(model)
    model_dtype = model_param.dtype
    noise_device = eval_device if eval_device.type != "cpu" else torch.device("cpu")
    generator = torch.Generator(device=noise_device).manual_seed(int(seed))
    predict_from_features_fn = getattr(predict_absolute_chunk_fn, "from_features", None)

    all_steps_acc = None
    first_step_acc = None
    total_fm_loss_sum = 0.0
    total_weight = 0.0
    dataset_fm_loss_sum: dict[str, float] = {}
    dataset_weight_sum: dict[str, float] = {}
    num_batches = 0

    for batch_idx, batch in enumerate(batches):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_1 = batch["action"].to(eval_device, dtype=model_dtype)
        x_1 = _normalize_action_targets(model, x_1)
        loss_mask = batch["loss_mask"].to(eval_device)
        pixel_values = batch["pixel_values"].to(eval_device)
        vlm_inputs = {
            key: value.to(eval_device) if torch.is_tensor(value) else value
            for key, value in batch["vlm_inputs"].items()
        }
        c_sem_mask = batch["c_sem_mask"].to(eval_device)
        frame_delay = batch["frame_delay"].to(eval_device)
        q_current = batch["q_current"].to(eval_device, dtype=model_dtype)

        f_vision = None
        c_sem = None
        can_reuse_features = isinstance(model, ServoVLA) and callable(predict_from_features_fn)
        if can_reuse_features:
            with _autocast_ctx(eval_device, amp_dtype):
                f_vision, c_sem = model.encode_observations(
                    pixel_values=pixel_values,
                    vlm_inputs=vlm_inputs,
                )
                f_vision = f_vision.to(device=eval_device, dtype=model_dtype)
                c_sem = c_sem.to(device=eval_device, dtype=model_dtype)

        target_abs = restore_absolute_actions(
            batch["action"],
            batch["q_current"],
            action_is_delta=action_is_delta,
            action_delta_state_indices=action_delta_state_indices,
        )
        if can_reuse_features:
            with _autocast_ctx(eval_device, amp_dtype):
                pred_abs = predict_from_features_fn(
                    batch,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
        else:
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

        batch_size, horizon, action_dim = x_1.shape
        x_0 = torch.randn(
            batch_size,
            horizon,
            action_dim,
            generator=generator,
            device=eval_device,
            dtype=model_dtype,
        )
        t = torch.rand(batch_size, generator=generator, device=eval_device, dtype=model_dtype)
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)

        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_true = x_1 - x_0
        with _autocast_ctx(eval_device, amp_dtype):
            if can_reuse_features:
                v_pred = model.forward_policy(
                    x_t=x_t,
                    t=t,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            else:
                v_pred = _predict_vector_field(
                    model,
                    x_t=x_t,
                    t=t,
                    pixel_values=pixel_values,
                    vlm_inputs=vlm_inputs,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )

        mse_loss = F.mse_loss(v_pred.float(), v_true.float(), reduction="none").mean(dim=-1)
        weighted_fm_loss_sum = float((mse_loss * loss_mask).sum().item())
        weight = float(loss_mask.sum().clamp(min=1.0).item())
        dataset_slugs = _dataset_slugs_for_batch(batch, batch_size)

        total_fm_loss_sum += weighted_fm_loss_sum
        total_weight += weight
        sample_fm_loss_sum = (mse_loss * loss_mask).sum(dim=1).detach().cpu()
        sample_weight = loss_mask.sum(dim=1).detach().cpu()
        for sample_idx, slug in enumerate(dataset_slugs):
            dataset_fm_loss_sum[slug] = dataset_fm_loss_sum.get(slug, 0.0) + float(
                sample_fm_loss_sum[sample_idx].item()
            )
            dataset_weight_sum[slug] = dataset_weight_sum.get(slug, 0.0) + float(
                sample_weight[sample_idx].item()
            )

        num_batches += 1

    summary = _finalize_action_summary(all_steps_acc, first_step_acc, num_batches=num_batches)
    summary.update(
        _finalize_fm_summary(
            dataset_fm_loss_sum,
            dataset_weight_sum,
            total_fm_loss_sum=total_fm_loss_sum,
            total_weight=total_weight,
            num_batches=num_batches,
        )
    )
    return summary


def evaluate_online_validation_batches(
    model: torch.nn.Module,
    batches: Iterable[TensorDict],
    *,
    device: str,
    action_is_delta: bool,
    action_delta_state_indices: list[int | None] | tuple[int | None, ...] | None = None,
    predict_absolute_chunk_fn: Callable[[TensorDict], torch.Tensor],
    max_batches: int | None = None,
    seed: int = 0,
    amp_dtype: torch.dtype | None = None,
    metrics: str = "both",
) -> dict[str, Any]:
    if metrics not in {"action", "fm", "both"}:
        raise ValueError(f"Unsupported eval metrics mode: {metrics!r}")

    if metrics == "action":
        summary = evaluate_model_on_batches(
            batches,
            predict_absolute_chunk_fn,
            action_is_delta=action_is_delta,
            action_delta_state_indices=action_delta_state_indices,
            max_batches=max_batches,
        )
        summary["action_loss_mean"] = float(summary["all_steps"]["mae_mean"])
        return summary

    if metrics == "fm":
        loss_summary = evaluate_loss_on_batches(
            model,
            batches,
            device=device,
            max_batches=max_batches,
            seed=seed,
            amp_dtype=amp_dtype,
        )
        return {
            "num_loss_batches": int(loss_summary["num_batches"]),
            "fm_loss_mean": float(loss_summary["fm_loss_mean"]),
            "fm_loss_by_dataset": {
                slug: float(value) for slug, value in loss_summary["fm_loss_by_dataset"].items()
            },
        }

    return evaluate_action_and_loss_on_batches(
        model,
        batches,
        predict_absolute_chunk_fn,
        device=device,
        action_is_delta=action_is_delta,
        action_delta_state_indices=action_delta_state_indices,
        max_batches=max_batches,
        seed=seed,
        amp_dtype=amp_dtype,
    )


def write_eval_summary(output_path: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
