#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_SOURCE_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import train as train_script
from servovla.architectures.servo_vla import ServoVLA
from servovla.evaluation.action_metrics import (
    _accumulate_metrics,
    _finalize_metrics,
    _init_accumulator,
    restore_absolute_actions,
)
from servovla.evaluation.online_eval import write_eval_summary

log = logging.getLogger("evaluate_checkpoints")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate all checkpoints from a ServoVLA run directory."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-device", default=None)
    parser.add_argument("--eval-batch-size", default=64, type=int)
    parser.add_argument("--eval-num-workers", default=4, type=int)
    parser.add_argument("--decode-num-workers", default=4, type=int)
    parser.add_argument("--checkpoint-glob", default="checkpoint_step*.pt")
    parser.add_argument(
        "--no-cache-encoded-val",
        action="store_true",
        help="Disable one-time validation encoder feature caching.",
    )
    parser.add_argument(
        "--encoded-val-cache-dir",
        default=None,
        type=Path,
        help="Directory for disk-backed encoded validation feature cache. Defaults to a temporary eval subdirectory.",
    )
    parser.add_argument(
        "--encoded-val-cache-chunk-batches",
        default=4,
        type=int,
        help="Number of encoded validation batches to load at once when sweeping checkpoints.",
    )
    parser.add_argument("--recompute-existing", action="store_true")
    parser.add_argument(
        "--weights",
        choices=("raw", "ema", "both"),
        default="both",
        help="Checkpoint weight variants to evaluate.",
    )
    parser.add_argument(
        "--metrics",
        choices=("fm", "action", "both"),
        default="both",
        help="Validation metrics to compute. Partial modes avoid unnecessary validation passes.",
    )
    return parser.parse_args(argv)


def _checkpoint_step(path: Path, checkpoint: dict[str, Any]) -> int:
    if checkpoint.get("step") is not None:
        return int(checkpoint["step"])
    match = re.search(r"checkpoint_step(\d+)\.pt$", path.name)
    if not match:
        raise ValueError(f"Cannot parse checkpoint step from {path.name}")
    return int(match.group(1))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint_step(\d+)\.pt$", path.name)
    step = int(match.group(1)) if match else -1
    return step, str(path)


def _collect_checkpoints(run_dir: Path, *, checkpoint_glob: str) -> list[Path]:
    by_step: dict[int, Path] = {}
    fallback: list[Path] = []
    for checkpoint_path in sorted(run_dir.resolve().glob(checkpoint_glob), key=_checkpoint_sort_key):
        match = re.search(r"checkpoint_step(\d+)\.pt$", checkpoint_path.name)
        if match:
            by_step[int(match.group(1))] = checkpoint_path.resolve()
        else:
            fallback.append(checkpoint_path.resolve())
    checkpoints = [by_step[step] for step in sorted(by_step)]
    checkpoints.extend(sorted(fallback))
    return checkpoints


def _eval_artifact_paths(
    run_dir: Path, *, step: int, weights: str, metrics: str
) -> tuple[Path, Path]:
    suffix_parts = []
    if weights != "both":
        suffix_parts.append(weights)
    if metrics != "both":
        suffix_parts.append(metrics)
    suffix = f".{'.'.join(suffix_parts)}" if suffix_parts else ""
    stem = f"step{step:07d}{suffix}"
    return run_dir / "eval" / f"{stem}.json", run_dir / "eval" / f"{stem}.error.json"


def _metric_for_log(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return "skipped"
    return f"{float(value):.6f}"


def _configure_eval_cfg(args: argparse.Namespace):
    run_dir = args.run_dir
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    checkpoints = _collect_checkpoints(
        run_dir,
        checkpoint_glob=str(args.checkpoint_glob),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r} for {run_dir}")
    final_checkpoint = checkpoints[-1]
    with open_dict(cfg):
        cfg.output_dir = str(run_dir)
    with open_dict(cfg.training):
        cfg.training.device = str(args.device)
        cfg.training.base_checkpoint = str(final_checkpoint)
        cfg.training.base_checkpoint_use_ema = False
        cfg.training.eval_batch_size = int(args.eval_batch_size)
        cfg.training.eval_num_workers = int(args.eval_num_workers)
        cfg.training.eval_prefetch_factor = int(cfg.training.get("eval_prefetch_factor", 1))
        if "decode" not in cfg.training:
            cfg.training.decode = {}
        with open_dict(cfg.training.decode):
            cfg.training.decode.device = str(args.decode_device or args.device)
            cfg.training.decode.num_workers = int(args.decode_num_workers)
    return cfg


def _tensor_bytes(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int(value.numel() * value.element_size())


def _to_cpu_tensor(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return list(value)
    return value


def _autocast_ctx(device: torch.device, dtype: torch.dtype | None):
    if dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


def _normalize_action_targets(model: torch.nn.Module, action: torch.Tensor) -> torch.Tensor:
    action_normalizer = getattr(model, "action_normalizer", None)
    if action_normalizer is None or not bool(getattr(action_normalizer, "enabled", False)):
        return action
    return action_normalizer.normalize(action)


class _EncodedValidationCache:
    def __init__(self, paths: list[Path], *, cache_dir: Path | None, cleanup: bool) -> None:
        self.paths = list(paths)
        self.cache_dir = cache_dir
        self.cleanup_enabled = bool(cleanup)

    def __iter__(self):
        for path in self.paths:
            yield torch.load(path, map_location="cpu", weights_only=False)

    def __len__(self) -> int:
        return len(self.paths)

    def cleanup(self) -> None:
        if self.cleanup_enabled and self.cache_dir is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)


class _EncodedEvalAccumulator:
    def __init__(
        self,
        *,
        model: ServoVLA,
        device: str,
        seed: int,
        metrics: str,
        action_delta_state_indices: tuple[int | None, ...] | list[int | None] | None = None,
    ) -> None:
        if metrics not in {"action", "fm", "both"}:
            raise ValueError(f"Unsupported eval metrics mode: {metrics!r}")
        self.metrics = str(metrics)
        self.eval_device = torch.device(device)
        self.model_dtype = next(model.policy_head.parameters()).dtype
        noise_device = self.eval_device if self.eval_device.type != "cpu" else torch.device("cpu")
        self.generator = torch.Generator(device=noise_device).manual_seed(int(seed))
        self.all_steps_acc = None
        self.first_step_acc = None
        self.total_fm_loss_sum = 0.0
        self.total_weight = 0.0
        self.dataset_fm_loss_sum: dict[str, float] = {}
        self.dataset_weight_sum: dict[str, float] = {}
        self.num_action_batches = 0
        self.num_loss_batches = 0
        self.action_delta_state_indices = action_delta_state_indices

    def accumulate(
        self,
        *,
        model: ServoVLA,
        batch: dict[str, Any],
        predict_from_features_fn,
        action_is_delta: bool,
        amp_dtype: torch.dtype | None,
    ) -> None:
        eval_device = self.eval_device
        model_dtype = self.model_dtype
        action_cpu = batch["action"]
        q_current_cpu = batch["q_current"]
        loss_mask_cpu = batch["loss_mask"]
        batch_size = int(action_cpu.shape[0])

        f_vision = batch["f_vision"].to(eval_device, dtype=model_dtype)
        c_sem = batch["c_sem"].to(eval_device, dtype=model_dtype)
        c_sem_mask = batch["c_sem_mask"].to(eval_device).bool()
        frame_delay = batch["frame_delay"].to(eval_device, dtype=torch.float32)
        q_current = q_current_cpu.to(eval_device, dtype=model_dtype)

        if self.metrics in {"action", "both"}:
            target_abs = restore_absolute_actions(
                action_cpu,
                q_current_cpu,
                action_is_delta=action_is_delta,
                action_delta_state_indices=self.action_delta_state_indices,
            )
            with _autocast_ctx(eval_device, amp_dtype):
                pred_abs = predict_from_features_fn(
                    batch,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            if pred_abs.ndim == 2:
                pred_abs = pred_abs.unsqueeze(1)
            pred_abs = pred_abs.detach().cpu().float()[..., : target_abs.shape[-1]]

            if self.all_steps_acc is None:
                action_dim = int(target_abs.shape[-1])
                self.all_steps_acc = _init_accumulator(action_dim)
                self.first_step_acc = _init_accumulator(action_dim)
            _accumulate_metrics(self.all_steps_acc, pred_abs, target_abs, loss_mask_cpu)
            _accumulate_metrics(
                self.first_step_acc, pred_abs[:, :1], target_abs[:, :1], loss_mask_cpu[:, :1]
            )
            self.num_action_batches += 1

        if self.metrics in {"fm", "both"}:
            x_1 = action_cpu.to(eval_device, dtype=model_dtype)
            x_1 = _normalize_action_targets(model, x_1)
            loss_mask = loss_mask_cpu.to(eval_device)
            batch_size, horizon, action_dim = x_1.shape
            x_0 = torch.randn(
                batch_size,
                horizon,
                action_dim,
                generator=self.generator,
                device=eval_device,
                dtype=model_dtype,
            )
            t = torch.rand(
                batch_size, generator=self.generator, device=eval_device, dtype=model_dtype
            )
            t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            v_true = x_1 - x_0
            with _autocast_ctx(eval_device, amp_dtype):
                v_pred = model.forward_policy(
                    x_t=x_t,
                    t=t,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            mse_loss = F.mse_loss(v_pred.float(), v_true.float(), reduction="none").mean(dim=-1)
            weighted_fm_loss_sum = float((mse_loss * loss_mask).sum().item())
            weight = float(loss_mask.sum().clamp(min=1.0).item())
            dataset_slugs = _dataset_slugs_for_batch(batch, batch_size)

            self.total_fm_loss_sum += weighted_fm_loss_sum
            self.total_weight += weight
            sample_fm_loss_sum = (mse_loss * loss_mask).sum(dim=1).detach().cpu()
            sample_weight = loss_mask.sum(dim=1).detach().cpu()
            for sample_idx, slug in enumerate(dataset_slugs):
                self.dataset_fm_loss_sum[slug] = self.dataset_fm_loss_sum.get(slug, 0.0) + float(
                    sample_fm_loss_sum[sample_idx].item()
                )
                self.dataset_weight_sum[slug] = self.dataset_weight_sum.get(slug, 0.0) + float(
                    sample_weight[sample_idx].item()
                )
            self.num_loss_batches += 1

    def finalize(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}
        if self.metrics in {"action", "both"}:
            summary.update(
                _finalize_action_summary(
                    self.all_steps_acc,
                    self.first_step_acc,
                    num_batches=self.num_action_batches,
                )
            )
        if self.metrics in {"fm", "both"}:
            if self.num_loss_batches == 0:
                raise ValueError("No batches were evaluated.")
            summary.update(
                {
                    "num_loss_batches": int(self.num_loss_batches),
                    "fm_loss_mean": float(self.total_fm_loss_sum / max(self.total_weight, 1.0)),
                    "fm_loss_by_dataset": {
                        slug: float(
                            self.dataset_fm_loss_sum[slug] / max(self.dataset_weight_sum[slug], 1.0)
                        )
                        for slug in sorted(self.dataset_fm_loss_sum)
                    },
                }
            )
        return summary


@dataclass
class _EncodedEvalTarget:
    checkpoint_index: int
    checkpoint_count: int
    checkpoint_path: Path
    step: int
    name: str
    used_ema: bool
    state_dict: dict[str, torch.Tensor]
    predict_from_features_fn: Any
    accumulator: _EncodedEvalAccumulator


def _dataset_slugs_for_batch(batch: dict[str, Any], batch_size: int) -> list[str]:
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


def _cache_encoded_validation_batches(
    *,
    model: ServoVLA,
    batches,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    cache_dir: Path,
    cleanup_cache_dir: bool,
) -> _EncodedValidationCache:
    if not isinstance(model, ServoVLA):
        raise TypeError("Encoded validation caching requires ServoVLA.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    existing_paths = sorted(cache_dir.glob("batch_*.pt"))
    if existing_paths:
        total_bytes = sum(path.stat().st_size for path in existing_paths)
        log.info(
            "reusing encoded val disk cache | batches=%d | cached=%.2f GiB | dir=%s",
            len(existing_paths),
            total_bytes / (1024**3),
            cache_dir,
        )
        return _EncodedValidationCache(
            existing_paths, cache_dir=cache_dir, cleanup=cleanup_cache_dir
        )

    paths: list[Path] = []
    total_bytes = 0
    start = time.monotonic()
    model.eval()
    with torch.inference_mode():
        with train_script.torch_compile_concurrency_guard():
            for batch_idx, batch in enumerate(batches, start=1):
                pixel_values = batch["pixel_values"].to(device)
                vlm_inputs = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch["vlm_inputs"].items()
                }
                with _autocast_ctx(device, amp_dtype):
                    f_vision, c_sem = model.encode_observations(
                        pixel_values=pixel_values,
                        vlm_inputs=vlm_inputs,
                    )

                encoded = {
                    "f_vision": f_vision.detach().cpu().contiguous(),
                    "c_sem": c_sem.detach().cpu().contiguous(),
                    "c_sem_mask": _to_cpu_tensor(batch["c_sem_mask"]),
                    "action": _to_cpu_tensor(batch["action"]),
                    "loss_mask": _to_cpu_tensor(batch["loss_mask"]),
                    "frame_delay": _to_cpu_tensor(batch["frame_delay"]),
                    "q_current": _to_cpu_tensor(batch["q_current"]),
                    "dataset_slug": _to_cpu_tensor(batch.get("dataset_slug", "unknown")),
                }
                path = cache_dir / f"batch_{batch_idx:06d}.pt"
                torch.save(encoded, path)
                batch_bytes = path.stat().st_size
                total_bytes += batch_bytes
                paths.append(path)
                if batch_idx == 1 or batch_idx % 25 == 0:
                    log.info(
                        "encoded val disk cache progress | batches=%d | cached=%.2f GiB | dir=%s | elapsed=%.1fs",
                        batch_idx,
                        total_bytes / (1024**3),
                        cache_dir,
                        time.monotonic() - start,
                    )
                del batch, pixel_values, vlm_inputs, f_vision, c_sem, encoded
                if batch_idx % 25 == 0:
                    gc.collect()

    if not paths:
        raise ValueError("No validation batches were cached.")
    log.info(
        "encoded val disk cache ready | batches=%d | cached=%.2f GiB | dir=%s | elapsed=%.1fs",
        len(paths),
        total_bytes / (1024**3),
        cache_dir,
        time.monotonic() - start,
    )
    return _EncodedValidationCache(paths, cache_dir=cache_dir, cleanup=cleanup_cache_dir)


def _evaluate_encoded_validation_batches(
    *,
    model: ServoVLA,
    encoded_batches,
    predict_absolute_chunk_fn,
    device: str,
    action_is_delta: bool,
    action_delta_state_indices: tuple[int | None, ...] | list[int | None] | None,
    seed: int,
    amp_dtype: torch.dtype | None,
    metrics: str,
) -> dict[str, Any]:
    if metrics not in {"action", "fm", "both"}:
        raise ValueError(f"Unsupported eval metrics mode: {metrics!r}")

    eval_device = torch.device(device)
    model_param = next(model.policy_head.parameters())
    model_dtype = model_param.dtype
    noise_device = eval_device if eval_device.type != "cpu" else torch.device("cpu")
    generator = torch.Generator(device=noise_device).manual_seed(int(seed))
    predict_from_features_fn = getattr(predict_absolute_chunk_fn, "from_features", None)
    if metrics in {"action", "both"} and not callable(predict_from_features_fn):
        raise TypeError(
            "Encoded validation action eval requires predict_absolute_chunk_fn.from_features."
        )

    all_steps_acc = None
    first_step_acc = None
    total_fm_loss_sum = 0.0
    total_weight = 0.0
    dataset_fm_loss_sum: dict[str, float] = {}
    dataset_weight_sum: dict[str, float] = {}
    num_action_batches = 0
    num_loss_batches = 0

    for batch in encoded_batches:
        action_cpu = batch["action"]
        q_current_cpu = batch["q_current"]
        loss_mask_cpu = batch["loss_mask"]
        batch_size = int(action_cpu.shape[0])

        f_vision = batch["f_vision"].to(eval_device, dtype=model_dtype)
        c_sem = batch["c_sem"].to(eval_device, dtype=model_dtype)
        c_sem_mask = batch["c_sem_mask"].to(eval_device).bool()
        frame_delay = batch["frame_delay"].to(eval_device, dtype=torch.float32)
        q_current = q_current_cpu.to(eval_device, dtype=model_dtype)

        if metrics in {"action", "both"}:
            target_abs = restore_absolute_actions(
                action_cpu,
                q_current_cpu,
                action_is_delta=action_is_delta,
                action_delta_state_indices=action_delta_state_indices,
            )
            with _autocast_ctx(eval_device, amp_dtype):
                pred_abs = predict_from_features_fn(
                    batch,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            if pred_abs.ndim == 2:
                pred_abs = pred_abs.unsqueeze(1)
            pred_abs = pred_abs.detach().cpu().float()[..., : target_abs.shape[-1]]

            if all_steps_acc is None:
                action_dim = int(target_abs.shape[-1])
                all_steps_acc = _init_accumulator(action_dim)
                first_step_acc = _init_accumulator(action_dim)
            _accumulate_metrics(all_steps_acc, pred_abs, target_abs, loss_mask_cpu)
            _accumulate_metrics(
                first_step_acc, pred_abs[:, :1], target_abs[:, :1], loss_mask_cpu[:, :1]
            )
            num_action_batches += 1

        if metrics in {"fm", "both"}:
            x_1 = action_cpu.to(eval_device, dtype=model_dtype)
            x_1 = _normalize_action_targets(model, x_1)
            loss_mask = loss_mask_cpu.to(eval_device)
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
                v_pred = model.forward_policy(
                    x_t=x_t,
                    t=t,
                    f_vision=f_vision,
                    c_sem=c_sem,
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
            num_loss_batches += 1

    summary: dict[str, Any] = {}
    if metrics in {"action", "both"}:
        summary.update(
            _finalize_action_summary(
                all_steps_acc,
                first_step_acc,
                num_batches=num_action_batches,
            )
        )
    if metrics in {"fm", "both"}:
        if num_loss_batches == 0:
            raise ValueError("No batches were evaluated.")
        summary.update(
            {
                "num_loss_batches": int(num_loss_batches),
                "fm_loss_mean": float(total_fm_loss_sum / max(total_weight, 1.0)),
                "fm_loss_by_dataset": {
                    slug: float(dataset_fm_loss_sum[slug] / max(dataset_weight_sum[slug], 1.0))
                    for slug in sorted(dataset_fm_loss_sum)
                },
            }
        )
    return summary


def _iter_encoded_cache_chunks(cache: _EncodedValidationCache, chunk_batches: int):
    chunk_batches = max(int(chunk_batches), 1)
    paths = list(cache.paths)
    total_chunks = (len(paths) + chunk_batches - 1) // chunk_batches
    for chunk_idx, start in enumerate(range(0, len(paths), chunk_batches), start=1):
        selected_paths = paths[start : start + chunk_batches]
        loaded = [
            torch.load(path, map_location="cpu", weights_only=False) for path in selected_paths
        ]
        yield chunk_idx, total_chunks, start + 1, start + len(selected_paths), loaded


def _prepare_encoded_eval_targets(
    *,
    checkpoints: list[Path],
    model: ServoVLA,
    cfg,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[_EncodedEvalTarget], int, int]:
    records: list[dict[str, Any]] = []
    targets: list[_EncodedEvalTarget] = []
    reused = 0
    failed = 0
    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            failed += 1
            log.error(
                "[%02d/%02d] skipping non-dict checkpoint: %s",
                idx,
                len(checkpoints),
                checkpoint_path,
            )
            continue

        step = _checkpoint_step(checkpoint_path, checkpoint)
        output_path, error_path = _eval_artifact_paths(
            checkpoint_path.parent,
            step=step,
            weights=str(args.weights),
            metrics=str(args.metrics),
        )
        record = {
            "idx": idx,
            "step": step,
            "checkpoint_path": checkpoint_path,
            "output_path": output_path,
            "error_path": error_path,
            "summary": None,
            "targets": {},
        }
        if output_path.exists() and output_path.stat().st_size > 0 and not args.recompute_existing:
            record["summary"] = _load_json(output_path)
            reused += 1
            log.info(
                "[%02d/%02d] step=%d reusing existing eval JSON: %s",
                idx,
                len(checkpoints),
                step,
                output_path,
            )
            records.append(record)
            del checkpoint
            continue

        for name, state_key, used_ema in (
            ("raw", "model_state", False),
            ("ema", "ema_state", True),
        ):
            if args.weights not in {name, "both"} or state_key not in checkpoint:
                continue
            accumulator = _EncodedEvalAccumulator(
                model=model,
                device=str(cfg.training.device),
                seed=int(cfg.training.eval_seed),
                metrics=str(args.metrics),
                action_delta_state_indices=train_script._resolve_action_delta_state_indices(cfg),
            )
            predict_fn = train_script._build_predict_absolute_chunk_fn(
                model,
                cfg,
                seed=int(cfg.training.eval_seed),
                zero_noise=bool(cfg.training.eval_zero_noise),
            )
            target = _EncodedEvalTarget(
                checkpoint_index=idx,
                checkpoint_count=len(checkpoints),
                checkpoint_path=checkpoint_path,
                step=step,
                name=name,
                used_ema=used_ema,
                state_dict=train_script._normalize_async_eval_state_dict_keys(
                    checkpoint[state_key]
                ),
                predict_from_features_fn=getattr(predict_fn, "from_features"),
                accumulator=accumulator,
            )
            record["targets"][name] = target
            targets.append(target)

        if not record["targets"]:
            failed += 1
            error_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "step": step,
                        "error": f"no requested {args.weights!r} state",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.error(
                "[%02d/%02d] step=%d has no requested %s state",
                idx,
                len(checkpoints),
                step,
                args.weights,
            )
        records.append(record)
        del checkpoint
    gc.collect()
    return records, targets, reused, failed


def _run_encoded_cache_checkpoint_sweep(
    *,
    checkpoints: list[Path],
    model: ServoVLA,
    cfg,
    args: argparse.Namespace,
    encoded_val_batches: _EncodedValidationCache,
) -> tuple[int, int, int]:
    records, targets, reused, failed = _prepare_encoded_eval_targets(
        checkpoints=checkpoints,
        model=model,
        cfg=cfg,
        args=args,
    )
    if not targets:
        return 0, reused, failed

    action_is_delta = train_script.action_mode_is_delta(cfg.dataset.action_mode)
    amp_dtype = train_script._training_amp_dtype(cfg)
    chunk_batches = max(int(args.encoded_val_cache_chunk_batches), 1)
    log.info(
        "encoded checkpoint sweep start | targets=%d | checkpoints=%d | cache_batches=%d | chunk_batches=%d",
        len(targets),
        len(checkpoints),
        len(encoded_val_batches),
        chunk_batches,
    )
    sweep_start = time.monotonic()
    model.eval()
    with torch.inference_mode():
        with train_script.torch_compile_concurrency_guard():
            for (
                chunk_idx,
                total_chunks,
                first_batch,
                last_batch,
                chunk,
            ) in _iter_encoded_cache_chunks(
                encoded_val_batches,
                chunk_batches,
            ):
                chunk_start = time.monotonic()
                for target_idx, target in enumerate(targets, start=1):
                    train_script.load_policy_head_state_dict(model, target.state_dict)
                    for batch in chunk:
                        target.accumulator.accumulate(
                            model=model,
                            batch=batch,
                            predict_from_features_fn=target.predict_from_features_fn,
                            action_is_delta=action_is_delta,
                            amp_dtype=amp_dtype,
                        )
                    if target_idx == 1 or target_idx == len(targets) or target_idx % 25 == 0:
                        log.info(
                            "encoded checkpoint sweep chunk %d/%d batches=%d-%d target=%d/%d step=%d %s",
                            chunk_idx,
                            total_chunks,
                            first_batch,
                            last_batch,
                            target_idx,
                            len(targets),
                            target.step,
                            target.name,
                        )
                del chunk
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log.info(
                    "encoded checkpoint sweep chunk %d/%d done | batches=%d-%d | elapsed=%.1fs | total_elapsed=%.1fs",
                    chunk_idx,
                    total_chunks,
                    first_batch,
                    last_batch,
                    time.monotonic() - chunk_start,
                    time.monotonic() - sweep_start,
                )

    completed = 0
    target_lookup = {(target.step, target.name): target for target in targets}
    for record in records:
        if record.get("summary") is not None:
            continue
        step = int(record["step"])
        output_path = record["output_path"]
        error_path = record["error_path"]
        summary: dict[str, Any] = {}
        try:
            for name, target in record["targets"].items():
                finalized = target_lookup[(step, name)].accumulator.finalize()
                finalized["used_ema"] = bool(target.used_ema)
                summary[name] = finalized
                log.info(
                    "[%02d/%02d] step=%d %s eval done | fm=%s action=%s",
                    record["idx"],
                    len(checkpoints),
                    step,
                    name,
                    _metric_for_log(finalized, "fm_loss_mean"),
                    _metric_for_log(finalized, "action_loss_mean"),
                )
            if not summary:
                raise KeyError(
                    f"{record['checkpoint_path']} has no requested {args.weights!r} state"
                )
            write_eval_summary(output_path, summary)
            error_path.unlink(missing_ok=True)
            completed += 1
            log.info(
                "[%02d/%02d] step=%d eval recorded | output=%s",
                record["idx"],
                len(checkpoints),
                step,
                output_path,
            )
        except Exception:
            failed += 1
            error_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(record["checkpoint_path"]),
                        "step": step,
                        "error": "see evaluate_checkpoints log",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.exception("[%02d/%02d] step=%d eval failed", record["idx"], len(checkpoints), step)

    return completed, reused, failed


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoints = _collect_checkpoints(
        run_dir,
        checkpoint_glob=str(args.checkpoint_glob),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matched {args.checkpoint_glob!r} for {run_dir}")

    cfg = _configure_eval_cfg(args)
    train_script.configure_hf_offline_env()
    seed = int(getattr(cfg, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(str(cfg.training.device)))
        torch.cuda.manual_seed(seed)

    device = torch.device(str(cfg.training.device))
    log.info(
        "Starting checkpoint eval | run_dir=%s | checkpoints=%d | device=%s | batch=%d | eval_workers=%d | decode_workers=%d | weights=%s | metrics=%s",
        run_dir,
        len(checkpoints),
        device,
        int(cfg.training.eval_batch_size),
        int(cfg.training.eval_num_workers),
        int(cfg.training.decode.num_workers),
        args.weights,
        args.metrics,
    )

    vlm_processor, vision_processor = train_script._load_processors(cfg)
    model = train_script._build_servovla_model(cfg, device=device)
    if bool(getattr(cfg.training, "compile", False)):
        model = train_script.TrainerLoop.compile_model(model, train_cfg=cfg.training)

    image_preprocessor = train_script._build_raw_image_gpu_preprocessor(
        cfg,
        vision_processor=vision_processor,
        vlm_processor=vlm_processor,
        device=device,
    )
    val_dataloader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=train_script.get_dataset_names(cfg.dataset, "val"),
        split=str(cfg.dataset.val_split),
        vlm_processor=vlm_processor,
        vision_processor=vision_processor,
        batch_size=int(cfg.training.eval_batch_size),
        num_workers=int(cfg.training.eval_num_workers),
        repeat=False,
        shuffle_episodes=False,
    )
    encoded_val_batches = None
    if not bool(args.no_cache_encoded_val) and isinstance(model, ServoVLA):
        if args.encoded_val_cache_dir is None:
            cache_dir = run_dir / "eval" / f"encoded_val_cache_{os.getpid()}"
            cleanup_cache_dir = True
        else:
            cache_dir = args.encoded_val_cache_dir.expanduser().resolve()
            cleanup_cache_dir = False
        encoded_val_batches = _cache_encoded_validation_batches(
            model=model,
            batches=train_script._prepare_eval_batches_for_gpu_preprocess(
                val_dataloader,
                image_preprocessor,
            ),
            device=device,
            amp_dtype=train_script._training_amp_dtype(cfg),
            cache_dir=cache_dir,
            cleanup_cache_dir=cleanup_cache_dir,
        )
    start_all = time.monotonic()
    completed = 0
    reused = 0
    failed = 0

    if encoded_val_batches is not None:
        completed, reused, failed = _run_encoded_cache_checkpoint_sweep(
            checkpoints=checkpoints,
            model=model,
            cfg=cfg,
            args=args,
            encoded_val_batches=encoded_val_batches,
        )
        encoded_val_batches.cleanup()
        log.info(
            "Checkpoint eval finished | completed=%d reused=%d failed=%d elapsed=%.1fs",
            completed,
            reused,
            failed,
            time.monotonic() - start_all,
        )
        return

    def run_state(state_dict: dict[str, torch.Tensor], *, used_ema: bool) -> dict[str, Any]:
        train_script.load_policy_head_state_dict(
            model,
            train_script._normalize_async_eval_state_dict_keys(state_dict),
        )
        model.eval()
        predict_absolute_chunk_fn = train_script._build_predict_absolute_chunk_fn(
            model,
            cfg,
            seed=int(cfg.training.eval_seed),
            zero_noise=bool(cfg.training.eval_zero_noise),
        )
        with torch.inference_mode():
            with train_script.torch_compile_concurrency_guard():
                if encoded_val_batches is not None:
                    summary = _evaluate_encoded_validation_batches(
                        model=model,
                        encoded_batches=encoded_val_batches,
                        predict_absolute_chunk_fn=predict_absolute_chunk_fn,
                        device=str(cfg.training.device),
                        action_is_delta=train_script.action_mode_is_delta(cfg.dataset.action_mode),
                        action_delta_state_indices=train_script._resolve_action_delta_state_indices(
                            cfg
                        ),
                        seed=int(cfg.training.eval_seed),
                        amp_dtype=train_script._training_amp_dtype(cfg),
                        metrics=str(args.metrics),
                    )
                else:
                    summary = train_script.evaluate_online_validation_batches(
                        model=model,
                        batches=train_script._prepare_eval_batches_for_gpu_preprocess(
                            val_dataloader,
                            image_preprocessor,
                        ),
                        device=str(cfg.training.device),
                        action_is_delta=train_script.action_mode_is_delta(cfg.dataset.action_mode),
                        action_delta_state_indices=train_script._resolve_action_delta_state_indices(
                            cfg
                        ),
                        predict_absolute_chunk_fn=predict_absolute_chunk_fn,
                        seed=int(cfg.training.eval_seed),
                        amp_dtype=train_script._training_amp_dtype(cfg),
                        metrics=str(args.metrics),
                    )
        summary["used_ema"] = bool(used_ema)
        return summary

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            failed += 1
            log.error(
                "[%02d/%02d] skipping non-dict checkpoint: %s",
                idx,
                len(checkpoints),
                checkpoint_path,
            )
            continue

        step = _checkpoint_step(checkpoint_path, checkpoint)
        output_path, error_path = _eval_artifact_paths(
            checkpoint_path.parent,
            step=step,
            weights=str(args.weights),
            metrics=str(args.metrics),
        )
        step_start = time.monotonic()
        try:
            if (
                output_path.exists()
                and output_path.stat().st_size > 0
                and not args.recompute_existing
            ):
                summary = _load_json(output_path)
                reused += 1
                log.info(
                    "[%02d/%02d] step=%d reusing existing eval JSON: %s",
                    idx,
                    len(checkpoints),
                    step,
                    output_path,
                )
            else:
                summary = {}
                if args.weights in {"raw", "both"} and "model_state" in checkpoint:
                    log.info(
                        "[%02d/%02d] step=%d raw eval start | checkpoint=%s",
                        idx,
                        len(checkpoints),
                        step,
                        checkpoint_path,
                    )
                    summary["raw"] = run_state(checkpoint["model_state"], used_ema=False)
                    log.info(
                        "[%02d/%02d] step=%d raw eval done | fm=%s action=%s",
                        idx,
                        len(checkpoints),
                        step,
                        _metric_for_log(summary["raw"], "fm_loss_mean"),
                        _metric_for_log(summary["raw"], "action_loss_mean"),
                    )
                if args.weights in {"ema", "both"} and "ema_state" in checkpoint:
                    log.info("[%02d/%02d] step=%d ema eval start", idx, len(checkpoints), step)
                    summary["ema"] = run_state(checkpoint["ema_state"], used_ema=True)
                    log.info(
                        "[%02d/%02d] step=%d ema eval done | fm=%s action=%s",
                        idx,
                        len(checkpoints),
                        step,
                        _metric_for_log(summary["ema"], "fm_loss_mean"),
                        _metric_for_log(summary["ema"], "action_loss_mean"),
                    )
                if not summary:
                    raise KeyError(f"{checkpoint_path} has no requested {args.weights!r} state")
                write_eval_summary(output_path, summary)
                error_path.unlink(missing_ok=True)
                completed += 1

            log.info(
                "[%02d/%02d] step=%d eval recorded | elapsed=%.1fs | output=%s",
                idx,
                len(checkpoints),
                step,
                time.monotonic() - step_start,
                output_path,
            )
        except Exception:
            failed += 1
            error_path.write_text(
                json.dumps(
                    {
                        "checkpoint": str(checkpoint_path),
                        "step": step,
                        "error": "see evaluate_checkpoints log",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            log.exception("[%02d/%02d] step=%d eval failed", idx, len(checkpoints), step)
        finally:
            del checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if encoded_val_batches is not None:
        encoded_val_batches.cleanup()
    log.info(
        "Checkpoint eval finished | completed=%d reused=%d failed=%d elapsed=%.1fs",
        completed,
        reused,
        failed,
        time.monotonic() - start_all,
    )


if __name__ == "__main__":
    main()
