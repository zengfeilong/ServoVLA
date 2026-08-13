#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import logging
import random
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_SOURCE_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from scripts import train as train_script
from servovla.evaluation.async_eval import AsyncEvalTask
from servovla.trainer.vlm_compile_warmup import VlmSignatureRegistry, make_vlm_input_signature


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one ServoVLA validation task in an isolated process."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _load_task(path: Path) -> AsyncEvalTask:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, AsyncEvalTask):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported async eval task payload: {type(payload)!r}")
    return AsyncEvalTask(
        step=int(payload["step"]),
        snapshot=dict(payload.get("snapshot") or {}),
        used_ema=bool(payload.get("used_ema", False)),
        eval_seed=int(payload.get("eval_seed", 0)),
        eval_zero_noise=bool(payload.get("eval_zero_noise", False)),
    )


def _run_eval_for_state(
    *,
    cfg,
    model,
    val_dataloader,
    task: AsyncEvalTask,
    state_dict: dict[str, torch.Tensor],
    used_ema: bool,
) -> dict[str, Any]:
    train_script.load_policy_head_state_dict(
        model,
        train_script._normalize_async_eval_state_dict_keys(state_dict),
    )
    model.eval()
    with torch.inference_mode():
        with train_script.torch_compile_concurrency_guard():
            summary = train_script.evaluate_online_validation_batches(
                model=model,
                batches=val_dataloader,
                device=str(cfg.training.device),
                action_is_delta=train_script.action_mode_is_delta(cfg.dataset.action_mode),
                action_delta_state_indices=train_script._resolve_action_delta_state_indices(cfg),
                predict_absolute_chunk_fn=train_script._build_predict_absolute_chunk_fn(
                    model,
                    cfg,
                    seed=int(task.eval_seed),
                    zero_noise=bool(task.eval_zero_noise),
                ),
                seed=int(task.eval_seed),
                amp_dtype=train_script._training_amp_dtype(cfg),
            )
    summary["used_ema"] = bool(used_ema)
    return summary


def _warm_eval_vlm_signatures(cfg, model, val_dataloader, *, image_preprocessor=None) -> None:
    warm_cfg = getattr(cfg.training, "vlm_compile", None)
    if warm_cfg is None or not bool(getattr(warm_cfg, "eval_warmup_enabled", False)):
        return

    max_batches = max(int(getattr(warm_cfg, "warmup_max_batches", 0)), 0)
    if max_batches <= 0:
        return

    registry = VlmSignatureRegistry(max_entries=int(getattr(warm_cfg, "signature_max_entries", 16)))
    warm_batches = list(itertools.islice(val_dataloader, max_batches))
    if not warm_batches:
        return

    for batch in warm_batches:
        signature = make_vlm_input_signature(batch)
        if registry.mark_warmed(signature) and bool(getattr(warm_cfg, "log_signatures", True)):
            logging.info("Eval VLM warmup signature %d | %s", registry.warmed_count, signature)

    with torch.inference_mode():
        with train_script.torch_compile_concurrency_guard():
            train_script.evaluate_online_validation_batches(
                model=model,
                batches=train_script._prepare_eval_batches_for_gpu_preprocess(
                    warm_batches,
                    image_preprocessor,
                ),
                device=str(cfg.training.device),
                action_is_delta=train_script.action_mode_is_delta(cfg.dataset.action_mode),
                action_delta_state_indices=train_script._resolve_action_delta_state_indices(cfg),
                predict_absolute_chunk_fn=train_script._build_predict_absolute_chunk_fn(
                    model,
                    cfg,
                    seed=int(getattr(cfg.training, "eval_seed", 0)),
                    zero_noise=bool(getattr(cfg.training, "eval_zero_noise", False)),
                ),
                seed=int(getattr(cfg.training, "eval_seed", 0)),
                amp_dtype=train_script._training_amp_dtype(cfg),
            )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _evaluate_task(cfg, task: AsyncEvalTask) -> dict[str, Any]:
    train_script.configure_hf_offline_env()
    seed = int(getattr(cfg, "seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(str(cfg.training.device))
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
    _warm_eval_vlm_signatures(cfg, model, val_dataloader, image_preprocessor=image_preprocessor)
    val_batches = train_script._prepare_eval_batches_for_gpu_preprocess(
        val_dataloader, image_preprocessor
    )

    snapshot = dict(task.snapshot or {})
    if "raw_state" in snapshot or "ema_state" in snapshot:
        summary: dict[str, Any] = {}
        raw_state = snapshot.get("raw_state")
        ema_state = snapshot.get("ema_state")
        if raw_state is not None:
            summary["raw"] = _run_eval_for_state(
                cfg=cfg,
                model=model,
                val_dataloader=val_batches,
                task=task,
                state_dict=raw_state,
                used_ema=False,
            )
        if ema_state is not None:
            summary["ema"] = _run_eval_for_state(
                cfg=cfg,
                model=model,
                val_dataloader=val_batches,
                task=task,
                state_dict=ema_state,
                used_ema=True,
            )
        return summary

    return _run_eval_for_state(
        cfg=cfg,
        model=model,
        val_dataloader=val_batches,
        task=task,
        state_dict=dict(snapshot.get("model_state", {})),
        used_ema=bool(task.used_ema),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    cfg = OmegaConf.load(args.config)
    logging.info(
        "Async eval worker starting | device=%s",
        cfg.training.device,
    )
    task = _load_task(args.task)
    summary = _evaluate_task(cfg, task)
    train_script.write_eval_summary(args.output, summary)


if __name__ == "__main__":
    main()
