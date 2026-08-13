from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from servovla.architectures.servo_vla import ServoVLA
from servovla.evaluation.async_eval import AsyncEvalTask, build_eval_log_payload
from servovla.evaluation.online_eval import write_eval_summary
from servovla.trainer.compile_guard import (
    apply_torch_compile_nested_fx_trace_fallback,
    apply_transformers_flash_attention_compile_graph_break,
    apply_transformers_qwen_visual_position_compile_graph_break,
    torch_compile_concurrency_guard,
)
from servovla.trainer.gpu_pipeline import (
    AsyncGpuFeatureProducer,
    EncodedFeatureBatch,
    SingleGpuFeatureProducer,
)
from servovla.trainer.policy_head_ema import (
    PolicyHeadEma,
    policy_head_state_dict,
)
from servovla.trainer.rollout_action_loss import (
    compute_rollout_action_loss,
    rollout_action_loss_weight,
)
from servovla.trainer.vlm_compile_warmup import VlmSignatureRegistry

logger = logging.getLogger(__name__)

_ASYNC_PRODUCER_COMPILE_GUARD_STEPS = 2


def summarize_batch_memory(batch: Mapping[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}

    def visit(prefix: str, value: Any) -> int:
        if isinstance(value, torch.Tensor):
            nbytes = int(value.numel() * value.element_size())
            result[f"tensor_bytes.{prefix}"] = nbytes
            return nbytes
        if isinstance(value, Mapping):
            total = 0
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                total += visit(child_prefix, child)
            return total
        return 0

    result["tensor_bytes.total"] = visit("", batch)

    def raw_group_bytes(value: Any) -> int:
        if not isinstance(value, Mapping):
            return 0
        total = 0
        for group in value.get("groups", []):
            if isinstance(group, Mapping) and torch.is_tensor(group.get("images")):
                images = group["images"]
                total += int(images.numel() * images.element_size())
        return total

    vision_raw = raw_group_bytes(batch.get("vision_images_uint8"))
    vlm_raw = raw_group_bytes(batch.get("vlm_images_uint8"))
    if vision_raw or vlm_raw:
        result["raw_image_bytes.vision"] = vision_raw
        result["raw_image_bytes.vlm"] = vlm_raw
        result["raw_image_bytes.total"] = vision_raw + vlm_raw

    raw_decode_stats = batch.get("raw_decode_stats")
    if isinstance(raw_decode_stats, Mapping):
        for key, value in raw_decode_stats.items():
            if (
                isinstance(key, str)
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ):
                result[f"raw_decode_stats.{key}"] = (
                    float(value) if isinstance(value, float) else int(value)
                )
    return result


def collect_process_memory_snapshot(worker_pids: Sequence[int] = ()) -> dict[str, int | None]:
    worker_pids = tuple(worker_pids)
    try:
        import psutil
    except ImportError:
        return {
            "rss_bytes.main": None,
            "rss_bytes.workers_total": 0 if not worker_pids else None,
            "rss_bytes.worker_count": len(worker_pids),
        }

    worker_rss = []
    for pid in worker_pids:
        try:
            worker_rss.append(int(psutil.Process(int(pid)).memory_info().rss))
        except psutil.Error:
            continue
    return {
        "rss_bytes.main": int(psutil.Process().memory_info().rss),
        "rss_bytes.workers_total": int(sum(worker_rss)),
        "rss_bytes.worker_count": len(worker_rss),
    }


def dataloader_worker_pids(data_iter: Any) -> tuple[int, ...]:
    workers = getattr(data_iter, "_workers", None)
    if not workers:
        return ()
    pids = []
    for worker in workers:
        pid = getattr(worker, "pid", None)
        if pid is not None:
            pids.append(int(pid))
    return tuple(pids)


class TrainerLoop:
    def __init__(self, model: nn.Module, train_cfg, image_preprocessor: Any | None = None) -> None:
        self.cfg = train_cfg
        self.image_preprocessor = image_preprocessor

        compile_enabled = bool(getattr(train_cfg, "compile", True))
        if compile_enabled:
            self.model = self.compile_model(model, train_cfg=train_cfg)
        else:
            self.model = model

        device_str = str(train_cfg.device) if hasattr(train_cfg, "device") else "cpu"
        self.device = torch.device(device_str)

        amp_dtype_str = str(getattr(train_cfg, "amp_dtype", "float32"))
        if self.device.type == "cuda" and amp_dtype_str == "bfloat16":
            self.amp_dtype = torch.bfloat16
            self.use_amp = True
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False

        self.output_dir = Path(str(train_cfg.output_dir))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=False)

    def build_optimizer(self) -> torch.optim.AdamW:
        opt_cfg = self.cfg.optimizer
        trainable_params = [param for param in self.model.parameters() if param.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found for optimizer.")
        return torch.optim.AdamW(
            trainable_params,
            lr=float(opt_cfg.lr),
            weight_decay=float(opt_cfg.weight_decay),
            betas=tuple(float(b) for b in opt_cfg.betas),
            eps=float(opt_cfg.eps),
        )

    def build_scheduler(self, optimizer: torch.optim.Optimizer):
        sched_cfg = self.cfg.scheduler
        base_lr = float(self.cfg.optimizer.lr)
        eta_min = float(sched_cfg.eta_min)
        total_steps = max(int(sched_cfg.T_max), 1)
        warmup_steps = max(int(getattr(sched_cfg, "warmup_steps", 0)), 0)
        warmup_start_factor = float(getattr(sched_cfg, "warmup_start_factor", 0.1))
        warmup_start_factor = min(max(warmup_start_factor, 1e-8), 1.0)

        if warmup_steps <= 0:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=eta_min,
            )

        cosine_steps = max(total_steps - warmup_steps, 1)
        min_lr_ratio = eta_min / max(base_lr, 1e-12)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                warmup_progress = float(current_step + 1) / float(max(warmup_steps, 1))
                return warmup_start_factor + (1.0 - warmup_start_factor) * warmup_progress

            cosine_step = min(current_step - warmup_steps, cosine_steps)
            cosine_progress = float(cosine_step) / float(cosine_steps)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_factor

        logger.info(
            "Using cosine LR scheduler with warmup: total_steps=%d, warmup_steps=%d, warmup_start_factor=%.3f, eta_min=%.2e",
            total_steps,
            warmup_steps,
            warmup_start_factor,
            eta_min,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def build_ema(self):
        ema_cfg = self.cfg.ema
        if isinstance(self.model, ServoVLA):
            return PolicyHeadEma(self.model, decay=float(ema_cfg.decay))

        from timm.utils import ModelEmaV2

        return ModelEmaV2(self.model, decay=float(ema_cfg.decay))

    def _autocast_ctx(self):
        if self.use_amp:
            return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)
        return torch.autocast(device_type=self.device.type, enabled=False)

    def _predict_vector_field_end2end(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        pixel_values: torch.Tensor,
        vlm_inputs: dict[str, torch.Tensor],
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.model, ServoVLA):
            return self.model(
                x_t=x_t,
                t=t,
                pixel_values=pixel_values,
                vlm_inputs=vlm_inputs,
                c_sem_mask=c_sem_mask,
                frame_delay=frame_delay,
                q_current=q_current,
            )
        return self.model(x_t, t, pixel_values, vlm_inputs, c_sem_mask, frame_delay, q_current)

    def _policy_dtype(self) -> torch.dtype:
        policy_head = getattr(self.model, "policy_head", None)
        if policy_head is not None:
            return next(policy_head.parameters()).dtype
        return next(self.model.parameters()).dtype

    def _action_normalizer(self):
        return getattr(self.model, "action_normalizer", None)

    def _train_step_policy_core(
        self,
        *,
        x_1: torch.Tensor,
        loss_mask: torch.Tensor,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
        dataset_slug,
        optimizer: torch.optim.Optimizer,
        defer_logging_tensors: bool,
        global_step: int = 0,
    ) -> dict[str, Any]:
        batch_size, horizon, action_dim = x_1.shape
        action_normalizer = self._action_normalizer()
        if action_normalizer is not None and bool(getattr(action_normalizer, "enabled", False)):
            x_1 = action_normalizer.normalize(x_1)
        x_0 = torch.randn_like(x_1)
        rollout_cfg = getattr(self.cfg, "rollout_action_loss", None)
        rollout_weight = rollout_action_loss_weight(rollout_cfg, int(global_step))
        t = torch.rand((batch_size,), device=self.device)
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_true = x_1 - x_0

        optimizer.zero_grad()
        with self._autocast_ctx():
            v_pred = self.model.forward_policy(
                x_t=x_t,
                t=t,
                f_vision=f_vision,
                c_sem=c_sem,
                c_sem_mask=c_sem_mask,
                frame_delay=frame_delay,
                q_current=q_current,
            )
            mse_loss = torch.nn.functional.mse_loss(v_pred, v_true, reduction="none").mean(dim=-1)
            fm_loss = (mse_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            if rollout_weight > 0.0:
                rollout_loss = compute_rollout_action_loss(
                    policy_head=self.model.policy_head,
                    x_1=x_1,
                    x_0=x_0,
                    loss_mask=loss_mask,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                    num_inference_steps=int(self.model.fm_solver.num_inference_steps),
                    loss_type=str(getattr(rollout_cfg, "loss_type", "smooth_l1")),
                    beta=float(getattr(rollout_cfg, "beta", 0.05)),
                )
            else:
                rollout_loss = fm_loss.new_zeros(())
            loss = fm_loss + float(rollout_weight) * rollout_loss

        loss.backward()
        grad_clip = float(getattr(self.cfg, "grad_clip_norm", 1.0))
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        if isinstance(dataset_slug, tuple):
            dataset_slug = list(dataset_slug)
        if isinstance(dataset_slug, list):
            dataset_slugs = [str(item) for item in dataset_slug]
        else:
            dataset_slugs = [str(dataset_slug)] * batch_size

        sample_weight = loss_mask.sum(dim=1).clamp(min=1.0)
        sample_loss = ((mse_loss * loss_mask).sum(dim=1) / sample_weight).detach()
        dataset_loss_by_slug: dict[str, Any] = {}
        dataset_count_by_slug: dict[str, int] = {}
        if defer_logging_tensors:
            for sample_idx, slug in enumerate(dataset_slugs):
                current_loss = sample_loss[sample_idx]
                if slug in dataset_loss_by_slug:
                    dataset_loss_by_slug[slug] = dataset_loss_by_slug[slug] + current_loss
                else:
                    dataset_loss_by_slug[slug] = current_loss
                dataset_count_by_slug[slug] = dataset_count_by_slug.get(slug, 0) + 1
            main_loss: Any = loss.detach()
            loss_fm: Any = fm_loss.detach()
            loss_rollout_action: Any = rollout_loss.detach()
            loss_total: Any = loss.detach()
        else:
            sample_loss_cpu = sample_loss.cpu()
            for sample_idx, slug in enumerate(dataset_slugs):
                dataset_loss_by_slug[slug] = dataset_loss_by_slug.get(slug, 0.0) + float(
                    sample_loss_cpu[sample_idx].item()
                )
                dataset_count_by_slug[slug] = dataset_count_by_slug.get(slug, 0) + 1
            main_loss = float(loss.item())
            loss_fm = float(fm_loss.item())
            loss_rollout_action = float(rollout_loss.item())
            loss_total = float(loss.item())

        return {
            "main_loss": main_loss,
            "loss_fm": loss_fm,
            "loss_rollout_action": loss_rollout_action,
            "loss_total": loss_total,
            "rollout_action_weight": float(rollout_weight),
            "dataset_slug": dataset_slugs if isinstance(dataset_slug, list) else dataset_slugs[0],
            "dataset_loss_by_slug": dataset_loss_by_slug,
            "dataset_count_by_slug": dataset_count_by_slug,
        }

    def train_step_from_features(
        self,
        encoded: EncodedFeatureBatch,
        optimizer: torch.optim.Optimizer,
        *,
        defer_logging_tensors: bool = False,
        global_step: int = 0,
    ) -> dict[str, Any]:
        policy_dtype = self._policy_dtype()
        encoded.wait_ready(self.device)
        return self._train_step_policy_core(
            x_1=encoded.action.to(self.device),
            loss_mask=encoded.loss_mask.to(self.device),
            f_vision=encoded.f_vision.to(self.device, dtype=policy_dtype),
            c_sem=encoded.c_sem.to(self.device, dtype=policy_dtype),
            c_sem_mask=encoded.c_sem_mask.to(self.device).bool(),
            frame_delay=encoded.frame_delay.to(self.device, dtype=torch.float32),
            q_current=encoded.q_current.to(self.device, dtype=policy_dtype),
            dataset_slug=encoded.dataset_slug,
            optimizer=optimizer,
            defer_logging_tensors=defer_logging_tensors,
            global_step=global_step,
        )

    def train_step_end2end(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        *,
        defer_logging_tensors: bool = False,
        global_step: int = 0,
    ) -> dict[str, Any]:
        model_dtype = next(self.model.parameters()).dtype
        x_1 = batch["action"].to(self.device)
        action_normalizer = self._action_normalizer()
        if action_normalizer is not None and bool(getattr(action_normalizer, "enabled", False)):
            x_1 = action_normalizer.normalize(x_1)
        loss_mask = batch["loss_mask"].to(self.device)
        pixel_values = batch["pixel_values"].to(self.device)
        vlm_inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in batch["vlm_inputs"].items()
        }
        c_sem_mask = batch["c_sem_mask"].to(self.device)
        frame_delay = batch["frame_delay"].to(self.device)
        q_current = batch["q_current"].to(self.device, dtype=model_dtype)

        batch_size, horizon, action_dim = x_1.shape

        x_0 = torch.randn_like(x_1)
        rollout_cfg = getattr(self.cfg, "rollout_action_loss", None)
        rollout_weight = rollout_action_loss_weight(rollout_cfg, int(global_step))
        t = torch.rand((batch_size,), device=self.device)
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)

        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        v_true = x_1 - x_0

        optimizer.zero_grad()
        with self._autocast_ctx():
            f_vision = None
            c_sem = None
            if isinstance(self.model, ServoVLA):
                f_vision, c_sem = self.model.encode_observations(
                    pixel_values=pixel_values,
                    vlm_inputs=vlm_inputs,
                )
                policy_param = next(iter(self.model.policy_head.parameters()), None)
                if policy_param is not None:
                    f_vision = f_vision.to(device=policy_param.device, dtype=policy_param.dtype)
                    c_sem = c_sem.to(device=policy_param.device, dtype=policy_param.dtype)
                v_pred = self.model.forward_policy(
                    x_t=x_t,
                    t=t,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            else:
                v_pred = self._predict_vector_field_end2end(
                    x_t=x_t,
                    t=t,
                    pixel_values=pixel_values,
                    vlm_inputs=vlm_inputs,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                )
            mse_loss = torch.nn.functional.mse_loss(v_pred, v_true, reduction="none").mean(dim=-1)
            fm_loss = (mse_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
            if rollout_weight > 0.0:
                if f_vision is None or c_sem is None:
                    raise RuntimeError(
                        "End-to-end rollout action loss requires model-facing encoded features."
                    )
                rollout_loss = compute_rollout_action_loss(
                    policy_head=self.model.policy_head,
                    x_1=x_1,
                    x_0=x_0,
                    loss_mask=loss_mask,
                    f_vision=f_vision,
                    c_sem=c_sem,
                    c_sem_mask=c_sem_mask,
                    frame_delay=frame_delay,
                    q_current=q_current,
                    num_inference_steps=int(self.model.fm_solver.num_inference_steps),
                    loss_type=str(getattr(rollout_cfg, "loss_type", "smooth_l1")),
                    beta=float(getattr(rollout_cfg, "beta", 0.05)),
                )
            else:
                rollout_loss = fm_loss.new_zeros(())
            loss = fm_loss + float(rollout_weight) * rollout_loss

        loss.backward()

        grad_clip = float(getattr(self.cfg, "grad_clip_norm", 1.0))
        nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()

        dataset_slug = batch.get("dataset_slug", "unknown")
        if isinstance(dataset_slug, tuple):
            dataset_slug = list(dataset_slug)
        if isinstance(dataset_slug, list):
            dataset_slugs = [str(item) for item in dataset_slug]
        else:
            dataset_slugs = [str(dataset_slug)] * batch_size

        sample_weight = loss_mask.sum(dim=1).clamp(min=1.0)
        sample_loss = ((mse_loss * loss_mask).sum(dim=1) / sample_weight).detach()
        dataset_loss_by_slug: dict[str, Any] = {}
        dataset_count_by_slug: dict[str, int] = {}
        if defer_logging_tensors:
            for sample_idx, slug in enumerate(dataset_slugs):
                current_loss = sample_loss[sample_idx]
                if slug in dataset_loss_by_slug:
                    dataset_loss_by_slug[slug] = dataset_loss_by_slug[slug] + current_loss
                else:
                    dataset_loss_by_slug[slug] = current_loss
                dataset_count_by_slug[slug] = dataset_count_by_slug.get(slug, 0) + 1
            main_loss: Any = loss.detach()
            loss_fm: Any = fm_loss.detach()
            loss_rollout_action: Any = rollout_loss.detach()
            loss_total: Any = loss.detach()
        else:
            sample_loss_cpu = sample_loss.cpu()
            for sample_idx, slug in enumerate(dataset_slugs):
                dataset_loss_by_slug[slug] = dataset_loss_by_slug.get(slug, 0.0) + float(
                    sample_loss_cpu[sample_idx].item()
                )
                dataset_count_by_slug[slug] = dataset_count_by_slug.get(slug, 0) + 1
            main_loss = float(loss.item())
            loss_fm = float(fm_loss.item())
            loss_rollout_action = float(rollout_loss.item())
            loss_total = float(loss.item())

        return {
            "main_loss": main_loss,
            "loss_fm": loss_fm,
            "loss_rollout_action": loss_rollout_action,
            "loss_total": loss_total,
            "rollout_action_weight": float(rollout_weight),
            "dataset_slug": dataset_slugs if isinstance(dataset_slug, list) else dataset_slugs[0],
            "dataset_loss_by_slug": dataset_loss_by_slug,
            "dataset_count_by_slug": dataset_count_by_slug,
        }

    def _maybe_run_eval(
        self,
        *,
        step: int,
        eval_every: int,
        eval_fn: Callable[[nn.Module, Iterable[dict[str, Any]], int, bool], dict[str, Any]] | None,
        eval_dataloader: Iterable[dict[str, Any]] | None,
        ema_model,
        use_ema: bool,
    ) -> dict[str, Any] | None:
        if eval_fn is None or eval_dataloader is None or eval_every <= 0 or step % eval_every != 0:
            return None

        eval_model = self.model
        was_training = eval_model.training
        eval_model.eval()
        try:
            with torch_compile_concurrency_guard():
                with torch.inference_mode():
                    raw_summary = eval_fn(eval_model, eval_dataloader, step, False)

            eval_summary = {"raw": raw_summary}
            ema_context = None
            if self._should_use_ema_for_eval(step=step, ema_model=ema_model, use_ema=use_ema):
                ema_context = (
                    ema_model.apply_to(eval_model) if hasattr(ema_model, "apply_to") else None
                )
            if ema_context is not None:
                with ema_context:
                    with torch_compile_concurrency_guard():
                        with torch.inference_mode():
                            eval_summary["ema"] = eval_fn(eval_model, eval_dataloader, step, True)
        finally:
            if was_training:
                eval_model.train()

        write_eval_summary(self.output_dir / "eval" / f"step{step:07d}.json", eval_summary)
        return eval_summary

    def _maybe_enqueue_eval(
        self,
        *,
        step: int,
        eval_every: int,
        eval_dataloader: Iterable[dict[str, Any]] | None,
        async_eval_enqueue_fn: Callable[..., None] | None,
        ema_model,
        use_ema: bool,
    ) -> bool:
        if (
            async_eval_enqueue_fn is None
            or eval_dataloader is None
            or eval_every <= 0
            or step % eval_every != 0
        ):
            return False

        async_eval_enqueue_fn(
            step=step,
            model=self.model,
            ema_model=ema_model,
            use_ema=self._should_use_ema_for_eval(step=step, ema_model=ema_model, use_ema=use_ema),
        )
        return True

    def _snapshot_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        if isinstance(model, ServoVLA):
            return policy_head_state_dict(model)
        return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    def _should_use_ema_for_eval(self, *, step: int, ema_model, use_ema: bool) -> bool:
        if not use_ema or ema_model is None:
            return False
        ema_cfg = self.cfg.ema if hasattr(self.cfg, "ema") else None
        ema_update_after_step = (
            int(getattr(ema_cfg, "update_after_step", 0)) if ema_cfg is not None else 0
        )
        return int(step) >= ema_update_after_step

    def _build_async_eval_task(
        self,
        *,
        step: int,
        ema_model,
        use_ema: bool,
    ) -> AsyncEvalTask:
        use_ema_now = self._should_use_ema_for_eval(step=step, ema_model=ema_model, use_ema=use_ema)
        snapshot = {"raw_state": self._snapshot_state_dict(self.model)}
        if use_ema_now and hasattr(ema_model, "state_dict"):
            snapshot["ema_state"] = ema_model.state_dict()
        return AsyncEvalTask(
            step=int(step),
            snapshot=snapshot,
            used_ema=use_ema_now,
            eval_seed=int(getattr(self.cfg, "eval_seed", 0)),
            eval_zero_noise=bool(getattr(self.cfg, "eval_zero_noise", False)),
        )

    def save_checkpoint(
        self,
        step: int,
        ema_model=None,
    ) -> str:
        ckpt = {
            "step": step,
            "model_state": policy_head_state_dict(self.model),
        }
        action_normalizer = getattr(self.model, "action_normalizer", None)
        if action_normalizer is not None and bool(getattr(action_normalizer, "enabled", False)):
            action_norm_cfg = action_normalizer.to_config()
            ckpt["action_normalization"] = {
                "enabled": bool(action_norm_cfg.enabled),
                "mean": action_norm_cfg.mean,
                "std": action_norm_cfg.std,
                "eps": float(action_norm_cfg.eps),
            }
        if ema_model is not None:
            ckpt["ema_state"] = ema_model.state_dict()

        ckpt_path = self.output_dir / f"checkpoint_step{step:07d}.pt"
        torch.save(ckpt, ckpt_path)
        logger.info("Saved checkpoint -> %s", ckpt_path)
        return str(ckpt_path)

    def run(
        self,
        dataloader: Iterable[dict[str, Any]] | DataLoader,
        max_steps: int | None = None,
        *,
        warmup_dataloader: Iterable[dict[str, Any]] | None = None,
        eval_dataloader: Iterable[dict[str, Any]] | None = None,
        eval_fn: Callable[[nn.Module, Iterable[dict[str, Any]], int, bool], dict[str, Any]]
        | None = None,
        async_eval_enqueue_fn: Callable[..., None] | None = None,
        async_eval_manager=None,
    ) -> str:
        optimizer = self.build_optimizer()
        scheduler = self.build_scheduler(optimizer)
        try:
            ema_model = self.build_ema()
        except ImportError:
            ema_model = None
            logger.warning("timm not found - EMA disabled")

        ema_cfg = self.cfg.ema if hasattr(self.cfg, "ema") else None
        ema_update_after_step = (
            int(getattr(ema_cfg, "update_after_step", 0)) if ema_cfg is not None else 0
        )
        ema_update_every = (
            max(int(getattr(ema_cfg, "update_every", 1)), 1) if ema_cfg is not None else 1
        )

        if max_steps is None:
            max_steps = int(self.cfg.max_steps)

        log_every = int(getattr(self.cfg, "log_every", 50))
        save_every = int(getattr(self.cfg, "save_every", 5000))
        eval_every = int(getattr(self.cfg, "eval_every", 0))
        eval_use_ema = bool(getattr(self.cfg, "eval_use_ema", True))
        async_eval_cfg = getattr(self.cfg, "async_eval", None)
        async_eval_enabled = bool(getattr(async_eval_cfg, "enabled", False))

        step = 0

        try:
            import wandb

            _use_wandb = wandb.run is not None
        except ImportError:
            _use_wandb = False

        self.model.train()
        data_iter: Iterator | None = None
        latest_batch_memory: dict[str, int | float] = {}
        latest_process_memory: dict[str, int | None] = {}

        def _ensure_sync_data_iter() -> Iterator:
            nonlocal data_iter
            if data_iter is None:
                data_iter = iter(dataloader)
            return data_iter

        def _next_batch_with_data_time() -> tuple[dict[str, Any], float]:
            nonlocal data_iter, latest_batch_memory, latest_process_memory
            data_t0 = time.perf_counter()
            try:
                next_batch = next(_ensure_sync_data_iter())
            except StopIteration:
                data_iter = iter(dataloader)
                next_batch = next(data_iter)
            if isinstance(next_batch, Mapping):
                latest_batch_memory = summarize_batch_memory(next_batch)
                latest_process_memory = collect_process_memory_snapshot(
                    worker_pids=dataloader_worker_pids(data_iter) if data_iter is not None else ()
                )
            return next_batch, time.perf_counter() - data_t0

        def _batch_memory_from_metadata(metadata: Mapping[str, Any]) -> dict[str, int | float]:
            result: dict[str, int | float] = {}
            for key, value in metadata.items():
                if not isinstance(key, str):
                    continue
                if not (
                    key.startswith("tensor_bytes.")
                    or key.startswith("raw_image_bytes.")
                    or key.startswith("raw_decode_stats.")
                    or key.startswith("producer_timing.")
                    or key.startswith("producer_state.")
                ):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                result[str(key)] = float(value) if isinstance(value, float) else int(value)
            return result

        gpu_pipeline_cfg = getattr(self.cfg, "gpu_pipeline", None)
        gpu_pipeline_enabled = (
            bool(getattr(gpu_pipeline_cfg, "enabled", False))
            and self.device.type == "cuda"
            and isinstance(self.model, ServoVLA)
        )
        raw_async_pipeline_enabled = (
            gpu_pipeline_enabled
            and bool(getattr(gpu_pipeline_cfg, "gpu_preprocess", False))
            and bool(getattr(gpu_pipeline_cfg, "async_producer", False))
        )
        if raw_async_pipeline_enabled and self.image_preprocessor is None:
            raise RuntimeError(
                "Raw async GPU preprocessing requires TrainerLoop(image_preprocessor=...)."
            )
        vlm_compile_cfg = getattr(self.cfg, "vlm_compile", None)
        vlm_warmup_max_batches = 0
        if (
            raw_async_pipeline_enabled
            and bool(getattr(self.cfg, "compile_vlm_encoder", False))
            and vlm_compile_cfg is not None
            and bool(getattr(vlm_compile_cfg, "warmup_enabled", False))
        ):
            vlm_warmup_max_batches = max(int(getattr(vlm_compile_cfg, "warmup_max_batches", 0)), 0)
        vlm_signature_registry = None
        if vlm_warmup_max_batches > 0:
            vlm_signature_registry = VlmSignatureRegistry(
                max_entries=int(getattr(vlm_compile_cfg, "signature_max_entries", 16))
            )
        feature_producer = None
        async_feature_producer = None
        if gpu_pipeline_enabled:
            feature_producer = SingleGpuFeatureProducer(
                model=self.model,
                device=self.device,
                amp_dtype=self.amp_dtype,
                queue_depth=int(getattr(gpu_pipeline_cfg, "feature_queue_depth", 1)),
                encoder_parallel_streams=bool(
                    getattr(gpu_pipeline_cfg, "encoder_parallel_streams", False)
                ),
                image_preprocessor=self.image_preprocessor if raw_async_pipeline_enabled else None,
                vision_encoder_micro_batch_size=int(
                    getattr(gpu_pipeline_cfg, "vision_encoder_micro_batch_size", 0) or 0
                ),
                vlm_encoder_micro_batch_size=int(
                    getattr(gpu_pipeline_cfg, "vlm_encoder_micro_batch_size", 0) or 0
                ),
                startup_trace_submit_limit=int(
                    getattr(gpu_pipeline_cfg, "startup_trace_submit_limit", 0) or 0
                ),
            )
            if raw_async_pipeline_enabled:
                async_feature_producer = AsyncGpuFeatureProducer(
                    dataloader=dataloader,
                    feature_producer=feature_producer,
                    feature_queue_depth=int(getattr(gpu_pipeline_cfg, "feature_queue_depth", 1)),
                    cpu_prefetch_depth=int(getattr(gpu_pipeline_cfg, "cpu_prefetch_depth", 1)),
                    max_steps=max(int(max_steps) - step, 0),
                    batch_metadata_fn=summarize_batch_memory,
                    warmup_dataloader=warmup_dataloader,
                    vlm_warmup_max_batches=vlm_warmup_max_batches,
                    vlm_signature_registry=vlm_signature_registry,
                    on_new_vlm_signature=str(
                        getattr(vlm_compile_cfg, "on_new_signature", "warn_and_warmup")
                    ),
                    log_vlm_signatures=bool(getattr(vlm_compile_cfg, "log_signatures", True)),
                    overlap_train_reader_during_warmup=bool(
                        getattr(vlm_compile_cfg, "overlap_train_reader", False)
                    ),
                )
        gpu_timing_log_every = (
            int(getattr(gpu_pipeline_cfg, "log_timing_every", 0))
            if (feature_producer is not None or async_feature_producer is not None)
            and gpu_pipeline_cfg is not None
            else 0
        )
        compile_concurrency_guard_steps = (
            _ASYNC_PRODUCER_COMPILE_GUARD_STEPS if bool(getattr(self.cfg, "compile", True)) else 0
        )

        def _submit_next_feature_batch() -> bool:
            if feature_producer is None or not feature_producer.can_submit():
                return False
            next_batch, next_data_time = _next_batch_with_data_time()
            feature_producer.submit(next_batch, metadata={"data_time": next_data_time})
            return True

        def _should_log(current_step: int) -> bool:
            if current_step <= 0:
                return False
            if current_step in {1, 2, 5, 10, 20, 50}:
                return True
            return current_step % log_every == 0

        last_checkpoint_path = ""
        startup_trace_initial_step = int(step)
        startup_trace_step_limit = (
            int(getattr(gpu_pipeline_cfg, "startup_trace_step_limit", 0) or 0)
            if gpu_pipeline_cfg is not None
            else 0
        )
        last_log_step = step
        last_log_wall_t0 = time.perf_counter()
        startup_t0 = last_log_wall_t0
        window_data_time = 0.0
        window_step_time = 0.0
        window_h2d_time = 0.0
        window_preprocess_time = 0.0
        window_encoder_time = 0.0
        window_ready_wait_time = 0.0
        window_vision_encoder_time = 0.0
        window_vlm_encoder_time = 0.0
        window_producer_wait_time = 0.0
        window_main_wait_time = 0.0
        train_loss_by_dataset: dict[str, Any] = {}
        train_count_by_dataset: dict[str, int] = {}
        latest_main_loss: Any = 0.0
        latest_loss_fm: Any = 0.0
        latest_loss_rollout_action: Any = 0.0
        latest_loss_total: Any = 0.0
        latest_rollout_action_weight: float = 0.0

        try:
            if async_feature_producer is not None:
                async_feature_producer.start()
                async_feature_producer.wait_warmup_complete(timeout=None)
            elif feature_producer is not None:
                while step + len(feature_producer) < max_steps and feature_producer.can_submit():
                    _submit_next_feature_batch()

            while step < max_steps:
                iter_t0 = time.perf_counter()
                trace_startup_step = (
                    int(step) - startup_trace_initial_step
                ) < startup_trace_step_limit
                step_trace_t0 = time.perf_counter()

                def _trace_startup_step(stage: str) -> None:
                    if trace_startup_step:
                        logger.info(
                            "trainer startup step %d | %s | %.3fs",
                            int(step),
                            stage,
                            time.perf_counter() - step_trace_t0,
                        )

                producer_wait_time = 0.0
                ready_wait_time = 0.0
                preprocess_time = 0.0
                vision_encoder_time = 0.0
                vlm_encoder_time = 0.0
                if async_feature_producer is not None:
                    wait_t0 = time.perf_counter()
                    _trace_startup_step("pop_start")
                    encoded_batch = async_feature_producer.pop_oldest()
                    producer_wait_time = time.perf_counter() - wait_t0
                    _trace_startup_step("pop_done")
                    ready_event_completed = getattr(encoded_batch, "ready_event_completed", None)
                    ready_at_pop = (
                        ready_event_completed() if callable(ready_event_completed) else None
                    )
                    if ready_at_pop is not None:
                        encoded_batch.metadata["producer_state.ready_at_pop"] = (
                            1 if ready_at_pop else 0
                        )
                    data_time = float(encoded_batch.metadata.get("data_time", 0.0))
                    latest_batch_memory = _batch_memory_from_metadata(encoded_batch.metadata)
                    step_t0 = time.perf_counter()
                    ready_wait_t0 = time.perf_counter()
                    _trace_startup_step("ready_wait_start")
                    encoded_batch.wait_ready(self.device)
                    ready_wait_time = time.perf_counter() - ready_wait_t0
                    _trace_startup_step("ready_wait_done")
                    compile_guard = (
                        torch_compile_concurrency_guard()
                        if step < compile_concurrency_guard_steps
                        else nullcontext()
                    )
                    _trace_startup_step("train_step_start")
                    with compile_guard:
                        step_result = self.train_step_from_features(
                            encoded_batch,
                            optimizer,
                            defer_logging_tensors=True,
                            global_step=step,
                        )
                    _trace_startup_step("train_step_done")
                    timings = encoded_batch.collect_timings(block=False)
                    h2d_time = float(timings.get("h2d_ms", 0.0)) / 1000.0
                    preprocess_time = float(timings.get("preprocess_ms", 0.0)) / 1000.0
                    encoder_time = float(timings.get("encoder_ms", 0.0)) / 1000.0
                    vision_encoder_time = float(timings.get("vision_encoder_ms", 0.0)) / 1000.0
                    vlm_encoder_time = float(timings.get("vlm_encoder_ms", 0.0)) / 1000.0
                elif feature_producer is not None:
                    if len(feature_producer) == 0:
                        _submit_next_feature_batch()
                    encoded_batch = feature_producer.pop_oldest()
                    ready_event_completed = getattr(encoded_batch, "ready_event_completed", None)
                    ready_at_pop = (
                        ready_event_completed() if callable(ready_event_completed) else None
                    )
                    if ready_at_pop is not None:
                        encoded_batch.metadata["producer_state.ready_at_pop"] = (
                            1 if ready_at_pop else 0
                        )
                    remaining_after_current = max(max_steps - step - 1, 0)
                    while (
                        len(feature_producer) < remaining_after_current
                        and feature_producer.can_submit()
                    ):
                        _submit_next_feature_batch()
                    data_time = float(encoded_batch.metadata.get("data_time", 0.0))
                    step_t0 = time.perf_counter()
                    ready_wait_t0 = time.perf_counter()
                    encoded_batch.wait_ready(self.device)
                    ready_wait_time = time.perf_counter() - ready_wait_t0
                    step_result = self.train_step_from_features(
                        encoded_batch,
                        optimizer,
                        defer_logging_tensors=True,
                        global_step=step,
                    )
                    timings = encoded_batch.collect_timings(block=False)
                    h2d_time = float(timings.get("h2d_ms", 0.0)) / 1000.0
                    preprocess_time = float(timings.get("preprocess_ms", 0.0)) / 1000.0
                    encoder_time = float(timings.get("encoder_ms", 0.0)) / 1000.0
                    vision_encoder_time = float(timings.get("vision_encoder_ms", 0.0)) / 1000.0
                    vlm_encoder_time = float(timings.get("vlm_encoder_ms", 0.0)) / 1000.0
                else:
                    batch, data_time = _next_batch_with_data_time()
                    step_t0 = time.perf_counter()
                    step_result = self.train_step_end2end(
                        batch,
                        optimizer,
                        defer_logging_tensors=True,
                        global_step=step,
                    )
                    h2d_time = 0.0
                    encoder_time = 0.0
                latest_main_loss = step_result["main_loss"]
                latest_loss_fm = step_result.get("loss_fm", latest_main_loss)
                latest_loss_rollout_action = step_result.get("loss_rollout_action", 0.0)
                latest_loss_total = step_result.get("loss_total", latest_main_loss)
                latest_rollout_action_weight = float(step_result.get("rollout_action_weight", 0.0))
                dataset_loss_by_slug = step_result.get("dataset_loss_by_slug") or {
                    str(step_result["dataset_slug"]): latest_main_loss
                }
                dataset_count_by_slug = step_result.get("dataset_count_by_slug") or {
                    str(step_result["dataset_slug"]): 1
                }
                for slug, total_loss in dataset_loss_by_slug.items():
                    if slug in train_loss_by_dataset:
                        train_loss_by_dataset[slug] = train_loss_by_dataset[slug] + total_loss
                    else:
                        train_loss_by_dataset[slug] = total_loss
                    train_count_by_dataset[slug] = train_count_by_dataset.get(slug, 0) + int(
                        dataset_count_by_slug.get(slug, 0)
                    )

                scheduler.step()

                current_step = step + 1
                if ema_model is not None:
                    should_update_ema = current_step >= ema_update_after_step and (
                        (current_step - ema_update_after_step) % ema_update_every == 0
                    )
                    if should_update_ema:
                        ema_model.update(self.model)

                step = current_step
                step_time = time.perf_counter() - step_t0
                iter_wall_time = time.perf_counter() - iter_t0
                main_wait_time = producer_wait_time + ready_wait_time
                if async_feature_producer is None:
                    main_wait_time += data_time
                io_ratio = main_wait_time / max(step_time + main_wait_time, 1e-8)
                window_data_time += data_time
                window_step_time += step_time
                window_h2d_time += h2d_time
                window_preprocess_time += preprocess_time
                window_encoder_time += encoder_time
                window_ready_wait_time += ready_wait_time
                window_vision_encoder_time += vision_encoder_time
                window_vlm_encoder_time += vlm_encoder_time
                window_producer_wait_time += producer_wait_time
                window_main_wait_time += main_wait_time
                active_feature_queue = (
                    async_feature_producer
                    if async_feature_producer is not None
                    else feature_producer
                )
                if (
                    active_feature_queue is not None
                    and gpu_timing_log_every > 0
                    and step % gpu_timing_log_every == 0
                ):
                    if vision_encoder_time > 0.0 or vlm_encoder_time > 0.0:
                        logger.info(
                            "gpu_pipeline timing step %7d | h2d %.3fs | preprocess %.3fs | encoder %.3fs | vision_encoder %.3fs | vlm_encoder %.3fs | producer_wait %.3fs | ready_wait %.3fs | queue_depth %d",
                            step,
                            h2d_time,
                            preprocess_time,
                            encoder_time,
                            vision_encoder_time,
                            vlm_encoder_time,
                            producer_wait_time,
                            ready_wait_time,
                            len(active_feature_queue),
                        )
                    else:
                        logger.info(
                            "gpu_pipeline timing step %7d | h2d %.3fs | preprocess %.3fs | encoder %.3fs | producer_wait %.3fs | ready_wait %.3fs | queue_depth %d",
                            step,
                            h2d_time,
                            preprocess_time,
                            encoder_time,
                            producer_wait_time,
                            ready_wait_time,
                            len(active_feature_queue),
                        )

                eval_summary = None
                queued_async_eval = False
                if async_eval_enabled and async_eval_manager is not None:
                    if eval_dataloader is not None and eval_every > 0 and step % eval_every == 0:
                        async_eval_manager.enqueue(
                            self._build_async_eval_task(
                                step=step,
                                ema_model=ema_model,
                                use_ema=eval_use_ema,
                            )
                        )
                        queued_async_eval = True
                        logger.info(
                            "queued async eval for step %7d | use_ema=%s", step, eval_use_ema
                        )
                elif async_eval_enabled and async_eval_enqueue_fn is not None:
                    queued_async_eval = self._maybe_enqueue_eval(
                        step=step,
                        eval_every=eval_every,
                        eval_dataloader=eval_dataloader,
                        async_eval_enqueue_fn=async_eval_enqueue_fn,
                        ema_model=ema_model,
                        use_ema=eval_use_ema,
                    )
                    if queued_async_eval:
                        logger.info(
                            "queued async eval for step %7d | use_ema=%s", step, eval_use_ema
                        )
                else:
                    eval_summary = self._maybe_run_eval(
                        step=step,
                        eval_every=eval_every,
                        eval_fn=eval_fn,
                        eval_dataloader=eval_dataloader,
                        ema_model=ema_model,
                        use_ema=eval_use_ema,
                    )
                    if eval_summary is not None:
                        logger.info(
                            "eval step %7d | raw_first_step_mae %.6f | raw_all_steps_mae %.6f | ema_first_step_mae %.6f | ema_all_steps_mae %.6f",
                            step,
                            float(eval_summary["raw"]["first_step"]["mae_mean"]),
                            float(eval_summary["raw"]["all_steps"]["mae_mean"]),
                            float(
                                eval_summary.get("ema", eval_summary["raw"])["first_step"][
                                    "mae_mean"
                                ]
                            ),
                            float(
                                eval_summary.get("ema", eval_summary["raw"])["all_steps"][
                                    "mae_mean"
                                ]
                            ),
                        )
                        if _use_wandb:
                            import wandb

                            eval_log_payload = build_eval_log_payload(eval_summary, eval_step=step)
                            wandb.log(eval_log_payload)

                if async_eval_enabled and async_eval_manager is not None and not queued_async_eval:
                    async_eval_manager.check_healthy()

                if _should_log(step):
                    lr = scheduler.get_last_lr()[0]
                    main_loss = float(
                        latest_main_loss.item()
                        if torch.is_tensor(latest_main_loss)
                        else latest_main_loss
                    )
                    loss_fm = float(
                        latest_loss_fm.item() if torch.is_tensor(latest_loss_fm) else latest_loss_fm
                    )
                    loss_rollout_action = float(
                        latest_loss_rollout_action.item()
                        if torch.is_tensor(latest_loss_rollout_action)
                        else latest_loss_rollout_action
                    )
                    loss_total = float(
                        latest_loss_total.item()
                        if torch.is_tensor(latest_loss_total)
                        else latest_loss_total
                    )
                    steps_since_last_log = max(step - last_log_step, 1)
                    window_wall_time = time.perf_counter() - last_log_wall_t0
                    avg_data_time = window_data_time / steps_since_last_log
                    avg_step_time = window_step_time / steps_since_last_log
                    avg_h2d_time = window_h2d_time / steps_since_last_log
                    avg_preprocess_time = window_preprocess_time / steps_since_last_log
                    avg_encoder_time = window_encoder_time / steps_since_last_log
                    avg_ready_wait_time = window_ready_wait_time / steps_since_last_log
                    avg_vision_encoder_time = window_vision_encoder_time / steps_since_last_log
                    avg_vlm_encoder_time = window_vlm_encoder_time / steps_since_last_log
                    avg_producer_wait_time = window_producer_wait_time / steps_since_last_log
                    avg_main_wait_time = window_main_wait_time / steps_since_last_log
                    avg_wall_time = window_wall_time / steps_since_last_log
                    stall_time = window_main_wait_time
                    avg_stall_time = stall_time / steps_since_last_log
                    stall_ratio = stall_time / max(window_wall_time, 1e-8)
                    startup_elapsed = time.perf_counter() - startup_t0
                    logger.info(
                        "step %7d | main_loss %.6f | lr %.2e | data %.3fs | step %.3fs | h2d %.3fs | preprocess %.3fs | encoder %.3fs | producer_wait %.3fs | ready_wait %.3fs | io_ratio %.2f | avg_data %.3fs | avg_step %.3fs | avg_h2d %.3fs | avg_preprocess %.3fs | avg_encoder %.3fs | avg_producer_wait %.3fs | avg_ready_wait %.3fs | avg_wall %.3fs | avg_stall %.3fs | stall_ratio %.2f | since_log %.1fs | since_start %.1fs",
                        step,
                        main_loss,
                        lr,
                        data_time,
                        step_time,
                        h2d_time,
                        preprocess_time,
                        encoder_time,
                        producer_wait_time,
                        ready_wait_time,
                        io_ratio,
                        avg_data_time,
                        avg_step_time,
                        avg_h2d_time,
                        avg_preprocess_time,
                        avg_encoder_time,
                        avg_producer_wait_time,
                        avg_ready_wait_time,
                        avg_wall_time,
                        avg_stall_time,
                        stall_ratio,
                        window_wall_time,
                        startup_elapsed,
                    )
                    if latest_batch_memory or latest_process_memory:
                        logger.info(
                            "memory step %7d | batch_tensor_bytes %s | pixel_values_bytes %s | vlm_pixel_values_bytes %s | raw_image_bytes %s | rss_main_bytes %s | rss_workers_bytes %s",
                            step,
                            latest_batch_memory.get("tensor_bytes.total"),
                            latest_batch_memory.get("tensor_bytes.pixel_values"),
                            latest_batch_memory.get("tensor_bytes.vlm_inputs.pixel_values"),
                            latest_batch_memory.get("raw_image_bytes.total"),
                            latest_process_memory.get("rss_bytes.main"),
                            latest_process_memory.get("rss_bytes.workers_total"),
                        )
                    if latest_batch_memory.get("raw_decode_stats.seconds") is not None:
                        logger.info(
                            "raw_decode step %7d | seconds %.3fs | calls %s | frames %s | videos %s | groups %s | max_span %.3fs | job_sum %.3fs | group_sum %.3fs | max_job %.3fs | max_group %.3fs",
                            step,
                            float(latest_batch_memory.get("raw_decode_stats.seconds", 0.0)),
                            latest_batch_memory.get("raw_decode_stats.calls"),
                            latest_batch_memory.get("raw_decode_stats.frames"),
                            latest_batch_memory.get("raw_decode_stats.videos"),
                            latest_batch_memory.get("raw_decode_stats.groups"),
                            float(latest_batch_memory.get("raw_decode_stats.max_span_s", 0.0)),
                            float(latest_batch_memory.get("raw_decode_stats.job_seconds_sum", 0.0)),
                            float(
                                latest_batch_memory.get("raw_decode_stats.group_seconds_sum", 0.0)
                            ),
                            float(latest_batch_memory.get("raw_decode_stats.max_job_seconds", 0.0)),
                            float(
                                latest_batch_memory.get("raw_decode_stats.max_group_seconds", 0.0)
                            ),
                        )
                    if latest_batch_memory.get("raw_decode_stats.cache_hits") is not None:
                        logger.info(
                            "raw_decode_cache step %7d | hits %s | misses %s | creates %s | evictions %s | fallbacks %s | cache_size %s | active %s",
                            step,
                            latest_batch_memory.get("raw_decode_stats.cache_hits"),
                            latest_batch_memory.get("raw_decode_stats.cache_misses"),
                            latest_batch_memory.get("raw_decode_stats.decoder_creates"),
                            latest_batch_memory.get("raw_decode_stats.decoder_evictions"),
                            latest_batch_memory.get("raw_decode_stats.fallbacks"),
                            latest_batch_memory.get("raw_decode_stats.cache_size"),
                            latest_batch_memory.get("raw_decode_stats.cache_active"),
                        )
                    if latest_batch_memory.get("producer_timing.reader_data_time") is not None:
                        logger.info(
                            "producer step %7d | reader_data %.3fs | cpu_queue_put %.3fs | encoder_cpu_wait %.3fs | submit %.3fs | feature_queue_put %.3fs | ready_at_pop %s",
                            step,
                            float(latest_batch_memory.get("producer_timing.reader_data_time", 0.0)),
                            float(
                                latest_batch_memory.get("producer_timing.cpu_queue_put_wait", 0.0)
                            ),
                            float(
                                latest_batch_memory.get(
                                    "producer_timing.encoder_cpu_queue_wait", 0.0
                                )
                            ),
                            float(latest_batch_memory.get("producer_timing.submit_time", 0.0)),
                            float(
                                latest_batch_memory.get(
                                    "producer_timing.feature_queue_put_wait", 0.0
                                )
                            ),
                            latest_batch_memory.get("producer_state.ready_at_pop"),
                        )
                    if _use_wandb:
                        import wandb

                        log_payload = {
                            "global_step": step,
                            "train/loss_main": main_loss,
                            "train/loss_fm": loss_fm,
                            "train/loss_rollout_action": loss_rollout_action,
                            "train/loss_total": loss_total,
                            "train/rollout_action_weight": latest_rollout_action_weight,
                            "train/lr": lr,
                            "perf/data_time": data_time,
                            "perf/step_time": step_time,
                            "perf/h2d_time": h2d_time,
                            "perf/preprocess_time": preprocess_time,
                            "perf/encoder_time": encoder_time,
                            "perf/ready_wait_time": ready_wait_time,
                            "perf/vision_encoder_time": vision_encoder_time,
                            "perf/vlm_encoder_time": vlm_encoder_time,
                            "perf/producer_wait_time": producer_wait_time,
                            "perf/main_wait_time": main_wait_time,
                            "perf/io_ratio": io_ratio,
                            "perf/iter_wall_time": iter_wall_time,
                            "perf/avg_data_time": avg_data_time,
                            "perf/avg_step_time": avg_step_time,
                            "perf/avg_h2d_time": avg_h2d_time,
                            "perf/avg_preprocess_time": avg_preprocess_time,
                            "perf/avg_encoder_time": avg_encoder_time,
                            "perf/avg_ready_wait_time": avg_ready_wait_time,
                            "perf/avg_vision_encoder_time": avg_vision_encoder_time,
                            "perf/avg_vlm_encoder_time": avg_vlm_encoder_time,
                            "perf/avg_producer_wait_time": avg_producer_wait_time,
                            "perf/avg_main_wait_time": avg_main_wait_time,
                            "perf/avg_wall_time": avg_wall_time,
                            "perf/avg_stall_time": avg_stall_time,
                            "perf/stall_ratio": stall_ratio,
                            "perf/time_since_start": startup_elapsed,
                        }
                        log_payload.update(
                            {
                                f"perf/{key}": value
                                for key, value in latest_batch_memory.items()
                                if value is not None
                            }
                        )
                        log_payload.update(
                            {
                                f"perf/{key}": value
                                for key, value in latest_process_memory.items()
                                if value is not None
                            }
                        )
                        for dataset_slug, total_loss in train_loss_by_dataset.items():
                            avg_loss = total_loss / max(train_count_by_dataset[dataset_slug], 1)
                            log_payload[f"train/loss/{dataset_slug}"] = float(
                                avg_loss.item() if torch.is_tensor(avg_loss) else avg_loss
                            )
                        wandb.log(log_payload, step=step)
                    last_log_step = step
                    last_log_wall_t0 = time.perf_counter()
                    window_data_time = 0.0
                    window_step_time = 0.0
                    window_h2d_time = 0.0
                    window_preprocess_time = 0.0
                    window_encoder_time = 0.0
                    window_ready_wait_time = 0.0
                    window_vision_encoder_time = 0.0
                    window_vlm_encoder_time = 0.0
                    window_producer_wait_time = 0.0
                    window_main_wait_time = 0.0
                    train_loss_by_dataset = {}
                    train_count_by_dataset = {}

                if save_every > 0 and step % save_every == 0:
                    last_checkpoint_path = self.save_checkpoint(step, ema_model)
        finally:
            if async_feature_producer is not None:
                async_feature_producer.stop()
            if async_eval_enabled and async_eval_manager is not None:
                async_eval_manager.finish()
                async_eval_manager.join()
                async_eval_manager.check_healthy()

        if step > 0 and save_every > 0 and step % save_every != 0:
            last_checkpoint_path = self.save_checkpoint(step, ema_model)

        logger.info("Training complete at step %d", step)
        return last_checkpoint_path

    @staticmethod
    def compile_model(model: nn.Module, *, train_cfg=None):
        """Compile critical submodules when possible."""
        if not hasattr(torch, "compile"):
            return model

        logger.info("Initiating torch.compile() ...")

        try:
            if isinstance(model, ServoVLA):
                compile_policy_head = bool(getattr(train_cfg, "compile_policy_head", True))
                compile_vision_encoder = bool(getattr(train_cfg, "compile_vision_encoder", True))
                compile_vlm_encoder = bool(getattr(train_cfg, "compile_vlm_encoder", True))
                compiled_modules: list[str] = []
                policy_head = model.policy_head
                vision_encoder = model.vision_encoder
                vlm_encoder = model.vlm_encoder
                if compile_policy_head and model.policy_head is not None:
                    policy_head = torch.compile(model.policy_head)
                    compiled_modules.append("policy_head")
                if compile_vision_encoder and model.vision_encoder is not None:
                    vision_encoder = torch.compile(model.vision_encoder)
                    compiled_modules.append("vision_encoder")
                if compile_vlm_encoder and model.vlm_encoder is not None:
                    vlm_compile_cfg = getattr(train_cfg, "vlm_compile", None)
                    if bool(getattr(vlm_compile_cfg, "flash_attention_graph_break", True)):
                        patched = apply_transformers_flash_attention_compile_graph_break()
                        if patched:
                            logger.info(
                                "Applied Transformers FlashAttention graph break before VLM torch.compile()."
                            )
                    if bool(getattr(vlm_compile_cfg, "qwen_visual_position_graph_break", True)):
                        patched = apply_transformers_qwen_visual_position_compile_graph_break()
                        if patched:
                            logger.info(
                                "Applied Qwen visual position graph break before VLM torch.compile()."
                            )
                    if bool(getattr(vlm_compile_cfg, "nested_fx_trace_fallback", True)):
                        patched = apply_torch_compile_nested_fx_trace_fallback()
                        if patched:
                            logger.info(
                                "Disabled Dynamo nested FX trace errors for VLM torch.compile fallback."
                            )
                    vlm_encoder = torch.compile(model.vlm_encoder)
                    compiled_modules.append("vlm_encoder")
                model.policy_head = policy_head
                model.vision_encoder = vision_encoder
                model.vlm_encoder = vlm_encoder
                logger.info(
                    "Model compilation complete (ServoVLA modules=%s)",
                    ",".join(compiled_modules) if compiled_modules else "none",
                )
                return model

            compiled = torch.compile(model)
            logger.info("Model compilation complete")
            return compiled
        except Exception as exc:
            logger.warning("torch.compile failed (%s); falling back to eager mode", exc)
            return model
