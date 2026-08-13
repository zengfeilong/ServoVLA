from __future__ import annotations

import logging
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from PIL import Image
from torch import Tensor
from transformers import AutoImageProcessor, AutoProcessor

from .configuration_servovla import ServoVLAConfig, _action_step_delay_support
from .runtime_semantic_delay import SemanticDelayRuntime

logger = logging.getLogger(__name__)


def _ensure_local_servovla_path() -> None:
    current = Path(__file__).resolve()
    candidates = [
        current.parents[1],
        current.parents[2] / "src",
    ]
    for candidate in candidates:
        if (candidate / "servovla").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return


_ensure_local_servovla_path()

from servovla.architectures.action_normalization import ActionNormalizer  # noqa: E402
from servovla.architectures.fm_solver import FlowMatchingEulerSolver  # noqa: E402
from servovla.architectures.inference_noise import make_inference_noise  # noqa: E402
from servovla.architectures.policy_head import FlowMatchingDiT  # noqa: E402
from servovla.architectures.policy_head_checkpoint import (  # noqa: E402
    load_policy_head_state_dict,
    prefixed_policy_head_state_dict,
)
from servovla.architectures.servo_vla import ServoVLA  # noqa: E402
from servovla.architectures.vision_encoder import VisionEncoder  # noqa: E402
from servovla.architectures.vlm_encoder import VLMEncoder  # noqa: E402
from servovla.architectures.vlm_forward import run_vlm_encoder_forward  # noqa: E402
from servovla.config.action_mode import normalize_action_delta_state_indices  # noqa: E402
from servovla.config.hf_offline import hf_from_pretrained_kwargs  # noqa: E402


class ServoVLAPolicy(PreTrainedPolicy):
    config_class = ServoVLAConfig
    name = "servovla"

    def __init__(self, config: ServoVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()

        vision_encoder = VisionEncoder(model_id=config.vision_model_name)
        vlm_encoder = VLMEncoder(model_id=config.vlm_model_name)
        policy_head = FlowMatchingDiT(
            action_dim=config.action_dim,
            state_dim=config.state_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            vision_feature_dim=config.vision_feature_dim,
            semantic_feature_dim=config.semantic_feature_dim,
            dropout=config.dropout,
            vision_grid_size=config.vision_grid_size,
            num_cameras=config.num_cameras,
        )
        solver = FlowMatchingEulerSolver(
            action_dim=config.action_dim,
            chunk_size=config.chunk_size,
            num_inference_steps=config.num_inference_steps,
        )
        action_normalizer = ActionNormalizer(
            enabled=bool(getattr(config, "action_normalization_enabled", False)),
            mean=getattr(config, "action_normalization_mean", None),
            std=getattr(config, "action_normalization_std", None),
            eps=float(getattr(config, "action_normalization_eps", 1.0e-6)),
        )
        self.model = ServoVLA(
            vision_encoder=vision_encoder,
            vlm_encoder=vlm_encoder,
            policy_head=policy_head,
            fm_solver=solver,
            action_is_delta=(config.action_mode == "delta"),
            action_delta_state_indices=config.action_delta_state_indices,
            action_normalizer=action_normalizer,
        )
        self.vision_processor = AutoImageProcessor.from_pretrained(
            config.vision_model_name,
            **hf_from_pretrained_kwargs(use_fast=True),
        )
        self.vlm_processor = AutoProcessor.from_pretrained(
            config.vlm_model_name,
            **hf_from_pretrained_kwargs(trust_remote_code=True),
        )
        self._semantic_runtime: SemanticDelayRuntime | None = None
        self.reset()
        self.configure_runtime()

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if args:
            destination = args[0]
        if len(args) > 1:
            prefix = args[1]
        if len(args) > 2:
            keep_vars = args[2]
        return prefixed_policy_head_state_dict(
            self,
            prefix=f"{prefix}model.policy_head.",
            destination=destination,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return load_policy_head_state_dict(
            self,
            state_dict,
            strict=strict,
            assign=assign,
        )

    def reset(self):
        self._queues = {ACTION: deque(maxlen=self.config.n_action_steps)}
        self._last_tasks: tuple[str, ...] | None = None
        self.model.latest_vlm_feature = None
        self.model.last_vlm_update_step = 0
        self.model.current_step = 0
        self._last_inference_debug: dict[str, Any] = {}
        if self._semantic_runtime is not None:
            self._semantic_runtime.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _select_device(self) -> torch.device:
        return next(self.parameters()).device

    def get_last_inference_debug(self) -> dict[str, Any]:
        return dict(self._last_inference_debug)

    def _extract_task_texts(self, batch: dict[str, Any]) -> list[str]:
        task = batch.get("task")
        batch_size = self._infer_batch_size(batch)
        if task is None:
            return [""] * batch_size
        if isinstance(task, str):
            return [task]
        if isinstance(task, (list, tuple)):
            return [str(item) for item in task]
        return [str(task)]

    def _extract_semantic_session_id(self, batch: dict[str, Any]) -> str:
        session = batch.get("semantic_session_id", batch.get("session_id", "default"))
        if isinstance(session, torch.Tensor):
            session = session.detach().cpu().reshape(-1)
            if session.numel() == 0:
                return "default"
            return str(int(session[0].item()))
        if isinstance(session, (list, tuple)):
            return "default" if not session else str(session[0])
        return str(session)

    def _infer_batch_size(self, batch: dict[str, Any]) -> int:
        for key in list(self.config.camera_keys) + [OBS_STATE, ACTION]:
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                if value.ndim == 1:
                    return 1
                return int(value.shape[0])
        task = batch.get("task")
        if isinstance(task, list):
            return len(task)
        return 1

    def _normalize_state(self, q_current: Any) -> Tensor:
        device = self._select_device()
        q_current = torch.as_tensor(q_current, dtype=torch.float32, device=device)
        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)
        return q_current[..., : self.config.state_dim]

    def _to_uint8_hwc(self, image: Any) -> np.ndarray:
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        if isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.ndim == 4:
                if tensor.shape[0] != 1:
                    raise ValueError(
                        f"Expected a single image tensor, got shape {tuple(tensor.shape)}"
                    )
                tensor = tensor.squeeze(0)
            if tensor.ndim != 3:
                raise ValueError(f"Expected image tensor with 3 dims, got {tuple(tensor.shape)}")
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            array = tensor.numpy()
        else:
            array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(f"Expected image array with 3 dims, got {array.shape}")

        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        if array.dtype != np.uint8:
            array = (
                np.clip(array, 0.0, 1.0)
                if np.issubdtype(array.dtype, np.floating)
                else np.clip(array, 0, 255)
            )
            scale = 255.0 if np.issubdtype(array.dtype, np.floating) else 1.0
            array = (array * scale).astype(np.uint8) if scale != 1.0 else array.astype(np.uint8)
        return array

    def _gather_camera_images(self, batch: dict[str, Any], camera_key: str) -> list[Image.Image]:
        if camera_key not in batch:
            raise KeyError(
                f"Missing expected camera key '{camera_key}' in batch keys: {list(batch.keys())}"
            )

        value = batch[camera_key]
        if isinstance(value, torch.Tensor):
            if value.ndim == 3:
                value = value.unsqueeze(0)
            if value.ndim != 4:
                raise ValueError(f"Expected image tensor (B,C,H,W), got {tuple(value.shape)}")
            return [Image.fromarray(self._to_uint8_hwc(sample)) for sample in value]

        if isinstance(value, list):
            return [Image.fromarray(self._to_uint8_hwc(sample)) for sample in value]

        return [Image.fromarray(self._to_uint8_hwc(value))]

    def _resize_pil_image(self, image: Image.Image, size: int) -> Image.Image:
        return image.resize((size, size), Image.Resampling.BILINEAR)

    def _build_pixel_values(self, batch: dict[str, Any]) -> Tensor:
        device = self._select_device()
        per_view_batches = []
        for camera_key in self.config.camera_keys:
            pil_images = self._gather_camera_images(batch, camera_key)
            encoded = self.vision_processor(
                images=pil_images,
                size={
                    "height": self.config.vision_image_size,
                    "width": self.config.vision_image_size,
                },
                return_tensors="pt",
            )["pixel_values"]
            per_view_batches.append(encoded)

        pixel_values = torch.stack(per_view_batches, dim=1)
        return pixel_values.to(device=device)

    def _build_vlm_inputs(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, Tensor], Tensor, tuple[str, ...]]:
        device = self._select_device()
        task_texts = self._extract_task_texts(batch)
        images_per_camera = [
            self._gather_camera_images(batch, camera_key) for camera_key in self.config.camera_keys
        ]
        batch_size = len(task_texts)

        image_batches: list[list[Image.Image]] = []
        prompts: list[str] = []
        for batch_idx in range(batch_size):
            sample_images = [
                self._resize_pil_image(
                    images_per_camera[camera_idx][batch_idx], self.config.vlm_image_size
                )
                for camera_idx in range(len(images_per_camera))
            ]
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in sample_images]
                    + [{"type": "text", "text": task_texts[batch_idx]}],
                }
            ]
            prompts.append(
                self.vlm_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )
            image_batches.append(sample_images)

        vlm_inputs = self.vlm_processor(
            text=prompts,
            images=image_batches,
            padding=self.config.tokenizer_padding,
            return_tensors="pt",
        )
        vlm_inputs = {
            key: value.to(device=device) if isinstance(value, torch.Tensor) else value
            for key, value in vlm_inputs.items()
        }
        c_sem_mask = vlm_inputs["attention_mask"].bool()
        return vlm_inputs, c_sem_mask, tuple(task_texts)

    def _prepare_batch_inputs(
        self,
        batch: dict[str, Any],
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor, tuple[str, ...], str]:
        pixel_values = batch.get("pixel_values")
        vlm_inputs = batch.get("vlm_inputs")
        c_sem_mask = batch.get("c_sem_mask")
        if pixel_values is None:
            pixel_values = self._build_pixel_values(batch)
        else:
            pixel_values = pixel_values.to(device=self._select_device())
        if vlm_inputs is None or c_sem_mask is None:
            vlm_inputs, c_sem_mask, task_texts = self._build_vlm_inputs(batch)
        else:
            device = self._select_device()
            vlm_inputs = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for key, value in vlm_inputs.items()
            }
            c_sem_mask = c_sem_mask.to(device=device)
            task_texts = tuple(self._extract_task_texts(batch))

        q_current = self._normalize_state(batch.get("q_current", batch[OBS_STATE]))
        session_id = self._extract_semantic_session_id(batch)
        return pixel_values, vlm_inputs, c_sem_mask, q_current, task_texts, session_id

    def _configured_delay_support(self) -> list[tuple[int, int]] | None:
        delay_max_chunks = getattr(self.config, "delay_max_chunks", None)
        delay_chunk_size_threshold = getattr(self.config, "delay_chunk_size_threshold", None)
        if delay_max_chunks is None and delay_chunk_size_threshold is None:
            return None
        if delay_max_chunks is None or delay_chunk_size_threshold is None:
            raise ValueError(
                "delay_max_chunks and delay_chunk_size_threshold must be configured together."
            )
        chunk_size = getattr(self.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("chunk_size is required when delay support is configured.")
        return _action_step_delay_support(
            chunk_size=int(chunk_size),
            chunk_size_threshold=float(delay_chunk_size_threshold),
            max_delay_chunks=int(delay_max_chunks),
        )

    def _frame_delay_is_supported(self, frame_delay: int, *, max_frame_delay: int) -> bool:
        frame_delay = int(frame_delay)
        if frame_delay < 0 or frame_delay > int(max_frame_delay):
            return False
        support = self._configured_delay_support()
        if support is None:
            return True
        return any(start <= frame_delay < end for start, end in support)

    def _compute_delta_targets(self, action: Tensor, q_current: Tensor) -> Tensor:
        if action.ndim == 2:
            action = action.unsqueeze(1)
        action = action[..., : self.config.action_dim]
        if self.config.action_is_delta:
            state_indices = normalize_action_delta_state_indices(
                self.config.action_delta_state_indices,
                action_dim=self.config.action_dim,
            )
            basis = torch.zeros(
                (q_current.shape[0], self.config.action_dim),
                device=q_current.device,
                dtype=q_current.dtype,
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
            return action - basis[:, None, :]
        return action

    def _compute_loss_mask(self, batch: dict[str, Tensor], action_target: Tensor) -> Tensor:
        if "loss_mask" in batch:
            loss_mask = batch["loss_mask"]
        elif "action_is_pad" in batch:
            loss_mask = (~batch["action_is_pad"]).float()
        elif "actions_is_pad" in batch:
            loss_mask = (~batch["actions_is_pad"]).float()
        else:
            loss_mask = torch.ones(action_target.shape[:2], dtype=torch.float32)
        if loss_mask.ndim == 1:
            loss_mask = loss_mask.unsqueeze(0)
        return loss_mask.to(device=self._select_device(), dtype=torch.float32)

    def _frame_delay_from_batch(self, batch: dict[str, Tensor], batch_size: int) -> Tensor:
        device = self._select_device()
        frame_delay = batch.get("frame_delay")
        if frame_delay is None:
            return torch.zeros(batch_size, device=device, dtype=torch.float32)

        frame_delay = torch.as_tensor(frame_delay, device=device, dtype=torch.float32)
        if frame_delay.ndim == 0:
            return frame_delay.expand(batch_size)

        frame_delay = frame_delay.reshape(-1)
        if frame_delay.numel() == 1:
            return frame_delay.expand(batch_size)
        if frame_delay.numel() != batch_size:
            raise ValueError(
                f"Expected frame_delay to have {batch_size} values, got {frame_delay.numel()}."
            )
        return frame_delay

    def _resolve_action_step(
        self, batch: dict[str, Any], action_step: int | None
    ) -> tuple[int, str]:
        if action_step is not None:
            return int(action_step), "runtime"
        batch_action_step = batch.get("action_step")
        if batch_action_step is not None:
            return int(torch.as_tensor(batch_action_step).reshape(-1)[0].item()), "batch"
        return int(self.model.current_step), "fallback_model_current_step"

    def _encode_semantic_tokens(self, vlm_inputs: dict[str, Tensor]) -> Tensor:
        if self.model.vlm_encoder is None:
            raise RuntimeError("ServoVLA semantic runtime requires a vlm_encoder.")
        with torch.no_grad():
            return run_vlm_encoder_forward(self.model.vlm_encoder, vlm_inputs)

    def configure_runtime(
        self,
        *,
        max_frame_delay: int | None = None,
        semantic_wait_warn_ms: int | None = None,
        semantic_wait_fail_ms: int | None = None,
    ) -> None:
        resolved_max_frame_delay = (
            self.config.max_frame_delay if max_frame_delay is None else int(max_frame_delay)
        )
        resolved_warn_ms = (
            self.config.semantic_wait_warn_ms
            if semantic_wait_warn_ms is None
            else int(semantic_wait_warn_ms)
        )
        resolved_fail_ms = (
            self.config.semantic_wait_fail_ms
            if semantic_wait_fail_ms is None
            else int(semantic_wait_fail_ms)
        )
        self.config.max_frame_delay = int(resolved_max_frame_delay)
        self.config.semantic_wait_warn_ms = int(resolved_warn_ms)
        self.config.semantic_wait_fail_ms = int(resolved_fail_ms)
        if self._semantic_runtime is None:
            self._semantic_runtime = SemanticDelayRuntime(
                encode_semantic=self._encode_semantic_tokens,
                max_frame_delay=self.config.max_frame_delay,
                semantic_wait_warn_ms=self.config.semantic_wait_warn_ms,
                semantic_wait_fail_ms=self.config.semantic_wait_fail_ms,
                logger=logger,
            )
        else:
            self._semantic_runtime.configure(
                max_frame_delay=self.config.max_frame_delay,
                semantic_wait_warn_ms=self.config.semantic_wait_warn_ms,
                semantic_wait_fail_ms=self.config.semantic_wait_fail_ms,
            )

    def shutdown_runtime(self) -> None:
        if self._semantic_runtime is not None:
            self._semantic_runtime.close()
            self._semantic_runtime = None

    def _semantic_runtime_or_raise(self) -> SemanticDelayRuntime:
        if self._semantic_runtime is None:
            self.configure_runtime()
        assert self._semantic_runtime is not None
        return self._semantic_runtime

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        pixel_values, vlm_inputs, c_sem_mask, q_current, _, _ = self._prepare_batch_inputs(batch)
        action = batch[ACTION].to(device=self._select_device(), dtype=torch.float32)
        action_target = self._compute_delta_targets(action, q_current)
        action_normalizer = getattr(self.model, "action_normalizer", None)
        if action_normalizer is not None and bool(getattr(action_normalizer, "enabled", False)):
            action_target = action_normalizer.normalize(action_target)
        loss_mask = self._compute_loss_mask(batch, action_target)

        batch_size, horizon, action_dim = action_target.shape
        x_0 = torch.randn_like(action_target)
        t = torch.rand((batch_size,), device=self._select_device(), dtype=torch.float32)
        t_expanded = t.view(batch_size, 1, 1).expand(batch_size, horizon, action_dim)
        x_t = (1.0 - t_expanded) * x_0 + t_expanded * action_target
        v_true = action_target - x_0

        frame_delay = self._frame_delay_from_batch(batch, batch_size)
        v_pred = self.model(
            x_t=x_t,
            t=t,
            pixel_values=pixel_values,
            vlm_inputs=vlm_inputs,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )
        mse = F.mse_loss(v_pred, v_true, reduction="none").mean(dim=-1)
        loss = (mse * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
        return loss, {"loss": float(loss.item())}

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], *, action_step: int | None = None, **kwargs
    ) -> Tensor:
        self.eval()
        pixel_values, vlm_inputs, c_sem_mask, q_current, task_texts, session_id = (
            self._prepare_batch_inputs(batch)
        )
        runtime = self._semantic_runtime_or_raise()
        admission_policy = (
            str(getattr(self.config, "semantic_admission_policy", "bsr")).strip().lower()
        )

        policy_step_before = int(self.model.current_step)
        current_action_step, action_step_source = self._resolve_action_step(batch, action_step)
        previous_snapshot = runtime.peek_snapshot()
        snapshot_matches_task = (
            previous_snapshot is not None and previous_snapshot.task_texts == task_texts
        )
        snapshot_matches_session = previous_snapshot is not None and str(
            previous_snapshot.session_id
        ) == str(session_id)
        last_vlm_update_before = (
            -1 if previous_snapshot is None else int(previous_snapshot.frame_id)
        )
        task_changed = self._last_tasks != task_texts
        action_step_went_backwards = (
            snapshot_matches_task
            and snapshot_matches_session
            and int(previous_snapshot.frame_id) > int(current_action_step)
        )
        step_delay = (
            None
            if previous_snapshot is None
            else int(current_action_step) - int(previous_snapshot.frame_id)
        )
        delay_support = self._configured_delay_support()
        delay_is_supported = None
        if delay_support is not None:

            def delay_is_supported(delay):
                return self._frame_delay_is_supported(
                    delay,
                    max_frame_delay=int(runtime.max_frame_delay),
                )

        if step_delay is None:
            refresh_due = False
            delay_outside_supported_range = False
        elif delay_support is None:
            refresh_due = False
            delay_outside_supported_range = snapshot_matches_task and step_delay > int(
                runtime.max_frame_delay
            )
        else:
            refresh_due = False
            delay_outside_supported_range = (
                snapshot_matches_task
                and not self._frame_delay_is_supported(
                    step_delay,
                    max_frame_delay=int(runtime.max_frame_delay),
                )
            )
        has_request_for_frame_delay = getattr(runtime, "has_request_for_frame_delay", None)
        has_current_request = False
        if callable(has_request_for_frame_delay):
            has_current_request = bool(
                has_request_for_frame_delay(
                    frame_id=current_action_step,
                    task_texts=task_texts,
                    session_id=session_id,
                    delay_is_supported=lambda delay: int(delay) == 0,
                )
            )
        snapshot_is_current = (
            snapshot_matches_task
            and snapshot_matches_session
            and previous_snapshot is not None
            and (int(previous_snapshot.frame_id) == int(current_action_step))
        )
        pending_refresh_will_be_supported = False
        if (
            delay_outside_supported_range
            and delay_is_supported is not None
            and callable(has_request_for_frame_delay)
        ):
            pending_refresh_will_be_supported = bool(
                has_request_for_frame_delay(
                    frame_id=current_action_step,
                    task_texts=task_texts,
                    session_id=session_id,
                    delay_is_supported=delay_is_supported,
                )
            )
        if admission_policy in {"latest_cache", "age_only"}:
            should_submit_refresh = True
        elif delay_support is None:
            should_submit_refresh = (
                task_changed
                or previous_snapshot is None
                or not snapshot_matches_task
                or not snapshot_matches_session
                or action_step_went_backwards
                or refresh_due
                or delay_outside_supported_range
            )
        else:
            should_submit_refresh = not snapshot_is_current and not has_current_request
        if should_submit_refresh:
            runtime.submit_latest(
                frame_id=current_action_step,
                task_texts=task_texts,
                session_id=session_id,
                vlm_inputs=vlm_inputs,
                c_sem_mask=c_sem_mask,
            )

        with torch.no_grad():
            f_vision = self.model.vision_encoder(pixel_values)

        snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(
            frame_id=current_action_step,
            task_texts=task_texts,
            session_id=session_id,
            admission_policy=admission_policy,
            delay_is_supported=delay_is_supported,
        )
        admission_debug = runtime.get_last_admission_debug()
        frame_delay_tensor = torch.full(
            (f_vision.shape[0],),
            float(frame_delay),
            device=f_vision.device,
            dtype=torch.float32,
        )
        policy_dtype = next(self.model.policy_head.parameters()).dtype
        noise = make_inference_noise(
            shape=(
                q_current.shape[0],
                self.model.fm_solver.chunk_size,
                self.model.fm_solver.action_dim,
            ),
            device=f_vision.device,
            dtype=policy_dtype,
            seed=getattr(self.config, "inference_noise_seed", None),
            seed_mode=getattr(self.config, "inference_noise_seed_mode", "step"),
            action_step=current_action_step,
        )
        action_chunk = self.model.sample_action_chunk_from_features(
            f_vision=f_vision,
            c_sem=snapshot.c_sem,
            c_sem_mask=snapshot.c_sem_mask,
            frame_delay=frame_delay_tensor,
            q_current=q_current,
            noise=noise,
        )
        self.model.latest_vlm_feature = snapshot.c_sem
        self.model.last_vlm_update_step = int(snapshot.frame_id)
        self.model.current_step = policy_step_before + 1
        self._last_tasks = task_texts

        action_chunk_size = int(
            getattr(self.config, "chunk_size", getattr(self.config, "n_action_steps", 1))
        )
        self._last_inference_debug = {
            "task_texts": list(task_texts),
            "semantic_session_id": session_id,
            "task_changed": bool(task_changed),
            "vlm_refreshed": int(snapshot.frame_id) != last_vlm_update_before,
            "current_action_step": int(current_action_step),
            "semantic_snapshot_action_step": int(snapshot.frame_id),
            "semantic_snapshot_task_texts": list(snapshot.task_texts),
            "semantic_snapshot_session_id": str(snapshot.session_id),
            "step_delay": int(frame_delay),
            "action_step_source": action_step_source,
            "semantic_admission_policy": admission_policy,
            "admit_reason": admission_debug.get("admit_reason"),
            "reject_reason": admission_debug.get("reject_reason"),
            "semantic_sample_mode": "sync_aligned" if int(frame_delay) == 0 else "async_stale",
            "action_chunk_size": action_chunk_size,
            "chunk_size_threshold": (
                None
                if getattr(self.config, "delay_chunk_size_threshold", None) is None
                else float(self.config.delay_chunk_size_threshold)
            ),
            "delay_max_chunks": (
                None
                if getattr(self.config, "delay_max_chunks", None) is None
                else int(self.config.delay_max_chunks)
            ),
            "policy_step_before": policy_step_before,
            "policy_step_after": int(self.model.current_step),
            "last_vlm_update_step_before": last_vlm_update_before,
            "last_vlm_update_step_after": int(snapshot.frame_id),
            "frame_delay": int(frame_delay),
            "max_frame_delay": int(runtime.max_frame_delay),
            "semantic_wait_ms": int(waited_ms),
            "waited_for_semantic": bool(waited_ms > 0),
            "pending_refresh_will_be_supported": bool(pending_refresh_will_be_supported),
            "vlm_prefetch_submitted": bool(should_submit_refresh),
            "vlm_prefetch_frame_id": int(current_action_step) if should_submit_refresh else None,
            "vlm_latest_submitted": bool(should_submit_refresh),
            "vlm_latest_frame_id": int(current_action_step) if should_submit_refresh else None,
            "actions_returned": int(self.config.n_action_steps),
        }
        logger.info(
            "Inference frame diff | current_action_step=%d | semantic_action_step=%d | step_delay=%d | action_step_source=%s | policy_step_before=%d | policy_step_after=%d | last_vlm_before=%d | last_vlm_after=%d | frame_delay=%d | max_frame_delay=%d | waited_for_semantic=%s | semantic_wait_ms=%d | vlm_refreshed=%s | vlm_latest_submitted=%s | vlm_latest_frame_id=%s | pending_refresh_will_be_supported=%s | task_changed=%s",
            self._last_inference_debug["current_action_step"],
            self._last_inference_debug["semantic_snapshot_action_step"],
            self._last_inference_debug["step_delay"],
            self._last_inference_debug["action_step_source"],
            self._last_inference_debug["policy_step_before"],
            self._last_inference_debug["policy_step_after"],
            self._last_inference_debug["last_vlm_update_step_before"],
            self._last_inference_debug["last_vlm_update_step_after"],
            self._last_inference_debug["frame_delay"],
            self._last_inference_debug["max_frame_delay"],
            self._last_inference_debug["waited_for_semantic"],
            self._last_inference_debug["semantic_wait_ms"],
            self._last_inference_debug["vlm_refreshed"],
            self._last_inference_debug["vlm_latest_submitted"],
            self._last_inference_debug["vlm_latest_frame_id"],
            self._last_inference_debug["pending_refresh_will_be_supported"],
            self._last_inference_debug["task_changed"],
        )
        return action_chunk[:, : self.config.n_action_steps, : self.config.action_dim]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        self.eval()
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        return self._queues[ACTION].popleft()
