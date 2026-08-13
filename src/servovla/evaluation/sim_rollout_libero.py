from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from PIL import Image
from transformers import AutoImageProcessor, AutoProcessor

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import add_envs_task, close_envs, preprocess_observation
from lerobot.processor.env_processor import LiberoProcessorStep
from lerobot.utils.constants import ACTION

from lerobot_policy_servovla import ServoVLAPolicy
from servovla.architectures.action_normalization import ActionNormalizer
from servovla.architectures.fm_solver import FlowMatchingEulerSolver
from servovla.architectures.policy_head import build_policy_head_from_cfg
from servovla.architectures.policy_head_checkpoint import load_policy_head_state_dict
from servovla.architectures.servo_vla import ServoVLA
from servovla.architectures.vision_encoder import VisionEncoder
from servovla.architectures.vlm_encoder import VLMEncoder
from servovla.config.action_mode import action_mode_is_delta, normalize_action_delta_state_indices
from servovla.config.hf_offline import hf_from_pretrained_kwargs
from servovla.config.paths import resolve_project_path
from servovla.data.sim_libero_adapter import load_libero_protocol
from servovla.deployment.deploy_utils import find_latest_checkpoint

log = logging.getLogger(__name__)


def _empty_runtime_metrics() -> dict[str, list[float]]:
    return {
        "chunk_latency_ms": [],
        "observed_frame_delay": [],
        "semantic_wait_ms": [],
    }


def _extend_runtime_metrics(dst: dict[str, list[float]], src: dict[str, list[float]]) -> None:
    for key, values in src.items():
        dst.setdefault(key, []).extend(float(value) for value in values)


def _append_jsonl(path: str | Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = resolve_project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _p95_or_none(values: list[float]) -> float | None:
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95)) if values else None


def _duration_steps_to_seconds(durations: dict[str, float], fps: float) -> dict[str, float]:
    if fps <= 0:
        raise ValueError(f"LIBERO duration FPS must be positive, got {fps}")
    return {
        task_name: float(duration_steps) / float(fps)
        for task_name, duration_steps in durations.items()
    }


def summarize_runtime_metrics(metrics: dict[str, list[float]]) -> dict[str, Any]:
    chunk_latencies = list(metrics.get("chunk_latency_ms") or [])
    frame_delays = list(metrics.get("observed_frame_delay") or [])
    semantic_waits = list(metrics.get("semantic_wait_ms") or [])
    return {
        "mean_chunk_latency_ms": _mean_or_none(chunk_latencies),
        "p95_chunk_latency_ms": _p95_or_none(chunk_latencies),
        "max_observed_frame_delay": int(max(frame_delays)) if frame_delays else None,
        "semantic_waittime_mean": _mean_or_none(semantic_waits),
        "semantic_waittime_p95": _p95_or_none(semantic_waits),
        "chunk_latency_sample_count": len(chunk_latencies),
        "semantic_wait_sample_count": len(semantic_waits),
    }


def _action_queue_len(policy: Any) -> int | None:
    target = getattr(policy, "policy", policy)
    queues = getattr(target, "_queues", None)
    if not isinstance(queues, dict) or ACTION not in queues:
        return None
    try:
        return len(queues[ACTION])
    except TypeError:
        return None


def _last_inference_debug(policy: Any) -> dict[str, Any]:
    target = getattr(policy, "policy", policy)
    getter = getattr(target, "get_last_inference_debug", None)
    if not callable(getter):
        return {}
    debug = getter()
    return debug if isinstance(debug, dict) else {}


def close_rollout_env(env: Any) -> None:
    try:
        close_envs(env)
    except NotImplementedError:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def default_sim_libero_eval_cfg_path() -> Path:
    return _PROJECT_ROOT / "configs" / "eval" / "sim_libero.yaml"


def resolve_checkpoint_path(run_dir: str | Path, checkpoint: str | Path | None) -> Path:
    if checkpoint is not None:
        return resolve_project_path(checkpoint)
    return find_latest_checkpoint(run_dir)


def _normalize_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        clean_key = key
        if clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        clean_key = clean_key.replace("._orig_mod", "")
        normalized[clean_key] = value
    return normalized


def _select_state_dict(
    checkpoint_path: Path, *, prefer_ema: bool, checkpoint: Any | None = None
) -> tuple[dict[str, torch.Tensor], bool]:
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and prefer_ema and "ema_state" in checkpoint:
        raw_state = checkpoint["ema_state"]
        used_ema = True
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        raw_state = checkpoint["model_state"]
        used_ema = False
    else:
        raw_state = checkpoint
        used_ema = False

    if not isinstance(raw_state, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict.")
    return _normalize_checkpoint_keys(raw_state), used_ema


def _checkpoint_action_normalizer(
    checkpoint: Any, *, checkpoint_path: Path, action_dim: int, chunk_size: int
) -> ActionNormalizer:
    """Build the inference normalizer saved with a raw training checkpoint."""
    stats = checkpoint.get("action_normalization") if isinstance(checkpoint, dict) else None
    if not isinstance(stats, dict) or not bool(stats.get("enabled", False)):
        return ActionNormalizer(enabled=False)

    mean = stats.get("mean")
    std = stats.get("std")
    if mean is None or std is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path} enables action normalization but lacks mean/std stats."
        )
    normalizer = ActionNormalizer(
        enabled=True,
        mean=mean,
        std=std,
        eps=float(stats.get("eps", 1.0e-6)),
    )
    if normalizer.action_dim != action_dim:
        raise ValueError(
            f"Checkpoint action normalization dim {normalizer.action_dim} does not match "
            f"policy action dim {action_dim}."
        )
    if normalizer.mean.ndim == 2 and normalizer.mean.shape[0] < chunk_size:
        raise ValueError(
            f"Checkpoint action normalization horizon {normalizer.mean.shape[0]} is shorter "
            f"than policy chunk_size {chunk_size}."
        )
    return normalizer


def _infer_policy_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    try:
        action_dim = int(state_dict["policy_head.final_layer.1.weight"].shape[0])
        state_dim = int(state_dict["policy_head.state_proj.0.weight"].shape[1])
    except KeyError as exc:
        raise KeyError("Unable to infer action/state dimensions from checkpoint.") from exc

    num_cameras = 1
    if "policy_head.view_embed.weight" in state_dict:
        num_cameras = max(1, int(state_dict["policy_head.view_embed.weight"].shape[0]))
    return action_dim, state_dim, num_cameras


def _validate_libero_policy_dims(action_dim: int, state_dim: int, num_cameras: int) -> None:
    expected = (7, 8, 2)
    actual = (int(action_dim), int(state_dim), int(num_cameras))
    if actual != expected:
        raise ValueError(
            "LIBERO rollout requires action_dim=7, state_dim=8, and num_cameras=2; "
            f"checkpoint provides action_dim={actual[0]}, state_dim={actual[1]}, "
            f"num_cameras={actual[2]}. Use a checkpoint trained with dataset=sim_libero."
        )


def _validate_libero_run_config(cfg: Any) -> None:
    benchmark = str(cfg.dataset.get("benchmark", ""))
    action_mode = str(cfg.dataset.get("action_mode", ""))
    if benchmark != "libero_40" or action_mode != "abs":
        raise ValueError(
            "LIBERO rollout requires a run trained with dataset=sim_libero "
            f"(benchmark=libero_40, action_mode=abs); got benchmark={benchmark!r}, "
            f"action_mode={action_mode!r}."
        )


class ServoVLALiberoRolloutPolicy:
    def __init__(
        self,
        *,
        model: ServoVLA,
        vision_processor,
        vlm_processor,
        device: torch.device,
        image_size: int,
        seed: int,
        camera_keys: tuple[str, ...],
    ) -> None:
        self.model = model.eval().to(device)
        self.vision_processor = vision_processor
        self.vlm_processor = vlm_processor
        self.device = device
        self.image_size = int(image_size)
        self.camera_keys = tuple(camera_keys)
        self.policy_dtype = next(self.model.policy_head.parameters()).dtype
        self.generator = torch.Generator(device=str(device))
        self.generator.manual_seed(int(seed))

    def _tensor_to_pil(self, image_tensor: torch.Tensor) -> Image.Image:
        image = image_tensor.detach().cpu()
        if image.ndim != 3:
            raise ValueError(f"Expected rank-3 image tensor, got {tuple(image.shape)}")
        if image.shape[0] in {1, 3}:
            image = image.permute(1, 2, 0)
        image = image.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
        return Image.fromarray(image).resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )

    def _pad_vlm_inputs(self, vlm_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = vlm_inputs["input_ids"]
        actual_len = input_ids.shape[1]
        target_len = (actual_len + 7) // 8 * 8
        pad_len = target_len - actual_len
        if pad_len <= 0:
            return vlm_inputs

        pad_token_id = self.vlm_processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.vlm_processor.tokenizer.eos_token_id

        padded = dict(vlm_inputs)
        for key, value in list(padded.items()):
            if not torch.is_tensor(value) or value.ndim != 2:
                continue
            if value.shape[1] != actual_len:
                continue
            pad_value = pad_token_id if key == "input_ids" else 0
            padded[key] = F.pad(value, (0, pad_len), value=pad_value)
        return padded

    @torch.inference_mode()
    def select_action(self, observation: dict[str, Any]) -> np.ndarray:
        tasks_value = observation["task"]
        tasks = (
            [tasks_value] if isinstance(tasks_value, str) else [str(task) for task in tasks_value]
        )

        per_camera_batches = []
        image_groups: list[list[Image.Image]] = [[] for _ in range(len(tasks))]
        for camera_key in self.camera_keys:
            image_batch = observation[camera_key]
            if image_batch.ndim == 3:
                image_batch = image_batch.unsqueeze(0)
            pil_images = [
                self._tensor_to_pil(image_batch[idx]) for idx in range(image_batch.shape[0])
            ]
            processed = self.vision_processor(images=pil_images, return_tensors="pt")[
                "pixel_values"
            ]
            per_camera_batches.append(processed)
            for idx, image in enumerate(pil_images):
                image_groups[idx].append(image)

        vision_inputs = torch.stack(per_camera_batches, dim=1).to(self.device)

        prompts = []
        for task in tasks:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in self.camera_keys]
                    + [{"type": "text", "text": task}],
                }
            ]
            prompts.append(
                self.vlm_processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        vlm_inputs = self.vlm_processor(
            text=prompts,
            images=image_groups,
            padding=True,
            return_tensors="pt",
        )
        vlm_inputs = self._pad_vlm_inputs(vlm_inputs)
        vlm_inputs = {key: value.to(self.device) for key, value in vlm_inputs.items()}
        c_sem_mask = vlm_inputs["attention_mask"].bool()

        q_current = observation["observation.state"]
        if q_current.ndim == 1:
            q_current = q_current.unsqueeze(0)
        q_current = q_current.to(device=self.device, dtype=self.policy_dtype)

        f_vision = self.model.vision_encoder(vision_inputs).to(dtype=self.policy_dtype)
        c_sem = self.model.vlm_encoder(vlm_inputs).to(dtype=self.policy_dtype)
        frame_delay = torch.zeros((q_current.shape[0],), device=self.device, dtype=torch.float32)
        noise = torch.randn(
            (q_current.shape[0], self.model.fm_solver.chunk_size, self.model.fm_solver.action_dim),
            generator=self.generator,
            device=self.device,
            dtype=self.policy_dtype,
        )
        pred = self.model.sample_action_chunk_from_features(
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
            noise=noise,
        )

        action = pred[:, 0, :].detach().cpu().float().numpy()
        return np.clip(action, -1.0, 1.0)


class ExportedServoVLALiberoRolloutPolicy:
    def __init__(self, *, policy: ServoVLAPolicy, device: torch.device) -> None:
        self.policy = policy.eval().to(device)
        self.device = device

    def _runtime_action_step(self, env_step: int) -> int:
        chunk_size = max(int(getattr(self.policy.config, "chunk_size", 1)), 1)
        chunk_index = max(int(env_step), 0) // chunk_size
        return chunk_index * max(chunk_size - 1, 1)

    @torch.inference_mode()
    def select_action(
        self, observation: dict[str, Any], *, action_step: int | None = None
    ) -> np.ndarray:
        batch: dict[str, Any] = {}
        tasks_value = observation["task"]
        batch["task"] = (
            [tasks_value] if isinstance(tasks_value, str) else [str(task) for task in tasks_value]
        )
        batch["observation.state"] = observation["observation.state"].to(self.device)
        if action_step is not None:
            batch["action_step"] = torch.tensor(
                self._runtime_action_step(int(action_step)), device=self.device
            )

        for camera_key in self.policy.config.camera_keys:
            if camera_key not in observation:
                raise KeyError(f"Missing expected camera key {camera_key!r} in LIBERO observation.")
            batch[camera_key] = observation[camera_key]

        action = self.policy.select_action(batch).detach().cpu().float().numpy()
        return np.clip(action, -1.0, 1.0)

    def reset(self) -> None:
        self.policy.reset()


def build_servovla_libero_policy(
    run_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str,
    prefer_ema: bool,
    seed: int,
) -> tuple[ServoVLALiberoRolloutPolicy, Any, bool]:
    run_dir = resolve_project_path(run_dir)
    checkpoint_path = resolve_project_path(checkpoint_path)
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Could not find Hydra config at {hydra_cfg_path}")

    cfg = OmegaConf.load(hydra_cfg_path)
    _validate_libero_run_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict, used_ema = _select_state_dict(
        checkpoint_path,
        prefer_ema=prefer_ema,
        checkpoint=checkpoint,
    )
    action_dim, state_dim, num_cameras = _infer_policy_dims(state_dict)
    _validate_libero_policy_dims(action_dim, state_dim, num_cameras)

    with open_dict(cfg):
        cfg.model.policy_head.action_dim = action_dim
        cfg.model.policy_head.state_dim = state_dim
        cfg.model.policy_head.num_cameras = num_cameras
        cfg.dataset.num_cameras = num_cameras
        cfg.dataset.camera_keys = ["observation.images.image", "observation.images.image2"]

    torch_device = torch.device(device)
    vision_encoder = VisionEncoder(
        model_id=cfg.model.vision_encoder.model_id,
        attn_implementation=cfg.model.vision_encoder.get("attn_implementation", None),
    ).to(torch_device)
    vlm_encoder = VLMEncoder(
        model_id=cfg.model.vlm_encoder.model_id,
        attn_implementation=cfg.model.vlm_encoder.get("attn_implementation", None),
    ).to(torch_device)
    policy_head = build_policy_head_from_cfg(cfg).to(torch_device)
    fm_solver = FlowMatchingEulerSolver(
        action_dim=action_dim,
        chunk_size=int(cfg.model.policy_head.chunk_size),
        num_inference_steps=int(cfg.model.policy_head.num_inference_steps),
    )
    action_normalizer = _checkpoint_action_normalizer(
        checkpoint,
        checkpoint_path=checkpoint_path,
        action_dim=action_dim,
        chunk_size=int(cfg.model.policy_head.chunk_size),
    )
    model = ServoVLA(
        vision_encoder=vision_encoder,
        vlm_encoder=vlm_encoder,
        policy_head=policy_head,
        fm_solver=fm_solver,
        action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
        action_delta_state_indices=normalize_action_delta_state_indices(
            cfg.dataset.get("action_delta_state_indices", None),
            action_dim=action_dim,
        ),
        action_normalizer=action_normalizer,
    ).to(torch_device)

    missing, unexpected = load_policy_head_state_dict(model, state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing checkpoint keys for LIBERO sim policy: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys for LIBERO sim policy: {unexpected}")

    vision_processor = AutoImageProcessor.from_pretrained(
        cfg.model.vision_encoder.model_id,
        **hf_from_pretrained_kwargs(),
    )
    vlm_processor = AutoProcessor.from_pretrained(
        cfg.model.vlm_encoder.model_id,
        **hf_from_pretrained_kwargs(trust_remote_code=True),
    )
    if not hasattr(vlm_processor, "pad_token") or vlm_processor.pad_token is None:
        vlm_processor.tokenizer.pad_token = vlm_processor.tokenizer.eos_token

    return (
        ServoVLALiberoRolloutPolicy(
            model=model,
            vision_processor=vision_processor,
            vlm_processor=vlm_processor,
            device=torch_device,
            image_size=int(cfg.model.vision_encoder.get("image_size", 256)),
            seed=seed,
            camera_keys=("observation.images.image", "observation.images.image2"),
        ),
        cfg,
        used_ema,
    )


def build_exported_servovla_libero_policy(
    pretrained_dir: str | Path,
    *,
    device: str,
    semantic_wait_fail_ms: int | None = None,
    delay_chunk_size_threshold: float | None = None,
    max_frame_delay: int | None = None,
    semantic_admission_policy: str | None = None,
    inference_noise_seed: int | None = None,
    inference_noise_seed_mode: str | None = None,
) -> ExportedServoVLALiberoRolloutPolicy:
    pretrained_dir = resolve_project_path(pretrained_dir)
    policy = ServoVLAPolicy.from_pretrained(pretrained_dir)
    if delay_chunk_size_threshold is not None:
        policy.config.delay_chunk_size_threshold = float(delay_chunk_size_threshold)
        if getattr(policy.config, "delay_max_chunks", None) is not None:
            policy.config.max_frame_delay = None
        post_init = getattr(policy.config, "__post_init__", None)
        if callable(post_init):
            post_init()
    if max_frame_delay is not None:
        policy.config.max_frame_delay = int(max_frame_delay)
    if semantic_admission_policy is not None:
        policy.config.semantic_admission_policy = str(semantic_admission_policy).strip().lower()
    if inference_noise_seed is not None:
        policy.config.inference_noise_seed = int(inference_noise_seed)
    if inference_noise_seed_mode is not None:
        policy.config.inference_noise_seed_mode = str(inference_noise_seed_mode)
    if semantic_wait_fail_ms is not None or max_frame_delay is not None:
        fail_ms = (
            int(policy.config.semantic_wait_fail_ms)
            if semantic_wait_fail_ms is None
            else int(semantic_wait_fail_ms)
        )
        configured_max_frame_delay = getattr(policy.config, "max_frame_delay", None)
        policy.configure_runtime(
            max_frame_delay=(
                None if configured_max_frame_delay is None else int(configured_max_frame_delay)
            ),
            semantic_wait_warn_ms=min(int(policy.config.semantic_wait_warn_ms), fail_ms),
            semantic_wait_fail_ms=fail_ms,
        )
    return ExportedServoVLALiberoRolloutPolicy(policy=policy, device=torch.device(device))


def summarize_libero_rollouts(
    task_successes: dict[str, list[bool]],
    task_lengths: dict[str, list[int]] | None = None,
    *,
    top_failure_count: int = 10,
    duration_fps: float | None = None,
) -> dict[str, Any]:
    per_task_success_rate = {
        task_name: float(sum(bool(item) for item in successes) / len(successes))
        for task_name, successes in task_successes.items()
        if successes
    }
    overall_balanced_success_rate = (
        float(sum(per_task_success_rate.values()) / len(per_task_success_rate))
        if per_task_success_rate
        else 0.0
    )

    per_task_episode_length = {}
    if task_lengths is not None:
        per_task_episode_length = {
            task_name: float(sum(lengths) / len(lengths))
            for task_name, lengths in task_lengths.items()
            if lengths
        }
    resolved_duration_fps = float(
        duration_fps if duration_fps is not None else getattr(LiberoEnv, "fps", 30)
    )
    per_task_duration_seconds = _duration_steps_to_seconds(
        per_task_episode_length, resolved_duration_fps
    )

    failure_order = sorted(per_task_success_rate.items(), key=lambda item: (item[1], item[0]))[
        : max(0, int(top_failure_count))
    ]

    return {
        "overall_balanced_success_rate": overall_balanced_success_rate,
        "per_task_success_rate": per_task_success_rate,
        "per_task_episode_length": per_task_episode_length,
        # Alias the rollout lengths as durations for paper/result aggregation.
        # Step values are simulator control steps, averaged over episodes for each task.
        # Second values are simulated/control time, not wall-clock runtime.
        "duration_fps": resolved_duration_fps,
        "per_task_duration_steps": per_task_episode_length,
        "per_task_duration_seconds": per_task_duration_seconds,
        "top_failures": [
            {
                "task": task_name,
                "success_rate": success_rate,
                "mean_episode_length": per_task_episode_length.get(task_name),
                "mean_duration_steps": per_task_episode_length.get(task_name),
                "mean_duration_seconds": per_task_duration_seconds.get(task_name),
            }
            for task_name, success_rate in failure_order
        ],
    }


def _preprocess_libero_observation(observation: dict[str, Any]) -> dict[str, Any]:
    processor = LiberoProcessorStep()
    return processor.observation(preprocess_observation(observation))


def _rollout_task(
    *,
    suite: str,
    task_spec,
    policy: ServoVLALiberoRolloutPolicy,
    obs_type: str,
    render_mode: str,
    camera_name: str,
    control_mode: str = "relative",
    n_episodes: int,
    use_async_envs: bool,
    seed: int,
    replan_every_step: bool = False,
    config_id: str | None = None,
    run_id: str | None = None,
    code_revision: str | None = None,
    checkpoint_id: str | None = None,
    runtime_event_jsonl: str | Path | None = None,
) -> tuple[list[bool], list[int], dict[str, list[float]]]:
    resolved_suite = str(getattr(task_spec, "suite", suite))
    env_cfg = LiberoEnv(
        task=resolved_suite,
        task_ids=[int(task_spec.suite_task_id)],
        obs_type=obs_type,
        render_mode=render_mode,
        camera_name=camera_name,
        control_mode=str(control_mode),
    )
    env_groups = make_env(env_cfg, n_envs=n_episodes, use_async_envs=use_async_envs)
    env = env_groups[resolved_suite][int(task_spec.suite_task_id)]

    done = np.zeros(env.num_envs, dtype=bool)
    successes = [False] * env.num_envs
    lengths = [0] * env.num_envs
    runtime_metrics = _empty_runtime_metrics()

    try:
        observation, _ = env.reset(seed=[seed + idx for idx in range(env.num_envs)])
        max_steps = int(env.call("_max_episode_steps")[0])
        step = 0
        while not np.all(done) and step < max_steps:
            policy_observation = _preprocess_libero_observation(observation)
            policy_observation = add_envs_task(env, policy_observation)
            session_id = f"{resolved_suite}:{int(task_spec.suite_task_id)}:seed{int(seed)}"
            policy_observation["semantic_session_id"] = session_id
            if bool(replan_every_step):
                reset = getattr(policy, "reset", None)
                if callable(reset):
                    reset()
            queue_len_before = _action_queue_len(policy)
            records_new_chunk = queue_len_before is None or queue_len_before == 0
            started_at = time.perf_counter()
            try:
                action = policy.select_action(policy_observation, action_step=step)
            except TypeError:
                action = policy.select_action(policy_observation)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            if records_new_chunk:
                runtime_metrics["chunk_latency_ms"].append(float(elapsed_ms))
                debug = _last_inference_debug(policy)
                frame_delay = debug.get("frame_delay", debug.get("step_delay"))
                if frame_delay is not None:
                    runtime_metrics["observed_frame_delay"].append(float(frame_delay))
                semantic_wait_ms = debug.get("semantic_wait_ms")
                if semantic_wait_ms is not None:
                    runtime_metrics["semantic_wait_ms"].append(float(semantic_wait_ms))
                _append_jsonl(
                    runtime_event_jsonl,
                    {
                        "run_id": run_id,
                        "code_revision": code_revision,
                        "config_id": config_id,
                        "checkpoint_id": checkpoint_id,
                        "seed": int(seed),
                        "suite": resolved_suite,
                        "task_id": int(task_spec.suite_task_id),
                        "task_name": str(getattr(task_spec, "task_name", task_spec.suite_task_id)),
                        "episode_id": None,
                        "action_step": int(step),
                        "source_action_step": debug.get("semantic_snapshot_action_step"),
                        "semantic_age": frame_delay,
                        "semantic_admission_policy": debug.get("semantic_admission_policy"),
                        "task_identity_match": (
                            debug.get("task_texts") == debug.get("semantic_snapshot_task_texts")
                        ),
                        "session_generation_match": (
                            str(debug.get("semantic_session_id"))
                            == str(debug.get("semantic_snapshot_session_id"))
                        ),
                        "ready_event_complete": True,
                        "admitted": True,
                        "admit_reason": debug.get("admit_reason"),
                        "reject_reason": debug.get("reject_reason"),
                        "semantic_wait_ms": semantic_wait_ms,
                        "chunk_latency_ms": float(elapsed_ms),
                        "timeout": False,
                        "action_queue_underflow": records_new_chunk,
                        "episode_success": None,
                    },
                )
            observation, _, terminated, truncated, info = env.step(action)

            finished_now = (~done) & (terminated | truncated)
            final_success = np.zeros(env.num_envs, dtype=bool)
            if (
                "final_info" in info
                and isinstance(info["final_info"], dict)
                and "is_success" in info["final_info"]
            ):
                final_success = np.asarray(info["final_info"]["is_success"]).astype(bool)

            for env_idx in np.where(finished_now)[0]:
                successes[env_idx] = bool(final_success[env_idx])
                lengths[env_idx] = step + 1

            done |= terminated | truncated
            step += 1

        for env_idx in np.where(~done)[0]:
            lengths[env_idx] = step
        return successes, lengths, runtime_metrics
    finally:
        close_rollout_env(env)


def evaluate_libero_run(
    run_dir: str | Path,
    checkpoint: str | Path | None = None,
    eval_cfg_path: str | Path | None = None,
    device: str | None = None,
    max_tasks: int | None = None,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    run_dir = resolve_project_path(run_dir)
    resolved_eval_cfg_path = (
        resolve_project_path(eval_cfg_path)
        if eval_cfg_path is not None
        else default_sim_libero_eval_cfg_path()
    )
    eval_cfg = OmegaConf.load(resolved_eval_cfg_path)
    if device is not None:
        eval_cfg.device = device
    if max_tasks is not None:
        eval_cfg.max_tasks = int(max_tasks)

    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    protocol_path = resolve_project_path(eval_cfg.protocol_path)
    protocol = load_libero_protocol(protocol_path)
    task_specs = list(protocol.tasks)
    if eval_cfg.get("max_tasks") is not None:
        task_specs = task_specs[: int(eval_cfg.max_tasks)]

    policy, run_cfg, used_ema = build_servovla_libero_policy(
        run_dir,
        checkpoint_path,
        device=str(eval_cfg.device),
        prefer_ema=bool(eval_cfg.get("prefer_ema", True)),
        seed=int(eval_cfg.seed),
    )

    task_successes: dict[str, list[bool]] = {}
    task_lengths: dict[str, list[int]] = {}
    runtime_metrics = _empty_runtime_metrics()
    for task_spec in task_specs:
        log.info("Running LIBERO rollout evaluation for task %s", task_spec.task_name)
        successes, lengths, task_metrics = _rollout_task(
            suite=str(getattr(task_spec, "suite", eval_cfg.suite)),
            task_spec=task_spec,
            policy=policy,
            obs_type=str(eval_cfg.obs_type),
            render_mode=str(eval_cfg.render_mode),
            camera_name=str(eval_cfg.camera_name),
            control_mode=str(eval_cfg.get("control_mode", "relative")),
            n_episodes=int(eval_cfg.n_episodes_per_task),
            use_async_envs=bool(eval_cfg.get("use_async_envs", False)),
            seed=int(eval_cfg.seed),
            replan_every_step=bool(eval_cfg.get("replan_every_step", False)),
        )
        task_successes[task_spec.language] = successes
        task_lengths[task_spec.language] = lengths
        _extend_runtime_metrics(runtime_metrics, task_metrics)

    summary = summarize_libero_rollouts(
        task_successes, task_lengths, top_failure_count=int(eval_cfg.get("top_failure_count", 10))
    )
    runtime_summary = summarize_runtime_metrics(runtime_metrics)
    result = {
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "used_ema": used_ema,
        "eval_cfg_path": str(resolved_eval_cfg_path),
        "protocol_path": str(protocol_path),
        "suite": str(eval_cfg.suite),
        "suites": list(protocol.suites)
        if protocol.suites
        else sorted({task.suite for task in task_specs}),
        "control_mode": str(eval_cfg.get("control_mode", "relative")),
        "replan_every_step": bool(eval_cfg.get("replan_every_step", False)),
        "task_count": len(task_specs),
        "n_episodes_per_task": int(eval_cfg.n_episodes_per_task),
        "action_mode": str(run_cfg.dataset.action_mode),
        **runtime_summary,
        **summary,
    }

    if output_json is not None:
        output_path = resolve_project_path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return result


def _load_export_metadata(pretrained_dir: Path) -> dict[str, Any]:
    metadata_path = pretrained_dir / "export_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata if isinstance(metadata, dict) else {}


def evaluate_libero_pretrained(
    pretrained_dir: str | Path,
    eval_cfg_path: str | Path | None = None,
    device: str | None = None,
    max_tasks: int | None = None,
    output_json: str | Path | None = None,
    semantic_wait_fail_ms: int | None = 60_000,
    replan_every_step: bool | None = None,
) -> dict[str, Any]:
    pretrained_dir = resolve_project_path(pretrained_dir)
    resolved_eval_cfg_path = (
        resolve_project_path(eval_cfg_path)
        if eval_cfg_path is not None
        else default_sim_libero_eval_cfg_path()
    )
    eval_cfg = OmegaConf.load(resolved_eval_cfg_path)
    if device is not None:
        eval_cfg.device = device
    if max_tasks is not None:
        eval_cfg.max_tasks = int(max_tasks)
    if replan_every_step is not None:
        eval_cfg.replan_every_step = bool(replan_every_step)

    protocol_path = resolve_project_path(eval_cfg.protocol_path)
    protocol = load_libero_protocol(protocol_path)
    task_specs = list(protocol.tasks)
    if eval_cfg.get("max_tasks") is not None:
        task_specs = task_specs[: int(eval_cfg.max_tasks)]

    policy = build_exported_servovla_libero_policy(
        pretrained_dir,
        device=str(eval_cfg.device),
        semantic_wait_fail_ms=semantic_wait_fail_ms,
        delay_chunk_size_threshold=(
            None
            if eval_cfg.get("delay_chunk_size_threshold") is None
            else float(eval_cfg.delay_chunk_size_threshold)
        ),
        max_frame_delay=(
            None if eval_cfg.get("max_frame_delay") is None else int(eval_cfg.max_frame_delay)
        ),
        semantic_admission_policy=(
            None
            if eval_cfg.get("semantic_admission_policy") is None
            else str(eval_cfg.semantic_admission_policy)
        ),
        inference_noise_seed=(
            None
            if eval_cfg.get("inference_noise_seed") is None
            else int(eval_cfg.inference_noise_seed)
        ),
        inference_noise_seed_mode=(
            None
            if eval_cfg.get("inference_noise_seed_mode") is None
            else str(eval_cfg.inference_noise_seed_mode)
        ),
    )
    metadata = _load_export_metadata(pretrained_dir)
    run_id = str(eval_cfg.get("run_id", ""))
    config_id = str(eval_cfg.get("config_id", ""))
    code_revision = str(eval_cfg.get("code_revision", ""))
    checkpoint_id = str(eval_cfg.get("checkpoint_id", metadata.get("checkpoint_path") or ""))
    runtime_event_jsonl_value = eval_cfg.get("runtime_event_jsonl")
    runtime_event_jsonl = (
        None
        if runtime_event_jsonl_value is None
        else resolve_project_path(runtime_event_jsonl_value)
    )
    episode_jsonl_value = eval_cfg.get("episode_jsonl")
    episode_jsonl = (
        None if episode_jsonl_value is None else resolve_project_path(episode_jsonl_value)
    )

    task_successes: dict[str, list[bool]] = {}
    task_lengths: dict[str, list[int]] = {}
    runtime_metrics = _empty_runtime_metrics()
    for task_spec in task_specs:
        log.info("Running LIBERO pretrained rollout evaluation for task %s", task_spec.task_name)
        successes, lengths, task_metrics = _rollout_task(
            suite=str(getattr(task_spec, "suite", eval_cfg.suite)),
            task_spec=task_spec,
            policy=policy,
            obs_type=str(eval_cfg.obs_type),
            render_mode=str(eval_cfg.render_mode),
            camera_name=str(eval_cfg.camera_name),
            control_mode=str(eval_cfg.get("control_mode", "relative")),
            n_episodes=int(eval_cfg.n_episodes_per_task),
            use_async_envs=bool(eval_cfg.get("use_async_envs", False)),
            seed=int(eval_cfg.seed),
            replan_every_step=bool(eval_cfg.get("replan_every_step", False)),
            config_id=config_id,
            run_id=run_id,
            code_revision=code_revision,
            checkpoint_id=checkpoint_id,
            runtime_event_jsonl=runtime_event_jsonl,
        )
        task_successes[task_spec.language] = successes
        task_lengths[task_spec.language] = lengths
        _extend_runtime_metrics(runtime_metrics, task_metrics)
        for episode_id, (success, length) in enumerate(zip(successes, lengths, strict=False)):
            _append_jsonl(
                episode_jsonl,
                {
                    "run_id": run_id,
                    "code_revision": code_revision,
                    "config_id": config_id,
                    "checkpoint_id": checkpoint_id,
                    "seed": int(eval_cfg.seed),
                    "suite": str(getattr(task_spec, "suite", eval_cfg.suite)),
                    "task_id": int(task_spec.suite_task_id),
                    "task_name": str(task_spec.task_name),
                    "episode_id": int(episode_id),
                    "episode_success": bool(success),
                    "episode_length_steps": int(length),
                },
            )

    summary = summarize_libero_rollouts(
        task_successes, task_lengths, top_failure_count=int(eval_cfg.get("top_failure_count", 10))
    )
    runtime_summary = summarize_runtime_metrics(runtime_metrics)
    result = {
        "pretrained_dir": str(pretrained_dir),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "used_ema": metadata.get("used_ema"),
        "eval_cfg_path": str(resolved_eval_cfg_path),
        "protocol_path": str(protocol_path),
        "suite": str(eval_cfg.suite),
        "suites": list(protocol.suites)
        if protocol.suites
        else sorted({task.suite for task in task_specs}),
        "control_mode": str(eval_cfg.get("control_mode", "relative")),
        "replan_every_step": bool(eval_cfg.get("replan_every_step", False)),
        "delay_chunk_size_threshold": (
            None
            if getattr(policy.policy.config, "delay_chunk_size_threshold", None) is None
            else float(policy.policy.config.delay_chunk_size_threshold)
        ),
        "max_frame_delay": int(getattr(policy.policy.config, "max_frame_delay", 0)),
        "semantic_admission_policy": str(
            getattr(policy.policy.config, "semantic_admission_policy", "bsr")
        ),
        "run_id": run_id,
        "config_id": config_id,
        "code_revision": code_revision,
        "checkpoint_id": checkpoint_id,
        "runtime_event_jsonl": None if runtime_event_jsonl is None else str(runtime_event_jsonl),
        "episode_jsonl": None if episode_jsonl is None else str(episode_jsonl),
        "task_count": len(task_specs),
        "n_episodes_per_task": int(eval_cfg.n_episodes_per_task),
        "action_mode": str(getattr(policy.policy.config, "action_mode", "unknown")),
        "inference_noise_seed": getattr(policy.policy.config, "inference_noise_seed", None),
        "inference_noise_seed_mode": str(
            getattr(policy.policy.config, "inference_noise_seed_mode", "unknown")
        ),
        **runtime_summary,
        **summary,
    }

    if output_json is not None:
        output_path = resolve_project_path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    return result
