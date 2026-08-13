#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import torch
import torch._dynamo

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, Dataset, Sampler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_SOURCE_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from servovla.architectures.action_normalization import ActionNormalizer
from servovla.architectures.fm_solver import FlowMatchingEulerSolver
from servovla.architectures.policy_head import build_policy_head_from_cfg
from servovla.architectures.servo_vla import ServoVLA
from servovla.config.action_mode import (
    action_mode_is_delta,
    get_dataset_names,
    normalize_action_delta_state_indices,
    normalize_action_mode,
    resolve_mode_root,
)
from servovla.config.hf_offline import (
    configure_hf_offline_env,
    hf_from_pretrained_kwargs,
    require_lerobot_local_dataset_root,
)
from servovla.data.action_normalization_stats import (
    compute_weighted_action_target_stats,
    write_action_target_stats,
)
from servovla.data.batch_mixing import (
    resolve_action_normalization_weights,
    resolve_sampler_train_weights,
    validate_batch_mixing_config,
)
from servovla.data.dataset_loader import (
    BatchAwareConcatDataset,
    OnlineVLACollator,
    OnlineVLADataset,
    SequentialEpisodeBatchSampler,
)
from servovla.data.raw_window_sampler import (
    DatasetEpisodeIndex,
    EpochChunkRawBatchSampler,
    ExplicitWeightedRawBatchSampler,
)
from servovla.data.video_decode import (
    SERVOVLA_PYAV_CUDA_BACKEND,
    SERVOVLA_TORCHCODEC_CUDA_BACKEND,
    install_lerobot_video_decode_adapter,
    resolve_video_decode_settings,
)
from servovla.deployment.deploy_utils import (
    build_export_kwargs_from_training_cfg,
    export_checkpoint_to_pretrained,
    find_latest_checkpoint,
    write_deployment_bundle,
)
from servovla.evaluation.async_eval import (
    AsyncEvalManager,
    AsyncEvalTask,
)
from servovla.evaluation.online_eval import evaluate_online_validation_batches, write_eval_summary
from servovla.trainer.compile_guard import torch_compile_concurrency_guard
from servovla.trainer.gpu_image_preprocess import (
    RawImageGpuPreprocessor,
    build_raw_image_preprocess_spec,
)
from servovla.trainer.policy_head_ema import load_policy_head_state_dict
from servovla.trainer.trainer_loop import TrainerLoop

log = logging.getLogger(__name__)

AutoImageProcessor = None
AutoProcessor = None
VisionEncoder = None
VLMEncoder = None


def _dataset_slug(dataset_name: str) -> str:
    return dataset_name.split("/")[-1]


def _resolve_amp_dtype(dtype_name: str | None) -> torch.dtype:
    normalized = str(dtype_name or "float32").lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    return torch.float32


def _cfg_get(container, key: str, default=None):
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def _raw_gpu_preprocess_enabled(
    cfg: DictConfig,
    *,
    require_async_producer: bool,
) -> bool:
    gpu_pipeline_cfg = _cfg_get(cfg.training, "gpu_pipeline", {})
    device_name = str(_cfg_get(cfg.training, "device", "cpu")).strip().lower()
    device_is_cuda = device_name == "cuda" or device_name.startswith("cuda:")
    enabled = (
        device_is_cuda
        and bool(_cfg_get(gpu_pipeline_cfg, "enabled", False))
        and bool(_cfg_get(gpu_pipeline_cfg, "gpu_preprocess", False))
    )
    if require_async_producer:
        enabled = enabled and bool(_cfg_get(gpu_pipeline_cfg, "async_producer", False))
    return enabled


def _training_raw_gpu_preprocess_enabled(cfg: DictConfig, *, repeat: bool) -> bool:
    return _raw_gpu_preprocess_enabled(
        cfg,
        require_async_producer=bool(repeat),
    )


def _training_amp_dtype(cfg) -> torch.dtype:
    return _resolve_amp_dtype(_cfg_get(cfg.training, "amp_dtype", "float32"))


def _training_policy_head_dtype(cfg) -> torch.dtype:
    return _resolve_amp_dtype(
        _cfg_get(
            cfg.training,
            "policy_head_param_dtype",
            _cfg_get(cfg.training, "amp_dtype", "float32"),
        )
    )


class _End2EndWorkerInit:
    def __init__(self, decode_settings=None):
        self.decode_settings = decode_settings

    def __call__(self, worker_id: int) -> None:
        _limit_dataloader_worker_threads(worker_id)
        if self.decode_settings is not None:
            install_lerobot_video_decode_adapter(self.decode_settings)


def _build_raw_image_gpu_preprocessor(
    cfg: DictConfig,
    *,
    vision_processor,
    vlm_processor,
    device: torch.device,
):
    if not _raw_gpu_preprocess_enabled(cfg, require_async_producer=False):
        return None
    spec = build_raw_image_preprocess_spec(
        vision_processor=vision_processor,
        vlm_processor=vlm_processor,
        vision_image_size=int(cfg.model.vision_encoder.image_size),
        vlm_image_size=int(cfg.model.vlm_encoder.image_size),
    )
    return RawImageGpuPreprocessor(spec=spec, device=device)


class _GpuPreprocessedEvalBatches:
    def __init__(
        self, batches: Iterable[dict[str, Any]], image_preprocessor: RawImageGpuPreprocessor
    ) -> None:
        self._batches = batches
        self._image_preprocessor = image_preprocessor

    def __iter__(self):
        for batch in self._batches:
            if "vision_images_uint8" in batch or "vlm_images_uint8" in batch:
                yield self._image_preprocessor.preprocess_batch(batch)
            else:
                yield batch


def _prepare_eval_batches_for_gpu_preprocess(
    batches: Iterable[dict[str, Any]],
    image_preprocessor: RawImageGpuPreprocessor | None,
) -> Iterable[dict[str, Any]]:
    if image_preprocessor is None:
        return batches
    return _GpuPreprocessedEvalBatches(batches, image_preprocessor)


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if dtype in {torch.bfloat16, torch.float16}:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return torch.autocast(device_type=device.type, enabled=False)


def _optional_int_list(value) -> list[int] | None:
    if value is None:
        return None
    return [int(item) for item in list(value)]


def _limit_dataloader_worker_threads(worker_id: int) -> None:
    del worker_id

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass


def _task_index_from_table(tasks_table, task_name: str) -> int | None:
    if tasks_table is None:
        return None
    if isinstance(tasks_table, dict):
        value = tasks_table.get(task_name)
        if value is None:
            return None
        if isinstance(value, dict):
            value = value.get("task_index")
        return int(value)
    if hasattr(tasks_table, "loc"):
        try:
            row = tasks_table.loc[task_name]
        except Exception:
            return None
        if hasattr(row, "task_index"):
            return int(row.task_index)
        if isinstance(row, dict) and "task_index" in row:
            return int(row["task_index"])
    return None


def _episode_task_indices(metadata, episode: dict) -> list[int]:
    if "task_index" in episode:
        value = episode["task_index"]
        if isinstance(value, list):
            return [int(item) for item in value]
        return [int(value)]

    raw_tasks = episode.get("tasks", [])
    if isinstance(raw_tasks, str):
        raw_tasks = [raw_tasks]

    indices: list[int] = []
    for task_name in raw_tasks:
        task_index = _task_index_from_table(getattr(metadata, "tasks", None), str(task_name))
        if task_index is not None:
            indices.append(task_index)
    return indices


def _resolve_episode_filter(cfg: DictConfig, dataset_name: str) -> list[int] | None:
    allowlist = _optional_int_list(cfg.dataset.get("task_index_allowlist"))
    max_per_task = cfg.dataset.get("max_episodes_per_task")
    max_per_task = int(max_per_task) if max_per_task is not None else None
    if allowlist is None and max_per_task is None:
        return None

    dataset_root = require_lerobot_local_dataset_root(dataset_name)
    metadata = LeRobotDatasetMetadata(dataset_name, root=dataset_root)
    allowset = set(allowlist) if allowlist is not None else None
    selected: list[int] = []
    counts_by_task: dict[int, int] = {}

    for episode_idx, episode in enumerate(getattr(metadata, "episodes", []) or []):
        task_indices = _episode_task_indices(metadata, episode)
        if allowset is not None:
            matching_tasks = [task_idx for task_idx in task_indices if task_idx in allowset]
            if not matching_tasks:
                continue
        else:
            matching_tasks = task_indices or [episode_idx]

        if max_per_task is not None:
            task_key = int(matching_tasks[0])
            if counts_by_task.get(task_key, 0) >= max_per_task:
                continue
            counts_by_task[task_key] = counts_by_task.get(task_key, 0) + 1

        selected.append(int(episode_idx))

    return selected


class _OffsetBatchSampler(Sampler[list[int]]):
    def __init__(self, samplers: list[Sampler[list[int]]], offsets: list[int]):
        self.samplers = samplers
        self.offsets = offsets

    def __iter__(self):
        for sampler, offset in zip(self.samplers, self.offsets, strict=True):
            for batch in sampler:
                yield [int(idx) + int(offset) for idx in batch]

    def __len__(self):
        return sum(len(sampler) for sampler in self.samplers)


def _task_texts_from_tasks_table(tasks_table: Any) -> list[str]:
    if tasks_table is None:
        return []
    if isinstance(tasks_table, dict):
        return [str(key) for key in tasks_table.keys()]
    if hasattr(tasks_table, "index"):
        try:
            return [str(value) for value in list(tasks_table.index)]
        except Exception:
            pass
    if isinstance(tasks_table, (list, tuple)):
        texts: list[str] = []
        for item in tasks_table:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                for key in ("task", "text", "name", "language_instruction"):
                    if key in item:
                        texts.append(str(item[key]))
                        break
        return texts
    return []


def _iter_online_vla_datasets(dataset: Any):
    if isinstance(dataset, OnlineVLADataset):
        yield dataset
        return
    for child in getattr(dataset, "datasets", []) or []:
        yield from _iter_online_vla_datasets(child)


def _synthetic_warmup_task_texts_from_train_loader(train_dataloader: Any | None) -> list[str]:
    if train_dataloader is None:
        return []
    dataset = getattr(train_dataloader, "dataset", None)
    texts: list[str] = []
    for online_dataset in _iter_online_vla_datasets(dataset):
        tasks_table = getattr(getattr(online_dataset.dataset, "meta", None), "tasks", None)
        texts.extend(_task_texts_from_tasks_table(tasks_table))
    return list(dict.fromkeys(text for text in texts if text))


def _build_vlm_prompt_for_task(vlm_processor: Any, *, num_cameras: int, task_text: str) -> str:
    image_token = str(getattr(vlm_processor, "image_token", "<image>"))
    fallback_text = (image_token * int(num_cameras)) + str(task_text)
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in range(int(num_cameras))]
            + [{"type": "text", "text": str(task_text)}],
        }
    ]
    try:
        prompt = vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        prompt = fallback_text
    if str(prompt).count(image_token) != int(num_cameras):
        return fallback_text
    return str(prompt)


class _SyntheticRawVlmWarmupDataset(Dataset):
    """Small raw-image dataset used only to trigger VLM compile signatures."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        vlm_processor: Any,
        length: int,
        batch_size: int,
        task_texts: Sequence[str] | None = None,
    ) -> None:
        self.length = max(int(length), 1)
        self.batch_size = max(int(batch_size), 1)
        self.action_horizon = int(cfg.model.policy_head.chunk_size)
        self.action_dim = int(cfg.model.policy_head.action_dim)
        self.proprio_dim = int(cfg.dataset.proprio_dim)
        self.vision_image_size = int(cfg.model.vision_encoder.image_size)
        self.vlm_image_size = int(cfg.model.vlm_encoder.image_size)
        self.num_cameras = len(list(cfg.dataset.camera_keys))
        texts = list(dict.fromkeys(str(text) for text in (task_texts or []) if str(text)))
        if not texts:
            texts = ["warmup"]
        self.vlm_texts = [
            _build_vlm_prompt_for_task(
                vlm_processor,
                num_cameras=self.num_cameras,
                task_text=task_text,
            )
            for task_text in texts
        ]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prompt_idx = (int(idx) // self.batch_size) % len(self.vlm_texts)
        vision_image = torch.zeros(
            3,
            self.vision_image_size,
            self.vision_image_size,
            dtype=torch.uint8,
        )
        vlm_image = torch.zeros(
            3,
            self.vlm_image_size,
            self.vlm_image_size,
            dtype=torch.uint8,
        )
        return {
            "action": torch.zeros(self.action_horizon, self.action_dim, dtype=torch.float32),
            "loss_mask": torch.zeros(self.action_horizon, dtype=torch.float32),
            "vlm_text": self.vlm_texts[prompt_idx],
            "frame_delay": torch.tensor(0.0, dtype=torch.float32),
            "q_current": torch.zeros(self.proprio_dim, dtype=torch.float32),
            "dataset_slug": "vlm_compile_warmup",
            "task_index": torch.tensor(0, dtype=torch.int64),
            "episode_index": torch.tensor(0, dtype=torch.int64),
            "episode_step_index": torch.tensor(0, dtype=torch.int64),
            "episode_length": torch.tensor(1, dtype=torch.int64),
            "vision_images_uint8": [vision_image.clone() for _ in range(self.num_cameras)],
            "vlm_images_uint8": [vlm_image.clone() for _ in range(self.num_cameras)],
        }


def _load_processors(cfg: DictConfig):
    global AutoImageProcessor, AutoProcessor
    if AutoProcessor is None or AutoImageProcessor is None:
        from transformers import AutoImageProcessor as _AutoImageProcessor
        from transformers import AutoProcessor as _AutoProcessor

        AutoImageProcessor = _AutoImageProcessor
        AutoProcessor = _AutoProcessor

    vlm_processor = AutoProcessor.from_pretrained(
        cfg.model.vlm_encoder.model_id,
        **hf_from_pretrained_kwargs(trust_remote_code=True),
    )
    if not hasattr(vlm_processor, "pad_token") or vlm_processor.pad_token is None:
        vlm_processor.tokenizer.pad_token = vlm_processor.tokenizer.eos_token
    vision_processor = AutoImageProcessor.from_pretrained(
        cfg.model.vision_encoder.model_id,
        **hf_from_pretrained_kwargs(),
    )
    return vlm_processor, vision_processor


def _resolve_runtime_roots(cfg: DictConfig, *, task_overrides: list[str] | None = None) -> None:
    del task_overrides
    action_mode = normalize_action_mode(cfg.dataset.action_mode)
    output_root = Path(str(cfg.output_root)).expanduser()
    if not output_root.is_absolute():
        output_root = _PROJECT_ROOT / output_root
    with open_dict(cfg):
        cfg.output_dir = str(
            resolve_mode_root(output_root, action_mode) / Path(str(cfg.output_dir)).name
        )


def _resolve_action_dim(cfg: DictConfig) -> int:
    configured_action_dim = _cfg_get(cfg.dataset, "action_dim")
    if configured_action_dim is not None:
        return int(configured_action_dim)
    model_cfg = _cfg_get(cfg, "model", None)
    policy_head_cfg = _cfg_get(model_cfg, "policy_head", {})
    return int(_cfg_get(policy_head_cfg, "action_dim", _cfg_get(cfg.dataset, "proprio_dim", 6)))


def _resolve_state_dim(cfg: DictConfig) -> int:
    model_cfg = _cfg_get(cfg, "model", None)
    policy_head_cfg = _cfg_get(model_cfg, "policy_head", {})
    return int(_cfg_get(cfg.dataset, "proprio_dim", _cfg_get(policy_head_cfg, "state_dim", 6)))


def _resolve_action_delta_state_indices(cfg: DictConfig) -> tuple[int | None, ...]:
    return normalize_action_delta_state_indices(
        _cfg_get(cfg.dataset, "action_delta_state_indices", None),
        action_dim=_resolve_action_dim(cfg),
    )


def _action_normalization_cfg(cfg: DictConfig):
    return cfg.training.get("action_normalization", {})


def _action_normalization_enabled(cfg: DictConfig) -> bool:
    return bool(_cfg_get(_action_normalization_cfg(cfg), "enabled", False))


def _action_normalization_stats_from_checkpoint(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    checkpoint_path = Path(str(path)).expanduser()
    if not checkpoint_path.exists():
        return None
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return None
    stats = checkpoint.get("action_normalization")
    if isinstance(stats, dict):
        return stats
    return None


def _optional_checkpoint_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _require_checkpoint_path(value: Any, *, field_name: str) -> Path | None:
    checkpoint_value = _optional_checkpoint_value(value)
    if checkpoint_value is None:
        return None
    checkpoint_path = Path(checkpoint_value).expanduser()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"training.{field_name} does not exist: {checkpoint_path}")
    return checkpoint_path


def _validate_base_checkpoint(cfg: DictConfig) -> None:
    _require_checkpoint_path(
        cfg.training.get("base_checkpoint", None),
        field_name="base_checkpoint",
    )


def _checkpoint_action_normalization_stats(cfg: DictConfig) -> dict[str, Any] | None:
    checkpoint_path = _require_checkpoint_path(
        cfg.training.get("base_checkpoint", None),
        field_name="base_checkpoint",
    )
    stats = _action_normalization_stats_from_checkpoint(
        str(checkpoint_path) if checkpoint_path else None
    )
    if stats is not None:
        log.info("Using action normalization stats from training.base_checkpoint=%s", checkpoint_path)
    return stats


def _resolve_action_normalization(
    cfg: DictConfig, *, dataset_names: list[str]
) -> ActionNormalizer | None:
    norm_cfg = _action_normalization_cfg(cfg)
    if not bool(_cfg_get(norm_cfg, "enabled", False)):
        return None

    mean = _cfg_get(norm_cfg, "mean", None)
    std = _cfg_get(norm_cfg, "std", None)
    if mean is None or std is None:
        checkpoint_stats = _checkpoint_action_normalization_stats(cfg)
        if checkpoint_stats is not None:
            mean = checkpoint_stats.get("mean")
            std = checkpoint_stats.get("std")
            if mean is not None and std is not None:
                eps = float(checkpoint_stats.get("eps", _cfg_get(norm_cfg, "eps", 1.0e-6)))
                with open_dict(cfg.training):
                    if "action_normalization" not in cfg.training:
                        cfg.training.action_normalization = {}
                    with open_dict(cfg.training.action_normalization):
                        cfg.training.action_normalization.enabled = True
                        cfg.training.action_normalization.mean = mean
                        cfg.training.action_normalization.std = std
                        cfg.training.action_normalization.action_mode = checkpoint_stats.get(
                            "action_mode", cfg.dataset.action_mode
                        )
                        cfg.training.action_normalization.chunk_size = int(
                            checkpoint_stats.get("chunk_size", cfg.model.policy_head.chunk_size)
                        )
                        cfg.training.action_normalization.action_dim = int(
                            checkpoint_stats.get("action_dim", _resolve_action_dim(cfg))
                        )
                        cfg.training.action_normalization.eps = eps
                return ActionNormalizer(
                    enabled=True,
                    mean=mean,
                    std=std,
                    eps=eps,
                )

    if mean is None or std is None:
        dataset_roots = []
        selected_episodes = []
        for dataset_name in dataset_names:
            dataset_root = require_lerobot_local_dataset_root(dataset_name)
            if dataset_root is None:
                metadata_dataset = LeRobotDataset(dataset_name, download_videos=False)
                dataset_root = Path(metadata_dataset.root)
            dataset_roots.append(dataset_root)
            selected_episodes.append(_resolve_episode_filter(cfg, dataset_name))
        dataset_weights = resolve_action_normalization_weights(cfg, dataset_names)
        norm_strategy = str(_cfg_get(norm_cfg, "dataset_weight_strategy", "equal_datasets"))
        log.info(
            "Resolved action normalization dataset weights | strategy=%s | entries=%s",
            norm_strategy,
            [
                {
                    "dataset": str(name),
                    "normalization_weight": float(weight),
                }
                for name, weight in zip(dataset_names, dataset_weights, strict=True)
            ],
        )
        stats = compute_weighted_action_target_stats(
            dataset_roots=dataset_roots,
            dataset_names=dataset_names,
            dataset_weights=dataset_weights,
            selected_episodes_by_dataset=selected_episodes,
            chunk_size=int(cfg.model.policy_head.chunk_size),
            action_dim=int(_resolve_action_dim(cfg)),
            action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
            action_delta_state_indices=_resolve_action_delta_state_indices(cfg),
            eps=float(_cfg_get(norm_cfg, "eps", 1.0e-6)),
        )
        with open_dict(cfg.training):
            if "action_normalization" not in cfg.training:
                cfg.training.action_normalization = {}
            with open_dict(cfg.training.action_normalization):
                cfg.training.action_normalization.enabled = True
                cfg.training.action_normalization.mean = stats.mean
                cfg.training.action_normalization.std = stats.std
                cfg.training.action_normalization.action_mode = stats.action_mode
                cfg.training.action_normalization.action_delta_state_indices = list(
                    stats.action_delta_state_indices
                )
                cfg.training.action_normalization.chunk_size = stats.chunk_size
                cfg.training.action_normalization.action_dim = stats.action_dim
                cfg.training.action_normalization.eps = float(_cfg_get(norm_cfg, "eps", 1.0e-6))
        write_action_target_stats(
            Path(str(cfg.output_dir)) / "action_normalization_stats.json", stats
        )
        mean = stats.mean
        std = stats.std

    return ActionNormalizer(
        enabled=True,
        mean=mean,
        std=std,
        eps=float(_cfg_get(norm_cfg, "eps", 1.0e-6)),
    )


def _load_base_checkpoint_into_model(cfg: DictConfig, model: ServoVLA) -> None:
    base_checkpoint_path = _require_checkpoint_path(
        cfg.training.get("base_checkpoint", None),
        field_name="base_checkpoint",
    )
    if base_checkpoint_path is None:
        return

    checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Expected checkpoint dict at {base_checkpoint_path}, got {type(checkpoint)!r}"
        )

    use_ema = bool(cfg.training.get("base_checkpoint_use_ema", False))
    state_key = "ema_state" if use_ema else "model_state"
    if state_key not in checkpoint:
        available_keys = ", ".join(sorted(str(key) for key in checkpoint.keys()))
        raise KeyError(
            f"Base checkpoint {base_checkpoint_path} does not contain {state_key!r}. "
            f"Available keys: {available_keys}"
        )

    load_policy_head_state_dict(model, checkpoint[state_key])
    source_step = int(checkpoint.get("step", 0))
    log.info(
        "Loaded base checkpoint weights from %s (%s, source step %d). "
        "Starting a fresh training run from step 0.",
        base_checkpoint_path,
        state_key,
        source_step,
    )


def _encoder_init_kwargs(encoder_cfg) -> dict[str, object]:
    kwargs: dict[str, object] = {"model_id": encoder_cfg.model_id}
    attn_implementation = encoder_cfg.get("attn_implementation", None)
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs


def _validate_raw_dataset_split(split: str, *, repeat: bool) -> None:
    field_name = "dataset.train_split" if repeat else "dataset.val_split"
    if str(split) not in {"train", "val"}:
        raise ValueError(
            f"{field_name}={split!r} is unsupported for raw end-to-end loading; "
            "only 'train' and 'val' are accepted, and dataset selection is controlled "
            "through train_names/val_names."
        )


def _resolve_lerobot_dataset_fps(
    dataset_name: str,
    dataset_root: Path,
) -> float:
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"LeRobot dataset metadata is required to compute action-step timestamps: {info_path}"
        )
    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if fps <= 0:
        raise ValueError(f"LeRobot dataset fps must be positive for {dataset_name!r}, got {fps}")
    return fps


def build_end2end_dataloader(
    cfg: DictConfig,
    dataset_names: list[str],
    split: str,
    *,
    repeat: bool,
    shuffle_episodes: bool,
    vlm_processor,
    vision_processor,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    _validate_raw_dataset_split(split, repeat=repeat)
    chunk_size = int(cfg.model.policy_head.chunk_size)
    semantic_delay_cfg = cfg.training.get("semantic_delay", {})
    drop_last = bool(cfg.training.get("drop_last", True)) if repeat else False
    raw_image_mode = _training_raw_gpu_preprocess_enabled(cfg, repeat=repeat)
    decode_cfg = cfg.training.get("decode", None)
    decode_settings = resolve_video_decode_settings(decode_cfg)
    video_backend = install_lerobot_video_decode_adapter(decode_settings)
    datasets = []
    lerobot_datasets = []
    samplers = []
    offsets = []
    next_offset = 0

    for dataset_name_value in dataset_names:
        dataset_name = str(dataset_name_value)
        dataset_slug = _dataset_slug(dataset_name)
        selected_episodes = _resolve_episode_filter(cfg, dataset_name)
        dataset_root = require_lerobot_local_dataset_root(dataset_name)
        if dataset_root is None:
            dataset_root = Path(LeRobotDatasetMetadata(dataset_name).root)
        dataset_fps = _resolve_lerobot_dataset_fps(
            dataset_name,
            dataset_root,
        )
        action_deltas = [i / dataset_fps for i in range(chunk_size)]
        lerobot_dataset = LeRobotDataset(
            dataset_name,
            root=dataset_root,
            episodes=selected_episodes,
            delta_timestamps={"action": action_deltas},
            video_backend=video_backend,
        )
        dataset = OnlineVLADataset(
            lerobot_dataset=lerobot_dataset,
            dataset_slug=dataset_slug,
            vlm_processor=vlm_processor,
            vision_processor=vision_processor,
            action_horizon=chunk_size,
            vision_image_size=int(cfg.model.vision_encoder.image_size),
            vlm_image_size=int(cfg.model.vlm_encoder.image_size),
            use_data_aug=bool(cfg.training.get("use_data_aug", False)),
            camera_keys=list(cfg.dataset.camera_keys),
            camera_key_aliases=cfg.dataset.get("camera_key_aliases"),
            state_keys=list(cfg.dataset.state_keys),
            task_key_candidates=list(cfg.dataset.task_key_candidates),
            proprio_dim=int(cfg.dataset.proprio_dim),
            action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
            action_delta_state_indices=_resolve_action_delta_state_indices(cfg),
            sample_frame_delay=True,
            prompt_cache_size=int(cfg.training.get("prompt_cache_size", 0)),
            delay_max_chunks=int(semantic_delay_cfg.get("max_delay_chunks", 1)),
            delay_chunk_size_threshold=float(semantic_delay_cfg.get("chunk_size_threshold", 0.0)),
            delay_sync_weight=float(semantic_delay_cfg.get("sync_weight", 1.0)),
            delay_async_bucket_weights=semantic_delay_cfg.get("async_bucket_weights", None),
            raw_image_mode=raw_image_mode,
            raw_video_batch_decode_max_span_s=float(
                decode_cfg.get("max_batch_decode_span_s", 2.0) if decode_cfg is not None else 2.0
            ),
            raw_video_batch_decode_parallelism=int(
                decode_cfg.get("intra_batch_parallelism", 1) if decode_cfg is not None else 1
            ),
        )
        sampler = SequentialEpisodeBatchSampler(
            lerobot_dataset=lerobot_dataset,
            batch_size=int(batch_size),
            drop_last=drop_last,
            shuffle_episodes=bool(shuffle_episodes),
        )
        lerobot_datasets.append(lerobot_dataset)
        datasets.append(dataset)
        samplers.append(sampler)
        offsets.append(next_offset)
        next_offset += len(dataset)

    if not datasets:
        raise ValueError(f"No datasets configured for split={split!r}.")

    dataset = datasets[0] if len(datasets) == 1 else BatchAwareConcatDataset(datasets)
    is_train_loader = bool(repeat)
    if is_train_loader:
        validate_batch_mixing_config(cfg)
        dataset_indices = [
            DatasetEpisodeIndex.from_lerobot_dataset(
                lerobot_dataset,
                dataset_id=dataset_idx,
                offset=offset,
            )
            for dataset_idx, (lerobot_dataset, offset) in enumerate(
                zip(lerobot_datasets, offsets, strict=True)
            )
        ]
        dataset_frame_counts = [
            sum(len(indices) for indices in index.episode_to_indices.values())
            for index in dataset_indices
        ]
        weights = resolve_sampler_train_weights(
            cfg,
            dataset_names,
            dataset_frame_counts=dataset_frame_counts,
        )
        train_weight_strategy = str(cfg.dataset.get("train_weight_strategy", "explicit_weights"))
        log.info(
            "Resolved train sampler weights | strategy=%s | entries=%s",
            train_weight_strategy,
            [
                {
                    "dataset": str(name),
                    "source_frames": int(frame_count),
                    "sampler_weight": float(weight),
                }
                for name, frame_count, weight in zip(
                    dataset_names, dataset_frame_counts, weights, strict=True
                )
            ],
        )
        raw_shuffle_cfg = cfg.training.get("raw_shuffle", {})
        raw_shuffle_strategy = str(raw_shuffle_cfg.get("strategy", "epoch_chunk"))
        if raw_shuffle_strategy == "epoch_chunk":
            sampler = EpochChunkRawBatchSampler(
                dataset_indices=dataset_indices,
                weights=weights,
                batch_size=int(batch_size),
                chunk_steps=int(
                    raw_shuffle_cfg.get("chunk_steps", raw_shuffle_cfg.get("window_steps", 128))
                ),
                seed=int(cfg.seed),
                repeat=True,
                drop_last=drop_last,
                shuffle_chunks=bool(raw_shuffle_cfg.get("shuffle_chunks", True)),
                reshuffle_each_epoch=bool(raw_shuffle_cfg.get("reshuffle_each_epoch", True)),
                epoch_boundary_offsets=bool(raw_shuffle_cfg.get("epoch_boundary_offsets", True)),
                seed_stride=int(raw_shuffle_cfg.get("seed_stride", 100003)),
                batch_dataset_strategy=str(raw_shuffle_cfg.get("batch_dataset_strategy", "quota")),
                batch_dataset_burst_batches=int(
                    raw_shuffle_cfg.get("batch_dataset_burst_batches", 1)
                ),
                quota_max_datasets_per_batch=int(
                    raw_shuffle_cfg.get("quota_max_datasets_per_batch", 0)
                ),
                max_episodes_per_batch=int(raw_shuffle_cfg.get("max_episodes_per_batch", 0)),
                episode_burst_batches=int(raw_shuffle_cfg.get("episode_burst_batches", 1)),
            )
        elif raw_shuffle_strategy == "active_window":
            sampler = ExplicitWeightedRawBatchSampler(
                dataset_indices=dataset_indices,
                weights=weights,
                batch_size=int(batch_size),
                window_steps=int(raw_shuffle_cfg.get("window_steps", 128)),
                active_episodes_per_dataset=int(
                    raw_shuffle_cfg.get("active_episodes_per_dataset", 8)
                ),
                window_boundary_offsets=bool(raw_shuffle_cfg.get("window_boundary_offsets", False)),
                seed=int(cfg.seed),
                repeat=True,
                drop_last=drop_last,
                shuffle=bool(shuffle_episodes),
                reactivate_when_remaining_below=int(
                    raw_shuffle_cfg.get("reactivate_when_remaining_below", 0)
                ),
                seed_stride=int(raw_shuffle_cfg.get("seed_stride", 0)),
                batch_dataset_strategy=str(raw_shuffle_cfg.get("batch_dataset_strategy", "quota")),
                batch_dataset_burst_batches=int(
                    raw_shuffle_cfg.get("batch_dataset_burst_batches", 1)
                ),
                quota_max_datasets_per_batch=int(
                    raw_shuffle_cfg.get("quota_max_datasets_per_batch", 0)
                ),
                max_episodes_per_batch=int(raw_shuffle_cfg.get("max_episodes_per_batch", 0)),
                episode_burst_batches=int(raw_shuffle_cfg.get("episode_burst_batches", 1)),
            )
        else:
            supported_strategies = ("epoch_chunk", "active_window")
            raise ValueError(
                "Unsupported training.raw_shuffle.strategy: "
                f"{raw_shuffle_strategy}. Supported values: {', '.join(supported_strategies)}"
            )
    else:
        sampler = samplers[0] if len(samplers) == 1 else _OffsetBatchSampler(samplers, offsets)
    collate_fn = OnlineVLACollator(
        vlm_processor=vlm_processor,
        vlm_processor_micro_batch_size=cfg.training.get("vlm_processor_micro_batch_size", None),
        vlm_sequence_padding_multiple=cfg.training.get("vlm_sequence_padding_multiple", None),
        raw_image_mode=raw_image_mode,
        vlm_image_size=int(cfg.model.vlm_encoder.image_size),
    )
    requested_num_workers = int(num_workers)
    nw = requested_num_workers
    if decode_settings.requires_adapter and decode_settings.num_workers is not None:
        nw = min(requested_num_workers, int(decode_settings.num_workers))
        if nw != requested_num_workers:
            log.info(
                "Capping DataLoader workers for video decode backend %s: requested=%d effective=%d device=%s",
                decode_settings.backend,
                requested_num_workers,
                nw,
                decode_settings.device,
            )
    prefetch_key = "prefetch_factor" if repeat else "eval_prefetch_factor"
    resolved_prefetch = (
        int(cfg.training.get(prefetch_key, cfg.training.prefetch_factor)) if nw > 0 else None
    )
    dataloader_in_order = bool(cfg.training.get("dataloader_in_order", True)) if repeat else True
    worker_init_fn = _End2EndWorkerInit(decode_settings) if nw > 0 else None
    multiprocessing_context = (
        "spawn"
        if nw > 0
        and decode_settings.backend
        in {SERVOVLA_PYAV_CUDA_BACKEND, SERVOVLA_TORCHCODEC_CUDA_BACKEND}
        else None
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=nw,
        pin_memory=bool(cfg.training.pin_memory) and str(cfg.training.device) != "cpu",
        prefetch_factor=resolved_prefetch,
        persistent_workers=nw > 0,
        in_order=dataloader_in_order,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=multiprocessing_context,
    )


def _build_vlm_compile_warmup_dataloader(
    cfg: DictConfig,
    *,
    vlm_processor,
    vision_processor,
    train_dataloader: DataLoader | None = None,
) -> DataLoader | None:
    vlm_compile_cfg = cfg.training.get("vlm_compile", {})
    warmup_enabled = (
        _training_raw_gpu_preprocess_enabled(cfg, repeat=True)
        and bool(cfg.training.get("compile_vlm_encoder", False))
        and bool(_cfg_get(vlm_compile_cfg, "warmup_enabled", False))
        and int(_cfg_get(vlm_compile_cfg, "warmup_max_batches", 0)) > 0
    )
    if not warmup_enabled:
        return None

    configured_batch_size = _cfg_get(vlm_compile_cfg, "warmup_batch_size", None)
    if configured_batch_size is None:
        gpu_pipeline_cfg = cfg.training.get("gpu_pipeline", {})
        micro_batch_size = int(_cfg_get(gpu_pipeline_cfg, "vlm_encoder_micro_batch_size", 0) or 0)
        batch_size = (
            min(int(cfg.training.batch_size), micro_batch_size)
            if micro_batch_size > 0
            else int(cfg.training.batch_size)
        )
    else:
        batch_size = int(configured_batch_size)
    if batch_size <= 0:
        raise ValueError(
            f"training.vlm_compile.warmup_batch_size must be positive, got {batch_size}"
        )
    num_workers = int(_cfg_get(vlm_compile_cfg, "warmup_num_workers", 0))
    if num_workers < 0:
        raise ValueError(f"training.vlm_compile.warmup_num_workers must be >= 0, got {num_workers}")

    shuffle_episodes = bool(_cfg_get(vlm_compile_cfg, "warmup_shuffle_episodes", False))
    synthetic_raw_batches = bool(_cfg_get(vlm_compile_cfg, "synthetic_raw_batches", False))
    log.info(
        "Building VLM compile warmup dataloader | batch_size=%d | num_workers=%d | shuffle_episodes=%s | synthetic_raw_batches=%s",
        batch_size,
        num_workers,
        shuffle_episodes,
        synthetic_raw_batches,
    )
    if synthetic_raw_batches:
        task_texts = _synthetic_warmup_task_texts_from_train_loader(train_dataloader)
        if task_texts:
            log.info(
                "Using %d train task prompts for synthetic VLM compile warmup.",
                len(task_texts),
            )
        collate_fn = OnlineVLACollator(
            vlm_processor=vlm_processor,
            vlm_processor_micro_batch_size=cfg.training.get("vlm_processor_micro_batch_size", None),
            vlm_sequence_padding_multiple=cfg.training.get("vlm_sequence_padding_multiple", None),
            raw_image_mode=True,
            vlm_image_size=int(cfg.model.vlm_encoder.image_size),
        )
        return DataLoader(
            _SyntheticRawVlmWarmupDataset(
                cfg,
                vlm_processor=vlm_processor,
                length=batch_size * max(int(_cfg_get(vlm_compile_cfg, "warmup_max_batches", 1)), 1),
                batch_size=batch_size,
                task_texts=task_texts,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    return build_end2end_dataloader(
        cfg,
        dataset_names=get_dataset_names(cfg.dataset, "train"),
        split=str(cfg.dataset.train_split),
        vlm_processor=vlm_processor,
        vision_processor=vision_processor,
        batch_size=batch_size,
        num_workers=num_workers,
        repeat=False,
        shuffle_episodes=shuffle_episodes,
    )


def _build_predict_absolute_chunk_fn(
    model_for_eval: ServoVLA,
    cfg: DictConfig,
    *,
    seed: int,
    zero_noise: bool,
):
    model_for_eval.eval()
    policy_param = next(model_for_eval.policy_head.parameters())
    policy_device = policy_param.device
    policy_dtype = policy_param.dtype
    amp_dtype = _training_amp_dtype(cfg)
    generator = None
    if not zero_noise:
        generator = torch.Generator(device=str(policy_device))
        generator.manual_seed(int(seed))

    chunk_size = int(cfg.model.policy_head.chunk_size)
    action_dim = int(cfg.model.policy_head.action_dim)

    def _sample_from_features(
        *,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        f_vision = f_vision.to(device=policy_device, dtype=policy_dtype)
        c_sem = c_sem.to(device=policy_device, dtype=policy_dtype)
        c_sem_mask = c_sem_mask.to(device=policy_device).bool()
        frame_delay = frame_delay.to(device=policy_device, dtype=torch.float32)
        q_current = q_current.to(device=policy_device, dtype=policy_dtype)

        if zero_noise:
            noise = torch.zeros(
                (q_current.shape[0], chunk_size, action_dim),
                device=policy_device,
                dtype=policy_dtype,
            )
        else:
            noise = torch.randn(
                (q_current.shape[0], chunk_size, action_dim),
                generator=generator,
                device=policy_device,
                dtype=policy_dtype,
            )

        return model_for_eval.sample_action_chunk_from_features(
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
            noise=noise,
        )

    def _predict(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        q_current = batch["q_current"].to(device=policy_device, dtype=policy_dtype)
        pixel_values = batch["pixel_values"].to(device=policy_device)
        vlm_inputs = {
            key: value.to(policy_device) if torch.is_tensor(value) else value
            for key, value in batch["vlm_inputs"].items()
        }
        c_sem_mask = batch["c_sem_mask"].to(device=policy_device).bool()
        frame_delay = batch["frame_delay"].to(device=policy_device, dtype=torch.float32)
        with _autocast_ctx(policy_device, amp_dtype):
            f_vision, c_sem = model_for_eval.encode_observations(
                pixel_values=pixel_values,
                vlm_inputs=vlm_inputs,
            )
            f_vision = f_vision.to(device=policy_device, dtype=policy_dtype)
            c_sem = c_sem.to(device=policy_device, dtype=policy_dtype)

            pred = _sample_from_features(
                f_vision=f_vision,
                c_sem=c_sem,
                c_sem_mask=c_sem_mask,
                frame_delay=frame_delay,
                q_current=q_current,
            )
        return pred.detach().cpu().float()

    def _predict_from_features(
        batch: dict[str, torch.Tensor],
        *,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        del batch
        pred = _sample_from_features(
            f_vision=f_vision,
            c_sem=c_sem,
            c_sem_mask=c_sem_mask,
            frame_delay=frame_delay,
            q_current=q_current,
        )
        return pred.detach().cpu().float()

    _predict.from_features = _predict_from_features
    return _predict


def _maybe_init_wandb(cfg: DictConfig) -> bool:
    try:
        import wandb

        init_timeout_s = float(os.environ.get("SERVOVLA_WANDB_INIT_TIMEOUT", "300"))
        # Keep live training W&B traffic limited to scalar history/config. The
        # default console/stats/code capture can overload free-account
        # filestream limits on long ServoVLA runs.
        wandb_settings = wandb.Settings(
            init_timeout=init_timeout_s,
            x_disable_stats=True,
            console="off",
            disable_git=True,
            disable_code=True,
            x_disable_meta=True,
            x_save_requirements=False,
        )
        init_kwargs = {
            "project": "ServoVLA",
            "config": OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
            "dir": str(cfg.output_dir),
            "settings": wandb_settings,
        }
        wandb.init(**init_kwargs)
        wandb.define_metric("global_step")
        wandb.define_metric("train/*", step_metric="global_step")
        wandb.define_metric("perf/*", step_metric="global_step")
        wandb.define_metric("val/step")
        wandb.define_metric("val/*", step_metric="val/step")
        log.info("wandb initialised (run=%s)", wandb.run.name)
        return True
    except Exception as exc:
        log.warning("wandb not available or failed to init (%s) - skipping", exc)
        return False


def _maybe_get_wandb_logger():
    try:
        import wandb

        if wandb.run is not None:
            return wandb.log
    except Exception:
        return None
    return None


def _build_servovla_model(cfg: DictConfig, *, device: torch.device) -> ServoVLA:
    global VisionEncoder, VLMEncoder
    if VisionEncoder is None or VLMEncoder is None:
        from servovla.architectures.vision_encoder import VisionEncoder as _VisionEncoder
        from servovla.architectures.vlm_encoder import VLMEncoder as _VLMEncoder

        VisionEncoder = _VisionEncoder
        VLMEncoder = _VLMEncoder

    action_dim = _resolve_action_dim(cfg)
    state_dim = _resolve_state_dim(cfg)
    with open_dict(cfg):
        cfg.model.policy_head.action_dim = action_dim
        cfg.model.policy_head.state_dim = state_dim
        cfg.model.policy_head.num_cameras = int(cfg.dataset.num_cameras)
    log.info("Resolved action dimension: %d", action_dim)
    log.info("Resolved proprio/state dimension: %d", state_dim)

    vision_encoder = VisionEncoder(**_encoder_init_kwargs(cfg.model.vision_encoder)).to(device)
    vlm_encoder = VLMEncoder(**_encoder_init_kwargs(cfg.model.vlm_encoder)).to(device)
    for param in vision_encoder.parameters():
        param.requires_grad_(False)
    for param in vlm_encoder.parameters():
        param.requires_grad_(False)

    policy_dtype = _training_policy_head_dtype(cfg)
    policy_head = build_policy_head_from_cfg(cfg).to(device=device, dtype=policy_dtype)
    fm_solver = FlowMatchingEulerSolver(
        action_dim=action_dim,
        chunk_size=int(cfg.model.policy_head.chunk_size),
        num_inference_steps=int(cfg.model.policy_head.num_inference_steps),
    )
    action_normalizer = None
    if _action_normalization_enabled(cfg):
        action_normalizer = _resolve_action_normalization(
            cfg, dataset_names=get_dataset_names(cfg.dataset, "train")
        )
    model = ServoVLA(
        vision_encoder=vision_encoder,
        vlm_encoder=vlm_encoder,
        policy_head=policy_head,
        fm_solver=fm_solver,
        action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
        action_delta_state_indices=_resolve_action_delta_state_indices(cfg),
        action_normalizer=action_normalizer,
    ).to(device)

    n_params = (
        model.policy_head.num_parameters()
        if hasattr(model.policy_head, "num_parameters")
        else sum(p.numel() for p in model.policy_head.parameters() if p.requires_grad)
    )
    log.info("Policy Head: %d trainable parameters (%.2f M)", n_params, n_params / 1e6)
    return model


def _normalize_async_eval_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if normalized.startswith("_orig_mod."):
            normalized = normalized[len("_orig_mod.") :]
        normalized = normalized.replace("._orig_mod.", ".")
        normalized = normalized.replace("._orig_mod", "")
        normalized_state[normalized] = value
    return normalized_state


def build_async_eval_manager(
    cfg,
    *,
    model,
    val_dataloader,
    image_preprocessor=None,
):
    async_eval_cfg = getattr(cfg.training, "async_eval", None)
    backend = (
        str(getattr(async_eval_cfg, "backend", "thread")).lower()
        if async_eval_cfg is not None
        else "thread"
    )
    if backend == "subprocess":
        return build_subprocess_async_eval_manager(cfg)

    def _run_eval_for_state(
        *,
        state_dict: dict[str, torch.Tensor],
        task: AsyncEvalTask,
        used_ema: bool,
    ) -> dict[str, object]:
        load_policy_head_state_dict(
            model,
            _normalize_async_eval_state_dict_keys(state_dict),
        )
        model.eval()
        with torch_compile_concurrency_guard():
            summary = evaluate_online_validation_batches(
                model=model,
                batches=_prepare_eval_batches_for_gpu_preprocess(
                    val_dataloader, image_preprocessor
                ),
                device=str(cfg.training.device),
                action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
                action_delta_state_indices=_resolve_action_delta_state_indices(cfg),
                predict_absolute_chunk_fn=_build_predict_absolute_chunk_fn(
                    model,
                    cfg,
                    seed=int(task.eval_seed),
                    zero_noise=bool(task.eval_zero_noise),
                ),
                seed=int(task.eval_seed),
                amp_dtype=_training_amp_dtype(cfg),
            )
        summary["used_ema"] = bool(used_ema)
        return summary

    def _evaluate_task(task: AsyncEvalTask) -> dict[str, object]:
        snapshot = dict(task.snapshot or {})
        if "raw_state" in snapshot or "ema_state" in snapshot:
            summary: dict[str, object] = {}
            raw_state = snapshot.get("raw_state")
            ema_state = snapshot.get("ema_state")
            if raw_state is not None:
                summary["raw"] = _run_eval_for_state(
                    state_dict=raw_state, task=task, used_ema=False
                )
            if ema_state is not None:
                summary["ema"] = _run_eval_for_state(state_dict=ema_state, task=task, used_ema=True)
            return summary

        model_state = snapshot.get("model_state", {})
        return _run_eval_for_state(state_dict=model_state, task=task, used_ema=bool(task.used_ema))

    return AsyncEvalManager(
        evaluator=_evaluate_task,
        result_writer=lambda step, summary: write_eval_summary(
            Path(str(cfg.output_dir)) / "eval" / f"step{int(step):07d}.json",
            summary,
        ),
        wandb_logger=_maybe_get_wandb_logger(),
    )


def _async_eval_task_payload(task: AsyncEvalTask) -> dict[str, object]:
    return {
        "step": int(task.step),
        "snapshot": dict(task.snapshot or {}),
        "used_ema": bool(task.used_ema),
        "eval_seed": int(task.eval_seed),
        "eval_zero_noise": bool(task.eval_zero_noise),
    }


def _config_to_container(cfg) -> dict[str, object]:
    if OmegaConf.is_config(cfg):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    if isinstance(cfg, dict):
        return {str(key): _config_to_container(value) for key, value in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [_config_to_container(value) for value in cfg]
    if hasattr(cfg, "__dict__"):
        return {str(key): _config_to_container(value) for key, value in vars(cfg).items()}
    return cfg


_MISSING_CONFIG_VALUE = object()


def _get_config_value(cfg, dotted_path: str):
    current = cfg
    for part in dotted_path.split("."):
        if OmegaConf.is_config(current):
            if part not in current:
                return _MISSING_CONFIG_VALUE
            current = current[part]
        elif isinstance(current, dict):
            if part not in current:
                return _MISSING_CONFIG_VALUE
            current = current[part]
        else:
            if not hasattr(current, part):
                return _MISSING_CONFIG_VALUE
            current = getattr(current, part)
    return current


def _require_config_value(cfg, dotted_path: str):
    value = _get_config_value(cfg, dotted_path)
    if value is _MISSING_CONFIG_VALUE:
        raise ValueError(f"{dotted_path} must be set in config.")
    return value


def _write_async_eval_config(cfg, path: Path, *, eval_device: str) -> None:
    cfg_for_eval = OmegaConf.create(_config_to_container(cfg))
    with open_dict(cfg_for_eval.training):
        cfg_for_eval.training.device = str(eval_device)
        decode_device = _get_config_value(cfg_for_eval, "training.async_eval.decode_device")
        if (
            decode_device is not _MISSING_CONFIG_VALUE
            and decode_device is not None
            and str(decode_device).strip()
        ):
            if "decode" not in cfg_for_eval.training:
                cfg_for_eval.training.decode = {}
            with open_dict(cfg_for_eval.training.decode):
                cfg_for_eval.training.decode.device = str(decode_device)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg_for_eval, f=str(path))


def _run_async_eval_subprocess(cfg, task: AsyncEvalTask) -> dict[str, object]:
    eval_device = str(_require_config_value(cfg, "training.async_eval.device"))
    async_eval_cfg = getattr(cfg.training, "async_eval", None)
    output_dir = Path(str(cfg.output_dir))
    task_dir = output_dir / str(getattr(async_eval_cfg, "task_dir", "eval_tasks"))
    task_dir.mkdir(parents=True, exist_ok=True)

    step = int(task.step)
    config_path = task_dir / "async_eval_config.yaml"
    task_path = task_dir / f"step{step:07d}.pt"
    output_path = task_dir / f"step{step:07d}.json"
    log_path = task_dir / f"step{step:07d}.log"

    _write_async_eval_config(cfg, config_path, eval_device=eval_device)
    torch.save(_async_eval_task_payload(task), task_path)
    output_path.unlink(missing_ok=True)

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("SERVOVLA_EVAL_DEVICE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(_SOURCE_ROOT), str(_PROJECT_ROOT), env.get("PYTHONPATH", "")) if path
    )

    cmd = [
        sys.executable,
        "-u",
        str(_PROJECT_ROOT / "scripts" / "async_eval_worker.py"),
        "--config",
        str(config_path),
        "--task",
        str(task_path),
        "--output",
        str(output_path),
    ]
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"async eval subprocess failed for step {step} with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    if not output_path.exists():
        raise RuntimeError(f"async eval subprocess did not write {output_path}; see {log_path}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def build_subprocess_async_eval_manager(cfg) -> AsyncEvalManager:
    eval_device = _require_config_value(cfg, "training.async_eval.device")
    log.info(
        "Async eval subprocess backend enabled | device=%s",
        eval_device,
    )
    return AsyncEvalManager(
        evaluator=lambda task: _run_async_eval_subprocess(cfg, task),
        result_writer=lambda step, summary: write_eval_summary(
            Path(str(cfg.output_dir)) / "eval" / f"step{int(step):07d}.json",
            summary,
        ),
        wandb_logger=_maybe_get_wandb_logger(),
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    configure_hf_offline_env()

    seed = int(cfg.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    _resolve_runtime_roots(cfg)
    _validate_base_checkpoint(cfg)

    device = torch.device(str(cfg.training.device))
    vlm_processor, vision_processor = _load_processors(cfg)
    model = _build_servovla_model(cfg, device=device)
    _load_base_checkpoint_into_model(cfg, model)

    train_dataloader = build_end2end_dataloader(
        cfg,
        dataset_names=get_dataset_names(cfg.dataset, "train"),
        split=str(cfg.dataset.train_split),
        vlm_processor=vlm_processor,
        vision_processor=vision_processor,
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
        repeat=True,
        shuffle_episodes=bool(cfg.training.get("shuffle_episodes", True)),
    )
    warmup_dataloader = _build_vlm_compile_warmup_dataloader(
        cfg,
        vlm_processor=vlm_processor,
        vision_processor=vision_processor,
        train_dataloader=train_dataloader,
    )

    val_dataloader = None
    if get_dataset_names(cfg.dataset, "val"):
        val_dataloader = build_end2end_dataloader(
            cfg,
            dataset_names=get_dataset_names(cfg.dataset, "val"),
            split=str(cfg.dataset.val_split),
            vlm_processor=vlm_processor,
            vision_processor=vision_processor,
            batch_size=int(cfg.training.eval_batch_size),
            num_workers=int(cfg.training.eval_num_workers),
            repeat=False,
            shuffle_episodes=False,
        )

    def run_online_eval(model_for_eval, dataloader, step, use_ema):
        with torch_compile_concurrency_guard():
            summary = evaluate_online_validation_batches(
                model=model_for_eval,
                batches=_prepare_eval_batches_for_gpu_preprocess(dataloader, image_preprocessor),
                device=str(cfg.training.device),
                action_is_delta=action_mode_is_delta(cfg.dataset.action_mode),
                action_delta_state_indices=_resolve_action_delta_state_indices(cfg),
                predict_absolute_chunk_fn=_build_predict_absolute_chunk_fn(
                    model_for_eval,
                    cfg,
                    seed=int(cfg.training.eval_seed),
                    zero_noise=bool(cfg.training.eval_zero_noise),
                ),
                seed=int(cfg.training.eval_seed),
                amp_dtype=_training_amp_dtype(cfg),
            )
        summary["used_ema"] = bool(use_ema)
        return summary

    _maybe_init_wandb(cfg)

    with open_dict(cfg.training):
        cfg.training.output_dir = cfg.output_dir

    image_preprocessor = _build_raw_image_gpu_preprocessor(
        cfg,
        vision_processor=vision_processor,
        vlm_processor=vlm_processor,
        device=device,
    )

    async_eval_manager = None
    if bool(getattr(cfg.training.async_eval, "enabled", False)) and val_dataloader is not None:
        async_eval_backend = str(getattr(cfg.training.async_eval, "backend", "thread")).lower()
        eval_model = (
            None
            if async_eval_backend == "subprocess"
            else _build_servovla_model(cfg, device=device)
        )
        async_eval_manager = build_async_eval_manager(
            cfg,
            model=eval_model,
            val_dataloader=val_dataloader,
            image_preprocessor=image_preprocessor,
        )
        async_eval_manager.start()

    trainer = TrainerLoop(
        model=model, train_cfg=cfg.training, image_preprocessor=image_preprocessor
    )

    last_checkpoint_path = trainer.run(
        dataloader=train_dataloader,
        warmup_dataloader=warmup_dataloader,
        max_steps=int(cfg.training.max_steps),
        eval_dataloader=val_dataloader,
        eval_fn=run_online_eval if val_dataloader is not None else None,
        async_eval_manager=async_eval_manager,
    )

    if bool(cfg.deployment.get("enabled", True)) and bool(cfg.deployment.get("auto_export", True)):
        if not last_checkpoint_path:
            last_checkpoint_path = str(find_latest_checkpoint(cfg.output_dir))
        pretrained_dir = (
            _PROJECT_ROOT
            / str(cfg.deployment.get("pretrained_root", "artifacts/pretrained"))
            / Path(str(cfg.output_dir)).name
        ).resolve()
        export_config = build_export_kwargs_from_training_cfg(cfg)
        exported_dir = export_checkpoint_to_pretrained(
            last_checkpoint_path,
            pretrained_dir,
            prefer_ema=bool(cfg.deployment.get("prefer_ema", False)),
            config_overrides=export_config,
        )
        deploy_dir = write_deployment_bundle(
            cfg,
            cfg.output_dir,
            last_checkpoint_path,
            exported_dir,
        )
        log.info("Exported LeRobot pretrained policy to: %s", exported_dir)
        log.info("Deployment bundle written to: %s", deploy_dir)

    log.info("Training finished. Outputs in: %s", cfg.output_dir)


if __name__ == "__main__":
    main()
