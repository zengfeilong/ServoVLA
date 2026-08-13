from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from servovla.architectures.policy_head_checkpoint import load_policy_head_state_dict
from servovla.config.action_mode import (
    action_mode_is_delta,
    normalize_action_delta_state_indices,
    normalize_action_mode,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_plugin_path() -> None:
    source_root = _project_root() / "src"
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)


def _load_plugin_symbols():
    _ensure_plugin_path()
    from lerobot.policies.factory import make_pre_post_processors

    from lerobot_policy_servovla import ServoVLAConfig, ServoVLAPolicy

    return ServoVLAConfig, ServoVLAPolicy, make_pre_post_processors


def _validate_chunk_size_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 0.5:
        raise ValueError(f"chunk_size_threshold must be in [0, 0.5], got {threshold}")
    return threshold


def _resolve_max_frame_delay(cfg: DictConfig) -> int:
    configured = OmegaConf.select(cfg, "deployment.server.max_frame_delay", default=None)
    if configured is not None:
        return int(configured)

    chunk_size = OmegaConf.select(cfg, "model.policy_head.chunk_size", default=None)
    max_delay_chunks = OmegaConf.select(
        cfg, "training.semantic_delay.max_delay_chunks", default=None
    )
    chunk_size_threshold = OmegaConf.select(
        cfg, "training.semantic_delay.chunk_size_threshold", default=None
    )
    if chunk_size is None or max_delay_chunks is None or chunk_size_threshold is None:
        return 0

    chunk_size = int(chunk_size)
    max_delay_chunks = int(max_delay_chunks)
    _validate_chunk_size_threshold(float(chunk_size_threshold))
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if max_delay_chunks < 1:
        raise ValueError(f"max_delay_chunks must be >= 1, got {max_delay_chunks}")

    boundary_stride = max(chunk_size - 1, 1)
    return int(max_delay_chunks * boundary_stride)


def _resolve_export_action_dim(cfg: DictConfig) -> int:
    dataset_action_dim = OmegaConf.select(cfg, "dataset.action_dim", default=None)
    if dataset_action_dim is not None:
        return int(dataset_action_dim)
    model_action_dim = OmegaConf.select(cfg, "model.policy_head.action_dim", default=None)
    if model_action_dim is not None:
        return int(model_action_dim)
    return int(OmegaConf.select(cfg, "dataset.proprio_dim", default=6))


def _resolve_export_state_dim(cfg: DictConfig) -> int:
    dataset_proprio_dim = OmegaConf.select(cfg, "dataset.proprio_dim", default=None)
    if dataset_proprio_dim is not None:
        return int(dataset_proprio_dim)
    return int(OmegaConf.select(cfg, "model.policy_head.state_dim", default=6))


def build_export_kwargs_from_training_cfg(cfg: DictConfig) -> dict[str, Any]:
    action_mode = normalize_action_mode(cfg.dataset.action_mode)
    action_dim = _resolve_export_action_dim(cfg)
    camera_keys = [str(key) for key in cfg.dataset.camera_keys]
    vision_image_size = int(cfg.model.vision_encoder.image_size)
    vlm_image_size = int(cfg.model.vlm_encoder.image_size)
    resolved_max_frame_delay = _resolve_max_frame_delay(cfg)
    delay_max_chunks = OmegaConf.select(
        cfg, "training.semantic_delay.max_delay_chunks", default=None
    )
    delay_chunk_size_threshold = OmegaConf.select(
        cfg, "training.semantic_delay.chunk_size_threshold", default=None
    )
    if (delay_max_chunks is None) != (delay_chunk_size_threshold is None):
        raise ValueError(
            "training.semantic_delay.max_delay_chunks and training.semantic_delay.chunk_size_threshold "
            "must be configured together"
        )
    semantic_wait_warn_ms = OmegaConf.select(
        cfg, "deployment.server.semantic_wait_warn_ms", default=500
    )
    semantic_wait_fail_ms = OmegaConf.select(
        cfg, "deployment.server.semantic_wait_fail_ms", default=3000
    )
    export_kwargs = {
        "device": "cpu",
        "chunk_size": int(cfg.model.policy_head.chunk_size),
        "n_action_steps": int(
            cfg.model.policy_head.get("n_action_steps", cfg.model.policy_head.chunk_size)
        ),
        "action_dim": action_dim,
        "state_dim": _resolve_export_state_dim(cfg),
        "num_cameras": len(camera_keys),
        "num_inference_steps": int(cfg.model.policy_head.num_inference_steps),
        "inference_noise_seed": OmegaConf.select(cfg, "inference.noise_seed", default=None),
        "inference_noise_seed_mode": str(
            OmegaConf.select(cfg, "inference.noise_seed_mode", default="step")
        ),
        "hidden_dim": int(cfg.model.policy_head.hidden_dim),
        "num_layers": int(cfg.model.policy_head.num_layers),
        "num_heads": int(cfg.model.policy_head.num_heads),
        "vision_feature_dim": int(cfg.model.vision_encoder.feature_dim),
        "semantic_feature_dim": int(cfg.model.vlm_encoder.feature_dim),
        "vision_image_size": vision_image_size,
        "vlm_image_size": vlm_image_size,
        "vision_grid_size": vision_image_size // 16,
        "vision_model_name": str(cfg.model.vision_encoder.model_id),
        "vlm_model_name": str(cfg.model.vlm_encoder.model_id),
        "camera_keys": camera_keys,
        "max_frame_delay": int(resolved_max_frame_delay),
        "semantic_wait_warn_ms": int(semantic_wait_warn_ms),
        "semantic_wait_fail_ms": int(semantic_wait_fail_ms),
        "action_mode": action_mode,
        "action_is_delta": action_mode_is_delta(action_mode),
        "action_delta_state_indices": list(
            normalize_action_delta_state_indices(
                OmegaConf.select(cfg, "dataset.action_delta_state_indices", default=None),
                action_dim=action_dim,
            )
        ),
    }
    action_norm_enabled = bool(
        OmegaConf.select(cfg, "training.action_normalization.enabled", default=False)
    )
    action_norm_mean = OmegaConf.select(cfg, "training.action_normalization.mean", default=None)
    action_norm_std = OmegaConf.select(cfg, "training.action_normalization.std", default=None)
    if action_norm_enabled:
        if action_norm_mean is None or action_norm_std is None:
            export_kwargs["_action_normalization_required"] = True
        else:
            export_kwargs["action_normalization_enabled"] = True
            export_kwargs["action_normalization_mean"] = OmegaConf.to_container(
                action_norm_mean, resolve=True
            )
            export_kwargs["action_normalization_std"] = OmegaConf.to_container(
                action_norm_std, resolve=True
            )
            export_kwargs["action_normalization_eps"] = float(
                OmegaConf.select(cfg, "training.action_normalization.eps", default=1.0e-6)
            )
    if delay_max_chunks is not None and delay_chunk_size_threshold is not None:
        export_kwargs["delay_max_chunks"] = int(delay_max_chunks)
        export_kwargs["delay_chunk_size_threshold"] = _validate_chunk_size_threshold(
            float(delay_chunk_size_threshold)
        )
    return export_kwargs


def _merge_checkpoint_action_normalization(
    cfg_kwargs: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(cfg_kwargs)
    required = bool(merged.pop("_action_normalization_required", False))
    if merged.get("action_normalization_enabled"):
        return merged
    action_norm = checkpoint.get("action_normalization")
    if not isinstance(action_norm, dict) or not bool(action_norm.get("enabled", False)):
        if required:
            raise ValueError(
                "training.action_normalization is enabled, but neither the training config nor "
                "the checkpoint provides action_normalization mean/std."
            )
        return merged
    mean = action_norm.get("mean")
    std = action_norm.get("std")
    if mean is None or std is None:
        if required:
            raise ValueError(
                "training.action_normalization is enabled, but neither the training config nor "
                "the checkpoint provides action_normalization mean/std."
            )
        return merged
    merged["action_normalization_enabled"] = True
    merged["action_normalization_mean"] = mean
    merged["action_normalization_std"] = std
    if "eps" in action_norm:
        merged["action_normalization_eps"] = float(action_norm["eps"])
    return merged


def remap_checkpoint_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        normalized = normalized.replace("._orig_mod", "")
        if not normalized.startswith("model."):
            normalized = f"model.{normalized}"
        remapped[normalized] = value
    return remapped


def load_policy_state_dict_for_export(
    policy: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    missing, unexpected = load_policy_head_state_dict(
        policy,
        remap_checkpoint_keys(state_dict),
        strict=False,
    )
    missing = [f"model.policy_head.{key}" for key in missing]
    unexpected = [f"model.policy_head.{key}" for key in unexpected]
    if not missing and not unexpected:
        return missing, unexpected

    sections: list[str] = ["Checkpoint state_dict is incompatible with the export policy."]
    if missing:
        sections.append("Missing keys during export:")
        sections.extend(f"  - {key}" for key in missing)
    if unexpected:
        sections.append("Unexpected keys during export:")
        sections.extend(f"  - {key}" for key in unexpected)
    raise RuntimeError("\n".join(sections))


def export_checkpoint_to_pretrained(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    prefer_ema: bool = False,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    ServoVLAConfig, ServoVLAPolicy, make_pre_post_processors = _load_plugin_symbols()

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and prefer_ema and "ema_state" in checkpoint:
        state_dict = checkpoint["ema_state"]
        used_ema = True
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        used_ema = False
    else:
        state_dict = checkpoint
        used_ema = False

    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a valid state_dict.")

    cfg_kwargs = _merge_checkpoint_action_normalization(dict(config_overrides or {}), checkpoint)
    policy_cfg = ServoVLAConfig(**cfg_kwargs)
    policy = ServoVLAPolicy(policy_cfg)
    missing, unexpected = load_policy_state_dict_for_export(policy, state_dict)

    preprocessor, postprocessor = make_pre_post_processors(policy_cfg)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    export_meta = {
        "checkpoint_path": str(checkpoint_path),
        "exported_pretrained_dir": str(output_dir),
        "used_ema": used_ema,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }
    with (output_dir / "export_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(export_meta, f, indent=2)

    return output_dir


def find_latest_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoints = sorted(output_dir.glob("checkpoint_step*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint_step*.pt found in {output_dir}")
    return checkpoints[-1]


def write_deployment_bundle(
    cfg: DictConfig,
    output_dir: str | Path,
    checkpoint_path: str | Path,
    pretrained_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    pretrained_dir = Path(pretrained_dir).expanduser().resolve()
    deploy_dir = output_dir / "deployment"
    deploy_dir.mkdir(parents=True, exist_ok=True)
    _validate_chunk_size_threshold(cfg.deployment.robot_client.chunk_size_threshold)
    resolved_max_frame_delay = _resolve_max_frame_delay(cfg)

    deployment_cfg = OmegaConf.to_container(cfg.deployment, resolve=True)
    manifest = {
        "policy_type": "servovla",
        "checkpoint_path": str(checkpoint_path),
        "pretrained_dir": str(pretrained_dir),
        "deployment": deployment_cfg,
    }
    with (deploy_dir / "deployment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    server_cfg = {
        "host": cfg.deployment.server.host,
        "port": int(cfg.deployment.server.port),
        "fps": int(cfg.deployment.server.fps),
        "inference_latency": float(cfg.deployment.server.inference_latency),
        "obs_queue_timeout": float(cfg.deployment.server.obs_queue_timeout),
        "max_frame_delay": resolved_max_frame_delay,
        "semantic_wait_warn_ms": int(cfg.deployment.server.get("semantic_wait_warn_ms", 500)),
        "semantic_wait_fail_ms": int(cfg.deployment.server.get("semantic_wait_fail_ms", 3000)),
    }
    OmegaConf.save(config=OmegaConf.create(server_cfg), f=str(deploy_dir / "server_config.yaml"))

    jetson_cfg = OmegaConf.load(_project_root() / "configs" / "deployment" / "jetson.yaml")
    jetson_cfg.async_client.server_address = (
        f"{cfg.deployment.server.public_host}:{int(cfg.deployment.server.port)}"
    )
    jetson_cfg.async_client.pretrained_name_or_path = str(pretrained_dir)
    OmegaConf.save(config=jetson_cfg, f=str(deploy_dir / "jetson.yaml"))
    shutil.copy2(_project_root() / "scripts" / "jetson_runner.py", deploy_dir / "jetson_runner.py")

    return deploy_dir
