from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from . import deploy_utils as deploy_utils_module
from .deploy_utils import build_export_kwargs_from_training_cfg, export_checkpoint_to_pretrained

PluginSymbolsLoader = Callable[
    [], tuple[type[Any], type[nn.Module], Callable[[Any], tuple[Any, Any]]]
]


class _SmokeConfig:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(self.kwargs, indent=2), encoding="utf-8")


class _SmokeProcessor:
    def __init__(self, filename: str):
        self.filename = filename

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / self.filename).write_text('{"ok": true}\n', encoding="utf-8")


class _SmokePolicy(nn.Module):
    def __init__(self, config: _SmokeConfig):
        super().__init__()
        self.config = config
        self.model = nn.Linear(4, 2)

    def save_pretrained(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.config.save_pretrained(output_dir)
        (output_dir / "model.safetensors").write_bytes(b"fake-weights")


def _make_smoke_pre_post_processors(
    _config: _SmokeConfig,
) -> tuple[_SmokeProcessor, _SmokeProcessor]:
    return _SmokeProcessor("policy_preprocessor.json"), _SmokeProcessor("policy_postprocessor.json")


def default_smoke_plugin_symbols_loader() -> tuple[
    type[_SmokeConfig], type[_SmokePolicy], Callable[[Any], tuple[Any, Any]]
]:
    return _SmokeConfig, _SmokePolicy, _make_smoke_pre_post_processors


def _build_minimal_training_cfg(action_mode: str) -> Any:
    return OmegaConf.create(
        {
            "dataset": {
                "action_mode": action_mode,
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "num_cameras": 2,
            },
            "model": {
                "policy_head": {
                    "chunk_size": 2,
                    "n_action_steps": 2,
                    "action_dim": 2,
                    "state_dim": 2,
                    "num_inference_steps": 2,
                    "hidden_dim": 8,
                    "num_layers": 1,
                    "num_heads": 1,
                },
                "vision_encoder": {
                    "feature_dim": 4,
                    "image_size": 32,
                    "model_id": "smoke/vision",
                },
                "vlm_encoder": {
                    "feature_dim": 4,
                    "image_size": 32,
                    "model_id": "smoke/vlm",
                },
            },
        }
    )


def run_minimal_export_smoke_check(
    workspace_dir: str | Path,
    *,
    action_mode: str = "delta",
    plugin_symbols_loader: PluginSymbolsLoader | None = None,
) -> Path:
    workspace_dir = Path(workspace_dir).expanduser().resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    cfg = _build_minimal_training_cfg(action_mode)
    config_overrides = build_export_kwargs_from_training_cfg(cfg)
    loader = plugin_symbols_loader or default_smoke_plugin_symbols_loader

    config_cls, policy_cls, _ = loader()
    seed_policy = policy_cls(config_cls(**config_overrides))

    checkpoint_path = workspace_dir / "synthetic_checkpoint.pt"
    pretrained_dir = workspace_dir / "pretrained_servovla"
    torch.save(
        {
            "model_state": seed_policy.state_dict(),
            "ema_state": seed_policy.state_dict(),
        },
        checkpoint_path,
    )

    original_loader = deploy_utils_module._load_plugin_symbols
    try:
        deploy_utils_module._load_plugin_symbols = loader
        export_checkpoint_to_pretrained(
            checkpoint_path,
            pretrained_dir,
            config_overrides=config_overrides,
        )
    finally:
        deploy_utils_module._load_plugin_symbols = original_loader

    return pretrained_dir
