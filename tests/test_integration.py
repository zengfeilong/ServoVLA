"""Integration tests for export, deployment, and raw end-to-end utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import servovla.deployment as deployment_pkg
import servovla.deployment.deploy_utils as deploy_utils_module
from scripts.export_lerobot_pretrained import build_config
from servovla.architectures.fm_solver import FlowMatchingEulerSolver
from servovla.data.dataset_loader import OnlineVLACollator
from servovla.deployment.deploy_utils import (
    build_export_kwargs_from_training_cfg,
    export_checkpoint_to_pretrained,
    load_policy_state_dict_for_export,
    remap_checkpoint_keys,
    write_deployment_bundle,
)


def test_build_export_kwargs_includes_action_mode_and_delta_flag():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "action_delta_state_indices": [0, 1, 2, 3, 4, 5],
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "num_cameras": 2,
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 10,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)
    assert kwargs["action_mode"] == "delta"
    assert kwargs["action_is_delta"] is True
    assert kwargs["action_delta_state_indices"] == [0, 1, 2, 3, 4, 5]


def test_build_export_kwargs_resolves_robot_dims_from_dataset_cfg():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "action_dim": 7,
                "proprio_dim": 8,
                "action_delta_state_indices": [0, 1, 2, 3, 4, 5, None],
                "camera_keys": ["observation.images.image", "observation.images.image2"],
                "num_cameras": 2,
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_cameras": 1,
                    "num_inference_steps": 10,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)

    assert kwargs["action_dim"] == 7
    assert kwargs["state_dim"] == 8
    assert kwargs["num_cameras"] == 2
    assert kwargs["action_delta_state_indices"] == [0, 1, 2, 3, 4, 5, None]


def test_build_export_kwargs_includes_action_normalization_stats():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": ["observation.images.front"],
                "num_cameras": 1,
            },
            "training": {
                "action_normalization": {
                    "enabled": True,
                    "mean": [[1.0, 2.0], [3.0, 4.0]],
                    "std": [[0.5, 0.25], [2.0, 4.0]],
                    "eps": 1.0e-5,
                },
            },
            "model": {
                "policy_head": {
                    "chunk_size": 2,
                    "n_action_steps": 2,
                    "action_dim": 2,
                    "state_dim": 2,
                    "num_inference_steps": 3,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "num_heads": 2,
                },
                "vision_encoder": {
                    "feature_dim": 64,
                    "image_size": 64,
                    "model_id": "vision",
                },
                "vlm_encoder": {
                    "feature_dim": 32,
                    "image_size": 64,
                    "model_id": "vlm",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)

    assert kwargs["action_normalization_enabled"] is True
    assert kwargs["action_normalization_mean"] == [[1.0, 2.0], [3.0, 4.0]]
    assert kwargs["action_normalization_std"] == [[0.5, 0.25], [2.0, 4.0]]
    assert kwargs["action_normalization_eps"] == pytest.approx(1.0e-5)


def test_build_export_kwargs_defers_action_normalization_stats_to_checkpoint():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": ["observation.images.front"],
            },
            "training": {
                "action_normalization": {
                    "enabled": True,
                    "mean": None,
                    "std": None,
                    "eps": 1.0e-5,
                },
            },
            "model": {
                "policy_head": {
                    "chunk_size": 2,
                    "n_action_steps": 2,
                    "action_dim": 2,
                    "state_dim": 2,
                    "num_inference_steps": 3,
                    "hidden_dim": 32,
                    "num_layers": 1,
                    "num_heads": 2,
                },
                "vision_encoder": {
                    "feature_dim": 64,
                    "image_size": 64,
                    "model_id": "vision",
                },
                "vlm_encoder": {
                    "feature_dim": 32,
                    "image_size": 64,
                    "model_id": "vlm",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)
    assert kwargs["_action_normalization_required"] is True
    assert "action_normalization_enabled" not in kwargs

    merged = deploy_utils_module._merge_checkpoint_action_normalization(
        kwargs,
        {
            "action_normalization": {
                "enabled": True,
                "mean": [[1.0, 2.0], [3.0, 4.0]],
                "std": [[0.5, 0.25], [2.0, 4.0]],
                "eps": 1.0e-5,
            }
        },
    )
    assert "_action_normalization_required" not in merged
    assert merged["action_normalization_enabled"] is True
    assert merged["action_normalization_mean"] == [[1.0, 2.0], [3.0, 4.0]]
    assert merged["action_normalization_std"] == [[0.5, 0.25], [2.0, 4.0]]
    assert merged["action_normalization_eps"] == pytest.approx(1.0e-5)


def test_required_action_normalization_stats_raise_when_checkpoint_lacks_stats():
    with pytest.raises(ValueError, match="action_normalization mean/std"):
        deploy_utils_module._merge_checkpoint_action_normalization(
            {"device": "cpu", "_action_normalization_required": True},
            {},
        )


def test_build_export_kwargs_writes_ordered_camera_keys_and_inference_seed():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": [
                    "observation.images.wrist",
                    "observation.images.side",
                    "observation.images.front",
                ],
                "num_cameras": 3,
            },
            "inference": {
                "noise_seed": 123,
                "noise_seed_mode": "step",
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 3,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 512,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 512,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)

    assert kwargs["camera_keys"] == [
        "observation.images.wrist",
        "observation.images.side",
        "observation.images.front",
    ]
    assert kwargs["num_cameras"] == 3
    assert kwargs["num_inference_steps"] == 3
    assert kwargs["inference_noise_seed"] == 123
    assert kwargs["inference_noise_seed_mode"] == "step"
    assert "front_camera_key" not in kwargs
    assert "wrist_camera_key" not in kwargs


def test_build_export_kwargs_separates_vision_and_vlm_image_sizes():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "num_cameras": 2,
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 10,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 320,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 96,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)

    assert kwargs["vision_image_size"] == 320
    assert kwargs["vlm_image_size"] == 96
    assert kwargs["vision_grid_size"] == 20
    assert "image_size" not in kwargs


def test_build_export_kwargs_includes_semantic_delay_runtime_defaults():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "num_cameras": 2,
            },
            "deployment": {
                "server": {
                    "max_frame_delay": 4,
                    "semantic_wait_warn_ms": 500,
                    "semantic_wait_fail_ms": 3000,
                }
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 10,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)
    assert kwargs["max_frame_delay"] == 4
    assert kwargs["semantic_wait_warn_ms"] == 500
    assert kwargs["semantic_wait_fail_ms"] == 3000


def test_build_export_kwargs_defaults_max_frame_delay_to_training_async_support():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "action_mode": "delta",
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "num_cameras": 2,
            },
            "deployment": {
                "server": {
                    "max_frame_delay": None,
                    "semantic_wait_warn_ms": 500,
                    "semantic_wait_fail_ms": 3000,
                }
            },
            "training": {
                "semantic_delay": {
                    "max_delay_chunks": 2,
                    "chunk_size_threshold": 0.2,
                }
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                    "n_action_steps": 16,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 10,
                    "hidden_dim": 512,
                    "num_layers": 8,
                    "num_heads": 8,
                },
                "vision_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
                },
                "vlm_encoder": {
                    "feature_dim": 1024,
                    "image_size": 256,
                    "model_id": "Qwen/Qwen3.5-0.8B",
                },
            },
        }
    )

    kwargs = build_export_kwargs_from_training_cfg(cfg)

    assert kwargs["max_frame_delay"] == 30
    assert kwargs["delay_max_chunks"] == 2
    assert kwargs["delay_chunk_size_threshold"] == 0.2
    assert "fast_rate_hz" not in kwargs
    assert "slow_rate_hz" not in kwargs
    assert "zoh_ratio" not in kwargs

    from lerobot_policy_servovla import ServoVLAConfig

    policy_cfg = ServoVLAConfig(**kwargs)
    assert policy_cfg.max_frame_delay == 30


def test_export_lerobot_build_config_uses_separate_image_size_fields():
    args = argparse.Namespace(
        device="cpu",
        chunk_size=16,
        n_action_steps=16,
        action_dim=6,
        state_dim=6,
        num_cameras=2,
        num_inference_steps=10,
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        vision_feature_dim=1024,
        semantic_feature_dim=1024,
        vision_image_size=320,
        vlm_image_size=96,
        vision_grid_size=20,
        vision_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        vlm_model_name="Qwen/Qwen3.5-0.8B",
        front_camera_key="observation.images.front",
        wrist_camera_key="observation.images.wrist",
    )

    cfg = build_config(args)

    assert cfg.vision_image_size == 320
    assert cfg.vlm_image_size == 96
    assert cfg.vision_grid_size == 20


def test_export_lerobot_build_config_accepts_camera_keys_list():
    args = argparse.Namespace(
        device="cpu",
        chunk_size=16,
        n_action_steps=16,
        action_dim=6,
        state_dim=6,
        num_cameras=2,
        num_inference_steps=3,
        inference_noise_seed=123,
        inference_noise_seed_mode="fixed",
        hidden_dim=512,
        num_layers=8,
        num_heads=8,
        vision_feature_dim=1024,
        semantic_feature_dim=1024,
        vision_image_size=320,
        vlm_image_size=96,
        vision_grid_size=20,
        vision_model_name="facebook/dinov3-vitl16-pretrain-lvd1689m",
        vlm_model_name="Qwen/Qwen3.5-0.8B",
        camera_keys=[
            "observation.images.wrist",
            "observation.images.side",
            "observation.images.front",
        ],
        front_camera_key="observation.images.front",
        wrist_camera_key="observation.images.wrist",
    )

    cfg = build_config(args)

    assert cfg.camera_keys == [
        "observation.images.wrist",
        "observation.images.side",
        "observation.images.front",
    ]
    assert cfg.num_cameras == 3
    assert cfg.inference_noise_seed == 123
    assert cfg.inference_noise_seed_mode == "fixed"


def test_export_lerobot_build_config_inherits_action_normalization_from_checkpoint():
    args = argparse.Namespace(
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        action_dim=2,
        state_dim=2,
        num_cameras=1,
        num_inference_steps=3,
        inference_noise_seed=None,
        inference_noise_seed_mode="step",
        hidden_dim=32,
        num_layers=1,
        num_heads=2,
        vision_feature_dim=64,
        semantic_feature_dim=32,
        vision_image_size=64,
        vlm_image_size=64,
        vision_grid_size=4,
        vision_model_name="vision",
        vlm_model_name="vlm",
        camera_keys=["observation.images.front"],
        front_camera_key="observation.images.front",
        wrist_camera_key="observation.images.wrist",
    )
    checkpoint = {
        "action_normalization": {
            "enabled": True,
            "mean": [[1.0, 2.0], [3.0, 4.0]],
            "std": [[0.5, 0.25], [2.0, 4.0]],
        }
    }

    cfg = build_config(args, checkpoint=checkpoint)

    assert cfg.action_normalization_enabled is True
    assert cfg.action_normalization_mean == [[1.0, 2.0], [3.0, 4.0]]
    assert cfg.action_normalization_std == [[0.5, 0.25], [2.0, 4.0]]


def test_run_minimal_export_smoke_check_exports_synthetic_ema_checkpoint(tmp_path):
    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

        def save_pretrained(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(
                json.dumps(self.kwargs, indent=2), encoding="utf-8"
            )

    class _FakeProcessor:
        def __init__(self, filename: str):
            self.filename = filename

        def save_pretrained(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.filename).write_text('{"ok": true}\n', encoding="utf-8")

    class _FakePolicy(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Linear(4, 2)

        def save_pretrained(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            self.config.save_pretrained(output_dir)
            (output_dir / "model.safetensors").write_bytes(b"fake-weights")

    def _fake_make_pre_post_processors(_config):
        return _FakeProcessor("policy_preprocessor.json"), _FakeProcessor(
            "policy_postprocessor.json"
        )

    def _fake_plugin_loader():
        return _FakeConfig, _FakePolicy, _fake_make_pre_post_processors

    assert hasattr(deployment_pkg, "run_minimal_export_smoke_check")

    exported_dir = deployment_pkg.run_minimal_export_smoke_check(
        tmp_path,
        plugin_symbols_loader=_fake_plugin_loader,
    )

    assert exported_dir.exists()
    assert (exported_dir / "config.json").exists()
    assert (exported_dir / "model.safetensors").exists()
    assert (exported_dir / "policy_preprocessor.json").exists()
    assert (exported_dir / "policy_postprocessor.json").exists()
    assert (exported_dir / "export_metadata.json").exists()

    export_meta = json.loads((exported_dir / "export_metadata.json").read_text(encoding="utf-8"))
    assert export_meta["used_ema"] is False
    assert export_meta["missing_keys"] == []
    assert export_meta["unexpected_keys"] == []

    exported_cfg = json.loads((exported_dir / "config.json").read_text(encoding="utf-8"))
    assert exported_cfg["action_mode"] == "delta"
    assert exported_cfg["action_is_delta"] is True


def test_export_checkpoint_allows_frozen_encoder_keys_to_be_absent(tmp_path):
    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _FakeProcessor:
        def __init__(self, filename: str):
            self.filename = filename

        def save_pretrained(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / self.filename).write_text('{"ok": true}\n', encoding="utf-8")

    class _FakePolicy(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.vision_encoder = nn.Linear(4, 4)
            self.model.vlm_encoder = nn.Linear(4, 4)
            self.model.policy_head = nn.Linear(4, 2)

        def save_pretrained(self, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.safetensors").write_bytes(b"fake-weights")

    def _fake_make_pre_post_processors(_config):
        return _FakeProcessor("policy_preprocessor.json"), _FakeProcessor(
            "policy_postprocessor.json"
        )

    def _fake_plugin_loader():
        return _FakeConfig, _FakePolicy, _fake_make_pre_post_processors

    checkpoint_path = tmp_path / "policy_only_checkpoint.pt"
    torch.save(
        {
            "ema_state": {
                "model.policy_head.weight": torch.zeros(2, 4),
                "model.policy_head.bias": torch.zeros(2),
            }
        },
        checkpoint_path,
    )

    original_loader = deploy_utils_module._load_plugin_symbols
    try:
        deploy_utils_module._load_plugin_symbols = _fake_plugin_loader
        exported_dir = export_checkpoint_to_pretrained(
            checkpoint_path,
            tmp_path / "pretrained_servovla",
            prefer_ema=True,
            config_overrides={},
        )
    finally:
        deploy_utils_module._load_plugin_symbols = original_loader

    assert (exported_dir / "model.safetensors").exists()
    export_meta = json.loads((exported_dir / "export_metadata.json").read_text(encoding="utf-8"))
    assert export_meta["missing_keys"] == []
    assert export_meta["unexpected_keys"] == []


def test_export_checkpoint_raises_on_mismatched_policy_keys(tmp_path):
    class _FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)

    class _FakeProcessor:
        def save_pretrained(self, _output_dir):
            pass

    class _FakePolicy(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.vision_encoder = nn.Linear(4, 4)
            self.model.vlm_encoder = nn.Linear(4, 4)
            self.model.policy_head = nn.Linear(4, 2)

        def save_pretrained(self, _output_dir):
            raise AssertionError("export must stop before saving mismatched weights")

    def _fake_make_pre_post_processors(_config):
        return _FakeProcessor(), _FakeProcessor()

    def _fake_plugin_loader():
        return _FakeConfig, _FakePolicy, _fake_make_pre_post_processors

    checkpoint_path = tmp_path / "mismatched_checkpoint.pt"
    torch.save(
        {
            "ema_state": {
                "model.policy_head.weight": torch.zeros(2, 4),
            }
        },
        checkpoint_path,
    )

    original_loader = deploy_utils_module._load_plugin_symbols
    try:
        deploy_utils_module._load_plugin_symbols = _fake_plugin_loader
        with pytest.raises(RuntimeError, match="Missing keys during export"):
            export_checkpoint_to_pretrained(
                checkpoint_path,
                tmp_path / "pretrained_servovla",
                prefer_ema=True,
                config_overrides={},
            )
    finally:
        deploy_utils_module._load_plugin_symbols = original_loader


class _DummyProcessor:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

    tokenizer = _Tokenizer()

    def __call__(self, *, text, images, padding, return_tensors):
        batch_size = len(text)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        }


def test_online_vla_collator_preserves_episode_metadata():
    collator = OnlineVLACollator(vlm_processor=_DummyProcessor())
    batch = [
        {
            "action": torch.zeros(8, 6),
            "loss_mask": torch.ones(8),
            "pixel_values": torch.zeros(2, 3, 8, 8),
            "vlm_text": "task a",
            "vlm_images": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(6),
            "task_index": torch.tensor(3),
            "episode_index": torch.tensor(7),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(2),
        },
        {
            "action": torch.zeros(8, 6),
            "loss_mask": torch.ones(8),
            "pixel_values": torch.zeros(2, 3, 8, 8),
            "vlm_text": "task a",
            "vlm_images": [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(6),
            "task_index": torch.tensor(3),
            "episode_index": torch.tensor(7),
            "episode_step_index": torch.tensor(1),
            "episode_length": torch.tensor(2),
        },
    ]

    output = collator(batch)

    assert output["task_index"].tolist() == [3, 3]
    assert output["episode_index"].tolist() == [7, 7]
    assert output["episode_step_index"].tolist() == [0, 1]
    assert output["episode_length"].tolist() == [2, 2]


def test_write_deployment_bundle_creates_server_and_jetson_assets(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "checkpoint_step000010.pt"
    checkpoint_path.write_bytes(b"placeholder")
    pretrained_dir = run_dir / "pretrained_servovla"
    pretrained_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "deployment": {
                "server": {
                    "host": "0.0.0.0",
                    "public_host": "10.0.0.2",
                    "port": 8080,
                    "fps": 10,
                    "inference_latency": 0.03,
                    "obs_queue_timeout": 1.0,
                    "max_frame_delay": 4,
                    "semantic_wait_warn_ms": 500,
                    "semantic_wait_fail_ms": 3000,
                    "cuda_visible_devices": "2",
                    "hf_offline": True,
                },
                "robot_client": {
                    "policy_device": "cuda",
                    "client_device": "cpu",
                    "actions_per_chunk": 8,
                    "chunk_size_threshold": 0.5,
                    "fps": 10,
                    "aggregate_fn_name": "weighted_average",
                    "task": "pick up the block",
                    "debug_visualize_queue_size": False,
                    "auto_align_to_leader_before_test": False,
                    "align_timeout_s": 1.5,
                    "align_tolerance": 0.25,
                    "align_settle_s": 0.1,
                },
            }
        }
    )

    deploy_dir = write_deployment_bundle(cfg, run_dir, checkpoint_path, pretrained_dir)

    assert {path.name for path in deploy_dir.iterdir()} == {
        "deployment_manifest.json",
        "server_config.yaml",
        "jetson.yaml",
        "jetson_runner.py",
    }

    manifest = json.loads((deploy_dir / "deployment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_type"] == "servovla"
    assert Path(manifest["checkpoint_path"]) == checkpoint_path.resolve()
    assert Path(manifest["pretrained_dir"]) == pretrained_dir.resolve()

    server_cfg = OmegaConf.load(deploy_dir / "server_config.yaml")
    assert server_cfg.host == "0.0.0.0"
    assert server_cfg.port == 8080
    assert server_cfg.max_frame_delay == 4
    assert server_cfg.semantic_wait_warn_ms == 500
    assert server_cfg.semantic_wait_fail_ms == 3000

    jetson_cfg = OmegaConf.load(deploy_dir / "jetson.yaml")
    assert jetson_cfg.async_client.server_address == "10.0.0.2:8080"
    assert Path(jetson_cfg.async_client.pretrained_name_or_path) == pretrained_dir.resolve()
    assert jetson_cfg.workspace.root == "."
    assert jetson_cfg.record.dataset_root == "data/recordings"
    assert (deploy_dir / "jetson_runner.py").read_bytes() == (
        _PROJECT_ROOT / "scripts" / "jetson_runner.py"
    ).read_bytes()


@pytest.mark.parametrize("threshold", [-0.01, 0.5001])
def test_write_deployment_bundle_rejects_chunk_threshold_outside_delay_contract(
    tmp_path, threshold
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "checkpoint_step000010.pt"
    checkpoint_path.write_bytes(b"placeholder")
    pretrained_dir = run_dir / "pretrained_servovla"
    pretrained_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "deployment": {
                "server": {
                    "host": "0.0.0.0",
                    "public_host": "10.0.0.2",
                    "port": 8080,
                    "fps": 10,
                    "inference_latency": 0.03,
                    "obs_queue_timeout": 1.0,
                    "max_frame_delay": 4,
                    "semantic_wait_warn_ms": 500,
                    "semantic_wait_fail_ms": 3000,
                    "cuda_visible_devices": "2",
                    "hf_offline": True,
                },
                "robot_client": {
                    "policy_device": "cuda",
                    "client_device": "cpu",
                    "actions_per_chunk": 8,
                    "chunk_size_threshold": threshold,
                    "fps": 10,
                    "aggregate_fn_name": "weighted_average",
                    "task": "pick up the block",
                    "debug_visualize_queue_size": False,
                    "auto_align_to_leader_before_test": False,
                    "align_timeout_s": 1.5,
                    "align_tolerance": 0.25,
                    "align_settle_s": 0.1,
                },
            }
        }
    )

    with pytest.raises(ValueError, match="chunk_size_threshold"):
        write_deployment_bundle(cfg, run_dir, checkpoint_path, pretrained_dir)


def test_write_deployment_bundle_defaults_max_frame_delay_to_training_async_support(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint_path = run_dir / "checkpoint_step000010.pt"
    checkpoint_path.write_bytes(b"placeholder")
    pretrained_dir = run_dir / "pretrained_servovla"
    pretrained_dir.mkdir()

    cfg = OmegaConf.create(
        {
            "dataset": {},
            "training": {
                "semantic_delay": {
                    "max_delay_chunks": 2,
                    "chunk_size_threshold": 0.2,
                }
            },
            "model": {
                "policy_head": {
                    "chunk_size": 16,
                },
            },
            "deployment": {
                "server": {
                    "host": "0.0.0.0",
                    "public_host": "10.0.0.2",
                    "port": 8080,
                    "fps": 10,
                    "inference_latency": 0.03,
                    "obs_queue_timeout": 1.0,
                    "max_frame_delay": None,
                    "semantic_wait_warn_ms": 500,
                    "semantic_wait_fail_ms": 3000,
                    "cuda_visible_devices": "2",
                    "hf_offline": True,
                },
                "robot_client": {
                    "policy_device": "cuda",
                    "client_device": "cpu",
                    "actions_per_chunk": 8,
                    "chunk_size_threshold": 0.5,
                    "fps": 10,
                    "aggregate_fn_name": "weighted_average",
                    "task": "pick up the block",
                    "debug_visualize_queue_size": False,
                    "auto_align_to_leader_before_test": False,
                    "align_timeout_s": 1.5,
                    "align_tolerance": 0.25,
                    "align_settle_s": 0.1,
                },
            },
        }
    )

    deploy_dir = write_deployment_bundle(cfg, run_dir, checkpoint_path, pretrained_dir)

    server_cfg = (deploy_dir / "server_config.yaml").read_text(encoding="utf-8")
    assert "max_frame_delay: 30" in server_cfg


def test_remap_checkpoint_keys_handles_compiled_policy_head_prefixes():
    state = {
        "module.policy_head._orig_mod.final_layer.1.weight": torch.ones(6, 8),
        "module.policy_head._orig_mod.final_layer.1.bias": torch.ones(6),
    }
    remapped = remap_checkpoint_keys(state)
    assert "model.policy_head.final_layer.1.weight" in remapped
    assert "model.policy_head.final_layer.1.bias" in remapped


def test_export_accepts_policy_head_only_checkpoint_keys():
    state = {
        "policy_head.final_layer.weight": torch.zeros(2, 2),
        "policy_head.final_layer.bias": torch.zeros(2),
    }
    remapped = remap_checkpoint_keys(state)

    assert "model.policy_head.final_layer.weight" in remapped
    assert "model.policy_head.final_layer.bias" in remapped
    assert not any("vision_encoder" in key for key in remapped)
    assert not any("vlm_encoder" in key for key in remapped)


def test_export_loader_ignores_encoder_weights_from_full_checkpoint():
    class _FakePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.vision_encoder = nn.Linear(4, 4)
            self.model.vlm_encoder = nn.Linear(4, 4)
            self.model.policy_head = nn.Linear(4, 2)

    policy = _FakePolicy()
    original_vision_weight = policy.model.vision_encoder.weight.detach().clone()
    original_vlm_weight = policy.model.vlm_encoder.weight.detach().clone()
    policy_weight = torch.full_like(policy.model.policy_head.weight, 0.125)
    policy_bias = torch.full_like(policy.model.policy_head.bias, -0.25)

    missing, unexpected = load_policy_state_dict_for_export(
        policy,
        {
            "model.vision_encoder.weight": torch.full_like(policy.model.vision_encoder.weight, 9.0),
            "model.vision_encoder.bias": torch.full_like(policy.model.vision_encoder.bias, 9.0),
            "model.vlm_encoder.weight": torch.full_like(policy.model.vlm_encoder.weight, 8.0),
            "model.vlm_encoder.bias": torch.full_like(policy.model.vlm_encoder.bias, 8.0),
            "model.policy_head.weight": policy_weight,
            "model.policy_head.bias": policy_bias,
        },
    )

    assert missing == []
    assert unexpected == []
    assert torch.equal(policy.model.vision_encoder.weight, original_vision_weight)
    assert torch.equal(policy.model.vlm_encoder.weight, original_vlm_weight)
    assert torch.allclose(policy.model.policy_head.weight, policy_weight)
    assert torch.allclose(policy.model.policy_head.bias, policy_bias)


class _ConstantVelocityHead(nn.Module):
    def forward(self, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
        return torch.ones_like(x_t)


def test_flow_matching_solver_moves_forward_from_noise_to_data():
    solver = FlowMatchingEulerSolver(action_dim=1, chunk_size=2, num_inference_steps=4)
    sample = solver.sample(
        policy_head=_ConstantVelocityHead(),
        f_vision=torch.zeros(1, 1, 1),
        c_sem=torch.zeros(1, 1, 1),
        c_sem_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_delay=torch.zeros(1),
        q_current=torch.zeros(1, 1),
        noise=torch.zeros(1, 2, 1),
    )

    expected = torch.ones(1, 2, 1)
    assert torch.allclose(sample, expected, atol=1e-6)
