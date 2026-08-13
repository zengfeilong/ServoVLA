from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf


def test_hf_is_online_by_default(monkeypatch):
    from servovla.config.hf_offline import configure_hf_offline_env, hf_from_pretrained_kwargs

    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(key, raising=False)

    assert configure_hf_offline_env() is False

    assert hf_from_pretrained_kwargs(trust_remote_code=True) == {
        "trust_remote_code": True,
    }


def test_explicit_hf_offline_updates_imported_hub_constants(monkeypatch):
    import datasets.config as datasets_config
    import huggingface_hub
    import huggingface_hub.constants as hub_constants

    from servovla.config.hf_offline import configure_hf_offline_env

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    monkeypatch.setenv("HF_HUB_DISABLE_TELEMETRY", "0")
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(hub_constants, "HF_HUB_DISABLE_TELEMETRY", False)
    monkeypatch.setattr(datasets_config, "HF_DATASETS_OFFLINE", False)
    monkeypatch.setattr(datasets_config, "HF_HUB_OFFLINE", False)

    assert configure_hf_offline_env() is True

    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["HF_DATASETS_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    assert __import__("os").environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert hub_constants.HF_HUB_OFFLINE is True
    assert hub_constants.HF_HUB_DISABLE_TELEMETRY is True
    assert hub_constants.is_offline_mode() is True
    assert huggingface_hub.is_offline_mode() is True
    assert datasets_config.HF_DATASETS_OFFLINE is True
    assert datasets_config.HF_HUB_OFFLINE is True


def test_model_encoders_load_with_local_files_only(monkeypatch):
    import servovla.architectures.vision_encoder as vision_module
    import servovla.architectures.vlm_encoder as vlm_module

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    captured = {"vision": None, "vlm": None}

    class _VisionModel(torch.nn.Module):
        config = SimpleNamespace(hidden_size=5)

    class _VLMModel(torch.nn.Module):
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=7))

    class _FakeBackbone:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vision"] = (model_id, kwargs)
            return _VisionModel()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vlm"] = (model_id, kwargs)
            return _VLMModel()

    monkeypatch.setattr(vision_module, "AutoBackbone", _FakeBackbone)
    monkeypatch.setattr(vlm_module, "AutoModel", _FakeAutoModel)

    vision = vision_module.VisionEncoder(model_id="vision-local")
    vlm = vlm_module.VLMEncoder(model_id="vlm-local")

    assert vision.feature_dim == 5
    assert vlm.feature_dim == 7
    assert captured["vision"][1]["local_files_only"] is True
    assert captured["vlm"][1]["local_files_only"] is True


def test_vlm_encoder_forward_requests_only_final_state_without_generation_cache(monkeypatch):
    import servovla.architectures.vlm_encoder as vlm_module

    captured = []
    expected = torch.ones(1, 2, 7)

    class _VLMModel(torch.nn.Module):
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=7))

        def forward(self, **kwargs):
            captured.append(kwargs)
            return SimpleNamespace(last_hidden_state=expected, hidden_states=None)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            del model_id, kwargs
            return _VLMModel()

    monkeypatch.setattr(vlm_module, "AutoModel", _FakeAutoModel)

    vlm = vlm_module.VLMEncoder(model_id="vlm-local")
    output = vlm({"input_ids": torch.ones(1, 2, dtype=torch.long)})

    assert captured[0]["output_hidden_states"] is False
    assert captured[0]["return_dict"] is True
    assert captured[0]["use_cache"] is False
    assert output is expected


def test_train_processors_load_with_local_files_only(monkeypatch):
    from scripts import train as train_script

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    captured = {"vision": None, "vlm": None}

    class _Processor:
        pad_token = "<pad>"
        tokenizer = SimpleNamespace(eos_token="</s>", pad_token="<pad>")

    class _FakeAutoProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vlm"] = (model_id, kwargs)
            return _Processor()

    class _FakeImageProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vision"] = (model_id, kwargs)
            return _Processor()

    monkeypatch.setattr(train_script, "AutoProcessor", _FakeAutoProcessor)
    monkeypatch.setattr(train_script, "AutoImageProcessor", _FakeImageProcessor)
    cfg = OmegaConf.create(
        {
            "model": {
                "vlm_encoder": {"model_id": "vlm-local"},
                "vision_encoder": {"model_id": "vision-local"},
            }
        }
    )

    train_script._load_processors(cfg)

    assert captured["vlm"][1]["local_files_only"] is True
    assert captured["vision"][1]["local_files_only"] is True


def test_lerobot_dataset_root_is_required_locally(tmp_path):
    from servovla.config.hf_offline import require_lerobot_local_dataset_root

    with pytest.raises(FileNotFoundError, match="configured local LeRobot dataset is missing"):
        require_lerobot_local_dataset_root("local/missing", hf_lerobot_home=tmp_path)

    root = tmp_path / "local" / "present"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")

    assert require_lerobot_local_dataset_root("local/present", hf_lerobot_home=tmp_path) == root


def test_lerobot_dataset_root_is_deferred_to_lerobot_when_online(monkeypatch):
    from servovla.config.hf_offline import require_lerobot_local_dataset_root

    for key in (
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_LEROBOT_HOME",
    ):
        monkeypatch.delenv(key, raising=False)

    assert require_lerobot_local_dataset_root("ServoVLA/so101_clean_train") is None


def test_lerobot_dataset_root_prefers_hf_lerobot_home_env(monkeypatch, tmp_path):
    from servovla.config.hf_offline import require_lerobot_local_dataset_root

    root = tmp_path / "lerobot" / "local" / "present"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "lerobot"))

    assert require_lerobot_local_dataset_root("local/present") == root


def test_plugin_policy_processors_load_with_local_files_only(monkeypatch):
    import lerobot_policy_servovla.modeling_servovla as plugin_module
    from lerobot_policy_servovla.configuration_servovla import ServoVLAConfig

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    captured = {"vision": None, "vlm": None}

    class _FakeVisionEncoder(torch.nn.Module):
        def __init__(self, model_id):
            super().__init__()

    class _FakeVLMEncoder(torch.nn.Module):
        def __init__(self, model_id):
            super().__init__()

    class _FakeProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vlm"] = (model_id, kwargs)
            return object()

    class _FakeImageProcessor:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            captured["vision"] = (model_id, kwargs)
            return object()

    monkeypatch.setattr(plugin_module, "VisionEncoder", _FakeVisionEncoder)
    monkeypatch.setattr(plugin_module, "VLMEncoder", _FakeVLMEncoder)
    monkeypatch.setattr(plugin_module, "AutoProcessor", _FakeProcessor)
    monkeypatch.setattr(plugin_module, "AutoImageProcessor", _FakeImageProcessor)
    config = ServoVLAConfig(
        vision_model_name="vision-local",
        vlm_model_name="vlm-local",
        action_dim=2,
        state_dim=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        vision_feature_dim=5,
        semantic_feature_dim=7,
        dropout=0.0,
        vision_grid_size=1,
        num_cameras=2,
        chunk_size=4,
        n_action_steps=1,
        num_inference_steps=2,
        action_mode="delta",
    )

    plugin_module.ServoVLAPolicy(config)

    assert captured["vision"][1]["local_files_only"] is True
    assert captured["vlm"][1]["local_files_only"] is True
