from __future__ import annotations

import json

import pytest
from lerobot.configs.policies import PreTrainedConfig

from lerobot_policy_servovla.configuration_servovla import ServoVLAConfig
from lerobot_policy_servovla.server_config_servovla import ServoVLAPolicyServerConfig


def test_servovla_config_defaults_semantic_delay_from_action_step_buckets():
    cfg = ServoVLAConfig(max_frame_delay=None)

    assert cfg.max_frame_delay == 30
    assert cfg.delay_max_chunks == 2
    assert cfg.delay_chunk_size_threshold == 0.2
    assert cfg.semantic_wait_warn_ms == 500
    assert cfg.semantic_wait_fail_ms == 3000


def test_servovla_config_accepts_bucketed_max_frame_delay():
    cfg = ServoVLAConfig(
        max_frame_delay=23,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.5,
    )

    assert cfg.max_frame_delay == 23
    assert cfg.delay_max_chunks == 3
    assert cfg.delay_chunk_size_threshold == 0.5


def test_servovla_config_rejects_negative_max_frame_delay():
    with pytest.raises(ValueError, match="max_frame_delay"):
        ServoVLAConfig(max_frame_delay=-1)


def test_servovla_config_threshold_zero_keeps_chunk_boundary_delay():
    cfg = ServoVLAConfig(
        max_frame_delay=None,
        delay_max_chunks=2,
        delay_chunk_size_threshold=0.0,
    )

    assert cfg.max_frame_delay == 30


def test_servovla_config_accepts_legacy_timing_fields_without_delay_semantics():
    cfg = ServoVLAConfig(
        fast_rate_hz=10,
        slow_rate_hz=2,
        zoh_ratio=5,
        max_frame_delay=None,
    )

    assert cfg.max_frame_delay == 30
    assert cfg.delay_max_chunks == 2
    assert cfg.delay_chunk_size_threshold == 0.2


def test_servovla_config_from_pretrained_accepts_legacy_timing_fields(tmp_path):
    cfg = ServoVLAConfig(max_frame_delay=None)
    cfg.save_pretrained(tmp_path)
    config_path = tmp_path / "config.json"
    config_json = json.loads(config_path.read_text())
    config_json["fast_rate_hz"] = 10
    config_json["slow_rate_hz"] = 2
    config_json["zoh_ratio"] = 5
    config_path.write_text(json.dumps(config_json))

    loaded = PreTrainedConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, ServoVLAConfig)
    assert loaded.max_frame_delay == 30
    assert loaded.delay_max_chunks == 2
    assert loaded.delay_chunk_size_threshold == 0.2


def test_servovla_config_defaults_visual_shapes_follow_vision_image_size():
    cfg = ServoVLAConfig(vision_image_size=128, vlm_image_size=96)

    assert cfg.vision_image_size == 128
    assert cfg.vlm_image_size == 96
    assert cfg.vision_grid_size == 8
    assert cfg.input_features[cfg.front_camera_key].shape == (3, 128, 128)
    assert cfg.input_features[cfg.wrist_camera_key].shape == (3, 128, 128)


def test_servovla_config_feature_shapes_follow_action_and_state_dims():
    cfg = ServoVLAConfig(action_dim=7, state_dim=8)

    assert cfg.action_feature.shape == (7,)
    assert cfg.robot_state_feature.shape == (8,)
    cfg.validate_features()


def test_servovla_config_preserves_action_normalization_stats():
    cfg = ServoVLAConfig(
        action_dim=2,
        action_normalization_enabled=True,
        action_normalization_mean=[[1.0, 2.0], [3.0, 4.0]],
        action_normalization_std=[[0.5, 0.25], [2.0, 4.0]],
    )

    assert cfg.action_normalization_enabled is True
    assert cfg.action_normalization_mean == [[1.0, 2.0], [3.0, 4.0]]
    assert cfg.action_normalization_std == [[0.5, 0.25], [2.0, 4.0]]


def test_servovla_config_requires_stats_when_action_normalization_enabled():
    with pytest.raises(ValueError, match="action_normalization_mean"):
        ServoVLAConfig(action_normalization_enabled=True)


def test_policy_server_config_keeps_runtime_delay_overrides():
    cfg = ServoVLAPolicyServerConfig(
        fps=10,
        max_frame_delay=0,
        semantic_wait_warn_ms=250,
        semantic_wait_fail_ms=3000,
    )

    assert cfg.environment_dt == 0.1
    assert cfg.max_frame_delay == 0
    assert cfg.semantic_wait_warn_ms == 250
    assert cfg.semantic_wait_fail_ms == 3000


def test_policy_server_config_defaults_to_loopback_host():
    cfg = ServoVLAPolicyServerConfig()

    assert cfg.host == "127.0.0.1"


def test_servovla_config_validates_inference_noise_seed_mode():
    from lerobot_policy_servovla.configuration_servovla import ServoVLAConfig

    cfg = ServoVLAConfig(inference_noise_seed=123, inference_noise_seed_mode="step")
    assert cfg.inference_noise_seed == 123
    assert cfg.inference_noise_seed_mode == "step"

    with pytest.raises(ValueError, match="inference_noise_seed_mode"):
        ServoVLAConfig(inference_noise_seed=123, inference_noise_seed_mode="episode")


def test_servovla_config_accepts_three_ordered_camera_keys():
    from lerobot_policy_servovla.configuration_servovla import ServoVLAConfig

    keys = [
        "observation.images.wrist",
        "observation.images.side",
        "observation.images.front",
    ]
    cfg = ServoVLAConfig(
        camera_keys=keys,
        num_cameras=99,
        vision_image_size=128,
    )

    assert cfg.camera_keys == keys
    assert cfg.num_cameras == 3
    for key in keys:
        assert cfg.input_features[key].shape == (3, 128, 128)


def test_servovla_config_rebuilds_visual_features_from_camera_keys():
    from lerobot.configs.types import FeatureType
    from lerobot.utils.constants import OBS_IMAGES

    from lerobot_policy_servovla.configuration_servovla import ServoVLAConfig

    cfg = ServoVLAConfig(camera_keys=[f"{OBS_IMAGES}.side"], vision_image_size=64)

    assert cfg.camera_keys == [f"{OBS_IMAGES}.side"]
    assert cfg.num_cameras == 1
    assert f"{OBS_IMAGES}.side" in cfg.input_features
    assert f"{OBS_IMAGES}.front" not in cfg.input_features
    assert f"{OBS_IMAGES}.wrist" not in cfg.input_features
    assert cfg.input_features[f"{OBS_IMAGES}.side"].type == FeatureType.VISUAL
