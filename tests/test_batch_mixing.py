from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import servovla.data.batch_mixing as batch_mixing
from servovla.data.batch_mixing import (
    BatchQuotaState,
    compute_batch_quotas,
    resolve_action_normalization_weights,
    resolve_sampler_train_weights,
    resolve_train_weights,
)


def test_single_train_dataset_defaults_to_weight_one():
    cfg = OmegaConf.create({"dataset": {"train_names": ["local/a"]}})

    assert resolve_train_weights(cfg, ["local/a"]) == [1.0]


def test_multiple_train_datasets_require_explicit_weights():
    cfg = OmegaConf.create({"dataset": {"train_names": ["local/a", "local/b"]}})

    with pytest.raises(ValueError, match="dataset.train_weights is required"):
        resolve_train_weights(cfg, ["local/a", "local/b"])


def test_train_weights_must_match_train_names_length():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_names": ["local/a", "local/b"],
                "train_weights": [1.0],
            }
        }
    )

    with pytest.raises(ValueError, match="same length"):
        resolve_train_weights(cfg, ["local/a", "local/b"])


def test_train_weights_must_be_positive():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_names": ["local/a", "local/b"],
                "train_weights": [1.0, 0.0],
            }
        }
    )

    with pytest.raises(ValueError, match="positive"):
        resolve_train_weights(cfg, ["local/a", "local/b"])


def test_equal_seen_frames_strategy_uses_equal_sampler_quota_for_unequal_source_frames():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_names": ["local/small", "local/medium", "local/large"],
                "train_weight_strategy": "equal_seen_frames",
            }
        }
    )

    weights = resolve_sampler_train_weights(
        cfg,
        ["local/small", "local/medium", "local/large"],
        dataset_frame_counts=[100, 1_000, 10_000],
    )

    assert weights == [1.0, 1.0, 1.0]


def test_action_normalization_weights_default_to_dataset_average_not_sampler_weights():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_names": ["local/a", "local/b"],
                "train_weights": [9.0, 1.0],
            },
            "training": {
                "action_normalization": {
                    "enabled": True,
                },
            },
        }
    )

    assert resolve_action_normalization_weights(cfg, ["local/a", "local/b"]) == [1.0, 1.0]


def test_largest_remainder_quotas_follow_explicit_weights():
    state = BatchQuotaState()

    first = compute_batch_quotas(batch_size=4, weights=[3.0, 1.0], state=state)
    second = compute_batch_quotas(batch_size=4, weights=[3.0, 1.0], state=state)

    assert first == [3, 1]
    assert second == [3, 1]


def test_batch_mixing_rejects_unsupported_mode():
    cfg = OmegaConf.create(
        {
            "training": {
                "batch_mixing": {
                    "mode": "implicit_num_steps",
                    "integerization": "largest_remainder",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="training.batch_mixing.mode"):
        batch_mixing.validate_batch_mixing_config(cfg)


def test_batch_mixing_rejects_unsupported_integerization():
    cfg = OmegaConf.create(
        {
            "training": {
                "batch_mixing": {
                    "mode": "explicit_weights",
                    "integerization": "round_robin",
                }
            }
        }
    )

    with pytest.raises(ValueError, match="training.batch_mixing.integerization"):
        batch_mixing.validate_batch_mixing_config(cfg)
