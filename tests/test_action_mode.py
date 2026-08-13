
import pytest
from omegaconf import OmegaConf

from servovla.config.action_mode import (
    action_mode_is_delta,
    get_dataset_names,
    normalize_action_mode,
    resolve_mode_root,
)


def test_normalize_action_mode_accepts_delta_and_abs():
    assert normalize_action_mode("delta") == "delta"
    assert normalize_action_mode("abs") == "abs"
    assert action_mode_is_delta("delta") is True
    assert action_mode_is_delta("abs") is False


def test_normalize_action_mode_rejects_invalid_values():
    with pytest.raises(ValueError, match="dataset.action_mode"):
        normalize_action_mode("weird")


def test_get_dataset_names_reads_train_and_val_lists():
    cfg = OmegaConf.create(
        {
            "dataset": {
                "train_names": ["local/train_a", "local/train_b"],
                "val_names": ["local/val_a"],
            }
        }
    )

    assert get_dataset_names(cfg.dataset, "train") == ["local/train_a", "local/train_b"]
    assert get_dataset_names(cfg.dataset, "val") == ["local/val_a"]


def test_resolve_mode_root_prefixes_top_level_directory(tmp_path):
    root = resolve_mode_root(tmp_path / "ServoVLA_Runs", "delta")
    assert root == tmp_path / "ServoVLA_Runs" / "delta"
