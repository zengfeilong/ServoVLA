from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _load_runner_module(monkeypatch):
    lerobot_module = types.ModuleType("lerobot")
    lerobot_utils_module = types.ModuleType("lerobot.utils")
    lerobot_import_utils_module = types.ModuleType("lerobot.utils.import_utils")
    lerobot_import_utils_module.register_third_party_plugins = lambda: None

    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.utils", lerobot_utils_module)
    monkeypatch.setitem(sys.modules, "lerobot.utils.import_utils", lerobot_import_utils_module)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "jetson_runner.py"
    spec = importlib.util.spec_from_file_location(
        "jetson_workspace_runner_template_test", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_teleop_diagnostics_cfg_uses_safe_defaults(monkeypatch):
    runner = _load_runner_module(monkeypatch)

    default_cfg = runner._build_teleop_diagnostics_cfg({})
    assert default_cfg == {
        "enabled": False,
        "log_every_n_steps": 10,
        "joint_keys": [],
    }

    explicit_cfg = runner._build_teleop_diagnostics_cfg(
        {
            "teleop_diagnostics": {
                "enabled": True,
                "log_every_n_steps": 3,
                "joint_keys": ["shoulder_lift.pos", "elbow_flex.pos"],
            }
        }
    )
    assert explicit_cfg == {
        "enabled": True,
        "log_every_n_steps": 3,
        "joint_keys": ["shoulder_lift.pos", "elbow_flex.pos"],
    }


@pytest.mark.parametrize("threshold", [-0.01, 0.5001])
def test_validate_chunk_size_threshold_rejects_values_outside_delay_contract(
    monkeypatch, threshold
):
    runner = _load_runner_module(monkeypatch)

    with pytest.raises(ValueError, match="chunk_size_threshold"):
        runner._validate_chunk_size_threshold(threshold)


def test_validate_chunk_size_threshold_accepts_sync_and_max_async_thresholds(monkeypatch):
    runner = _load_runner_module(monkeypatch)

    assert runner._validate_chunk_size_threshold(0.0) == 0.0
    assert runner._validate_chunk_size_threshold(0.5) == 0.5


def test_robot_client_delay_debug_reports_queue_side_fields(monkeypatch):
    runner = _load_runner_module(monkeypatch)
    client = types.SimpleNamespace(
        latest_executed_action_timestep=99,
        action_queue_size=3,
        config=types.SimpleNamespace(actions_per_chunk=16, chunk_size_threshold=0.5),
    )

    assert runner._robot_client_delay_debug(client) == {
        "latest_executed_action_timestep": 99,
        "action_queue_size": 3,
        "action_chunk_size": 16,
        "chunk_size_threshold": 0.5,
    }


def test_build_teleop_joint_diagnostic_record_filters_joint_keys_and_computes_errors(monkeypatch):
    runner = _load_runner_module(monkeypatch)

    diag_cfg = {
        "enabled": True,
        "log_every_n_steps": 2,
        "joint_keys": ["shoulder_lift.pos", "elbow_flex.pos"],
    }
    leader_action = {
        "shoulder_lift.pos": -105.0,
        "elbow_flex.pos": 97.0,
        "wrist_roll.pos": -8.0,
    }
    follower_observation = {
        "shoulder_lift.pos": -102.0,
        "elbow_flex.pos": 95.5,
        "wrist_roll.pos": -7.5,
    }
    robot_action_to_send = {
        "shoulder_lift.pos": -104.5,
        "elbow_flex.pos": 96.5,
        "wrist_roll.pos": -8.2,
    }

    skipped = runner._build_teleop_joint_diagnostic_record(
        step_idx=3,
        leader_action=leader_action,
        follower_observation=follower_observation,
        robot_action_to_send=robot_action_to_send,
        robot_joint_keys=["shoulder_lift.pos", "elbow_flex.pos", "wrist_roll.pos"],
        diag_cfg=diag_cfg,
    )
    assert skipped is None

    record = runner._build_teleop_joint_diagnostic_record(
        step_idx=4,
        leader_action=leader_action,
        follower_observation=follower_observation,
        robot_action_to_send=robot_action_to_send,
        robot_joint_keys=["shoulder_lift.pos", "elbow_flex.pos", "wrist_roll.pos"],
        diag_cfg=diag_cfg,
    )

    assert record["step"] == 4
    assert list(record["leader"]) == ["shoulder_lift.pos", "elbow_flex.pos"]
    assert list(record["follower"]) == ["shoulder_lift.pos", "elbow_flex.pos"]
    assert list(record["command"]) == ["shoulder_lift.pos", "elbow_flex.pos"]
    assert record["leader_error"] == {
        "shoulder_lift.pos": -3.0,
        "elbow_flex.pos": 1.5,
    }
    assert record["command_error"] == {
        "shoulder_lift.pos": -2.5,
        "elbow_flex.pos": 1.0,
    }
