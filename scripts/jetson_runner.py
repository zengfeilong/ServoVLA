#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from lerobot.utils.import_utils import register_third_party_plugins

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_project_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified Jetson launcher for teleop, data collection, replay, and ServoVLA client."
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    def add_common_config_arg(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--config", required=True, help="Path to jetson_workspace.yaml")

    teleop_parser = subparsers.add_parser(
        "teleop", help="Run direct teleoperation on the follower robot."
    )
    add_common_config_arg(teleop_parser)
    teleop_parser.add_argument("--fps", type=int, default=None, help="Override teleoperation FPS.")
    teleop_parser.add_argument(
        "--display-data", action="store_true", help="Display camera and state data."
    )

    record_parser = subparsers.add_parser(
        "record", help="Record a teleoperated dataset for a named task."
    )
    add_common_config_arg(record_parser)
    record_parser.add_argument("--task", required=True, help="Task key from the tasks section.")
    record_parser.add_argument(
        "--episodes", type=int, default=None, help="Override number of episodes to record."
    )
    record_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing dataset instead of creating a new one.",
    )
    record_parser.add_argument("--root", default=None, help="Override dataset root directory.")
    record_parser.add_argument("--repo-id", default=None, help="Override dataset repo_id.")
    record_parser.add_argument(
        "--instruction", default=None, help="Override the task instruction used for recording."
    )
    record_parser.add_argument(
        "--fps", type=int, default=None, help="Override dataset recording FPS."
    )
    record_parser.add_argument(
        "--display-data",
        action="store_true",
        help="Display camera and state data during recording.",
    )

    replay_parser = subparsers.add_parser(
        "replay", help="Replay one recorded episode for a named task."
    )
    add_common_config_arg(replay_parser)
    replay_parser.add_argument("--task", required=True, help="Task key from the tasks section.")
    replay_parser.add_argument(
        "--episode", type=int, required=True, help="Episode index to replay."
    )
    replay_parser.add_argument("--root", default=None, help="Override dataset root directory.")
    replay_parser.add_argument("--repo-id", default=None, help="Override dataset repo_id.")
    replay_parser.add_argument("--fps", type=int, default=None, help="Override replay FPS.")

    async_parser = subparsers.add_parser(
        "async-client",
        help="Run the remote ServoVLA robot client using the mode configured in jetson_workspace.yaml.",
    )
    add_common_config_arg(async_parser)
    async_parser.add_argument(
        "--task", default=None, help="Task key from tasks section; uses its instruction text."
    )
    async_parser.add_argument(
        "--task-text", default=None, help="Raw task text override for remote policy execution."
    )

    list_parser = subparsers.add_parser("list-tasks", help="Print available task keys.")
    add_common_config_arg(list_parser)
    return parser.parse_args()


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {config_path}, got {type(data)!r}")
    workspace_cfg = data.get("workspace", {})
    if isinstance(workspace_cfg, dict):
        for key in ("root", "log_dir"):
            if workspace_cfg.get(key):
                workspace_cfg[key] = _resolve_project_path(workspace_cfg[key])
    record_cfg = data.get("record", {})
    if isinstance(record_cfg, dict) and record_cfg.get("dataset_root"):
        record_cfg["dataset_root"] = _resolve_project_path(record_cfg["dataset_root"])
    async_cfg = data.get("async_client", {})
    if isinstance(async_cfg, dict) and async_cfg.get("pretrained_name_or_path"):
        async_cfg["pretrained_name_or_path"] = _resolve_project_path(
            async_cfg["pretrained_name_or_path"]
        )
    return data


def _copy_section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping, got {type(value)!r}")
    return copy.deepcopy(value)


def _validate_chunk_size_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 0.5:
        raise ValueError(f"chunk_size_threshold must be in [0, 0.5], got {threshold}")
    return threshold


def _robot_client_delay_debug(client) -> dict[str, object]:
    return {
        "latest_executed_action_timestep": getattr(client, "latest_executed_action_timestep", None),
        "action_queue_size": getattr(client, "action_queue_size", None),
        "action_chunk_size": int(client.config.actions_per_chunk),
        "chunk_size_threshold": float(client.config.chunk_size_threshold),
    }


def _resolve_task(cfg: dict[str, Any], task_name: str | None) -> tuple[str | None, dict[str, Any]]:
    tasks = cfg.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("Expected 'tasks' section to be a mapping.")

    resolved_name = task_name or cfg.get("workspace", {}).get("default_task")
    if resolved_name is None:
        return None, {}

    if resolved_name not in tasks:
        available = ", ".join(sorted(tasks))
        raise KeyError(f"Unknown task '{resolved_name}'. Available tasks: {available}")

    task_cfg = tasks[resolved_name]
    if not isinstance(task_cfg, dict):
        raise ValueError(f"Task '{resolved_name}' must be a mapping, got {type(task_cfg)!r}")
    return resolved_name, copy.deepcopy(task_cfg)


def _build_camera_config(cfg: dict[str, Any]):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    camera_type = str(cfg.get("type", "opencv")).lower()
    common = {
        "fps": int(cfg["fps"]),
        "width": int(cfg["width"]),
        "height": int(cfg["height"]),
    }

    if camera_type == "opencv":
        return OpenCVCameraConfig(
            index_or_path=cfg["index_or_path"],
            color_mode=cfg.get("color_mode", "rgb"),
            rotation=cfg.get("rotation", 0),
            warmup_s=int(cfg.get("warmup_s", 1)),
            fourcc=cfg.get("fourcc"),
            backend=cfg.get("backend", 0),
            **common,
        )

    if camera_type in {"intelrealsense", "realsense"}:
        return RealSenseCameraConfig(
            serial_number_or_name=str(cfg["serial_number_or_name"]),
            color_mode=cfg.get("color_mode", "rgb"),
            use_depth=bool(cfg.get("use_depth", False)),
            rotation=cfg.get("rotation", 0),
            warmup_s=int(cfg.get("warmup_s", 1)),
            **common,
        )

    raise ValueError(f"Unsupported camera type '{camera_type}'. Use 'opencv' or 'intelrealsense'.")


def _build_robot_config(cfg: dict[str, Any]):
    from lerobot.robots.so_follower import SO100FollowerConfig, SO101FollowerConfig

    robot_type = str(cfg["type"]).lower()
    cameras = {
        camera_name: _build_camera_config(camera_cfg)
        for camera_name, camera_cfg in cfg.get("cameras", {}).items()
    }
    kwargs = {
        "id": str(cfg.get("id", "")) or None,
        "port": str(cfg["port"]),
        "cameras": cameras,
        "disable_torque_on_disconnect": bool(cfg.get("disable_torque_on_disconnect", True)),
        "use_degrees": bool(cfg.get("use_degrees", True)),
    }
    if cfg.get("max_relative_target") is not None:
        kwargs["max_relative_target"] = cfg["max_relative_target"]

    if robot_type == "so101_follower":
        return SO101FollowerConfig(**kwargs)
    if robot_type == "so100_follower":
        return SO100FollowerConfig(**kwargs)

    raise ValueError(
        f"Unsupported robot type '{robot_type}'. This launcher supports so101_follower and so100_follower."
    )


def _build_teleop_config(cfg: dict[str, Any]):
    from lerobot.teleoperators.so_leader import SO100LeaderConfig, SO101LeaderConfig

    teleop_type = str(cfg["type"]).lower()
    kwargs = {
        "id": str(cfg.get("id", "")) or None,
        "port": str(cfg["port"]),
        "use_degrees": bool(cfg.get("use_degrees", True)),
    }

    if teleop_type == "so101_leader":
        return SO101LeaderConfig(**kwargs)
    if teleop_type == "so100_leader":
        return SO100LeaderConfig(**kwargs)

    raise ValueError(
        f"Unsupported teleop type '{teleop_type}'. This launcher supports so101_leader and so100_leader."
    )


def _extract_joint_state_snapshot(
    raw_observation: dict[str, Any], joint_keys: list[str]
) -> OrderedDict[str, float]:
    snapshot: OrderedDict[str, float] = OrderedDict()
    for key in joint_keys:
        value = raw_observation.get(key)
        if value is None:
            continue
        if hasattr(value, "item"):
            value = value.item()
        snapshot[key] = float(value)
    return snapshot


def _build_teleop_diagnostics_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = cfg.get("teleop_diagnostics", {})
    if raw_cfg is None:
        raw_cfg = {}
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"Expected 'teleop_diagnostics' to be a mapping, got {type(raw_cfg)!r}")
    return {
        "enabled": bool(raw_cfg.get("enabled", False)),
        "log_every_n_steps": max(1, int(raw_cfg.get("log_every_n_steps", 10))),
        "joint_keys": [str(key) for key in raw_cfg.get("joint_keys", []) or []],
    }


def _joint_delta_snapshot(
    reference_snapshot: OrderedDict[str, float],
    follower_snapshot: OrderedDict[str, float],
) -> OrderedDict[str, float]:
    return OrderedDict(
        (key, reference_snapshot[key] - follower_snapshot[key])
        for key in reference_snapshot
        if key in follower_snapshot
    )


def _build_teleop_joint_diagnostic_record(
    *,
    step_idx: int,
    leader_action: dict[str, Any],
    follower_observation: dict[str, Any],
    robot_action_to_send: dict[str, Any],
    robot_joint_keys: list[str],
    diag_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not diag_cfg.get("enabled", False):
        return None

    log_every_n_steps = max(1, int(diag_cfg.get("log_every_n_steps", 10)))
    if step_idx % log_every_n_steps != 0:
        return None

    selected_joint_keys = [str(key) for key in diag_cfg.get("joint_keys", []) or []]
    if not selected_joint_keys:
        selected_joint_keys = list(robot_joint_keys)

    leader_snapshot = _extract_joint_state_snapshot(leader_action, selected_joint_keys)
    follower_snapshot = _extract_joint_state_snapshot(follower_observation, selected_joint_keys)
    command_snapshot = _extract_joint_state_snapshot(robot_action_to_send, selected_joint_keys)

    return {
        "step": int(step_idx),
        "leader": leader_snapshot,
        "follower": follower_snapshot,
        "command": command_snapshot,
        "leader_error": _joint_delta_snapshot(leader_snapshot, follower_snapshot),
        "command_error": _joint_delta_snapshot(command_snapshot, follower_snapshot),
    }


def _log_teleop_joint_diagnostic_record(logger: logging.Logger, record: dict[str, Any]) -> None:
    logger.info(
        "Teleop joint diag | step=%s | leader=%s | follower=%s | command=%s | leader_error=%s | command_error=%s",
        record["step"],
        dict(record["leader"]),
        dict(record["follower"]),
        dict(record["command"]),
        dict(record["leader_error"]),
        dict(record["command_error"]),
    )


def _maybe_align_follower_to_leader(client, cfg: dict[str, Any]) -> None:
    import time

    from lerobot.teleoperators import make_teleoperator_from_config

    async_cfg = _copy_section(cfg, "async_client")
    if not bool(async_cfg.get("auto_align_to_leader_before_test", True)):
        client.logger.info("Skipping leader-to-follower alignment before remote test.")
        return

    align_timeout_s = float(async_cfg.get("align_timeout_s", 12.0))
    align_tolerance = float(async_cfg.get("align_tolerance", 3.0))
    align_settle_s = float(async_cfg.get("align_settle_s", 0.5))
    joint_keys = list(client.robot.action_features.keys())

    teleop = make_teleoperator_from_config(_build_teleop_config(_copy_section(cfg, "teleop")))
    teleop.connect(calibrate=False)
    client.logger.info(
        "Aligning follower to leader before remote test | timeout=%.1fs | tolerance=%.2f",
        align_timeout_s,
        align_tolerance,
    )

    try:
        deadline = time.time() + align_timeout_s
        while time.time() < deadline:
            leader_action = teleop.get_action()
            leader_joint_snapshot = OrderedDict(
                (key, float(leader_action[key])) for key in joint_keys if key in leader_action
            )
            client.robot.send_action(leader_action)
            time.sleep(align_settle_s)
            follower_raw = client.robot.get_observation()
            follower_joint_snapshot = _extract_joint_state_snapshot(follower_raw, joint_keys)

            common_keys = [
                key
                for key in joint_keys
                if key in leader_joint_snapshot and key in follower_joint_snapshot
            ]
            if not common_keys:
                raise RuntimeError(
                    "No overlapping joint keys were found during leader-to-follower alignment."
                )

            max_err = max(
                abs(leader_joint_snapshot[key] - follower_joint_snapshot[key])
                for key in common_keys
            )
            client.logger.info(
                "Alignment step | max_err=%.3f | leader=%s | follower=%s",
                max_err,
                dict(leader_joint_snapshot),
                dict(follower_joint_snapshot),
            )
            if max_err <= align_tolerance:
                client.logger.info("Leader-to-follower alignment complete.")
                return

        raise RuntimeError(
            f"Failed to align follower to leader within {align_timeout_s:.1f}s (tolerance={align_tolerance:.2f})."
        )
    finally:
        teleop.disconnect()


def _task_instruction(task_cfg: dict[str, Any], fallback: str = "") -> str:
    for key in ("language_instruction", "single_task", "task"):
        value = task_cfg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def _resolve_dataset_root(
    *,
    task_cfg: dict[str, Any],
    record_defaults: dict[str, Any],
    repo_id: str,
) -> str | None:
    explicit_root = task_cfg.get("root")
    if explicit_root:
        return str(explicit_root)

    base_root = record_defaults.get("dataset_root")
    if not base_root:
        return None

    repo_leaf = repo_id.split("/")[-1]
    return str(Path(base_root).expanduser() / repo_leaf)


def _install_record_rerun_patch(record_module, record_defaults: dict[str, Any]) -> None:
    display_every_n_frames = max(1, int(record_defaults.get("display_every_n_frames", 1)))
    display_downsample_factor = max(1, int(record_defaults.get("display_downsample_factor", 1)))

    if display_every_n_frames == 1 and display_downsample_factor == 1:
        return

    import cv2
    import numpy as np

    original_log_rerun_data = record_module.log_rerun_data
    state = {"frame_idx": 0}

    def _is_image_array(value: Any) -> bool:
        return isinstance(value, np.ndarray) and value.ndim == 3

    def _downsample_image(image: np.ndarray) -> np.ndarray:
        resized = image
        channels_first = False

        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            channels_first = True
            resized = np.transpose(image, (1, 2, 0))

        if display_downsample_factor > 1:
            height, width = resized.shape[:2]
            target_width = max(1, width // display_downsample_factor)
            target_height = max(1, height // display_downsample_factor)
            resized = cv2.resize(
                resized, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

        if channels_first:
            resized = np.transpose(resized, (2, 0, 1))

        return resized

    def patched_log_rerun_data(
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        compress_images: bool = False,
    ) -> None:
        frame_idx = state["frame_idx"]
        state["frame_idx"] = frame_idx + 1
        should_emit_images = frame_idx % display_every_n_frames == 0

        patched_observation = observation
        if observation:
            patched_observation = {}
            for key, value in observation.items():
                if _is_image_array(value):
                    if not should_emit_images:
                        continue
                    patched_observation[key] = _downsample_image(value)
                else:
                    patched_observation[key] = value

        original_log_rerun_data(
            observation=patched_observation,
            action=action,
            compress_images=compress_images,
        )

    record_module.log_rerun_data = patched_log_rerun_data
    logging.info(
        "Installed Rerun display patch for record: every_n_frames=%s downsample_factor=%s",
        display_every_n_frames,
        display_downsample_factor,
    )


def _build_robot_client_cfg(cfg: dict[str, Any], args: argparse.Namespace):
    from lerobot.async_inference.configs import RobotClientConfig

    _, task_cfg = _resolve_task(cfg, args.task)
    async_cfg = _copy_section(cfg, "async_client")

    task_text = args.task_text
    if not task_text:
        task_text = _task_instruction(task_cfg, fallback=str(async_cfg.get("task", "")))

    client_cfg = RobotClientConfig(
        policy_type=str(async_cfg["policy_type"]),
        pretrained_name_or_path=str(async_cfg["pretrained_name_or_path"]),
        robot=_build_robot_config(_copy_section(cfg, "robot")),
        actions_per_chunk=int(async_cfg["actions_per_chunk"]),
        task=task_text,
        server_address=str(async_cfg["server_address"]),
        policy_device=str(async_cfg.get("policy_device", "cuda")),
        client_device=str(async_cfg.get("client_device", "cpu")),
        chunk_size_threshold=_validate_chunk_size_threshold(
            async_cfg.get("chunk_size_threshold", 0.5)
        ),
        fps=int(async_cfg.get("fps", 10)),
        aggregate_fn_name=str(async_cfg.get("aggregate_fn_name", "weighted_average")),
        debug_visualize_queue_size=bool(async_cfg.get("debug_visualize_queue_size", False)),
    )
    return client_cfg, async_cfg


def _run_async_client(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    import threading

    from lerobot.async_inference.helpers import visualize_action_queue_size
    from lerobot.async_inference.robot_client import RobotClient

    client_cfg, _async_cfg = _build_robot_client_cfg(cfg, args)

    client = RobotClient(client_cfg)
    _maybe_align_follower_to_leader(client, cfg)
    if not client.start():
        raise RuntimeError("Failed to start RobotClient.")

    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()

    try:
        client.control_loop(task=client_cfg.task)
    finally:
        logging.info("RobotClient delay debug | %s", _robot_client_delay_debug(client))
        client.stop()
        action_receiver_thread.join()
        if client_cfg.debug_visualize_queue_size:
            visualize_action_queue_size(client.action_queue_size)


def _run_sync_client(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    import pickle
    import time

    from lerobot.async_inference.helpers import TimedObservation
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.transport import services_pb2

    client_cfg, async_cfg = _build_robot_client_cfg(cfg, args)
    sync_get_actions_timeout_s = float(
        async_cfg.get(
            "sync_get_actions_timeout_s",
            max(2.0, client_cfg.environment_dt * max(client_cfg.actions_per_chunk, 1) * 2.0),
        )
    )
    sync_poll_interval_s = float(async_cfg.get("sync_poll_interval_s", 0.01))

    client = RobotClient(client_cfg)
    _maybe_align_follower_to_leader(client, cfg)
    if not client.start():
        raise RuntimeError("Failed to start RobotClient.")

    client.logger.info(
        "Running RobotClient in sync mode | actions_per_chunk=%d | timeout=%.2fs | poll_interval=%.3fs",
        client_cfg.actions_per_chunk,
        sync_get_actions_timeout_s,
        sync_poll_interval_s,
    )

    sync_timestep = 0
    try:
        while client.running:
            loop_start = time.perf_counter()
            raw_observation = client.robot.get_observation()
            raw_observation["task"] = client_cfg.task

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=sync_timestep,
                must_go=True,
            )
            if not client.send_observation(observation):
                raise RuntimeError(f"Failed to send observation #{observation.get_timestep()}")

            wait_start = time.perf_counter()
            timed_actions = None
            while time.perf_counter() - wait_start < sync_get_actions_timeout_s and client.running:
                actions_chunk = client.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    time.sleep(sync_poll_interval_s)
                    continue
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                if timed_actions:
                    break
                time.sleep(sync_poll_interval_s)

            if not timed_actions:
                raise RuntimeError(
                    f"Timed out waiting for sync actions after observation #{observation.get_timestep()}"
                )

            selected_action = timed_actions[0]
            client.robot.send_action(
                client._action_tensor_to_action_dict(selected_action.get_action())
            )
            with client.latest_action_lock:
                client.latest_action = sync_timestep

            action_tensor_cpu = selected_action.get_action().detach().cpu().flatten()
            q_current_cpu = _extract_joint_state_snapshot(
                raw_observation, list(client.robot.action_features.keys())
            )
            client.logger.info(
                "Sync step | obs_timestep=%s | action_timestep=%s | chunk_len=%s | wait_time=%.3fs | q_current=%s | action=%s",
                observation.get_timestep(),
                selected_action.get_timestep(),
                len(timed_actions),
                time.perf_counter() - wait_start,
                q_current_cpu,
                action_tensor_cpu.tolist(),
            )

            sync_timestep += 1
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.0, client.config.environment_dt - elapsed))
    finally:
        client.stop()


def _run_teleop(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig, teleoperate

    teleop_cfg = _build_teleop_config(_copy_section(cfg, "teleop"))
    robot_cfg = _build_robot_config(_copy_section(cfg, "robot"))
    record_defaults = _copy_section(cfg, "record")
    teleop_diag_cfg = _build_teleop_diagnostics_cfg(cfg)

    teleop_cfg_obj = TeleoperateConfig(
        teleop=teleop_cfg,
        robot=robot_cfg,
        fps=int(args.fps or record_defaults.get("fps", 30)),
        display_data=bool(args.display_data or record_defaults.get("display_data", False)),
        display_ip=record_defaults.get("display_ip"),
        display_port=record_defaults.get("display_port"),
    )
    if not teleop_diag_cfg["enabled"]:
        teleoperate(teleop_cfg_obj)
        return

    import time
    from dataclasses import asdict
    from pprint import pformat

    import rerun as rr
    from lerobot.processor import make_default_processors
    from lerobot.robots import make_robot_from_config
    from lerobot.teleoperators import make_teleoperator_from_config
    from lerobot.utils.robot_utils import precise_sleep
    from lerobot.utils.utils import init_logging, move_cursor_up
    from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

    init_logging()
    logging.info(pformat(asdict(teleop_cfg_obj)))
    logging.info("Teleop joint diagnostics enabled: %s", teleop_diag_cfg)

    if teleop_cfg_obj.display_data:
        init_rerun(
            session_name="teleoperation",
            ip=teleop_cfg_obj.display_ip,
            port=teleop_cfg_obj.display_port,
        )
    display_compressed_images = (
        True
        if (
            teleop_cfg_obj.display_data
            and teleop_cfg_obj.display_ip is not None
            and teleop_cfg_obj.display_port is not None
        )
        else teleop_cfg_obj.display_compressed_images
    )

    teleop = make_teleoperator_from_config(teleop_cfg_obj.teleop)
    robot = make_robot_from_config(teleop_cfg_obj.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    teleop.connect()
    robot.connect()

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
    step_idx = 0

    try:
        while True:
            loop_start = time.perf_counter()

            obs = robot.get_observation()
            raw_action = teleop.get_action()
            teleop_action = teleop_action_processor((raw_action, obs))
            robot_action_to_send = robot_action_processor((teleop_action, obs))
            robot.send_action(robot_action_to_send)

            record = _build_teleop_joint_diagnostic_record(
                step_idx=step_idx,
                leader_action=teleop_action,
                follower_observation=obs,
                robot_action_to_send=robot_action_to_send,
                robot_joint_keys=list(robot.action_features.keys()),
                diag_cfg=teleop_diag_cfg,
            )
            if record is not None:
                _log_teleop_joint_diagnostic_record(logging.getLogger(__name__), record)

            if teleop_cfg_obj.display_data:
                obs_transition = robot_observation_processor(obs)
                log_rerun_data(
                    observation=obs_transition,
                    action=teleop_action,
                    compress_images=display_compressed_images,
                )

                print("\n" + "-" * (display_len + 10))
                print(f"{'NAME':<{display_len}} | {'NORM':>7}")
                for motor, value in robot_action_to_send.items():
                    print(f"{motor:<{display_len}} | {value:>7.2f}")
                move_cursor_up(len(robot_action_to_send) + 3)

            dt_s = time.perf_counter() - loop_start
            precise_sleep(max(1 / teleop_cfg_obj.fps - dt_s, 0.0))
            loop_s = time.perf_counter() - loop_start
            print(f"Teleop loop time: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")
            move_cursor_up(1)

            step_idx += 1
            if (
                teleop_cfg_obj.teleop_time_s is not None
                and time.perf_counter() - start >= teleop_cfg_obj.teleop_time_s
            ):
                return
    except KeyboardInterrupt:
        pass
    finally:
        if teleop_cfg_obj.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def _run_record(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    import lerobot.scripts.lerobot_record as record_module

    _, task_cfg = _resolve_task(cfg, args.task)
    record_defaults = _copy_section(cfg, "record")
    repo_id = str(args.repo_id or task_cfg.get("repo_id") or task_cfg.get("dataset_repo_id"))

    _install_record_rerun_patch(record_module, record_defaults)

    dataset_cfg = record_module.DatasetRecordConfig(
        repo_id=repo_id,
        single_task=str(args.instruction or _task_instruction(task_cfg)),
        root=args.root
        or _resolve_dataset_root(
            task_cfg=task_cfg, record_defaults=record_defaults, repo_id=repo_id
        ),
        fps=int(args.fps or task_cfg.get("fps") or record_defaults.get("fps", 30)),
        episode_time_s=float(
            task_cfg.get("episode_time_s", record_defaults.get("episode_time_s", 60))
        ),
        reset_time_s=float(task_cfg.get("reset_time_s", record_defaults.get("reset_time_s", 60))),
        num_episodes=int(
            args.episodes or task_cfg.get("num_episodes", record_defaults.get("num_episodes", 50))
        ),
        video=bool(task_cfg.get("video", record_defaults.get("video", True))),
        push_to_hub=bool(task_cfg.get("push_to_hub", record_defaults.get("push_to_hub", False))),
        private=bool(task_cfg.get("private", record_defaults.get("private", False))),
        tags=list(task_cfg.get("tags", record_defaults.get("tags", [])) or []),
        num_image_writer_processes=int(record_defaults.get("num_image_writer_processes", 0)),
        num_image_writer_threads_per_camera=int(
            record_defaults.get("num_image_writer_threads_per_camera", 4)
        ),
        video_encoding_batch_size=int(record_defaults.get("video_encoding_batch_size", 1)),
        vcodec=str(record_defaults.get("vcodec", "auto")),
        streaming_encoding=bool(record_defaults.get("streaming_encoding", True)),
        encoder_queue_maxsize=int(record_defaults.get("encoder_queue_maxsize", 30)),
        encoder_threads=record_defaults.get("encoder_threads", None),
        rename_map=dict(record_defaults.get("rename_map", {}) or {}),
    )

    record_cfg = record_module.RecordConfig(
        robot=_build_robot_config(_copy_section(cfg, "robot")),
        teleop=_build_teleop_config(_copy_section(cfg, "teleop")),
        dataset=dataset_cfg,
        display_data=bool(args.display_data or record_defaults.get("display_data", False)),
        display_ip=record_defaults.get("display_ip"),
        display_port=record_defaults.get("display_port"),
        play_sounds=bool(record_defaults.get("play_sounds", True)),
        resume=bool(args.resume or record_defaults.get("resume", False)),
    )
    record_module.record(record_cfg)


def _run_replay(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    from lerobot.scripts.lerobot_replay import DatasetReplayConfig, ReplayConfig, replay

    _, task_cfg = _resolve_task(cfg, args.task)
    record_defaults = _copy_section(cfg, "record")
    repo_id = str(args.repo_id or task_cfg.get("repo_id") or task_cfg.get("dataset_repo_id"))
    replay_cfg = ReplayConfig(
        robot=_build_robot_config(_copy_section(cfg, "robot")),
        dataset=DatasetReplayConfig(
            repo_id=repo_id,
            root=args.root
            or _resolve_dataset_root(
                task_cfg=task_cfg, record_defaults=record_defaults, repo_id=repo_id
            ),
            episode=int(args.episode),
            fps=int(args.fps or task_cfg.get("fps") or record_defaults.get("fps", 30)),
        ),
        play_sounds=bool(record_defaults.get("play_sounds", True)),
    )
    replay(replay_cfg)


def main() -> None:
    register_third_party_plugins()
    args = parse_args()
    cfg = _load_config(args.config)

    if args.command == "teleop":
        _run_teleop(cfg, args)
        return
    if args.command == "record":
        _run_record(cfg, args)
        return
    if args.command == "replay":
        _run_replay(cfg, args)
        return
    if args.command == "async-client":
        async_cfg = _copy_section(cfg, "async_client")
        client_mode = str(async_cfg.get("mode", "sync")).strip().lower()
        if client_mode == "async":
            _run_async_client(cfg, args)
            return
        if client_mode == "sync":
            _run_sync_client(cfg, args)
            return
        raise ValueError(f"Unsupported async_client.mode '{client_mode}'. Use 'async' or 'sync'.")
    if args.command == "list-tasks":
        tasks = cfg.get("tasks", {})
        if not isinstance(tasks, dict):
            raise ValueError("Expected 'tasks' section to be a mapping.")
        for task_name in sorted(tasks):
            print(task_name)
        return

    raise ValueError(f"Unsupported command '{args.command}'")


if __name__ == "__main__":
    main()
