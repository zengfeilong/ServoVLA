from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from servovla.config.paths import PROJECT_ROOT, resolve_project_path


@dataclass(frozen=True)
class LiberoTaskSpec:
    suite: str
    suite_task_id: int
    task_index: int
    task_name: str
    language: str


@dataclass(frozen=True)
class LiberoProtocol:
    suite: str
    tasks: tuple[LiberoTaskSpec, ...]
    source: str | None = None
    suites: tuple[str, ...] = ()


def default_libero_10_protocol_path() -> Path:
    return PROJECT_ROOT / "configs" / "eval" / "libero_10_tasks.json"


def default_libero_40_protocol_path() -> Path:
    return PROJECT_ROOT / "configs" / "eval" / "libero_40_tasks.json"


def load_libero_protocol(path: str | Path | None = None) -> LiberoProtocol:
    protocol_path = (
        resolve_project_path(path) if path is not None else default_libero_40_protocol_path()
    )
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    tasks = tuple(
        LiberoTaskSpec(
            suite=str(task.get("suite", payload.get("suite", "libero_40"))),
            suite_task_id=int(task["suite_task_id"]),
            task_index=int(task["task_index"]),
            task_name=str(task["task_name"]),
            language=str(task["language"]),
        )
        for task in payload["tasks"]
    )
    return LiberoProtocol(
        suite=str(payload.get("suite", "libero_40")),
        tasks=tasks,
        source=payload.get("source"),
        suites=tuple(str(item) for item in payload.get("suites", ())),
    )


def load_libero_10_protocol(path: str | Path | None = None) -> LiberoProtocol:
    return load_libero_protocol(default_libero_10_protocol_path() if path is None else path)


def load_libero_40_protocol(path: str | Path | None = None) -> LiberoProtocol:
    return load_libero_protocol(default_libero_40_protocol_path() if path is None else path)


def libero_10_task_indices(path: str | Path | None = None) -> tuple[int, ...]:
    return tuple(task.task_index for task in load_libero_10_protocol(path).tasks)


def libero_40_task_indices(path: str | Path | None = None) -> tuple[int, ...]:
    return tuple(task.task_index for task in load_libero_40_protocol(path).tasks)


def _as_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _normalize_task_index(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def validate_libero_item(
    item: dict[str, Any],
    *,
    camera_keys: tuple[str, str] = ("observation.images.image", "observation.images.image2"),
    proprio_dim: int = 8,
    action_dim: int = 7,
    allowed_task_indices: tuple[int, ...] | None = None,
) -> None:
    if "task" not in item:
        raise KeyError("LIBERO item is missing task text under key 'task'.")

    task = item["task"]
    if isinstance(task, (list, tuple)):
        if not task or not isinstance(task[0], str):
            raise TypeError("LIBERO task field must contain at least one string instruction.")
    elif not isinstance(task, str):
        raise TypeError("LIBERO task field must be a string or a sequence of strings.")

    if "task_index" not in item:
        raise KeyError("LIBERO item is missing 'task_index'.")
    task_index = _normalize_task_index(item["task_index"])
    if allowed_task_indices is not None and task_index not in set(
        int(x) for x in allowed_task_indices
    ):
        raise ValueError(f"LIBERO task_index {task_index} is outside the allowed benchmark subset.")

    for camera_key in camera_keys:
        if camera_key not in item:
            raise KeyError(f"LIBERO item is missing camera key '{camera_key}'.")
        image = _as_tensor(item[camera_key])
        if image.ndim != 3:
            raise ValueError(f"LIBERO image must be rank-3, got shape {tuple(image.shape)}")
        if image.shape[0] not in {1, 3} and image.shape[-1] not in {1, 3}:
            raise ValueError(
                f"LIBERO image must be CHW or HWC with 1/3 channels, got {tuple(image.shape)}"
            )

    if "observation.state" not in item:
        raise KeyError("LIBERO item is missing 'observation.state'.")
    state = _as_tensor(item["observation.state"], dtype=torch.float32).flatten()
    if state.numel() < proprio_dim:
        raise ValueError(
            f"LIBERO state has {state.numel()} values, expected at least {proprio_dim}."
        )

    if "action" not in item:
        raise KeyError("LIBERO item is missing 'action'.")
    action = _as_tensor(item["action"], dtype=torch.float32)
    if action.ndim != 2:
        raise ValueError(f"LIBERO action chunk must be rank-2, got shape {tuple(action.shape)}")
    if action.shape[-1] != action_dim:
        raise ValueError(f"LIBERO action dim must be {action_dim}, got {action.shape[-1]}.")
