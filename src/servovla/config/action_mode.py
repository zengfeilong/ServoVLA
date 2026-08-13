from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

VALID_ACTION_MODES = {"delta", "abs"}


def normalize_action_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_ACTION_MODES:
        raise ValueError(
            f"dataset.action_mode must be one of {sorted(VALID_ACTION_MODES)}, got {value!r}"
        )
    return mode


def action_mode_is_delta(value: str) -> bool:
    return normalize_action_mode(value) == "delta"


def normalize_action_delta_state_indices(
    value: Sequence[int | None] | None,
    *,
    action_dim: int,
) -> tuple[int | None, ...]:
    action_dim = int(action_dim)
    if action_dim <= 0:
        raise ValueError(f"action_dim must be positive, got {action_dim}.")
    if value is None:
        return tuple(range(action_dim))

    indices = list(value)
    if len(indices) != action_dim:
        raise ValueError(
            f"dataset.action_delta_state_indices must have length action_dim={action_dim}, "
            f"got {len(indices)}."
        )

    normalized: list[int | None] = []
    for idx, item in enumerate(indices):
        if item is None:
            normalized.append(None)
            continue
        if isinstance(item, str) and item.strip().lower() in {"none", "null"}:
            normalized.append(None)
            continue
        state_idx = int(item)
        if state_idx < 0:
            raise ValueError(
                f"dataset.action_delta_state_indices[{idx}] must be non-negative or null, got {item!r}."
            )
        normalized.append(state_idx)
    return tuple(normalized)


def get_dataset_names(dataset_cfg, partition: str) -> list[str]:
    if partition not in {"train", "val"}:
        raise ValueError(f"partition must be 'train' or 'val', got {partition!r}")

    key = f"{partition}_names"
    names = list(getattr(dataset_cfg, key, []))
    if partition == "train" and not names:
        raise ValueError("dataset.train_names must not be empty")

    return [str(name) for name in names]


def resolve_mode_root(base_root: str | Path, action_mode: str) -> Path:
    return Path(base_root).expanduser().resolve() / normalize_action_mode(action_mode)
