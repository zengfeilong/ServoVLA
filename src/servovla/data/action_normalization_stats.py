from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from servovla.config.action_mode import normalize_action_delta_state_indices


@dataclass(frozen=True, slots=True)
class ActionTargetStats:
    mean: list[list[float]]
    std: list[list[float]]
    count: list[int]
    dataset_names: list[str]
    dataset_weights: list[float]
    action_mode: str
    action_delta_state_indices: list[int | None]
    chunk_size: int
    action_dim: int

    def to_dict(self) -> dict[str, object]:
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "dataset_names": self.dataset_names,
            "dataset_weights": self.dataset_weights,
            "action_mode": self.action_mode,
            "action_delta_state_indices": self.action_delta_state_indices,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
        }


def _fixed_size_list_array_to_numpy(column, *, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        list_size = int(array.type.list_size)
        values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        return values.reshape(len(array), list_size)[:, :width]
    return np.asarray(array.to_numpy(zero_copy_only=False).tolist(), dtype=np.float64)[:, :width]


def _read_dataset_arrays(
    dataset_root: Path,
    *,
    action_dim: int,
    state_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parquet_paths = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No LeRobot parquet files found under {dataset_root / 'data'}")

    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=["action", "observation.state", "episode_index", "frame_index"],
            )
            for path in parquet_paths
        ]
    )
    action = _fixed_size_list_array_to_numpy(table["action"], width=action_dim)
    state_width = max(int(state_dim or action_dim), action_dim)
    state = _fixed_size_list_array_to_numpy(table["observation.state"], width=state_width)
    episode_index = np.asarray(
        table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )
    frame_index = np.asarray(
        table["frame_index"].combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64
    )
    order = np.lexsort((frame_index, episode_index))
    return action[order], state[order], episode_index[order], frame_index[order]


def _dataset_target_sums(
    *,
    dataset_root: Path,
    selected_episodes: Iterable[int] | None,
    chunk_size: int,
    action_dim: int,
    action_is_delta: bool,
    action_delta_state_indices: Sequence[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized_state_indices = normalize_action_delta_state_indices(
        action_delta_state_indices,
        action_dim=action_dim,
    )
    max_state_idx = max(
        (int(idx) for idx in normalized_state_indices if idx is not None), default=-1
    )
    action, state, episode_index, frame_index = _read_dataset_arrays(
        dataset_root,
        action_dim=action_dim,
        state_dim=max(action_dim, max_state_idx + 1),
    )
    selected = None if selected_episodes is None else {int(value) for value in selected_episodes}
    target_sum = np.zeros((chunk_size, action_dim), dtype=np.float64)
    target_sumsq = np.zeros((chunk_size, action_dim), dtype=np.float64)
    target_count = np.zeros((chunk_size,), dtype=np.float64)

    for episode_id in np.unique(episode_index):
        if selected is not None and int(episode_id) not in selected:
            continue
        positions = np.where(episode_index == episode_id)[0]
        if positions.size == 0:
            continue
        positions = positions[np.argsort(frame_index[positions])]
        episode_action = action[positions, :action_dim]
        episode_state = state[positions]
        episode_len = int(positions.size)
        for horizon_idx in range(min(chunk_size, episode_len)):
            current_count = episode_len - horizon_idx
            future_action = episode_action[horizon_idx:]
            if action_is_delta:
                basis = np.zeros((current_count, action_dim), dtype=np.float64)
                for action_idx, state_idx in enumerate(normalized_state_indices):
                    if state_idx is None:
                        continue
                    if int(state_idx) >= episode_state.shape[-1]:
                        raise ValueError(
                            f"Action delta state index {state_idx} for action dim {action_idx} exceeds "
                            f"state dim {episode_state.shape[-1]}"
                        )
                    basis[:, action_idx] = episode_state[:current_count, int(state_idx)]
                target = future_action - basis
            else:
                target = future_action
            target_sum[horizon_idx] += target.sum(axis=0)
            target_sumsq[horizon_idx] += np.square(target).sum(axis=0)
            target_count[horizon_idx] += current_count

    return target_sum, target_sumsq, target_count


def compute_weighted_action_target_stats(
    *,
    dataset_roots: Sequence[Path],
    dataset_names: Sequence[str],
    dataset_weights: Sequence[float],
    selected_episodes_by_dataset: Sequence[Iterable[int] | None],
    chunk_size: int,
    action_dim: int,
    action_is_delta: bool,
    action_delta_state_indices: Sequence[int | None] | None = None,
    eps: float = 1.0e-6,
) -> ActionTargetStats:
    if not dataset_roots:
        raise ValueError(
            "At least one dataset root is required to compute action normalization stats."
        )
    if not (
        len(dataset_roots)
        == len(dataset_names)
        == len(dataset_weights)
        == len(selected_episodes_by_dataset)
    ):
        raise ValueError(
            "dataset_roots, dataset_names, dataset_weights, and selected_episodes_by_dataset must match."
        )

    chunk_size = int(chunk_size)
    action_dim = int(action_dim)
    if chunk_size <= 0 or action_dim <= 0:
        raise ValueError(
            f"chunk_size and action_dim must be positive, got {chunk_size=} {action_dim=}."
        )
    normalized_state_indices = normalize_action_delta_state_indices(
        action_delta_state_indices,
        action_dim=action_dim,
    )

    weighted_mean_sum = np.zeros((chunk_size, action_dim), dtype=np.float64)
    weighted_second_sum = np.zeros((chunk_size, action_dim), dtype=np.float64)
    weighted_presence = np.zeros((chunk_size,), dtype=np.float64)
    combined_count = np.zeros((chunk_size,), dtype=np.float64)

    for dataset_root, dataset_weight, selected_episodes in zip(
        dataset_roots,
        dataset_weights,
        selected_episodes_by_dataset,
        strict=True,
    ):
        target_sum, target_sumsq, target_count = _dataset_target_sums(
            dataset_root=Path(dataset_root),
            selected_episodes=selected_episodes,
            chunk_size=chunk_size,
            action_dim=action_dim,
            action_is_delta=action_is_delta,
            action_delta_state_indices=normalized_state_indices,
        )
        valid = target_count > 0
        if not valid.any():
            continue
        weight = float(dataset_weight)
        dataset_mean = np.zeros_like(target_sum)
        dataset_second = np.zeros_like(target_sumsq)
        dataset_mean[valid] = target_sum[valid] / target_count[valid, None]
        dataset_second[valid] = target_sumsq[valid] / target_count[valid, None]
        weighted_mean_sum[valid] += weight * dataset_mean[valid]
        weighted_second_sum[valid] += weight * dataset_second[valid]
        weighted_presence[valid] += weight
        combined_count[valid] += target_count[valid]

    valid = weighted_presence > 0
    if not valid.any():
        raise ValueError("No valid action targets were found for action normalization stats.")
    mean = np.zeros((chunk_size, action_dim), dtype=np.float64)
    second = np.ones((chunk_size, action_dim), dtype=np.float64)
    mean[valid] = weighted_mean_sum[valid] / weighted_presence[valid, None]
    second[valid] = weighted_second_sum[valid] / weighted_presence[valid, None]
    variance = np.maximum(second - np.square(mean), float(eps) ** 2)
    std = np.sqrt(variance)

    return ActionTargetStats(
        mean=mean.tolist(),
        std=std.tolist(),
        count=[int(value) for value in combined_count.tolist()],
        dataset_names=[str(value) for value in dataset_names],
        dataset_weights=[float(value) for value in dataset_weights],
        action_mode="delta" if action_is_delta else "abs",
        action_delta_state_indices=list(normalized_state_indices),
        chunk_size=chunk_size,
        action_dim=action_dim,
    )


def write_action_target_stats(path: str | Path, stats: ActionTargetStats) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")
    return output_path
