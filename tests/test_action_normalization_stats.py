from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from servovla.data.action_normalization_stats import compute_weighted_action_target_stats


def _write_lerobot_parquet(root, *, actions, states):
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True)
    action_dim = len(actions[0])
    state_dim = len(states[0])
    table = pa.table(
        {
            "action": pa.array(actions, type=pa.list_(pa.float32(), action_dim)),
            "observation.state": pa.array(states, type=pa.list_(pa.float32(), state_dim)),
            "episode_index": pa.array([0 for _ in actions], type=pa.int64()),
            "frame_index": pa.array(list(range(len(actions))), type=pa.int64()),
        }
    )
    pq.write_table(table, data_dir / "file-000.parquet")


def test_compute_weighted_action_target_stats_uses_per_horizon_delta_targets(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_lerobot_parquet(
        dataset_root,
        actions=[[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]],
        states=[[0.0, 8.0], [2.0, 10.0], [4.0, 12.0]],
    )

    stats = compute_weighted_action_target_stats(
        dataset_roots=[dataset_root],
        dataset_names=["local/unit"],
        dataset_weights=[1.0],
        selected_episodes_by_dataset=[None],
        chunk_size=2,
        action_dim=2,
        action_is_delta=True,
    )

    assert stats.count == [3, 2]
    assert stats.mean[0] == pytest.approx([1.0, 4.0])
    assert stats.mean[1] == pytest.approx([3.0, 7.0])
    assert stats.std[0] == pytest.approx([0.0, 1.632993], abs=1e-5)
    assert stats.std[1] == pytest.approx([0.0, 1.0], abs=1e-5)


def test_compute_weighted_action_target_stats_uses_mapped_state_indices_wider_than_action(tmp_path):
    dataset_root = tmp_path / "dataset"
    _write_lerobot_parquet(
        dataset_root,
        actions=[[10.0, 20.0, -1.0], [11.0, 22.0, 1.0]],
        states=[
            [0.0, 1.0, 2.0, 100.0, 200.0],
            [0.0, 1.0, 2.0, 101.0, 202.0],
        ],
    )

    stats = compute_weighted_action_target_stats(
        dataset_roots=[dataset_root],
        dataset_names=["local/unit"],
        dataset_weights=[1.0],
        selected_episodes_by_dataset=[None],
        chunk_size=1,
        action_dim=3,
        action_is_delta=True,
        action_delta_state_indices=[3, 4, None],
    )

    assert stats.count == [2]
    assert stats.mean[0] == pytest.approx([-90.0, -180.0, 0.0])
    assert stats.action_delta_state_indices == [3, 4, None]
