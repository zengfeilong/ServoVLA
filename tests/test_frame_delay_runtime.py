from __future__ import annotations

import random

import pytest
import torch

from servovla.data.dataset_loader import (
    OnlineVLADataset,
    action_step_delay_buckets,
    action_step_delay_support,
)
from tests.conftest import CHUNK_SIZE


class _FakeVisionProcessor:
    def __call__(self, *, images, return_tensors: str, size):
        return {"pixel_values": torch.zeros(len(images), 3, size["height"], size["width"])}


class _FakeVLMProcessor:
    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        return messages[0]["content"][-1]["text"]


class _FakeEpisodeDataset:
    def __init__(self, length: int = 10):
        self.length = length
        self.num_episodes = 1
        self.episode_data_index = {"from": [0], "to": [length]}

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        return {
            "episode_index": 0,
            "observation.images.front": torch.zeros(3, 8, 8),
            "observation.images.wrist": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(6),
            "action": torch.zeros(CHUNK_SIZE, 6),
            "task": f"task-{idx}",
        }


def test_action_step_delay_buckets_are_half_open_integer_windows():
    assert action_step_delay_buckets(
        chunk_size=16,
        chunk_size_threshold=0.25,
        max_delay_chunks=2,
    ) == [(12, 16), (27, 31)]


@pytest.mark.parametrize("threshold", [-0.01, 0.5001])
def test_action_step_delay_support_rejects_threshold_outside_contract(threshold):
    with pytest.raises(ValueError, match="chunk_size_threshold"):
        action_step_delay_support(
            chunk_size=16,
            chunk_size_threshold=threshold,
            max_delay_chunks=3,
        )


def test_action_step_delay_support_threshold_zero_keeps_async_chunk_boundaries():
    assert action_step_delay_support(
        chunk_size=16,
        chunk_size_threshold=0.0,
        max_delay_chunks=2,
    ) == [(0, 1), (15, 16), (30, 31)]
    assert action_step_delay_buckets(
        chunk_size=16,
        chunk_size_threshold=0.0,
        max_delay_chunks=2,
    ) == [(15, 16), (30, 31)]


def test_action_step_delay_support_threshold_point_two_keeps_narrow_two_chunk_windows():
    assert action_step_delay_support(
        chunk_size=16,
        chunk_size_threshold=0.2,
        max_delay_chunks=2,
    ) == [(0, 1), (13, 16), (28, 31)]


def test_online_vla_dataset_samples_action_step_delay_from_bucket_support():
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(length=64),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_FakeVisionProcessor(),
        action_horizon=16,
        vision_image_size=8,
        vlm_image_size=8,
        action_is_delta=True,
        delay_max_chunks=2,
        delay_chunk_size_threshold=0.25,
    )

    observed = []
    for seed in range(256):
        random.seed(seed)
        observed.append(int(dataset[40]["frame_delay"].item()))

    support = {0, *range(12, 16), *range(27, 31)}
    assert set(observed).issubset(support)
    assert any(delay >= 12 for delay in observed)


def test_online_vla_dataset_does_not_clamp_async_delay_into_support_gap_at_episode_start():
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(length=64),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_FakeVisionProcessor(),
        action_horizon=16,
        vision_image_size=8,
        vlm_image_size=8,
        action_is_delta=True,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.25,
        delay_sync_weight=0.0,
    )

    observed = []
    for seed in range(64):
        random.seed(seed)
        observed.append(int(dataset[5]["frame_delay"].item()))

    assert observed == [0] * 64
    assert dataset[5]["vlm_text"] == "task-5"


def test_online_vla_dataset_can_use_current_step_semantics():
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(length=10),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_FakeVisionProcessor(),
        action_horizon=CHUNK_SIZE,
        vision_image_size=8,
        vlm_image_size=8,
        action_is_delta=True,
        sample_frame_delay=False,
    )

    sample = dataset[4]

    assert sample["frame_delay"].item() == 0.0
    assert sample["vlm_text"] == "task-4"
