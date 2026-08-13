from __future__ import annotations

import bisect
import logging
import math
import random
import time
from collections import OrderedDict, defaultdict
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from lerobot.datasets import dataset_reader as lerobot_dataset_reader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Sampler

from servovla.config.action_mode import normalize_action_delta_state_indices
from servovla.config.compile_padding import ATTENTION_SEQUENCE_MULTIPLE, round_up_to_multiple
from servovla.data import video_decode as servovla_video_decode
from servovla.data.indexing import PaddedSampleIndex, is_padded_sample_index, sample_index_value
from servovla.data.qwen_image_tokens import expand_qwen_image_tokens, qwen_image_grid_thw_for_square

log = logging.getLogger(__name__)

_RAW_VIDEO_BATCH_DECODE_MAX_SPAN_S = 2.0


class BatchAwareConcatDataset(ConcatDataset):
    def __getitems__(self, indices: Sequence[Any]) -> list[Any]:
        index_list = list(indices)
        if not index_list:
            return []

        total_length = len(self)
        grouped: dict[int, list[tuple[int, int | PaddedSampleIndex]]] = defaultdict(list)
        for position, raw_idx in enumerate(index_list):
            idx = sample_index_value(raw_idx)
            if idx < 0:
                idx += total_length
            if idx < 0 or idx >= total_length:
                raise IndexError(f"index {idx} is out of bounds for dataset of size {total_length}")
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            prev_cumulative = 0 if dataset_idx == 0 else int(self.cumulative_sizes[dataset_idx - 1])
            local_idx = idx - prev_cumulative
            local_ref: int | PaddedSampleIndex = (
                PaddedSampleIndex(local_idx) if is_padded_sample_index(raw_idx) else local_idx
            )
            grouped[dataset_idx].append((position, local_ref))

        results: list[Any] = [None] * len(index_list)
        for dataset_idx, entries in grouped.items():
            child = self.datasets[dataset_idx]
            local_indices = [local_idx for _position, local_idx in entries]
            if hasattr(child, "__getitems__"):
                child_results = child.__getitems__(local_indices)
            else:
                child_results = [
                    child[sample_index_value(local_idx)] for local_idx in local_indices
                ]
            if len(child_results) != len(entries):
                raise ValueError(
                    f"Concat child dataset {dataset_idx} returned {len(child_results)} items for {len(entries)} indices"
                )
            for (position, _local_idx), item in zip(entries, child_results, strict=True):
                results[position] = item
        return results


def action_step_delay_support(
    *,
    chunk_size: int,
    chunk_size_threshold: float,
    max_delay_chunks: int,
) -> list[tuple[int, int]]:
    chunk_size = int(chunk_size)
    chunk_size_threshold = float(chunk_size_threshold)
    max_delay_chunks = int(max_delay_chunks)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if max_delay_chunks < 1:
        raise ValueError(f"max_delay_chunks must be >= 1, got {max_delay_chunks}")
    if not 0.0 <= chunk_size_threshold <= 0.5:
        raise ValueError(f"chunk_size_threshold must be in [0, 0.5], got {chunk_size_threshold}")

    # Intervals are half-open: (28, 31) samples delays 28, 29, 30.
    # Runtime logs with actions_per_chunk=16 show one stale chunk lands at
    # delay 14/15, so the boundary stride is chunk_size - 1.
    support: list[tuple[int, int]] = [(0, 1)]
    bucket_width = max(1, int(math.floor(chunk_size * chunk_size_threshold)))
    boundary_stride = max(chunk_size - 1, 1)
    for bucket_idx in range(1, max_delay_chunks + 1):
        boundary = bucket_idx * boundary_stride
        start = max(1, boundary - bucket_width + 1)
        support.append((start, boundary + 1))
    return support


def action_step_delay_buckets(
    *,
    chunk_size: int,
    chunk_size_threshold: float,
    max_delay_chunks: int,
) -> list[tuple[int, int]]:
    return [
        interval
        for interval in action_step_delay_support(
            chunk_size=chunk_size,
            chunk_size_threshold=chunk_size_threshold,
            max_delay_chunks=max_delay_chunks,
        )
        if interval != (0, 1)
    ]


def _num_episodes(lerobot_dataset: LeRobotDataset) -> int:
    selected_episodes = getattr(lerobot_dataset, "episodes", None)
    if selected_episodes is not None:
        return len(selected_episodes)
    if hasattr(lerobot_dataset, "meta") and hasattr(lerobot_dataset.meta, "episodes"):
        episodes = lerobot_dataset.meta.episodes
        if isinstance(episodes, dict):
            first_column = next(iter(episodes.values()), [])
            return len(first_column)
        return len(episodes)
    hf_episode_ids = _episode_ids_from_hf_dataset(lerobot_dataset)
    if hf_episode_ids is not None:
        return len(hf_episode_ids)
    return lerobot_dataset.num_episodes


def _selected_episode_ids(lerobot_dataset: LeRobotDataset) -> list[int]:
    selected_episodes = getattr(lerobot_dataset, "episodes", None)
    if selected_episodes is not None:
        return [int(ep_idx) for ep_idx in selected_episodes]
    hf_episode_ids = _episode_ids_from_hf_dataset(lerobot_dataset)
    if hf_episode_ids is not None:
        return [int(ep_idx) for ep_idx in hf_episode_ids]
    return [int(ep_idx) for ep_idx in range(_num_episodes(lerobot_dataset))]


def selected_episode_ids(lerobot_dataset: LeRobotDataset) -> list[int]:
    return _selected_episode_ids(lerobot_dataset)


def _extract_scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, list):
        return int(value[0])
    if isinstance(value, tuple):
        return int(value[0])
    return int(value)


def _episode_column_bounds(episode_column: Sequence[Any]) -> dict[int, tuple[int, int]]:
    if len(episode_column) == 0:
        return {}

    bounds: dict[int, tuple[int, int]] = {}
    current_episode = _extract_scalar_int(episode_column[0])
    start_idx = 0
    for rel_idx, value in enumerate(episode_column[1:], start=1):
        episode_id = _extract_scalar_int(value)
        if episode_id == current_episode:
            continue
        bounds[current_episode] = (start_idx, rel_idx)
        current_episode = episode_id
        start_idx = rel_idx

    bounds[current_episode] = (start_idx, len(episode_column))
    return bounds


def _episode_bounds_from_hf_dataset(
    lerobot_dataset: LeRobotDataset,
) -> dict[int, tuple[int, int]] | None:
    hf_dataset = getattr(lerobot_dataset, "hf_dataset", None)
    if hf_dataset is None:
        return None

    try:
        episode_column = hf_dataset["episode_index"]
    except Exception:
        return None

    return _episode_column_bounds(episode_column)


def _episode_ids_from_hf_dataset(lerobot_dataset: LeRobotDataset) -> list[int] | None:
    bounds = _episode_bounds_from_hf_dataset(lerobot_dataset)
    if bounds is None:
        return None
    return list(bounds.keys())


def _filtered_episode_row_bounds(
    lerobot_dataset: LeRobotDataset,
) -> dict[int, tuple[int, int]] | None:
    if getattr(lerobot_dataset, "episodes", None) is None:
        return None
    return _episode_bounds_from_hf_dataset(lerobot_dataset)


def episode_row_bounds(lerobot_dataset: LeRobotDataset) -> dict[int, tuple[int, int]]:
    filtered_bounds = _filtered_episode_row_bounds(lerobot_dataset)
    if filtered_bounds is not None:
        return filtered_bounds

    selected_episode_ids_list = _selected_episode_ids(lerobot_dataset)
    bounds: dict[int, tuple[int, int]] = {}
    meta_episodes = (
        lerobot_dataset.meta.episodes
        if hasattr(lerobot_dataset, "meta") and hasattr(lerobot_dataset.meta, "episodes")
        else None
    )
    if meta_episodes is not None:
        for ep_idx in selected_episode_ids_list:
            try:
                start_idx = int(meta_episodes["dataset_from_index"][ep_idx])
                end_idx = int(meta_episodes["dataset_to_index"][ep_idx])
            except (KeyError, TypeError, IndexError):
                try:
                    start_idx = int(meta_episodes[ep_idx]["dataset_from_index"])
                    end_idx = int(meta_episodes[ep_idx]["dataset_to_index"])
                except (KeyError, TypeError, IndexError):
                    break
            bounds[int(ep_idx)] = (start_idx, end_idx)
        if len(bounds) == len(selected_episode_ids_list):
            return bounds

    hf_bounds = _episode_bounds_from_hf_dataset(lerobot_dataset)
    if hf_bounds is not None:
        return {
            int(ep_idx): hf_bounds[int(ep_idx)]
            for ep_idx in selected_episode_ids_list
            if int(ep_idx) in hf_bounds
        }

    for ep_idx in selected_episode_ids_list:
        start_idx = int(lerobot_dataset.episode_data_index["from"][ep_idx])
        end_idx = int(lerobot_dataset.episode_data_index["to"][ep_idx])
        bounds[int(ep_idx)] = (start_idx, end_idx)
    return bounds


def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _normalize_camera_aliases(
    camera_keys: Sequence[str],
    camera_key_aliases: dict[str, Sequence[str]] | None,
) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}
    for camera_key in camera_keys:
        canonical_name = camera_key.split(".")[-1]
        aliases = [camera_key]
        if camera_key_aliases and canonical_name in camera_key_aliases:
            aliases.extend(list(camera_key_aliases[canonical_name]))
        alias_map[camera_key] = list(dict.fromkeys(aliases))
    return alias_map


def _extract_camera_tensor(item: dict[str, Any], aliases: Sequence[str]) -> torch.Tensor:
    for key in aliases:
        if key in item:
            image = _to_tensor(item[key])
            if image.ndim == 2:
                if image.dtype != torch.uint8:
                    raise ValueError(
                        f"Expected uint8 NV12 image tensor for key '{key}', got {image.dtype}"
                    )
                return image.contiguous()
            if image.ndim != 3:
                raise ValueError(
                    f"Expected image tensor with 3 dims for key '{key}', got {tuple(image.shape)}"
                )
            if image.shape[0] not in {1, 3} and image.shape[-1] in {1, 3}:
                image = image.permute(2, 0, 1)
            if image.shape[0] not in {1, 3}:
                raise ValueError(
                    f"Unsupported image layout for key '{key}', got {tuple(image.shape)}"
                )
            return image.contiguous()
    raise KeyError(
        f"None of the camera aliases {list(aliases)} were found in dataset item keys={list(item.keys())}"
    )


def _tensor_to_pil(image: torch.Tensor, image_size: int) -> Image.Image:
    image = image.detach().cpu()
    if image.is_floating_point():
        image = image.clamp(0.0, 1.0).mul(255.0).round()
    image = image.to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image).resize((image_size, image_size), Image.Resampling.BILINEAR)


def _tensor_to_uint8_chw(image: torch.Tensor) -> torch.Tensor:
    image = image.detach().cpu()
    if image.is_floating_point():
        image = image.clamp(0.0, 1.0).mul(255.0).round()
    image = image.to(torch.uint8)
    if image.ndim == 2:
        height_times_3 = int(image.shape[0]) * 2
        if height_times_3 % 3 != 0:
            raise ValueError(
                f"Expected NV12 image height to be 1.5x RGB height, got {tuple(image.shape)}"
            )
        rgb_height = height_times_3 // 3
        if rgb_height % 2 != 0 or int(image.shape[1]) % 2 != 0:
            raise ValueError(f"Expected even NV12 image dimensions, got {tuple(image.shape)}")
        return image.contiguous()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image tensor with 3 dims, got {tuple(image.shape)}")
    if image.shape[0] == 1:
        image = image.expand(3, -1, -1)
    if image.shape[0] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {tuple(image.shape)}")
    return image.contiguous()


def _extract_task_text(item: dict[str, Any], task_key_candidates: Sequence[str]) -> str:
    for key in task_key_candidates:
        if key in item:
            value = item[key]
            if isinstance(value, str):
                return value
            if isinstance(value, Sequence) and value and isinstance(value[0], str):
                return value[0]
            return str(value)
    return ""


def _extract_q_current(
    item: dict[str, Any],
    state_keys: Sequence[str],
    proprio_dim: int,
) -> torch.Tensor:
    if "observation.state" in item:
        state = _to_tensor(item["observation.state"], dtype=torch.float32).flatten()
        if state.numel() < proprio_dim:
            raise ValueError(
                f"'observation.state' only has {state.numel()} elements, expected at least {proprio_dim}"
            )
        return state[:proprio_dim].contiguous()

    missing = [key for key in state_keys if key not in item]
    if missing:
        raise KeyError(
            "Unable to recover proprio state. Missing keys "
            f"{missing} and 'observation.state' was not present."
        )
    values = [
        _to_tensor(item[key], dtype=torch.float32).reshape(-1)[0]
        for key in state_keys[:proprio_dim]
    ]
    return torch.stack(values, dim=0).contiguous()


def _extract_action_chunk(item: dict[str, Any]) -> torch.Tensor:
    actions_chunk_abs = item["action"]
    if isinstance(actions_chunk_abs, list):
        if actions_chunk_abs and isinstance(actions_chunk_abs[0], torch.Tensor):
            actions_chunk_abs = torch.stack(actions_chunk_abs)
        else:
            actions_chunk_abs = torch.tensor(actions_chunk_abs)
    elif not isinstance(actions_chunk_abs, torch.Tensor):
        actions_chunk_abs = torch.tensor(actions_chunk_abs)
    return actions_chunk_abs.float().contiguous()


class SequentialEpisodeBatchSampler(Sampler):
    """
    Sequential stratified sampler optimized for LeRobot v3 and PyAV decoding.

    Episodes remain shuffled while frames within each episode are read in
    timestamp order, improving sequential video decode throughput.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        batch_size: int,
        drop_last: bool = True,
        shuffle_episodes: bool = True,
        is_ddp: bool = False,
        rank: int = 0,
        num_replicas: int = 1,
        episode_indices: Sequence[int] | None = None,
        start_step_offset: int = 0,
    ):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle_episodes = bool(shuffle_episodes)
        self.is_ddp = is_ddp
        self.rank = rank
        self.num_replicas = num_replicas
        self.start_step_offset = max(int(start_step_offset), 0)

        self.ep_to_indices = defaultdict(list)
        self.filtered_episode_bounds = _filtered_episode_row_bounds(lerobot_dataset)
        available_episode_ids = _selected_episode_ids(lerobot_dataset)

        for ep_idx in available_episode_ids:
            if self.filtered_episode_bounds is not None:
                bounds = self.filtered_episode_bounds.get(int(ep_idx))
                if bounds is None:
                    continue
                start_idx, end_idx = bounds
            elif hasattr(lerobot_dataset, "meta") and hasattr(lerobot_dataset.meta, "episodes"):
                try:
                    start_idx = int(lerobot_dataset.meta.episodes["dataset_from_index"][ep_idx])
                    end_idx = int(lerobot_dataset.meta.episodes["dataset_to_index"][ep_idx])
                except (KeyError, TypeError):
                    start_idx = int(lerobot_dataset.meta.episodes[ep_idx]["dataset_from_index"])
                    end_idx = int(lerobot_dataset.meta.episodes[ep_idx]["dataset_to_index"])
            else:
                start_idx = int(lerobot_dataset.episode_data_index["from"][ep_idx])
                end_idx = int(lerobot_dataset.episode_data_index["to"][ep_idx])

            self.ep_to_indices[str(int(ep_idx))] = list(range(start_idx, end_idx))

        selected_episode_indices = (
            [int(ep_idx) for ep_idx in episode_indices]
            if episode_indices is not None
            else available_episode_ids
        )
        self.episodes = [
            str(ep_idx) for ep_idx in selected_episode_indices if str(ep_idx) in self.ep_to_indices
        ]

    def __iter__(self) -> Iterator[list[int]]:
        episodes = self.episodes.copy()
        if self.shuffle_episodes:
            random.shuffle(episodes)

        if self.is_ddp:
            episodes = episodes[self.rank :: self.num_replicas]

        remaining_skip = int(self.start_step_offset)
        for ep in episodes:
            indices = self.ep_to_indices[ep].copy()
            if remaining_skip > 0:
                if remaining_skip >= len(indices):
                    remaining_skip -= len(indices)
                    continue
                indices = indices[remaining_skip:]
                remaining_skip = 0
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        count = 0
        episodes = self.episodes[self.rank :: self.num_replicas] if self.is_ddp else self.episodes
        for ep in episodes:
            full_batches, tail = divmod(len(self.ep_to_indices[ep]), self.batch_size)
            count += full_batches
            if tail and not self.drop_last:
                count += 1
        return count


class OnlineVLADataset(Dataset):
    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        vlm_processor: Any,
        vision_processor: Any,
        action_horizon: int,
        dataset_slug: str = "unknown",
        vision_image_size: int = 256,
        vlm_image_size: int = 256,
        use_data_aug: bool = False,
        camera_keys: Sequence[str] | None = None,
        camera_key_aliases: dict[str, Sequence[str]] | None = None,
        state_keys: Sequence[str] | None = None,
        task_key_candidates: Sequence[str] | None = None,
        proprio_dim: int = 6,
        action_is_delta: bool = True,
        action_delta_state_indices: Sequence[int | None] | None = None,
        sample_frame_delay: bool = True,
        prompt_cache_size: int = 0,
        delay_max_chunks: int | None = None,
        delay_chunk_size_threshold: float | None = None,
        delay_sync_weight: float = 1.0,
        delay_async_bucket_weights: Sequence[float] | None = None,
        raw_image_mode: bool = False,
        raw_video_batch_decode_max_span_s: float = _RAW_VIDEO_BATCH_DECODE_MAX_SPAN_S,
        raw_video_batch_decode_parallelism: int = 1,
    ):
        self.dataset = lerobot_dataset
        self.dataset_slug = str(dataset_slug)
        self.vlm_processor = vlm_processor
        self.vision_processor = vision_processor
        self.action_horizon = action_horizon
        self.vision_image_size = int(vision_image_size)
        self.vlm_image_size = int(vlm_image_size)
        self.use_data_aug = use_data_aug
        self.camera_keys = list(
            camera_keys or ("observation.images.front", "observation.images.wrist")
        )
        self.camera_alias_map = _normalize_camera_aliases(self.camera_keys, camera_key_aliases)
        self.state_keys = list(
            state_keys
            or (
                "shoulder_pan.pos",
                "shoulder_lift.pos",
                "elbow_flex.pos",
                "wrist_flex.pos",
                "wrist_roll.pos",
                "gripper.pos",
            )
        )
        self.task_key_candidates = list(task_key_candidates or ("task", "language_instruction"))
        self.proprio_dim = int(proprio_dim)
        self.action_is_delta = bool(action_is_delta)
        self.action_delta_state_indices = action_delta_state_indices
        self.sample_frame_delay = bool(sample_frame_delay)
        self.raw_image_mode = bool(raw_image_mode)
        self.raw_video_batch_decode_max_span_s = max(float(raw_video_batch_decode_max_span_s), 0.0)
        self.raw_video_batch_decode_parallelism = max(int(raw_video_batch_decode_parallelism), 1)
        self.prompt_cache_size = max(int(prompt_cache_size), 0)
        self._prompt_cache: OrderedDict[tuple[str, tuple[str, ...]], str] = OrderedDict()
        self.filtered_episode_bounds = _filtered_episode_row_bounds(lerobot_dataset)
        try:
            self.episode_bounds_by_id = episode_row_bounds(lerobot_dataset)
        except (AttributeError, KeyError, TypeError, IndexError, ValueError):
            self.episode_bounds_by_id = {}
        self.episode_bounds_by_start = sorted(
            (int(start), int(end), int(ep_idx))
            for ep_idx, (start, end) in self.episode_bounds_by_id.items()
        )
        self.delay_max_chunks = 1 if delay_max_chunks is None else int(delay_max_chunks)
        self.delay_chunk_size_threshold = (
            0.0 if delay_chunk_size_threshold is None else float(delay_chunk_size_threshold)
        )
        self.delay_support = action_step_delay_support(
            chunk_size=int(self.action_horizon),
            chunk_size_threshold=self.delay_chunk_size_threshold,
            max_delay_chunks=self.delay_max_chunks,
        )
        async_bucket_count = len(self.delay_support) - 1
        if delay_async_bucket_weights is None:
            self.delay_async_bucket_weights = [1.0] * async_bucket_count
        else:
            self.delay_async_bucket_weights = [
                float(weight) for weight in delay_async_bucket_weights
            ]
            if len(self.delay_async_bucket_weights) != async_bucket_count:
                raise ValueError(
                    "delay_async_bucket_weights must match the number of async delay buckets "
                    f"({async_bucket_count}), got {len(self.delay_async_bucket_weights)}"
                )
        self.delay_weights = [float(delay_sync_weight)] + self.delay_async_bucket_weights

        if self.raw_image_mode and self.use_data_aug:
            raise ValueError(
                "raw_image_mode does not support CPU use_data_aug; disable use_data_aug or gpu_preprocess"
            )

        if self.use_data_aug:
            self.vision_transform = T.Compose(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)]
            )
        else:
            self.vision_transform = None

    def __len__(self):
        return len(self.dataset)

    def _locate_episode_bounds_for_index(self, idx: int) -> tuple[int, int, int] | None:
        idx = int(idx)
        for ep_start, ep_end, ep_idx in self.episode_bounds_by_start:
            if ep_start <= idx < ep_end:
                return ep_idx, ep_start, ep_end
        return None

    def _get_episode_bounds(self, idx: int, ep_idx: int) -> tuple[int, int]:
        cached_bounds = self.episode_bounds_by_id.get(int(ep_idx))
        if cached_bounds is not None:
            return int(cached_bounds[0]), int(cached_bounds[1])

        if self.filtered_episode_bounds is not None:
            bounds = self.filtered_episode_bounds.get(int(ep_idx))
            if bounds is not None:
                return bounds

        if hasattr(self.dataset, "meta") and hasattr(self.dataset.meta, "episodes"):
            try:
                ep_start = int(self.dataset.meta.episodes["dataset_from_index"][ep_idx])
                ep_end = int(self.dataset.meta.episodes["dataset_to_index"][ep_idx])
            except (KeyError, TypeError):
                ep_start = int(self.dataset.meta.episodes[ep_idx]["dataset_from_index"])
                ep_end = int(self.dataset.meta.episodes[ep_idx]["dataset_to_index"])
        else:
            ep_start = int(self.dataset.episode_data_index["from"][ep_idx])
            ep_end = int(self.dataset.episode_data_index["to"][ep_idx])
        return ep_start, ep_end

    def _cached_prompt(self, task_text: str) -> str:
        key = (task_text, tuple(self.camera_keys))
        cached = self._prompt_cache.get(key)
        if cached is not None:
            self._prompt_cache.move_to_end(key)
            return cached

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"} for _ in self.camera_keys]
                + [{"type": "text", "text": task_text}],
            }
        ]
        prompt = self.vlm_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if self.prompt_cache_size > 0:
            self._prompt_cache[key] = prompt
            self._prompt_cache.move_to_end(key)
            while len(self._prompt_cache) > self.prompt_cache_size:
                self._prompt_cache.popitem(last=False)
        return prompt

    def _sample_delay_interval(self, feasible_intervals: list[tuple[int, int]]) -> tuple[int, int]:
        if not feasible_intervals:
            return (0, 1)

        weights = []
        for support_idx, support_interval in enumerate(self.delay_support):
            for interval in feasible_intervals:
                if interval[0] == support_interval[0]:
                    weights.append(max(float(self.delay_weights[support_idx]), 0.0))
                    break

        if len(weights) != len(feasible_intervals) or sum(weights) <= 0.0:
            for interval in feasible_intervals:
                if interval == (0, 1):
                    return interval
            return feasible_intervals[0]

        return random.choices(feasible_intervals, weights=weights, k=1)[0]

    def _sample_slow_index(self, idx: int, ep_start: int) -> tuple[int, int]:
        if not self.sample_frame_delay:
            return 0, int(idx)

        history_steps = int(idx - ep_start)
        feasible_intervals: list[tuple[int, int]] = []
        for start, end in self.delay_support:
            clipped_end = min(int(end), history_steps + 1)
            if int(start) < clipped_end:
                feasible_intervals.append((int(start), clipped_end))
        selected_start, selected_end = self._sample_delay_interval(feasible_intervals)
        sampled_frame_delay = random.randint(int(selected_start), int(selected_end) - 1)
        return int(sampled_frame_delay), int(idx - sampled_frame_delay)

    @staticmethod
    def _mark_padding_sample(sample: dict[str, Any]) -> dict[str, Any]:
        marked = dict(sample)
        loss_mask = marked.get("loss_mask")
        if torch.is_tensor(loss_mask):
            marked["loss_mask"] = torch.zeros_like(loss_mask)
        marked["is_padding_sample"] = True
        return marked

    def __getitem__(self, idx: Any) -> dict[str, Any]:
        return self.materialize_index(idx)

    def __getitems__(self, indices: Sequence[Any]) -> list[dict[str, Any]]:
        index_refs = list(indices)
        index_list = [sample_index_value(idx) for idx in index_refs]
        if self.raw_image_mode:
            batched = self._materialize_raw_video_batch(index_list)
            if batched is not None:
                return [
                    self._mark_padding_sample(sample)
                    if is_padded_sample_index(index_ref)
                    else sample
                    for sample, index_ref in zip(batched, index_refs, strict=True)
                ]
        return self.materialize_batch(index_refs)

    def materialize_index(self, idx: Any) -> dict[str, Any]:
        if is_padded_sample_index(idx):
            return self._mark_padding_sample(self.materialize_index(sample_index_value(idx)))
        idx = int(idx)
        located_bounds = self._locate_episode_bounds_for_index(idx)
        if located_bounds is None:
            item_t = self.dataset[idx]
            ep_idx = item_t["episode_index"]
            ep_idx = ep_idx.item() if hasattr(ep_idx, "item") else ep_idx
            ep_start, ep_end = self._get_episode_bounds(idx, ep_idx)
            _sampled_frame_delay, slow_global_idx = self._sample_slow_index(idx, ep_start)
            item_slow = item_t if slow_global_idx == idx else self.dataset[slow_global_idx]
        else:
            ep_idx, ep_start, ep_end = located_bounds
            _sampled_frame_delay, slow_global_idx = self._sample_slow_index(idx, ep_start)
            if slow_global_idx < idx:
                item_slow = self.dataset[slow_global_idx]
                item_t = self.dataset[idx]
            else:
                item_t = self.dataset[idx]
                item_slow = item_t if slow_global_idx == idx else self.dataset[slow_global_idx]
            item_ep_idx = item_t["episode_index"]
            ep_idx = item_ep_idx.item() if hasattr(item_ep_idx, "item") else item_ep_idx
        return self._materialize_from_items(
            idx=idx,
            ep_idx=int(ep_idx),
            ep_start=int(ep_start),
            ep_end=int(ep_end),
            item_t=item_t,
            item_slow=item_slow,
            slow_global_idx=int(slow_global_idx),
        )

    def _materialize_from_items(
        self,
        *,
        idx: int,
        ep_idx: int,
        ep_start: int,
        ep_end: int,
        item_t: dict[str, Any],
        item_slow: dict[str, Any],
        slow_global_idx: int,
    ) -> dict[str, Any]:
        frame_delay = int(idx) - int(slow_global_idx)

        vision_images: list[Image.Image] = []
        vlm_images: list[Image.Image] = []
        vision_images_uint8: list[torch.Tensor] = []
        vlm_images_uint8: list[torch.Tensor] = []
        shared_image_size = self.vision_image_size == self.vlm_image_size
        for camera_key in self.camera_keys:
            current_image = _extract_camera_tensor(item_t, self.camera_alias_map[camera_key])
            slow_image = (
                current_image
                if slow_global_idx == idx
                else _extract_camera_tensor(
                    item_slow,
                    self.camera_alias_map[camera_key],
                )
            )

            if self.raw_image_mode:
                current_uint8 = _tensor_to_uint8_chw(current_image)
                vision_images_uint8.append(current_uint8)
                if slow_global_idx == idx:
                    vlm_images_uint8.append(current_uint8)
                else:
                    vlm_images_uint8.append(_tensor_to_uint8_chw(slow_image))
                continue

            if self.use_data_aug and self.vision_transform is not None:
                current_image = self.vision_transform(current_image)
            vision_pil = _tensor_to_pil(current_image, self.vision_image_size)
            vision_images.append(vision_pil)

            if slow_global_idx == idx and shared_image_size:
                vlm_images.append(vision_pil)
            else:
                vlm_images.append(_tensor_to_pil(slow_image, self.vlm_image_size))

        vision_pixel_values = None
        if not self.raw_image_mode:
            vision_pixel_values = self.vision_processor(
                images=vision_images,
                size={"height": self.vision_image_size, "width": self.vision_image_size},
                return_tensors="pt",
            )["pixel_values"]

        q_current = _extract_q_current(item_t, self.state_keys, self.proprio_dim)

        task_text = _extract_task_text(item_slow, self.task_key_candidates)
        text_prompt = self._cached_prompt(task_text)

        actions_chunk_abs = _extract_action_chunk(item_t)
        if self.action_is_delta:
            state_indices = normalize_action_delta_state_indices(
                self.action_delta_state_indices,
                action_dim=actions_chunk_abs.shape[-1],
            )
            delta_basis = torch.zeros(actions_chunk_abs.shape[-1], dtype=q_current.dtype)
            for action_idx, state_idx in enumerate(state_indices):
                if state_idx is None:
                    continue
                if int(state_idx) >= q_current.shape[0]:
                    raise ValueError(
                        f"Action delta state index {state_idx} for action dim {action_idx} exceeds "
                        f"q_current dim {q_current.shape[0]}"
                    )
                delta_basis[action_idx] = q_current[int(state_idx)]
            actions_target = actions_chunk_abs - delta_basis.unsqueeze(0)
        else:
            actions_target = actions_chunk_abs

        valid_len = min(self.action_horizon, ep_end - idx)
        if valid_len < self.action_horizon:
            loss_mask = torch.cat(
                [
                    torch.ones(valid_len, dtype=torch.float32),
                    torch.zeros(self.action_horizon - valid_len, dtype=torch.float32),
                ],
                dim=0,
            )
        else:
            loss_mask = torch.ones(self.action_horizon, dtype=torch.float32)

        episode_step_index = int(idx - ep_start)
        episode_length = int(ep_end - ep_start)
        task_index = _extract_scalar_int(item_t["task_index"]) if "task_index" in item_t else 0

        result = {
            "action": actions_target.clone(),
            "action_abs": actions_chunk_abs.clone(),
            "loss_mask": loss_mask,
            "vlm_text": text_prompt,
            "frame_delay": torch.tensor(frame_delay, dtype=torch.float32),
            "q_current": q_current,
            "dataset_slug": self.dataset_slug,
            "task_index": torch.tensor(task_index, dtype=torch.int64),
            "episode_index": torch.tensor(int(ep_idx), dtype=torch.int64),
            "episode_step_index": torch.tensor(episode_step_index, dtype=torch.int64),
            "episode_length": torch.tensor(episode_length, dtype=torch.int64),
        }
        if self.raw_image_mode:
            result["vision_images_uint8"] = vision_images_uint8
            result["vlm_images_uint8"] = vlm_images_uint8
        else:
            result["pixel_values"] = vision_pixel_values
            result["vlm_images"] = vlm_images
        return result

    def _reader_for_raw_video_batch(self):
        ensure_reader = getattr(self.dataset, "_ensure_reader", None)
        if not callable(ensure_reader):
            return None
        reader = ensure_reader()
        if getattr(reader, "hf_dataset", None) is None:
            load_and_activate = getattr(reader, "load_and_activate", None)
            if not callable(load_and_activate):
                return None
            load_and_activate()
        return reader

    def _raw_video_key_by_camera(self, video_keys: set[str]) -> dict[str, str] | None:
        key_by_camera: dict[str, str] = {}
        for camera_key, aliases in self.camera_alias_map.items():
            matched = next((alias for alias in aliases if alias in video_keys), None)
            if matched is None:
                return None
            key_by_camera[camera_key] = matched
        return key_by_camera

    def _task_text_for_reader_item(self, item: dict[str, Any], reader: Any) -> str:
        task_text = _extract_task_text(item, self.task_key_candidates)
        if task_text:
            return task_text
        if "task_index" not in item:
            return ""
        task_idx = _extract_scalar_int(item["task_index"])
        tasks = getattr(getattr(reader, "_meta", None), "tasks", None)
        if tasks is None:
            return str(task_idx)
        try:
            return str(tasks.iloc[task_idx].name)
        except Exception:
            return str(task_idx)

    def _reader_base_item(
        self,
        reader: Any,
        rel_idx: int,
    ) -> tuple[dict[str, Any], dict[str, list[int]] | None]:
        item = dict(reader.hf_dataset[int(rel_idx)])
        ep_idx = _extract_scalar_int(item["episode_index"])
        abs_idx = _extract_scalar_int(item["index"]) if "index" in item else int(rel_idx)
        query_indices = None
        if getattr(reader, "delta_indices", None) is not None:
            query_indices, padding = reader._get_query_indices(abs_idx, ep_idx)
            query_result = reader._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, value in query_result.items():
                item[key] = value
        if "task" not in item:
            item["task"] = self._task_text_for_reader_item(item, reader)
        return item, query_indices

    def _split_video_decode_entries(self, entries: list[tuple[float, int, str, str]]):
        sorted_entries = sorted(entries, key=lambda entry: entry[0])
        groups: list[list[tuple[float, int, str, str]]] = []
        current_group: list[tuple[float, int, str, str]] = []
        current_start: float | None = None
        for entry in sorted_entries:
            timestamp = float(entry[0])
            if (
                current_group
                and current_start is not None
                and timestamp - current_start > self.raw_video_batch_decode_max_span_s
            ):
                groups.append(current_group)
                current_group = []
                current_start = None
            if not current_group:
                current_start = timestamp
            current_group.append(entry)
        if current_group:
            groups.append(current_group)
        return groups

    def _materialize_raw_video_batch(self, indices: Sequence[int]) -> list[dict[str, Any]] | None:
        if not indices:
            return []

        reader = self._reader_for_raw_video_batch()
        if reader is None:
            return None

        meta = getattr(reader, "_meta", None)
        video_keys = set(getattr(meta, "video_keys", []) or [])
        key_by_camera = self._raw_video_key_by_camera(video_keys)
        if meta is None or key_by_camera is None:
            return None

        root = getattr(reader, "root", getattr(self.dataset, "root", None))
        if root is None:
            return None
        tolerance_s = float(
            getattr(reader, "_tolerance_s", getattr(self.dataset, "tolerance_s", 1e-4))
        )
        video_backend = getattr(
            reader, "_video_backend", getattr(self.dataset, "_video_backend", None)
        )

        sample_specs: list[dict[str, int]] = []
        required_indices: set[int] = set()
        for idx in indices:
            located_bounds = self._locate_episode_bounds_for_index(int(idx))
            if located_bounds is None:
                return None
            ep_idx, ep_start, ep_end = located_bounds
            _sampled_delay, slow_idx = self._sample_slow_index(int(idx), int(ep_start))
            sample_specs.append(
                {
                    "idx": int(idx),
                    "ep_idx": int(ep_idx),
                    "ep_start": int(ep_start),
                    "ep_end": int(ep_end),
                    "slow_idx": int(slow_idx),
                }
            )
            required_indices.add(int(idx))
            required_indices.add(int(slow_idx))

        item_cache: dict[int, dict[str, Any]] = {}
        query_cache: dict[int, dict[str, list[int]] | None] = {}
        for rel_idx in sorted(required_indices):
            item, query_indices = self._reader_base_item(reader, int(rel_idx))
            item_cache[int(rel_idx)] = item
            query_cache[int(rel_idx)] = query_indices

        decode_groups: dict[str, dict[str, Any]] = {}
        frame_slots: dict[tuple[int, str], torch.Tensor] = {}
        for rel_idx, item in item_cache.items():
            ep_idx = _extract_scalar_int(item["episode_index"])
            current_ts = _to_tensor(item["timestamp"], dtype=torch.float32).reshape(-1)[0].item()
            query_timestamps = reader._get_query_timestamps(float(current_ts), query_cache[rel_idx])
            episode_meta = meta.episodes[ep_idx]
            for camera_key, video_key in key_by_camera.items():
                timestamps = query_timestamps.get(video_key, [float(current_ts)])
                if len(timestamps) != 1:
                    return None
                from_timestamp = float(episode_meta[f"videos/{video_key}/from_timestamp"])
                shifted_ts = from_timestamp + float(timestamps[0])
                video_path = str(root / meta.get_video_file_path(ep_idx, video_key))
                group = decode_groups.setdefault(
                    video_path, {"video_path": video_path, "entries": []}
                )
                group["entries"].append((shifted_ts, rel_idx, video_key, camera_key))

        decode_jobs: list[dict[str, Any]] = []
        raw_decode_stats = {
            "seconds": 0.0,
            "calls": 0,
            "frames": 0,
            "videos": len(decode_groups),
            "groups": 0,
            "max_span_s": 0.0,
            "job_seconds_sum": 0.0,
            "group_seconds_sum": 0.0,
            "max_job_seconds": 0.0,
            "max_group_seconds": 0.0,
        }
        for group in decode_groups.values():
            split_groups = self._split_video_decode_entries(group["entries"])
            decode_jobs.append(
                {
                    "video_path": group["video_path"],
                    "split_groups": split_groups,
                }
            )
            for entries in split_groups:
                timestamps = [float(entry[0]) for entry in entries]
                raw_decode_stats["calls"] += 1
                raw_decode_stats["frames"] += len(timestamps)
                raw_decode_stats["groups"] += 1
                if timestamps:
                    raw_decode_stats["max_span_s"] = max(
                        raw_decode_stats["max_span_s"],
                        max(timestamps) - min(timestamps),
                    )

        def _decode_video_job(
            job: dict[str, Any],
        ) -> tuple[list[tuple[int, str, torch.Tensor]], dict[str, float]]:
            job_t0 = time.perf_counter()
            decoded_slots: list[tuple[int, str, torch.Tensor]] = []
            job_stats = {
                "job_seconds": 0.0,
                "group_seconds_sum": 0.0,
                "max_group_seconds": 0.0,
            }
            for entries in job["split_groups"]:
                timestamps = [float(entry[0]) for entry in entries]
                group_t0 = time.perf_counter()
                frames = lerobot_dataset_reader.decode_video_frames(
                    job["video_path"],
                    timestamps,
                    tolerance_s,
                    video_backend,
                )
                if frames.ndim == 3:
                    is_batched_nv12 = (
                        int(frames.shape[0]) == len(entries)
                        and int(frames.shape[1]) * 2 % 3 == 0
                        and (int(frames.shape[1]) * 2 // 3) % 2 == 0
                        and int(frames.shape[2]) % 2 == 0
                    )
                    if not is_batched_nv12:
                        frames = frames.unsqueeze(0)
                if int(frames.shape[0]) != len(entries):
                    raise ValueError(
                        f"Decoded frame count mismatch for {job['video_path']}: "
                        f"got {int(frames.shape[0])}, expected {len(entries)}"
                    )
                group_seconds = time.perf_counter() - group_t0
                job_stats["group_seconds_sum"] += float(group_seconds)
                job_stats["max_group_seconds"] = max(
                    float(job_stats["max_group_seconds"]),
                    float(group_seconds),
                )
                for frame, (_timestamp, rel_idx, video_key, _camera_key) in zip(
                    frames, entries, strict=True
                ):
                    decoded_slots.append((int(rel_idx), str(video_key), frame))
            job_stats["job_seconds"] = time.perf_counter() - job_t0
            return decoded_slots, job_stats

        def _record_decode_job_stats(job_stats: dict[str, float]) -> None:
            job_seconds = float(job_stats.get("job_seconds", 0.0))
            group_seconds_sum = float(job_stats.get("group_seconds_sum", 0.0))
            max_group_seconds = float(job_stats.get("max_group_seconds", 0.0))
            raw_decode_stats["job_seconds_sum"] += job_seconds
            raw_decode_stats["group_seconds_sum"] += group_seconds_sum
            raw_decode_stats["max_job_seconds"] = max(
                float(raw_decode_stats["max_job_seconds"]),
                job_seconds,
            )
            raw_decode_stats["max_group_seconds"] = max(
                float(raw_decode_stats["max_group_seconds"]),
                max_group_seconds,
            )

        decode_stats_before = servovla_video_decode.get_decode_stats_snapshot()
        decode_t0 = time.perf_counter()
        if len(decode_jobs) > 1 and self.raw_video_batch_decode_parallelism > 1:
            max_workers = min(self.raw_video_batch_decode_parallelism, len(decode_jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_decode_video_job, job) for job in decode_jobs]
                for future in futures:
                    decoded_slots, job_stats = future.result()
                    _record_decode_job_stats(job_stats)
                    for rel_idx, video_key, frame in decoded_slots:
                        frame_slots[(int(rel_idx), str(video_key))] = frame
        else:
            for job in decode_jobs:
                decoded_slots, job_stats = _decode_video_job(job)
                _record_decode_job_stats(job_stats)
                for rel_idx, video_key, frame in decoded_slots:
                    frame_slots[(int(rel_idx), str(video_key))] = frame
        raw_decode_stats["seconds"] = time.perf_counter() - decode_t0
        decode_stats_delta = servovla_video_decode.diff_decode_stats(
            decode_stats_before,
            servovla_video_decode.get_decode_stats_snapshot(),
        )
        if any(
            int(decode_stats_delta.get(key, 0)) != 0
            for key in (
                "cache_hits",
                "cache_misses",
                "decoder_creates",
                "decoder_evictions",
                "fallbacks",
            )
        ):
            raw_decode_stats.update(decode_stats_delta)

        for rel_idx, item in item_cache.items():
            for _camera_key, video_key in key_by_camera.items():
                item[video_key] = frame_slots[(int(rel_idx), str(video_key))]

        samples = [
            self._materialize_from_items(
                idx=spec["idx"],
                ep_idx=spec["ep_idx"],
                ep_start=spec["ep_start"],
                ep_end=spec["ep_end"],
                item_t=item_cache[spec["idx"]],
                item_slow=item_cache[spec["slow_idx"]],
                slow_global_idx=spec["slow_idx"],
            )
            for spec in sample_specs
        ]
        if samples:
            samples[0]["raw_decode_stats"] = raw_decode_stats
        return samples

    def materialize_batch(self, indices: Sequence[Any]) -> list[dict[str, Any]]:
        return [self.materialize_index(idx) for idx in indices]


class OnlineVLACollator:
    """
    Assemble multi-camera image and text batches through multimodal processors.
    """

    def __init__(
        self,
        vlm_processor,
        vlm_processor_micro_batch_size: int | None = None,
        vlm_sequence_padding_multiple: int | None = None,
        raw_image_mode: bool = False,
        vlm_image_size: int | None = None,
    ):
        self.vlm_processor = vlm_processor
        self.raw_image_mode = bool(raw_image_mode)
        self.vlm_image_size = int(vlm_image_size) if vlm_image_size is not None else 0
        if self.raw_image_mode and self.vlm_image_size <= 0:
            raise ValueError("vlm_image_size must be provided when raw_image_mode=True")
        self.vlm_processor_micro_batch_size = 0
        if vlm_processor_micro_batch_size is not None:
            self.vlm_processor_micro_batch_size = int(vlm_processor_micro_batch_size)
            if self.vlm_processor_micro_batch_size < 0:
                raise ValueError("vlm_processor_micro_batch_size must be >= 0 when configured")
        self.vlm_sequence_padding_multiple = (
            ATTENTION_SEQUENCE_MULTIPLE
            if vlm_sequence_padding_multiple is None
            else int(vlm_sequence_padding_multiple)
        )
        if self.vlm_sequence_padding_multiple <= 0:
            raise ValueError("vlm_sequence_padding_multiple must be > 0 when configured")

    def _sequence_pad_value(self, key: str) -> int:
        if key == "input_ids":
            pad_token_id = self.vlm_processor.tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self.vlm_processor.tokenizer.eos_token_id
            return int(pad_token_id)
        return 0

    def _sequence_padding_side(self) -> str:
        padding_side = str(getattr(self.vlm_processor.tokenizer, "padding_side", "right")).lower()
        return "left" if padding_side == "left" else "right"

    def _pad_sequence_tensor(self, key: str, value: torch.Tensor, target_len: int) -> torch.Tensor:
        pad_len = int(target_len) - int(value.shape[1])
        if pad_len <= 0:
            return value
        pad_value = self._sequence_pad_value(key)
        if self._sequence_padding_side() == "left":
            return F.pad(value, (pad_len, 0), value=pad_value)
        return F.pad(value, (0, pad_len), value=pad_value)

    def _concat_vlm_processor_tensors(
        self,
        key: str,
        values: list[torch.Tensor],
        chunk_batch_sizes: list[int],
    ) -> torch.Tensor:
        if (
            all(value.ndim == 2 for value in values)
            and all(
                value.shape[0] == chunk_size
                for value, chunk_size in zip(values, chunk_batch_sizes, strict=True)
            )
            and len({int(value.shape[1]) for value in values}) > 1
        ):
            target_len = max(int(value.shape[1]) for value in values)
            values = [self._pad_sequence_tensor(key, value, target_len) for value in values]
        return torch.cat(values, dim=0)

    def _merge_vlm_processor_chunks(
        self,
        chunks: list[dict[str, Any]],
        chunk_batch_sizes: list[int],
    ) -> dict[str, Any]:
        if len(chunks) == 1:
            return chunks[0]

        merged: dict[str, Any] = {}
        for key in chunks[0].keys():
            values = [chunk[key] for chunk in chunks]
            if all(torch.is_tensor(value) for value in values):
                merged[key] = self._concat_vlm_processor_tensors(
                    key,
                    values,
                    chunk_batch_sizes,
                )
            elif all(isinstance(value, (list, tuple)) for value in values):
                merged_list = []
                for value in values:
                    merged_list.extend(list(value))
                merged[key] = merged_list
            else:
                merged[key] = values
        return merged

    def _build_vlm_inputs(
        self, text_list: list[str], image_list: list[list[Image.Image]]
    ) -> dict[str, Any]:
        micro_batch_size = int(self.vlm_processor_micro_batch_size)
        if micro_batch_size <= 0 or len(text_list) <= micro_batch_size:
            return dict(
                self.vlm_processor(
                    text=text_list,
                    images=image_list,
                    padding=True,
                    return_tensors="pt",
                )
            )

        chunks: list[dict[str, Any]] = []
        chunk_batch_sizes: list[int] = []
        for start in range(0, len(text_list), micro_batch_size):
            end = min(start + micro_batch_size, len(text_list))
            chunks.append(
                dict(
                    self.vlm_processor(
                        text=text_list[start:end],
                        images=image_list[start:end],
                        padding=True,
                        return_tensors="pt",
                    )
                )
            )
            chunk_batch_sizes.append(end - start)
        return self._merge_vlm_processor_chunks(chunks, chunk_batch_sizes)

    def _raw_vlm_returns_mm_token_type_ids(self) -> bool:
        processor_module = str(type(self.vlm_processor).__module__).lower()
        return (
            "qwen3_vl" in processor_module
            and getattr(self.vlm_processor, "image_token_id", None) is not None
        )

    def _build_raw_mm_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        image_token_id = getattr(self.vlm_processor, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == int(image_token_id)] = 1
        video_token_id = getattr(self.vlm_processor, "video_token_id", None)
        if video_token_id is not None:
            mm_token_type_ids[input_ids == int(video_token_id)] = 2
        return mm_token_type_ids

    def _group_uint8_images(self, batch: list[dict[str, Any]], key: str) -> dict[str, Any]:
        groups_by_shape: dict[tuple[str, int, int], list[tuple[int, int, torch.Tensor]]] = (
            defaultdict(list)
        )
        num_cameras = len(batch[0][key])
        for sample_idx, item in enumerate(batch):
            images = item[key]
            if len(images) != num_cameras:
                raise ValueError(f"Inconsistent camera count for {key}")
            for camera_idx, image in enumerate(images):
                if image.dtype != torch.uint8:
                    raise TypeError(f"{key} images must be uint8, got {image.dtype}")
                if image.ndim == 3 and int(image.shape[0]) == 3:
                    image_format = "rgb"
                    height = int(image.shape[-2])
                    width = int(image.shape[-1])
                elif image.ndim == 2:
                    height_times_3 = int(image.shape[0]) * 2
                    if height_times_3 % 3 != 0:
                        raise ValueError(f"{key} NV12 image has invalid shape {tuple(image.shape)}")
                    image_format = "nv12"
                    height = height_times_3 // 3
                    width = int(image.shape[1])
                    if height % 2 != 0 or width % 2 != 0:
                        raise ValueError(
                            f"{key} NV12 image must have even RGB dimensions, got {tuple(image.shape)}"
                        )
                else:
                    raise ValueError(
                        f"{key} images must be CHW RGB or NV12 tensors, got {tuple(image.shape)}"
                    )
                groups_by_shape[(image_format, height, width)].append(
                    (sample_idx, camera_idx, image.contiguous())
                )

        groups = []
        for (image_format, height, width), entries in groups_by_shape.items():
            restore_indices = torch.tensor(
                [[sample_idx, camera_idx] for sample_idx, camera_idx, _image in entries],
                dtype=torch.long,
            )
            images = torch.stack([image for _sample_idx, _camera_idx, image in entries], dim=0)
            groups.append(
                {
                    "format": image_format,
                    "height": int(height),
                    "width": int(width),
                    "images": images,
                    "restore_indices": restore_indices,
                }
            )
        return {"batch_size": len(batch), "num_cameras": int(num_cameras), "groups": groups}

    def _build_raw_vlm_inputs(self, text_list: list[str], *, image_count: int) -> dict[str, Any]:
        image_processor = self.vlm_processor.image_processor
        merge_size = int(getattr(image_processor, "merge_size", 2))
        image_token = str(getattr(self.vlm_processor, "image_token", "<image>"))
        grid = qwen_image_grid_thw_for_square(image_processor, image_size=self.vlm_image_size)
        image_grid_thw = torch.tensor([grid] * int(image_count), dtype=torch.long)
        expanded_text = expand_qwen_image_tokens(
            text_list,
            image_grid_thw=image_grid_thw,
            image_token=image_token,
            merge_size=merge_size,
        )
        tokenized = dict(
            self.vlm_processor.tokenizer(
                expanded_text,
                padding=True,
                return_tensors="pt",
            )
        )
        tokenized["image_grid_thw"] = image_grid_thw
        if self._raw_vlm_returns_mm_token_type_ids():
            tokenized["mm_token_type_ids"] = self._build_raw_mm_token_type_ids(
                tokenized["input_ids"]
            )
        return tokenized

    def _aggregate_raw_decode_stats(self, batch: list[dict[str, Any]]) -> dict[str, float | int]:
        totals: dict[str, float | int] = {}
        max_keys = {"max_span_s", "max_job_seconds", "max_group_seconds"}
        for item in batch:
            stats = item.get("raw_decode_stats")
            if not isinstance(stats, dict):
                continue
            for key, value in stats.items():
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if key in max_keys:
                    totals[key] = max(float(totals.get(key, 0.0)), float(value))
                elif isinstance(value, float):
                    totals[key] = float(totals.get(key, 0.0)) + float(value)
                else:
                    totals[key] = int(totals.get(key, 0)) + int(value)
        return totals

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        actions = torch.stack([item["action"] for item in batch])
        loss_masks = torch.stack([item["loss_mask"] for item in batch])
        frame_delays = torch.stack([item["frame_delay"] for item in batch])
        q_current = torch.stack([item["q_current"] for item in batch])
        dataset_slugs = [str(item.get("dataset_slug", "unknown")) for item in batch]
        task_indices = torch.stack([item["task_index"] for item in batch])
        episode_indices = torch.stack([item["episode_index"] for item in batch])
        episode_step_indices = torch.stack([item["episode_step_index"] for item in batch])
        episode_lengths = torch.stack([item["episode_length"] for item in batch])

        text_list = [item["vlm_text"] for item in batch]
        if self.raw_image_mode:
            vision_image_groups = self._group_uint8_images(batch, "vision_images_uint8")
            vlm_image_groups = self._group_uint8_images(batch, "vlm_images_uint8")
            image_count = int(vlm_image_groups["batch_size"]) * int(vlm_image_groups["num_cameras"])
            batched_vlm_inputs = self._build_raw_vlm_inputs(text_list, image_count=image_count)
            pixel_values = None
        else:
            pixel_values = torch.stack([item["pixel_values"] for item in batch])
            image_list = [item["vlm_images"] for item in batch]
            batched_vlm_inputs = self._build_vlm_inputs(text_list, image_list)

        input_ids = batched_vlm_inputs["input_ids"]
        actual_len = input_ids.shape[1]
        target_len = round_up_to_multiple(actual_len, self.vlm_sequence_padding_multiple)
        pad_len = target_len - actual_len

        if pad_len > 0:
            for key, value in list(batched_vlm_inputs.items()):
                if not torch.is_tensor(value):
                    continue
                if value.ndim != 2:
                    continue
                if value.shape[0] != input_ids.shape[0]:
                    continue
                if value.shape[1] != actual_len:
                    continue

                batched_vlm_inputs[key] = self._pad_sequence_tensor(key, value, target_len)

        c_sem_mask = batched_vlm_inputs["attention_mask"].bool()
        vlm_inputs_dict = {k: v for k, v in batched_vlm_inputs.items()}

        result = {
            "action": actions,
            "loss_mask": loss_masks,
            "vlm_inputs": vlm_inputs_dict,
            "c_sem_mask": c_sem_mask,
            "frame_delay": frame_delays,
            "q_current": q_current,
            "dataset_slug": dataset_slugs,
            "task_index": task_indices,
            "episode_index": episode_indices,
            "episode_step_index": episode_step_indices,
            "episode_length": episode_lengths,
        }
        if self.raw_image_mode:
            result["vision_images_uint8"] = vision_image_groups
            result["vlm_images_uint8"] = vlm_image_groups
            raw_decode_stats = self._aggregate_raw_decode_stats(batch)
            if raw_decode_stats:
                result["raw_decode_stats"] = raw_decode_stats
        else:
            result["pixel_values"] = pixel_values
        return result
