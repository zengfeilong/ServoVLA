from __future__ import annotations

import threading
import time
from pathlib import Path

import torch

from servovla.data.dataset_loader import (
    BatchAwareConcatDataset,
    OnlineVLACollator,
    OnlineVLADataset,
    episode_row_bounds,
    selected_episode_ids,
)
from servovla.data.indexing import PaddedSampleIndex


class _RecordingVisionProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, return_tensors: str, size):
        image_list = images if isinstance(images, list) else [images]
        self.calls.append(
            {
                "size": size,
                "pil_sizes": [image.size for image in image_list],
                "batch_size": len(image_list),
            }
        )
        return {"pixel_values": torch.zeros(len(image_list), 3, size["height"], size["width"])}


class _FakeVLMProcessor:
    def __init__(self):
        self.templates = 0

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        self.templates += 1
        return messages[0]["content"][-1]["text"]


class _FakeEpisodeDataset:
    def __len__(self) -> int:
        return 1

    @property
    def episode_data_index(self):
        return {"from": [0], "to": [1]}

    def __getitem__(self, idx: int):
        return {
            "episode_index": 0,
            "observation.images.front": torch.zeros(3, 8, 8),
            "observation.images.wrist": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(6),
            "action": torch.zeros(16, 6),
            "task": "pick",
        }


class _FakeHFDataset:
    def __init__(self, episode_ids):
        self._episode_ids = list(episode_ids)

    def __len__(self):
        return len(self._episode_ids)

    def __getitem__(self, key):
        if key == "episode_index":
            return self._episode_ids
        raise KeyError(key)


class _MetaWithoutEpisodes:
    pass


class _ReaderStyleDataset:
    def __init__(self, episode_ids, *, episodes=None):
        self.episodes = episodes
        self.meta = _MetaWithoutEpisodes()
        self.hf_dataset = _FakeHFDataset(episode_ids)

    def __len__(self):
        return len(self.hf_dataset)


class _BatchVideoHFDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        if isinstance(item, list):
            return {key: [self.rows[idx][key] for idx in item] for key in self.rows[0]}
        return dict(self.rows[int(item)])


class _BatchVideoMeta:
    fps = 15
    total_frames = 8
    total_episodes = 1
    video_keys = ["observation.images.front", "observation.images.wrist"]
    camera_keys = video_keys
    features = {}

    def __init__(self):
        self.episodes = [
            {
                "dataset_from_index": 0,
                "dataset_to_index": 8,
                "videos/observation.images.front/from_timestamp": 0.0,
                "videos/observation.images.wrist/from_timestamp": 0.0,
            }
        ]

    def get_video_file_path(self, ep_idx: int, vid_key: str) -> Path:
        camera_name = vid_key.split(".")[-1]
        return Path("videos") / camera_name / f"episode-{ep_idx}.mp4"


class _BatchVideoReader:
    def __init__(self, rows, root):
        self.hf_dataset = _BatchVideoHFDataset(rows)
        self._meta = _BatchVideoMeta()
        self.root = Path(root)
        self._tolerance_s = 1e-4
        self._video_backend = "servovla_pyav_cuda"
        self._absolute_to_relative_idx = None
        self.delta_indices = {"action": list(range(4))}

    def _get_query_indices(self, abs_idx: int, ep_idx: int):
        del ep_idx
        query_indices = {
            "action": [min(len(self.hf_dataset) - 1, int(abs_idx) + offset) for offset in range(4)]
        }
        padding = {"action_is_pad": torch.zeros(4, dtype=torch.bool)}
        return query_indices, padding

    def _query_hf_dataset(self, query_indices):
        action_indices = query_indices["action"]
        actions = [torch.full((6,), float(idx), dtype=torch.float32) for idx in action_indices]
        return {"action": torch.stack(actions, dim=0)}

    def _get_query_timestamps(self, current_ts: float, query_indices=None):
        del query_indices
        return {key: [float(current_ts)] for key in self._meta.video_keys}


class _BatchVideoDataset:
    def __init__(self, rows, root):
        self.reader = _BatchVideoReader(rows, root)
        self.meta = self.reader._meta
        self.root = Path(root)
        self.episodes = None
        self.tolerance_s = self.reader._tolerance_s
        self._video_backend = self.reader._video_backend
        self.delta_timestamps = {"action": [0.0, 1 / 15, 2 / 15, 3 / 15]}
        self.image_transforms = None
        self.episode_data_index = {"from": [0], "to": [len(rows)]}

    def __len__(self):
        return len(self.reader.hf_dataset)

    def _ensure_reader(self):
        return self.reader

    @property
    def hf_dataset(self):
        return self.reader.hf_dataset

    def __getitem__(self, idx):
        raise AssertionError(f"batched raw path should not call per-sample __getitem__, got {idx}")


def test_batch_aware_concat_dataset_dispatches_batched_fetches_in_request_order():
    class _BulkDataset:
        def __init__(self, name, length):
            self.name = name
            self.length = int(length)
            self.bulk_calls = []
            self.item_calls = []

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            self.item_calls.append(int(idx))
            return (self.name, "item", int(idx))

        def __getitems__(self, indices):
            local_indices = [int(idx) for idx in indices]
            self.bulk_calls.append(local_indices)
            return [(self.name, "bulk", idx) for idx in local_indices]

    class _ItemDataset:
        def __init__(self, name, length):
            self.name = name
            self.length = int(length)
            self.item_calls = []

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            self.item_calls.append(int(idx))
            return (self.name, "item", int(idx))

    left = _BulkDataset("left", 2)
    right = _ItemDataset("right", 2)
    dataset = BatchAwareConcatDataset([left, right])

    result = dataset.__getitems__([0, 3, 1, 2])

    assert result == [
        ("left", "bulk", 0),
        ("right", "item", 1),
        ("left", "bulk", 1),
        ("right", "item", 0),
    ]
    assert left.bulk_calls == [[0, 1]]
    assert left.item_calls == []
    assert right.item_calls == [1, 0]


def test_batch_aware_concat_dataset_preserves_padded_indices_for_child_batches():
    class _BulkDataset:
        def __init__(self, length):
            self.length = int(length)
            self.bulk_calls = []

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            return ("item", idx)

        def __getitems__(self, indices):
            self.bulk_calls.append(list(indices))
            return list(indices)

    left = _BulkDataset(2)
    right = _BulkDataset(2)
    dataset = BatchAwareConcatDataset([left, right])

    result = dataset.__getitems__([0, PaddedSampleIndex(3)])

    assert result[0] == 0
    assert isinstance(result[1], PaddedSampleIndex)
    assert result[1].index == 1
    assert left.bulk_calls == [[0]]
    assert right.bulk_calls == [[PaddedSampleIndex(1)]]


def test_episode_bounds_derive_from_hf_dataset_when_meta_episodes_missing():
    dataset = _ReaderStyleDataset([10, 10, 11, 11, 11, 13])

    assert selected_episode_ids(dataset) == [10, 11, 13]
    assert episode_row_bounds(dataset) == {
        10: (0, 2),
        11: (2, 5),
        13: (5, 6),
    }


def test_filtered_episode_bounds_use_relative_hf_rows_for_reader_style_dataset():
    dataset = _ReaderStyleDataset([11, 11, 13], episodes=[11, 13])

    assert selected_episode_ids(dataset) == [11, 13]
    assert episode_row_bounds(dataset) == {
        11: (0, 2),
        13: (2, 3),
    }


def test_online_vla_dataset_uses_split_vision_and_vlm_image_sizes():
    vision_processor = _RecordingVisionProcessor()
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=vision_processor,
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
    )

    sample = dataset[0]

    assert vision_processor.calls[0]["size"] == {"height": 12, "width": 12}
    assert sample["pixel_values"].shape == (2, 3, 12, 12)
    assert [img.size for img in sample["vlm_images"]] == [(20, 20), (20, 20)]


def test_online_vla_dataset_materialize_batch_preserves_contract():
    vision_processor = _RecordingVisionProcessor()
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=vision_processor,
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=12,
        action_is_delta=True,
    )

    batch = dataset.materialize_batch([0])

    assert len(batch) == 1
    assert batch[0]["action"].shape == (16, 6)
    assert batch[0]["pixel_values"].shape == (2, 3, 12, 12)
    assert batch[0]["dataset_slug"] == "unknown"


def test_online_vla_dataset_zeroes_loss_mask_for_padded_sample_index():
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=12,
        action_is_delta=True,
    )

    real, padded = dataset.materialize_batch([0, PaddedSampleIndex(0)])

    assert real["loss_mask"].sum().item() > 0
    assert torch.all(padded["loss_mask"] == 0)
    assert padded["is_padding_sample"] is True


def test_online_vla_dataset_batches_multi_camera_vision_processor_call():
    vision_processor = _RecordingVisionProcessor()
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=vision_processor,
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=12,
        action_is_delta=True,
    )

    sample = dataset[0]

    assert len(vision_processor.calls) == 1
    assert vision_processor.calls[0]["batch_size"] == 2
    assert vision_processor.calls[0]["pil_sizes"] == [(12, 12), (12, 12)]
    assert sample["pixel_values"].shape == (2, 3, 12, 12)


def test_online_vla_dataset_emits_dataset_slug():
    sample_ds = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=12,
        action_is_delta=True,
        dataset_slug="unit_dataset",
    )

    sample = sample_ds[0]

    assert sample["dataset_slug"] == "unit_dataset"


def test_online_vla_dataset_raw_mode_emits_uint8_images_without_processors():
    vision_processor = _RecordingVisionProcessor()
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=vision_processor,
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        raw_image_mode=True,
    )

    sample = dataset[0]

    assert vision_processor.calls == []
    assert "pixel_values" not in sample
    assert "vlm_images" not in sample
    assert len(sample["vision_images_uint8"]) == 2
    assert len(sample["vlm_images_uint8"]) == 2
    assert sample["vision_images_uint8"][0].dtype == torch.uint8
    assert sample["vision_images_uint8"][0].shape == (3, 8, 8)
    assert sample["vlm_images_uint8"][1].dtype == torch.uint8
    assert sample["vlm_text"] == "pick"
    assert sample["action"].shape == (16, 6)
    assert sample["dataset_slug"] == "unknown"


def test_online_vla_dataset_raw_mode_reuses_uint8_conversion_for_sync_semantic_frame():
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        raw_image_mode=True,
    )

    sample = dataset[0]

    assert int(sample["frame_delay"].item()) == 0
    assert sample["vision_images_uint8"][0] is sample["vlm_images_uint8"][0]
    assert sample["vision_images_uint8"][1] is sample["vlm_images_uint8"][1]


def test_online_vla_dataset_raw_getitems_batches_video_decodes(monkeypatch, tmp_path):
    rows = [
        {
            "episode_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "timestamp": torch.tensor(float(idx) / 15.0),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.full((6,), float(idx)),
            "task": f"task-{idx}",
            "task_index": torch.tensor(0),
        }
        for idx in range(8)
    ]
    decode_calls = []

    def _decode_video_frames(video_path, timestamps, tolerance_s, backend):
        del tolerance_s
        decode_calls.append((str(video_path), list(timestamps), backend))
        frames = []
        for timestamp in timestamps:
            frame_idx = int(round(float(timestamp) * 15.0))
            frames.append(torch.full((3, 4, 4), frame_idx / 255.0, dtype=torch.float32))
        return torch.stack(frames, dim=0)

    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.decode_video_frames",
        _decode_video_frames,
    )
    dataset = OnlineVLADataset(
        lerobot_dataset=_BatchVideoDataset(rows, tmp_path),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        delay_max_chunks=1,
        camera_keys=["observation.images.front", "observation.images.wrist"],
    )

    samples = dataset.__getitems__([0, 1, 2])

    assert len(samples) == 3
    assert len(decode_calls) == 2
    assert sorted(len(call[1]) for call in decode_calls) == [3, 3]
    assert samples[0]["raw_decode_stats"]["calls"] == 2
    assert samples[0]["raw_decode_stats"]["frames"] == 6
    assert samples[0]["raw_decode_stats"]["videos"] == 2
    assert samples[0]["raw_decode_stats"]["groups"] == 2
    assert samples[0]["raw_decode_stats"]["max_span_s"] > 0.0
    assert "raw_decode_stats" not in samples[1]
    assert [sample["vlm_text"] for sample in samples] == ["task-0", "task-1", "task-2"]
    assert torch.equal(
        samples[2]["vision_images_uint8"][0], torch.full((3, 4, 4), 2, dtype=torch.uint8)
    )
    assert samples[1]["vision_images_uint8"][1] is samples[1]["vlm_images_uint8"][1]
    assert samples[2]["action"].shape == (4, 6)


def test_online_vla_dataset_raw_getitems_parallelizes_different_video_decodes(
    monkeypatch, tmp_path
):
    rows = [
        {
            "episode_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "timestamp": torch.tensor(float(idx) / 15.0),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.full((6,), float(idx)),
            "task": f"task-{idx}",
            "task_index": torch.tensor(0),
        }
        for idx in range(8)
    ]
    active = 0
    max_active = 0
    lock = threading.Lock()

    def _decode_video_frames(video_path, timestamps, tolerance_s, backend):
        nonlocal active, max_active
        del video_path, tolerance_s, backend
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            time.sleep(0.05)
            frames = []
            for timestamp in timestamps:
                frame_idx = int(round(float(timestamp) * 15.0))
                frames.append(torch.full((3, 4, 4), frame_idx / 255.0, dtype=torch.float32))
            return torch.stack(frames, dim=0)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.decode_video_frames",
        _decode_video_frames,
    )
    dataset = OnlineVLADataset(
        lerobot_dataset=_BatchVideoDataset(rows, tmp_path),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        delay_max_chunks=1,
        camera_keys=["observation.images.front", "observation.images.wrist"],
        raw_video_batch_decode_parallelism=2,
    )

    samples = dataset.__getitems__([0, 1, 2])

    assert max_active == 2
    assert len(samples) == 3
    assert samples[0]["raw_decode_stats"]["calls"] == 2
    assert samples[0]["raw_decode_stats"]["frames"] == 6
    assert samples[0]["raw_decode_stats"]["videos"] == 2
    assert samples[0]["raw_decode_stats"]["groups"] == 2
    assert torch.equal(
        samples[2]["vision_images_uint8"][0], torch.full((3, 4, 4), 2, dtype=torch.uint8)
    )


def test_online_vla_dataset_raw_getitems_accepts_batched_nv12_decode(monkeypatch, tmp_path):
    rows = [
        {
            "episode_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "timestamp": torch.tensor(float(idx) / 15.0),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.full((6,), float(idx)),
            "task": f"task-{idx}",
            "task_index": torch.tensor(0),
        }
        for idx in range(8)
    ]

    def _decode_video_frames(video_path, timestamps, tolerance_s, backend):
        del video_path, tolerance_s, backend
        frames = []
        for timestamp in timestamps:
            frame_idx = int(round(float(timestamp) * 15.0))
            frames.append(torch.full((6, 4), frame_idx, dtype=torch.uint8))
        return torch.stack(frames, dim=0)

    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.decode_video_frames",
        _decode_video_frames,
    )
    dataset = OnlineVLADataset(
        lerobot_dataset=_BatchVideoDataset(rows, tmp_path),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        delay_max_chunks=1,
        camera_keys=["observation.images.front", "observation.images.wrist"],
    )

    samples = dataset.__getitems__([0, 1, 2])

    assert len(samples) == 3
    assert samples[0]["vision_images_uint8"][0].shape == (6, 4)
    assert torch.equal(
        samples[2]["vision_images_uint8"][0], torch.full((6, 4), 2, dtype=torch.uint8)
    )


def test_online_vla_dataset_raw_getitems_splits_wide_video_decode_spans(monkeypatch, tmp_path):
    rows = [
        {
            "episode_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "timestamp": torch.tensor(float(idx) * 5.0),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.full((6,), float(idx)),
            "task": f"task-{idx}",
            "task_index": torch.tensor(0),
        }
        for idx in range(8)
    ]
    decode_calls = []

    def _decode_video_frames(video_path, timestamps, tolerance_s, backend):
        del tolerance_s, backend
        decode_calls.append((str(video_path), list(timestamps)))
        frames = [
            torch.full((3, 4, 4), float(i), dtype=torch.float32) for i, _ts in enumerate(timestamps)
        ]
        return torch.stack(frames, dim=0)

    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.decode_video_frames",
        _decode_video_frames,
    )
    dataset = OnlineVLADataset(
        lerobot_dataset=_BatchVideoDataset(rows, tmp_path),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        delay_max_chunks=1,
        camera_keys=["observation.images.front", "observation.images.wrist"],
    )

    dataset.__getitems__([0, 1, 2])

    assert len(decode_calls) == 6
    assert all(len(timestamps) == 1 for _path, timestamps in decode_calls)


def test_online_vla_dataset_raw_getitems_uses_configured_decode_span(monkeypatch, tmp_path):
    rows = [
        {
            "episode_index": torch.tensor(0),
            "index": torch.tensor(idx),
            "timestamp": torch.tensor(float(idx) * 5.0),
            "observation.state": torch.full((6,), float(idx)),
            "action": torch.full((6,), float(idx)),
            "task": f"task-{idx}",
            "task_index": torch.tensor(0),
        }
        for idx in range(8)
    ]
    decode_calls = []

    def _decode_video_frames(video_path, timestamps, tolerance_s, backend):
        del tolerance_s, backend
        decode_calls.append((str(video_path), list(timestamps)))
        frames = [
            torch.full((3, 4, 4), float(i), dtype=torch.float32) for i, _ts in enumerate(timestamps)
        ]
        return torch.stack(frames, dim=0)

    monkeypatch.setattr(
        "lerobot.datasets.dataset_reader.decode_video_frames",
        _decode_video_frames,
    )
    dataset = OnlineVLADataset(
        lerobot_dataset=_BatchVideoDataset(rows, tmp_path),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        delay_max_chunks=1,
        camera_keys=["observation.images.front", "observation.images.wrist"],
        raw_video_batch_decode_max_span_s=10.0,
    )

    dataset.__getitems__([0, 1, 2])

    assert len(decode_calls) == 2
    assert sorted(len(timestamps) for _path, timestamps in decode_calls) == [3, 3]


def test_prompt_cache_key_includes_ordered_camera_keys():
    class _Processor(_FakeVLMProcessor):
        def __init__(self):
            self.prompts = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            prompt = ",".join(item["type"] for item in messages[0]["content"])
            self.prompts.append(prompt)
            return prompt

    processor = _Processor()
    dataset = OnlineVLADataset(
        lerobot_dataset=_FakeEpisodeDataset(),
        vlm_processor=processor,
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=4,
        raw_image_mode=True,
        prompt_cache_size=8,
        camera_keys=["observation.images.front", "observation.images.wrist"],
    )

    first = dataset._cached_prompt("pick")
    dataset.camera_keys = ["observation.images.wrist", "observation.images.front"]
    second = dataset._cached_prompt("pick")

    assert first == "image,image,text"
    assert second == "image,image,text"
    assert len(processor.prompts) == 2


def test_online_vla_dataset_raw_mode_preserves_delayed_semantic_frame(monkeypatch):
    class _DelayDataset:
        def __init__(self):
            self.num_episodes = 1
            self.episode_data_index = {"from": [0], "to": [64]}

        def __len__(self):
            return 64

        def __getitem__(self, idx):
            value = int(idx)
            image = torch.full((3, 8, 8), value, dtype=torch.uint8)
            return {
                "episode_index": 0,
                "observation.images.front": image,
                "observation.images.wrist": image + 1,
                "observation.state": torch.zeros(6),
                "action": torch.zeros(16, 6),
                "task": f"task-{idx}",
            }

    monkeypatch.setattr(
        "servovla.data.dataset_loader.random.choices",
        lambda population, weights, k: [
            next(interval for interval in population if interval[0] > 0)
        ],
    )
    monkeypatch.setattr("servovla.data.dataset_loader.random.randint", lambda start, _end: start)

    dataset = OnlineVLADataset(
        lerobot_dataset=_DelayDataset(),
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        raw_image_mode=True,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.25,
        delay_sync_weight=0.0,
    )

    sample = dataset[20]

    assert int(sample["frame_delay"].item()) == 12
    assert torch.equal(
        sample["vision_images_uint8"][0], torch.full((3, 8, 8), 20, dtype=torch.uint8)
    )
    assert torch.equal(sample["vlm_images_uint8"][0], torch.full((3, 8, 8), 8, dtype=torch.uint8))
    assert sample["vlm_text"] == "task-8"


def test_qwen_grid_uses_post_smart_resize_dimensions_for_configured_square():
    from servovla.data.qwen_image_tokens import qwen_image_grid_thw_for_square

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    grid = qwen_image_grid_thw_for_square(_ImageProcessor(), image_size=512)

    assert grid == (1, 36, 36)


def test_online_vla_collator_raw_mode_groups_uint8_images_and_skips_cpu_vlm_pixels():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def __call__(self, text, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            max_len = max(len(item) for item in text)
            input_ids = torch.zeros(len(text), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(text), max_len, dtype=torch.long)
            for row, item in enumerate(text):
                input_ids[row, : len(item)] = 7
                attention_mask[row, : len(item)] = 1
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _Processor:
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    def _item(offset: int):
        return {
            "action": torch.zeros(4, 2),
            "loss_mask": torch.ones(4),
            "vision_images_uint8": [
                torch.full((3, 8, 8), offset, dtype=torch.uint8),
                torch.full((3, 10, 8), offset + 1, dtype=torch.uint8),
            ],
            "vlm_images_uint8": [
                torch.full((3, 8, 8), offset + 2, dtype=torch.uint8),
                torch.full((3, 10, 8), offset + 3, dtype=torch.uint8),
            ],
            "vlm_text": "<image><image> pick",
            "frame_delay": torch.tensor(float(offset)),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit_dataset",
        }

    first = _item(0)
    first["raw_decode_stats"] = {
        "seconds": 1.25,
        "calls": 2,
        "frames": 6,
        "videos": 2,
        "groups": 2,
        "max_span_s": 0.5,
        "job_seconds_sum": 1.4,
        "group_seconds_sum": 1.3,
        "max_job_seconds": 0.8,
        "max_group_seconds": 0.7,
    }
    second = _item(10)
    second["raw_decode_stats"] = {
        "seconds": 0.75,
        "calls": 1,
        "frames": 3,
        "videos": 1,
        "groups": 1,
        "max_span_s": 0.25,
        "job_seconds_sum": 0.9,
        "group_seconds_sum": 0.8,
        "max_job_seconds": 0.9,
        "max_group_seconds": 0.8,
    }

    collator = OnlineVLACollator(
        vlm_processor=_Processor(),
        raw_image_mode=True,
        vlm_image_size=512,
    )

    batch = collator([first, second])

    assert "pixel_values" not in batch
    assert "vlm_images" not in batch
    assert "vision_images_uint8" in batch
    assert "vlm_images_uint8" in batch
    assert batch["vision_images_uint8"]["batch_size"] == 2
    assert batch["vision_images_uint8"]["num_cameras"] == 2
    assert len(batch["vision_images_uint8"]["groups"]) == 2
    assert batch["vlm_inputs"]["image_grid_thw"].shape == (4, 3)
    assert torch.equal(
        batch["vlm_inputs"]["image_grid_thw"][0],
        torch.tensor([1, 36, 36], dtype=torch.long),
    )
    assert "pixel_values" not in batch["vlm_inputs"]
    assert batch["c_sem_mask"].dtype == torch.bool
    assert batch["raw_decode_stats"] == {
        "seconds": 2.0,
        "calls": 3,
        "frames": 9,
        "videos": 3,
        "groups": 3,
        "max_span_s": 0.5,
        "job_seconds_sum": 2.3,
        "group_seconds_sum": 2.1,
        "max_job_seconds": 0.9,
        "max_group_seconds": 0.8,
    }


def test_raw_collator_preserves_nv12_image_groups_for_gpu_conversion():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def __call__(self, text, padding, return_tensors):
            del text
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.ones(1, 4, dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
            }

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _Processor:
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    sample = {
        "action": torch.zeros(2, 1),
        "loss_mask": torch.ones(2),
        "vision_images_uint8": [torch.zeros(6, 4, dtype=torch.uint8)],
        "vlm_images_uint8": [torch.zeros(6, 4, dtype=torch.uint8)],
        "vlm_text": "<image> pick",
        "frame_delay": torch.tensor(0.0),
        "q_current": torch.zeros(3),
        "task_index": torch.tensor(0),
        "episode_index": torch.tensor(0),
        "episode_step_index": torch.tensor(0),
        "episode_length": torch.tensor(1),
        "dataset_slug": "unit",
    }
    collator = OnlineVLACollator(
        vlm_processor=_Processor(),
        raw_image_mode=True,
        vlm_image_size=512,
    )

    batch = collator([sample])

    group = batch["vision_images_uint8"]["groups"][0]
    assert group["format"] == "nv12"
    assert group["height"] == 4
    assert group["width"] == 4
    assert group["images"].shape == (1, 6, 4)


def test_raw_collator_preserves_unusual_camera_order_in_restore_indices():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def __call__(self, text, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.ones(len(text), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 4, dtype=torch.long),
            }

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _Processor:
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    def _item(offset: int):
        return {
            "action": torch.zeros(2, 1),
            "loss_mask": torch.ones(2),
            "vision_images_uint8": [
                torch.full((3, 2, 2), offset + 20, dtype=torch.uint8),
                torch.full((3, 2, 2), offset + 30, dtype=torch.uint8),
                torch.full((3, 2, 2), offset + 10, dtype=torch.uint8),
            ],
            "vlm_images_uint8": [
                torch.full((3, 2, 2), offset + 20, dtype=torch.uint8),
                torch.full((3, 2, 2), offset + 30, dtype=torch.uint8),
                torch.full((3, 2, 2), offset + 10, dtype=torch.uint8),
            ],
            "vlm_text": "<image><image><image> pick",
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit",
        }

    collator = OnlineVLACollator(
        vlm_processor=_Processor(),
        raw_image_mode=True,
        vlm_image_size=512,
    )

    batch = collator([_item(0)])
    group = batch["vision_images_uint8"]["groups"][0]
    values_by_camera = {
        int(camera_idx): int(group["images"][row_idx, 0, 0, 0].item())
        for row_idx, (_sample_idx, camera_idx) in enumerate(group["restore_indices"].tolist())
    }

    assert batch["vision_images_uint8"]["num_cameras"] == 3
    assert values_by_camera == {0: 20, 1: 30, 2: 10}


def test_online_vla_collator_raw_mode_preserves_qwen3_mm_token_type_ids():
    image_token_id = 151655

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def __call__(self, text, padding, return_tensors):
            del text
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor(
                    [[10, image_token_id, 20, image_token_id, 30]], dtype=torch.long
                ),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _Processor:
        __module__ = "transformers.models.qwen3_vl.processing_qwen3_vl"
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    _Processor.image_token_id = image_token_id

    item = {
        "action": torch.zeros(4, 2),
        "loss_mask": torch.ones(4),
        "vision_images_uint8": [
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            torch.zeros(3, 8, 8, dtype=torch.uint8),
        ],
        "vlm_images_uint8": [
            torch.zeros(3, 8, 8, dtype=torch.uint8),
            torch.zeros(3, 8, 8, dtype=torch.uint8),
        ],
        "vlm_text": "<image> pick <image>",
        "frame_delay": torch.tensor(0.0),
        "q_current": torch.zeros(3),
        "task_index": torch.tensor(0),
        "episode_index": torch.tensor(0),
        "episode_step_index": torch.tensor(0),
        "episode_length": torch.tensor(1),
        "dataset_slug": "unit_dataset",
    }

    batch = OnlineVLACollator(
        vlm_processor=_Processor(),
        raw_image_mode=True,
        vlm_image_size=512,
    )([item])

    assert torch.equal(
        batch["vlm_inputs"]["mm_token_type_ids"],
        torch.tensor([[0, 1, 0, 1, 0, 0, 0, 0]], dtype=torch.long),
    )


def test_online_vla_collator_emits_pixel_values_key():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.ones(len(text), 3, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 3, dtype=torch.long),
            }

    item = {
        "action": torch.zeros(4, 2),
        "loss_mask": torch.ones(4),
        "pixel_values": torch.zeros(2, 3, 12, 12),
        "vlm_text": "pick",
        "vlm_images": [],
        "frame_delay": torch.tensor(0.0),
        "q_current": torch.zeros(3),
        "task_index": torch.tensor(0),
        "episode_index": torch.tensor(0),
        "episode_step_index": torch.tensor(0),
        "episode_length": torch.tensor(1),
        "dataset_slug": "unit_dataset",
    }

    batch = OnlineVLACollator(vlm_processor=_Processor())([item, item])

    assert "pixel_values" in batch
    assert "f_vision" not in batch
    assert batch["pixel_values"].shape == (2, 2, 3, 12, 12)
    assert batch["dataset_slug"] == ["unit_dataset", "unit_dataset"]


def test_online_vla_collator_micro_batches_vlm_processor_outputs():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    class _Processor:
        tokenizer = _Tokenizer()

        def __init__(self):
            self.calls: list[list[str]] = []

        def __call__(self, *, text, images, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            self.calls.append(list(text))
            token_len = max(len(item) for item in text)
            input_ids = torch.ones(len(text), token_len, dtype=torch.long)
            attention_mask = torch.ones(len(text), token_len, dtype=torch.long)
            image_count = sum(len(sample_images) for sample_images in images)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": torch.zeros(image_count, 3, 4, 4),
                "image_grid_thw": torch.ones(image_count, 3, dtype=torch.long),
            }

    def _item(text: str) -> dict[str, object]:
        return {
            "action": torch.zeros(4, 2),
            "loss_mask": torch.ones(4),
            "pixel_values": torch.zeros(2, 3, 12, 12),
            "vlm_text": text,
            "vlm_images": [object(), object()],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit_dataset",
        }

    processor = _Processor()
    collator = OnlineVLACollator(vlm_processor=processor, vlm_processor_micro_batch_size=2)

    batch = collator([_item("a"), _item("abcd"), _item("ab"), _item("abcdef"), _item("abc")])

    assert processor.calls == [["a", "abcd"], ["ab", "abcdef"], ["abc"]]
    assert batch["vlm_inputs"]["input_ids"].shape == (5, 8)
    assert batch["vlm_inputs"]["attention_mask"].shape == (5, 8)
    assert batch["vlm_inputs"]["pixel_values"].shape == (10, 3, 4, 4)
    assert batch["vlm_inputs"]["image_grid_thw"].shape == (10, 3)


def test_online_vla_collator_micro_batches_preserve_left_padding_layout():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            lengths = [len(item) for item in text]
            max_len = max(lengths)
            input_ids = torch.zeros(len(text), max_len, dtype=torch.long)
            attention_mask = torch.zeros(len(text), max_len, dtype=torch.long)
            image_count = sum(len(sample_images) for sample_images in images)
            pixel_values = torch.zeros(image_count, 3, 4, 4)
            image_grid_thw = torch.ones(image_count, 3, dtype=torch.long)
            for row, length in enumerate(lengths):
                input_ids[row, max_len - length :] = length
                attention_mask[row, max_len - length :] = 1
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

    def _item(text: str) -> dict[str, object]:
        return {
            "action": torch.zeros(4, 2),
            "loss_mask": torch.ones(4),
            "pixel_values": torch.zeros(2, 3, 12, 12),
            "vlm_text": text,
            "vlm_images": [object(), object()],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit_dataset",
        }

    batch = [_item("a"), _item("abcd"), _item("ab"), _item("abcdef"), _item("abc")]
    processor = _Processor()
    full = OnlineVLACollator(vlm_processor=processor)(batch)
    micro = OnlineVLACollator(vlm_processor=processor, vlm_processor_micro_batch_size=2)(batch)

    assert torch.equal(micro["vlm_inputs"]["input_ids"], full["vlm_inputs"]["input_ids"])
    assert torch.equal(micro["vlm_inputs"]["attention_mask"], full["vlm_inputs"]["attention_mask"])


def test_online_vla_collator_final_alignment_padding_respects_left_padding_side():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "left"

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor(
                    [
                        [0, 0, 3, 3, 3, 3],
                        [6, 6, 6, 6, 6, 6],
                    ],
                    dtype=torch.long,
                ),
                "attention_mask": torch.tensor(
                    [
                        [0, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1],
                    ],
                    dtype=torch.long,
                ),
            }

    def _item(text: str) -> dict[str, object]:
        return {
            "action": torch.zeros(4, 2),
            "loss_mask": torch.ones(4),
            "pixel_values": torch.zeros(2, 3, 12, 12),
            "vlm_text": text,
            "vlm_images": [],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit_dataset",
        }

    batch = OnlineVLACollator(vlm_processor=_Processor())([_item("abcd"), _item("abcdef")])

    assert torch.equal(
        batch["vlm_inputs"]["input_ids"],
        torch.tensor(
            [
                [0, 0, 0, 0, 3, 3, 3, 3],
                [0, 0, 6, 6, 6, 6, 6, 6],
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        batch["vlm_inputs"]["attention_mask"],
        torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.bool,
        ),
    )


def test_online_vla_collator_final_alignment_padding_uses_configured_multiple():
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

    class _Processor:
        tokenizer = _Tokenizer()

        def __call__(self, *, text, images, padding, return_tensors):
            del images
            assert padding is True
            assert return_tensors == "pt"
            max_len = max(len(item) for item in text)
            input_ids = torch.ones(len(text), max_len, dtype=torch.long)
            attention_mask = torch.ones(len(text), max_len, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

    def _item(text: str) -> dict[str, object]:
        return {
            "action": torch.zeros(4, 2),
            "loss_mask": torch.ones(4),
            "pixel_values": torch.zeros(2, 3, 12, 12),
            "vlm_text": text,
            "vlm_images": [],
            "frame_delay": torch.tensor(0.0),
            "q_current": torch.zeros(3),
            "task_index": torch.tensor(0),
            "episode_index": torch.tensor(0),
            "episode_step_index": torch.tensor(0),
            "episode_length": torch.tensor(1),
            "dataset_slug": "unit_dataset",
        }

    batch = OnlineVLACollator(
        vlm_processor=_Processor(),
        vlm_sequence_padding_multiple=16,
    )([_item("abcdefghi")])

    assert batch["vlm_inputs"]["input_ids"].shape == (1, 16)
    assert batch["vlm_inputs"]["attention_mask"].shape == (1, 16)


class _CountingDataset:
    def __init__(self):
        self.calls: list[int] = []
        self.episode_data_index = {"from": [0], "to": [1]}

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        self.calls.append(idx)
        return {
            "episode_index": 0,
            "observation.images.front": torch.zeros(3, 8, 8),
            "observation.images.wrist": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(6),
            "action": torch.zeros(16, 6),
            "task": "pick",
        }


def test_online_vla_dataset_pack_mode_reads_source_sample_once():
    dataset = _CountingDataset()
    vlm = _FakeVLMProcessor()
    sample_ds = OnlineVLADataset(
        lerobot_dataset=dataset,
        vlm_processor=vlm,
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        sample_frame_delay=False,
    )

    _ = sample_ds[0]

    assert dataset.calls == [0]


def test_online_vla_dataset_delay_zero_reads_source_sample_once():
    dataset = _CountingDataset()
    sample_ds = OnlineVLADataset(
        lerobot_dataset=dataset,
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        sample_frame_delay=True,
        delay_max_chunks=1,
        delay_chunk_size_threshold=0.0,
    )

    _ = sample_ds[0]

    assert dataset.calls == [0]


def test_online_vla_dataset_reads_slow_frame_before_fast_frame_when_delay_is_nonzero(monkeypatch):
    class _CountingLongDataset:
        def __init__(self):
            self.calls: list[int] = []
            self.num_episodes = 1
            self.episode_data_index = {"from": [0], "to": [64]}

        def __len__(self) -> int:
            return 64

        def __getitem__(self, idx: int):
            self.calls.append(int(idx))
            return {
                "episode_index": 0,
                "observation.images.front": torch.zeros(3, 8, 8),
                "observation.images.wrist": torch.zeros(3, 8, 8),
                "observation.state": torch.zeros(6),
                "action": torch.zeros(16, 6),
                "task": f"task-{idx}",
            }

    def _choose_first_async_bucket(population, weights, k):
        return [next(interval for interval in population if interval[0] > 0)]

    monkeypatch.setattr("servovla.data.dataset_loader.random.choices", _choose_first_async_bucket)
    monkeypatch.setattr("servovla.data.dataset_loader.random.randint", lambda start, _end: start)

    dataset = _CountingLongDataset()
    sample_ds = OnlineVLADataset(
        lerobot_dataset=dataset,
        vlm_processor=_FakeVLMProcessor(),
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        sample_frame_delay=True,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.25,
        delay_sync_weight=0.0,
    )

    sample = sample_ds[20]

    assert int(sample["frame_delay"].item()) == 12
    assert dataset.calls == [8, 20]


def test_online_vla_dataset_pack_mode_caches_prompt_text():
    dataset = _CountingDataset()
    vlm = _FakeVLMProcessor()
    sample_ds = OnlineVLADataset(
        lerobot_dataset=dataset,
        vlm_processor=vlm,
        vision_processor=_RecordingVisionProcessor(),
        action_horizon=16,
        vision_image_size=12,
        vlm_image_size=20,
        action_is_delta=True,
        sample_frame_delay=False,
        prompt_cache_size=8,
    )

    _ = sample_ds[0]
    dataset.calls.clear()
    _ = sample_ds[0]

    assert vlm.templates == 1


def test_raw_collator_model_facing_vlm_keys_match_qwen3_processor_contract():
    IMAGE_TOKEN_ID = 151655

    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        padding_side = "right"

        def __call__(self, text, padding, return_tensors):
            del text
            assert padding is True
            assert return_tensors == "pt"
            return {
                "input_ids": torch.tensor(
                    [[10, IMAGE_TOKEN_ID, 20, IMAGE_TOKEN_ID, 30]], dtype=torch.long
                ),
                "attention_mask": torch.ones(1, 5, dtype=torch.long),
            }

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _Processor:
        __module__ = "transformers.models.qwen3_vl.processing_qwen3_vl"
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    _Processor.image_token_id = IMAGE_TOKEN_ID

    item = {
        "action": torch.zeros(4, 2),
        "loss_mask": torch.ones(4),
        "vision_images_uint8": [torch.zeros(3, 8, 8, dtype=torch.uint8)],
        "vlm_images_uint8": [torch.zeros(3, 8, 8, dtype=torch.uint8)],
        "vlm_text": "<image> pick",
        "frame_delay": torch.tensor(0.0),
        "q_current": torch.zeros(3),
        "task_index": torch.tensor(0),
        "episode_index": torch.tensor(0),
        "episode_step_index": torch.tensor(0),
        "episode_length": torch.tensor(1),
        "dataset_slug": "unit",
    }

    batch = OnlineVLACollator(vlm_processor=_Processor(), raw_image_mode=True, vlm_image_size=512)(
        [item]
    )

    assert set(["input_ids", "attention_mask", "image_grid_thw", "mm_token_type_ids"]).issubset(
        batch["vlm_inputs"]
    )
    assert batch["vlm_inputs"]["input_ids"].dtype == torch.long
    assert batch["vlm_inputs"]["attention_mask"].dtype == torch.long
    assert batch["vlm_inputs"]["image_grid_thw"].dtype == torch.long
    assert batch["vlm_inputs"]["mm_token_type_ids"].dtype == torch.long
    assert batch["vlm_inputs"]["image_grid_thw"].shape == (1, 3)
    assert batch["vlm_inputs"]["mm_token_type_ids"].shape == batch["vlm_inputs"]["input_ids"].shape
