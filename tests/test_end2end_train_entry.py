from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from scripts import train as train_script


class _Processor:
    class _Tokenizer:
        pad_token_id = 0
        eos_token_id = 1
        pad_token = "<pad>"
        eos_token = "</s>"

    tokenizer = _Tokenizer()
    pad_token = "<pad>"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        del tokenize, add_generation_prompt
        return messages[0]["content"][-1]["text"]

    def __call__(self, **kwargs):
        if "text" in kwargs:
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.ones(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }
        images = kwargs["images"]
        size = kwargs["size"]
        return {"pixel_values": torch.zeros(len(images), 3, size["height"], size["width"])}


class _LeRobotDataset:
    instances = []

    def __init__(
        self,
        name,
        root=None,
        episodes=None,
        delta_timestamps=None,
        download_videos=True,
        video_backend=None,
    ):
        self.name = name
        self.root = root
        self.episodes = episodes
        self.delta_timestamps = delta_timestamps
        self.download_videos = download_videos
        self.video_backend = video_backend
        max_episode = max(episodes) if episodes else 0
        self.num_episodes = max_episode + 1
        self.episode_data_index = {
            "from": [idx * 2 for idx in range(max_episode + 1)],
            "to": [idx * 2 + 2 for idx in range(max_episode + 1)],
        }
        _LeRobotDataset.instances.append(self)
        self.fps = 30

    def __len__(self):
        return 2

    def __getitem__(self, idx):
        return {
            "episode_index": 0,
            "observation.images.front": torch.zeros(3, 8, 8),
            "observation.images.wrist": torch.zeros(3, 8, 8),
            "observation.state": torch.zeros(6),
            "action": torch.zeros(8, 6),
            "task": "pick",
            "task_index": 0,
        }


@pytest.fixture(autouse=True)
def _stub_lerobot_local_root(monkeypatch, tmp_path):
    def _dataset_root(dataset_name):
        root = tmp_path / str(dataset_name).replace("/", "_")
        (root / "meta").mkdir(parents=True, exist_ok=True)
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            info_path.write_text('{"fps": 30}', encoding="utf-8")
        return root

    monkeypatch.setattr(
        train_script,
        "require_lerobot_local_dataset_root",
        _dataset_root,
    )


def _cfg(tmp_path):
    return OmegaConf.create(
        {
            "seed": 1,
            "output_dir": str(tmp_path / "current"),
            "output_root": str(tmp_path),
            "dataset": {
                "train_names": ["local/train"],
                "val_names": ["local/val"],
                "train_split": "train",
                "val_split": "val",
                "action_mode": "delta",
                "action_dim": 6,
                "camera_keys": ["observation.images.front", "observation.images.wrist"],
                "camera_key_aliases": {
                    "front": ["observation.images.front"],
                    "wrist": ["observation.images.wrist"],
                },
                "state_keys": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
                "task_key_candidates": ["task", "language_instruction"],
                "proprio_dim": 6,
                "num_cameras": 2,
            },
            "model": {
                "policy_head": {
                    "chunk_size": 8,
                    "action_dim": 6,
                    "state_dim": 6,
                    "num_inference_steps": 2,
                    "hidden_dim": 64,
                    "num_layers": 1,
                    "num_heads": 4,
                    "dropout": 0.0,
                    "vision_feature_dim": 64,
                    "semantic_feature_dim": 32,
                    "vision_grid_size": 1,
                    "num_cameras": 2,
                },
                "vision_encoder": {
                    "model_id": "vision",
                    "feature_dim": 64,
                    "image_size": 8,
                },
                "vlm_encoder": {
                    "model_id": "vlm",
                    "feature_dim": 32,
                    "image_size": 8,
                },
            },
            "training": {
                "device": "cpu",
                "amp_dtype": "float32",
                "compile": False,
                "batch_size": 2,
                "num_workers": 0,
                "pin_memory": False,
                "prefetch_factor": 2,
                "drop_last": False,
                "shuffle_episodes": True,
                "eval_batch_size": 2,
                "eval_num_workers": 0,
                "eval_prefetch_factor": 2,
                "prompt_cache_size": 4,
                "semantic_delay": {
                    "max_delay_chunks": 3,
                    "chunk_size_threshold": 0.5,
                    "sync_weight": 1.0,
                    "async_bucket_weights": None,
                },
                "optimizer": {
                    "lr": 1e-3,
                    "weight_decay": 0.0,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                },
                "scheduler": {
                    "T_max": 2,
                    "eta_min": 1e-6,
                    "warmup_steps": 0,
                    "warmup_start_factor": 0.1,
                },
                "ema": {
                    "decay": 0.9,
                    "update_after_step": 0,
                    "update_every": 1,
                },
                "async_eval": {"enabled": False},
                "max_steps": 1,
                "grad_clip_norm": 1.0,
                "log_every": 1,
                "save_every": 1,
                "eval_every": 0,
                "eval_seed": 123,
                "eval_zero_noise": True,
                "eval_use_ema": True,
            },
            "deployment": {
                "enabled": False,
                "auto_export": False,
            },
        }
    )


class _TinyPolicyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = torch.nn.Linear(2, 1)


@pytest.mark.parametrize(
    ("use_ema", "expected_weight", "expected_bias"),
    [(False, [[3.0, 4.0]], [5.0]), (True, [[6.0, 7.0]], [8.0])],
)
def test_load_base_checkpoint_initializes_raw_or_ema_weights(
    tmp_path, use_ema, expected_weight, expected_bias
):
    checkpoint_path = tmp_path / "base.pt"
    torch.save(
        {
            "step": 123,
            "model_state": {
                "policy_head.weight": torch.tensor([[3.0, 4.0]]),
                "policy_head.bias": torch.tensor([5.0]),
            },
            "ema_state": {
                "policy_head.weight": torch.tensor([[6.0, 7.0]]),
                "policy_head.bias": torch.tensor([8.0]),
            },
        },
        checkpoint_path,
    )
    model = _TinyPolicyModel()
    with torch.no_grad():
        model.policy_head.weight.zero_()
        model.policy_head.bias.zero_()
    cfg = OmegaConf.create(
        {
            "training": {
                "base_checkpoint": str(checkpoint_path),
                "base_checkpoint_use_ema": use_ema,
            }
        }
    )

    train_script._load_base_checkpoint_into_model(cfg, model)

    assert torch.equal(model.policy_head.weight, torch.tensor(expected_weight))
    assert torch.equal(model.policy_head.bias, torch.tensor(expected_bias))


def test_action_normalization_can_seed_from_base_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "base.pt"
    mean = [[0.1] * 6 for _ in range(8)]
    std = [[1.5] * 6 for _ in range(8)]
    torch.save(
        {
            "step": 123,
            "model_state": {},
            "action_normalization": {
                "enabled": True,
                "mean": mean,
                "std": std,
                "eps": 1.0e-5,
                "action_mode": "delta",
                "chunk_size": 8,
                "action_dim": 6,
            },
        },
        checkpoint_path,
    )
    cfg = _cfg(tmp_path)
    cfg.training.base_checkpoint = str(checkpoint_path)
    cfg.training.base_checkpoint_use_ema = False
    cfg.training.action_normalization = {
        "enabled": True,
        "source": "train_dataset",
        "mode": "per_horizon_dim",
        "eps": 1.0e-6,
        "mean": None,
        "std": None,
    }

    normalizer = train_script._resolve_action_normalization(cfg, dataset_names=["local/train"])

    assert normalizer.enabled is True
    assert normalizer.mean.shape == (8, 6)
    assert normalizer.std.shape == (8, 6)
    assert cfg.training.action_normalization.mean == mean
    assert cfg.training.action_normalization.std == std
    assert cfg.training.action_normalization.eps == 1.0e-5


def test_build_end2end_dataloader_uses_raw_lerobot_dataset(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )
    batch = next(iter(loader))

    assert "pixel_values" in batch
    assert "vlm_inputs" in batch
    assert batch["action"].shape == (2, 8, 6)
    assert batch["dataset_slug"] == ["train", "train"]


def test_build_end2end_dataloader_uses_lerobot_dataset_fps_for_raw_action_deltas(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    dataset_root = tmp_path / "local_train"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "meta" / "info.json").write_text('{"fps": 30}', encoding="utf-8")
    processor = _Processor()

    train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    assert _LeRobotDataset.instances[0].delta_timestamps == {"action": [i / 30 for i in range(8)]}


def test_build_end2end_dataloader_passes_vlm_processor_micro_batch_size(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.vlm_processor_micro_batch_size = 3
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.collate_fn.vlm_processor_micro_batch_size == 3


def test_build_end2end_dataloader_passes_vlm_sequence_padding_multiple(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.vlm_sequence_padding_multiple = 16
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.collate_fn.vlm_sequence_padding_multiple == 16


def test_build_end2end_dataloader_passes_configured_video_decode_backend(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.decode = {
        "video_backend": "pyav_cuda",
        "device": "cuda:2",
        "fallback_backend": "torchcodec",
    }
    processor = _Processor()

    train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    assert _LeRobotDataset.instances[0].video_backend == "servovla_pyav_cuda"


def test_build_end2end_dataloader_uses_spawn_workers_for_cuda_video_decode(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.decode = {
        "video_backend": "pyav_cuda",
        "device": "cuda:2",
        "fallback_backend": "torchcodec",
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=1,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_build_end2end_dataloader_caps_workers_for_cuda_video_decode(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.decode = {
        "video_backend": "pyav_cuda",
        "device": "cuda:2",
        "fallback_backend": "torchcodec",
        "num_workers": 1,
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=8,
        repeat=True,
        shuffle_episodes=False,
    )

    assert loader.num_workers == 1
    assert loader.multiprocessing_context.get_start_method() == "spawn"


def test_build_end2end_dataloader_installs_worker_thread_limiter(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    processor = _Processor()
    calls = []

    monkeypatch.setattr(
        train_script.torch, "set_num_threads", lambda value: calls.append(("intra", value))
    )
    monkeypatch.setattr(
        train_script.torch,
        "set_num_interop_threads",
        lambda value: calls.append(("interop", value)),
    )

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=1,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.worker_init_fn is not None
    loader.worker_init_fn(0)
    assert calls == [("intra", 1), ("interop", 1)]


def test_build_end2end_dataloader_enables_raw_mode_only_when_both_gpu_flags_true(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda"
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
    }

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

    class _RawProcessor(_Processor):
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<image><image> pick"

    processor = _RawProcessor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )

    batch = next(iter(loader))

    assert loader.dataset.raw_image_mode is True
    assert loader.collate_fn.raw_image_mode is True
    assert "vision_images_uint8" in batch
    assert "pixel_values" not in batch
    assert "pixel_values" not in batch["vlm_inputs"]


def test_build_end2end_dataloader_passes_raw_decode_span_to_dataset(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.decode = {"max_batch_decode_span_s": 6.5}
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )

    assert loader.dataset.raw_video_batch_decode_max_span_s == 6.5


def test_build_end2end_dataloader_keeps_cpu_path_when_async_producer_false(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda"
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": False,
        "cpu_prefetch_depth": 2,
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )
    batch = next(iter(loader))

    assert loader.dataset.raw_image_mode is False
    assert loader.collate_fn.raw_image_mode is False
    assert "pixel_values" in batch
    assert "vision_images_uint8" not in batch


def test_build_end2end_dataloader_keeps_cpu_path_on_non_cuda_devices(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "mps"
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
    }

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

    class _RawCapableProcessor:
        __module__ = "transformers.models.qwen3_vl.processing_qwen3_vl"
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<image><image> pick"

        def __call__(self, **kwargs):
            if "text" in kwargs:
                text = kwargs["text"]
                assert kwargs["padding"] is True
                assert kwargs["return_tensors"] == "pt"
                image_count = sum(len(sample_images) for sample_images in kwargs.get("images", []))
                return {
                    "input_ids": torch.ones(len(text), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), 4, dtype=torch.long),
                    "pixel_values": torch.zeros(image_count, 3, 8, 8),
                    "image_grid_thw": torch.ones(image_count, 3, dtype=torch.long),
                }
            images = kwargs["images"]
            size = kwargs["size"]
            assert kwargs["return_tensors"] == "pt"
            return {
                "pixel_values": torch.zeros(len(images), 3, size["height"], size["width"]),
            }

    processor = _RawCapableProcessor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )
    batch = next(iter(loader))

    assert loader.dataset.raw_image_mode is False
    assert loader.collate_fn.raw_image_mode is False
    assert "pixel_values" in batch
    assert "vision_images_uint8" not in batch


def test_build_end2end_dataloader_enables_raw_mode_for_cuda_eval_gpu_preprocess(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda:0"
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
    }

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

    class _RawCapableProcessor:
        __module__ = "transformers.models.qwen3_vl.processing_qwen3_vl"
        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "<image><image> pick"

    processor = _RawCapableProcessor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )
    batch = next(iter(loader))

    assert loader.dataset.raw_image_mode is True
    assert loader.collate_fn.raw_image_mode is True
    assert "vision_images_uint8" in batch
    assert "pixel_values" not in batch


def test_build_vlm_compile_warmup_dataloader_uses_sequential_raw_train_loader(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda:1"
    cfg.training.compile = True
    cfg.training.compile_vlm_encoder = True
    cfg.training.vlm_compile = {
        "warmup_enabled": True,
        "warmup_max_batches": 2,
        "warmup_batch_size": None,
        "warmup_num_workers": 0,
    }
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
    }
    processor = _Processor()

    loader = train_script._build_vlm_compile_warmup_dataloader(
        cfg,
        vlm_processor=processor,
        vision_processor=processor,
    )

    assert loader is not None
    assert loader.dataset.raw_image_mode is True
    assert loader.collate_fn.raw_image_mode is True
    assert loader.num_workers == 0
    assert loader.batch_sampler.batch_size == cfg.training.batch_size
    assert _LeRobotDataset.instances[-1].name == "local/train"


def test_build_vlm_compile_warmup_dataloader_defaults_to_vlm_micro_batch(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda:1"
    cfg.training.batch_size = 64
    cfg.training.compile = True
    cfg.training.compile_vlm_encoder = True
    cfg.training.vlm_compile = {
        "warmup_enabled": True,
        "warmup_max_batches": 2,
        "warmup_batch_size": None,
        "warmup_num_workers": 0,
    }
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
        "vlm_encoder_micro_batch_size": 8,
    }
    processor = _Processor()

    loader = train_script._build_vlm_compile_warmup_dataloader(
        cfg,
        vlm_processor=processor,
        vision_processor=processor,
    )

    assert loader is not None
    assert loader.batch_sampler.batch_size == 8


def test_build_vlm_compile_warmup_dataloader_can_use_synthetic_raw_batches(monkeypatch, tmp_path):
    class _ExplodingLeRobotDataset(_LeRobotDataset):
        def __getitem__(self, idx):
            raise AssertionError("synthetic VLM warmup must not materialize real dataset items")

    _ExplodingLeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _ExplodingLeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda:1"
    cfg.training.batch_size = 64
    cfg.training.compile = True
    cfg.training.compile_vlm_encoder = True
    cfg.training.vlm_compile = {
        "warmup_enabled": True,
        "warmup_max_batches": 2,
        "warmup_batch_size": None,
        "warmup_num_workers": 0,
        "synthetic_raw_batches": True,
    }
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
        "vlm_encoder_micro_batch_size": 8,
    }

    class _RawCapableProcessor(_Processor):
        class _Tokenizer(_Processor._Tokenizer):
            def __call__(self, text, padding, return_tensors):
                assert padding is True
                assert return_tensors == "pt"
                return {
                    "input_ids": torch.ones(len(text), 4, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), 4, dtype=torch.long),
                }

        class _ImageProcessor:
            patch_size = 2
            merge_size = 2
            temporal_patch_size = 2
            size = {"shortest_edge": 8 * 8, "longest_edge": 8 * 8}

        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()
        image_token = "<image>"

    processor = _RawCapableProcessor()

    loader = train_script._build_vlm_compile_warmup_dataloader(
        cfg,
        vlm_processor=processor,
        vision_processor=processor,
    )

    assert loader is not None
    batch = next(iter(loader))
    assert batch["action"].shape == (8, 8, 6)
    assert batch["vision_images_uint8"]["batch_size"] == 8
    assert batch["vlm_images_uint8"]["num_cameras"] == 2
    assert batch["vlm_inputs"]["input_ids"].shape[0] == 8
    assert batch["vlm_inputs"]["image_grid_thw"].shape[0] == 16
    assert "pixel_values" not in batch


def test_synthetic_vlm_compile_warmup_uses_real_task_prompt_lengths(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda:1"
    cfg.training.batch_size = 16
    cfg.training.vlm_sequence_padding_multiple = 16
    cfg.training.compile = True
    cfg.training.compile_vlm_encoder = True
    cfg.training.vlm_compile = {
        "warmup_enabled": True,
        "warmup_max_batches": 2,
        "warmup_batch_size": 8,
        "warmup_num_workers": 0,
        "synthetic_raw_batches": True,
        "signature_max_entries": 8,
    }
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
        "vlm_encoder_micro_batch_size": 8,
    }

    class _LengthAwareProcessor(_Processor):
        image_token = "<image>"
        image_token_id = 99

        class _Tokenizer(_Processor._Tokenizer):
            def __call__(self, text, padding, return_tensors):
                del padding, return_tensors
                lengths = [24 if "long instruction" in value else 9 for value in text]
                max_len = max(lengths)
                return {
                    "input_ids": torch.ones(len(text), max_len, dtype=torch.long),
                    "attention_mask": torch.ones(len(text), max_len, dtype=torch.long),
                }

        class _ImageProcessor:
            patch_size = 2
            merge_size = 2
            temporal_patch_size = 2
            size = {"shortest_edge": 8 * 8, "longest_edge": 8 * 8}

        tokenizer = _Tokenizer()
        image_processor = _ImageProcessor()

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del tokenize, add_generation_prompt
            image_count = sum(1 for entry in messages[0]["content"] if entry.get("type") == "image")
            text = messages[0]["content"][-1]["text"]
            return (self.image_token * image_count) + text

    processor = _LengthAwareProcessor()
    lerobot_dataset = _LeRobotDataset("local/train", episodes=[0])
    lerobot_dataset.meta = SimpleNamespace(
        tasks=SimpleNamespace(index=["short task", "long instruction with more words"])
    )
    train_dataset = train_script.OnlineVLADataset(
        lerobot_dataset=lerobot_dataset,
        vlm_processor=processor,
        vision_processor=processor,
        action_horizon=8,
        camera_keys=list(cfg.dataset.camera_keys),
        task_key_candidates=list(cfg.dataset.task_key_candidates),
        raw_image_mode=True,
        vision_image_size=8,
        vlm_image_size=8,
    )

    loader = train_script._build_vlm_compile_warmup_dataloader(
        cfg,
        vlm_processor=processor,
        vision_processor=processor,
        train_dataloader=SimpleNamespace(dataset=train_dataset),
    )

    assert loader is not None
    observed_lengths = [int(batch["vlm_inputs"]["input_ids"].shape[1]) for batch in loader]
    assert sorted(set(observed_lengths)) == [16, 32]


def test_train_entry_builds_raw_image_preprocessor_when_raw_async_flags_enabled(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.training.device = "cuda"
    cfg.training.gpu_pipeline = {
        "enabled": True,
        "feature_queue_depth": 2,
        "encoder_parallel_streams": True,
        "gpu_preprocess": True,
        "async_producer": True,
        "cpu_prefetch_depth": 2,
    }
    cfg.model.vision_encoder.image_size = 8
    cfg.model.vlm_encoder.image_size = 512

    class _VisionProcessor:
        image_mean = [0.1, 0.2, 0.3]
        image_std = [0.4, 0.5, 0.6]
        rescale_factor = 1.0 / 255.0

    class _ImageProcessor:
        patch_size = 14
        merge_size = 2
        temporal_patch_size = 2
        image_mean = [0.1, 0.2, 0.3]
        image_std = [0.4, 0.5, 0.6]
        rescale_factor = 1.0 / 255.0
        size = {"shortest_edge": 56 * 56, "longest_edge": 28 * 28 * 1280}

    class _VLMProcessor:
        image_processor = _ImageProcessor()

    preprocessor = train_script._build_raw_image_gpu_preprocessor(
        cfg,
        vision_processor=_VisionProcessor(),
        vlm_processor=_VLMProcessor(),
        device=torch.device("cuda"),
    )

    assert preprocessor is not None
    assert preprocessor.device.type == "cuda"
    assert preprocessor.spec.vision_image_size == 8
    assert preprocessor.spec.vlm_image_size == 512
    assert preprocessor.spec.qwen_smart_height == 504
    assert preprocessor.spec.qwen_smart_width == 504


@pytest.mark.parametrize("threshold", [-0.01, 0.5001])
def test_build_end2end_dataloader_rejects_training_semantic_delay_threshold_outside_contract(
    monkeypatch,
    tmp_path,
    threshold,
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.semantic_delay.chunk_size_threshold = threshold
    processor = _Processor()

    with pytest.raises(ValueError, match="chunk_size_threshold"):
        train_script.build_end2end_dataloader(
            cfg,
            dataset_names=["local/train"],
            split="train",
            vlm_processor=processor,
            vision_processor=processor,
            batch_size=2,
            num_workers=0,
            repeat=False,
            shuffle_episodes=False,
        )


def test_build_end2end_dataloader_honors_dataset_episode_filters(monkeypatch, tmp_path):
    class _Metadata:
        def __init__(self, repo_id, root=None):
            self.repo_id = repo_id
            self.root = root
            self.episodes = [
                {"tasks": ["task_a"]},
                {"tasks": ["task_a"]},
                {"tasks": ["task_b"]},
                {"tasks": ["task_b"]},
            ]
            self.tasks = {"task_a": 3, "task_b": 7}

    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    monkeypatch.setattr(train_script, "LeRobotDatasetMetadata", _Metadata, raising=False)
    cfg = _cfg(tmp_path)
    cfg.dataset.task_index_allowlist = [3, 7]
    cfg.dataset.max_episodes_per_task = 1
    processor = _Processor()

    train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    assert _LeRobotDataset.instances[0].episodes == [0, 2]


def test_build_end2end_dataloader_supports_multiple_dataset_names(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train_a", "local/train_b"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )
    batches = list(iter(loader))

    assert [dataset.name for dataset in _LeRobotDataset.instances] == [
        "local/train_a",
        "local/train_b",
    ]
    assert [batch["dataset_slug"] for batch in batches] == [
        ["train_a", "train_a"],
        ["train_b", "train_b"],
    ]


def test_train_dataloader_requires_weights_for_multiple_train_datasets(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.train_names = ["local/train_a", "local/train_b"]
    processor = _Processor()

    with pytest.raises(ValueError, match="dataset.train_weights is required"):
        train_script.build_end2end_dataloader(
            cfg,
            dataset_names=["local/train_a", "local/train_b"],
            split="train",
            vlm_processor=processor,
            vision_processor=processor,
            batch_size=2,
            num_workers=0,
            repeat=True,
            shuffle_episodes=True,
        )


def test_train_dataloader_supports_equal_seen_frames_strategy(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.train_names = ["local/train_a", "local/train_b"]
    cfg.dataset.train_weight_strategy = "equal_seen_frames"
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train_a", "local/train_b"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=True,
    )

    assert loader.batch_sampler.weights == [1.0, 1.0]


def test_train_dataloader_mixes_datasets_by_explicit_weights(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.train_names = ["local/train_a", "local/train_b"]
    cfg.dataset.train_weights = [1.0, 1.0]
    cfg.training.raw_shuffle = {
        "window_steps": 4,
        "active_episodes_per_dataset": 1,
        "window_boundary_offsets": True,
        "reactivate_when_remaining_below": 3,
        "seed_stride": 17,
        "strategy": "active_window",
        "quota_max_datasets_per_batch": 2,
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train_a", "local/train_b"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )

    batch = next(iter(loader))
    assert sorted(batch["dataset_slug"]) == ["train_a", "train_b"]
    assert loader.batch_sampler.reactivate_when_remaining_below == 3
    assert loader.batch_sampler.seed_stride == 17
    assert loader.batch_sampler.window_boundary_offsets is True
    assert loader.batch_sampler.quota_max_datasets_per_batch == 2


def test_train_dataloader_passes_active_window_single_dataset_strategy(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.train_names = ["local/train_a", "local/train_b"]
    cfg.dataset.train_weights = [1.0, 1.0]
    cfg.training.raw_shuffle = {
        "window_steps": 4,
        "active_episodes_per_dataset": 1,
        "strategy": "active_window",
        "batch_dataset_strategy": "single_dataset",
        "batch_dataset_burst_batches": 1,
        "max_episodes_per_batch": 1,
        "episode_burst_batches": 2,
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train_a", "local/train_b"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=True,
        shuffle_episodes=False,
    )

    batch = next(iter(loader))
    assert len(set(batch["dataset_slug"])) == 1
    assert loader.batch_sampler.batch_dataset_strategy == "single_dataset"
    assert loader.batch_sampler.batch_dataset_burst_batches == 1
    assert loader.batch_sampler.max_episodes_per_batch == 1
    assert loader.batch_sampler.episode_burst_batches == 2


def test_build_end2end_dataloader_uses_epoch_chunk_sampler_when_configured(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.raw_shuffle = {
        "strategy": "epoch_chunk",
        "chunk_steps": 4,
        "shuffle_chunks": True,
        "reshuffle_each_epoch": False,
        "epoch_boundary_offsets": True,
        "seed_stride": 17,
        "batch_dataset_strategy": "single_dataset",
        "batch_dataset_burst_batches": 5,
        "max_episodes_per_batch": 2,
        "episode_burst_batches": 3,
    }
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        ["local/unit"],
        "train",
        repeat=True,
        shuffle_episodes=True,
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
    )

    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    assert isinstance(loader.batch_sampler, EpochChunkRawBatchSampler)
    assert loader.batch_sampler.chunk_steps == 4
    assert loader.batch_sampler.seed_stride == 17
    assert loader.batch_sampler.reshuffle_each_epoch is False
    assert loader.batch_sampler.batch_dataset_strategy == "single_dataset"
    assert loader.batch_sampler.batch_dataset_burst_batches == 5
    assert loader.batch_sampler.max_episodes_per_batch == 2
    assert loader.batch_sampler.episode_burst_batches == 3


def test_build_end2end_dataloader_uses_batch_aware_concat_for_multiple_train_datasets(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.train_weights = [1.0, 1.0]
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        ["local/train_a", "local/train_b"],
        "train",
        repeat=True,
        shuffle_episodes=True,
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
    )

    from servovla.data.dataset_loader import BatchAwareConcatDataset

    assert isinstance(loader.dataset, BatchAwareConcatDataset)
    assert hasattr(loader.dataset, "__getitems__")


def test_build_end2end_dataloader_defaults_to_epoch_chunk_sampler(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.raw_shuffle = {}
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        ["local/unit"],
        "train",
        repeat=True,
        shuffle_episodes=False,
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
    )

    from servovla.data.raw_window_sampler import EpochChunkRawBatchSampler

    assert isinstance(loader.batch_sampler, EpochChunkRawBatchSampler)
    assert loader.batch_sampler.chunk_steps == 128
    assert loader.batch_sampler.shuffle_chunks is True
    assert loader.batch_sampler.reshuffle_each_epoch is True
    assert loader.batch_sampler.epoch_boundary_offsets is True
    assert loader.batch_sampler.seed_stride == 100003


def test_validation_dataloader_allows_multiple_val_datasets_without_train_weights(
    monkeypatch, tmp_path
):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.dataset.val_names = ["local/val_a", "local/val_b"]
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val_a", "local/val_b"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    batches = list(iter(loader))
    assert [batch["dataset_slug"] for batch in batches] == [["val_a", "val_a"], ["val_b", "val_b"]]


def test_build_end2end_validation_dataloader_keeps_tail_batch(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.drop_last = True
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=4,
        num_workers=0,
        repeat=False,
        shuffle_episodes=False,
    )

    batches = list(iter(loader))
    assert len(batches) == 1
    assert batches[0]["action"].shape[0] == 2


def test_build_end2end_validation_dataloader_uses_eval_prefetch_factor(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.prefetch_factor = 9
    cfg.training.eval_prefetch_factor = 3
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=1,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.prefetch_factor == 3


def test_build_end2end_train_dataloader_can_disable_in_order_delivery(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.dataloader_in_order = False
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/train"],
        split="train",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=1,
        repeat=True,
        shuffle_episodes=False,
    )

    assert loader.in_order is False


def test_build_end2end_validation_dataloader_stays_in_order(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    cfg.training.dataloader_in_order = False
    processor = _Processor()

    loader = train_script.build_end2end_dataloader(
        cfg,
        dataset_names=["local/val"],
        split="val",
        vlm_processor=processor,
        vision_processor=processor,
        batch_size=2,
        num_workers=1,
        repeat=False,
        shuffle_episodes=False,
    )

    assert loader.in_order is True


def test_build_end2end_dataloader_rejects_unsupported_train_split(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    processor = _Processor()

    with pytest.raises(ValueError, match="dataset.train_split"):
        train_script.build_end2end_dataloader(
            cfg,
            dataset_names=["local/train"],
            split="custom",
            vlm_processor=processor,
            vision_processor=processor,
            batch_size=2,
            num_workers=0,
            repeat=True,
            shuffle_episodes=False,
        )


def test_build_end2end_dataloader_rejects_unsupported_val_split(monkeypatch, tmp_path):
    _LeRobotDataset.instances = []
    monkeypatch.setattr(train_script, "LeRobotDataset", _LeRobotDataset)
    cfg = _cfg(tmp_path)
    processor = _Processor()

    with pytest.raises(ValueError, match="dataset.val_split"):
        train_script.build_end2end_dataloader(
            cfg,
            dataset_names=["local/val"],
            split="custom",
            vlm_processor=processor,
            vision_processor=processor,
            batch_size=2,
            num_workers=0,
            repeat=False,
            shuffle_episodes=False,
        )


def test_build_servovla_model_loads_online_encoders(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)

    class _Vision(torch.nn.Module):
        feature_dim = 64

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixel_values):
            return torch.zeros(pixel_values.shape[0], 2, 64)

    class _VLM(torch.nn.Module):
        feature_dim = 32

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, vlm_inputs):
            return torch.zeros(vlm_inputs["input_ids"].shape[0], 4, 32)

    monkeypatch.setattr(train_script, "VisionEncoder", _Vision)
    monkeypatch.setattr(train_script, "VLMEncoder", _VLM)

    model = train_script._build_servovla_model(cfg, device=torch.device("cpu"))

    assert model.vision_encoder is not None
    assert model.vlm_encoder is not None
    assert all(not param.requires_grad for param in model.vision_encoder.parameters())
    assert all(not param.requires_grad for param in model.vlm_encoder.parameters())


def test_build_servovla_model_passes_encoder_attention_backend(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.model.vision_encoder.attn_implementation = "flash_attention_2"
    cfg.model.vlm_encoder.attn_implementation = "flash_attention_2"
    captured = {"vision": None, "vlm": None}

    class _Vision(torch.nn.Module):
        feature_dim = 64

        def __init__(self, model_id, attn_implementation=None):
            super().__init__()
            captured["vision"] = (model_id, attn_implementation)
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixel_values):
            return torch.zeros(pixel_values.shape[0], 2, 64)

    class _VLM(torch.nn.Module):
        feature_dim = 32

        def __init__(self, model_id, attn_implementation=None):
            super().__init__()
            captured["vlm"] = (model_id, attn_implementation)
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, vlm_inputs):
            return torch.zeros(vlm_inputs["input_ids"].shape[0], 4, 32)

    monkeypatch.setattr(train_script, "VisionEncoder", _Vision)
    monkeypatch.setattr(train_script, "VLMEncoder", _VLM)

    train_script._build_servovla_model(cfg, device=torch.device("cpu"))

    assert captured["vision"] == ("vision", "flash_attention_2")
    assert captured["vlm"] == ("vlm", "flash_attention_2")


def test_build_servovla_model_keeps_policy_head_fp32_under_bfloat16_amp(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.training.amp_dtype = "bfloat16"
    cfg.training.policy_head_param_dtype = "float32"

    class _Vision(torch.nn.Module):
        feature_dim = 64

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixel_values):
            return torch.zeros(pixel_values.shape[0], 2, 64)

    class _VLM(torch.nn.Module):
        feature_dim = 32

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, vlm_inputs):
            return torch.zeros(vlm_inputs["input_ids"].shape[0], 4, 32)

    monkeypatch.setattr(train_script, "VisionEncoder", _Vision)
    monkeypatch.setattr(train_script, "VLMEncoder", _VLM)

    model = train_script._build_servovla_model(cfg, device=torch.device("cpu"))

    assert train_script._training_amp_dtype(cfg) == torch.bfloat16
    assert next(model.policy_head.parameters()).dtype == torch.float32


def test_build_servovla_model_defaults_policy_head_dtype_to_amp_dtype(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.training.amp_dtype = "bfloat16"
    if "policy_head_param_dtype" in cfg.training:
        del cfg.training["policy_head_param_dtype"]

    class _Vision(torch.nn.Module):
        feature_dim = 64

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, pixel_values):
            return torch.zeros(pixel_values.shape[0], 2, 64)

    class _VLM(torch.nn.Module):
        feature_dim = 32

        def __init__(self, model_id):
            super().__init__()
            self.model_id = model_id
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, vlm_inputs):
            return torch.zeros(vlm_inputs["input_ids"].shape[0], 4, 32)

    monkeypatch.setattr(train_script, "VisionEncoder", _Vision)
    monkeypatch.setattr(train_script, "VLMEncoder", _VLM)

    model = train_script._build_servovla_model(cfg, device=torch.device("cpu"))

    assert next(model.policy_head.parameters()).dtype == torch.bfloat16


def test_validation_predictor_runs_online_encoders_under_amp_autocast(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path)
    cfg.training.amp_dtype = "bfloat16"

    autocast_active = {"value": False}

    class _Autocast:
        def __init__(self, *, device_type, dtype, enabled=True):
            self.enabled = bool(enabled)

        def __enter__(self):
            autocast_active["value"] = self.enabled

        def __exit__(self, exc_type, exc, tb):
            autocast_active["value"] = False

    monkeypatch.setattr(train_script.torch, "autocast", _Autocast)

    class _Vision(torch.nn.Module):
        def forward(self, pixel_values):
            assert autocast_active["value"]
            return torch.zeros(pixel_values.shape[0], 2, 64)

    class _VLM(torch.nn.Module):
        def forward(self, vlm_inputs):
            assert autocast_active["value"]
            return torch.zeros(vlm_inputs["input_ids"].shape[0], 4, 32)

    class _Solver:
        def sample(self, **kwargs):
            q_current = kwargs["q_current"]
            return torch.zeros(
                q_current.shape[0], 8, 6, dtype=q_current.dtype, device=q_current.device
            )

    model = train_script.ServoVLA(
        vision_encoder=_Vision(),
        vlm_encoder=_VLM(),
        policy_head=torch.nn.Linear(1, 1).to(dtype=torch.bfloat16),
        fm_solver=_Solver(),
        action_is_delta=False,
    )
    predict = train_script._build_predict_absolute_chunk_fn(model, cfg, seed=1, zero_noise=True)
    batch = {
        "q_current": torch.zeros(2, 6),
        "pixel_values": torch.zeros(2, 2, 3, 8, 8),
        "vlm_inputs": {
            "input_ids": torch.ones(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4, dtype=torch.long),
        },
        "c_sem_mask": torch.ones(2, 4, dtype=torch.bool),
        "frame_delay": torch.zeros(2),
    }

    pred = predict(batch)

    assert pred.shape == (2, 8, 6)


def test_resolve_action_dim_prefers_dataset_action_dim_over_policy_placeholder(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.dataset.action_dim = 7
    cfg.model.policy_head.action_dim = 6

    assert train_script._resolve_action_dim(cfg) == 7


def test_resolve_runtime_roots_preserves_action_mode_output_dir(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.output_dir = str(tmp_path / "current")

    train_script._resolve_runtime_roots(cfg, task_overrides=[])

    assert cfg.output_dir == str(tmp_path / "delta" / "current")
