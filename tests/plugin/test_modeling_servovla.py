from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from lerobot.utils.constants import ACTION

from lerobot_policy_servovla.modeling_servovla import ServoVLAPolicy
from lerobot_policy_servovla.runtime_semantic_delay import SemanticSnapshot
from servovla.architectures.inference_noise import make_inference_noise


class _RecordingVisionProcessor:
    def __init__(self):
        self.calls = []

    def __call__(self, *, images, return_tensors: str, size):
        self.calls.append({"size": size, "image_count": len(images)})
        return {"pixel_values": torch.zeros(len(images), 3, size["height"], size["width"])}


class _RecordingVLMProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
        return messages[0]["content"][-1]["text"]

    def __call__(self, *, text, images, padding, return_tensors, truncation=None, max_length=None):
        self.calls.append(
            {
                "images": images,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            }
        )
        batch = len(text)
        return {
            "input_ids": torch.ones(batch, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch, 4, dtype=torch.long),
        }


def _make_policy():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(
        camera_keys=["observation.images.front", "observation.images.wrist"],
        front_camera_key="observation.images.front",
        wrist_camera_key="observation.images.wrist",
        vision_image_size=128,
        vlm_image_size=96,
        tokenizer_padding="longest",
        tokenizer_max_length=32,
        state_dim=6,
    )
    policy.vision_processor = _RecordingVisionProcessor()
    policy.vlm_processor = _RecordingVLMProcessor()
    policy._select_device = lambda: torch.device("cpu")
    return policy


def test_build_pixel_values_passes_explicit_vision_size_to_processor():
    policy = _make_policy()
    batch = {
        "observation.images.front": torch.zeros(1, 3, 24, 24),
        "observation.images.wrist": torch.zeros(1, 3, 24, 24),
    }

    pixel_values = policy._build_pixel_values(batch)

    assert pixel_values.shape == (1, 2, 3, 128, 128)
    assert policy.vision_processor.calls[0]["size"] == {"height": 128, "width": 128}
    assert policy.vision_processor.calls[1]["size"] == {"height": 128, "width": 128}


def test_build_vlm_inputs_resizes_images_with_vlm_image_size():
    policy = _make_policy()
    batch = {
        "observation.images.front": torch.zeros(1, 3, 24, 24),
        "observation.images.wrist": torch.zeros(1, 3, 24, 24),
        "task": ["pick"],
    }

    _, c_sem_mask, task_texts = policy._build_vlm_inputs(batch)

    assert c_sem_mask.shape == (1, 4)
    assert task_texts == ("pick",)
    first_call = policy.vlm_processor.calls[0]
    assert first_call["images"][0][0].size == (96, 96)
    assert first_call["images"][0][1].size == (96, 96)


def test_policy_uses_identical_order_for_vision_and_vlm_camera_inputs():
    class _OrderVisionProcessor:
        def __init__(self):
            self.values = []

        def __call__(self, *, images, return_tensors: str, size):
            del return_tensors, size
            value = int(images[0].getpixel((0, 0))[0])
            self.values.append(value)
            return {"pixel_values": torch.full((len(images), 3, 2, 2), float(value))}

    class _OrderVLMProcessor(_RecordingVLMProcessor):
        def __call__(
            self, *, text, images, padding, return_tensors, truncation=None, max_length=None
        ):
            self.calls.append(
                {
                    "values": [int(image.getpixel((0, 0))[0]) for image in images[0]],
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                }
            )
            return {
                "input_ids": torch.ones(len(text), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(text), 4, dtype=torch.long),
            }

    policy = _make_policy()
    policy.config.camera_keys = [
        "observation.images.wrist",
        "observation.images.side",
        "observation.images.front",
    ]
    policy.config.state_dim = 6
    policy.vision_processor = _OrderVisionProcessor()
    policy.vlm_processor = _OrderVLMProcessor()
    batch = {
        "observation.images.front": torch.full((1, 3, 2, 2), 10.0 / 255.0),
        "observation.images.wrist": torch.full((1, 3, 2, 2), 20.0 / 255.0),
        "observation.images.side": torch.full((1, 3, 2, 2), 30.0 / 255.0),
        "task": ["pick"],
    }

    pixel_values = policy._build_pixel_values(batch)
    policy._build_vlm_inputs(batch)

    assert pixel_values[:, 0].mean().item() == pytest.approx(20.0)
    assert pixel_values[:, 1].mean().item() == pytest.approx(30.0)
    assert pixel_values[:, 2].mean().item() == pytest.approx(10.0)
    assert policy.vision_processor.values == [20, 30, 10]
    assert policy.vlm_processor.calls[0]["values"] == [20, 30, 10]


def test_build_vlm_inputs_does_not_truncate_expanded_image_tokens():
    class _QwenLikeVLMProcessor:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool) -> str:
            del messages, tokenize, add_generation_prompt
            return "<image>" * 512 + " pick"

        def __call__(
            self,
            *,
            text,
            images,
            padding,
            return_tensors,
            truncation=None,
            max_length=None,
        ):
            self.calls.append({"truncation": truncation, "max_length": max_length})
            if truncation and max_length is not None:
                raise ValueError(
                    "Mismatch in `image` token count between text and `input_ids`. "
                    "Got ids=[506] and text=[512]."
                )
            batch = len(text)
            return {
                "input_ids": torch.ones(batch, 518, dtype=torch.long),
                "attention_mask": torch.ones(batch, 518, dtype=torch.long),
            }

    policy = _make_policy()
    policy.vlm_processor = _QwenLikeVLMProcessor()
    batch = {
        "observation.images.front": torch.zeros(1, 3, 24, 24),
        "observation.images.wrist": torch.zeros(1, 3, 24, 24),
        "task": ["pick"],
    }

    vlm_inputs, c_sem_mask, _ = policy._build_vlm_inputs(batch)

    assert vlm_inputs["input_ids"].shape == (1, 518)
    assert c_sem_mask.shape == (1, 518)
    assert policy.vlm_processor.calls == [{"truncation": None, "max_length": None}]


def test_policy_state_dict_and_load_state_dict_are_policy_head_only():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    torch.nn.Module.__init__(policy)
    policy.model = torch.nn.Module()
    policy.model.vision_encoder = torch.nn.Linear(2, 2)
    policy.model.vlm_encoder = torch.nn.Linear(2, 2)
    policy.model.policy_head = torch.nn.Linear(2, 1)

    state = policy.state_dict()

    assert state
    assert all(key.startswith("model.policy_head.") for key in state)
    assert not any("vision_encoder" in key for key in state)
    assert not any("vlm_encoder" in key for key in state)

    original_vision_weight = policy.model.vision_encoder.weight.detach().clone()
    original_vlm_weight = policy.model.vlm_encoder.weight.detach().clone()
    policy_weight = torch.full_like(policy.model.policy_head.weight, 0.75)
    policy_bias = torch.full_like(policy.model.policy_head.bias, -0.125)

    policy.load_state_dict(
        {
            "model.vision_encoder.weight": torch.full_like(policy.model.vision_encoder.weight, 9.0),
            "model.vlm_encoder.weight": torch.full_like(policy.model.vlm_encoder.weight, 8.0),
            "model.policy_head.weight": policy_weight,
            "model.policy_head.bias": policy_bias,
        }
    )

    assert torch.equal(policy.model.vision_encoder.weight, original_vision_weight)
    assert torch.equal(policy.model.vlm_encoder.weight, original_vlm_weight)
    assert torch.allclose(policy.model.policy_head.weight, policy_weight)
    assert torch.allclose(policy.model.policy_head.bias, policy_bias)


class _RecordingForwardModel:
    def __init__(self):
        self.frame_delay = None

    def __call__(
        self,
        *,
        x_t,
        t,
        pixel_values,
        vlm_inputs,
        c_sem_mask,
        frame_delay,
        q_current,
    ):
        self.frame_delay = frame_delay.detach().clone()
        return torch.zeros_like(x_t)


def test_forward_passes_sampled_frame_delay_to_model():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(action_dim=2, action_is_delta=False)
    policy.model = _RecordingForwardModel()
    policy._select_device = lambda: torch.device("cpu")
    q_current = torch.zeros(2, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(2, 2, 3, 8, 8),
            {"input_ids": torch.ones(2, 4, dtype=torch.long)},
            torch.ones(2, 4, dtype=torch.bool),
            q_current,
            ("pick", "place"),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)
    batch = {
        ACTION: torch.zeros(2, 3, 2),
        "frame_delay": torch.tensor([1.0, 3.0]),
    }

    policy.forward(batch)

    assert torch.equal(policy.model.frame_delay, torch.tensor([1.0, 3.0]))


def test_encode_semantic_tokens_uses_shared_vlm_forward_helper(monkeypatch):
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    vlm_encoder = object()
    policy.model = SimpleNamespace(vlm_encoder=vlm_encoder)
    observed = {}

    def _run_vlm_encoder_forward(observed_encoder, observed_inputs):
        observed["encoder"] = observed_encoder
        observed["inputs"] = observed_inputs
        return torch.ones(1, 2, 3)

    monkeypatch.setattr(
        "lerobot_policy_servovla.modeling_servovla.run_vlm_encoder_forward",
        _run_vlm_encoder_forward,
        raising=False,
    )
    vlm_inputs = {"input_ids": torch.ones(1, 2, dtype=torch.long)}

    c_sem = policy._encode_semantic_tokens(vlm_inputs)

    assert torch.equal(c_sem, torch.ones(1, 2, 3))
    assert observed == {"encoder": vlm_encoder, "inputs": vlm_inputs}


class _SynchronousRuntime:
    def __init__(self, *, max_frame_delay: int):
        self.max_frame_delay = max_frame_delay
        self.submitted_frame_ids = []
        self._snapshot = None

    def peek_snapshot(self):
        return self._snapshot

    def submit_latest(self, *, frame_id, task_texts, session_id="default", vlm_inputs, c_sem_mask):
        self.submitted_frame_ids.append(int(frame_id))
        self._snapshot = SemanticSnapshot(
            frame_id=int(frame_id),
            task_texts=tuple(task_texts),
            session_id=str(session_id),
            c_sem=torch.ones(1, 4, 3),
            c_sem_mask=c_sem_mask,
        )

    def has_request_for_frame_delay(
        self, *, frame_id, task_texts, session_id="default", delay_is_supported=None
    ):
        del frame_id, task_texts, session_id, delay_is_supported
        return False

    def wait_for_snapshot(
        self,
        *,
        frame_id,
        task_texts,
        session_id="default",
        admission_policy="bsr",
        delay_is_supported=None,
    ):
        del admission_policy
        assert self._snapshot is not None
        frame_delay = int(frame_id) - int(self._snapshot.frame_id)
        assert frame_delay <= self.max_frame_delay
        if delay_is_supported is not None:
            assert delay_is_supported(frame_delay)
        assert self._snapshot.task_texts == tuple(task_texts)
        assert self._snapshot.session_id == str(session_id)
        return self._snapshot, frame_delay, 0

    def get_last_admission_debug(self):
        return {}


class _DeferredRefreshRuntime(_SynchronousRuntime):
    def __init__(self, *, max_frame_delay: int):
        super().__init__(max_frame_delay=max_frame_delay)
        self._active_snapshot = None
        self._pending_snapshot = None

    def submit_latest(self, *, frame_id, task_texts, session_id="default", vlm_inputs, c_sem_mask):
        self.submitted_frame_ids.append(int(frame_id))
        snapshot = SemanticSnapshot(
            frame_id=int(frame_id),
            task_texts=tuple(task_texts),
            session_id=str(session_id),
            c_sem=torch.ones(1, 4, 3),
            c_sem_mask=c_sem_mask,
        )
        if self._active_snapshot is None:
            self._active_snapshot = snapshot
        else:
            self._pending_snapshot = snapshot

    def has_request_for_frame_delay(
        self, *, frame_id, task_texts, session_id="default", delay_is_supported=None
    ):
        for snapshot in (self._active_snapshot, self._pending_snapshot):
            if (
                snapshot is None
                or snapshot.task_texts != tuple(task_texts)
                or snapshot.session_id != str(session_id)
            ):
                continue
            frame_delay = int(frame_id) - int(snapshot.frame_id)
            if 0 <= frame_delay <= self.max_frame_delay and (
                delay_is_supported is None or delay_is_supported(frame_delay)
            ):
                return True
        return False

    def wait_for_snapshot(
        self,
        *,
        frame_id,
        task_texts,
        session_id="default",
        admission_policy="bsr",
        delay_is_supported=None,
    ):
        del admission_policy
        if (
            self._snapshot is not None
            and self._snapshot.task_texts == tuple(task_texts)
            and self._snapshot.session_id == str(session_id)
        ):
            frame_delay = int(frame_id) - int(self._snapshot.frame_id)
            if 0 <= frame_delay <= self.max_frame_delay and (
                delay_is_supported is None or delay_is_supported(frame_delay)
            ):
                return self._snapshot, frame_delay, 0
        assert self._active_snapshot is not None
        self._snapshot = self._active_snapshot
        self._active_snapshot = self._pending_snapshot
        self._pending_snapshot = None
        frame_delay = int(frame_id) - int(self._snapshot.frame_id)
        assert 0 <= frame_delay <= self.max_frame_delay
        if delay_is_supported is not None:
            assert delay_is_supported(frame_delay)
        return self._snapshot, frame_delay, 1


class _PredictModel:
    def __init__(self):
        self.current_step = 0
        self.last_vlm_update_step = 0
        self.latest_vlm_feature = None
        self.frame_delay = None
        self.noise = None
        self.policy_head = torch.nn.Linear(1, 1)
        self.fm_solver = SimpleNamespace(chunk_size=1, action_dim=2)
        self.vision_encoder = lambda pixel_values: torch.zeros(pixel_values.shape[0], 4, 3)

    def sample_action_chunk_from_features(
        self, *, f_vision, c_sem, c_sem_mask, frame_delay, q_current, noise=None
    ):
        self.frame_delay = frame_delay.detach().cpu().clone()
        self.noise = None if noise is None else noise.detach().cpu().clone()
        return torch.zeros(q_current.shape[0], 1, q_current.shape[-1])


def test_predict_action_chunk_without_bucket_support_refreshes_at_runtime_delay_bound():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1, action_dim=2)
    policy.model = _PredictModel()
    policy._last_tasks = None
    runtime = _SynchronousRuntime(max_frame_delay=2)
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    for _ in range(4):
        policy.predict_action_chunk({})

    assert runtime.submitted_frame_ids == [0, 3]


def test_predict_action_chunk_uses_supplied_action_step_for_semantic_delay():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1, action_dim=2)
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _SynchronousRuntime(max_frame_delay=20)
    runtime._snapshot = SemanticSnapshot(
        frame_id=95,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=100)

    assert torch.equal(policy.model.frame_delay, torch.tensor([5.0]))
    debug = policy.get_last_inference_debug()
    assert debug["current_action_step"] == 100
    assert debug["semantic_snapshot_action_step"] == 95
    assert debug["step_delay"] == 5
    assert debug["frame_delay"] == 5
    assert debug["action_step_source"] == "runtime"
    assert debug["semantic_sample_mode"] == "async_stale"


def test_predict_action_chunk_refreshes_gap_delay_outside_bucket_support():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(
        n_action_steps=1,
        action_dim=2,
        chunk_size=16,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.5,
    )
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _DeferredRefreshRuntime(max_frame_delay=23)
    runtime._snapshot = SemanticSnapshot(
        frame_id=0,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=1)

    assert runtime.submitted_frame_ids == [1]
    assert torch.equal(policy.model.frame_delay, torch.tensor([0.0]))
    debug = policy.get_last_inference_debug()
    assert debug["current_action_step"] == 1
    assert debug["semantic_snapshot_action_step"] == 1
    assert debug["step_delay"] == 0
    assert debug["semantic_sample_mode"] == "sync_aligned"


def test_predict_action_chunk_waits_past_cached_gap_delay_after_refresh_submit():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(
        n_action_steps=1,
        action_dim=2,
        chunk_size=16,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.5,
    )
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _DeferredRefreshRuntime(max_frame_delay=23)
    runtime._snapshot = SemanticSnapshot(
        frame_id=0,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=1)

    assert runtime.submitted_frame_ids == [1]
    assert torch.equal(policy.model.frame_delay, torch.tensor([0.0]))
    debug = policy.get_last_inference_debug()
    assert debug["semantic_snapshot_action_step"] == 1
    assert debug["step_delay"] == 0


def test_predict_action_chunk_reuses_delay_inside_bucket_support():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(
        n_action_steps=1,
        action_dim=2,
        chunk_size=16,
        delay_max_chunks=3,
        delay_chunk_size_threshold=0.5,
        inference_noise_seed=123,
        inference_noise_seed_mode="step",
    )
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _DeferredRefreshRuntime(max_frame_delay=23)
    runtime._snapshot = SemanticSnapshot(
        frame_id=0,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=8)

    assert runtime.submitted_frame_ids == [8]
    assert torch.equal(policy.model.frame_delay, torch.tensor([8.0]))
    expected_noise = make_inference_noise(
        shape=(1, 1, 2),
        device=torch.device("cpu"),
        dtype=policy.model.policy_head.weight.dtype,
        seed=123,
        seed_mode="step",
        action_step=8,
    )
    assert policy.model.noise is not None
    assert policy.model.noise.shape == (1, 1, 2)
    assert policy.model.noise.dtype == expected_noise.dtype
    assert torch.equal(policy.model.noise, expected_noise)
    debug = policy.get_last_inference_debug()
    assert debug["current_action_step"] == 8
    assert debug["semantic_snapshot_action_step"] == 0
    assert debug["step_delay"] == 8
    assert debug["semantic_sample_mode"] == "async_stale"
    assert debug["vlm_latest_submitted"] is True
    assert debug["vlm_latest_frame_id"] == 8


def test_predict_action_chunk_uses_latest_completed_vlm_while_background_worker_tracks_newest_frame():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(
        n_action_steps=1,
        action_dim=2,
        chunk_size=16,
        delay_max_chunks=2,
        delay_chunk_size_threshold=0.2,
    )
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _DeferredRefreshRuntime(max_frame_delay=30)
    runtime._snapshot = SemanticSnapshot(
        frame_id=0,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=30)

    assert runtime.submitted_frame_ids == [30]
    assert torch.equal(policy.model.frame_delay, torch.tensor([30.0]))
    debug = policy.get_last_inference_debug()
    assert debug["semantic_snapshot_action_step"] == 0
    assert debug["vlm_prefetch_submitted"] is True
    assert debug["vlm_prefetch_frame_id"] == 30

    policy.predict_action_chunk({}, action_step=45)

    assert runtime.submitted_frame_ids == [30, 45]
    assert torch.equal(policy.model.frame_delay, torch.tensor([15.0]))
    debug = policy.get_last_inference_debug()
    assert debug["semantic_snapshot_action_step"] == 30
    assert debug["frame_delay"] == 15
    assert debug["pending_refresh_will_be_supported"] is True
    assert debug["vlm_latest_submitted"] is True
    assert debug["vlm_latest_frame_id"] == 45


def test_predict_action_chunk_refreshes_when_action_step_goes_backwards_for_same_task():
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1, action_dim=2)
    policy.model = _PredictModel()
    policy._last_tasks = ("pick",)
    runtime = _SynchronousRuntime(max_frame_delay=100)
    runtime._snapshot = SemanticSnapshot(
        frame_id=95,
        task_texts=("pick",),
        session_id="default",
        c_sem=torch.ones(1, 4, 3),
        c_sem_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    policy.predict_action_chunk({}, action_step=10)

    assert runtime.submitted_frame_ids == [10]
    assert torch.equal(policy.model.frame_delay, torch.tensor([0.0]))
    debug = policy.get_last_inference_debug()
    assert debug["current_action_step"] == 10
    assert debug["semantic_snapshot_action_step"] == 10
    assert debug["step_delay"] == 0
    assert debug["semantic_sample_mode"] == "sync_aligned"


@pytest.mark.parametrize(
    ("max_frame_delay", "num_calls", "expected_submitted_frame_ids"),
    [
        (0, 2, [0, 1]),
        (1, 3, [0, 2]),
    ],
)
def test_predict_action_chunk_refreshes_before_waiting_when_delay_bound_would_be_exceeded(
    max_frame_delay,
    num_calls,
    expected_submitted_frame_ids,
):
    policy = ServoVLAPolicy.__new__(ServoVLAPolicy)
    policy.config = SimpleNamespace(n_action_steps=1, action_dim=2)
    policy.model = _PredictModel()
    policy._last_tasks = None
    runtime = _SynchronousRuntime(max_frame_delay=max_frame_delay)
    policy._semantic_runtime_or_raise = lambda: runtime
    policy.eval = lambda: None
    q_current = torch.zeros(1, 2)

    def _prepare_batch_inputs(self, batch):
        return (
            torch.zeros(1, 2, 3, 8, 8),
            {"input_ids": torch.ones(1, 4, dtype=torch.long)},
            torch.ones(1, 4, dtype=torch.bool),
            q_current,
            ("pick",),
            "default",
        )

    policy._prepare_batch_inputs = MethodType(_prepare_batch_inputs, policy)

    for _ in range(num_calls):
        policy.predict_action_chunk({})

    assert runtime.submitted_frame_ids == expected_submitted_frame_ids
