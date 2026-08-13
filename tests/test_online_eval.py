import pytest
import torch
import torch.nn as nn

import servovla.evaluation.online_eval as online_eval
from servovla.architectures.fm_solver import FlowMatchingEulerSolver
from servovla.architectures.servo_vla import ServoVLA
from servovla.evaluation.online_eval import (
    evaluate_loss_on_batches,
    evaluate_model_on_batches,
    evaluate_online_validation_batches,
)


class _DeltaModel:
    def predict_chunk(self, batch):
        return batch["action"] + batch["q_current"].unsqueeze(1)


class _AbsModel:
    def predict_chunk(self, batch):
        return batch["action"]


class _FrameDelayField(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, x_t, t, pixel_values, vlm_inputs, c_sem_mask, frame_delay, q_current):
        del t, pixel_values, vlm_inputs, c_sem_mask, q_current
        return x_t + frame_delay.view(-1, 1, 1)


def _make_loss_batch(*, dataset_slug, frame_delay):
    batch_size = len(frame_delay)
    return {
        "action": torch.ones(batch_size, 1, 1),
        "loss_mask": torch.ones(batch_size, 1),
        "pixel_values": torch.zeros(batch_size, 1, 3, 4, 4),
        "vlm_inputs": {
            "input_ids": torch.ones(batch_size, 1, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 1, dtype=torch.long),
        },
        "c_sem_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "frame_delay": torch.tensor(frame_delay, dtype=torch.float32),
        "q_current": torch.zeros(batch_size, 1),
        "dataset_slug": dataset_slug,
    }


def test_evaluate_model_on_batches_restores_delta_predictions_to_absolute():
    q_current = torch.tensor([[1.0, 2.0, 3.0]])
    batch = {
        "action": torch.tensor([[[4.0, 5.0, 6.0]]]),
        "q_current": q_current,
        "loss_mask": torch.ones(1, 1),
    }

    summary = evaluate_model_on_batches([batch], _DeltaModel().predict_chunk, action_is_delta=True)
    assert summary["first_step"]["mae_mean"] == 0.0


def test_evaluate_model_on_batches_uses_mapped_delta_state_indices_for_libero_gripper():
    q_current = torch.tensor([[0.4, 0.5, 0.6, 3.0, -1.0, 0.2, 0.03, -0.03]])
    delta_target = torch.tensor([[[0.1, -0.1, 0.2, 0.01, 0.02, -0.03, -1.0]]])
    absolute_target = torch.tensor([[[0.5, 0.4, 0.8, 3.01, -0.98, 0.17, -1.0]]])
    batch = {
        "action": delta_target,
        "q_current": q_current,
        "loss_mask": torch.ones(1, 1),
    }

    summary = evaluate_model_on_batches(
        [batch],
        lambda _batch: absolute_target.clone(),
        action_is_delta=True,
        action_delta_state_indices=[0, 1, 2, 3, 4, 5, None],
    )

    assert summary["first_step"]["mae_mean"] == 0.0


def test_evaluate_model_on_batches_handles_absolute_targets_directly():
    batch = {
        "action": torch.tensor([[[4.0, 5.0, 6.0]]]),
        "q_current": torch.zeros(1, 3),
        "loss_mask": torch.ones(1, 1),
    }

    summary = evaluate_model_on_batches([batch], _AbsModel().predict_chunk, action_is_delta=False)
    assert summary["all_steps"]["mae_mean"] == 0.0


def test_evaluate_loss_on_raw_end2end_batches():
    from tests.test_end2end_training import _batch, _model

    model = _model()
    summary = evaluate_loss_on_batches(
        model=model,
        batches=[_batch(batch_size=3)],
        device="cpu",
        seed=123,
    )

    assert summary["num_batches"] == 1
    assert summary["fm_loss_mean"] >= 0.0
    assert summary["fm_loss_by_dataset"]["unit"] >= 0.0


def test_evaluate_loss_on_batches_uses_requested_autocast(monkeypatch):
    autocast_active = {"value": False}

    class _Autocast:
        def __init__(self, *, device_type, dtype, enabled=True):
            self.enabled = bool(enabled)

        def __enter__(self):
            autocast_active["value"] = self.enabled

        def __exit__(self, exc_type, exc, tb):
            autocast_active["value"] = False

    class _AmpRequiredField(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))

        def forward(self, x_t, t, pixel_values, vlm_inputs, c_sem_mask, frame_delay, q_current):
            del t, pixel_values, vlm_inputs, c_sem_mask, frame_delay, q_current
            assert autocast_active["value"]
            return torch.zeros_like(x_t)

    monkeypatch.setattr(online_eval.torch, "autocast", _Autocast)

    summary = evaluate_loss_on_batches(
        _AmpRequiredField(),
        [_make_loss_batch(dataset_slug="amp", frame_delay=[0.0])],
        device="cpu",
        seed=0,
        amp_dtype=torch.bfloat16,
    )

    assert summary["num_batches"] == 1


def test_evaluate_online_validation_batches_uses_raw_predict_function():
    from tests.test_end2end_training import _batch, _model

    batch = _batch(batch_size=3)
    model = _model()

    def predict_absolute_chunk(current_batch):
        return current_batch["action"] + current_batch["q_current"][:, None, :2]

    summary = evaluate_online_validation_batches(
        model=model,
        batches=[batch],
        device="cpu",
        action_is_delta=True,
        predict_absolute_chunk_fn=predict_absolute_chunk,
        seed=123,
    )

    assert summary["first_step"]["mae_mean"] == 0.0
    assert summary["all_steps"]["mae_mean"] == 0.0
    assert summary["num_loss_batches"] == 1


def test_evaluate_loss_on_batches_reports_fm_loss_without_episode_summary(monkeypatch):
    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    batches = [
        _make_loss_batch(dataset_slug="smoke_a", frame_delay=[0.0, 0.0]),
        _make_loss_batch(dataset_slug="smoke_a", frame_delay=[2.0]),
    ]

    summary = evaluate_loss_on_batches(
        _FrameDelayField(), batches, device="cpu", max_batches=8, seed=0
    )

    assert summary["fm_loss_mean"] == pytest.approx(4.0 / 3.0)
    assert "episode" not in summary


def test_evaluate_loss_on_batches_does_not_report_duplicate_mae_loss(monkeypatch):
    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    batches = [
        _make_loss_batch(dataset_slug="smoke_a", frame_delay=[0.0, 0.0]),
        _make_loss_batch(dataset_slug="smoke_a", frame_delay=[2.0]),
    ]

    summary = evaluate_loss_on_batches(
        _FrameDelayField(), batches, device="cpu", max_batches=8, seed=0
    )

    assert summary["fm_loss_mean"] == pytest.approx(4.0 / 3.0)
    assert "mae_loss_mean" not in summary


def test_evaluate_loss_on_batches_accepts_per_sample_dataset_slug_lists(monkeypatch):
    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    batch = _make_loss_batch(dataset_slug=["smoke_a", "smoke_b"], frame_delay=[1.0, 2.0])

    summary = evaluate_loss_on_batches(
        _FrameDelayField(), [batch], device="cpu", max_batches=8, seed=0
    )

    assert summary["fm_loss_by_dataset"]["smoke_a"] == pytest.approx(1.0)
    assert summary["fm_loss_by_dataset"]["smoke_b"] == pytest.approx(4.0)


def test_evaluate_online_validation_batches_keeps_global_sections_only(monkeypatch):
    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    batch = _make_loss_batch(dataset_slug="smoke_a", frame_delay=[0.0])

    summary = evaluate_online_validation_batches(
        model=_FrameDelayField(),
        batches=[batch],
        device="cpu",
        action_is_delta=False,
        predict_absolute_chunk_fn=lambda current_batch: current_batch["action"],
        max_batches=4,
        seed=0,
    )

    assert summary["all_steps"]["mae_mean"] == 0.0
    assert summary["action_loss_mean"] == 0.0
    assert summary["fm_loss_mean"] == 0.0
    assert "mae_loss_mean" not in summary
    assert "episode" not in summary


def test_evaluate_online_validation_batches_consumes_batches_once_for_both_metrics(monkeypatch):
    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    class _SinglePassBatches:
        def __init__(self, batch):
            self.batch = batch
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            if self.iterations > 1:
                raise AssertionError("validation batches were iterated more than once")
            yield self.batch

    batch = _make_loss_batch(dataset_slug="smoke_a", frame_delay=[0.0])
    batches = _SinglePassBatches(batch)

    summary = evaluate_online_validation_batches(
        model=_FrameDelayField(),
        batches=batches,
        device="cpu",
        action_is_delta=False,
        predict_absolute_chunk_fn=lambda current_batch: current_batch["action"],
        max_batches=4,
        seed=0,
    )

    assert batches.iterations == 1
    assert summary["action_loss_mean"] == 0.0
    assert summary["fm_loss_mean"] == 0.0


def test_evaluate_online_validation_batches_reuses_encoded_features_for_both_metrics(monkeypatch):
    class _Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, pixel_values):
            self.calls += 1
            return pixel_values[:, :, 0]

    class _VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))
            self.calls = 0

        def forward(self, vlm_inputs):
            self.calls += 1
            input_ids = vlm_inputs["input_ids"]
            return torch.zeros(input_ids.shape[0], input_ids.shape[1], 1)

    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(()))

        def forward(self, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
            del t, f_vision, c_sem, c_sem_mask, frame_delay, q_current
            return torch.zeros_like(x_t)

    monkeypatch.setattr(
        online_eval.torch,
        "randn",
        lambda *args, **kwargs: torch.zeros(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )
    monkeypatch.setattr(
        online_eval.torch,
        "rand",
        lambda *args, **kwargs: torch.ones(*args, device=kwargs["device"], dtype=kwargs["dtype"]),
    )

    vision = _Vision()
    vlm = _VLM()
    model = ServoVLA(
        vision_encoder=vision,
        vlm_encoder=vlm,
        policy_head=_Policy(),
        fm_solver=FlowMatchingEulerSolver(action_dim=1, chunk_size=1, num_inference_steps=1),
        action_is_delta=False,
    )
    batch = _make_loss_batch(dataset_slug="smoke_a", frame_delay=[0.0])

    def predict_absolute_chunk(current_batch):
        raise AssertionError("combined eval should use the feature-level predictor")

    def from_features(current_batch, *, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
        del f_vision, c_sem, c_sem_mask, frame_delay, q_current
        return current_batch["action"]

    predict_absolute_chunk.from_features = from_features

    summary = evaluate_online_validation_batches(
        model=model,
        batches=[batch],
        device="cpu",
        action_is_delta=False,
        predict_absolute_chunk_fn=predict_absolute_chunk,
        seed=0,
    )

    assert vision.calls == 1
    assert vlm.calls == 1
    assert summary["action_loss_mean"] == 0.0
    assert summary["fm_loss_mean"] == 1.0
