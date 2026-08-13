from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from servovla.evaluation.action_metrics import (
    evaluate_action_prediction_batches,
    restore_absolute_actions,
)
from tests.conftest import ACTION_DIM, CHUNK_SIZE


def _make_batch(
    *, batch_size: int = 2, valid_steps: int = CHUNK_SIZE, action_is_delta: bool = True
):
    q_current = torch.arange(batch_size * ACTION_DIM, dtype=torch.float32).view(
        batch_size, ACTION_DIM
    )
    absolute_target = torch.full((batch_size, CHUNK_SIZE, ACTION_DIM), 10.0, dtype=torch.float32)
    if action_is_delta:
        stored_action = absolute_target - q_current.unsqueeze(1)
    else:
        stored_action = absolute_target.clone()

    loss_mask = torch.zeros(batch_size, CHUNK_SIZE, dtype=torch.float32)
    loss_mask[:, :valid_steps] = 1.0
    return {
        "action": stored_action,
        "q_current": q_current,
        "loss_mask": loss_mask,
    }, absolute_target


def test_restore_absolute_actions_recovers_ground_truth_for_delta_targets():
    batch, absolute_target = _make_batch(action_is_delta=True)
    restored = restore_absolute_actions(batch["action"], batch["q_current"], action_is_delta=True)
    assert torch.allclose(restored, absolute_target)


def test_restore_absolute_actions_uses_mapped_delta_state_indices_and_keeps_gripper_direct():
    q_current = torch.tensor([[0.4, 0.5, 0.6, 3.0, -1.0, 0.2, 0.03, -0.03]], dtype=torch.float32)
    absolute_target = torch.tensor([[[0.5, 0.4, 0.8, 3.1, -0.8, 0.1, -1.0]]], dtype=torch.float32)
    basis = torch.tensor([[[0.4, 0.5, 0.6, 3.0, -1.0, 0.2, 0.0]]], dtype=torch.float32)
    stored_action = absolute_target - basis

    restored = restore_absolute_actions(
        stored_action,
        q_current,
        action_is_delta=True,
        action_delta_state_indices=[0, 1, 2, 3, 4, 5, None],
    )

    assert torch.allclose(restored, absolute_target)
    assert torch.allclose(restored[..., 6], stored_action[..., 6])


def test_evaluate_action_prediction_batches_reports_zero_error_for_exact_predictions():
    batch, absolute_target = _make_batch(action_is_delta=True)

    def predict_fn(current_batch):
        return absolute_target.clone()

    summary = evaluate_action_prediction_batches([batch], predict_fn, action_is_delta=True)

    assert summary["num_batches"] == 1
    assert summary["num_valid_steps"] == batch["loss_mask"].sum().item()
    assert summary["all_steps"]["mae_mean"] == 0.0
    assert summary["first_step"]["mae_mean"] == 0.0


def test_evaluate_action_prediction_batches_respects_loss_mask_and_first_step_metrics():
    batch, absolute_target = _make_batch(valid_steps=3, action_is_delta=False)
    offset = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=torch.float32)

    def predict_fn(current_batch):
        return absolute_target + offset.view(1, 1, -1)

    summary = evaluate_action_prediction_batches([batch], predict_fn, action_is_delta=False)

    expected_mae = offset.abs().mean().item()
    expected_signed = offset.mean().item()

    assert summary["num_valid_steps"] == 2 * 3
    assert summary["num_valid_first_steps"] == 2
    assert summary["all_steps"]["mae_mean"] == expected_mae
    assert summary["first_step"]["mae_mean"] == expected_mae
    assert summary["all_steps"]["signed_mean"] == expected_signed


def test_evaluation_package_still_exports_rollout_helper_names():
    from servovla import evaluation

    assert "resolve_checkpoint_path" in evaluation.__all__
    assert "evaluate_libero_run" in evaluation.__all__


def test_importing_action_metrics_does_not_emit_sentencepiece_warnings():
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "default",
            "-c",
            'from servovla.evaluation.action_metrics import evaluate_action_prediction_batches; print("ok")',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"
    assert "SwigPyPacked" not in completed.stderr
    assert "SwigPyObject" not in completed.stderr
    assert "swigvarlink" not in completed.stderr
