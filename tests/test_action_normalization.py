from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from servovla.architectures.action_normalization import ActionNormalizer
from servovla.architectures.servo_vla import ServoVLA
from servovla.trainer.trainer_loop import TrainerLoop


class _RecordingSolver:
    def __init__(self, trajectory: torch.Tensor):
        self.trajectory = trajectory

    def sample(self, **_kwargs):
        return self.trajectory.clone()


class _RecordingPolicyModel(nn.Module):
    def __init__(self, normalizer: ActionNormalizer):
        super().__init__()
        self.action_normalizer = normalizer
        self.policy_head = nn.Linear(1, 1)
        self.recorded_x_t = None

    def forward_policy(
        self,
        *,
        x_t,
        t,
        f_vision,
        c_sem,
        c_sem_mask,
        frame_delay,
        q_current,
    ):
        del t, f_vision, c_sem, c_sem_mask, frame_delay, q_current
        self.recorded_x_t = x_t.detach().clone()
        return x_t * self.policy_head.weight.reshape(1, 1, 1)


def test_action_normalizer_standardizes_and_restores_action_space():
    normalizer = ActionNormalizer(
        enabled=True,
        mean=torch.tensor([10.0, -2.0]),
        std=torch.tensor([2.0, 4.0]),
    )
    raw = torch.tensor([[[12.0, 2.0], [8.0, -6.0]]])

    normalized = normalizer.normalize(raw)

    assert torch.allclose(normalized, torch.tensor([[[1.0, 1.0], [-1.0, -1.0]]]))
    assert torch.allclose(normalizer.denormalize(normalized), raw)


def test_trainer_policy_core_trains_in_normalized_action_space(monkeypatch):
    normalizer = ActionNormalizer(
        enabled=True,
        mean=torch.tensor([10.0, -2.0]),
        std=torch.tensor([2.0, 4.0]),
    )
    model = _RecordingPolicyModel(normalizer)
    train_cfg = OmegaConf.create(
        {
            "device": "cpu",
            "amp_dtype": "float32",
            "compile": False,
            "output_dir": ".pytest_action_normalization",
            "grad_clip_norm": 1.0,
            "ema": {},
            "optimizer": {"lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.99], "eps": 1e-8},
            "scheduler": {"warmup_steps": 0, "T_max": 1, "eta_min": 1e-6},
        }
    )
    trainer = TrainerLoop(
        model=model,
        train_cfg=train_cfg,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    monkeypatch.setattr(torch, "randn_like", lambda value: torch.zeros_like(value))
    monkeypatch.setattr(
        torch,
        "rand",
        lambda shape, device=None, dtype=None: torch.ones(
            shape, device=device, dtype=dtype or torch.float32
        ),
    )

    trainer._train_step_policy_core(
        x_1=torch.tensor([[[12.0, 2.0]]]),
        loss_mask=torch.ones(1, 1),
        f_vision=torch.zeros(1, 1, 1),
        c_sem=torch.zeros(1, 1, 1),
        c_sem_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_delay=torch.zeros(1),
        q_current=torch.zeros(1, 2),
        dataset_slug=["unit"],
        optimizer=optimizer,
        defer_logging_tensors=False,
    )

    assert torch.allclose(model.recorded_x_t, torch.tensor([[[1.0, 1.0]]]))


def test_sample_action_chunk_from_features_denormalizes_before_delta_restore():
    normalizer = ActionNormalizer(
        enabled=True,
        mean=torch.tensor([10.0, -2.0]),
        std=torch.tensor([2.0, 4.0]),
    )
    normalized_trajectory = torch.tensor([[[1.0, -1.0], [0.0, 0.5]]])
    model = ServoVLA(
        vision_encoder=None,
        vlm_encoder=None,
        policy_head=nn.Linear(1, 1),
        fm_solver=_RecordingSolver(normalized_trajectory),
        action_is_delta=True,
        action_normalizer=normalizer,
    )
    q_current = torch.tensor([[100.0, 200.0]])

    out = model.sample_action_chunk_from_features(
        f_vision=torch.zeros(1, 1, 1),
        c_sem=torch.zeros(1, 1, 1),
        c_sem_mask=torch.ones(1, 1, dtype=torch.bool),
        frame_delay=torch.zeros(1),
        q_current=q_current,
    )

    expected_delta = torch.tensor([[[12.0, -6.0], [10.0, 0.0]]])
    assert torch.allclose(out, expected_delta + q_current.unsqueeze(1))
