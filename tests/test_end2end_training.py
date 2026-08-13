from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from servovla.architectures.fm_solver import FlowMatchingEulerSolver
from servovla.architectures.servo_vla import ServoVLA
from servovla.trainer.trainer_loop import TrainerLoop

ACTION_DIM = 2
STATE_DIM = 3
CHUNK_SIZE = 4
VISION_DIM = 5
SEM_DIM = 6


class _FrozenVisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        per_view = pixel_values.mean(dim=(-1, -2))
        flat = per_view.reshape(batch_size, -1)
        features = flat[:, :VISION_DIM]
        if features.shape[1] < VISION_DIM:
            features = torch.nn.functional.pad(features, (0, VISION_DIM - features.shape[1]))
        return features.unsqueeze(1) * self.scale


class _FrozenVLMEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, vlm_inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = vlm_inputs["input_ids"].float()
        batch_size, seq_len = input_ids.shape
        base = input_ids.unsqueeze(-1).expand(batch_size, seq_len, SEM_DIM)
        return base * self.scale / 100.0


class _TinyPolicyHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        in_dim = VISION_DIM + SEM_DIM + STATE_DIM + 2
        self.linear = nn.Linear(in_dim, CHUNK_SIZE * ACTION_DIM)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        f_vision: torch.Tensor,
        c_sem: torch.Tensor,
        c_sem_mask: torch.Tensor,
        frame_delay: torch.Tensor,
        q_current: torch.Tensor,
    ) -> torch.Tensor:
        del x_t
        sem_mask = c_sem_mask.float().unsqueeze(-1)
        sem = (c_sem * sem_mask).sum(dim=1) / sem_mask.sum(dim=1).clamp(min=1.0)
        pooled = torch.cat(
            [
                f_vision.mean(dim=1),
                sem,
                q_current,
                frame_delay.unsqueeze(-1).float(),
                t.unsqueeze(-1).float(),
            ],
            dim=-1,
        )
        return self.linear(pooled).view(-1, CHUNK_SIZE, ACTION_DIM)


def _train_cfg(tmp_path):
    return SimpleNamespace(
        device="cpu",
        amp_dtype="float32",
        compile=False,
        optimizer=SimpleNamespace(
            lr=1e-3,
            weight_decay=0.0,
            betas=[0.9, 0.999],
            eps=1e-8,
        ),
        scheduler=SimpleNamespace(T_max=8, eta_min=1e-6, warmup_steps=0),
        ema=SimpleNamespace(decay=0.9, update_after_step=0, update_every=1),
        async_eval=SimpleNamespace(enabled=False),
        output_dir=tmp_path,
        grad_clip_norm=1.0,
        log_every=50,
        save_every=99,
        eval_every=0,
        eval_use_ema=True,
    )


def _batch(batch_size: int = 3) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | list[str]]:
    return {
        "action": torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM),
        "loss_mask": torch.ones(batch_size, CHUNK_SIZE),
        "pixel_values": torch.randn(batch_size, 2, 3, 8, 8),
        "vlm_inputs": {
            "input_ids": torch.arange(batch_size * 7).view(batch_size, 7),
            "attention_mask": torch.ones(batch_size, 7, dtype=torch.long),
        },
        "c_sem_mask": torch.ones(batch_size, 7, dtype=torch.bool),
        "frame_delay": torch.arange(batch_size, dtype=torch.float32),
        "q_current": torch.randn(batch_size, STATE_DIM),
        "dataset_slug": ["unit"] * batch_size,
    }


def _model() -> ServoVLA:
    policy_head = _TinyPolicyHead()
    return ServoVLA(
        vision_encoder=_FrozenVisionEncoder(),
        vlm_encoder=_FrozenVLMEncoder(),
        policy_head=policy_head,
        fm_solver=FlowMatchingEulerSolver(
            action_dim=ACTION_DIM,
            chunk_size=CHUNK_SIZE,
            num_inference_steps=2,
        ),
        action_is_delta=True,
    )


def test_train_step_end2end_updates_policy_head_but_not_frozen_encoders(tmp_path):
    model = _model()
    trainer = TrainerLoop(model=model, train_cfg=_train_cfg(tmp_path))
    optimizer = trainer.build_optimizer()
    policy_before = model.policy_head.linear.weight.detach().clone()
    vision_before = model.vision_encoder.scale.detach().clone()
    vlm_before = model.vlm_encoder.scale.detach().clone()

    result = trainer.train_step_end2end(_batch(), optimizer)

    assert result["dataset_count_by_slug"] == {"unit": 3}
    assert torch.isfinite(torch.as_tensor(result["main_loss"]))
    assert not torch.allclose(model.policy_head.linear.weight.detach(), policy_before)
    assert torch.equal(model.vision_encoder.scale.detach(), vision_before)
    assert torch.equal(model.vlm_encoder.scale.detach(), vlm_before)
    assert model.vision_encoder.scale.grad is None
    assert model.vlm_encoder.scale.grad is None


def test_train_step_from_features_preserves_dataset_metadata_and_updates_policy(tmp_path):
    model = _model()
    trainer = TrainerLoop(model=model, train_cfg=_train_cfg(tmp_path))
    optimizer = trainer.build_optimizer()
    batch = _batch(batch_size=2)

    with torch.no_grad():
        f_vision, c_sem = model.encode_observations(
            pixel_values=batch["pixel_values"],
            vlm_inputs=batch["vlm_inputs"],
        )

    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    encoded = EncodedFeatureBatch(
        f_vision=f_vision,
        c_sem=c_sem,
        c_sem_mask=batch["c_sem_mask"],
        action=batch["action"],
        loss_mask=batch["loss_mask"],
        q_current=batch["q_current"],
        frame_delay=batch["frame_delay"],
        dataset_slug=batch["dataset_slug"],
        timings={},
    )
    before = model.policy_head.linear.weight.detach().clone()

    result = trainer.train_step_from_features(encoded, optimizer)

    assert result["dataset_count_by_slug"] == {"unit": 2}
    assert torch.isfinite(torch.as_tensor(result["main_loss"]))
    assert not torch.allclose(model.policy_head.linear.weight.detach(), before)


def test_optimizer_only_contains_trainable_policy_head_params(tmp_path):
    model = _model()
    trainer = TrainerLoop(model=model, train_cfg=_train_cfg(tmp_path))
    optimizer = trainer.build_optimizer()

    optimized_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    assert id(model.policy_head.linear.weight) in optimized_ids
    assert id(model.policy_head.linear.bias) in optimized_ids
    assert id(model.vision_encoder.scale) not in optimized_ids
    assert id(model.vlm_encoder.scale) not in optimized_ids


def test_policy_head_only_checkpoint_excludes_training_recovery_state(tmp_path):
    model = _model()
    trainer = TrainerLoop(model=model, train_cfg=_train_cfg(tmp_path))
    optimizer = trainer.build_optimizer()
    trainer.train_step_end2end(_batch(), optimizer)
    ema = trainer.build_ema()
    ema.update(model)
    saved = trainer.save_checkpoint(step=1, ema_model=ema)

    payload = torch.load(saved, map_location="cpu", weights_only=False)
    assert payload["model_state"]
    assert all(key.startswith("policy_head.") for key in payload["model_state"])
    assert not any("vision_encoder" in key for key in payload["model_state"])
    assert not any("vlm_encoder" in key for key in payload["model_state"])
    assert payload["ema_state"]
    assert all(key.startswith("policy_head.") for key in payload["ema_state"])
    assert "decay" not in payload["ema_state"]
    assert "module" not in payload["ema_state"]
    assert set(payload) == {"step", "model_state", "ema_state"}


def test_policy_head_ema_shadow_and_checkpoint_are_fp32_for_bfloat16_policy_head(tmp_path):
    model = _model()
    model.policy_head.to(dtype=torch.bfloat16)
    trainer = TrainerLoop(model=model, train_cfg=_train_cfg(tmp_path))
    ema = trainer.build_ema()

    assert {param.dtype for param in ema.module.parameters()} == {torch.float32}

    ema.update(model)
    saved = trainer.save_checkpoint(step=1, ema_model=ema)
    payload = torch.load(saved, map_location="cpu", weights_only=False)

    assert {tensor.dtype for tensor in payload["model_state"].values()} == {torch.bfloat16}
    assert {tensor.dtype for tensor in payload["ema_state"].values()} == {torch.float32}


def test_servovla_state_dict_exposes_policy_head_only():
    model = _model()

    state = model.state_dict()

    assert state
    assert all(key.startswith("policy_head.") for key in state)
    assert not any(key.startswith("vision_encoder.") for key in state)
    assert not any(key.startswith("vlm_encoder.") for key in state)


def test_servovla_load_state_dict_ignores_encoder_weights():
    model = _model()
    original_vision_scale = model.vision_encoder.scale.detach().clone()
    original_vlm_scale = model.vlm_encoder.scale.detach().clone()
    policy_weight = torch.full_like(model.policy_head.linear.weight, 0.25)
    policy_bias = torch.full_like(model.policy_head.linear.bias, -0.5)

    model.load_state_dict(
        {
            "vision_encoder.scale": torch.tensor(99.0),
            "vlm_encoder.scale": torch.tensor(88.0),
            "policy_head.linear.weight": policy_weight,
            "policy_head.linear.bias": policy_bias,
        }
    )

    assert torch.equal(model.vision_encoder.scale, original_vision_scale)
    assert torch.equal(model.vlm_encoder.scale, original_vlm_scale)
    assert torch.allclose(model.policy_head.linear.weight, policy_weight)
    assert torch.allclose(model.policy_head.linear.bias, policy_bias)
