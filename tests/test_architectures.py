"""Tests for DiT projection layers, adaLN-Zero, and forward pass."""

import torch
import torch.nn as nn

import servovla.architectures.policy_head as policy_head_module
from servovla.architectures.policy_head import (
    DiTBlock,
    FlashCompatibleAttention,
    FlowMatchingDiT,
    ModulateAdaLNZero,
)
from servovla.architectures.servo_vla import ServoVLA
from tests.conftest import (
    ACTION_DIM,
    CHUNK_SIZE,
    D_SEM,
    D_VISION,
    HIDDEN_DIM,
    NUM_CAMERAS,
    NUM_HEADS,
    NUM_LAYERS,
    STATE_DIM,
    VISION_SEQ,
)

BATCH = 4


class TestModulateAdaLNZero:
    def test_zero_init(self):
        ada = ModulateAdaLNZero(HIDDEN_DIM)
        x_dummy = torch.randn(BATCH, CHUNK_SIZE, HIDDEN_DIM)
        c = torch.randn(BATCH, HIDDEN_DIM)
        outs = ada(x_dummy, c)

        assert len(outs) == 6
        for o in outs:
            assert o.abs().max().item() == 0.0


class TestFlowMatchingDiT:
    def _make_model(self):
        return FlowMatchingDiT(
            action_dim=ACTION_DIM,
            state_dim=STATE_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            num_heads=NUM_HEADS,
            vision_feature_dim=D_VISION,
            semantic_feature_dim=D_SEM,
            vision_grid_size=4,
            num_cameras=NUM_CAMERAS,
        )

    def _make_batch(self):
        x_t = torch.randn(BATCH, CHUNK_SIZE, ACTION_DIM)
        t = torch.rand(BATCH)
        f_vision = torch.randn(BATCH, VISION_SEQ, D_VISION)
        c_sem = torch.randn(BATCH, 12, D_SEM)
        c_sem_mask = torch.ones(BATCH, 12, dtype=torch.bool)
        frame_delay = torch.randint(0, 5, (BATCH,), dtype=torch.float32)
        q_current = torch.randn(BATCH, STATE_DIM)
        return x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current

    def test_forward_output_shape_and_finite(self):
        policy_head = self._make_model()
        x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current = self._make_batch()
        u = policy_head(x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current)
        assert u.shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
        assert torch.isfinite(u).all()

    def test_output_zero_init(self):
        policy_head = self._make_model()
        x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current = self._make_batch()
        u = policy_head(x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current)
        assert u.abs().max().item() == 0.0

    def test_context_projections_have_layernorm(self):
        policy_head = self._make_model()
        assert isinstance(policy_head.vision_proj, nn.Sequential)
        assert isinstance(policy_head.vision_proj[-1], nn.LayerNorm)
        assert isinstance(policy_head.sem_proj, nn.Sequential)
        assert isinstance(policy_head.sem_proj[-1], nn.LayerNorm)

    def test_policy_head_attention_no_longer_uses_torch_multihead_attention(self):
        policy_head = self._make_model()

        assert not any(
            isinstance(module, nn.MultiheadAttention) for module in policy_head.modules()
        )

    def test_forward_supports_bfloat16_policy_head(self):
        policy_head = self._make_model().to(dtype=torch.bfloat16)
        x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current = self._make_batch()

        u = policy_head(
            x_t.to(dtype=torch.bfloat16),
            t.to(dtype=torch.bfloat16),
            f_vision.to(dtype=torch.bfloat16),
            c_sem.to(dtype=torch.bfloat16),
            c_sem_mask,
            frame_delay.to(dtype=torch.bfloat16),
            q_current.to(dtype=torch.bfloat16),
        )

        assert u.shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
        assert u.dtype == torch.bfloat16
        assert torch.isfinite(u).all()


def test_dit_block_cpu_attention_fallback_handles_context_mask():
    block = DiTBlock(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.0)
    x = torch.randn(BATCH, CHUNK_SIZE, HIDDEN_DIM)
    c = torch.randn(BATCH, HIDDEN_DIM)
    context = torch.randn(BATCH, 17, HIDDEN_DIM)
    context_mask = torch.ones(BATCH, 17, dtype=torch.bool)
    context_mask[0, 3:7] = False

    out = block(x, c, context, context_mask)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_flash_varlen_cu_seqlens_are_built_without_integer_pad(monkeypatch):
    attn = FlashCompatibleAttention(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.0)
    q = torch.randn(2, 3, NUM_HEADS, HIDDEN_DIM // NUM_HEADS)
    k = torch.randn(2, 5, NUM_HEADS, HIDDEN_DIM // NUM_HEADS)
    v = torch.randn(2, 5, NUM_HEADS, HIDDEN_DIM // NUM_HEADS)
    key_padding_mask = torch.tensor(
        [
            [False, False, False, True, True],
            [False, False, False, False, True],
        ],
        dtype=torch.bool,
    )
    captured: dict[str, torch.Tensor] = {}

    def _fail_integer_pad(input_tensor, *args, **kwargs):
        if input_tensor.dtype in {torch.int32, torch.int64}:
            raise AssertionError("flash varlen cu_seqlens must not be built with F.pad")
        return original_pad(input_tensor, *args, **kwargs)

    def _fake_varlen(q_unpad, k_unpad, v_unpad, cu_q, cu_k, *args, **kwargs):
        captured["cu_q"] = cu_q
        captured["cu_k"] = cu_k
        return torch.zeros_like(q_unpad)

    original_pad = policy_head_module.F.pad
    monkeypatch.setattr(policy_head_module.F, "pad", _fail_integer_pad)
    monkeypatch.setattr(policy_head_module, "_flash_attn_varlen_func", _fake_varlen)

    out = attn._flash_attention(q, k, v, key_padding_mask)

    assert out.shape == q.shape
    assert captured["cu_q"].dtype == torch.int32
    assert captured["cu_k"].dtype == torch.int32
    assert captured["cu_q"].tolist() == [0, 3, 6]
    assert captured["cu_k"].tolist() == [0, 3, 7]


def test_flash_attention_can_use_bfloat16_kernel_with_float32_module(monkeypatch):
    attn = FlashCompatibleAttention(hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, dropout=0.0)
    attn.set_flash_attention_dtype(torch.bfloat16)
    query = torch.randn(2, 3, HIDDEN_DIM)
    captured = {}

    def _fake_can_use_flash(q):
        return True

    def _fake_flash(q, k, v, *args, **kwargs):
        captured["q_dtype"] = q.dtype
        captured["k_dtype"] = k.dtype
        captured["v_dtype"] = v.dtype
        return torch.zeros_like(q)

    monkeypatch.setattr(attn, "_can_use_flash", _fake_can_use_flash)
    monkeypatch.setattr(policy_head_module, "_flash_attn_func", _fake_flash)

    out = attn(query, query, query)

    assert captured == {
        "q_dtype": torch.bfloat16,
        "k_dtype": torch.bfloat16,
        "v_dtype": torch.bfloat16,
    }
    assert out.dtype == torch.float32
    assert next(attn.parameters()).dtype == torch.float32


def test_policy_head_pads_attention_sequences_to_multiple_of_eight_and_slices_output():
    class _CaptureBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_shape = None
            self.x_mask_shape = None
            self.context_shape = None
            self.context_mask_shape = None
            self.padded_x_mask_tail = None
            self.padded_mask_tail = None

        def forward(self, x, c, context, context_mask=None, x_mask=None):
            self.x_shape = tuple(x.shape)
            self.x_mask_shape = tuple(x_mask.shape)
            self.context_shape = tuple(context.shape)
            self.context_mask_shape = tuple(context_mask.shape)
            self.padded_x_mask_tail = x_mask[:, -6:].detach().clone()
            self.padded_mask_tail = context_mask[:, -1:].detach().clone()
            return x

    horizon = 10
    policy_head = FlowMatchingDiT(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=1,
        num_heads=NUM_HEADS,
        vision_feature_dim=D_VISION,
        semantic_feature_dim=D_SEM,
        vision_grid_size=4,
        num_cameras=NUM_CAMERAS,
    )
    capture_block = _CaptureBlock()
    policy_head.blocks = nn.ModuleList([capture_block])
    x_t = torch.randn(BATCH, horizon, ACTION_DIM)
    t = torch.rand(BATCH)
    f_vision = torch.randn(BATCH, VISION_SEQ + NUM_CAMERAS, D_VISION)
    c_sem = torch.randn(BATCH, 12, D_SEM)
    c_sem_mask = torch.ones(BATCH, 12, dtype=torch.bool)
    frame_delay = torch.zeros(BATCH)
    q_current = torch.randn(BATCH, STATE_DIM)

    out = policy_head(x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current)

    assert out.shape == (BATCH, horizon, ACTION_DIM)
    assert capture_block.x_shape[1] % 8 == 0
    assert capture_block.x_mask_shape == capture_block.x_shape[:2]
    assert capture_block.padded_x_mask_tail.eq(False).all()
    assert capture_block.context_shape[1] % 8 == 0
    assert capture_block.context_mask_shape == capture_block.context_shape[:2]
    assert capture_block.padded_mask_tail.eq(False).all()


class _RecordingSolver:
    def __init__(self, trajectory: torch.Tensor):
        self.trajectory = trajectory
        self.calls: list[dict[str, torch.Tensor]] = []

    def sample(self, **kwargs):
        self.calls.append(kwargs)
        return self.trajectory.clone()


def test_sample_action_chunk_from_features_passes_mask_and_frame_delay():
    solver = _RecordingSolver(torch.zeros(BATCH, CHUNK_SIZE, ACTION_DIM))
    model = ServoVLA(
        vision_encoder=None,
        vlm_encoder=None,
        policy_head=nn.Linear(1, 1),
        fm_solver=solver,
        action_is_delta=False,
    )

    out = model.sample_action_chunk_from_features(
        torch.randn(BATCH, VISION_SEQ, D_VISION),
        torch.randn(BATCH, 12, D_SEM),
        torch.ones(BATCH, 12, dtype=torch.bool),
        torch.full((BATCH,), 3.0),
        torch.randn(BATCH, STATE_DIM),
    )

    assert out.shape == (BATCH, CHUNK_SIZE, ACTION_DIM)
    assert torch.equal(solver.calls[0]["frame_delay"], torch.full((BATCH,), 3.0))
    assert torch.equal(solver.calls[0]["c_sem_mask"], torch.ones(BATCH, 12, dtype=torch.bool))


def test_sample_action_chunk_from_features_reconstructs_absolute_targets_in_delta_mode():
    solver = _RecordingSolver(torch.full((1, CHUNK_SIZE, ACTION_DIM), 0.25))
    model = ServoVLA(
        vision_encoder=None,
        vlm_encoder=None,
        policy_head=nn.Linear(1, 1),
        fm_solver=solver,
        action_is_delta=True,
    )
    q_current = torch.full((1, STATE_DIM), 1.0)

    out = model.sample_action_chunk_from_features(
        torch.randn(1, VISION_SEQ, D_VISION),
        torch.randn(1, 12, D_SEM),
        torch.ones(1, 12, dtype=torch.bool),
        torch.tensor([4.0]),
        q_current,
    )

    assert torch.allclose(out, solver.trajectory + q_current.unsqueeze(1))


def test_sample_action_chunk_from_features_restores_delta_with_state_wider_than_action():
    action_dim = 7
    state_dim = 8
    solver = _RecordingSolver(torch.full((1, CHUNK_SIZE, action_dim), 0.25))
    model = ServoVLA(
        vision_encoder=None,
        vlm_encoder=None,
        policy_head=nn.Linear(1, 1),
        fm_solver=solver,
        action_is_delta=True,
        action_delta_state_indices=[0, 1, 2, 3, 4, 5, None],
    )
    q_current = torch.arange(state_dim, dtype=torch.float32).unsqueeze(0)

    out = model.sample_action_chunk_from_features(
        torch.randn(1, VISION_SEQ, D_VISION),
        torch.randn(1, 12, D_SEM),
        torch.ones(1, 12, dtype=torch.bool),
        torch.tensor([4.0]),
        q_current,
    )

    assert out.shape == (1, CHUNK_SIZE, action_dim)
    expected_basis = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0]]])
    assert torch.allclose(out, solver.trajectory + expected_basis)
    assert torch.allclose(out[..., 6], solver.trajectory[..., 6])


def test_sample_action_chunk_from_features_can_leave_selected_delta_dims_unrestored():
    solver = _RecordingSolver(torch.full((1, CHUNK_SIZE, ACTION_DIM), 0.25))
    model = ServoVLA(
        vision_encoder=None,
        vlm_encoder=None,
        policy_head=nn.Linear(1, 1),
        fm_solver=solver,
        action_is_delta=True,
        action_delta_state_indices=[None] * ACTION_DIM,
    )
    q_current = torch.full((1, STATE_DIM), 1.0)

    out = model.sample_action_chunk_from_features(
        torch.randn(1, VISION_SEQ, D_VISION),
        torch.randn(1, 12, D_SEM),
        torch.ones(1, 12, dtype=torch.bool),
        torch.tensor([4.0]),
        q_current,
    )

    assert torch.allclose(out, solver.trajectory)


def test_servovla_forward_runs_frozen_online_encoders_without_encoder_grads():
    class _Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0), requires_grad=False)

        def forward(self, pixel_values):
            return pixel_values.mean(dim=(-1, -2)).mean(dim=2)[:, :, None] * self.weight

    class _VLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(3.0), requires_grad=False)

        def forward(self, vlm_inputs):
            ids = vlm_inputs["input_ids"].float()
            return ids[:, :, None] * self.weight

    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))

        def forward(self, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
            del t, c_sem_mask, frame_delay, q_current
            return x_t + self.weight * (f_vision.mean() + c_sem.mean())

    model = ServoVLA(
        vision_encoder=_Vision(),
        vlm_encoder=_VLM(),
        policy_head=_Policy(),
        fm_solver=None,
        action_is_delta=True,
    )
    x_t = torch.zeros(2, 4, 3)
    out = model(
        x_t=x_t,
        t=torch.zeros(2),
        pixel_values=torch.ones(2, 2, 3, 8, 8),
        vlm_inputs={
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        },
        c_sem_mask=torch.ones(2, 5, dtype=torch.bool),
        frame_delay=torch.zeros(2),
        q_current=torch.zeros(2, 3),
    )
    loss = out.sum()
    loss.backward()

    assert model.policy_head.weight.grad is not None
    assert model.vision_encoder.weight.grad is None
    assert model.vlm_encoder.weight.grad is None


def test_servovla_forward_casts_online_encoder_features_to_policy_dtype():
    class _Vision(nn.Module):
        def forward(self, pixel_values):
            return torch.ones(pixel_values.shape[0], 2, D_VISION, dtype=torch.bfloat16)

    class _VLM(nn.Module):
        def forward(self, vlm_inputs):
            return torch.ones(vlm_inputs["input_ids"].shape[0], 3, D_SEM, dtype=torch.bfloat16)

    class _Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        def forward(self, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
            del t, c_sem_mask, frame_delay, q_current
            assert f_vision.dtype == self.weight.dtype
            assert c_sem.dtype == self.weight.dtype
            return x_t + self.weight

    model = ServoVLA(
        vision_encoder=_Vision(),
        vlm_encoder=_VLM(),
        policy_head=_Policy(),
        fm_solver=None,
        action_is_delta=True,
    )

    out = model(
        x_t=torch.zeros(2, 4, ACTION_DIM),
        t=torch.zeros(2),
        pixel_values=torch.ones(2, 2, 3, 8, 8),
        vlm_inputs={"input_ids": torch.ones(2, 5, dtype=torch.long)},
        c_sem_mask=torch.ones(2, 3, dtype=torch.bool),
        frame_delay=torch.zeros(2),
        q_current=torch.zeros(2, STATE_DIM),
    )

    assert out.dtype == torch.float32


def test_servovla_encode_observations_returns_frozen_features():
    class _Vision(nn.Module):
        def forward(self, pixel_values):
            return pixel_values.mean(dim=(-1, -2)).mean(dim=2)[:, :, None]

    class _VLM(nn.Module):
        def forward(self, vlm_inputs):
            return vlm_inputs["input_ids"].float().unsqueeze(-1)

    model = ServoVLA(
        vision_encoder=_Vision(),
        vlm_encoder=_VLM(),
        policy_head=nn.Linear(1, 1),
        fm_solver=None,
        action_is_delta=True,
    )

    f_vision, c_sem = model.encode_observations(
        pixel_values=torch.ones(2, 2, 3, 8, 8),
        vlm_inputs={"input_ids": torch.ones(2, 5, dtype=torch.long)},
    )

    assert f_vision.shape == (2, 2, 1)
    assert c_sem.shape == (2, 5, 1)
    assert f_vision.requires_grad is False
    assert c_sem.requires_grad is False


def test_servovla_encode_observations_uses_shared_vlm_forward_gate(monkeypatch):
    calls = []

    class _Vision(nn.Module):
        def forward(self, pixel_values):
            return pixel_values.mean(dim=(-1, -2)).mean(dim=2)[:, :, None]

    class _VLM(nn.Module):
        def forward(self, vlm_inputs):
            return vlm_inputs["input_ids"].float().unsqueeze(-1)

    def _locked_vlm_forward(vlm_encoder, vlm_inputs):
        calls.append(vlm_encoder)
        return vlm_encoder(vlm_inputs)

    monkeypatch.setattr(
        "servovla.architectures.servo_vla.run_vlm_encoder_forward",
        _locked_vlm_forward,
        raising=False,
    )
    model = ServoVLA(
        vision_encoder=_Vision(),
        vlm_encoder=_VLM(),
        policy_head=nn.Linear(1, 1),
        fm_solver=None,
        action_is_delta=True,
    )

    _, c_sem = model.encode_observations(
        pixel_values=torch.ones(2, 2, 3, 8, 8),
        vlm_inputs={"input_ids": torch.ones(2, 5, dtype=torch.long)},
    )

    assert calls == [model.vlm_encoder]
    assert c_sem.shape == (2, 5, 1)


def test_inference_noise_fixed_seed_repeats_and_step_seed_changes():
    from servovla.architectures.inference_noise import make_inference_noise

    cpu = torch.device("cpu")
    rng_state_before = torch.random.get_rng_state()
    fixed_a = make_inference_noise(
        shape=(1, 2, 3),
        device=cpu,
        dtype=torch.float64,
        seed=123,
        seed_mode="fixed",
        action_step=0,
    )
    rng_state_after = torch.random.get_rng_state()
    fixed_b = make_inference_noise(
        shape=(1, 2, 3),
        device=cpu,
        dtype=torch.float64,
        seed=123,
        seed_mode="fixed",
        action_step=99,
    )
    step_a = make_inference_noise(
        shape=(1, 2, 3),
        device=cpu,
        dtype=torch.float64,
        seed=123,
        seed_mode="step",
        action_step=0,
    )
    step_b = make_inference_noise(
        shape=(1, 2, 3),
        device=cpu,
        dtype=torch.float64,
        seed=123,
        seed_mode="step",
        action_step=1,
    )

    assert fixed_a.device == cpu
    assert fixed_a.dtype == torch.float64
    assert torch.equal(rng_state_before, rng_state_after)
    assert torch.equal(fixed_a, fixed_b)
    assert torch.equal(fixed_a, step_a)
    assert not torch.equal(step_a, step_b)


def test_inference_noise_seeded_generator_preserves_cuda_device_index(monkeypatch):
    import servovla.architectures.inference_noise as inference_noise

    generator_devices = []

    class _FakeGenerator:
        def __init__(self, *, device):
            self.device = device
            generator_devices.append(device)

        def manual_seed(self, seed):
            self.seed = seed
            return self

    def _fake_randn(shape, *, generator=None, device=None, dtype=None):
        assert generator is not None
        assert device == torch.device("cuda:1")
        return torch.empty(shape, dtype=dtype)

    monkeypatch.setattr(inference_noise.torch, "Generator", _FakeGenerator)
    monkeypatch.setattr(inference_noise.torch, "randn", _fake_randn)

    out = inference_noise.make_inference_noise(
        shape=(1, 2, 3),
        device=torch.device("cuda:1"),
        dtype=torch.float32,
        seed=123,
        seed_mode="fixed",
        action_step=0,
    )

    assert generator_devices == [torch.device("cuda:1")]
    assert out.shape == (1, 2, 3)
