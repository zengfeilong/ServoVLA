from __future__ import annotations

import dataclasses
import inspect
import logging
import pickle
from types import SimpleNamespace

import torch
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation

from lerobot_policy_servovla.async_policy_server import ServoVLAPolicyServer, serve
from lerobot_policy_servovla.server_config_servovla import ServoVLAPolicyServerConfig


class _FakePolicy:
    def __init__(self):
        self.config = SimpleNamespace(image_features={})
        self.model = SimpleNamespace(
            vision_encoder=_FakeModule("vision_encoder"),
            vlm_encoder=_FakeModule("vlm_encoder"),
            policy_head=_FakePolicyHead(),
        )
        self.runtime_calls = []
        self.reset_calls = 0
        self.device = None
        self.predict_kwargs = None

    def to(self, device):
        self.device = device

    def configure_runtime(self, **kwargs):
        self.runtime_calls.append(kwargs)

    def shutdown_runtime(self):
        self.runtime_calls.append({"shutdown": True})

    def reset(self):
        self.reset_calls += 1

    def predict_action_chunk(self, observation, **kwargs):
        self.predict_kwargs = dict(kwargs)
        return torch.zeros(1, 2, 1)

    def get_last_inference_debug(self):
        action_step = None
        if self.predict_kwargs is not None:
            action_step = self.predict_kwargs.get("action_step")
        return {
            "current_action_step": action_step,
            "semantic_snapshot_action_step": action_step,
            "step_delay": 0,
            "frame_delay": 0,
            "action_step_source": "runtime",
            "semantic_sample_mode": "sync_aligned",
        }


class _FakePolicyHead:
    def __init__(self):
        self.to_calls = []
        self.attn = _FakeFlashAttentionModule()

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def modules(self):
        return iter([self, self.attn])


class _FakeModule:
    def __init__(self, name):
        self.name = name

    def modules(self):
        return iter([self])


class _CompiledModule:
    def __init__(self, original):
        self.original = original

    def modules(self):
        return self.original.modules()


class _FakeFlashAttentionModule:
    def __init__(self):
        self.flash_attention_dtype_calls = []

    def set_flash_attention_dtype(self, dtype):
        self.flash_attention_dtype_calls.append(dtype)


class _FakePolicyClass:
    last_policy = None
    policies = []

    @classmethod
    def from_pretrained(cls, path):
        cls.last_policy = _FakePolicy()
        cls.last_policy.path = path
        cls.policies.append(cls.last_policy)
        return cls.last_policy


class _FakeContext:
    def peer(self):
        return "test-client"


class _IdentityProcessor:
    def __call__(self, value):
        return value


def test_send_policy_instructions_configures_policy_runtime(monkeypatch):
    _FakePolicyClass.policies = []
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(
        ServoVLAPolicyServerConfig(
            fps=10,
            max_frame_delay=0,
            semantic_wait_warn_ms=250,
            semantic_wait_fail_ms=3000,
        )
    )
    request = SimpleNamespace(
        data=pickle.dumps(
            RemotePolicyConfig(
                policy_type="servovla",
                pretrained_name_or_path="/tmp/pretrained",
                lerobot_features={},
                actions_per_chunk=8,
                device="cpu",
                rename_map={},
            )
        )
    )

    server.SendPolicyInstructions(request, _FakeContext())

    assert _FakePolicyClass.last_policy.runtime_calls == [
        {
            "max_frame_delay": 0,
            "semantic_wait_warn_ms": 250,
            "semantic_wait_fail_ms": 3000,
        }
    ]
    assert _FakePolicyClass.last_policy.reset_calls == 1


def test_send_policy_instructions_reuses_same_loaded_policy(monkeypatch):
    _FakePolicyClass.policies = []
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    request = SimpleNamespace(
        data=pickle.dumps(
            RemotePolicyConfig(
                policy_type="servovla",
                pretrained_name_or_path="/tmp/pretrained",
                lerobot_features={},
                actions_per_chunk=8,
                device="cpu",
                rename_map={},
            )
        )
    )

    server.SendPolicyInstructions(request, _FakeContext())
    server.SendPolicyInstructions(request, _FakeContext())

    assert len(_FakePolicyClass.policies) == 1
    assert _FakePolicyClass.last_policy.runtime_calls == [
        {
            "max_frame_delay": None,
            "semantic_wait_warn_ms": 500,
            "semantic_wait_fail_ms": 3000,
        },
        {
            "max_frame_delay": None,
            "semantic_wait_warn_ms": 500,
            "semantic_wait_fail_ms": 3000,
        },
    ]
    assert _FakePolicyClass.last_policy.reset_calls == 2


def test_send_policy_instructions_configures_policy_head_attention_dtype_from_env(monkeypatch):
    _FakePolicyClass.policies = []
    monkeypatch.setenv("SERVOVLA_POLICY_HEAD_DTYPE", "off")
    monkeypatch.setenv("SERVOVLA_POLICY_HEAD_ATTN_DTYPE", "bfloat16")
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    request = SimpleNamespace(
        data=pickle.dumps(
            RemotePolicyConfig(
                policy_type="servovla",
                pretrained_name_or_path="/tmp/pretrained",
                lerobot_features={},
                actions_per_chunk=8,
                device="cuda",
                rename_map={},
            )
        )
    )

    server.SendPolicyInstructions(request, _FakeContext())

    assert _FakePolicyClass.last_policy.model.policy_head.to_calls == []
    assert _FakePolicyClass.last_policy.model.policy_head.attn.flash_attention_dtype_calls == [
        torch.bfloat16
    ]


def test_send_policy_instructions_compiles_all_model_modules_once(monkeypatch):
    _FakePolicyClass.policies = []
    compiled_modules = []

    def _fake_compile(module, **kwargs):
        compiled = _CompiledModule(module)
        compiled_modules.append((module, kwargs, compiled))
        return compiled

    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE", "1")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_DYNAMIC", "false")
    monkeypatch.setattr(torch, "compile", _fake_compile)
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    request = SimpleNamespace(
        data=pickle.dumps(
            RemotePolicyConfig(
                policy_type="servovla",
                pretrained_name_or_path="/tmp/pretrained",
                lerobot_features={},
                actions_per_chunk=8,
                device="cuda",
                rename_map={},
            )
        )
    )

    server.SendPolicyInstructions(request, _FakeContext())
    server.SendPolicyInstructions(request, _FakeContext())

    assert len(compiled_modules) == 3
    compiled_by_original = {module: compiled for module, _, compiled in compiled_modules}
    assert (
        _FakePolicyClass.last_policy.model.vision_encoder
        is compiled_by_original[compiled_modules[0][0]]
    )
    assert (
        _FakePolicyClass.last_policy.model.vlm_encoder
        is compiled_by_original[compiled_modules[1][0]]
    )
    assert (
        _FakePolicyClass.last_policy.model.policy_head
        is compiled_by_original[compiled_modules[2][0]]
    )
    assert [kwargs for _, kwargs, _ in compiled_modules] == [
        {"mode": "reduce-overhead", "dynamic": False},
        {"mode": "reduce-overhead", "dynamic": False},
        {"mode": "reduce-overhead", "dynamic": False},
    ]


def test_send_policy_instructions_allows_module_compile_backend_overrides(monkeypatch):
    _FakePolicyClass.policies = []
    compiled_modules = []

    def _fake_compile(module, **kwargs):
        compiled = _CompiledModule(module)
        compiled_modules.append((getattr(module, "name", "policy_head"), kwargs, compiled))
        return compiled

    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE", "1")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_BACKEND", "inductor")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_MODE", "reduce-overhead")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_DYNAMIC", "false")
    monkeypatch.setenv("SERVOVLA_TORCH_COMPILE_VLM_ENCODER_BACKEND", "eager")
    monkeypatch.setattr(torch, "compile", _fake_compile)
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    request = SimpleNamespace(
        data=pickle.dumps(
            RemotePolicyConfig(
                policy_type="servovla",
                pretrained_name_or_path="/tmp/pretrained",
                lerobot_features={},
                actions_per_chunk=8,
                device="cuda",
                rename_map={},
            )
        )
    )

    server.SendPolicyInstructions(request, _FakeContext())

    assert [(name, kwargs) for name, kwargs, _ in compiled_modules] == [
        ("vision_encoder", {"backend": "inductor", "mode": "reduce-overhead", "dynamic": False}),
        ("vlm_encoder", {"backend": "eager", "dynamic": False}),
        ("policy_head", {"backend": "inductor", "mode": "reduce-overhead", "dynamic": False}),
    ]


def test_send_policy_instructions_releases_policy_when_pretrained_changes(monkeypatch):
    _FakePolicyClass.policies = []
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.get_policy_class",
        lambda _name: _FakePolicyClass,
    )
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.make_pre_post_processors",
        lambda *args, **kwargs: (_IdentityProcessor(), _IdentityProcessor()),
    )

    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))

    def request_for(path: str):
        return SimpleNamespace(
            data=pickle.dumps(
                RemotePolicyConfig(
                    policy_type="servovla",
                    pretrained_name_or_path=path,
                    lerobot_features={},
                    actions_per_chunk=8,
                    device="cpu",
                    rename_map={},
                )
            )
        )

    server.SendPolicyInstructions(request_for("/tmp/pretrained-a"), _FakeContext())
    first_policy = _FakePolicyClass.last_policy
    server.SendPolicyInstructions(request_for("/tmp/pretrained-b"), _FakeContext())

    assert len(_FakePolicyClass.policies) == 2
    assert first_policy.runtime_calls[-1] == {"shutdown": True}
    assert _FakePolicyClass.last_policy.path == "/tmp/pretrained-b"


def test_stop_shuts_down_policy_runtime():
    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    fake_policy = _FakePolicy()
    server.policy = fake_policy

    server.stop()

    assert fake_policy.runtime_calls == [{"shutdown": True}]


def test_policy_server_passes_observation_timestep_as_action_step(monkeypatch, caplog):
    monkeypatch.setattr(
        "lerobot_policy_servovla.async_policy_server.raw_observation_to_observation",
        lambda raw, lerobot_features, policy_image_features: raw,
    )
    server = ServoVLAPolicyServer(ServoVLAPolicyServerConfig(fps=10))
    fake_policy = _FakePolicy()
    server.policy = fake_policy
    server.lerobot_features = {}
    server.actions_per_chunk = 2
    server.preprocessor = _IdentityProcessor()
    server.postprocessor = _IdentityProcessor()

    observation = TimedObservation(
        timestamp=123.0,
        timestep=37,
        observation={"observation.state": torch.zeros(1)},
    )

    caplog.set_level(logging.INFO, logger=server.logger.name)
    server._predict_action_chunk(observation)

    assert fake_policy.predict_kwargs["action_step"] == 37
    assert "current_action_step=37" in caplog.text
    assert "step_delay=0" in caplog.text
    assert "action_step_source=runtime" in caplog.text
    assert "action_chunk_size=2" in caplog.text


def test_serve_wrapper_exposes_real_config_type_for_draccus():
    wrapped = serve.__wrapped__
    cfg_annotation = inspect.getfullargspec(wrapped).annotations["cfg"]

    assert cfg_annotation is ServoVLAPolicyServerConfig
    assert dataclasses.is_dataclass(cfg_annotation)
