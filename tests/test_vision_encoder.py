from types import SimpleNamespace

import torch
import torch.nn as nn

import servovla.architectures.vision_encoder as vision_module
from servovla.architectures.vision_encoder import VisionEncoder


class _FakeVisionBackbone(nn.Module):
    def __init__(self, *, hidden_size: int, tokens_per_view: int):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.tokens_per_view = int(tokens_per_view)

    def forward(self, pixel_values: torch.Tensor):
        batch_size = pixel_values.shape[0]
        hidden = self.config.hidden_size
        values = torch.arange(batch_size * self.tokens_per_view * hidden, dtype=torch.float32)
        feature_map = values.view(batch_size, self.tokens_per_view, hidden)
        return SimpleNamespace(feature_maps=(feature_map,))


def _build_encoder_with_fake_backbone(*, hidden_size: int, tokens_per_view: int):
    encoder = VisionEncoder.__new__(VisionEncoder)
    nn.Module.__init__(encoder)
    encoder.model = _FakeVisionBackbone(
        hidden_size=hidden_size,
        tokens_per_view=tokens_per_view,
    )
    encoder.feature_dim = hidden_size
    encoder.model_id = "fake/dinov3"
    return encoder


def test_forward_returns_grid_tokens_from_backbone_feature_map():
    encoder = _build_encoder_with_fake_backbone(hidden_size=3, tokens_per_view=256)

    output = encoder(torch.zeros(2, 3, 256, 256))

    assert output.shape == (2, 256, 3)
    assert torch.equal(output[:, 0], torch.tensor([[0.0, 1.0, 2.0], [768.0, 769.0, 770.0]]))


def test_forward_multiview_flattens_backbone_feature_maps_per_camera():
    encoder = _build_encoder_with_fake_backbone(hidden_size=2, tokens_per_view=256)

    output = encoder(torch.zeros(2, 2, 3, 256, 256))

    assert output.shape == (2, 512, 2)


def test_init_falls_back_to_mapped_backbone_class_when_auto_backbone_offline_load_fails(
    monkeypatch,
):
    class _FakeConfig:
        pass

    class _DirectBackbone(nn.Module):
        calls = []

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=7)
            self.param = nn.Parameter(torch.ones(()))

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.calls.append((model_id, kwargs))
            return cls()

        def forward(self, pixel_values):
            raise NotImplementedError

    class _BrokenAutoBackbone:
        _model_mapping = {_FakeConfig: _DirectBackbone}

        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise RuntimeError("offline repo_exists")

    class _AutoConfig:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return _FakeConfig()

    monkeypatch.setattr(vision_module, "AutoBackbone", _BrokenAutoBackbone)
    monkeypatch.setattr(vision_module, "AutoConfig", _AutoConfig, raising=False)

    encoder = VisionEncoder("facebook/dinov3", attn_implementation="flash_attention_2")

    assert encoder.feature_dim == 7
    assert isinstance(encoder.model, _DirectBackbone)
    assert _DirectBackbone.calls[0][0] == "facebook/dinov3"
    assert "config" not in _DirectBackbone.calls[0][1]
    assert _DirectBackbone.calls[0][1]["local_files_only"] is True
    assert _DirectBackbone.calls[0][1]["attn_implementation"] == "flash_attention_2"
    assert _DirectBackbone.calls[0][1]["reshape_hidden_states"] is False


def test_init_falls_back_from_flash_attention_to_sdpa_before_eager(monkeypatch):
    calls = []

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(hidden_size=7)
            self.param = nn.Parameter(torch.ones(()))

        def forward(self, pixel_values):
            raise NotImplementedError

    def _from_pretrained(model_id, **kwargs):
        calls.append(kwargs["attn_implementation"])
        if kwargs["attn_implementation"] == "flash_attention_2":
            raise RuntimeError("flash attention unavailable")
        if kwargs["attn_implementation"] == "sdpa":
            return _Backbone()
        raise AssertionError("eager should not be tried before sdpa")

    monkeypatch.setattr(
        vision_module,
        "_load_backbone_from_pretrained",
        _from_pretrained,
    )

    encoder = VisionEncoder("facebook/dinov3", attn_implementation="flash_attention_2")

    assert encoder.attn_implementation == "sdpa"
    assert calls == ["flash_attention_2", "sdpa"]
