import logging
import os

import torch
import torch.nn as nn
from transformers import AutoBackbone, AutoConfig

from servovla.config.hf_offline import hf_from_pretrained_kwargs

log = logging.getLogger(__name__)


def _load_backbone_from_pretrained(model_id: str, **kwargs):
    try:
        return AutoBackbone.from_pretrained(model_id, **kwargs)
    except Exception as auto_exc:
        config_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "local_files_only", "revision", "token")
            if key in kwargs
        }
        config = AutoConfig.from_pretrained(model_id, **config_kwargs)
        backbone_cls = AutoBackbone._model_mapping[type(config)]
        log.info(
            "AutoBackbone load failed for %s (%s); retrying with mapped class %s",
            model_id,
            auto_exc,
            backbone_cls.__name__,
        )
        return backbone_cls.from_pretrained(model_id, **kwargs)


class VisionEncoder(nn.Module):
    """
    Frozen visual feature extractor used for end-to-end training and deployment inference.
    Default to stable attention backends for deployment; FlashAttention can be re-enabled explicitly.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        requested_attn_impl = (
            attn_implementation
            or os.environ.get("SERVOVLA_VISION_ATTN_IMPLEMENTATION")
            or os.environ.get("SERVOVLA_ATTN_IMPLEMENTATION")
            or "sdpa"
        )
        fallback_order = [requested_attn_impl]
        for fallback_attn_impl in ("sdpa", "eager"):
            if fallback_attn_impl not in fallback_order:
                fallback_order.append(fallback_attn_impl)

        last_exc = None
        for attn_impl in fallback_order:
            try:
                log.info(
                    "Loading Frozen Vision Encoder: %s | dtype=bf16 | attn_implementation=%s",
                    model_id,
                    attn_impl,
                )
                self.model = _load_backbone_from_pretrained(
                    model_id,
                    **hf_from_pretrained_kwargs(),
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                    reshape_hidden_states=False,
                )
                self.attn_implementation = attn_impl
                break
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Failed to load Vision Encoder with attn_implementation=%s (%s)",
                    attn_impl,
                    exc,
                )
        else:
            raise RuntimeError(
                f"Unable to load VisionEncoder for {model_id} with any supported attention implementation."
            ) from last_exc

        self.feature_dim = self.model.config.hidden_size

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.eval()

    def _extract_grid_tokens(self, outputs) -> torch.Tensor:
        feature_maps = getattr(outputs, "feature_maps", None)
        if not feature_maps:
            raise ValueError("Expected DINO backbone outputs with at least one feature map.")
        grid_tokens = feature_maps[-1]
        if grid_tokens.ndim != 3:
            raise ValueError(
                f"Expected final feature map shaped (B, N, D), got {tuple(grid_tokens.shape)}"
            )
        return grid_tokens

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.eval()
        if pixel_values.ndim == 5:
            batch_size, num_views = pixel_values.shape[:2]
            flat_pixel_values = pixel_values.flatten(0, 1)
            outputs = self.model(pixel_values=flat_pixel_values)
            f_vision = self._extract_grid_tokens(outputs)
            f_vision = f_vision.view(batch_size, num_views, f_vision.shape[1], f_vision.shape[2])
            return f_vision.flatten(1, 2)

        outputs = self.model(pixel_values=pixel_values)
        return self._extract_grid_tokens(outputs)
