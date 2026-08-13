import logging
import os

import torch
import torch.nn as nn
from transformers import AutoModel

from servovla.config.hf_offline import hf_from_pretrained_kwargs

log = logging.getLogger(__name__)


class VLMEncoder(nn.Module):
    """
    Frozen semantic / language encoder used for end-to-end training and deployment inference.
    Default to stable attention backends for deployment; FlashAttention can be re-enabled explicitly.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-0.8B",
        attn_implementation: str | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        requested_attn_impl = (
            attn_implementation
            or os.environ.get("SERVOVLA_VLM_ATTN_IMPLEMENTATION")
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
                    "Loading Frozen VLM Encoder: %s | dtype=bf16 | attn_implementation=%s",
                    model_id,
                    attn_impl,
                )
                self.model = AutoModel.from_pretrained(
                    model_id,
                    **hf_from_pretrained_kwargs(),
                    dtype=torch.bfloat16,
                    attn_implementation=attn_impl,
                )
                self.attn_implementation = attn_impl
                break
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "Failed to load VLM Encoder with attn_implementation=%s (%s)",
                    attn_impl,
                    exc,
                )
        else:
            raise RuntimeError(
                f"Unable to load VLMEncoder for {model_id} with any supported attention implementation."
            ) from last_exc

        self.feature_dim = self.model.config.text_config.hidden_size

        for param in self.model.parameters():
            param.requires_grad_(False)

        self.eval()

    @torch.no_grad()
    def forward(self, vlm_inputs: dict) -> torch.Tensor:
        self.eval()
        outputs = self.model(
            **vlm_inputs, output_hidden_states=False, return_dict=True, use_cache=False
        )

        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            c_sem = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            c_sem = outputs.hidden_states[-1]
        else:
            raise RuntimeError("VLM encoder output does not include last_hidden_state.")

        return c_sem
