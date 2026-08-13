from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

from servovla.architectures.policy_head_checkpoint import (
    load_policy_head_state_dict as _load_policy_head_state_dict,
)
from servovla.architectures.policy_head_checkpoint import (
    prefixed_policy_head_state_dict,
)


def policy_head_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return dict(
        prefixed_policy_head_state_dict(
            model,
            prefix="policy_head.",
            clone_to_cpu=True,
        )
    )


def load_policy_head_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    _load_policy_head_state_dict(model, state_dict, strict=True)


class PolicyHeadEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        policy_head = getattr(model, "policy_head", None)
        if policy_head is None:
            raise TypeError("PolicyHeadEma requires a model with a policy_head attribute.")
        self.decay = float(decay)
        # Keep the EMA shadow in fp32 even when the trainable policy head runs in
        # bf16. Updating the shadow in bf16 quantizes the small EMA increments and
        # can produce unusable EMA checkpoints for high-decay settings.
        self.module = copy.deepcopy(policy_head).eval().float()
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = getattr(model, "policy_head")
        source_state = source.state_dict()
        ema_state = self.module.state_dict()
        for key, ema_value in ema_state.items():
            source_value = (
                source_state[key].detach().to(device=ema_value.device, dtype=ema_value.dtype)
            )
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
            else:
                ema_value.copy_(source_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(
            prefixed_policy_head_state_dict(
                self.module,
                prefix="policy_head.",
                clone_to_cpu=True,
            )
        )

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        _load_policy_head_state_dict(self.module, state_dict, strict=True)
        self.module.float()

    @contextmanager
    def apply_to(self, model: nn.Module) -> Iterator[nn.Module]:
        policy_head = getattr(model, "policy_head")
        original = {key: value.detach().clone() for key, value in policy_head.state_dict().items()}
        policy_head.load_state_dict(self.module.state_dict(), strict=True)
        try:
            yield model
        finally:
            policy_head.load_state_dict(original, strict=True)
