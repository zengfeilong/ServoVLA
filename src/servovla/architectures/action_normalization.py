from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn


def _as_stat_tensor(values: Iterable[float] | torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim not in {1, 2} or tensor.numel() == 0:
        raise ValueError(f"{name} must be a non-empty 1D or 2D tensor/sequence.")
    return tensor


@dataclass(frozen=True, slots=True)
class ActionNormalizationConfig:
    enabled: bool
    mean: list
    std: list
    eps: float


class ActionNormalizer(nn.Module):
    def __init__(
        self,
        *,
        enabled: bool = False,
        mean: Iterable[float] | torch.Tensor | None = None,
        std: Iterable[float] | torch.Tensor | None = None,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.eps = float(eps)
        mean_tensor = (
            torch.zeros(1, dtype=torch.float32)
            if mean is None
            else _as_stat_tensor(mean, name="mean")
        )
        std_tensor = (
            torch.ones_like(mean_tensor) if std is None else _as_stat_tensor(std, name="std")
        )
        if tuple(mean_tensor.shape) != tuple(std_tensor.shape):
            raise ValueError("ActionNormalizer mean and std must have the same shape.")
        self.register_buffer("mean", mean_tensor, persistent=False)
        self.register_buffer("std", std_tensor.clamp_min(self.eps), persistent=False)

    @property
    def action_dim(self) -> int:
        return int(self.mean.shape[-1])

    def _stats_for(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean.to(device=action.device, dtype=action.dtype)
        std = self.std.to(device=action.device, dtype=action.dtype)
        if mean.ndim == 1:
            if int(action.shape[-1]) != int(mean.shape[0]):
                raise ValueError(
                    f"Action dim {int(action.shape[-1])} does not match normalizer dim {int(mean.shape[0])}."
                )
            return mean, std

        if action.ndim < 2:
            raise ValueError(
                "Per-horizon action normalization requires an action tensor with a horizon dimension."
            )
        horizon = int(action.shape[-2])
        action_dim = int(action.shape[-1])
        if horizon > int(mean.shape[0]) or action_dim != int(mean.shape[1]):
            raise ValueError(
                "Action shape is incompatible with per-horizon normalizer stats: "
                f"action horizon/dim=({horizon}, {action_dim}), stats={tuple(mean.shape)}."
            )
        return mean[:horizon], std[:horizon]

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return action
        mean, std = self._stats_for(action)
        return (action - mean) / std

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return action
        mean, std = self._stats_for(action)
        return action * std + mean

    def to_config(self) -> ActionNormalizationConfig:
        return ActionNormalizationConfig(
            enabled=self.enabled,
            mean=self.mean.detach().cpu().tolist(),
            std=self.std.detach().cpu().tolist(),
            eps=float(self.eps),
        )

    @classmethod
    def from_config(
        cls,
        *,
        enabled: bool,
        mean: Iterable[float] | torch.Tensor | None = None,
        std: Iterable[float] | torch.Tensor | None = None,
        eps: float = 1.0e-6,
    ) -> ActionNormalizer:
        return cls(enabled=enabled, mean=mean, std=std, eps=eps)
