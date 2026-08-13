from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class BatchQuotaState:
    residuals: list[float] = field(default_factory=list)


def _as_list(value) -> list:
    if value is None:
        return []
    return list(value)


def _cfg_get(cfg, key: str, default=None):
    getter = getattr(cfg, "get", None)
    if getter is not None:
        return getter(key, default)
    return getattr(cfg, key, default)


def validate_batch_mixing_config(cfg) -> None:
    training_cfg = _cfg_get(cfg, "training", cfg)
    batch_mixing_cfg = _cfg_get(training_cfg, "batch_mixing", {})
    mode = str(_cfg_get(batch_mixing_cfg, "mode", "explicit_weights"))
    integerization = str(_cfg_get(batch_mixing_cfg, "integerization", "largest_remainder"))
    if mode != "explicit_weights":
        raise ValueError(
            "training.batch_mixing.mode only supports 'explicit_weights' for raw end-to-end loading."
        )
    if integerization != "largest_remainder":
        raise ValueError(
            "training.batch_mixing.integerization only supports 'largest_remainder' for raw end-to-end loading."
        )


def resolve_train_weights(cfg, dataset_names: Sequence[str]) -> list[float]:
    return resolve_sampler_train_weights(cfg, dataset_names)


def _resolve_explicit_train_weights(cfg, dataset_names: Sequence[str]) -> list[float]:
    names = [str(name) for name in dataset_names]
    if len(names) == 1 and cfg.dataset.get("train_weights") is None:
        return [1.0]

    raw_weights = cfg.dataset.get("train_weights")
    if raw_weights is None:
        raise ValueError(
            "dataset.train_weights is required when multiple dataset.train_names are configured."
        )

    weights = [float(value) for value in _as_list(raw_weights)]
    if len(weights) != len(names):
        raise ValueError("dataset.train_weights must have the same length as dataset.train_names.")
    for weight in weights:
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("dataset.train_weights values must be finite and positive.")
    return weights


def _validate_dataset_frame_counts(
    dataset_frame_counts: Sequence[int] | None,
    dataset_names: Sequence[str],
) -> list[int]:
    if dataset_frame_counts is None:
        raise ValueError(
            "dataset_frame_counts is required when dataset.train_weight_strategy='equal_seen_frames'."
        )
    counts = [int(value) for value in dataset_frame_counts]
    if len(counts) != len(dataset_names):
        raise ValueError("dataset_frame_counts must have the same length as dataset.train_names.")
    if any(count <= 0 for count in counts):
        raise ValueError("dataset_frame_counts values must be positive.")
    return counts


def resolve_sampler_train_weights(
    cfg,
    dataset_names: Sequence[str],
    *,
    dataset_frame_counts: Sequence[int] | None = None,
) -> list[float]:
    names = [str(name) for name in dataset_names]
    dataset_cfg = _cfg_get(cfg, "dataset", cfg)
    strategy = _cfg_get(dataset_cfg, "train_weight_strategy", None)
    if strategy is None:
        return _resolve_explicit_train_weights(cfg, names)

    strategy = str(strategy)
    if strategy in {"explicit", "explicit_weights"}:
        return _resolve_explicit_train_weights(cfg, names)
    if strategy == "equal_seen_frames":
        _validate_dataset_frame_counts(dataset_frame_counts, names)
        return [1.0 for _ in names]
    raise ValueError(
        "dataset.train_weight_strategy must be one of "
        "['explicit_weights', 'equal_seen_frames'], got "
        f"{strategy!r}."
    )


def resolve_action_normalization_weights(cfg, dataset_names: Sequence[str]) -> list[float]:
    names = [str(name) for name in dataset_names]
    training_cfg = _cfg_get(cfg, "training", {})
    norm_cfg = _cfg_get(training_cfg, "action_normalization", {})
    strategy = str(_cfg_get(norm_cfg, "dataset_weight_strategy", "equal_datasets"))

    if strategy == "equal_datasets":
        return [1.0 for _ in names]
    if strategy in {"explicit", "explicit_weights"}:
        raw_weights = _cfg_get(norm_cfg, "dataset_weights", None)
        if raw_weights is None:
            raise ValueError(
                "training.action_normalization.dataset_weights is required when "
                "dataset_weight_strategy='explicit_weights'."
            )
        weights = [float(value) for value in _as_list(raw_weights)]
        if len(weights) != len(names):
            raise ValueError(
                "training.action_normalization.dataset_weights must have the same length as dataset.train_names."
            )
        if any((not math.isfinite(weight)) or weight <= 0.0 for weight in weights):
            raise ValueError(
                "training.action_normalization.dataset_weights values must be finite and positive."
            )
        return weights
    raise ValueError(
        "training.action_normalization.dataset_weight_strategy must be one of "
        "['equal_datasets', 'explicit_weights'], got "
        f"{strategy!r}."
    )


def compute_batch_quotas(
    *, batch_size: int, weights: Sequence[float], state: BatchQuotaState
) -> list[int]:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    weights = [float(weight) for weight in weights]
    if not weights:
        raise ValueError("weights must not be empty")
    if any((not math.isfinite(weight)) or weight <= 0.0 for weight in weights):
        raise ValueError("weights must be finite and positive")

    if len(state.residuals) != len(weights):
        state.residuals = [0.0 for _ in weights]

    total_weight = sum(weights)
    exact = [
        float(batch_size) * weight / total_weight + state.residuals[idx]
        for idx, weight in enumerate(weights)
    ]
    quotas = [int(value) for value in exact]
    remaining = int(batch_size) - sum(quotas)
    ranked = sorted(
        ((exact[idx] - quotas[idx], -idx, idx) for idx in range(len(weights))),
        reverse=True,
    )
    for _, _, idx in ranked[:remaining]:
        quotas[idx] += 1
    state.residuals = [exact[idx] - float(quotas[idx]) for idx in range(len(weights))]
    return quotas
