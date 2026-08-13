from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def _action_step_delay_support(
    *,
    chunk_size: int,
    chunk_size_threshold: float,
    max_delay_chunks: int,
) -> list[tuple[int, int]]:
    chunk_size = int(chunk_size)
    chunk_size_threshold = float(chunk_size_threshold)
    max_delay_chunks = int(max_delay_chunks)

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if max_delay_chunks < 1:
        raise ValueError(f"max_delay_chunks must be >= 1, got {max_delay_chunks}")
    if not 0.0 <= chunk_size_threshold <= 0.5:
        raise ValueError(f"chunk_size_threshold must be in [0, 0.5], got {chunk_size_threshold}")

    # Intervals are half-open: (28, 31) accepts delays 28, 29, 30.
    # Runtime action_step logs with actions_per_chunk=16 place one stale chunk
    # at delay 14/15, so the boundary stride is chunk_size - 1.
    support: list[tuple[int, int]] = [(0, 1)]
    bucket_width = max(1, int(math.floor(chunk_size * chunk_size_threshold)))
    boundary_stride = max(chunk_size - 1, 1)
    for bucket_idx in range(1, max_delay_chunks + 1):
        boundary = bucket_idx * boundary_stride
        start = max(1, boundary - bucket_width + 1)
        support.append((start, boundary + 1))
    return support


def _delay_support_max(support: list[tuple[int, int]]) -> int:
    return max(end - 1 for _, end in support)


@PreTrainedConfig.register_subclass("servovla")
@dataclass
class ServoVLAConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    chunk_size: int = 16
    n_action_steps: int = 16
    action_dim: int = 6
    state_dim: int = 6
    num_cameras: int = 2
    num_inference_steps: int = 10
    inference_noise_seed: int | None = None
    inference_noise_seed_mode: str = "step"
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    vision_feature_dim: int = 1024
    semantic_feature_dim: int = 1024
    vision_image_size: int = 256
    vlm_image_size: int = 256
    vision_grid_size: int = 16
    dropout: float = 0.0
    # Deprecated legacy export fields. Accepted for old config.json files; ignored for delay semantics.
    fast_rate_hz: int | None = None
    slow_rate_hz: int | None = None
    zoh_ratio: int | None = None
    max_frame_delay: int | None = None
    delay_max_chunks: int | None = 2
    delay_chunk_size_threshold: float | None = 0.2
    semantic_admission_policy: str = "bsr"
    semantic_wait_warn_ms: int = 500
    semantic_wait_fail_ms: int = 3000
    action_mode: str = "abs"
    action_is_delta: bool = False
    action_delta_state_indices: list[int | None] | None = None
    action_normalization_enabled: bool = False
    action_normalization_mean: list[Any] | None = None
    action_normalization_std: list[Any] | None = None
    action_normalization_eps: float = 1.0e-6
    front_camera_key: str = f"{OBS_IMAGES}.front"
    wrist_camera_key: str = f"{OBS_IMAGES}.wrist"
    camera_keys: list[str] | None = None
    vision_model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    vlm_model_name: str = "Qwen/Qwen3.5-0.8B"
    tokenizer_padding: str = "longest"
    tokenizer_max_length: int = 512
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 1e-5

    input_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        }
    )
    output_features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        }
    )
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.vision_image_size = int(self.vision_image_size)
        self.vlm_image_size = int(self.vlm_image_size)
        self.vision_grid_size = self.vision_image_size // 16
        if self.camera_keys is None:
            self.camera_keys = [self.front_camera_key, self.wrist_camera_key]
        else:
            self.camera_keys = [str(key) for key in self.camera_keys]
        if not self.camera_keys:
            raise ValueError("ServoVLA requires at least one camera key.")
        self.num_cameras = len(self.camera_keys)
        self.front_camera_key = self.camera_keys[0]
        self.wrist_camera_key = (
            self.camera_keys[1] if len(self.camera_keys) > 1 else self.camera_keys[0]
        )
        visual_shape = (3, self.vision_image_size, self.vision_image_size)
        retained_features = {}
        for key, feature in self.input_features.items():
            feature_type = getattr(feature, "type", None)
            if feature_type != FeatureType.VISUAL and str(feature_type).upper() != "VISUAL":
                retained_features[key] = feature
        self.input_features = retained_features
        for camera_key in self.camera_keys:
            self.input_features[camera_key] = PolicyFeature(
                type=FeatureType.VISUAL, shape=visual_shape
            )
        self.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(int(self.state_dim),)
        )
        self.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(int(self.action_dim),)
        )
        if self.inference_noise_seed is not None:
            self.inference_noise_seed = int(self.inference_noise_seed)
        self.inference_noise_seed_mode = str(self.inference_noise_seed_mode).lower()
        if self.inference_noise_seed_mode not in {"fixed", "step"}:
            raise ValueError("`inference_noise_seed_mode` must be 'fixed' or 'step'.")
        if self.chunk_size <= 0:
            raise ValueError("`chunk_size` must be strictly positive.")
        if self.n_action_steps <= 0 or self.n_action_steps > self.chunk_size:
            raise ValueError("`n_action_steps` must be in the range [1, chunk_size].")
        if self.action_dim <= 0:
            raise ValueError("`action_dim` must be strictly positive.")
        if self.state_dim <= 0:
            raise ValueError("`state_dim` must be strictly positive.")
        self.action_mode = self.action_mode.strip().lower()
        if self.action_mode not in {"delta", "abs"}:
            raise ValueError("`action_mode` must be 'delta' or 'abs'.")
        if self.action_mode == "abs" and bool(self.action_is_delta):
            self.action_mode = "delta"
        self.action_is_delta = self.action_mode == "delta"
        if self.action_delta_state_indices is None:
            self.action_delta_state_indices = list(range(int(self.action_dim)))
        else:
            if len(self.action_delta_state_indices) != int(self.action_dim):
                raise ValueError(
                    "`action_delta_state_indices` length must match `action_dim` "
                    f"({self.action_dim}), got {len(self.action_delta_state_indices)}."
                )
            normalized_indices: list[int | None] = []
            for idx, item in enumerate(self.action_delta_state_indices):
                if item is None:
                    normalized_indices.append(None)
                    continue
                state_idx = int(item)
                if state_idx < 0:
                    raise ValueError(
                        f"`action_delta_state_indices[{idx}]` must be non-negative or null."
                    )
                normalized_indices.append(state_idx)
            self.action_delta_state_indices = normalized_indices
        self.action_normalization_enabled = bool(self.action_normalization_enabled)
        self.action_normalization_eps = float(self.action_normalization_eps)
        if self.action_normalization_enabled and (
            self.action_normalization_mean is None or self.action_normalization_std is None
        ):
            raise ValueError(
                "action_normalization_mean and action_normalization_std are required when "
                "action_normalization_enabled=True."
            )
        has_delay_max_chunks = self.delay_max_chunks is not None
        has_delay_threshold = self.delay_chunk_size_threshold is not None
        if has_delay_max_chunks != has_delay_threshold:
            raise ValueError(
                "delay_max_chunks and delay_chunk_size_threshold must be configured together"
            )
        bucketed_max_supported_delay = None
        if has_delay_max_chunks and has_delay_threshold:
            self.delay_max_chunks = int(self.delay_max_chunks)
            self.delay_chunk_size_threshold = float(self.delay_chunk_size_threshold)
            support = _action_step_delay_support(
                chunk_size=int(self.chunk_size),
                chunk_size_threshold=self.delay_chunk_size_threshold,
                max_delay_chunks=self.delay_max_chunks,
            )
            bucketed_max_supported_delay = _delay_support_max(support)
        if self.max_frame_delay is None:
            self.max_frame_delay = (
                bucketed_max_supported_delay if bucketed_max_supported_delay is not None else 0
            )
        self.max_frame_delay = int(self.max_frame_delay)
        if self.max_frame_delay < 0:
            raise ValueError(f"max_frame_delay must be >= 0, got {self.max_frame_delay}")
        self.semantic_admission_policy = str(self.semantic_admission_policy).strip().lower()
        allowed_admission_policies = {"bsr", "age_only", "latest_cache"}
        if self.semantic_admission_policy not in allowed_admission_policies:
            raise ValueError(
                "semantic_admission_policy must be one of "
                f"{sorted(allowed_admission_policies)}, got {self.semantic_admission_policy!r}"
            )
        if int(self.semantic_wait_warn_ms) < 0:
            raise ValueError("semantic_wait_warn_ms must be non-negative")
        if int(self.semantic_wait_fail_ms) <= 0:
            raise ValueError("semantic_wait_fail_ms must be strictly positive")
        if int(self.semantic_wait_fail_ms) < int(self.semantic_wait_warn_ms):
            raise ValueError("semantic_wait_fail_ms must be >= semantic_wait_warn_ms")

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("ServoVLA requires `observation.state` in input_features.")
        for camera_key in self.camera_keys:
            if camera_key not in self.image_features:
                raise ValueError(f"Missing camera feature: {camera_key}")
        action_feature = self.action_feature
        if action_feature is None:
            raise ValueError("ServoVLA requires an `action` output feature.")
        if action_feature.shape[0] != self.action_dim:
            raise ValueError(
                f"`action_dim` ({self.action_dim}) must match action feature dim ({action_feature.shape[0]})."
            )
        if self.robot_state_feature.shape[0] != self.state_dim:
            raise ValueError(
                f"`state_dim` ({self.state_dim}) must match state feature dim ({self.robot_state_feature.shape[0]})."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
        )

    @property
    def observation_delta_indices(self) -> list[int] | None:
        return None

    @property
    def action_delta_indices(self) -> list[int] | None:
        if not self.action_is_delta:
            return None
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> list[int] | None:
        return None
