from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.async_inference.configs import PolicyServerConfig


@dataclass
class ServoVLAPolicyServerConfig(PolicyServerConfig):
    host: str = field(default="127.0.0.1")
    max_frame_delay: int | None = field(default=None)
    semantic_wait_warn_ms: int = field(default=500)
    semantic_wait_fail_ms: int = field(default=3000)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.max_frame_delay is not None and self.max_frame_delay < 0:
            raise ValueError(f"max_frame_delay must be >= 0, got {self.max_frame_delay}")
        if self.semantic_wait_warn_ms < 0:
            raise ValueError(
                f"semantic_wait_warn_ms must be >= 0, got {self.semantic_wait_warn_ms}"
            )
        if self.semantic_wait_fail_ms <= 0:
            raise ValueError(f"semantic_wait_fail_ms must be > 0, got {self.semantic_wait_fail_ms}")
        if self.semantic_wait_fail_ms < self.semantic_wait_warn_ms:
            raise ValueError(
                f"semantic_wait_fail_ms ({self.semantic_wait_fail_ms}) must be >= semantic_wait_warn_ms ({self.semantic_wait_warn_ms})"
            )
