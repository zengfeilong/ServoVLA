from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PaddedSampleIndex:
    index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", int(self.index))

    def __int__(self) -> int:
        return int(self.index)


def is_padded_sample_index(index: Any) -> bool:
    return isinstance(index, PaddedSampleIndex)


def sample_index_value(index: Any) -> int:
    if isinstance(index, PaddedSampleIndex):
        return int(index.index)
    return int(index)
