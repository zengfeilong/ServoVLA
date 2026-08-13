from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "evaluate_libero_run",
    "resolve_checkpoint_path",
    "summarize_libero_rollouts",
]

_LIBERO_EXPORTS = {
    "evaluate_libero_run",
    "summarize_libero_rollouts",
}


def __getattr__(name: str) -> Any:
    if name in _LIBERO_EXPORTS:
        module = import_module("servovla.evaluation.sim_rollout_libero")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
