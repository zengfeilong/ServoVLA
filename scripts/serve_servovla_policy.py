#!/usr/bin/env python

from __future__ import annotations

import sys
from pathlib import Path


def _add_plugin_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))


_add_plugin_path()

from lerobot_policy_servovla.async_policy_server import serve  # noqa: E402

if __name__ == "__main__":
    serve()
