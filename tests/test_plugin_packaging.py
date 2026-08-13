from __future__ import annotations

import importlib.util
from pathlib import Path


def test_lerobot_policy_servovla_is_merged_into_servovla_repo() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_package_dir = repo_root / "src" / "lerobot_policy_servovla"

    spec = importlib.util.find_spec("lerobot_policy_servovla")

    assert spec is not None
    assert spec.origin is not None
    assert Path(spec.origin).resolve().parent == expected_package_dir
