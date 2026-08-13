from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
_OFFLINE_KEYS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")


def hf_offline_enabled() -> bool:
    return any(
        os.environ.get(key, "").strip().lower() in {"1", "true", "yes", "on"}
        for key in _OFFLINE_KEYS
    )


def configure_hf_offline_env() -> bool:
    if not hf_offline_enabled():
        return False

    for key, value in _OFFLINE_ENV.items():
        os.environ[key] = value

    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        setattr(hub_constants, "HF_HUB_OFFLINE", True)
        setattr(hub_constants, "HF_HUB_DISABLE_TELEMETRY", True)

    datasets_config = sys.modules.get("datasets.config")
    if datasets_config is not None:
        setattr(datasets_config, "HF_DATASETS_OFFLINE", True)
        setattr(datasets_config, "HF_HUB_OFFLINE", True)
    return True


def hf_from_pretrained_kwargs(**kwargs: Any) -> dict[str, Any]:
    if configure_hf_offline_env():
        return {**kwargs, "local_files_only": True}
    return kwargs


def require_lerobot_local_dataset_root(
    repo_id: str,
    *,
    root: str | Path | None = None,
    hf_lerobot_home: str | Path | None = None,
) -> Path | None:
    offline = configure_hf_offline_env()
    if root is not None:
        resolved = Path(root).expanduser()
    else:
        if hf_lerobot_home is None:
            env_hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
            if env_hf_lerobot_home:
                hf_lerobot_home = env_hf_lerobot_home
        if hf_lerobot_home is None and not offline:
            return None
        if hf_lerobot_home is None:
            try:
                from lerobot.utils.constants import HF_LEROBOT_HOME
            except Exception:
                hf_lerobot_home = None
            else:
                hf_lerobot_home = HF_LEROBOT_HOME
        if hf_lerobot_home is None:
            return None
        resolved = Path(hf_lerobot_home).expanduser() / repo_id

    info_path = resolved / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(
            "A configured local LeRobot dataset is missing. "
            f"repo_id={repo_id!r}, expected metadata at {info_path}"
        )
    return resolved
