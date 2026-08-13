#!/usr/bin/env python

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
for _path in (_SOURCE_ROOT, _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from servovla.deployment.deploy_utils import (  # noqa: E402
    build_export_kwargs_from_training_cfg,
    export_checkpoint_to_pretrained,
    find_latest_checkpoint,
    write_deployment_bundle,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create pretrained/export deployment artifacts from a training run."
    )
    parser.add_argument(
        "--run-dir", required=True, help="Hydra output dir of a ServoVLA training run."
    )
    parser.add_argument("--checkpoint", default=None, help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--pretrained-dir", default=None, help="Optional explicit export directory."
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Export checkpoint model_state even when an ema_state is present.",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Export checkpoint ema_state when present.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()
    hydra_cfg_path = run_dir / ".hydra" / "config.yaml"
    if not hydra_cfg_path.exists():
        raise FileNotFoundError(f"Could not find Hydra config at {hydra_cfg_path}")

    cfg = OmegaConf.load(hydra_cfg_path)
    if bool(args.use_raw) and bool(args.use_ema):
        raise ValueError("--use-raw and --use-ema are mutually exclusive.")
    if bool(args.use_raw):
        OmegaConf.update(cfg, "deployment.prefer_ema", False, force_add=True)
    if bool(args.use_ema):
        OmegaConf.update(cfg, "deployment.prefer_ema", True, force_add=True)
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else find_latest_checkpoint(run_dir)
    )
    pretrained_dir = (
        Path(args.pretrained_dir).expanduser().resolve()
        if args.pretrained_dir
        else (
            _PROJECT_ROOT
            / str(cfg.deployment.get("pretrained_root", "artifacts/pretrained"))
            / run_dir.name
        ).resolve()
    )

    exported_dir = export_checkpoint_to_pretrained(
        checkpoint_path,
        pretrained_dir,
        prefer_ema=bool(cfg.deployment.get("prefer_ema", False)),
        config_overrides=build_export_kwargs_from_training_cfg(cfg),
    )
    print(f"Export complete: {exported_dir}")
    if bool(cfg.deployment.get("enabled", True)):
        deploy_dir = write_deployment_bundle(cfg, run_dir, checkpoint_path, exported_dir)
        print(f"Deployment bundle: {deploy_dir}")
    else:
        print("Deployment bundle skipped: deployment.enabled=false")


if __name__ == "__main__":
    main()
