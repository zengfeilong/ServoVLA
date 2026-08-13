#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "src"
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from servovla.evaluation.sim_rollout_libero import evaluate_libero_pretrained

DEFAULT_EVAL_CFG = _PROJECT_ROOT / "configs" / "eval" / "sim_libero.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an exported ServoVLA pretrained policy with LIBERO rollouts."
    )
    parser.add_argument(
        "--pretrained-dir",
        required=True,
        help="Exported LeRobot pretrained ServoVLA policy directory.",
    )
    parser.add_argument(
        "--eval-cfg", default=str(DEFAULT_EVAL_CFG), help="Path to the simulation eval config YAML."
    )
    parser.add_argument("--device", default=None, help="Optional torch device override.")
    parser.add_argument(
        "--max-tasks", type=int, default=None, help="Optional cap for quick smoke evaluation."
    )
    parser.add_argument(
        "--semantic-wait-fail-ms",
        type=int,
        default=60000,
        help="Semantic runtime wait timeout for exported-policy rollout evaluation.",
    )
    parser.add_argument(
        "--replan-every-step",
        action="store_true",
        help="Reset the exported policy action queue before every environment step.",
    )
    parser.add_argument(
        "--output-json", default=None, help="Optional path to save the JSON summary."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_libero_pretrained(
        pretrained_dir=args.pretrained_dir,
        eval_cfg_path=args.eval_cfg,
        device=args.device,
        max_tasks=args.max_tasks,
        output_json=args.output_json,
        semantic_wait_fail_ms=args.semantic_wait_fail_ms,
        replan_every_step=args.replan_every_step,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
