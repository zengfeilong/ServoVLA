#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch


def _add_repo_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (repo_root / "src", repo_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_add_repo_paths()

from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

from lerobot_policy_servovla import ServoVLAConfig, ServoVLAPolicy  # noqa: E402
from servovla.deployment.deploy_utils import load_policy_state_dict_for_export  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a ServoVLA checkpoint to a LeRobot-style pretrained dir."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to ServoVLA .pt checkpoint.")
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write config/model/processors."
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to initialize the export policy on."
    )
    parser.add_argument("--vision-model-name", default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--vlm-model-name", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--front-camera-key", default="observation.images.front")
    parser.add_argument("--wrist-camera-key", default="observation.images.wrist")
    parser.add_argument("--camera-keys", nargs="+", default=None)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--n-action-steps", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--state-dim", type=int, default=6)
    parser.add_argument("--num-cameras", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--inference-noise-seed", type=int, default=None)
    parser.add_argument("--inference-noise-seed-mode", default="step", choices=["fixed", "step"])
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--vision-feature-dim", type=int, default=1024)
    parser.add_argument("--semantic-feature-dim", type=int, default=1024)
    parser.add_argument("--vision-image-size", type=int, default=256)
    parser.add_argument("--vlm-image-size", type=int, default=256)
    parser.add_argument("--vision-grid-size", type=int, default=16)
    parser.add_argument("--action-mode", choices=["abs", "delta"], default="abs")
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Export checkpoint ema_state when present. Defaults to raw model_state.",
    )
    parser.add_argument(
        "--use-raw",
        action="store_true",
        help="Export checkpoint model_state even when an ema_state is present.",
    )
    parser.add_argument(
        "--action-delta-state-indices",
        nargs="+",
        default=None,
        help="Per-action state index for delta restore; use null for direct command dimensions.",
    )
    return parser.parse_args()


def _checkpoint_action_normalization_kwargs(checkpoint) -> dict[str, object]:
    if not isinstance(checkpoint, dict):
        return {}
    action_norm = checkpoint.get("action_normalization")
    if not isinstance(action_norm, dict) or not bool(action_norm.get("enabled", False)):
        return {}
    mean = action_norm.get("mean")
    std = action_norm.get("std")
    if mean is None or std is None:
        return {}
    kwargs: dict[str, object] = {
        "action_normalization_enabled": True,
        "action_normalization_mean": mean,
        "action_normalization_std": std,
    }
    if "eps" in action_norm:
        kwargs["action_normalization_eps"] = float(action_norm["eps"])
    return kwargs


def build_config(args: argparse.Namespace, *, checkpoint=None) -> ServoVLAConfig:
    camera_keys = getattr(args, "camera_keys", None) or [
        args.front_camera_key,
        args.wrist_camera_key,
    ]
    action_norm_kwargs = _checkpoint_action_normalization_kwargs(checkpoint)
    action_delta_state_indices = getattr(args, "action_delta_state_indices", None)
    if action_delta_state_indices is not None:
        action_delta_state_indices = [
            None if str(item).strip().lower() in {"none", "null"} else int(item)
            for item in action_delta_state_indices
        ]
    return ServoVLAConfig(
        device=args.device,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        num_cameras=len(camera_keys),
        num_inference_steps=args.num_inference_steps,
        inference_noise_seed=getattr(args, "inference_noise_seed", None),
        inference_noise_seed_mode=getattr(args, "inference_noise_seed_mode", "step"),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        vision_feature_dim=args.vision_feature_dim,
        semantic_feature_dim=args.semantic_feature_dim,
        vision_image_size=args.vision_image_size,
        vlm_image_size=args.vlm_image_size,
        vision_grid_size=args.vision_grid_size,
        vision_model_name=args.vision_model_name,
        vlm_model_name=args.vlm_model_name,
        camera_keys=list(camera_keys),
        action_mode=getattr(args, "action_mode", "abs"),
        action_delta_state_indices=action_delta_state_indices,
        **action_norm_kwargs,
    )


def main() -> None:
    args = parse_args()
    if bool(args.use_raw) and bool(args.use_ema):
        raise ValueError("--use-raw and --use-ema are mutually exclusive.")
    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and bool(args.use_ema) and "ema_state" in checkpoint:
        state_dict = checkpoint["ema_state"]
    elif isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Could not locate a state_dict in the checkpoint.")

    cfg = build_config(args, checkpoint=checkpoint)
    policy = ServoVLAPolicy(cfg)
    load_policy_state_dict_for_export(policy, state_dict)

    preprocessor, postprocessor = make_pre_post_processors(cfg)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)

    print(f"Exported ServoVLA policy to: {output_dir}")
    print("Files:")
    for name in sorted(os.listdir(output_dir)):
        print(f"  - {name}")


if __name__ == "__main__":
    main()
