from __future__ import annotations

import torch


def resolved_inference_seed(
    seed: int | None, *, seed_mode: str, action_step: int | None
) -> int | None:
    if seed is None:
        return None
    mode = str(seed_mode).lower()
    base_seed = int(seed)
    if mode == "fixed":
        return base_seed
    if mode == "step":
        return base_seed + int(0 if action_step is None else action_step)
    raise ValueError(f"Unsupported inference_noise_seed_mode: {seed_mode}")


def make_inference_noise(
    *,
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    seed: int | None,
    seed_mode: str,
    action_step: int | None,
) -> torch.Tensor:
    resolved_seed = resolved_inference_seed(seed, seed_mode=seed_mode, action_step=action_step)
    if resolved_seed is None:
        return torch.randn(shape, device=device, dtype=dtype)
    if device.type in {"cpu", "cuda"}:
        generator_device = device
    else:
        generator_device = torch.device("cpu")
    try:
        generator = torch.Generator(device=generator_device)
    except (RuntimeError, TypeError, ValueError):
        if generator_device == torch.device("cpu"):
            raise
        generator = torch.Generator(device=torch.device("cpu"))
    generator.manual_seed(int(resolved_seed))
    return torch.randn(shape, generator=generator, device=device, dtype=dtype)
