from __future__ import annotations

from typing import Any

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize


def qwen_image_grid_thw_for_square(
    image_processor: Any, *, image_size: int
) -> tuple[int, int, int]:
    patch_size = int(getattr(image_processor, "patch_size", 14))
    merge_size = int(getattr(image_processor, "merge_size", 2))
    size = dict(getattr(image_processor, "size", {}) or {})
    min_pixels = int(size.get("shortest_edge", 56 * 56))
    max_pixels = int(size.get("longest_edge", 28 * 28 * 1280))
    resized_h, resized_w = smart_resize(
        int(image_size),
        int(image_size),
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return (1, int(resized_h) // patch_size, int(resized_w) // patch_size)


def qwen_image_token_count(grid_thw: tuple[int, int, int], *, merge_size: int) -> int:
    grid = torch.tensor(grid_thw, dtype=torch.long)
    return int(grid.prod().item() // (int(merge_size) ** 2))


def expand_qwen_image_tokens(
    texts: list[str],
    *,
    image_grid_thw: torch.Tensor,
    image_token: str,
    merge_size: int,
) -> list[str]:
    expanded = list(texts)
    image_index = 0
    for row, text in enumerate(expanded):
        while image_token in text:
            if image_index >= int(image_grid_thw.shape[0]):
                raise ValueError("Not enough image_grid_thw entries for Qwen image tokens")
            num_tokens = int(image_grid_thw[image_index].prod().item() // (int(merge_size) ** 2))
            text = text.replace(image_token, "<|placeholder|>" * num_tokens, 1)
            image_index += 1
        expanded[row] = text.replace("<|placeholder|>", image_token)
    if image_index != int(image_grid_thw.shape[0]):
        raise ValueError(
            f"Unused image_grid_thw entries: used={image_index}, total={int(image_grid_thw.shape[0])}"
        )
    return expanded
