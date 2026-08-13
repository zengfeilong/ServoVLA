from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from servovla.data.qwen_image_tokens import qwen_image_grid_thw_for_square


@dataclass(frozen=True, slots=True)
class RawImagePreprocessSpec:
    vision_image_size: int
    vlm_image_size: int
    vision_image_mean: tuple[float, float, float]
    vision_image_std: tuple[float, float, float]
    vision_rescale_factor: float
    qwen_patch_size: int
    qwen_temporal_patch_size: int
    qwen_merge_size: int
    qwen_image_mean: tuple[float, float, float]
    qwen_image_std: tuple[float, float, float]
    qwen_rescale_factor: float
    qwen_smart_height: int
    qwen_smart_width: int


def _triple(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value), float(value))
    values = tuple(float(item) for item in value)
    if len(values) != 3:
        raise ValueError(f"Expected 3 channel values, got {values}")
    return values


def build_raw_image_preprocess_spec(
    *,
    vision_processor: Any,
    vlm_processor: Any,
    vision_image_size: int,
    vlm_image_size: int,
) -> RawImagePreprocessSpec:
    image_processor = vlm_processor.image_processor
    qwen_grid = qwen_image_grid_thw_for_square(image_processor, image_size=int(vlm_image_size))
    qwen_patch_size = int(getattr(image_processor, "patch_size", 14))
    return RawImagePreprocessSpec(
        vision_image_size=int(vision_image_size),
        vlm_image_size=int(vlm_image_size),
        vision_image_mean=_triple(
            getattr(vision_processor, "image_mean", None), (0.485, 0.456, 0.406)
        ),
        vision_image_std=_triple(
            getattr(vision_processor, "image_std", None), (0.229, 0.224, 0.225)
        ),
        vision_rescale_factor=float(getattr(vision_processor, "rescale_factor", 1.0 / 255.0)),
        qwen_patch_size=qwen_patch_size,
        qwen_temporal_patch_size=int(getattr(image_processor, "temporal_patch_size", 2)),
        qwen_merge_size=int(getattr(image_processor, "merge_size", 2)),
        qwen_image_mean=_triple(
            getattr(image_processor, "image_mean", None), (0.48145466, 0.4578275, 0.40821073)
        ),
        qwen_image_std=_triple(
            getattr(image_processor, "image_std", None), (0.26862954, 0.26130258, 0.27577711)
        ),
        qwen_rescale_factor=float(getattr(image_processor, "rescale_factor", 1.0 / 255.0)),
        qwen_smart_height=int(qwen_grid[1]) * qwen_patch_size,
        qwen_smart_width=int(qwen_grid[2]) * qwen_patch_size,
    )


class RawImageGpuPreprocessor:
    def __init__(self, *, spec: RawImagePreprocessSpec, device: torch.device) -> None:
        self.spec = spec
        self.device = device
        self._normalization_cache: dict[
            tuple[tuple[float, float, float], torch.dtype], torch.Tensor
        ] = {}

    def _move_group_images(self, group: Mapping[str, Any]) -> torch.Tensor:
        return group["images"].to(device=self.device, non_blocking=True)

    def _move_grouped_images_to_device(self, grouped: Mapping[str, Any]) -> dict[str, Any]:
        moved = dict(grouped)
        moved_groups = []
        for group in grouped["groups"]:
            moved_group = dict(group)
            moved_group["images"] = group["images"].to(device=self.device, non_blocking=True)
            moved_group["restore_indices"] = group["restore_indices"].to(
                device=self.device, non_blocking=True
            )
            moved_groups.append(moved_group)
        moved["groups"] = moved_groups
        return moved

    def move_raw_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        moved = dict(batch)
        moved["vision_images_uint8"] = self._move_grouped_images_to_device(
            batch["vision_images_uint8"]
        )
        moved["vlm_images_uint8"] = self._move_grouped_images_to_device(batch["vlm_images_uint8"])
        return moved

    def _normalization_tensor(
        self,
        values: tuple[float, float, float],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        key = (tuple(float(item) for item in values), dtype)
        cached = self._normalization_cache.get(key)
        if cached is None:
            cached = torch.tensor(key[0], device=self.device, dtype=dtype).view(1, 3, 1, 1)
            self._normalization_cache[key] = cached
        return cached

    def _normalize(
        self,
        images: torch.Tensor,
        *,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        rescale_factor: float,
    ) -> torch.Tensor:
        images = images.to(dtype=torch.float32).mul(float(rescale_factor))
        mean_tensor = self._normalization_tensor(mean, dtype=images.dtype).to(
            device=images.device,
            dtype=images.dtype,
        )
        std_tensor = self._normalization_tensor(std, dtype=images.dtype).to(
            device=images.device,
            dtype=images.dtype,
        )
        return (images - mean_tensor) / std_tensor

    def _resize_group(
        self,
        images: torch.Tensor,
        *,
        size: tuple[int, int],
        mode: str,
        quantize_uint8: bool = True,
    ) -> torch.Tensor:
        images = images.to(dtype=torch.float32)
        target_h, target_w = int(size[0]), int(size[1])
        current_h, current_w = int(images.shape[-2]), int(images.shape[-1])
        quantized = False
        if mode in {"bilinear", "bicubic"}:
            kwargs: dict[str, Any] = {
                "mode": mode,
                "align_corners": False,
                "antialias": True,
            }
            if current_w != target_w:
                images = F.interpolate(images, size=(current_h, target_w), **kwargs)
                if quantize_uint8:
                    images = images.round().clamp(0.0, 255.0)
                    quantized = True
            if current_h != target_h:
                images = F.interpolate(images, size=(target_h, target_w), **kwargs)
                if quantize_uint8:
                    images = images.round().clamp(0.0, 255.0)
                    quantized = True
        elif tuple(images.shape[-2:]) != (target_h, target_w):
            images = F.interpolate(images, size=(target_h, target_w), mode=mode)
            quantized = False
        if quantize_uint8 and not quantized:
            images = images.round().clamp(0.0, 255.0)
        return images

    def _nv12_to_rgb_chw(
        self,
        images: torch.Tensor,
        *,
        height: int,
        width: int,
    ) -> torch.Tensor:
        height = int(height)
        width = int(width)
        if images.ndim != 3:
            raise ValueError(
                f"NV12 groups must have shape (N, H*3/2, W), got {tuple(images.shape)}"
            )
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                f"NV12 RGB dimensions must be even, got height={height}, width={width}"
            )
        if int(images.shape[1]) != height + height // 2 or int(images.shape[2]) != width:
            raise ValueError(
                f"NV12 group shape {tuple(images.shape)} does not match height={height}, width={width}"
            )

        nv12 = images.to(dtype=torch.int32)
        y = nv12[:, :height, :width]
        uv = nv12[:, height : height + height // 2, :width]
        u = uv[:, :, 0::2].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        v = uv[:, :, 1::2].repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
        u = u[:, :height, :width]
        v = v[:, :height, :width]

        c = y - 16
        d = u - 128
        e = v - 128
        r = torch.div(298 * c + 409 * e + 128, 256, rounding_mode="floor")
        g = torch.div(298 * c - 100 * d - 208 * e + 128, 256, rounding_mode="floor")
        b = torch.div(298 * c + 516 * d + 128, 256, rounding_mode="floor")
        rgb = torch.stack((r, g, b), dim=1).clamp(0, 255)
        return rgb.to(dtype=torch.float32)

    def _group_images_to_rgb(self, images: torch.Tensor, group: Mapping[str, Any]) -> torch.Tensor:
        image_format = str(group.get("format", "rgb")).lower()
        if image_format == "rgb":
            return images
        if image_format == "nv12":
            return self._nv12_to_rgb_chw(
                images,
                height=int(group["height"]),
                width=int(group["width"]),
            )
        raise ValueError(f"Unsupported raw image group format {image_format!r}")

    def _restore_grouped(
        self,
        grouped: Mapping[str, Any],
        *,
        target_size: tuple[int, int],
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
        rescale_factor: float,
        resize_mode: str,
    ) -> torch.Tensor:
        batch_size = int(grouped["batch_size"])
        num_cameras = int(grouped["num_cameras"])
        output = torch.empty(
            batch_size,
            num_cameras,
            3,
            int(target_size[0]),
            int(target_size[1]),
            device=self.device,
            dtype=torch.float32,
        )
        for group in grouped["groups"]:
            images = self._move_group_images(group)
            images = self._group_images_to_rgb(images, group)
            images = self._resize_group(images, size=target_size, mode=resize_mode)
            images = self._normalize(images, mean=mean, std=std, rescale_factor=rescale_factor)
            restore_indices = group["restore_indices"].to(device=self.device, non_blocking=True)
            output[restore_indices[:, 0], restore_indices[:, 1]] = images
        return output

    def preprocess_vision(self, grouped: Mapping[str, Any]) -> torch.Tensor:
        size = (int(self.spec.vision_image_size), int(self.spec.vision_image_size))
        return self._restore_grouped(
            grouped,
            target_size=size,
            mean=self.spec.vision_image_mean,
            std=self.spec.vision_image_std,
            rescale_factor=self.spec.vision_rescale_factor,
            resize_mode="bilinear",
        )

    def preprocess_vlm_images(self, grouped: Mapping[str, Any]) -> torch.Tensor:
        square = (int(self.spec.vlm_image_size), int(self.spec.vlm_image_size))
        smart = (int(self.spec.qwen_smart_height), int(self.spec.qwen_smart_width))
        square_images = self._restore_grouped(
            grouped,
            target_size=square,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            rescale_factor=1.0,
            resize_mode="bilinear",
        )
        flat = square_images.reshape(
            -1, 3, int(self.spec.vlm_image_size), int(self.spec.vlm_image_size)
        )
        flat = self._resize_group(flat, size=smart, mode="bicubic", quantize_uint8=True)
        flat = self._normalize(
            flat,
            mean=self.spec.qwen_image_mean,
            std=self.spec.qwen_image_std,
            rescale_factor=self.spec.qwen_rescale_factor,
        )
        return self.patchify_qwen_images(flat)

    def patchify_qwen_images(self, images: torch.Tensor) -> torch.Tensor:
        patch_size = int(self.spec.qwen_patch_size)
        temporal_patch_size = int(self.spec.qwen_temporal_patch_size)
        merge_size = int(self.spec.qwen_merge_size)
        patches = images.unsqueeze(1)
        if patches.shape[1] % temporal_patch_size != 0:
            repeats = patches[:, -1:].repeat(
                1,
                temporal_patch_size - (patches.shape[1] % temporal_patch_size),
                1,
                1,
                1,
            )
            patches = torch.cat([patches, repeats], dim=1)
        batch_size, grid_t, channel = patches.shape[:3]
        grid_t = grid_t // temporal_patch_size
        grid_h = int(images.shape[-2]) // patch_size
        grid_w = int(images.shape[-1]) // patch_size
        patches = patches.view(
            batch_size,
            grid_t,
            temporal_patch_size,
            channel,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches = patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return patches.reshape(
            batch_size * grid_t * grid_h * grid_w,
            channel * temporal_patch_size * patch_size * patch_size,
        )

    def preprocess_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        processed = dict(batch)
        processed["pixel_values"] = self.preprocess_vision(batch["vision_images_uint8"])
        vlm_inputs = dict(batch["vlm_inputs"])
        vlm_inputs["pixel_values"] = self.preprocess_vlm_images(batch["vlm_images_uint8"])
        processed["vlm_inputs"] = vlm_inputs
        processed.pop("vision_images_uint8", None)
        processed.pop("vlm_images_uint8", None)
        return processed
