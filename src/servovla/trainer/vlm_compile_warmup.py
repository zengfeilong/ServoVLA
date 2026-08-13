from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VlmInputSignature:
    input_ids_shape: tuple[int, ...]
    attention_mask_shape: tuple[int, ...] | None
    image_grid_shape: tuple[int, ...] | None
    grid_values: tuple[tuple[int, int, int], ...]
    pixel_values_shape: tuple[int, ...] | None
    pixel_values_dtype: str | None


def _shape(value: Any) -> tuple[int, ...] | None:
    return tuple(int(dim) for dim in value.shape) if torch.is_tensor(value) else None


def _grid_values(value: Any) -> tuple[tuple[int, int, int], ...]:
    if not torch.is_tensor(value):
        return ()
    grid_cpu = value.detach().to("cpu", dtype=torch.long)
    if grid_cpu.ndim != 2 or grid_cpu.shape[1] != 3:
        return ()
    return tuple(tuple(int(cell) for cell in row.tolist()) for row in grid_cpu)


def make_vlm_input_signature(batch: Mapping[str, Any]) -> VlmInputSignature:
    vlm_inputs = batch.get("vlm_inputs", {})
    if not isinstance(vlm_inputs, Mapping):
        raise TypeError("batch['vlm_inputs'] must be a mapping")
    input_ids = vlm_inputs.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise TypeError("batch['vlm_inputs']['input_ids'] must be a tensor")
    pixel_values = vlm_inputs.get("pixel_values")
    return VlmInputSignature(
        input_ids_shape=tuple(int(dim) for dim in input_ids.shape),
        attention_mask_shape=_shape(vlm_inputs.get("attention_mask")),
        image_grid_shape=_shape(vlm_inputs.get("image_grid_thw")),
        grid_values=_grid_values(vlm_inputs.get("image_grid_thw")),
        pixel_values_shape=_shape(pixel_values),
        pixel_values_dtype=str(pixel_values.dtype) if torch.is_tensor(pixel_values) else None,
    )


def _unique_signatures(signatures: list[VlmInputSignature]) -> tuple[VlmInputSignature, ...]:
    unique: list[VlmInputSignature] = []
    seen: set[VlmInputSignature] = set()
    for signature in signatures:
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(signature)
    return tuple(unique)


def _slice_vlm_inputs_for_signature(
    vlm_inputs: Mapping[str, Any],
    *,
    sample_start: int,
    sample_end: int,
    batch_size: int,
    image_rows_per_sample: int | None,
    pixel_rows_per_sample: int | None,
) -> dict[str, Any]:
    chunk_inputs: dict[str, Any] = {}
    for key, value in vlm_inputs.items():
        if not torch.is_tensor(value) or value.ndim == 0:
            chunk_inputs[key] = value
        elif key == "image_grid_thw" and image_rows_per_sample is not None:
            row_start = sample_start * image_rows_per_sample
            row_end = sample_end * image_rows_per_sample
            chunk_inputs[key] = value[row_start:row_end]
        elif key == "pixel_values" and pixel_rows_per_sample is not None:
            row_start = sample_start * pixel_rows_per_sample
            row_end = sample_end * pixel_rows_per_sample
            chunk_inputs[key] = value[row_start:row_end]
        elif int(value.shape[0]) == batch_size:
            chunk_inputs[key] = value[sample_start:sample_end]
        else:
            chunk_inputs[key] = value
    return chunk_inputs


def make_vlm_input_signatures(
    batch: Mapping[str, Any],
    *,
    micro_batch_size: int = 0,
) -> tuple[VlmInputSignature, ...]:
    vlm_inputs = batch.get("vlm_inputs", {})
    if not isinstance(vlm_inputs, Mapping):
        raise TypeError("batch['vlm_inputs'] must be a mapping")
    input_ids = vlm_inputs.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise TypeError("batch['vlm_inputs']['input_ids'] must be a tensor")

    batch_size = int(input_ids.shape[0]) if input_ids.ndim > 0 else 0
    chunk_size = int(micro_batch_size)
    if batch_size <= 0 or chunk_size <= 0 or chunk_size >= batch_size:
        return (make_vlm_input_signature(batch),)

    image_grid_thw = vlm_inputs.get("image_grid_thw")
    image_rows_per_sample: int | None = None
    if torch.is_tensor(image_grid_thw):
        if image_grid_thw.ndim == 0 or int(image_grid_thw.shape[0]) % batch_size != 0:
            return (make_vlm_input_signature(batch),)
        image_rows_per_sample = int(image_grid_thw.shape[0]) // batch_size

    pixel_values = vlm_inputs.get("pixel_values")
    pixel_rows_per_sample: int | None = None
    if torch.is_tensor(pixel_values):
        if pixel_values.ndim == 0 or int(pixel_values.shape[0]) % batch_size != 0:
            return (make_vlm_input_signature(batch),)
        pixel_rows_per_sample = int(pixel_values.shape[0]) // batch_size

    signatures: list[VlmInputSignature] = []
    for sample_start in range(0, batch_size, chunk_size):
        sample_end = min(sample_start + chunk_size, batch_size)
        chunk_inputs = _slice_vlm_inputs_for_signature(
            vlm_inputs,
            sample_start=sample_start,
            sample_end=sample_end,
            batch_size=batch_size,
            image_rows_per_sample=image_rows_per_sample,
            pixel_rows_per_sample=pixel_rows_per_sample,
        )
        signatures.append(make_vlm_input_signature({"vlm_inputs": chunk_inputs}))
    return _unique_signatures(signatures)


class VlmSignatureRegistry:
    def __init__(self, *, max_entries: int) -> None:
        self.max_entries = max(int(max_entries), 1)
        self._seen: set[VlmInputSignature] = set()
        self._warmed: set[VlmInputSignature] = set()

    @property
    def seen_count(self) -> int:
        return len(self._seen)

    @property
    def warmed_count(self) -> int:
        return len(self._warmed)

    def mark_seen(self, signature: VlmInputSignature) -> bool:
        if signature in self._seen:
            return False
        if len(self._seen) >= self.max_entries:
            raise RuntimeError(
                f"VLM signature limit exceeded: max_entries={self.max_entries}, new_signature={signature!r}"
            )
        self._seen.add(signature)
        return True

    def mark_warmed(self, signature: VlmInputSignature) -> bool:
        self.mark_seen(signature)
        if signature in self._warmed:
            return False
        self._warmed.add(signature)
        return True

    def is_warmed(self, signature: VlmInputSignature) -> bool:
        return signature in self._warmed


def handle_runtime_vlm_signature(
    batch: Mapping[str, Any],
    *,
    registry: VlmSignatureRegistry,
    on_new_signature: str,
    logger_name: str,
) -> VlmInputSignature:
    signature = make_vlm_input_signature(batch)
    if registry.is_warmed(signature):
        return signature

    policy = str(on_new_signature)
    if policy == "error":
        raise RuntimeError(f"Unexpected VLM input signature after warmup: {signature!r}")

    is_new = registry.mark_seen(signature)
    if is_new:
        logging.getLogger(logger_name).warning(
            "Observed new VLM input signature after warmup; policy=%s | %s",
            policy,
            signature,
        )

    if policy == "warn_and_warmup":
        registry.mark_warmed(signature)
    elif policy == "warn_only":
        pass
    else:
        raise ValueError(
            "training.vlm_compile.on_new_signature must be one of: warn_and_warmup, warn_only, error"
        )
    return signature


def handle_runtime_vlm_signatures(
    batch: Mapping[str, Any],
    *,
    registry: VlmSignatureRegistry,
    on_new_signature: str,
    logger_name: str,
    micro_batch_size: int = 0,
) -> tuple[VlmInputSignature, ...]:
    signatures = make_vlm_input_signatures(batch, micro_batch_size=int(micro_batch_size))
    for signature in signatures:
        if registry.is_warmed(signature):
            continue

        policy = str(on_new_signature)
        if policy == "error":
            raise RuntimeError(f"Unexpected VLM input signature after warmup: {signature!r}")

        is_new = registry.mark_seen(signature)
        if is_new:
            logging.getLogger(logger_name).warning(
                "Observed new VLM input signature after warmup; policy=%s | %s",
                policy,
                signature,
            )

        if policy == "warn_and_warmup":
            registry.mark_warmed(signature)
        elif policy == "warn_only":
            pass
        else:
            raise ValueError(
                "training.vlm_compile.on_new_signature must be one of: warn_and_warmup, warn_only, error"
            )
    return signatures
