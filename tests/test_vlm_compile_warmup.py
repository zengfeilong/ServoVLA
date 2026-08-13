from __future__ import annotations

import torch

from servovla.trainer.vlm_compile_warmup import (
    VlmInputSignature,
    VlmSignatureRegistry,
    handle_runtime_vlm_signatures,
    make_vlm_input_signature,
    make_vlm_input_signatures,
)


def _batch(seq_len: int, *, token_offset: int = 0, grid=None):
    if grid is None:
        grid = [[1, 18, 18], [1, 18, 18]]
    input_ids = torch.arange(2 * seq_len, dtype=torch.long).reshape(2, seq_len) + token_offset
    return {
        "vlm_inputs": {
            "input_ids": input_ids,
            "attention_mask": torch.ones(2, seq_len, dtype=torch.long),
            "image_grid_thw": torch.tensor(grid, dtype=torch.long),
        }
    }


def _batched_vlm_batch(batch_size: int, seq_len: int, *, grids_per_sample: int):
    grid = [[1, 32, 32] for _ in range(batch_size * grids_per_sample)]
    input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)
    return {
        "vlm_inputs": {
            "input_ids": input_ids,
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_grid_thw": torch.tensor(grid, dtype=torch.long),
        }
    }


def test_vlm_signature_ignores_token_values_but_tracks_shape_and_grid_values():
    first = make_vlm_input_signature(_batch(128, token_offset=0))
    second = make_vlm_input_signature(_batch(128, token_offset=1000))
    different_length = make_vlm_input_signature(_batch(144, token_offset=0))
    different_grid = make_vlm_input_signature(_batch(128, grid=[[1, 18, 18], [1, 16, 18]]))

    assert first == second
    assert first != different_length
    assert first != different_grid
    assert first.input_ids_shape == (2, 128)
    assert first.grid_values == ((1, 18, 18), (1, 18, 18))


def test_vlm_signature_includes_vlm_pixel_values_when_present():
    batch = _batch(128)
    batch["vlm_inputs"]["pixel_values"] = torch.zeros(648, 1176, dtype=torch.bfloat16)

    signature = make_vlm_input_signature(batch)

    assert signature.pixel_values_shape == (648, 1176)
    assert signature.pixel_values_dtype == "torch.bfloat16"


def test_vlm_micro_batch_signatures_use_effective_forward_shape():
    batch = _batched_vlm_batch(batch_size=64, seq_len=800, grids_per_sample=3)

    signatures = make_vlm_input_signatures(batch, micro_batch_size=8)

    assert len(signatures) == 1
    signature = signatures[0]
    assert signature.input_ids_shape == (8, 800)
    assert signature.attention_mask_shape == (8, 800)
    assert signature.image_grid_shape == (24, 3)


def test_runtime_vlm_signature_accepts_full_batch_when_micro_batch_is_warmed():
    registry = VlmSignatureRegistry(max_entries=4)
    warmup_batch = _batched_vlm_batch(batch_size=8, seq_len=800, grids_per_sample=3)
    train_batch = _batched_vlm_batch(batch_size=64, seq_len=800, grids_per_sample=3)
    for signature in make_vlm_input_signatures(warmup_batch, micro_batch_size=8):
        registry.mark_warmed(signature)

    handle_runtime_vlm_signatures(
        train_batch,
        registry=registry,
        on_new_signature="error",
        logger_name=__name__,
        micro_batch_size=8,
    )

    assert registry.seen_count == 1


def test_signature_registry_does_not_hardcode_signature_count():
    registry = VlmSignatureRegistry(max_entries=3)
    signatures = [
        VlmInputSignature((1, 64), (1, 64), (1, 3), ((1, 18, 18),), None, None),
        VlmInputSignature((1, 80), (1, 80), (1, 3), ((1, 18, 18),), None, None),
        VlmInputSignature((1, 96), (1, 96), (1, 3), ((1, 18, 18),), None, None),
    ]

    for signature in signatures:
        assert registry.mark_seen(signature) is True

    assert registry.mark_seen(signatures[0]) is False
    assert registry.seen_count == 3
