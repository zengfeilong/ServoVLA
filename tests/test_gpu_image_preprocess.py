from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from servovla.trainer.gpu_image_preprocess import RawImageGpuPreprocessor, RawImagePreprocessSpec


def _raw_group(images, restore_indices):
    return {
        "height": int(images.shape[-2]),
        "width": int(images.shape[-1]),
        "images": images,
        "restore_indices": torch.tensor(restore_indices, dtype=torch.long),
    }


def test_restore_grouped_uint8_images_to_batch_camera_order_on_cpu():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0 / 255.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0 / 255.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))
    groups = {
        "batch_size": 2,
        "num_cameras": 2,
        "groups": [
            _raw_group(
                torch.stack(
                    [
                        torch.full((3, 4, 4), 1, dtype=torch.uint8),
                        torch.full((3, 4, 4), 4, dtype=torch.uint8),
                    ],
                    dim=0,
                ),
                [[0, 0], [1, 1]],
            ),
            _raw_group(
                torch.stack(
                    [
                        torch.full((3, 2, 2), 2, dtype=torch.uint8),
                        torch.full((3, 2, 2), 3, dtype=torch.uint8),
                    ],
                    dim=0,
                ),
                [[0, 1], [1, 0]],
            ),
        ],
    }

    restored = preprocessor.preprocess_vision(groups)

    assert restored.shape == (2, 2, 3, 4, 4)
    assert torch.allclose(restored[0, 0], torch.full((3, 4, 4), 1 / 255.0))
    assert torch.allclose(restored[0, 1], torch.full((3, 4, 4), 2 / 255.0))
    assert torch.allclose(restored[1, 0], torch.full((3, 4, 4), 3 / 255.0))
    assert torch.allclose(restored[1, 1], torch.full((3, 4, 4), 4 / 255.0))


def test_preprocess_vision_matches_pil_bilinear_resize_and_uint8_quantization_on_cpu():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0 / 255.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0 / 255.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))
    raw = torch.arange(3 * 5 * 7, dtype=torch.uint8).reshape(1, 3, 5, 7)
    groups = {
        "batch_size": 1,
        "num_cameras": 1,
        "groups": [_raw_group(raw, [[0, 0]])],
    }
    pil_image = Image.fromarray(raw[0].permute(1, 2, 0).numpy())
    expected = torch.from_numpy(
        np.asarray(pil_image.resize((4, 4), Image.Resampling.BILINEAR)).copy()
    ).permute(2, 0, 1)
    expected = expected.unsqueeze(0).unsqueeze(0).to(dtype=torch.float32).div(255.0)

    restored = preprocessor.preprocess_vision(groups)

    diff = (restored.float().cpu() - expected.float()).abs()
    assert float(diff.mean().item()) < 1.0e-6
    assert float(diff.max().item()) < 1.0e-6


def test_preprocess_vision_converts_nv12_groups_to_rgb_on_device():
    spec = RawImagePreprocessSpec(
        vision_image_size=2,
        vlm_image_size=2,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0,
        qwen_patch_size=1,
        qwen_temporal_patch_size=1,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0,
        qwen_smart_height=2,
        qwen_smart_width=2,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))
    y = torch.tensor([[16, 235], [16, 235]], dtype=torch.uint8)
    uv = torch.tensor([[128, 128]], dtype=torch.uint8)
    nv12 = torch.cat([y, uv], dim=0).unsqueeze(0)
    groups = {
        "batch_size": 1,
        "num_cameras": 1,
        "groups": [
            {
                "format": "nv12",
                "height": 2,
                "width": 2,
                "images": nv12,
                "restore_indices": torch.tensor([[0, 0]], dtype=torch.long),
            }
        ],
    }

    restored = preprocessor.preprocess_vision(groups)

    expected = torch.tensor(
        [
            [
                [
                    [[0.0, 255.0], [0.0, 255.0]],
                    [[0.0, 255.0], [0.0, 255.0]],
                    [[0.0, 255.0], [0.0, 255.0]],
                ]
            ]
        ]
    )
    assert torch.allclose(restored, expected)


def test_raw_preprocessor_caches_normalization_stat_tensors_on_device():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.1, 0.2, 0.3),
        vision_image_std=(0.4, 0.5, 0.6),
        vision_rescale_factor=1.0 / 255.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0 / 255.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))

    first = preprocessor._normalization_tensor(spec.vision_image_mean)
    second = preprocessor._normalization_tensor(spec.vision_image_mean)

    assert first is second
    assert first.device.type == "cpu"
    assert first.shape == (1, 3, 1, 1)


def test_qwen_patchify_order_matches_transformers_fast_layout_on_cpu():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)

    patches = preprocessor.patchify_qwen_images(images)

    expected = (
        images.unsqueeze(1)
        .repeat(1, 2, 1, 1, 1)
        .view(2, 1, 2, 3, 2, 1, 2, 2, 1, 2)
        .permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        .reshape(8, 24)
    )
    assert patches.shape == (8, 24)
    assert torch.equal(patches, expected)


def test_preprocess_raw_batch_injects_gpu_pixel_values_without_cpu_vlm_pixels():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0 / 255.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0 / 255.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cpu"))
    raw_group = {
        "batch_size": 1,
        "num_cameras": 1,
        "groups": [_raw_group(torch.full((1, 3, 4, 4), 8, dtype=torch.uint8), [[0, 0]])],
    }
    batch = {
        "vision_images_uint8": raw_group,
        "vlm_images_uint8": raw_group,
        "vlm_inputs": {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        },
        "action": torch.zeros(1, 4, 2),
        "loss_mask": torch.ones(1, 4),
        "c_sem_mask": torch.ones(1, 4, dtype=torch.bool),
        "q_current": torch.zeros(1, 3),
        "frame_delay": torch.zeros(1),
        "dataset_slug": ["unit"],
    }

    processed = preprocessor.preprocess_batch(batch)

    assert processed["pixel_values"].shape == (1, 1, 3, 4, 4)
    assert processed["vlm_inputs"]["pixel_values"].shape == (4, 24)
    assert processed["vlm_inputs"]["input_ids"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_raw_preprocessor_outputs_cuda_tensors():
    spec = RawImagePreprocessSpec(
        vision_image_size=4,
        vlm_image_size=4,
        vision_image_mean=(0.0, 0.0, 0.0),
        vision_image_std=(1.0, 1.0, 1.0),
        vision_rescale_factor=1.0 / 255.0,
        qwen_patch_size=2,
        qwen_temporal_patch_size=2,
        qwen_merge_size=1,
        qwen_image_mean=(0.0, 0.0, 0.0),
        qwen_image_std=(1.0, 1.0, 1.0),
        qwen_rescale_factor=1.0 / 255.0,
        qwen_smart_height=4,
        qwen_smart_width=4,
    )
    preprocessor = RawImageGpuPreprocessor(spec=spec, device=torch.device("cuda"))
    raw_group = {
        "batch_size": 1,
        "num_cameras": 1,
        "groups": [_raw_group(torch.full((1, 3, 4, 4), 8, dtype=torch.uint8), [[0, 0]])],
    }

    output = preprocessor.preprocess_vision(raw_group)

    assert output.is_cuda
