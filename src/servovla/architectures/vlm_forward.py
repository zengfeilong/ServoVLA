from __future__ import annotations

import threading
from typing import Any

import torch
import torch.nn as nn

_VLM_FORWARD_LOCK = threading.RLock()


def run_vlm_encoder_forward(vlm_encoder: nn.Module, vlm_inputs: dict[str, Any]) -> torch.Tensor:
    with _VLM_FORWARD_LOCK:
        return vlm_encoder(vlm_inputs)
