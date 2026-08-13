from __future__ import annotations

import pytest

from servovla.architectures.policy_head import FlowMatchingDiT

D_VISION = 64
D_SEM = 32
ACTION_DIM = 6
STATE_DIM = 6
CHUNK_SIZE = 8
NUM_CAMERAS = 2
VISION_SEQ_PER_CAMERA = 16
VISION_SEQ = NUM_CAMERAS * VISION_SEQ_PER_CAMERA

HIDDEN_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2


@pytest.fixture
def policy_head() -> FlowMatchingDiT:
    return FlowMatchingDiT(
        action_dim=ACTION_DIM,
        state_dim=STATE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        vision_feature_dim=D_VISION,
        semantic_feature_dim=D_SEM,
        vision_grid_size=4,
        num_cameras=NUM_CAMERAS,
    )
