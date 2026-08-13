from __future__ import annotations

import importlib
import importlib.metadata as metadata
import inspect


def test_installed_lerobot_version_is_051():
    assert metadata.version("lerobot") == "0.5.1"


def test_lerobot_051_import_surface_used_by_servovla_is_available():
    modules = [
        "lerobot.datasets.lerobot_dataset",
        "lerobot.envs.configs",
        "lerobot.envs.factory",
        "lerobot.envs.utils",
        "lerobot.processor",
        "lerobot.policies.factory",
        "lerobot.async_inference.helpers",
        "lerobot.async_inference.configs",
        "lerobot.transport.utils",
        "lerobot.transport.services_pb2",
        "lerobot.transport.services_pb2_grpc",
        "lerobot_policy_servovla.configuration_servovla",
        "lerobot_policy_servovla.modeling_servovla",
        "lerobot_policy_servovla.processor_servovla",
        "lerobot_policy_servovla.server_config_servovla",
        "lerobot_policy_servovla.async_policy_server",
        "scripts.export_lerobot_pretrained",
    ]

    for module_name in modules:
        importlib.import_module(module_name)


def test_lerobot_051_dataset_and_processor_signatures_cover_servovla_usage():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline

    dataset_params = inspect.signature(LeRobotDataset.__init__).parameters
    for name in [
        "repo_id",
        "root",
        "episodes",
        "image_transforms",
        "delta_timestamps",
        "revision",
        "download_videos",
        "video_backend",
    ]:
        assert name in dataset_params

    processor_params = inspect.signature(make_pre_post_processors).parameters
    assert "policy_cfg" in processor_params
    assert "pretrained_path" in processor_params

    pipeline_params = inspect.signature(PolicyProcessorPipeline.__init__).parameters
    assert "origin" in pipeline_params
    assert "args" in pipeline_params
