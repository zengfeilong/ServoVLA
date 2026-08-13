from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_load_libero_10_protocol_returns_ordered_tasks():
    from servovla.data.sim_libero_adapter import LiberoTaskSpec, load_libero_10_protocol

    protocol = load_libero_10_protocol()

    assert len(protocol.tasks) == 10
    assert protocol.tasks[0] == LiberoTaskSpec(
        suite="libero_10",
        suite_task_id=0,
        task_index=5,
        task_name="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        language="put both the alphabet soup and the tomato sauce in the basket",
    )
    assert protocol.tasks[-1].suite_task_id == 9
    assert protocol.tasks[-1].task_index == 2
    assert (
        protocol.tasks[-1].language == "put the yellow and white mug in the microwave and close it"
    )


def test_load_libero_40_protocol_returns_fixed_four_suite_tasks():
    from servovla.data.sim_libero_adapter import load_libero_40_protocol, load_libero_protocol

    protocol = load_libero_40_protocol()
    default_protocol = load_libero_protocol()

    assert default_protocol == protocol
    assert protocol.suite == "libero_40"
    assert protocol.suites == ("libero_10", "libero_spatial", "libero_object", "libero_goal")
    assert len(protocol.tasks) == 40
    assert [task.task_index for task in protocol.tasks] == list(range(40))
    assert {task.suite for task in protocol.tasks} == {
        "libero_10",
        "libero_spatial",
        "libero_object",
        "libero_goal",
    }
    assert protocol.tasks[0].suite == "libero_10"
    assert protocol.tasks[10].suite == "libero_spatial"
    assert protocol.tasks[20].suite == "libero_object"
    assert protocol.tasks[30].suite == "libero_goal"


def test_sim_libero_dataset_cfg_is_dual_camera_abs_controller_mode():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "dataset" / "sim_libero.yaml")

    assert cfg.train_names == ["HuggingFaceVLA/libero"]
    assert cfg.benchmark == "libero_40"
    assert cfg.action_mode == "abs"
    assert cfg.action_delta_state_indices is None
    assert cfg.action_dim == 7
    assert cfg.num_cameras == 2
    assert list(cfg.camera_keys) == ["observation.images.image", "observation.images.image2"]
    assert cfg.proprio_dim == 8
    assert list(cfg.task_index_allowlist) == list(range(40))


def test_sim_libero_deployment_cfg_disables_auto_export():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "deployment" / "sim_libero.yaml")

    assert cfg.enabled is False
    assert cfg.auto_export is False


def test_libero_rollout_uses_relative_controller(monkeypatch):
    import servovla.evaluation.sim_rollout_libero as module

    captured = {}

    class FakePolicy:
        def select_action(self, observation):
            batch_size = int(observation["observation.state"].shape[0])
            return torch.zeros(batch_size, 7).numpy()

    class FakeEnv:
        num_envs = 1

        def reset(self, seed=None):
            return {"stub": True}, {}

        def call(self, name):
            assert name == "_max_episode_steps"
            return [0]

        def close(self):
            pass

    def fake_make_env(env_cfg, *, n_envs, use_async_envs):
        captured["task"] = env_cfg.task
        captured["task_ids"] = list(env_cfg.task_ids)
        captured["control_mode"] = env_cfg.control_mode
        captured["n_envs"] = n_envs
        captured["use_async_envs"] = use_async_envs
        return {"libero_spatial": {2: FakeEnv()}}

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "close_rollout_env", lambda _env: None)

    module._rollout_task(
        suite="libero_40",
        task_spec=SimpleNamespace(suite="libero_spatial", suite_task_id=2),
        policy=FakePolicy(),
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        n_episodes=1,
        use_async_envs=False,
        seed=1234,
    )

    assert captured["task"] == "libero_spatial"
    assert captured["task_ids"] == [2]
    assert captured["control_mode"] == "relative"


def test_libero_rollout_can_replan_every_step(monkeypatch):
    import servovla.evaluation.sim_rollout_libero as module

    class FakePolicy:
        def __init__(self):
            self.reset_calls = 0
            self.select_calls = 0

        def reset(self):
            self.reset_calls += 1

        def select_action(self, observation):
            self.select_calls += 1
            batch_size = int(observation["observation.state"].shape[0])
            return torch.zeros(batch_size, 7).numpy()

    class FakeEnv:
        num_envs = 1

        def __init__(self):
            self.step_calls = 0

        def reset(self, seed=None):
            return {"stub": True}, {}

        def call(self, name):
            assert name == "_max_episode_steps"
            return [2]

        def step(self, action):
            self.step_calls += 1
            terminated = torch.tensor([self.step_calls >= 2]).numpy()
            truncated = torch.tensor([False]).numpy()
            info = {"final_info": {"is_success": torch.tensor([True]).numpy()}}
            return {"stub": True}, None, terminated, truncated, info

        def close(self):
            pass

    fake_env = FakeEnv()

    def fake_make_env(env_cfg, *, n_envs, use_async_envs):
        del env_cfg, n_envs, use_async_envs
        return {"libero_10": {0: fake_env}}

    monkeypatch.setattr(module, "make_env", fake_make_env)
    monkeypatch.setattr(module, "close_rollout_env", lambda _env: None)
    monkeypatch.setattr(
        module,
        "_preprocess_libero_observation",
        lambda _observation: {"observation.state": torch.zeros(1, 8)},
    )
    monkeypatch.setattr(
        module,
        "add_envs_task",
        lambda _env, observation: {**observation, "task": ["demo task"]},
    )

    policy = FakePolicy()
    successes, lengths, runtime_metrics = module._rollout_task(
        suite="libero_10",
        task_spec=SimpleNamespace(suite_task_id=0),
        policy=policy,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        camera_name="agentview_image,robot0_eye_in_hand_image",
        n_episodes=1,
        use_async_envs=False,
        seed=1234,
        replan_every_step=True,
    )

    assert successes == [True]
    assert lengths == [2]
    assert len(runtime_metrics["chunk_latency_ms"]) == 2
    assert policy.select_calls == 2
    assert policy.reset_calls == 2


def test_validate_libero_item_accepts_expected_contract():
    from servovla.data.sim_libero_adapter import validate_libero_item

    item = {
        "task": "put both the alphabet soup and the tomato sauce in the basket",
        "task_index": 5,
        "observation.images.image": torch.zeros(3, 64, 64),
        "observation.images.image2": torch.zeros(3, 64, 64),
        "observation.state": torch.zeros(8),
        "action": torch.zeros(16, 7),
    }

    validate_libero_item(item)


def test_validate_libero_item_rejects_missing_wrist_camera_key():
    from servovla.data.sim_libero_adapter import validate_libero_item

    item = {
        "task": "put both the alphabet soup and the tomato sauce in the basket",
        "task_index": 5,
        "observation.images.image": torch.zeros(3, 64, 64),
        "observation.state": torch.zeros(8),
        "action": torch.zeros(16, 7),
    }

    with pytest.raises(KeyError):
        validate_libero_item(item)


def test_balanced_success_rate_averages_per_task_success():
    from servovla.evaluation.sim_rollout_libero import summarize_libero_rollouts

    summary = summarize_libero_rollouts(
        {
            "task_a": [True, False],
            "task_b": [True, True],
        }
    )

    assert summary["overall_balanced_success_rate"] == pytest.approx(0.75)
    assert summary["per_task_success_rate"] == {
        "task_a": pytest.approx(0.5),
        "task_b": pytest.approx(1.0),
    }


def test_libero_duration_steps_are_reported_as_control_seconds():
    from servovla.evaluation.sim_rollout_libero import summarize_libero_rollouts

    summary = summarize_libero_rollouts(
        {
            "task_a": [True, False],
            "task_b": [True, True],
        },
        {
            "task_a": [30, 60],
            "task_b": [90, 120],
        },
        duration_fps=30,
    )

    assert summary["duration_fps"] == pytest.approx(30)
    assert summary["per_task_duration_steps"] == {
        "task_a": pytest.approx(45),
        "task_b": pytest.approx(105),
    }
    assert summary["per_task_duration_seconds"] == {
        "task_a": pytest.approx(1.5),
        "task_b": pytest.approx(3.5),
    }
    assert summary["top_failures"][0]["mean_duration_seconds"] == pytest.approx(1.5)
    assert "per_task_success_duration_steps" not in summary
    assert "per_task_success_duration_seconds" not in summary
    assert "mean_success_duration_steps" not in summary["top_failures"][0]
    assert "mean_success_duration_seconds" not in summary["top_failures"][0]


def test_resolve_checkpoint_prefers_latest_when_not_provided(tmp_path):
    from servovla.evaluation.sim_rollout_libero import resolve_checkpoint_path

    (tmp_path / "checkpoint_step0000002.pt").write_bytes(b"x")
    (tmp_path / "checkpoint_step0000010.pt").write_bytes(b"y")

    assert resolve_checkpoint_path(tmp_path, None).name == "checkpoint_step0000010.pt"


def test_checkpoint_action_normalizer_restores_saved_libero_stats(tmp_path):
    from servovla.evaluation.sim_rollout_libero import _checkpoint_action_normalizer

    mean = torch.arange(16 * 7, dtype=torch.float32).view(16, 7)
    std = torch.full((16, 7), 2.0)
    checkpoint = {
        "action_normalization": {
            "enabled": True,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "eps": 1.0e-5,
        }
    }

    normalizer = _checkpoint_action_normalizer(
        checkpoint,
        checkpoint_path=tmp_path / "checkpoint.pt",
        action_dim=7,
        chunk_size=16,
    )

    assert normalizer.enabled is True
    assert normalizer.eps == pytest.approx(1.0e-5)
    assert torch.equal(normalizer.mean, mean)
    assert torch.equal(normalizer.std, std)


def test_libero_checkpoint_contract_rejects_real_robot_dimensions():
    from servovla.evaluation.sim_rollout_libero import _validate_libero_policy_dims

    with pytest.raises(ValueError, match="dataset=sim_libero"):
        _validate_libero_policy_dims(action_dim=6, state_dim=6, num_cameras=3)


def test_libero_run_contract_rejects_real_robot_config():
    from servovla.evaluation.sim_rollout_libero import _validate_libero_run_config

    cfg = OmegaConf.create({"dataset": {"action_mode": "delta"}})

    with pytest.raises(ValueError, match="benchmark=libero_40"):
        _validate_libero_run_config(cfg)


def test_evaluate_libero_pretrained_loads_exported_policy(monkeypatch, tmp_path):
    import servovla.evaluation.sim_rollout_libero as module

    pretrained_dir = tmp_path / "pretrained_servovla_raw"
    pretrained_dir.mkdir()
    (pretrained_dir / "export_metadata.json").write_text(
        '{"checkpoint_path": "/tmp/checkpoint.pt", "used_ema": false}\n',
        encoding="utf-8",
    )

    eval_cfg_path = tmp_path / "eval.yaml"
    eval_cfg_path.write_text(
        "\n".join(
            [
                "protocol_path: unused",
                "suite: libero_40",
                "device: cuda",
                "obs_type: pixels_agent_pos",
                "render_mode: rgb_array",
                "camera_name: agentview_image,robot0_eye_in_hand_image",
                "control_mode: relative",
                "n_episodes_per_task: 10",
                "use_async_envs: false",
                "delay_chunk_size_threshold: 0.0",
                "prefer_ema: true",
                "seed: 1234",
                "max_tasks: 0",
                "top_failure_count: 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    captured = {}

    class FakePolicy:
        config = SimpleNamespace(action_mode="abs", semantic_wait_warn_ms=500)

        @classmethod
        def from_pretrained(cls, path):
            captured["pretrained_dir"] = Path(path)
            return cls()

        def configure_runtime(self, **kwargs):
            captured["runtime_kwargs"] = kwargs

        def eval(self):
            return self

        def to(self, _device):
            return self

    monkeypatch.setattr(module, "ServoVLAPolicy", FakePolicy, raising=False)
    monkeypatch.setattr(
        module,
        "load_libero_protocol",
        lambda _path: type(
            "Protocol", (), {"tasks": [], "suites": ("libero_10", "libero_spatial")}
        )(),
    )

    result = module.evaluate_libero_pretrained(
        pretrained_dir=pretrained_dir,
        eval_cfg_path=eval_cfg_path,
        device="cuda:0",
        semantic_wait_fail_ms=12000,
    )

    assert captured["pretrained_dir"] == pretrained_dir.resolve()
    assert result["checkpoint_path"] == "/tmp/checkpoint.pt"
    assert result["pretrained_dir"] == str(pretrained_dir.resolve())
    assert result["used_ema"] is False
    assert result["suite"] == "libero_40"
    assert result["suites"] == ["libero_10", "libero_spatial"]
    assert result["delay_chunk_size_threshold"] == 0.0
    assert captured["runtime_kwargs"]["semantic_wait_fail_ms"] == 12000


def test_sim_libero_eval_cfg_defaults_to_relative_controller():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "eval" / "sim_libero.yaml")

    assert cfg.suite == "libero_40"
    assert cfg.protocol_path.endswith("configs/eval/libero_40_tasks.json")
    assert cfg.run_root == "runs/eval/libero_40"
    assert cfg.control_mode == "relative"


def test_sim_libero_eval_cfg_defaults_to_open_loop_chunk_execution():
    cfg = OmegaConf.load(PROJECT_ROOT / "configs" / "eval" / "sim_libero.yaml")

    assert cfg.replan_every_step is False
    assert cfg.delay_chunk_size_threshold == 0.0


def test_exported_libero_policy_maps_env_step_to_chunk_boundary_action_step():
    from servovla.evaluation.sim_rollout_libero import ExportedServoVLALiberoRolloutPolicy

    policy = SimpleNamespace(config=SimpleNamespace(chunk_size=16), eval=lambda: None)
    policy.eval = lambda: policy
    policy.to = lambda _device: policy
    wrapper = ExportedServoVLALiberoRolloutPolicy(policy=policy, device=torch.device("cpu"))

    assert wrapper._runtime_action_step(0) == 0
    assert wrapper._runtime_action_step(15) == 0
    assert wrapper._runtime_action_step(16) == 15
    assert wrapper._runtime_action_step(31) == 15
    assert wrapper._runtime_action_step(32) == 30


def test_close_rollout_env_falls_back_to_direct_close(monkeypatch):
    class DummyEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    env = DummyEnv()

    import servovla.evaluation.sim_rollout_libero as module

    def _raise_not_implemented(_env):
        raise NotImplementedError("unsupported env type")

    monkeypatch.setattr(module, "close_envs", _raise_not_implemented)
    module.close_rollout_env(env)

    assert env.closed is True


def test_sequential_episode_batch_sampler_respects_episode_allowlist():
    from servovla.data.dataset_loader import SequentialEpisodeBatchSampler

    class DummyDataset:
        def __init__(self):
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [0, 3, 6],
                "dataset_to_index": [3, 6, 9],
            }

    sampler = SequentialEpisodeBatchSampler(
        lerobot_dataset=DummyDataset(),
        batch_size=3,
        drop_last=True,
        is_ddp=False,
        episode_indices=[1],
    )

    batches = list(iter(sampler))
    assert batches == [[3, 4, 5]]


def test_sequential_episode_batch_sampler_can_preserve_episode_order():
    from servovla.data.dataset_loader import SequentialEpisodeBatchSampler

    class DummyDataset:
        def __init__(self):
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [0, 3, 6],
                "dataset_to_index": [3, 6, 9],
            }

    sampler = SequentialEpisodeBatchSampler(
        lerobot_dataset=DummyDataset(),
        batch_size=3,
        drop_last=True,
        shuffle_episodes=False,
        is_ddp=False,
    )

    assert list(iter(sampler)) == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]


def test_sequential_episode_batch_sampler_uses_filtered_dataset_row_indices():
    from servovla.data.dataset_loader import SequentialEpisodeBatchSampler

    class DummyHFDataset:
        column_names = ["episode_index"]

        def __init__(self, episode_index):
            self._episode_index = [torch.tensor(value) for value in episode_index]

        def __getitem__(self, key):
            if key != "episode_index":
                raise KeyError(key)
            return self._episode_index

        def __len__(self):
            return len(self._episode_index)

    class DummyDataset:
        def __init__(self):
            self.episodes = [5, 7]
            self.hf_dataset = DummyHFDataset([5, 5, 7, 7, 7])
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [idx * 10 for idx in range(8)],
                "dataset_to_index": [idx * 10 + 10 for idx in range(8)],
            }

    sampler = SequentialEpisodeBatchSampler(
        lerobot_dataset=DummyDataset(),
        batch_size=2,
        drop_last=False,
        is_ddp=False,
        episode_indices=[7],
    )

    assert list(iter(sampler)) == [[2, 3], [4]]


def test_online_vla_dataset_uses_filtered_episode_bounds():
    from servovla.data.dataset_loader import OnlineVLADataset

    class DummyHFDataset:
        column_names = ["episode_index"]

        def __init__(self, episode_index):
            self._episode_index = [torch.tensor(value) for value in episode_index]

        def __getitem__(self, key):
            if key != "episode_index":
                raise KeyError(key)
            return self._episode_index

        def __len__(self):
            return len(self._episode_index)

    class DummyDataset:
        def __init__(self):
            self.episodes = [5, 7]
            self.hf_dataset = DummyHFDataset([5, 5, 7, 7, 7])
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [idx * 10 for idx in range(8)],
                "dataset_to_index": [idx * 10 + 10 for idx in range(8)],
            }

    dataset = OnlineVLADataset(
        lerobot_dataset=DummyDataset(),
        vlm_processor=None,
        vision_processor=None,
        action_horizon=4,
    )

    assert dataset._get_episode_bounds(idx=2, ep_idx=7) == (2, 5)


def test_online_vla_dataset_allows_abs_actions_narrower_than_state():
    from servovla.data.dataset_loader import OnlineVLADataset

    class DummyVisionProcessor:
        def __call__(self, *, images, return_tensors, size):
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}

    class DummyVLMProcessor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "demo prompt"

    class DummyDataset:
        def __init__(self):
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [0],
                "dataset_to_index": [1],
            }
            item = {
                "episode_index": torch.tensor(0),
                "task": "demo task",
                "observation.images.image": torch.zeros(3, 4, 4),
                "observation.images.image2": torch.zeros(3, 4, 4),
                "observation.state": torch.zeros(8),
                "action": torch.zeros(4, 7),
            }
            self.items = [item]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    dataset = OnlineVLADataset(
        lerobot_dataset=DummyDataset(),
        vlm_processor=DummyVLMProcessor(),
        vision_processor=DummyVisionProcessor(),
        action_horizon=4,
        camera_keys=["observation.images.image", "observation.images.image2"],
        proprio_dim=8,
        action_is_delta=False,
    )

    sample = dataset[0]
    assert tuple(sample["action"].shape) == (4, 7)
    assert tuple(sample["q_current"].shape) == (8,)


def test_online_vla_dataset_can_keep_delta_controller_actions_without_state_subtraction():
    from servovla.data.dataset_loader import OnlineVLADataset

    class DummyVisionProcessor:
        def __call__(self, *, images, return_tensors, size):
            del images, return_tensors, size
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}

    class DummyVLMProcessor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "demo prompt"

    class DummyDataset:
        def __init__(self):
            self.meta = type("Meta", (), {})()
            self.meta.episodes = {
                "dataset_from_index": [0],
                "dataset_to_index": [1],
            }
            self.action = torch.tensor(
                [[0.1, -0.2, 0.3, 0.01, -0.02, 0.03, -1.0]],
                dtype=torch.float32,
            ).repeat(4, 1)
            item = {
                "episode_index": torch.tensor(0),
                "task": "demo task",
                "observation.images.image": torch.zeros(3, 4, 4),
                "observation.images.image2": torch.zeros(3, 4, 4),
                "observation.state": torch.tensor([0.4, 0.5, 0.6, 3.0, -1.0, 0.2, 0.03, -0.03]),
                "action": self.action,
            }
            self.items = [item]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    raw_dataset = DummyDataset()
    dataset = OnlineVLADataset(
        lerobot_dataset=raw_dataset,
        vlm_processor=DummyVLMProcessor(),
        vision_processor=DummyVisionProcessor(),
        action_horizon=4,
        camera_keys=["observation.images.image", "observation.images.image2"],
        proprio_dim=8,
        action_is_delta=True,
        action_delta_state_indices=[0, 1, 2, 3, 4, 5, None],
    )

    sample = dataset[0]
    expected_basis = torch.tensor([0.4, 0.5, 0.6, 3.0, -1.0, 0.2, 0.0], dtype=torch.float32)
    assert torch.allclose(sample["action"], raw_dataset.action - expected_basis)
    assert torch.allclose(sample["action"][:, 6], raw_dataset.action[:, 6])
    assert torch.allclose(sample["action_abs"], raw_dataset.action)
