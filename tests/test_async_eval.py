from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts import train as train_script
from servovla.evaluation.async_eval import AsyncEvalTask, build_eval_log_payload


def _make_cfg(tmp_path, *, async_enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=str(tmp_path),
        dataset=SimpleNamespace(action_mode="delta", val_split="val"),
        deployment=SimpleNamespace(enabled=True, auto_export=True),
        training=SimpleNamespace(
            device="cpu",
            eval_seed=1234,
            eval_zero_noise=False,
            async_eval=SimpleNamespace(enabled=async_enabled),
        ),
    )


def test_eval_manager_processes_tasks_in_fifo_order():
    calls: list[int] = []

    def _evaluator(task):
        calls.append(task.step)
        return {
            "first_step": {"mae_mean": float(task.step)},
            "all_steps": {"mae_mean": float(task.step)},
        }

    manager = train_script.AsyncEvalManager(
        evaluator=_evaluator,
        result_writer=lambda step, summary: None,
        wandb_logger=None,
        max_retries=1,
    )

    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=1000,
            snapshot={"model_state": {}},
            used_ema=True,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.enqueue(
        AsyncEvalTask(
            step=2000,
            snapshot={"model_state": {}},
            used_ema=True,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()

    assert calls == [1000, 2000]


def test_eval_manager_sets_fatal_state_after_retry_exhaustion():
    attempts: list[int] = []

    def _evaluator(task):
        attempts.append(task.step)
        raise RuntimeError("boom")

    manager = train_script.AsyncEvalManager(
        evaluator=_evaluator,
        result_writer=lambda step, summary: None,
        wandb_logger=None,
        max_retries=2,
    )

    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=1000,
            snapshot={"model_state": {}},
            used_ema=True,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()

    assert attempts == [1000, 1000]
    assert manager.fatal_error is not None
    with pytest.raises(RuntimeError, match="async evaluation failed"):
        manager.check_healthy()


def test_build_async_eval_manager_loads_compiled_policy_head_snapshot_and_writes_result(
    monkeypatch, tmp_path
):
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2)

    cfg = _make_cfg(tmp_path)
    source_model = _DummyModel()
    target_model = _DummyModel()

    with torch.no_grad():
        source_model.policy_head.weight.fill_(1.5)
        source_model.policy_head.bias.fill_(-0.25)
        target_model.policy_head.weight.zero_()
        target_model.policy_head.bias.zero_()

    monkeypatch.setattr(
        train_script, "_build_predict_absolute_chunk_fn", lambda *args, **kwargs: "predictor"
    )
    monkeypatch.setattr(
        train_script,
        "evaluate_online_validation_batches",
        lambda **kwargs: {"fm_loss_mean": 1.0, "action_loss_mean": 3.0},
    )
    monkeypatch.setattr(train_script, "_maybe_get_wandb_logger", lambda: None)

    manager = train_script.build_async_eval_manager(cfg, model=target_model, val_dataloader=[])
    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=12,
            snapshot={
                "model_state": {
                    "policy_head._orig_mod.weight": source_model.policy_head.weight.detach().clone(),
                    "policy_head._orig_mod.bias": source_model.policy_head.bias.detach().clone(),
                }
            },
            used_ema=True,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()
    manager.check_healthy()

    summary = json.loads((tmp_path / "eval" / "step0000012.json").read_text(encoding="utf-8"))

    assert summary["fm_loss_mean"] == 1.0
    assert torch.allclose(target_model.policy_head.weight, source_model.policy_head.weight)
    assert torch.allclose(target_model.policy_head.bias, source_model.policy_head.bias)


def test_importing_train_script_does_not_emit_sentencepiece_warnings():
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "default",
            "-c",
            'import scripts.train; print("ok")',
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "ok"
    assert "SwigPyPacked" not in completed.stderr
    assert "SwigPyObject" not in completed.stderr
    assert "swigvarlink" not in completed.stderr


def test_maybe_init_wandb_starts_fresh_run_with_timeout(monkeypatch, tmp_path):
    captured = {}
    defined = []

    class _FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def _fake_init(**kwargs):
        captured.update(kwargs)

    fake_wandb = types.SimpleNamespace(
        init=_fake_init,
        define_metric=lambda *args, **kwargs: defined.append((args, kwargs)),
        Settings=_FakeSettings,
        run=types.SimpleNamespace(name="new-run"),
    )
    cfg = _make_cfg(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setenv("SERVOVLA_WANDB_INIT_TIMEOUT", "123")
    monkeypatch.setattr(train_script.OmegaConf, "to_container", lambda *args, **kwargs: {"seed": 1})

    ok = train_script._maybe_init_wandb(cfg)

    assert ok is True
    assert "id" not in captured
    assert "resume" not in captured
    assert captured["settings"].kwargs["init_timeout"] == 123
    assert (("global_step",), {}) in defined
    assert (("train/*",), {"step_metric": "global_step"}) in defined
    assert (("perf/*",), {"step_metric": "global_step"}) in defined
    assert (("val/*",), {"step_metric": "val/step"}) in defined


def test_maybe_init_wandb_does_not_persist_current_run_id(monkeypatch, tmp_path):
    class _FakeSettings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_wandb = types.SimpleNamespace(
        init=lambda **kwargs: None,
        define_metric=lambda *args, **kwargs: None,
        Settings=_FakeSettings,
        run=types.SimpleNamespace(id="run-abc123", name="current-run"),
    )
    cfg = _make_cfg(tmp_path)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    monkeypatch.setattr(train_script.OmegaConf, "to_container", lambda *args, **kwargs: {"seed": 1})

    ok = train_script._maybe_init_wandb(cfg)

    assert ok is True
    assert list(tmp_path.iterdir()) == []


def test_build_async_eval_manager_logs_dataset_global_metrics_only(monkeypatch, tmp_path):
    model = torch.nn.Linear(2, 2)
    logged = []
    cfg = _make_cfg(tmp_path)

    monkeypatch.setattr(
        train_script, "_build_predict_absolute_chunk_fn", lambda *args, **kwargs: "predictor"
    )
    monkeypatch.setattr(
        train_script,
        "evaluate_online_validation_batches",
        lambda **kwargs: {
            "fm_loss_mean": 1.0,
            "action_loss_mean": 3.0,
            "first_step": {"mae_mean": 2.0},
            "all_steps": {"mae_mean": 3.0},
        },
    )
    monkeypatch.setattr(
        train_script,
        "_maybe_get_wandb_logger",
        lambda: lambda payload, step=None: logged.append((step, payload)),
    )

    manager = train_script.build_async_eval_manager(cfg, model=model, val_dataloader=[])
    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=4,
            snapshot={"model_state": model.state_dict()},
            used_ema=False,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()
    manager.check_healthy()

    assert logged[0][0] is None
    assert logged[0][1]["val/step"] == 4
    assert logged[0][1]["val/raw/fm_loss"] == 1.0
    assert logged[0][1]["val/raw/action_loss"] == 3.0
    assert "val/loss" not in logged[0][1]
    assert "val/episode/first_step/mae_mean" not in logged[0][1]
    assert "val/episode/all_steps/rmse_mean" not in logged[0][1]


def test_build_eval_log_payload_records_only_raw_and_ema_eval_variants():
    payload = build_eval_log_payload(
        {
            "raw": {
                "fm_loss_mean": 1.0,
                "fm_loss_by_dataset": {"smoke_a": 1.5},
                "action_loss_mean": 3.0,
            },
            "ema": {
                "fm_loss_mean": 4.0,
                "fm_loss_by_dataset": {"smoke_a": 4.5},
                "action_loss_mean": 6.0,
            },
        },
        eval_step=7,
    )

    assert payload["val/step"] == 7
    assert payload["val/raw/fm_loss"] == 1.0
    assert payload["val/raw/fm_loss/smoke_a"] == 1.5
    assert payload["val/raw/action_loss"] == 3.0
    assert payload["val/ema/fm_loss"] == 4.0
    assert payload["val/ema/fm_loss/smoke_a"] == 4.5
    assert payload["val/ema/action_loss"] == 6.0
    assert "val/loss" not in payload
    assert "val/first_step/mae_mean" not in payload
    assert "val/raw/loss" not in payload
    assert "val/raw/mae_loss" not in payload
    assert "val/raw/all_steps/mae_mean" not in payload


def test_build_async_eval_manager_evaluates_raw_and_ema_snapshots(monkeypatch, tmp_path):
    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_head = torch.nn.Linear(2, 2)

    cfg = _make_cfg(tmp_path)
    model = _DummyModel()
    raw_state = {
        "policy_head.weight": torch.full_like(model.policy_head.weight, 1.0),
        "policy_head.bias": torch.full_like(model.policy_head.bias, 1.0),
    }
    ema_state = {
        "policy_head.weight": torch.full_like(model.policy_head.weight, 2.0),
        "policy_head.bias": torch.full_like(model.policy_head.bias, 2.0),
    }
    logged = []

    monkeypatch.setattr(
        train_script, "_build_predict_absolute_chunk_fn", lambda *args, **kwargs: "predictor"
    )

    def _fake_eval(**kwargs):
        eval_model = kwargs["model"]
        value = float(eval_model.policy_head.weight[0, 0].item())
        return {
            "fm_loss_mean": value,
            "action_loss_mean": value + 0.5,
        }

    monkeypatch.setattr(train_script, "evaluate_online_validation_batches", _fake_eval)
    monkeypatch.setattr(
        train_script,
        "_maybe_get_wandb_logger",
        lambda: lambda payload, step=None: logged.append((step, payload)),
    )

    manager = train_script.build_async_eval_manager(cfg, model=model, val_dataloader=[])
    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=9,
            snapshot={"raw_state": raw_state, "ema_state": ema_state},
            used_ema=True,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()
    manager.check_healthy()

    summary = json.loads((tmp_path / "eval" / "step0000009.json").read_text(encoding="utf-8"))
    assert summary["raw"]["fm_loss_mean"] == 1.0
    assert summary["ema"]["fm_loss_mean"] == 2.0
    assert logged[0][1]["val/raw/fm_loss"] == 1.0
    assert logged[0][1]["val/ema/fm_loss"] == 2.0
    assert "val/loss" not in logged[0][1]


def test_build_async_eval_manager_uses_configured_subprocess_eval_device(monkeypatch, tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.training.async_eval.backend = "subprocess"
    cfg.training.async_eval.device = "cuda:3"

    captured: dict[str, object] = {}

    def _fake_run(cmd, *, env, stdout, stderr, check):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(env)
        captured["config"] = train_script.OmegaConf.load(cmd[cmd.index("--config") + 1])
        output_path = Path(cmd[cmd.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "raw": {
                        "fm_loss_mean": 1.0,
                        "action_loss_mean": 3.0,
                    }
                }
            ),
            encoding="utf-8",
        )
        if stdout is not None:
            stdout.write("worker ok\n")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_script.subprocess, "run", _fake_run)
    monkeypatch.setattr(train_script, "_maybe_get_wandb_logger", lambda: None)

    manager = train_script.build_async_eval_manager(cfg, model=None, val_dataloader=None)
    manager.start()
    manager.enqueue(
        AsyncEvalTask(
            step=11,
            snapshot={"raw_state": {"policy_head.weight": torch.ones(1)}},
            used_ema=False,
            eval_seed=1234,
            eval_zero_noise=False,
        )
    )
    manager.finish()
    manager.join()
    manager.check_healthy()

    assert "CUDA_VISIBLE_DEVICES" not in captured["env"]
    assert "SERVOVLA_EVAL_DEVICE" not in captured["env"]
    assert captured["config"].training.device == "cuda:3"
    assert Path(captured["cmd"][2]).name == "async_eval_worker.py"
    assert (
        json.loads((tmp_path / "eval" / "step0000011.json").read_text(encoding="utf-8"))["raw"][
            "fm_loss_mean"
        ]
        == 1.0
    )


def test_async_eval_worker_receives_vlm_compile_warmup_config(tmp_path):
    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.async_eval.backend = "subprocess"
    cfg.training.async_eval.device = "cuda:0"
    cfg.training.vlm_compile = SimpleNamespace(
        warmup_enabled=True,
        warmup_max_batches=32,
        signature_max_entries=16,
        on_new_signature="warn_and_warmup",
        log_signatures=True,
        eval_warmup_enabled=True,
        position_cache_enabled=False,
    )

    config_path = tmp_path / "async_eval_config.yaml"
    train_script._write_async_eval_config(cfg, config_path, eval_device="cuda:0")
    written = train_script.OmegaConf.load(config_path)

    assert written.training.device == "cuda:0"
    assert written.training.vlm_compile.eval_warmup_enabled is True
    assert written.training.vlm_compile.signature_max_entries == 16


def test_async_eval_config_routes_decode_to_eval_decode_device(tmp_path):
    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.async_eval.backend = "subprocess"
    cfg.training.async_eval.device = "cuda:0"
    cfg.training.async_eval.decode_device = "cuda:1"
    cfg.training.decode = SimpleNamespace(device="cuda:2", video_backend="auto")

    config_path = tmp_path / "async_eval_config.yaml"
    train_script._write_async_eval_config(cfg, config_path, eval_device="cuda:0")
    written = train_script.OmegaConf.load(config_path)

    assert written.training.device == "cuda:0"
    assert written.training.decode.device == "cuda:1"


def test_async_eval_config_preserves_decode_device_when_eval_decode_device_unset(tmp_path):
    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.async_eval.backend = "subprocess"
    cfg.training.async_eval.device = "cuda:0"
    cfg.training.decode = SimpleNamespace(device="cuda:2", video_backend="auto")

    config_path = tmp_path / "async_eval_config.yaml"
    train_script._write_async_eval_config(cfg, config_path, eval_device="cuda:0")
    written = train_script.OmegaConf.load(config_path)

    assert written.training.device == "cuda:0"
    assert written.training.decode.device == "cuda:2"


def test_async_eval_worker_warms_vlm_signatures_with_eval_path(monkeypatch, tmp_path):
    from scripts import async_eval_worker

    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.vlm_compile = SimpleNamespace(
        eval_warmup_enabled=True,
        warmup_max_batches=2,
        signature_max_entries=4,
        log_signatures=False,
    )
    captured = {}
    batches = [
        {
            "vlm_inputs": {
                "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 18, 18]], dtype=torch.long),
            }
        }
        for seq_len in (64, 80, 96)
    ]

    monkeypatch.setattr(
        train_script, "_build_predict_absolute_chunk_fn", lambda *args, **kwargs: "predictor"
    )
    monkeypatch.setattr(
        train_script,
        "evaluate_online_validation_batches",
        lambda **kwargs: captured.update(kwargs) or {"loss_mean": 0.0},
    )

    async_eval_worker._warm_eval_vlm_signatures(cfg, torch.nn.Linear(1, 1), batches)

    assert len(captured["batches"]) == 2
    assert captured["device"] == "cpu"
    assert captured["predict_absolute_chunk_fn"] == "predictor"
    assert captured["seed"] == 1234


def test_async_eval_worker_warmup_preprocesses_batches_lazily(monkeypatch, tmp_path):
    from scripts import async_eval_worker

    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.vlm_compile = SimpleNamespace(
        eval_warmup_enabled=True,
        warmup_max_batches=2,
        signature_max_entries=4,
        log_signatures=False,
    )
    raw_batches = [
        {
            "batch_id": batch_id,
            "vision_images_uint8": {"groups": []},
            "vlm_inputs": {
                "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
                "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
                "image_grid_thw": torch.tensor([[1, 18, 18]], dtype=torch.long),
            },
        }
        for batch_id, seq_len in enumerate((64, 80, 96))
    ]
    preprocess_calls = []
    captured = {}

    class _Preprocessor:
        def preprocess_batch(self, batch):
            preprocess_calls.append(batch["batch_id"])
            processed = dict(batch)
            processed["processed"] = True
            return processed

    def _fake_eval(**kwargs):
        evaluated = list(kwargs["batches"])
        captured["evaluated"] = evaluated
        return {"loss_mean": 0.0}

    monkeypatch.setattr(
        train_script, "_build_predict_absolute_chunk_fn", lambda *args, **kwargs: "predictor"
    )
    monkeypatch.setattr(train_script, "evaluate_online_validation_batches", _fake_eval)

    async_eval_worker._warm_eval_vlm_signatures(
        cfg,
        torch.nn.Linear(1, 1),
        raw_batches,
        image_preprocessor=_Preprocessor(),
    )

    assert [batch["batch_id"] for batch in captured["evaluated"]] == [0, 1]
    assert all(batch["processed"] for batch in captured["evaluated"])
    assert preprocess_calls == [0, 1]


def test_async_eval_worker_calls_vlm_warmup_before_task_eval(monkeypatch, tmp_path):
    from scripts import async_eval_worker

    cfg = _make_cfg(tmp_path, async_enabled=True)
    cfg.training.eval_batch_size = 1
    cfg.training.eval_num_workers = 0
    cfg.training.vlm_compile = SimpleNamespace(eval_warmup_enabled=True)
    calls = []
    model = torch.nn.Linear(1, 1)
    val_loader = [{"batch_id": 0}]

    monkeypatch.setattr(train_script, "configure_hf_offline_env", lambda: None)
    monkeypatch.setattr(train_script, "_load_processors", lambda cfg: (object(), object()))
    monkeypatch.setattr(train_script, "_build_servovla_model", lambda cfg, *, device: model)
    monkeypatch.setattr(train_script, "get_dataset_names", lambda dataset, split: ["unit"])
    monkeypatch.setattr(
        train_script, "build_end2end_dataloader", lambda *args, **kwargs: val_loader
    )
    monkeypatch.setattr(
        async_eval_worker,
        "_warm_eval_vlm_signatures",
        lambda cfg, model, val_dataloader, **kwargs: calls.append(("warm", val_dataloader)),
    )
    monkeypatch.setattr(
        async_eval_worker,
        "_run_eval_for_state",
        lambda **kwargs: calls.append(("eval", kwargs["val_dataloader"])) or {"loss_mean": 1.0},
    )

    summary = async_eval_worker._evaluate_task(
        cfg,
        AsyncEvalTask(
            step=1,
            snapshot={"model_state": {}},
            used_ema=False,
            eval_seed=1234,
            eval_zero_noise=False,
        ),
    )

    assert summary["loss_mean"] == 1.0
    assert calls == [("warm", val_loader), ("eval", val_loader)]


def test_subprocess_eval_requires_device_config(tmp_path):
    cfg = _make_cfg(tmp_path)
    cfg.training.async_eval.backend = "subprocess"

    with pytest.raises(ValueError, match="training.async_eval.device"):
        train_script.build_async_eval_manager(cfg, model=None, val_dataloader=None)
