"""Tests for TrainerLoop raw end-to-end training path."""

from __future__ import annotations

import sys
import time
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from servovla.trainer.trainer_loop import (
    TrainerLoop,
    collect_process_memory_snapshot,
    dataloader_worker_pids,
    summarize_batch_memory,
)
from tests.test_end2end_training import _batch as _end2end_batch
from tests.test_end2end_training import _model as _end2end_model


def test_batch_profile_reports_tensor_bytes_for_raw_pipeline_fields():
    batch = {
        "pixel_values": torch.zeros(2, 2, 3, 224, 224, dtype=torch.float32),
        "vlm_inputs": {
            "input_ids": torch.zeros(2, 16, dtype=torch.long),
            "pixel_values": torch.zeros(4, 3, 336, 336, dtype=torch.float32),
        },
        "frame_delay": torch.zeros(2, dtype=torch.float32),
    }

    profile = summarize_batch_memory(batch)

    assert profile["tensor_bytes.total"] > 0
    assert (
        profile["tensor_bytes.pixel_values"]
        == batch["pixel_values"].numel() * batch["pixel_values"].element_size()
    )
    assert (
        profile["tensor_bytes.vlm_inputs.pixel_values"]
        == batch["vlm_inputs"]["pixel_values"].numel()
        * batch["vlm_inputs"]["pixel_values"].element_size()
    )


def test_summarize_batch_memory_reports_raw_image_bytes():
    batch = {
        "vision_images_uint8": {
            "groups": [
                {"images": torch.zeros(2, 3, 4, 4, dtype=torch.uint8)},
                {"images": torch.zeros(1, 3, 2, 2, dtype=torch.uint8)},
            ]
        },
        "vlm_images_uint8": {
            "groups": [
                {"images": torch.zeros(2, 3, 4, 4, dtype=torch.uint8)},
            ]
        },
        "action": torch.zeros(2, 4, 2),
    }

    profile = summarize_batch_memory(batch)

    assert profile["raw_image_bytes.vision"] == (2 * 3 * 4 * 4) + (1 * 3 * 2 * 2)
    assert profile["raw_image_bytes.vlm"] == 2 * 3 * 4 * 4
    assert profile["raw_image_bytes.total"] == (2 * 3 * 4 * 4) + (1 * 3 * 2 * 2) + (2 * 3 * 4 * 4)


def test_summarize_batch_memory_reports_raw_decode_stats():
    batch = {
        "raw_decode_stats": {
            "seconds": 1.25,
            "calls": 2,
            "frames": 6,
            "videos": 2,
            "groups": 2,
            "max_span_s": 0.5,
            "job_seconds_sum": 1.4,
            "group_seconds_sum": 1.3,
            "max_job_seconds": 0.8,
            "max_group_seconds": 0.7,
        }
    }

    profile = summarize_batch_memory(batch)

    assert profile["raw_decode_stats.seconds"] == 1.25
    assert profile["raw_decode_stats.calls"] == 2
    assert profile["raw_decode_stats.frames"] == 6
    assert profile["raw_decode_stats.videos"] == 2
    assert profile["raw_decode_stats.groups"] == 2
    assert profile["raw_decode_stats.max_span_s"] == 0.5
    assert profile["raw_decode_stats.job_seconds_sum"] == 1.4
    assert profile["raw_decode_stats.group_seconds_sum"] == 1.3
    assert profile["raw_decode_stats.max_job_seconds"] == 0.8
    assert profile["raw_decode_stats.max_group_seconds"] == 0.7


def test_process_memory_snapshot_reports_main_and_worker_keys():
    snapshot = collect_process_memory_snapshot(worker_pids=[])

    assert "rss_bytes.main" in snapshot
    assert "rss_bytes.workers_total" in snapshot
    assert snapshot["rss_bytes.workers_total"] == 0


def test_dataloader_worker_pids_reads_worker_processes_from_iterator():
    data_iter = SimpleNamespace(
        _workers=[
            SimpleNamespace(pid=123),
            SimpleNamespace(pid=456),
        ]
    )

    assert dataloader_worker_pids(data_iter) == (123, 456)


def test_run_passes_dataloader_worker_pids_to_memory_snapshot(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    observed_worker_pids = []

    class _WorkerIterator:
        def __init__(self):
            self._workers = [SimpleNamespace(pid=123), SimpleNamespace(pid=456)]
            self._items = iter([_end2end_batch()])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._items)

    class _WorkerDataloader:
        def __iter__(self):
            return _WorkerIterator()

    def _fake_memory_snapshot(*, worker_pids):
        observed_worker_pids.append(tuple(worker_pids))
        return {
            "rss_bytes.main": 1,
            "rss_bytes.workers_total": 2,
            "rss_bytes.worker_count": len(tuple(worker_pids)),
        }

    monkeypatch.setattr(
        trainer_loop_module, "collect_process_memory_snapshot", _fake_memory_snapshot
    )

    trainer.run(dataloader=_WorkerDataloader(), max_steps=1)

    assert observed_worker_pids == [(123, 456)]


def test_trainer_compile_can_compile_all_servovla_submodules(monkeypatch, tmp_path):
    model = _end2end_model()
    compiled = []

    def _fake_compile(module):
        compiled.append(module)
        return module

    monkeypatch.setattr(torch, "compile", _fake_compile)
    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.compile = True
    cfg.compile_policy_head = True
    cfg.compile_vision_encoder = True
    cfg.compile_vlm_encoder = True
    cfg.vlm_compile = SimpleNamespace(
        flash_attention_graph_break=False,
        qwen_visual_position_graph_break=False,
        nested_fx_trace_fallback=False,
    )

    trainer = TrainerLoop(model=model, train_cfg=cfg)

    assert trainer.model is model
    assert compiled == [model.policy_head, model.vision_encoder, model.vlm_encoder]


def test_trainer_compile_failure_keeps_servovla_submodules_unmodified(monkeypatch, tmp_path):
    model = _end2end_model()
    original_policy_head = model.policy_head
    original_vision_encoder = model.vision_encoder
    original_vlm_encoder = model.vlm_encoder
    calls = []

    class _CompiledWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    def _fake_compile(module):
        calls.append(module)
        if module is original_vlm_encoder:
            raise RuntimeError("compile failed")
        return _CompiledWrapper(module)

    monkeypatch.setattr(torch, "compile", _fake_compile)
    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.compile = True
    cfg.compile_policy_head = True
    cfg.compile_vision_encoder = True
    cfg.compile_vlm_encoder = True
    cfg.vlm_compile = SimpleNamespace(
        flash_attention_graph_break=False,
        qwen_visual_position_graph_break=False,
        nested_fx_trace_fallback=False,
    )

    trainer = TrainerLoop(model=model, train_cfg=cfg)

    assert calls == [original_policy_head, original_vision_encoder, original_vlm_encoder]
    assert trainer.model is model
    assert model.policy_head is original_policy_head
    assert model.vision_encoder is original_vision_encoder
    assert model.vlm_encoder is original_vlm_encoder


def test_trainer_compile_patches_transformers_vlm_graph_breaks_before_vlm_compile(
    monkeypatch, tmp_path
):
    import servovla.trainer.trainer_loop as trainer_loop_module

    model = _end2end_model()
    calls = []

    def _fake_compile(module):
        calls.append(("compile", module))
        return module

    def _fake_patch():
        calls.append(("patch_fa", None))
        return True

    def _fake_qwen_patch():
        calls.append(("patch_qwen", None))
        return True

    def _fake_nested_fx_patch():
        calls.append(("patch_nested_fx", None))
        return True

    monkeypatch.setattr(torch, "compile", _fake_compile)
    monkeypatch.setattr(
        trainer_loop_module,
        "apply_transformers_flash_attention_compile_graph_break",
        _fake_patch,
    )
    monkeypatch.setattr(
        trainer_loop_module,
        "apply_transformers_qwen_visual_position_compile_graph_break",
        _fake_qwen_patch,
    )
    monkeypatch.setattr(
        trainer_loop_module,
        "apply_torch_compile_nested_fx_trace_fallback",
        _fake_nested_fx_patch,
    )
    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.compile = True
    cfg.compile_policy_head = True
    cfg.compile_vision_encoder = True
    cfg.compile_vlm_encoder = True
    cfg.vlm_compile = SimpleNamespace(
        flash_attention_graph_break=True,
        qwen_visual_position_graph_break=True,
        nested_fx_trace_fallback=True,
    )

    TrainerLoop(model=model, train_cfg=cfg)

    assert calls == [
        ("compile", model.policy_head),
        ("compile", model.vision_encoder),
        ("patch_fa", None),
        ("patch_qwen", None),
        ("patch_nested_fx", None),
        ("compile", model.vlm_encoder),
    ]


def test_transformers_flash_attention_compile_graph_break_is_idempotent(monkeypatch):
    import sys

    from servovla.trainer.compile_guard import (
        apply_transformers_flash_attention_compile_graph_break,
    )

    fake_module = types.ModuleType("transformers.modeling_flash_attention_utils")
    fake_integration = types.ModuleType("transformers.integrations.flash_attention")
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")

    def fake_flash_attention_forward():
        return "ok"

    def fake_registered_flash_attention_forward():
        return "registered"

    def fake_sdpa_forward():
        return "sdpa"

    fake_module._flash_attention_forward = fake_flash_attention_forward
    fake_integration.flash_attention_forward = fake_registered_flash_attention_forward
    fake_modeling_utils.ALL_ATTENTION_FUNCTIONS = SimpleNamespace(
        _global_mapping={
            "flash_attention_2": fake_registered_flash_attention_forward,
            "flash_attention_3": fake_registered_flash_attention_forward,
            "sdpa": fake_sdpa_forward,
        },
        _local_mapping={},
    )
    monkeypatch.setitem(sys.modules, "transformers.modeling_flash_attention_utils", fake_module)
    monkeypatch.setitem(sys.modules, "transformers.integrations.flash_attention", fake_integration)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)
    disabled = []

    def fake_disable(fn, recursive=True):
        disabled.append((fn, recursive))

        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch.compiler, "disable", fake_disable)

    assert apply_transformers_flash_attention_compile_graph_break() is True
    assert apply_transformers_flash_attention_compile_graph_break() is False
    assert disabled == [
        (fake_registered_flash_attention_forward, True),
        (fake_flash_attention_forward, True),
    ]
    assert fake_module._flash_attention_forward() == "ok"
    assert fake_integration.flash_attention_forward() == "registered"
    assert (
        fake_modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping["flash_attention_2"]()
        == "registered"
    )
    assert (
        fake_modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping["flash_attention_2"]
        is fake_modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping["flash_attention_3"]
    )
    assert fake_modeling_utils.ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] is fake_sdpa_forward
    assert (
        getattr(fake_module._flash_attention_forward, "_servovla_compile_graph_break", False)
        is True
    )


def test_transformers_qwen_visual_position_compile_graph_break_is_idempotent(monkeypatch):
    import sys

    from servovla.trainer.compile_guard import (
        apply_transformers_qwen_visual_position_compile_graph_break,
    )

    fake_module = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")

    class FakeVisionModel:
        def rot_pos_emb(self):
            return "rot"

        def fast_pos_embed_interpolate(self):
            return "pos"

    fake_module.FakeVisionModel = FakeVisionModel
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_5.modeling_qwen3_5", fake_module)
    for module_name in (
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
    ):
        monkeypatch.setitem(sys.modules, module_name, types.ModuleType(module_name))
    disabled = []

    def fake_disable(fn, recursive=True):
        disabled.append((fn, recursive))

        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(torch.compiler, "disable", fake_disable)

    assert apply_transformers_qwen_visual_position_compile_graph_break() is True
    assert apply_transformers_qwen_visual_position_compile_graph_break() is False
    assert disabled == [
        (FakeVisionModel.__dict__["rot_pos_emb"]._servovla_original, True),
        (FakeVisionModel.__dict__["fast_pos_embed_interpolate"]._servovla_original, True),
    ]
    assert FakeVisionModel().rot_pos_emb() == "rot"
    assert FakeVisionModel().fast_pos_embed_interpolate() == "pos"
    assert getattr(FakeVisionModel.rot_pos_emb, "_servovla_compile_graph_break", False) is True


def _cfg_with_eval(tmp_dir, *, eval_every: int, save_every: int, async_enabled: bool = False):
    return SimpleNamespace(
        device="cpu",
        amp_dtype="float32",
        compile=False,
        optimizer=SimpleNamespace(
            lr=1e-3,
            weight_decay=1e-2,
            betas=[0.9, 0.999],
            eps=1e-8,
        ),
        scheduler=SimpleNamespace(T_max=10, eta_min=1e-6),
        ema=SimpleNamespace(decay=0.9, update_after_step=0, update_every=1),
        async_eval=SimpleNamespace(enabled=async_enabled),
        output_dir=tmp_dir,
        grad_clip_norm=1.0,
        log_every=50,
        save_every=save_every,
        eval_every=eval_every,
        eval_use_ema=True,
    )


def test_run_triggers_eval_every_without_waiting_for_save_every(tmp_path):
    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=2, save_every=99),
    )
    train_loader = [_end2end_batch(), _end2end_batch(), _end2end_batch()]
    val_loader = [_end2end_batch()]
    calls = []

    def eval_fn(model, dataloader, step, use_ema):
        calls.append((step, use_ema))
        value = 10.0 if use_ema else 1.0
        return {"first_step": {"mae_mean": value}, "all_steps": {"mae_mean": value + 1.0}}

    trainer.run(dataloader=train_loader, max_steps=3, eval_dataloader=val_loader, eval_fn=eval_fn)

    assert calls == [(2, False), (2, True)]
    assert (Path(tmp_path) / "eval" / "step0000002.json").exists()


def test_run_skips_ema_eval_before_ema_update_starts(tmp_path):
    cfg = _cfg_with_eval(tmp_path, eval_every=2, save_every=99)
    cfg.ema.update_after_step = 5
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    train_loader = [_end2end_batch(), _end2end_batch(), _end2end_batch()]
    val_loader = [_end2end_batch()]
    calls = []

    def eval_fn(model, dataloader, step, use_ema):
        calls.append((step, use_ema))
        return {"first_step": {"mae_mean": 1.0}, "all_steps": {"mae_mean": 2.0}}

    trainer.run(dataloader=train_loader, max_steps=3, eval_dataloader=val_loader, eval_fn=eval_fn)

    assert calls == [(2, False)]
    summary_path = Path(tmp_path) / "eval" / "step0000002.json"
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert '"raw"' in content
    assert '"ema"' not in content


def test_run_async_eval_enqueues_task_instead_of_running_eval(tmp_path):
    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=2, save_every=99, async_enabled=True),
    )
    train_loader = [_end2end_batch(), _end2end_batch(), _end2end_batch()]
    val_loader = [_end2end_batch()]
    queued = []
    eval_calls = []

    def eval_fn(model, dataloader, step, use_ema):
        eval_calls.append(step)
        return {"first_step": {"mae_mean": 1.0}, "all_steps": {"mae_mean": 2.0}}

    def enqueue_fn(*, step, model, ema_model, use_ema):
        queued.append((step, use_ema, ema_model is not None, model is trainer.model))

    trainer.run(
        dataloader=train_loader,
        max_steps=3,
        eval_dataloader=val_loader,
        eval_fn=eval_fn,
        async_eval_enqueue_fn=enqueue_fn,
    )

    assert queued == [(2, True, True, True)]
    assert eval_calls == []
    assert not (Path(tmp_path) / "eval" / "step0000002.json").exists()


def test_run_async_eval_manager_skips_ema_snapshot_before_ema_update_starts(tmp_path):
    cfg = _cfg_with_eval(tmp_path, eval_every=2, save_every=99, async_enabled=True)
    cfg.ema.update_after_step = 5
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    events = []

    class _DummyManager:
        fatal_error = None

        def enqueue(self, task):
            events.append((task.step, task.used_ema, sorted(task.snapshot.keys())))

        def check_healthy(self):
            pass

        def finish(self):
            pass

        def join(self):
            pass

    trainer.run(
        dataloader=[_end2end_batch(), _end2end_batch(), _end2end_batch()],
        max_steps=3,
        eval_dataloader=[_end2end_batch()],
        async_eval_manager=_DummyManager(),
    )

    assert events == [(2, False, ["raw_state"])]


def test_run_async_eval_uses_manager_and_finishes_it(tmp_path):
    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=2, save_every=99, async_enabled=True),
    )
    train_loader = [_end2end_batch(), _end2end_batch(), _end2end_batch()]
    val_loader = [_end2end_batch()]
    events = []
    eval_calls = []

    def eval_fn(model, dataloader, step, use_ema):
        eval_calls.append(step)
        return {"first_step": {"mae_mean": 1.0}, "all_steps": {"mae_mean": 2.0}}

    class _DummyManager:
        fatal_error = None

        def enqueue(self, task):
            snapshot_keys = sorted(task.snapshot.keys())
            events.append(("enqueue", task.step, task.used_ema, snapshot_keys))

        def check_healthy(self):
            events.append("check")

        def finish(self):
            events.append("finish")

        def join(self):
            events.append("join")

    trainer.run(
        dataloader=train_loader,
        max_steps=3,
        eval_dataloader=val_loader,
        eval_fn=eval_fn,
        async_eval_manager=_DummyManager(),
    )

    assert eval_calls == []
    assert events == [
        "check",
        ("enqueue", 2, True, ["ema_state", "raw_state"]),
        "check",
        "finish",
        "join",
        "check",
    ]
    assert not (Path(tmp_path) / "eval" / "step0000002.json").exists()


def test_trainer_stops_when_eval_manager_enters_fatal_state(tmp_path):
    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=1, save_every=99, async_enabled=True),
    )

    class _FatalManager:
        fatal_error = RuntimeError("eval failed")

        def enqueue(self, task):
            pass

        def check_healthy(self):
            raise self.fatal_error

        def finish(self):
            pass

        def join(self):
            pass

    with pytest.raises(RuntimeError, match="eval failed"):
        trainer.run(
            dataloader=[_end2end_batch(), _end2end_batch()],
            max_steps=2,
            eval_dataloader=[_end2end_batch()],
            async_eval_manager=_FatalManager(),
        )


def test_run_sync_eval_path_evaluates_raw_and_ema_when_async_disabled(tmp_path):
    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=2, save_every=99, async_enabled=False),
    )
    train_loader = [_end2end_batch(), _end2end_batch(), _end2end_batch()]
    val_loader = [_end2end_batch()]
    queued = []
    eval_calls = []

    def eval_fn(model, dataloader, step, use_ema):
        eval_calls.append((step, use_ema))
        value = 10.0 if use_ema else 1.0
        return {"first_step": {"mae_mean": value}, "all_steps": {"mae_mean": value + 1.0}}

    def enqueue_fn(**kwargs):
        queued.append(kwargs)

    trainer.run(
        dataloader=train_loader,
        max_steps=3,
        eval_dataloader=val_loader,
        eval_fn=eval_fn,
        async_eval_enqueue_fn=enqueue_fn,
    )

    assert eval_calls == [(2, False), (2, True)]
    assert queued == []
    summary_path = Path(tmp_path) / "eval" / "step0000002.json"
    assert summary_path.exists()
    content = summary_path.read_text(encoding="utf-8")
    assert '"raw"' in content
    assert '"ema"' in content


def test_run_logs_dataset_global_metrics_to_wandb(monkeypatch, tmp_path):
    logged = []
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda payload, step=None: logged.append((step, payload)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    trainer = TrainerLoop(
        model=_end2end_model(),
        train_cfg=_cfg_with_eval(tmp_path, eval_every=1, save_every=99),
    )

    def eval_fn(model, dataloader, step, use_ema):
        value = 2.0 if use_ema else 1.0
        return {
            "fm_loss_mean": value,
            "fm_loss_by_dataset": {"smoke_a": value},
            "action_loss_mean": value + 0.5,
            "first_step": {"mae_mean": value},
            "all_steps": {"mae_mean": value + 0.5},
        }

    trainer.run(
        dataloader=[_end2end_batch()],
        max_steps=1,
        eval_dataloader=[_end2end_batch()],
        eval_fn=eval_fn,
    )

    train_payloads = [(step, payload) for step, payload in logged if "train/loss_main" in payload]
    assert len(train_payloads) == 1
    train_step, train_payload = train_payloads[0]
    assert train_step == 1
    assert train_payload["global_step"] == 1
    assert "train/loss_main" in train_payload

    eval_payloads = [(step, payload) for step, payload in logged if "val/raw/fm_loss" in payload]
    assert len(eval_payloads) == 1

    eval_step, payload = eval_payloads[0]
    assert eval_step is None
    assert payload["val/step"] == 1
    assert payload["val/raw/fm_loss"] == 1.0
    assert payload["val/raw/fm_loss/smoke_a"] == 1.0
    assert payload["val/raw/action_loss"] == 1.5
    assert payload["val/ema/fm_loss"] == 2.0
    assert payload["val/ema/fm_loss/smoke_a"] == 2.0
    assert payload["val/ema/action_loss"] == 2.5
    assert "val/loss" not in payload
    assert "val/raw/loss" not in payload
    assert "val/raw/mae_loss" not in payload
    assert "val/first_step/mae_mean" not in payload
    assert "val/episode/loss_mean" not in payload
    assert "val/episode/first_step/mae_mean" not in payload
    assert all("train/loss_aux" not in payload for _, payload in logged)


def test_run_uses_sync_path_when_gpu_pipeline_disabled(tmp_path):
    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=False,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)

    calls = {"sync": 0, "features": 0}
    original_sync = trainer.train_step_end2end
    original_features = trainer.train_step_from_features

    def wrapped_sync(*args, **kwargs):
        calls["sync"] += 1
        return original_sync(*args, **kwargs)

    def wrapped_features(*args, **kwargs):
        calls["features"] += 1
        return original_features(*args, **kwargs)

    trainer.train_step_end2end = wrapped_sync
    trainer.train_step_from_features = wrapped_features
    trainer.run(dataloader=[_end2end_batch()], max_steps=1)

    assert calls == {"sync": 1, "features": 0}


def test_run_keeps_existing_feature_path_when_async_producer_false(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
        log_timing_every=0,
        gpu_preprocess=True,
        async_producer=False,
        cpu_prefetch_depth=1,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    events = []

    class _FakeEncoded:
        metadata = {"data_time": 0.0}

        def wait_ready(self, device):
            events.append(("wait", device.type))

        def collect_timings(self, *, block=False):
            return {}

    class _FakeProducer:
        def __init__(self, *, queue_depth, **kwargs):
            del kwargs
            events.append(("producer_init", queue_depth))
            self.queue = []
            self.queue_depth = queue_depth

        def can_submit(self):
            return len(self.queue) < self.queue_depth

        def submit(self, batch, *, metadata=None):
            del batch, metadata
            events.append(("submit",))
            self.queue.append(_FakeEncoded())

        def pop_oldest(self):
            events.append(("pop",))
            return self.queue.pop(0)

        def __len__(self):
            return len(self.queue)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(
        trainer_loop_module,
        "AsyncGpuFeatureProducer",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("async producer should not be constructed")
        ),
        raising=False,
    )

    def fake_train_step(encoded, optimizer, *, defer_logging_tensors=False, global_step=0):
        del encoded, optimizer, defer_logging_tensors, global_step
        events.append(("train_features",))
        return {
            "main_loss": torch.tensor(0.0),
            "loss_fm": torch.tensor(0.0),
            "loss_rollout_action": torch.tensor(0.0),
            "loss_total": torch.tensor(0.0),
            "rollout_action_weight": 0.0,
            "dataset_slug": "unit",
            "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
            "dataset_count_by_slug": {"unit": 1},
        }

    trainer.train_step_from_features = fake_train_step
    trainer.run(dataloader=[{"batch_id": 0}], max_steps=1)

    assert ("producer_init", 1) in events
    assert ("train_features",) in events


def test_run_uses_async_producer_without_main_thread_dataloader_next(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=2,
        encoder_parallel_streams=False,
        log_timing_every=1,
        gpu_preprocess=True,
        async_producer=True,
        cpu_prefetch_depth=1,
    )
    cfg.compile = True
    cfg.compile_vlm_encoder = True
    cfg.vlm_compile = SimpleNamespace(
        warmup_enabled=True,
        warmup_max_batches=4,
        signature_max_entries=8,
        on_new_signature="warn_and_warmup",
        log_signatures=False,
        eval_warmup_enabled=True,
        position_cache_enabled=False,
        overlap_train_reader=True,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg, image_preprocessor=object())
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    events = []

    class _ExplodingLoader:
        def __iter__(self):
            events.append(("main_iter",))
            return self

        def __next__(self):
            raise AssertionError("main TrainerLoop must not call next() in async raw mode")

    class _FakeProducer:
        def __init__(self, **kwargs):
            events.append(("producer_init", kwargs.get("image_preprocessor") is not None))

    class _FakeAsync:
        def __init__(self, **kwargs):
            events.append(
                (
                    "async_init",
                    kwargs["feature_queue_depth"],
                    kwargs["cpu_prefetch_depth"],
                    kwargs["vlm_warmup_max_batches"],
                )
            )
            self.items = [
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={"h2d_ms": 1.0, "preprocess_ms": 2.0, "encoder_ms": 3.0},
                    metadata={"data_time": 0.0, "raw_image_bytes": 12},
                )
            ]

        def start(self):
            events.append(("start",))

        def wait_warmup_complete(self, timeout=None):
            events.append(("wait_warmup", timeout is None or timeout > 0))

        def pop_oldest(self):
            events.append(("pop",))
            return self.items.pop(0)

        def stop(self):
            events.append(("stop",))

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(trainer_loop_module, "AsyncGpuFeatureProducer", _FakeAsync, raising=False)

    def fake_train_step(encoded, optimizer, *, defer_logging_tensors=False, global_step=0):
        del optimizer, defer_logging_tensors, global_step
        events.append(("train", encoded.dataset_slug[0]))
        return {
            "main_loss": torch.tensor(0.0),
            "loss_fm": torch.tensor(0.0),
            "loss_rollout_action": torch.tensor(0.0),
            "loss_total": torch.tensor(0.0),
            "rollout_action_weight": 0.0,
            "dataset_slug": "unit",
            "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
            "dataset_count_by_slug": {"unit": 1},
        }

    trainer.train_step_from_features = fake_train_step
    trainer.run(dataloader=_ExplodingLoader(), max_steps=1)

    assert ("producer_init", True) in events
    assert ("async_init", 2, 1, 4) in events
    assert ("start",) in events
    assert ("wait_warmup", True) in events
    assert ("pop",) in events
    assert ("train", "unit") in events
    assert ("stop",) in events
    assert ("main_iter",) not in events


def test_run_passes_vlm_encoder_micro_batch_size_to_feature_producer(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=2,
        encoder_parallel_streams=True,
        log_timing_every=1,
        gpu_preprocess=True,
        async_producer=True,
        cpu_prefetch_depth=1,
        vision_encoder_micro_batch_size=8,
        vlm_encoder_micro_batch_size=8,
        startup_trace_submit_limit=3,
        startup_trace_step_limit=2,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg, image_preprocessor=object())
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    events = []

    class _FakeProducer:
        def __init__(self, **kwargs):
            events.append(
                ("vision_micro_batch_size", kwargs.get("vision_encoder_micro_batch_size"))
            )
            events.append(("vlm_micro_batch_size", kwargs.get("vlm_encoder_micro_batch_size")))
            events.append(("startup_trace_submit_limit", kwargs.get("startup_trace_submit_limit")))

    class _FakeAsync:
        def __init__(self, **kwargs):
            del kwargs
            self.items = [
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata={"data_time": 0.0},
                )
            ]

        def start(self):
            pass

        def wait_warmup_complete(self, timeout=None):
            del timeout

        def pop_oldest(self):
            return self.items.pop(0)

        def stop(self):
            pass

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(trainer_loop_module, "AsyncGpuFeatureProducer", _FakeAsync, raising=False)

    trainer.train_step_from_features = lambda encoded, optimizer, **kwargs: {
        "main_loss": torch.tensor(0.0),
        "loss_fm": torch.tensor(0.0),
        "loss_rollout_action": torch.tensor(0.0),
        "loss_total": torch.tensor(0.0),
        "rollout_action_weight": 0.0,
        "dataset_slug": "unit",
        "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
        "dataset_count_by_slug": {"unit": 1},
    }
    trainer.run(dataloader=[{"batch_id": "train"}], max_steps=1)

    assert ("vision_micro_batch_size", 8) in events
    assert ("vlm_micro_batch_size", 8) in events
    assert ("startup_trace_submit_limit", 3) in events


def test_run_passes_separate_warmup_dataloader_to_async_producer(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=2,
        encoder_parallel_streams=False,
        log_timing_every=1,
        gpu_preprocess=True,
        async_producer=True,
        cpu_prefetch_depth=1,
    )
    cfg.compile = True
    cfg.compile_vlm_encoder = True
    cfg.vlm_compile = SimpleNamespace(
        warmup_enabled=True,
        warmup_max_batches=2,
        signature_max_entries=8,
        on_new_signature="warn_and_warmup",
        log_signatures=False,
        eval_warmup_enabled=True,
        overlap_train_reader=True,
        position_cache_enabled=False,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg, image_preprocessor=object())
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    warmup_loader = [{"batch_id": "warmup"}]
    events = []

    class _FakeProducer:
        def __init__(self, **kwargs):
            del kwargs

    class _FakeAsync:
        def __init__(self, **kwargs):
            events.append(("warmup_loader", kwargs.get("warmup_dataloader") is warmup_loader))
            events.append(
                ("overlap_train_reader", kwargs.get("overlap_train_reader_during_warmup"))
            )
            self.items = [
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata={"data_time": 0.0},
                )
            ]

        def start(self):
            pass

        def wait_warmup_complete(self, timeout=None):
            del timeout

        def pop_oldest(self):
            return self.items.pop(0)

        def stop(self):
            pass

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(trainer_loop_module, "AsyncGpuFeatureProducer", _FakeAsync, raising=False)

    trainer.train_step_from_features = lambda encoded, optimizer, **kwargs: {
        "main_loss": torch.tensor(0.0),
        "loss_fm": torch.tensor(0.0),
        "loss_rollout_action": torch.tensor(0.0),
        "loss_total": torch.tensor(0.0),
        "rollout_action_weight": 0.0,
        "dataset_slug": "unit",
        "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
        "dataset_count_by_slug": {"unit": 1},
    }
    trainer.run(dataloader=[{"batch_id": "train"}], warmup_dataloader=warmup_loader, max_steps=1)

    assert ("warmup_loader", True) in events
    assert ("overlap_train_reader", True) in events


def test_run_logs_async_raw_batch_memory_from_encoded_metadata(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    logged = []
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda payload, step=None: logged.append((step, payload)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
        log_timing_every=0,
        gpu_preprocess=True,
        async_producer=True,
        cpu_prefetch_depth=1,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg, image_preprocessor=object())
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )

    class _FakeProducer:
        def __init__(self, **kwargs):
            del kwargs

    class _FakeAsync:
        def __init__(self, **kwargs):
            del kwargs
            self.items = [
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={"h2d_ms": 0.0, "preprocess_ms": 0.0, "encoder_ms": 0.0},
                    metadata={
                        "data_time": 0.0,
                        "tensor_bytes.total": 16,
                        "raw_image_bytes.vision": 12,
                        "raw_image_bytes.vlm": 20,
                        "raw_image_bytes.total": 32,
                    },
                )
            ]

        def start(self):
            pass

        def wait_warmup_complete(self, timeout=None):
            pass

        def pop_oldest(self):
            return self.items.pop(0)

        def stop(self):
            pass

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(trainer_loop_module, "AsyncGpuFeatureProducer", _FakeAsync, raising=False)

    def fake_train_step(encoded, optimizer, *, defer_logging_tensors=False, global_step=0):
        del encoded, optimizer, defer_logging_tensors, global_step
        return {
            "main_loss": torch.tensor(0.0),
            "loss_fm": torch.tensor(0.0),
            "loss_rollout_action": torch.tensor(0.0),
            "loss_total": torch.tensor(0.0),
            "rollout_action_weight": 0.0,
            "dataset_slug": "unit",
            "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
            "dataset_count_by_slug": {"unit": 1},
        }

    trainer.train_step_from_features = fake_train_step
    trainer.run(dataloader=[{"unused": True}], max_steps=1)

    train_payloads = [payload for _step, payload in logged if "train/loss_main" in payload]
    assert train_payloads
    assert train_payloads[0]["perf/raw_image_bytes.total"] == 32
    assert train_payloads[0]["perf/tensor_bytes.total"] == 16


def test_run_logs_async_stall_from_main_thread_waits(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    logged = []
    fake_wandb = types.SimpleNamespace(
        run=object(),
        log=lambda payload, step=None: logged.append((step, payload)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.log_every = 1
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
        log_timing_every=0,
        gpu_preprocess=True,
        async_producer=True,
        cpu_prefetch_depth=1,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg, image_preprocessor=object())
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )

    class _FakeProducer:
        def __init__(self, **kwargs):
            del kwargs

    class _FakeAsync:
        def __init__(self, **kwargs):
            del kwargs
            self.items = [
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata={"data_time": 10.0},
                )
            ]

        def start(self):
            pass

        def wait_warmup_complete(self, timeout=None):
            pass

        def pop_oldest(self):
            time.sleep(0.03)
            return self.items.pop(0)

        def stop(self):
            pass

        def __len__(self):
            return len(self.items)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(trainer_loop_module, "AsyncGpuFeatureProducer", _FakeAsync, raising=False)

    trainer.train_step_from_features = lambda encoded, optimizer, **kwargs: {
        "main_loss": torch.tensor(0.0),
        "loss_fm": torch.tensor(0.0),
        "loss_rollout_action": torch.tensor(0.0),
        "loss_total": torch.tensor(0.0),
        "rollout_action_weight": 0.0,
        "dataset_slug": "unit",
        "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
        "dataset_count_by_slug": {"unit": 1},
    }
    trainer.run(dataloader=[{"unused": True}], max_steps=1)

    train_payloads = [payload for _step, payload in logged if "train/loss_main" in payload]
    assert train_payloads
    assert train_payloads[0]["perf/avg_producer_wait_time"] >= 0.02
    assert train_payloads[0]["perf/avg_stall_time"] >= 0.02
    assert train_payloads[0]["perf/stall_ratio"] > 0.0


def test_run_prefetches_next_feature_batch_before_training_current(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    events = []

    class _FakeEncoded:
        def __init__(self, batch_id):
            self.batch_id = batch_id
            self.timings = {}
            self.metadata = {"data_time": 0.0}

        def wait_ready(self, device):
            events.append(("wait", self.batch_id, device.type))

        def collect_timings(self, *, block=False):
            events.append(("timings", self.batch_id, block))
            return {}

    class _FakeProducer:
        def __init__(self, *, queue_depth, **kwargs):
            events.append(("init", queue_depth))
            self.queue = []
            self.queue_depth = queue_depth

        def can_submit(self):
            return len(self.queue) < self.queue_depth

        def submit(self, batch, *, metadata=None):
            del metadata
            events.append(("submit", batch["batch_id"]))
            self.queue.append(_FakeEncoded(batch["batch_id"]))

        def pop_oldest(self):
            encoded = self.queue.pop(0)
            events.append(("pop", encoded.batch_id))
            return encoded

        def __len__(self):
            return len(self.queue)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)

    def fake_train_step(encoded, optimizer, *, defer_logging_tensors=False, global_step=0):
        del optimizer, defer_logging_tensors, global_step
        events.append(("train", encoded.batch_id))
        return {
            "main_loss": torch.tensor(0.0),
            "loss_fm": torch.tensor(0.0),
            "loss_rollout_action": torch.tensor(0.0),
            "loss_total": torch.tensor(0.0),
            "rollout_action_weight": 0.0,
            "dataset_slug": "unit",
            "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
            "dataset_count_by_slug": {"unit": 1},
        }

    trainer.train_step_from_features = fake_train_step
    dataloader = [{"batch_id": 0}, {"batch_id": 1}, {"batch_id": 2}]

    trainer.run(dataloader=dataloader, max_steps=2)

    assert events[:8] == [
        ("init", 1),
        ("submit", 0),
        ("pop", 0),
        ("submit", 1),
        ("wait", 0, "cuda"),
        ("train", 0),
        ("timings", 0, False),
        ("pop", 1),
    ]
    assert events[8:11] == [
        ("wait", 1, "cuda"),
        ("train", 1),
        ("timings", 1, False),
    ]


def test_run_honors_gpu_pipeline_log_timing_every(monkeypatch, tmp_path):
    import servovla.trainer.trainer_loop as trainer_loop_module

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.log_every = 99
    cfg.gpu_pipeline = SimpleNamespace(
        enabled=True,
        feature_queue_depth=1,
        encoder_parallel_streams=False,
        log_timing_every=3,
    )
    trainer = TrainerLoop(model=_end2end_model(), train_cfg=cfg)
    trainer.device = torch.device("cuda")
    trainer.build_scheduler = lambda optimizer: SimpleNamespace(
        step=lambda: None,
        get_last_lr=lambda: [1e-3],
        state_dict=lambda: {},
    )
    timing_logs = []

    class _FakeEncoded:
        def __init__(self, batch_id):
            self.batch_id = batch_id
            self.timings = {}
            self.metadata = {"data_time": 0.0}

        def wait_ready(self, device):
            pass

        def collect_timings(self, *, block=False):
            return {"h2d_ms": 10.0, "encoder_ms": 20.0}

    class _FakeProducer:
        def __init__(self, *, queue_depth, **kwargs):
            self.queue = []
            self.queue_depth = queue_depth

        def can_submit(self):
            return len(self.queue) < self.queue_depth

        def submit(self, batch, *, metadata=None):
            del metadata
            self.queue.append(_FakeEncoded(batch["batch_id"]))

        def pop_oldest(self):
            return self.queue.pop(0)

        def __len__(self):
            return len(self.queue)

    monkeypatch.setattr(trainer_loop_module, "SingleGpuFeatureProducer", _FakeProducer)
    monkeypatch.setattr(
        trainer_loop_module.logger,
        "info",
        lambda message, *args: (
            timing_logs.append((message, args)) if "gpu_pipeline timing" in message else None
        ),
    )

    def fake_train_step(encoded, optimizer, *, defer_logging_tensors=False, global_step=0):
        del encoded, optimizer, defer_logging_tensors, global_step
        return {
            "main_loss": torch.tensor(0.0),
            "loss_fm": torch.tensor(0.0),
            "loss_rollout_action": torch.tensor(0.0),
            "loss_total": torch.tensor(0.0),
            "rollout_action_weight": 0.0,
            "dataset_slug": "unit",
            "dataset_loss_by_slug": {"unit": torch.tensor(0.0)},
            "dataset_count_by_slug": {"unit": 1},
        }

    trainer.train_step_from_features = fake_train_step

    trainer.run(dataloader=[{"batch_id": 0}, {"batch_id": 1}, {"batch_id": 2}], max_steps=3)

    assert len(timing_logs) == 1
    assert timing_logs[0][1][0] == 3


def test_encoded_feature_batch_keeps_dataset_slug_as_cpu_metadata():
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    batch = EncodedFeatureBatch(
        f_vision=torch.zeros(1, 1, 5),
        c_sem=torch.zeros(1, 2, 6),
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
        action=torch.zeros(1, 4, 2),
        loss_mask=torch.ones(1, 4),
        q_current=torch.zeros(1, 3),
        frame_delay=torch.zeros(1),
        dataset_slug=["unit"],
        timings={},
    )

    assert batch.dataset_slug == ["unit"]
    assert not torch.is_tensor(batch.dataset_slug)


def test_encoded_feature_batch_waits_with_cuda_event_without_host_sync(monkeypatch):
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    events = []

    class _FakeEvent:
        def synchronize(self):
            raise AssertionError("wait_ready must not synchronize the host")

        def query(self):
            return True

    class _FakeStream:
        def wait_event(self, event):
            events.append(event)

    ready_event = _FakeEvent()
    fake_stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: fake_stream)

    batch = EncodedFeatureBatch(
        f_vision=torch.zeros(1, 1, 5),
        c_sem=torch.zeros(1, 2, 6),
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
        action=torch.zeros(1, 4, 2),
        loss_mask=torch.ones(1, 4),
        q_current=torch.zeros(1, 3),
        frame_delay=torch.zeros(1),
        dataset_slug=["unit"],
        ready_event=ready_event,
        timings={},
    )

    batch.wait_ready(torch.device("cuda"))

    assert events == [ready_event]


def test_encoded_feature_batch_reports_split_encoder_timings(monkeypatch):
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    class _FakeEvent:
        def __init__(self, name):
            self.name = name

        def query(self):
            return True

        def elapsed_time(self, other):
            return {
                ("start", "h2d"): 5.0,
                ("h2d", "preprocess"): 7.0,
                ("preprocess", "vision"): 11.0,
                ("preprocess", "vlm"): 13.0,
                ("preprocess", "encode"): 17.0,
            }[(self.name, other.name)]

    batch = EncodedFeatureBatch(
        f_vision=torch.zeros(1, 1, 5),
        c_sem=torch.zeros(1, 2, 6),
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
        action=torch.zeros(1, 4, 2),
        loss_mask=torch.ones(1, 4),
        q_current=torch.zeros(1, 3),
        frame_delay=torch.zeros(1),
        dataset_slug=["unit"],
        timings={},
        start_event=_FakeEvent("start"),
        after_h2d_event=_FakeEvent("h2d"),
        after_preprocess_event=_FakeEvent("preprocess"),
        after_encode_event=_FakeEvent("encode"),
        after_vision_encode_event=_FakeEvent("vision"),
        after_vlm_encode_event=_FakeEvent("vlm"),
    )

    timings = batch.collect_timings(block=False)

    assert timings["h2d_ms"] == 5.0
    assert timings["preprocess_ms"] == 7.0
    assert timings["encoder_ms"] == 17.0
    assert timings["vision_encoder_ms"] == 11.0
    assert timings["vlm_encoder_ms"] == 13.0


def test_feature_producer_preprocesses_raw_batches_before_encoding(monkeypatch):
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    events = []

    class _FakeStream:
        def __init__(self, *, device=None):
            self.device = device

        def wait_event(self, event):
            events.append(("wait_event", event.name))

    class _FakeEvent:
        def __init__(self, *, enable_timing=False):
            del enable_timing
            self.name = f"event-{len(events)}"

        def record(self):
            events.append(("record", self.name))

    class _FakeTensor:
        is_cuda = True

        def __init__(self, name):
            self.name = name

        def bool(self):
            return self

        def record_stream(self, stream):
            pass

    class _FakePreprocessor:
        def preprocess_batch(self, batch):
            events.append(("preprocess", "raw"))
            return {
                **batch,
                "pixel_values": torch.zeros(1),
                "vlm_inputs": {
                    "input_ids": torch.ones(1, dtype=torch.long),
                    "pixel_values": torch.zeros(1),
                },
            }

    class _FakeModel:
        def encode_observations(self, *, pixel_values, vlm_inputs):
            events.append(("encode", pixel_values.name, vlm_inputs["pixel_values"].name))
            return _FakeTensor("f_vision"), _FakeTensor("c_sem")

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    producer = SingleGpuFeatureProducer(
        model=_FakeModel(),
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        queue_depth=1,
        encoder_parallel_streams=True,
        image_preprocessor=_FakePreprocessor(),
    )
    batch = {
        "vision_images_uint8": {"groups": []},
        "vlm_images_uint8": {"groups": []},
        "vlm_inputs": {"input_ids": torch.ones(1, dtype=torch.long)},
        "action": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "c_sem_mask": torch.ones(1),
        "q_current": torch.zeros(1),
        "frame_delay": torch.zeros(1),
    }
    names_by_id = {}

    def fake_move(value, dtype=None):
        del dtype
        name = names_by_id.setdefault(id(value), f"tensor-{len(names_by_id)}")
        return _FakeTensor(name)

    producer._move_tensor = fake_move
    producer.submit(batch)
    encoded = producer.pop_oldest()

    assert ("preprocess", "raw") in events
    assert any(event[0] == "encode" for event in events)
    assert encoded.after_preprocess_event is not None


def test_raw_feature_producer_splits_preprocess_and_encoder_streams(monkeypatch):
    from servovla.trainer import gpu_pipeline as gpu_pipeline_module
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    calls = []
    streams = []
    active_streams = []

    class _FakeStream:
        def __init__(self, *, device=None):
            del device
            self.name = f"stream{len(streams)}"
            streams.append(self)

        def wait_event(self, event):
            calls.append(("wait_event", self.name, getattr(event, "name", None)))

    class _FakeEvent:
        def __init__(self, *, enable_timing=False):
            del enable_timing
            self.name = f"event{len(calls)}"

        def record(self):
            calls.append(("record", active_streams[-1], self.name))

    class _FakeTensor:
        is_cuda = True

        def __init__(self, name):
            self.name = name

        def bool(self):
            return self

        def record_stream(self, stream):
            calls.append(("record_stream", self.name, stream.name))

    class _FakePreprocessor:
        def move_raw_batch_to_device(self, batch):
            calls.append(("move_raw", active_streams[-1]))
            return batch

        def preprocess_batch(self, batch):
            del batch
            raise AssertionError("raw preprocess should split vision and VLM work")

        def preprocess_vision(self, grouped):
            del grouped
            calls.append(("preprocess_vision", active_streams[-1]))
            return torch.zeros(1)

        def preprocess_vlm_images(self, grouped):
            del grouped
            calls.append(("preprocess_vlm", active_streams[-1]))
            return torch.zeros(1)

    class _FakeVisionEncoder:
        def __call__(self, pixel_values):
            calls.append(("vision", active_streams[-1], pixel_values.name))
            return _FakeTensor("f_vision")

    class _FakeVlmEncoder:
        def __call__(self, vlm_inputs):
            calls.append(("vlm", active_streams[-1], vlm_inputs["pixel_values"].name))
            return _FakeTensor("c_sem")

    class _FakeModel:
        vision_encoder = _FakeVisionEncoder()
        vlm_encoder = _FakeVlmEncoder()

        def encode_observations(self, *, pixel_values, vlm_inputs):
            calls.append(
                ("combined", active_streams[-1], pixel_values.name, vlm_inputs["pixel_values"].name)
            )
            return _FakeTensor("f_vision"), _FakeTensor("c_sem")

    @contextmanager
    def fake_stream_context(stream):
        active_streams.append(stream.name)
        try:
            yield
        finally:
            active_streams.pop()

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_context)

    def fake_locked_vlm_forward(vlm_encoder, vlm_inputs):
        calls.append(("locked_vlm", active_streams[-1]))
        return vlm_encoder(vlm_inputs)

    monkeypatch.setattr(
        gpu_pipeline_module,
        "run_vlm_encoder_forward",
        fake_locked_vlm_forward,
        raising=False,
    )

    producer = SingleGpuFeatureProducer(
        model=_FakeModel(),
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        queue_depth=1,
        encoder_parallel_streams=True,
        image_preprocessor=_FakePreprocessor(),
    )
    batch = {
        "vision_images_uint8": {"groups": []},
        "vlm_images_uint8": {"groups": []},
        "vlm_inputs": {"input_ids": torch.ones(1, dtype=torch.long)},
        "action": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "c_sem_mask": torch.ones(1),
        "q_current": torch.zeros(1),
        "frame_delay": torch.zeros(1),
    }
    names_by_id = {}

    def fake_move(value, dtype=None):
        del dtype
        name = names_by_id.setdefault(id(value), f"tensor-{len(names_by_id)}")
        return _FakeTensor(name)

    producer._move_tensor = fake_move

    producer.submit(batch)

    preprocess_calls = [
        call for call in calls if call[0] in {"preprocess_vision", "preprocess_vlm"}
    ]
    encoder_calls = [call for call in calls if call[0] in {"vision", "vlm", "combined"}]
    assert [call[0] for call in preprocess_calls] == ["preprocess_vision", "preprocess_vlm"]
    assert preprocess_calls[0][1] != preprocess_calls[1][1]
    assert preprocess_calls[0][1] != producer.h2d_stream.name
    assert preprocess_calls[1][1] != producer.h2d_stream.name
    assert any(call[0] == "locked_vlm" for call in calls)
    assert [call[0] for call in encoder_calls] == ["vision", "vlm"]
    assert encoder_calls[0][1] != encoder_calls[1][1]


def test_vlm_encoder_micro_batch_splits_inputs_on_sample_boundaries(monkeypatch):
    from servovla.trainer import gpu_pipeline as gpu_pipeline_module

    calls = []
    batch_size = 4
    images_per_sample = 3
    patches_per_image = 5
    hidden_dim = 2
    seq_len = 6
    input_ids = torch.arange(batch_size * seq_len).view(batch_size, seq_len)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    image_grid_thw = torch.ones(batch_size * images_per_sample, 3, dtype=torch.long)
    pixel_values = torch.arange(
        batch_size * images_per_sample * patches_per_image,
        dtype=torch.float32,
    ).view(batch_size * images_per_sample * patches_per_image, 1)
    vlm_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_token_type_ids,
        "image_grid_thw": image_grid_thw,
        "pixel_values": pixel_values,
    }

    def fake_forward(_vlm_encoder, chunk_inputs):
        calls.append(
            {
                "input_ids": chunk_inputs["input_ids"].clone(),
                "image_grid_thw": chunk_inputs["image_grid_thw"].clone(),
                "pixel_values": chunk_inputs["pixel_values"].clone(),
            }
        )
        token_values = chunk_inputs["input_ids"].to(dtype=torch.float32).unsqueeze(-1)
        return token_values.expand(-1, -1, hidden_dim)

    monkeypatch.setattr(gpu_pipeline_module, "run_vlm_encoder_forward", fake_forward)

    encoded = gpu_pipeline_module.run_vlm_encoder_forward_micro_batched(
        object(),
        vlm_inputs,
        micro_batch_size=2,
    )

    assert len(calls) == 2
    assert torch.equal(calls[0]["input_ids"], input_ids[:2])
    assert torch.equal(calls[1]["input_ids"], input_ids[2:])
    assert calls[0]["image_grid_thw"].shape[0] == 2 * images_per_sample
    assert calls[1]["image_grid_thw"].shape[0] == 2 * images_per_sample
    assert torch.equal(
        calls[0]["pixel_values"], pixel_values[: 2 * images_per_sample * patches_per_image]
    )
    assert torch.equal(
        calls[1]["pixel_values"], pixel_values[2 * images_per_sample * patches_per_image :]
    )
    assert torch.equal(
        encoded, input_ids.to(dtype=torch.float32).unsqueeze(-1).expand(-1, -1, hidden_dim)
    )


def test_vision_encoder_micro_batch_splits_pixel_values_on_sample_boundaries():
    from servovla.trainer import gpu_pipeline as gpu_pipeline_module

    calls = []
    batch_size = 4
    num_views = 3
    patches = 5
    hidden_dim = 2
    pixel_values = torch.arange(
        batch_size * num_views * 3 * 4 * 4,
        dtype=torch.float32,
    ).view(batch_size, num_views, 3, 4, 4)

    class _VisionEncoder:
        def __call__(self, chunk_pixel_values):
            calls.append(chunk_pixel_values.clone())
            chunk_size = int(chunk_pixel_values.shape[0])
            values = chunk_pixel_values[:, :, 0, 0, 0].reshape(chunk_size, num_views, 1, 1)
            return values.expand(chunk_size, num_views, patches, hidden_dim).reshape(
                chunk_size,
                num_views * patches,
                hidden_dim,
            )

    encoded = gpu_pipeline_module.run_vision_encoder_forward_micro_batched(
        _VisionEncoder(),
        pixel_values,
        micro_batch_size=2,
    )

    assert len(calls) == 2
    assert torch.equal(calls[0], pixel_values[:2])
    assert torch.equal(calls[1], pixel_values[2:])

    expected = pixel_values[:, :, 0, 0, 0].reshape(batch_size, num_views, 1, 1)
    expected = expected.expand(batch_size, num_views, patches, hidden_dim).reshape(
        batch_size,
        num_views * patches,
        hidden_dim,
    )
    assert torch.equal(encoded, expected)


def test_async_feature_producer_owns_iterator_and_returns_encoded_batches():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    events = []

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            events.append(("submit", batch["batch_id"], "data_time" in metadata))
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[{"batch_id": 0}, {"batch_id": 1}],
        feature_producer=_Producer(),
        feature_queue_depth=2,
        cpu_prefetch_depth=1,
        max_steps=2,
    )
    producer.start()
    first = producer.pop_oldest()
    second = producer.pop_oldest()
    producer.stop()

    assert first.metadata["data_time"] >= 0.0
    assert second.metadata["data_time"] >= 0.0
    assert [event[1] for event in events] == [0, 1]


def test_async_feature_producer_prefills_internal_feature_queue_when_cpu_batches_are_ready():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    events = []

    class _Producer:
        def __init__(self):
            self.items = []
            self.queue_depth = 2

        def can_submit(self):
            return len(self.items) < self.queue_depth

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            events.append(("submit", batch_id))
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            events.append(("pop", int(self.items[0].f_vision.flatten()[0].item())))
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[{"batch_id": 0}, {"batch_id": 1}, {"batch_id": 2}],
        feature_producer=_Producer(),
        feature_queue_depth=3,
        cpu_prefetch_depth=3,
        max_steps=3,
    )
    producer.start()
    encoded_ids = [int(producer.pop_oldest().f_vision.flatten()[0].item()) for _ in range(3)]
    producer.stop()

    assert encoded_ids == [0, 1, 2]
    assert events.index(("submit", 1)) < events.index(("pop", 0))


def test_async_feature_producer_reports_internal_timing_metadata():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[{"batch_id": 0}],
        feature_producer=_Producer(),
        feature_queue_depth=1,
        cpu_prefetch_depth=1,
        max_steps=1,
    )
    producer.start()
    encoded = producer.pop_oldest()
    producer.stop()

    for key in [
        "producer_timing.reader_data_time",
        "producer_timing.encoder_cpu_queue_wait",
        "producer_timing.submit_time",
        "producer_timing.feature_queue_put_wait",
    ]:
        assert key in encoded.metadata
        assert float(encoded.metadata[key]) >= 0.0


def test_async_feature_producer_discards_warmup_batches_without_reducing_training_count():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    submitted_ids = []

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            submitted_ids.append(batch_id)
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[
            {"batch_id": 0},
            {"batch_id": 1},
            {"batch_id": 2},
            {"batch_id": 3},
            {"batch_id": 4},
        ],
        feature_producer=_Producer(),
        feature_queue_depth=2,
        cpu_prefetch_depth=1,
        max_steps=2,
        vlm_warmup_max_batches=3,
    )
    producer.start()
    producer.wait_warmup_complete(timeout=2.0)
    encoded_ids = [int(producer.pop_oldest().f_vision.flatten()[0].item()) for _ in range(2)]
    producer.stop()

    assert encoded_ids == [3, 4]
    assert submitted_ids == [0, 1, 2, 3, 4]


def test_async_feature_producer_can_use_separate_warmup_dataloader():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    submitted_ids = []

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            submitted_ids.append(batch_id)
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[{"batch_id": 100}, {"batch_id": 101}],
        warmup_dataloader=[{"batch_id": 0}, {"batch_id": 1}, {"batch_id": 2}],
        feature_producer=_Producer(),
        feature_queue_depth=2,
        cpu_prefetch_depth=1,
        max_steps=2,
        vlm_warmup_max_batches=3,
    )
    producer.start()
    producer.wait_warmup_complete(timeout=2.0)
    encoded_ids = [int(producer.pop_oldest().f_vision.flatten()[0].item()) for _ in range(2)]
    producer.stop()

    assert encoded_ids == [100, 101]
    assert submitted_ids == [0, 1, 2, 100, 101]


def test_async_feature_producer_waits_for_separate_warmup_before_train_reader():
    import threading

    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    train_iterated = threading.Event()
    allow_warmup_pop = threading.Event()

    class _TrainLoader:
        def __iter__(self):
            train_iterated.set()
            return iter([{"batch_id": 100}])

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            batch_id = int(self.items[0].f_vision.flatten()[0].item())
            if batch_id == 0:
                assert allow_warmup_pop.wait(timeout=2.0)
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=_TrainLoader(),
        warmup_dataloader=[{"batch_id": 0}],
        feature_producer=_Producer(),
        feature_queue_depth=2,
        cpu_prefetch_depth=2,
        max_steps=1,
        vlm_warmup_max_batches=1,
    )
    producer.start()
    assert not train_iterated.wait(timeout=0.1)
    allow_warmup_pop.set()
    producer.wait_warmup_complete(timeout=2.0)
    encoded = producer.pop_oldest()
    producer.stop()

    assert int(encoded.f_vision.flatten()[0].item()) == 100


def test_async_feature_producer_can_overlap_train_reader_with_separate_warmup():
    import threading

    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    train_iterated = threading.Event()
    allow_warmup_pop = threading.Event()

    class _TrainLoader:
        def __iter__(self):
            train_iterated.set()
            return iter([{"batch_id": 100}])

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            batch_id = int(batch["batch_id"])
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.full((1, 1, 5), float(batch_id)),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            batch_id = int(self.items[0].f_vision.flatten()[0].item())
            if batch_id == 0:
                assert allow_warmup_pop.wait(timeout=2.0)
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=_TrainLoader(),
        warmup_dataloader=[{"batch_id": 0}],
        feature_producer=_Producer(),
        feature_queue_depth=2,
        cpu_prefetch_depth=2,
        max_steps=1,
        vlm_warmup_max_batches=1,
        overlap_train_reader_during_warmup=True,
    )
    producer.start()
    assert train_iterated.wait(timeout=2.0)
    allow_warmup_pop.set()
    producer.wait_warmup_complete(timeout=2.0)
    encoded = producer.pop_oldest()
    producer.stop()

    assert int(encoded.f_vision.flatten()[0].item()) == 100


def test_runtime_vlm_signature_error_policy_raises_clear_error():
    from servovla.trainer.vlm_compile_warmup import (
        VlmSignatureRegistry,
        handle_runtime_vlm_signature,
        make_vlm_input_signature,
    )

    registry = VlmSignatureRegistry(max_entries=4)
    first = {
        "vlm_inputs": {
            "input_ids": torch.zeros(1, 64, dtype=torch.long),
            "attention_mask": torch.ones(1, 64, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 18, 18]], dtype=torch.long),
        }
    }
    second = {
        "vlm_inputs": {
            "input_ids": torch.zeros(1, 80, dtype=torch.long),
            "attention_mask": torch.ones(1, 80, dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 18, 18]], dtype=torch.long),
        }
    }
    registry.mark_warmed(make_vlm_input_signature(first))

    with pytest.raises(RuntimeError, match="Unexpected VLM input signature"):
        handle_runtime_vlm_signature(
            second,
            registry=registry,
            on_new_signature="error",
            logger_name="test",
        )


def test_async_feature_producer_attaches_batch_metadata_from_callback():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer, EncodedFeatureBatch

    class _Producer:
        def __init__(self):
            self.items = []

        def submit(self, batch, *, metadata=None):
            self.items.append(
                EncodedFeatureBatch(
                    f_vision=torch.zeros(1, 1, 5),
                    c_sem=torch.zeros(1, 2, 6),
                    c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
                    action=torch.zeros(1, 4, 2),
                    loss_mask=torch.ones(1, 4),
                    q_current=torch.zeros(1, 3),
                    frame_delay=torch.zeros(1),
                    dataset_slug=["unit"],
                    timings={},
                    metadata=dict(metadata or {}),
                )
            )

        def pop_oldest(self):
            return self.items.pop(0)

    producer = AsyncGpuFeatureProducer(
        dataloader=[{"batch_id": 0}],
        feature_producer=_Producer(),
        feature_queue_depth=1,
        cpu_prefetch_depth=1,
        max_steps=1,
        batch_metadata_fn=lambda batch: {
            "tensor_bytes.total": 16,
            "raw_image_bytes.total": int(batch["batch_id"]) + 32,
        },
    )
    producer.start()
    encoded = producer.pop_oldest()
    producer.stop()

    assert encoded.metadata["data_time"] >= 0.0
    assert encoded.metadata["tensor_bytes.total"] == 16
    assert encoded.metadata["raw_image_bytes.total"] == 32


def test_async_feature_producer_propagates_iterator_exceptions():
    from servovla.trainer.gpu_pipeline import AsyncGpuFeatureProducer

    class _BadLoader:
        def __iter__(self):
            raise RuntimeError("loader failed")

    class _Producer:
        def submit(self, batch, *, metadata=None):
            raise AssertionError("submit should not be reached")

        def pop_oldest(self):
            raise AssertionError("pop should not be reached")

    producer = AsyncGpuFeatureProducer(
        dataloader=_BadLoader(),
        feature_producer=_Producer(),
        feature_queue_depth=1,
        cpu_prefetch_depth=1,
        max_steps=1,
    )
    producer.start()
    with pytest.raises(RuntimeError, match="loader failed"):
        producer.pop_oldest()
    producer.stop()


def test_feature_producer_records_encoder_inputs_on_encoder_stream(monkeypatch):
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    recorded = []
    streams = []

    class _FakeStream:
        def __init__(self, *, device=None):
            del device
            self.name = f"stream{len(streams)}"
            streams.append(self)

        def wait_event(self, event):
            pass

    class _FakeEvent:
        def __init__(self, *, enable_timing=False):
            del enable_timing

        def record(self):
            pass

    class _FakeTensor:
        is_cuda = True

        def __init__(self, name):
            self.name = name

        def bool(self):
            return self

        def record_stream(self, stream):
            recorded.append((self.name, stream.name))

    class _FakeModel:
        def encode_observations(self, *, pixel_values, vlm_inputs):
            assert pixel_values.name == "pixel_values"
            assert vlm_inputs["input_ids"].name == "input_ids"
            return _FakeTensor("f_vision"), _FakeTensor("c_sem")

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    producer = SingleGpuFeatureProducer(
        model=_FakeModel(),
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        queue_depth=1,
        encoder_parallel_streams=True,
    )
    batch = {
        "pixel_values": torch.zeros(1),
        "vlm_inputs": {
            "input_ids": torch.zeros(1, dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
        },
        "action": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "c_sem_mask": torch.ones(1),
        "q_current": torch.zeros(1),
        "frame_delay": torch.zeros(1),
    }
    names_by_id = {
        id(batch["pixel_values"]): "pixel_values",
        id(batch["vlm_inputs"]["input_ids"]): "input_ids",
        id(batch["vlm_inputs"]["attention_mask"]): "attention_mask",
        id(batch["action"]): "action",
        id(batch["loss_mask"]): "loss_mask",
        id(batch["c_sem_mask"]): "c_sem_mask",
        id(batch["q_current"]): "q_current",
        id(batch["frame_delay"]): "frame_delay",
    }
    producer._move_tensor = lambda value, dtype=None: _FakeTensor(names_by_id[id(value)])

    producer.submit(batch)

    encoder_stream_name = producer.encoder_stream.name
    assert ("pixel_values", encoder_stream_name) in recorded
    assert ("input_ids", encoder_stream_name) in recorded
    assert ("attention_mask", encoder_stream_name) in recorded


def test_feature_producer_splits_frozen_encoders_across_streams(monkeypatch):
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    calls = []
    streams = []
    active_streams = []

    class _FakeStream:
        def __init__(self, *, device=None):
            del device
            self.name = f"stream{len(streams)}"
            streams.append(self)

        def wait_event(self, event):
            calls.append(("wait_event", self.name, getattr(event, "name", None)))

    class _FakeEvent:
        def __init__(self, *, enable_timing=False):
            del enable_timing
            self.name = f"event{len(calls)}"

        def record(self):
            calls.append(("record", active_streams[-1], self.name))

    class _FakeTensor:
        is_cuda = True

        def __init__(self, name):
            self.name = name

        def bool(self):
            return self

        def record_stream(self, stream):
            calls.append(("record_stream", self.name, stream.name))

    class _FakeVisionEncoder:
        def __call__(self, pixel_values):
            calls.append(("vision", active_streams[-1], pixel_values.name))
            return _FakeTensor("f_vision")

    class _FakeVlmEncoder:
        def __call__(self, vlm_inputs):
            calls.append(("vlm", active_streams[-1], vlm_inputs["input_ids"].name))
            return _FakeTensor("c_sem")

    class _FakeModel:
        vision_encoder = _FakeVisionEncoder()
        vlm_encoder = _FakeVlmEncoder()

        def encode_observations(self, *, pixel_values, vlm_inputs):
            calls.append(
                ("combined", active_streams[-1], pixel_values.name, vlm_inputs["input_ids"].name)
            )
            return _FakeTensor("f_vision"), _FakeTensor("c_sem")

    @contextmanager
    def fake_stream_context(stream):
        active_streams.append(stream.name)
        try:
            yield
        finally:
            active_streams.pop()

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", fake_stream_context)

    producer = SingleGpuFeatureProducer(
        model=_FakeModel(),
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        queue_depth=1,
        encoder_parallel_streams=True,
    )
    batch = {
        "pixel_values": torch.zeros(1),
        "vlm_inputs": {
            "input_ids": torch.zeros(1, dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
        },
        "action": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "c_sem_mask": torch.ones(1),
        "q_current": torch.zeros(1),
        "frame_delay": torch.zeros(1),
    }
    names_by_id = {
        id(batch["pixel_values"]): "pixel_values",
        id(batch["vlm_inputs"]["input_ids"]): "input_ids",
        id(batch["vlm_inputs"]["attention_mask"]): "attention_mask",
        id(batch["action"]): "action",
        id(batch["loss_mask"]): "loss_mask",
        id(batch["c_sem_mask"]): "c_sem_mask",
        id(batch["q_current"]): "q_current",
        id(batch["frame_delay"]): "frame_delay",
    }
    producer._move_tensor = lambda value, dtype=None: _FakeTensor(names_by_id[id(value)])

    producer.submit(batch)

    encoder_calls = [call for call in calls if call[0] in {"combined", "vision", "vlm"}]
    assert [call[0] for call in encoder_calls] == ["vision", "vlm"]
    assert encoder_calls[0][1] != encoder_calls[1][1]
    assert encoder_calls[0][1] != streams[0].name
    assert encoder_calls[1][1] != streams[0].name


def test_feature_producer_reuses_h2d_stream_when_encoder_parallel_disabled(monkeypatch):
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    recorded = []
    streams = []

    class _FakeStream:
        def __init__(self, *, device=None):
            del device
            self.name = f"stream{len(streams)}"
            streams.append(self)

        def wait_event(self, event):
            pass

    class _FakeEvent:
        def __init__(self, *, enable_timing=False):
            del enable_timing

        def record(self):
            pass

    class _FakeTensor:
        is_cuda = True

        def __init__(self, name):
            self.name = name

        def bool(self):
            return self

        def record_stream(self, stream):
            recorded.append((self.name, stream.name))

    class _FakeModel:
        def encode_observations(self, *, pixel_values, vlm_inputs):
            return _FakeTensor("f_vision"), _FakeTensor("c_sem")

    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())

    producer = SingleGpuFeatureProducer(
        model=_FakeModel(),
        device=torch.device("cuda"),
        amp_dtype=torch.float32,
        queue_depth=1,
        encoder_parallel_streams=False,
    )
    batch = {
        "pixel_values": torch.zeros(1),
        "vlm_inputs": {
            "input_ids": torch.zeros(1, dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
        },
        "action": torch.zeros(1),
        "loss_mask": torch.ones(1),
        "c_sem_mask": torch.ones(1),
        "q_current": torch.zeros(1),
        "frame_delay": torch.zeros(1),
    }
    names_by_id = {
        id(batch["pixel_values"]): "pixel_values",
        id(batch["vlm_inputs"]["input_ids"]): "input_ids",
        id(batch["vlm_inputs"]["attention_mask"]): "attention_mask",
        id(batch["action"]): "action",
        id(batch["loss_mask"]): "loss_mask",
        id(batch["c_sem_mask"]): "c_sem_mask",
        id(batch["q_current"]): "q_current",
        id(batch["frame_delay"]): "frame_delay",
    }
    producer._move_tensor = lambda value, dtype=None: _FakeTensor(names_by_id[id(value)])

    producer.submit(batch)

    assert len(streams) == 1
    assert ("pixel_values", "stream0") in recorded
    assert ("input_ids", "stream0") in recorded
    assert ("attention_mask", "stream0") in recorded


def test_train_step_from_features_keeps_action_reference_dtype(monkeypatch, tmp_path):
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    model = _end2end_model()
    model.policy_head.to(dtype=torch.bfloat16)
    trainer = TrainerLoop(
        model=model, train_cfg=_cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    )
    batch = EncodedFeatureBatch(
        f_vision=torch.zeros(1, 1, 5, dtype=torch.bfloat16),
        c_sem=torch.zeros(1, 2, 6, dtype=torch.bfloat16),
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
        action=torch.zeros(1, 4, 2, dtype=torch.float32),
        loss_mask=torch.ones(1, 4),
        q_current=torch.zeros(1, 3, dtype=torch.bfloat16),
        frame_delay=torch.zeros(1),
        dataset_slug=["unit"],
        timings={},
    )
    captured = {}

    def fake_policy_core(**kwargs):
        captured["action_dtype"] = kwargs["x_1"].dtype
        return {"main_loss": 0.0}

    monkeypatch.setattr(trainer, "_train_step_policy_core", fake_policy_core)

    trainer.train_step_from_features(batch, object())

    assert captured["action_dtype"] == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_feature_producer_outputs_gpu_resident_features(tmp_path):
    from servovla.trainer.gpu_pipeline import SingleGpuFeatureProducer

    model = _end2end_model().to("cuda")
    producer = SingleGpuFeatureProducer(
        model=model,
        device=torch.device("cuda"),
        amp_dtype=torch.bfloat16,
        queue_depth=1,
        encoder_parallel_streams=False,
    )
    encoded = producer.encode_next(_end2end_batch(batch_size=2))

    assert encoded.f_vision.is_cuda
    assert encoded.c_sem.is_cuda
    assert encoded.action.is_cuda
    assert encoded.dataset_slug == ["unit", "unit"]


def test_rollout_action_loss_weight_uses_cosine_ramp():
    from servovla.trainer.rollout_action_loss import rollout_action_loss_weight

    cfg = SimpleNamespace(
        enabled=True,
        max_weight=0.2,
        start_step=10,
        end_step=30,
        schedule="cosine",
    )

    assert rollout_action_loss_weight(cfg, 9) == 0.0
    assert rollout_action_loss_weight(cfg, 10) == 0.0
    assert rollout_action_loss_weight(cfg, 20) == pytest.approx(0.1)
    assert rollout_action_loss_weight(cfg, 30) == pytest.approx(0.2)
    assert rollout_action_loss_weight(cfg, 99) == pytest.approx(0.2)


def test_differentiable_rollout_action_loss_updates_policy_parameters():
    from servovla.trainer.rollout_action_loss import compute_rollout_action_loss

    class _TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.25))

        def forward(self, *, x_t, t, f_vision, c_sem, c_sem_mask, frame_delay, q_current):
            del t, f_vision, c_sem, c_sem_mask, frame_delay, q_current
            return x_t * self.weight

    policy = _TinyPolicy()
    target = torch.ones(2, 3, 1)
    noise = torch.full_like(target, 0.5)
    loss_mask = torch.ones(2, 3)

    loss = compute_rollout_action_loss(
        policy_head=policy,
        x_1=target,
        x_0=noise,
        loss_mask=loss_mask,
        f_vision=torch.zeros(2, 1, 4),
        c_sem=torch.zeros(2, 1, 4),
        c_sem_mask=torch.ones(2, 1, dtype=torch.bool),
        frame_delay=torch.zeros(2),
        q_current=torch.zeros(2, 6),
        num_inference_steps=3,
        loss_type="smooth_l1",
        beta=0.05,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.weight.grad is not None
    assert torch.isfinite(policy.weight.grad)
    assert policy.weight.grad.abs().item() > 0.0


def test_train_step_from_features_logs_rollout_loss_components(tmp_path):
    from servovla.trainer.gpu_pipeline import EncodedFeatureBatch

    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.compile = False
    cfg.rollout_action_loss = SimpleNamespace(
        enabled=True,
        loss_type="smooth_l1",
        beta=0.05,
        max_weight=0.2,
        start_step=0,
        end_step=10,
        schedule="cosine",
        noise_source="fm_noise",
    )
    model = _end2end_model()
    model.fm_solver.num_inference_steps = 3
    trainer = TrainerLoop(model=model, train_cfg=cfg)
    optimizer = torch.optim.SGD(model.policy_head.parameters(), lr=1e-3)

    encoded = EncodedFeatureBatch(
        f_vision=torch.randn(2, 2, 5),
        c_sem=torch.randn(2, 3, 6),
        c_sem_mask=torch.ones(2, 3, dtype=torch.bool),
        action=torch.randn(2, 4, 2),
        loss_mask=torch.ones(2, 4),
        q_current=torch.randn(2, 3),
        frame_delay=torch.zeros(2),
        dataset_slug=["unit", "unit"],
    )

    result = trainer.train_step_from_features(
        encoded,
        optimizer,
        defer_logging_tensors=False,
        global_step=10,
    )

    assert result["rollout_action_weight"] == pytest.approx(0.2)
    assert result["loss_fm"] >= 0.0
    assert result["loss_rollout_action"] >= 0.0
    assert result["main_loss"] == pytest.approx(result["loss_total"])


def test_train_step_end2end_logs_rollout_loss_components_without_double_encoding(tmp_path):
    cfg = _cfg_with_eval(tmp_path, eval_every=0, save_every=99)
    cfg.compile = False
    cfg.rollout_action_loss = SimpleNamespace(
        enabled=True,
        loss_type="smooth_l1",
        beta=0.05,
        max_weight=0.2,
        start_step=0,
        end_step=10,
        schedule="cosine",
        noise_source="fm_noise",
    )
    model = _end2end_model()
    model.fm_solver.num_inference_steps = 3
    trainer = TrainerLoop(model=model, train_cfg=cfg)
    optimizer = torch.optim.SGD(model.policy_head.parameters(), lr=1e-3)
    batch = _end2end_batch(batch_size=2)
    encode_calls = 0
    original_encode_observations = model.encode_observations

    def _counting_encode_observations(*, pixel_values, vlm_inputs):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode_observations(pixel_values=pixel_values, vlm_inputs=vlm_inputs)

    model.encode_observations = _counting_encode_observations

    result = trainer.train_step_end2end(
        batch,
        optimizer,
        defer_logging_tensors=False,
        global_step=10,
    )

    assert encode_calls == 1
    assert result["rollout_action_weight"] == pytest.approx(0.2)
    assert result["loss_fm"] >= 0.0
    assert result["loss_rollout_action"] >= 0.0
    assert result["main_loss"] == pytest.approx(result["loss_total"])
