from __future__ import annotations

import logging
import threading
import time

import pytest
import torch

from lerobot_policy_servovla.runtime_semantic_delay import SemanticDelayRuntime


def test_runtime_reuses_cached_snapshot_within_delay_bound():
    runtime = SemanticDelayRuntime(
        encode_semantic=lambda vlm_inputs: torch.full((1, 2, 3), float(vlm_inputs["frame_id"])),
        max_frame_delay=4,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=200,
    )

    runtime.submit_latest(
        frame_id=0,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 0},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    assert snapshot.frame_id == 0
    assert frame_delay == 0
    assert waited_ms >= 0

    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(frame_id=4, task_texts=("pick",))
    assert snapshot.frame_id == 0
    assert frame_delay == 4
    assert waited_ms >= 0

    runtime.close()


def test_runtime_blocks_until_fresh_snapshot_when_delay_would_exceed_bound():
    gate = threading.Event()

    def encode_semantic(vlm_inputs):
        if vlm_inputs["frame_id"] == 5:
            gate.wait(timeout=1.0)
        return torch.full((1, 2, 3), float(vlm_inputs["frame_id"]))

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=4,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=500,
    )
    runtime.submit_latest(
        frame_id=0,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 0},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    runtime.submit_latest(
        frame_id=5,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 5},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    threading.Thread(target=lambda: (time.sleep(0.05), gate.set()), daemon=True).start()
    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(frame_id=5, task_texts=("pick",))

    assert snapshot.frame_id == 5
    assert frame_delay == 0
    assert waited_ms > 0

    runtime.close()


def test_runtime_blocks_until_fresh_snapshot_when_delay_fails_support_predicate():
    gate = threading.Event()

    def encode_semantic(vlm_inputs):
        if vlm_inputs["frame_id"] == 1:
            gate.wait(timeout=1.0)
        return torch.full((1, 2, 3), float(vlm_inputs["frame_id"]))

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=4,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=500,
    )
    runtime.submit_latest(
        frame_id=0,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 0},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    runtime.submit_latest(
        frame_id=1,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 1},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    threading.Thread(target=lambda: (time.sleep(0.05), gate.set()), daemon=True).start()
    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(
        frame_id=1,
        task_texts=("pick",),
        delay_is_supported=lambda delay: delay == 0,
    )

    assert snapshot.frame_id == 1
    assert frame_delay == 0
    assert waited_ms > 0

    runtime.close()


def test_runtime_does_not_accept_future_snapshot_when_action_step_goes_backwards():
    gate = threading.Event()

    def encode_semantic(vlm_inputs):
        if vlm_inputs["frame_id"] == 10:
            gate.wait(timeout=1.0)
        return torch.full((1, 2, 3), float(vlm_inputs["frame_id"]))

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=100,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=500,
    )
    runtime.submit_latest(
        frame_id=95,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 95},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=95, task_texts=("pick",))

    runtime.submit_latest(
        frame_id=10,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 10},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    threading.Thread(target=lambda: (time.sleep(0.05), gate.set()), daemon=True).start()
    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(frame_id=10, task_texts=("pick",))

    assert snapshot.frame_id == 10
    assert frame_delay == 0
    assert waited_ms > 0

    runtime.close()


def test_latest_cache_policy_accepts_latest_snapshot_without_task_or_age_checks():
    runtime = SemanticDelayRuntime(
        encode_semantic=lambda vlm_inputs: torch.full((1, 2, 3), float(vlm_inputs["frame_id"])),
        max_frame_delay=4,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=200,
    )
    runtime.submit_latest(
        frame_id=95,
        task_texts=("old task",),
        session_id="old-session",
        vlm_inputs={"frame_id": 95},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=95, task_texts=("old task",), session_id="old-session")

    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(
        frame_id=10,
        task_texts=("new task",),
        session_id="new-session",
        admission_policy="latest_cache",
    )

    assert snapshot.frame_id == 95
    assert snapshot.task_texts == ("old task",)
    assert snapshot.session_id == "old-session"
    assert frame_delay == -85
    assert waited_ms >= 0
    assert runtime.get_last_admission_debug()["admit_reason"] == "admitted_latest_cache"
    runtime.close()


def test_age_only_policy_accepts_age_compatible_snapshot_without_task_or_session_checks():
    runtime = SemanticDelayRuntime(
        encode_semantic=lambda vlm_inputs: torch.full((1, 2, 3), float(vlm_inputs["frame_id"])),
        max_frame_delay=15,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=200,
    )
    runtime.submit_latest(
        frame_id=0,
        task_texts=("old task",),
        session_id="old-session",
        vlm_inputs={"frame_id": 0},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=0, task_texts=("old task",), session_id="old-session")

    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(
        frame_id=15,
        task_texts=("new task",),
        session_id="new-session",
        admission_policy="age_only",
        delay_is_supported=lambda delay: delay in {0, 15},
    )

    assert snapshot.frame_id == 0
    assert snapshot.task_texts == ("old task",)
    assert snapshot.session_id == "old-session"
    assert frame_delay == 15
    assert waited_ms >= 0
    assert runtime.get_last_admission_debug()["admit_reason"] == "admitted_age_only"
    runtime.close()


def test_bsr_policy_rejects_session_mismatch_until_compatible_snapshot_arrives():
    gate = threading.Event()

    def encode_semantic(vlm_inputs):
        if vlm_inputs["frame_id"] == 0 and vlm_inputs["session"] == "new-session":
            gate.wait(timeout=1.0)
        return torch.full((1, 2, 3), float(vlm_inputs["frame_id"]))

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=15,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=500,
    )
    runtime.submit_latest(
        frame_id=0,
        task_texts=("same task",),
        session_id="old-session",
        vlm_inputs={"frame_id": 0, "session": "old-session"},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=0, task_texts=("same task",), session_id="old-session")
    runtime.submit_latest(
        frame_id=0,
        task_texts=("same task",),
        session_id="new-session",
        vlm_inputs={"frame_id": 0, "session": "new-session"},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    threading.Thread(target=lambda: (time.sleep(0.05), gate.set()), daemon=True).start()
    snapshot, frame_delay, waited_ms = runtime.wait_for_snapshot(
        frame_id=0,
        task_texts=("same task",),
        session_id="new-session",
        admission_policy="bsr",
    )

    assert snapshot.session_id == "new-session"
    assert frame_delay == 0
    assert waited_ms > 0
    assert runtime.get_last_admission_debug()["admit_reason"] == "admitted_bsr"
    runtime.close()


def test_runtime_raises_timeout_when_refresh_takes_too_long():
    gate = threading.Event()

    def encode_semantic(vlm_inputs):
        gate.wait(timeout=0.2)
        return torch.zeros(1, 2, 3)

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=0,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=20,
    )
    runtime.submit_latest(
        frame_id=0,
        task_texts=("pick",),
        vlm_inputs={"frame_id": 0},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(TimeoutError, match="semantic"):
        runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    gate.set()
    runtime.close()


def test_runtime_synchronizes_snapshot_ready_event_before_return(monkeypatch):
    class _ReadyEvent:
        def __init__(self):
            self.synchronized = False

        def synchronize(self):
            self.synchronized = True

    ready_event = _ReadyEvent()
    runtime = SemanticDelayRuntime(
        encode_semantic=lambda _vlm_inputs: torch.zeros(1, 2, 3),
        max_frame_delay=0,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=200,
    )
    monkeypatch.setattr(runtime, "_record_ready_event", lambda _tensor: ready_event, raising=False)

    runtime.submit_latest(
        frame_id=0,
        task_texts=("pick",),
        vlm_inputs={},
        c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    assert ready_event.synchronized is True
    runtime.close()


def test_runtime_logs_worker_exception_with_original_error(caplog):
    logger = logging.getLogger("test_runtime_logs_worker_exception")

    def encode_semantic(_vlm_inputs):
        raise ValueError("semantic encoder exploded")

    runtime = SemanticDelayRuntime(
        encode_semantic=encode_semantic,
        max_frame_delay=0,
        semantic_wait_warn_ms=1,
        semantic_wait_fail_ms=200,
        logger=logger,
    )

    with caplog.at_level(logging.ERROR, logger=logger.name):
        runtime.submit_latest(
            frame_id=0,
            task_texts=("pick",),
            vlm_inputs={},
            c_sem_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        with pytest.raises(RuntimeError, match="semantic refresh failed"):
            runtime.wait_for_snapshot(frame_id=0, task_texts=("pick",))

    assert "Semantic refresh failed in worker" in caplog.text
    assert "semantic encoder exploded" in caplog.text
    runtime.close()
