from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class SemanticRefreshRequest:
    frame_id: int
    task_texts: tuple[str, ...]
    session_id: str
    vlm_inputs: dict[str, Any]
    c_sem_mask: torch.Tensor


@dataclass(slots=True)
class SemanticSnapshot:
    frame_id: int
    task_texts: tuple[str, ...]
    session_id: str
    c_sem: torch.Tensor
    c_sem_mask: torch.Tensor
    ready_event: Any | None = None


class SemanticDelayRuntime:
    def __init__(
        self,
        *,
        encode_semantic: Callable[[dict[str, Any]], torch.Tensor],
        max_frame_delay: int,
        semantic_wait_warn_ms: int,
        semantic_wait_fail_ms: int,
        logger: logging.Logger | None = None,
        time_fn: Callable[[], float] | None = None,
    ) -> None:
        self.encode_semantic = encode_semantic
        self.logger = logger or logging.getLogger(__name__)
        self.time_fn = time_fn or time.monotonic

        self._condition = threading.Condition()
        self._latest_snapshot: SemanticSnapshot | None = None
        self._pending_request: SemanticRefreshRequest | None = None
        self._active_request: SemanticRefreshRequest | None = None
        self._worker: threading.Thread | None = None
        self._closed = False
        self._last_error: Exception | None = None
        self._last_admission_debug: dict[str, Any] = {}

        self.configure(
            max_frame_delay=max_frame_delay,
            semantic_wait_warn_ms=semantic_wait_warn_ms,
            semantic_wait_fail_ms=semantic_wait_fail_ms,
        )

    def configure(
        self, *, max_frame_delay: int, semantic_wait_warn_ms: int, semantic_wait_fail_ms: int
    ) -> None:
        max_frame_delay = int(max_frame_delay)
        semantic_wait_warn_ms = int(semantic_wait_warn_ms)
        semantic_wait_fail_ms = int(semantic_wait_fail_ms)
        if max_frame_delay < 0:
            raise ValueError(f"max_frame_delay must be >= 0, got {max_frame_delay}")
        if semantic_wait_warn_ms < 0:
            raise ValueError(f"semantic_wait_warn_ms must be >= 0, got {semantic_wait_warn_ms}")
        if semantic_wait_fail_ms <= 0:
            raise ValueError(f"semantic_wait_fail_ms must be > 0, got {semantic_wait_fail_ms}")
        if semantic_wait_fail_ms < semantic_wait_warn_ms:
            raise ValueError(
                f"semantic_wait_fail_ms ({semantic_wait_fail_ms}) must be >= semantic_wait_warn_ms ({semantic_wait_warn_ms})"
            )
        self.max_frame_delay = max_frame_delay
        self.semantic_wait_warn_ms = semantic_wait_warn_ms
        self.semantic_wait_fail_ms = semantic_wait_fail_ms

    def reset(self) -> None:
        with self._condition:
            self._latest_snapshot = None
            self._pending_request = None
            self._active_request = None
            self._last_error = None
            self._last_admission_debug = {}
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)

    def peek_snapshot(self) -> SemanticSnapshot | None:
        with self._condition:
            return self._latest_snapshot

    def get_last_admission_debug(self) -> dict[str, Any]:
        with self._condition:
            return dict(self._last_admission_debug)

    @staticmethod
    def _normalize_admission_policy(admission_policy: str) -> str:
        policy = str(admission_policy).strip().lower()
        if policy not in {"bsr", "age_only", "latest_cache"}:
            raise ValueError(f"Unsupported semantic admission policy: {admission_policy!r}")
        return policy

    def _check_snapshot_admission(
        self,
        snapshot: SemanticSnapshot | None,
        *,
        frame_id: int,
        task_texts: tuple[str, ...],
        session_id: str,
        admission_policy: str,
        delay_is_supported: Callable[[int], bool] | None = None,
    ) -> tuple[bool, str, int | None]:
        if snapshot is None:
            return False, "no_snapshot", None

        policy = self._normalize_admission_policy(admission_policy)
        frame_delay = int(frame_id) - int(snapshot.frame_id)
        if policy == "latest_cache":
            return True, "admitted_latest_cache", frame_delay

        if frame_delay < 0:
            return False, "future_snapshot", frame_delay
        if frame_delay > self.max_frame_delay:
            return False, "over_age", frame_delay
        if delay_is_supported is not None and not delay_is_supported(frame_delay):
            return False, "unsupported_age_bucket", frame_delay
        if policy == "bsr":
            if snapshot.task_texts != tuple(task_texts):
                return False, "task_mismatch", frame_delay
            if str(snapshot.session_id) != str(session_id):
                return False, "session_mismatch", frame_delay
        return True, f"admitted_{policy}", frame_delay

    def has_request_for_frame_delay(
        self,
        *,
        frame_id: int,
        task_texts: tuple[str, ...],
        session_id: str = "default",
        delay_is_supported: Callable[[int], bool] | None = None,
    ) -> bool:
        frame_id = int(frame_id)
        task_texts = tuple(task_texts)
        session_id = str(session_id)
        with self._condition:
            requests = (self._pending_request, self._active_request)
            for request in requests:
                if (
                    request is None
                    or request.task_texts != task_texts
                    or str(request.session_id) != session_id
                ):
                    continue
                frame_delay = frame_id - int(request.frame_id)
                if 0 <= frame_delay <= self.max_frame_delay and (
                    delay_is_supported is None or delay_is_supported(frame_delay)
                ):
                    return True
        return False

    def submit_latest(
        self,
        *,
        frame_id: int,
        task_texts: tuple[str, ...],
        session_id: str = "default",
        vlm_inputs: dict[str, Any],
        c_sem_mask: torch.Tensor,
    ) -> None:
        self._ensure_worker_started()
        request = SemanticRefreshRequest(
            frame_id=int(frame_id),
            task_texts=tuple(task_texts),
            session_id=str(session_id),
            vlm_inputs={
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in vlm_inputs.items()
            },
            c_sem_mask=c_sem_mask.detach().clone(),
        )
        with self._condition:
            self._pending_request = request
            self._last_error = None
            self._condition.notify_all()

    def wait_for_snapshot(
        self,
        *,
        frame_id: int,
        task_texts: tuple[str, ...],
        session_id: str = "default",
        admission_policy: str = "bsr",
        delay_is_supported: Callable[[int], bool] | None = None,
    ) -> tuple[SemanticSnapshot, int, int]:
        started_at = self.time_fn()
        warned = False
        task_texts = tuple(task_texts)
        session_id = str(session_id)
        admission_policy = self._normalize_admission_policy(admission_policy)
        last_reject_reason = "no_snapshot"
        last_frame_delay = None
        with self._condition:
            while True:
                if self._last_error is not None:
                    raise RuntimeError("semantic refresh failed") from self._last_error

                snapshot = self._latest_snapshot
                admitted, reason, candidate_frame_delay = self._check_snapshot_admission(
                    snapshot,
                    frame_id=frame_id,
                    task_texts=task_texts,
                    session_id=session_id,
                    admission_policy=admission_policy,
                    delay_is_supported=delay_is_supported,
                )
                if admitted:
                    frame_delay = int(
                        candidate_frame_delay if candidate_frame_delay is not None else 0
                    )
                    waited_ms = int((self.time_fn() - started_at) * 1000)
                    self._last_admission_debug = {
                        "admission_policy": admission_policy,
                        "admitted": True,
                        "admit_reason": reason,
                        "reject_reason": None,
                        "request_frame_id": int(frame_id),
                        "snapshot_frame_id": int(snapshot.frame_id)
                        if snapshot is not None
                        else None,
                        "frame_delay": frame_delay,
                        "request_task_texts": list(task_texts),
                        "snapshot_task_texts": list(snapshot.task_texts)
                        if snapshot is not None
                        else None,
                        "request_session_id": session_id,
                        "snapshot_session_id": str(snapshot.session_id)
                        if snapshot is not None
                        else None,
                    }
                    break
                last_reject_reason = reason
                last_frame_delay = candidate_frame_delay

                elapsed_ms = int((self.time_fn() - started_at) * 1000)
                if not warned and elapsed_ms >= self.semantic_wait_warn_ms:
                    self.logger.warning(
                        "Semantic refresh wait exceeded warn threshold | frame_id=%s | task=%s | elapsed_ms=%s | max_frame_delay=%s",
                        frame_id,
                        task_texts,
                        elapsed_ms,
                        self.max_frame_delay,
                    )
                    warned = True
                if elapsed_ms >= self.semantic_wait_fail_ms:
                    self._last_admission_debug = {
                        "admission_policy": admission_policy,
                        "admitted": False,
                        "admit_reason": None,
                        "reject_reason": last_reject_reason,
                        "request_frame_id": int(frame_id),
                        "snapshot_frame_id": int(snapshot.frame_id)
                        if snapshot is not None
                        else None,
                        "frame_delay": last_frame_delay,
                        "request_task_texts": list(task_texts),
                        "snapshot_task_texts": list(snapshot.task_texts)
                        if snapshot is not None
                        else None,
                        "request_session_id": session_id,
                        "snapshot_session_id": str(snapshot.session_id)
                        if snapshot is not None
                        else None,
                    }
                    raise TimeoutError(
                        f"timed out waiting for semantic snapshot after {elapsed_ms}ms "
                        f"(max_frame_delay={self.max_frame_delay}, task={task_texts}, "
                        f"admission_policy={admission_policy}, reject_reason={last_reject_reason})"
                    )

                remaining = max(
                    0.0, (self.semantic_wait_fail_ms / 1000.0) - (self.time_fn() - started_at)
                )
                self._condition.wait(timeout=min(0.01, remaining))

        self._synchronize_ready_event(snapshot)
        return snapshot, frame_delay, waited_ms

    def _record_ready_event(self, tensor: torch.Tensor) -> Any | None:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(tensor.device))
            return event
        return None

    def _synchronize_ready_event(self, snapshot: SemanticSnapshot) -> None:
        ready_event = snapshot.ready_event
        if ready_event is not None:
            ready_event.synchronize()

    def _ensure_worker_started(self) -> None:
        with self._condition:
            if self._worker is not None and self._worker.is_alive():
                return
            self._closed = False
            self._worker = threading.Thread(
                target=self._worker_loop, name="servovla-semantic-worker", daemon=True
            )
            self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                while not self._closed and self._pending_request is None:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending_request
                self._pending_request = None
                self._active_request = request

            assert request is not None
            try:
                c_sem = self.encode_semantic(request.vlm_inputs)
                ready_event = self._record_ready_event(c_sem)
                snapshot = SemanticSnapshot(
                    frame_id=request.frame_id,
                    task_texts=request.task_texts,
                    session_id=request.session_id,
                    c_sem=c_sem.detach(),
                    c_sem_mask=request.c_sem_mask.detach().clone(),
                    ready_event=ready_event,
                )
                with self._condition:
                    self._latest_snapshot = snapshot
                    self._active_request = None
                    self._last_error = None
                    self._condition.notify_all()
            except Exception as exc:  # pragma: no cover - exercised through wait_for_snapshot
                self.logger.exception(
                    "Semantic refresh failed in worker | frame_id=%s | task=%s",
                    request.frame_id,
                    request.task_texts,
                )
                with self._condition:
                    self._active_request = None
                    self._last_error = exc
                    self._condition.notify_all()
