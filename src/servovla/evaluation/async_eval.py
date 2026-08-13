from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AsyncEvalTask:
    step: int
    snapshot: dict[str, Any]
    used_ema: bool
    eval_seed: int
    eval_zero_noise: bool


class AsyncEvalManager:
    def __init__(
        self,
        *,
        evaluator: Callable[[AsyncEvalTask], dict[str, Any]],
        result_writer: Callable[[int, dict[str, Any]], None],
        wandb_logger: Callable[..., None] | None,
        max_retries: int = 2,
    ) -> None:
        self._evaluator = evaluator
        self._result_writer = result_writer
        self._wandb_logger = wandb_logger
        self._max_retries = max(1, int(max_retries))
        self._queue: queue.Queue[AsyncEvalTask | None] = queue.Queue()
        self._fatal_error: Exception | None = None
        self._thread = threading.Thread(target=self._worker_loop, name="servovla-eval", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._thread.start()
        self._started = True

    def enqueue(self, task: AsyncEvalTask) -> None:
        if self._fatal_error is not None:
            raise RuntimeError("evaluation manager is in fatal state") from self._fatal_error
        self._queue.put(task)

    def finish(self) -> None:
        self._queue.put(None)

    def join(self) -> None:
        if self._started:
            self._thread.join()

    @property
    def fatal_error(self) -> Exception | None:
        return self._fatal_error

    def check_healthy(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError("async evaluation failed") from self._fatal_error

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return

            for attempt_idx in range(self._max_retries):
                try:
                    summary = self._evaluator(task)
                    self._result_writer(int(task.step), summary)
                    if self._wandb_logger is not None:
                        try:
                            self._wandb_logger(
                                build_eval_log_payload(summary, eval_step=int(task.step)),
                            )
                        except Exception as exc:
                            logger.warning(
                                "async eval wandb logging failed for step %s: %s", task.step, exc
                            )
                    break
                except Exception as exc:
                    if attempt_idx + 1 >= self._max_retries:
                        self._fatal_error = exc
                        logger.exception("async evaluation failed for step %s", task.step)
                        return


def _add_eval_summary_to_payload(
    payload: dict[str, Any], summary: dict[str, Any], *, prefix: str
) -> None:
    action_loss = summary.get("action_loss_mean")
    if action_loss is None and "all_steps" in summary:
        action_loss = summary["all_steps"].get("mae_mean")
    if action_loss is not None:
        payload[f"{prefix}/action_loss"] = float(action_loss)

    fm_loss = summary.get("fm_loss_mean", summary.get("loss_mean"))
    if fm_loss is not None:
        payload[f"{prefix}/fm_loss"] = float(fm_loss)
    fm_loss_by_dataset = summary.get("fm_loss_by_dataset", summary.get("loss_by_dataset", {}))
    for dataset_slug, avg_loss in fm_loss_by_dataset.items():
        payload[f"{prefix}/fm_loss/{dataset_slug}"] = float(avg_loss)


def build_eval_log_payload(
    summary: dict[str, Any], *, eval_step: int | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if eval_step is not None:
        payload["val/step"] = int(eval_step)
    if "raw" in summary or "ema" in summary:
        for variant in ("raw", "ema"):
            if variant in summary:
                _add_eval_summary_to_payload(payload, summary[variant], prefix=f"val/{variant}")
        return payload

    _add_eval_summary_to_payload(payload, summary, prefix="val/raw")
    return payload
