from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from servovla.architectures.vlm_forward import run_vlm_encoder_forward
from servovla.trainer.compile_guard import torch_compile_concurrency_guard
from servovla.trainer.vlm_compile_warmup import (
    handle_runtime_vlm_signatures,
    make_vlm_input_signatures,
)

logger = logging.getLogger(__name__)


def _iter_vlm_input_micro_batches(
    vlm_inputs: Mapping[str, Any],
    *,
    micro_batch_size: int,
) -> Iterable[dict[str, Any]]:
    input_ids = vlm_inputs.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim == 0:
        yield dict(vlm_inputs)
        return

    batch_size = int(input_ids.shape[0])
    chunk_size = int(micro_batch_size)
    if batch_size <= 0 or chunk_size <= 0 or chunk_size >= batch_size:
        yield dict(vlm_inputs)
        return

    image_grid_thw = vlm_inputs.get("image_grid_thw")
    image_rows_per_sample: int | None = None
    if torch.is_tensor(image_grid_thw):
        if image_grid_thw.ndim == 0 or int(image_grid_thw.shape[0]) % batch_size != 0:
            yield dict(vlm_inputs)
            return
        image_rows_per_sample = int(image_grid_thw.shape[0]) // batch_size

    pixel_values = vlm_inputs.get("pixel_values")
    pixel_rows_per_sample: int | None = None
    if torch.is_tensor(pixel_values):
        if pixel_values.ndim == 0 or int(pixel_values.shape[0]) % batch_size != 0:
            yield dict(vlm_inputs)
            return
        pixel_rows_per_sample = int(pixel_values.shape[0]) // batch_size

    for sample_start in range(0, batch_size, chunk_size):
        sample_end = min(sample_start + chunk_size, batch_size)
        chunk_inputs: dict[str, Any] = {}
        for key, value in vlm_inputs.items():
            if not torch.is_tensor(value) or value.ndim == 0:
                chunk_inputs[key] = value
            elif key == "image_grid_thw" and image_rows_per_sample is not None:
                row_start = sample_start * image_rows_per_sample
                row_end = sample_end * image_rows_per_sample
                chunk_inputs[key] = value[row_start:row_end]
            elif key == "pixel_values" and pixel_rows_per_sample is not None:
                row_start = sample_start * pixel_rows_per_sample
                row_end = sample_end * pixel_rows_per_sample
                chunk_inputs[key] = value[row_start:row_end]
            elif int(value.shape[0]) == batch_size:
                chunk_inputs[key] = value[sample_start:sample_end]
            else:
                chunk_inputs[key] = value
        yield chunk_inputs


def run_vlm_encoder_forward_micro_batched(
    vlm_encoder,
    vlm_inputs: Mapping[str, Any],
    *,
    micro_batch_size: int,
) -> torch.Tensor:
    chunks = list(_iter_vlm_input_micro_batches(vlm_inputs, micro_batch_size=int(micro_batch_size)))
    if len(chunks) <= 1:
        return run_vlm_encoder_forward(vlm_encoder, chunks[0])
    outputs = [run_vlm_encoder_forward(vlm_encoder, chunk) for chunk in chunks]
    return torch.cat(outputs, dim=0)


def run_vision_encoder_forward_micro_batched(
    vision_encoder,
    pixel_values: torch.Tensor,
    *,
    micro_batch_size: int,
) -> torch.Tensor:
    if not torch.is_tensor(pixel_values):
        return vision_encoder(pixel_values)
    batch_size = int(pixel_values.shape[0]) if pixel_values.ndim > 0 else 0
    chunk_size = int(micro_batch_size)
    if batch_size <= 0 or chunk_size <= 0 or chunk_size >= batch_size:
        return vision_encoder(pixel_values)
    outputs = [
        vision_encoder(pixel_values[sample_start : min(sample_start + chunk_size, batch_size)])
        for sample_start in range(0, batch_size, chunk_size)
    ]
    return torch.cat(outputs, dim=0)


@dataclass(slots=True)
class EncodedFeatureBatch:
    f_vision: torch.Tensor
    c_sem: torch.Tensor
    c_sem_mask: torch.Tensor
    action: torch.Tensor
    loss_mask: torch.Tensor
    q_current: torch.Tensor
    frame_delay: torch.Tensor
    dataset_slug: str | list[str]
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    ready_event: torch.cuda.Event | None = None
    start_event: torch.cuda.Event | None = None
    after_h2d_event: torch.cuda.Event | None = None
    after_preprocess_event: torch.cuda.Event | None = None
    after_vision_preprocess_event: torch.cuda.Event | None = None
    after_vlm_preprocess_event: torch.cuda.Event | None = None
    after_encode_event: torch.cuda.Event | None = None
    after_vision_encode_event: torch.cuda.Event | None = None
    after_vlm_encode_event: torch.cuda.Event | None = None
    ready_waited: bool = False

    def _tensors(self) -> list[torch.Tensor]:
        return [
            self.f_vision,
            self.c_sem,
            self.action,
            self.loss_mask,
            self.c_sem_mask,
            self.q_current,
            self.frame_delay,
        ]

    def wait_ready(self, device: torch.device) -> None:
        if self.ready_waited:
            return
        if self.ready_event is not None:
            current_stream = torch.cuda.current_stream(device)
            current_stream.wait_event(self.ready_event)
            for tensor in self._tensors():
                if tensor.is_cuda:
                    tensor.record_stream(current_stream)
        self.ready_waited = True

    def ready_event_completed(self) -> bool | None:
        if self.ready_event is None:
            return None
        return bool(self.ready_event.query())

    def collect_timings(self, *, block: bool = False) -> dict[str, float]:
        if (
            self.start_event is None
            or self.after_h2d_event is None
            or self.after_encode_event is None
        ):
            return self.timings
        if (
            "h2d_ms" in self.timings
            and "preprocess_ms" in self.timings
            and "encoder_ms" in self.timings
        ):
            return self.timings

        after_preprocess = self.after_preprocess_event or self.after_h2d_event
        if block:
            self.after_encode_event.synchronize()
        elif not self.after_encode_event.query():
            return {}

        self.timings["h2d_ms"] = float(self.start_event.elapsed_time(self.after_h2d_event))
        self.timings["preprocess_ms"] = float(self.after_h2d_event.elapsed_time(after_preprocess))
        self.timings["encoder_ms"] = float(after_preprocess.elapsed_time(self.after_encode_event))
        if self.after_vision_preprocess_event is not None:
            self.timings["vision_preprocess_ms"] = float(
                self.after_h2d_event.elapsed_time(self.after_vision_preprocess_event)
            )
        if self.after_vlm_preprocess_event is not None:
            self.timings["vlm_preprocess_ms"] = float(
                self.after_h2d_event.elapsed_time(self.after_vlm_preprocess_event)
            )
        if self.after_vision_encode_event is not None and self.after_vlm_encode_event is not None:
            vision_start = self.after_vision_preprocess_event or after_preprocess
            vlm_start = self.after_vlm_preprocess_event or after_preprocess
            self.timings["vision_encoder_ms"] = float(
                vision_start.elapsed_time(self.after_vision_encode_event)
            )
            self.timings["vlm_encoder_ms"] = float(
                vlm_start.elapsed_time(self.after_vlm_encode_event)
            )
        return self.timings


class SingleGpuFeatureProducer:
    def __init__(
        self,
        *,
        model,
        device: torch.device,
        amp_dtype: torch.dtype,
        queue_depth: int,
        encoder_parallel_streams: bool,
        image_preprocessor: Any | None = None,
        vision_encoder_micro_batch_size: int | None = None,
        vlm_encoder_micro_batch_size: int | None = None,
        startup_trace_submit_limit: int | None = None,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("SingleGpuFeatureProducer requires a CUDA device.")
        if int(queue_depth) < 1:
            raise ValueError("feature_queue_depth must be at least 1.")
        if vlm_encoder_micro_batch_size is not None and int(vlm_encoder_micro_batch_size) < 0:
            raise ValueError("vlm_encoder_micro_batch_size must be non-negative.")
        if vision_encoder_micro_batch_size is not None and int(vision_encoder_micro_batch_size) < 0:
            raise ValueError("vision_encoder_micro_batch_size must be non-negative.")
        if startup_trace_submit_limit is not None and int(startup_trace_submit_limit) < 0:
            raise ValueError("startup_trace_submit_limit must be non-negative.")
        self.model = model
        self.device = device
        self.amp_dtype = amp_dtype
        self.queue_depth = int(queue_depth)
        self.encoder_parallel_streams = bool(encoder_parallel_streams)
        self.image_preprocessor = image_preprocessor
        self.vision_encoder_micro_batch_size = int(vision_encoder_micro_batch_size or 0)
        self.vlm_encoder_micro_batch_size = int(vlm_encoder_micro_batch_size or 0)
        self._startup_trace_submit_limit = int(startup_trace_submit_limit or 0)
        self._submit_count = 0
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.vision_preprocess_stream = (
            torch.cuda.Stream(device=device) if self.encoder_parallel_streams else self.h2d_stream
        )
        self.vlm_preprocess_stream = (
            torch.cuda.Stream(device=device) if self.encoder_parallel_streams else self.h2d_stream
        )
        self.preprocess_join_stream = (
            torch.cuda.Stream(device=device) if self.encoder_parallel_streams else self.h2d_stream
        )
        self.encoder_stream = (
            torch.cuda.Stream(device=device) if self.encoder_parallel_streams else self.h2d_stream
        )
        self.vlm_encoder_stream = (
            torch.cuda.Stream(device=device)
            if self.encoder_parallel_streams
            else self.encoder_stream
        )
        self.completion_stream = (
            torch.cuda.Stream(device=device)
            if self.encoder_parallel_streams
            else self.encoder_stream
        )
        self._queue: deque[EncodedFeatureBatch] = deque()

    def _move_tensor(
        self, value: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        if dtype is None:
            return value.to(device=self.device, non_blocking=True)
        return value.to(device=self.device, dtype=dtype, non_blocking=True)

    def _move_vlm_inputs(self, vlm_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: self._move_tensor(value) if torch.is_tensor(value) else value
            for key, value in vlm_inputs.items()
        }

    def _record_stream_recursive(self, value: Any, stream: torch.cuda.Stream) -> None:
        if isinstance(value, dict):
            for item in value.values():
                self._record_stream_recursive(item, stream)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                self._record_stream_recursive(item, stream)
            return
        record_stream = getattr(value, "record_stream", None)
        if record_stream is not None and bool(getattr(value, "is_cuda", False)):
            record_stream(stream)

    def __len__(self) -> int:
        return len(self._queue)

    def can_submit(self) -> bool:
        return len(self._queue) < self.queue_depth

    def _can_split_frozen_encoders(self) -> bool:
        return (
            self.encoder_parallel_streams
            and getattr(self.model, "vision_encoder", None) is not None
            and getattr(self.model, "vlm_encoder", None) is not None
        )

    def _can_split_raw_preprocess(self) -> bool:
        if not self.encoder_parallel_streams or self.image_preprocessor is None:
            return False
        return callable(getattr(self.image_preprocessor, "preprocess_vision", None)) and callable(
            getattr(self.image_preprocessor, "preprocess_vlm_images", None)
        )

    def submit(self, batch: dict[str, Any], *, metadata: dict[str, Any] | None = None) -> None:
        if not self.can_submit():
            raise RuntimeError("feature queue is full.")

        self._submit_count += 1
        submit_index = self._submit_count
        trace_startup_submit = submit_index <= self._startup_trace_submit_limit
        submit_trace_t0 = time.perf_counter()

        def _trace_startup_submit(stage: str) -> None:
            if trace_startup_submit:
                logger.info(
                    "gpu_pipeline startup submit %d | %s | %.3fs",
                    submit_index,
                    stage,
                    time.perf_counter() - submit_trace_t0,
                )

        split_frozen_encoders = self._can_split_frozen_encoders()
        split_raw_preprocess = self._can_split_raw_preprocess()
        start = torch.cuda.Event(enable_timing=True)
        after_h2d = torch.cuda.Event(enable_timing=True)
        after_preprocess = torch.cuda.Event(enable_timing=True)
        after_vision_preprocess = torch.cuda.Event(enable_timing=True)
        after_vlm_preprocess = torch.cuda.Event(enable_timing=True)
        after_encode = torch.cuda.Event(enable_timing=True)
        after_vision_encode = torch.cuda.Event(enable_timing=True)
        after_vlm_encode = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self.h2d_stream):
            _trace_startup_submit("h2d_start")
            start.record()
            is_raw_batch = "vision_images_uint8" in batch or "vlm_images_uint8" in batch
            if is_raw_batch:
                if self.image_preprocessor is None:
                    raise RuntimeError("Raw image batch requires image_preprocessor")
                move_raw = getattr(self.image_preprocessor, "move_raw_batch_to_device", None)
                if callable(move_raw):
                    batch = move_raw(batch)
                action = self._move_tensor(batch["action"])
                loss_mask = self._move_tensor(batch["loss_mask"])
                c_sem_mask = self._move_tensor(batch["c_sem_mask"]).bool()
                q_current = self._move_tensor(batch["q_current"])
                frame_delay = self._move_tensor(batch["frame_delay"])
                vlm_inputs = self._move_vlm_inputs(batch["vlm_inputs"])
                after_h2d.record()
                _trace_startup_submit("h2d_queued")

                if not split_raw_preprocess:
                    _trace_startup_submit("combined_preprocess_start")
                    processed_batch = self.image_preprocessor.preprocess_batch(batch)
                    pixel_values = self._move_tensor(processed_batch["pixel_values"])
                    vlm_pixel_values = self._move_tensor(
                        processed_batch["vlm_inputs"]["pixel_values"]
                    )
                    vlm_inputs = dict(vlm_inputs)
                    vlm_inputs["pixel_values"] = vlm_pixel_values
                    after_preprocess.record()
                    _trace_startup_submit("combined_preprocess_queued")
            else:
                pixel_values = self._move_tensor(batch["pixel_values"])
                vlm_inputs = self._move_vlm_inputs(batch["vlm_inputs"])
                action = self._move_tensor(batch["action"])
                loss_mask = self._move_tensor(batch["loss_mask"])
                c_sem_mask = self._move_tensor(batch["c_sem_mask"]).bool()
                q_current = self._move_tensor(batch["q_current"])
                frame_delay = self._move_tensor(batch["frame_delay"])
                after_h2d.record()
                after_preprocess.record()
                _trace_startup_submit("h2d_queued")

        if is_raw_batch and split_raw_preprocess:
            self.vision_preprocess_stream.wait_event(after_h2d)
            self.vlm_preprocess_stream.wait_event(after_h2d)
            with torch.cuda.stream(self.vision_preprocess_stream):
                _trace_startup_submit("vision_preprocess_start")
                pixel_values = self.image_preprocessor.preprocess_vision(
                    batch["vision_images_uint8"]
                )
                pixel_values = self._move_tensor(pixel_values)
                after_vision_preprocess.record()
                self._record_stream_recursive(
                    batch["vision_images_uint8"], self.vision_preprocess_stream
                )
                _trace_startup_submit("vision_preprocess_queued")

            with torch.cuda.stream(self.vlm_preprocess_stream):
                _trace_startup_submit("vlm_preprocess_start")
                vlm_pixel_values = self.image_preprocessor.preprocess_vlm_images(
                    batch["vlm_images_uint8"]
                )
                vlm_pixel_values = self._move_tensor(vlm_pixel_values)
                vlm_inputs = dict(vlm_inputs)
                vlm_inputs["pixel_values"] = vlm_pixel_values
                after_vlm_preprocess.record()
                self._record_stream_recursive(batch["vlm_images_uint8"], self.vlm_preprocess_stream)
                _trace_startup_submit("vlm_preprocess_queued")

            self.preprocess_join_stream.wait_event(after_vision_preprocess)
            self.preprocess_join_stream.wait_event(after_vlm_preprocess)
            with torch.cuda.stream(self.preprocess_join_stream):
                after_preprocess.record()
        else:
            after_vision_preprocess = None
            after_vlm_preprocess = None

        autocast_enabled = self.amp_dtype in {torch.bfloat16, torch.float16}
        with torch_compile_concurrency_guard():
            if split_frozen_encoders:
                self.encoder_stream.wait_event(after_vision_preprocess or after_preprocess)
                self.vlm_encoder_stream.wait_event(after_vlm_preprocess or after_preprocess)
                with torch.cuda.stream(self.encoder_stream):
                    _trace_startup_submit("vision_encoder_start")
                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=autocast_enabled,
                        ):
                            f_vision = run_vision_encoder_forward_micro_batched(
                                self.model.vision_encoder,
                                pixel_values,
                                micro_batch_size=self.vision_encoder_micro_batch_size,
                            )
                    after_vision_encode.record()
                    self._record_stream_recursive(pixel_values, self.encoder_stream)
                    _trace_startup_submit("vision_encoder_queued")

                with torch.cuda.stream(self.vlm_encoder_stream):
                    _trace_startup_submit("vlm_encoder_start")
                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=autocast_enabled,
                        ):
                            c_sem = run_vlm_encoder_forward_micro_batched(
                                self.model.vlm_encoder,
                                vlm_inputs,
                                micro_batch_size=self.vlm_encoder_micro_batch_size,
                            )
                    after_vlm_encode.record()
                    self._record_stream_recursive(vlm_inputs, self.vlm_encoder_stream)
                    _trace_startup_submit("vlm_encoder_queued")

                self.completion_stream.wait_event(after_preprocess)
                self.completion_stream.wait_event(after_vision_encode)
                self.completion_stream.wait_event(after_vlm_encode)
                with torch.cuda.stream(self.completion_stream):
                    after_encode.record()
                _trace_startup_submit("submit_queued")
            else:
                self.encoder_stream.wait_event(after_preprocess)
                with torch.cuda.stream(self.encoder_stream):
                    _trace_startup_submit("combined_encoder_start")
                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=self.amp_dtype,
                            enabled=autocast_enabled,
                        ):
                            f_vision, c_sem = self.model.encode_observations(
                                pixel_values=pixel_values,
                                vlm_inputs=vlm_inputs,
                            )
                    after_encode.record()
                    self._record_stream_recursive(pixel_values, self.encoder_stream)
                    self._record_stream_recursive(vlm_inputs, self.encoder_stream)
                    _trace_startup_submit("combined_encoder_queued")

        self._queue.append(
            EncodedFeatureBatch(
                f_vision=f_vision,
                c_sem=c_sem,
                c_sem_mask=c_sem_mask,
                action=action,
                loss_mask=loss_mask,
                q_current=q_current,
                frame_delay=frame_delay,
                dataset_slug=batch.get("dataset_slug", "unknown"),
                timings={},
                metadata=dict(metadata or {}),
                ready_event=after_encode,
                start_event=start,
                after_h2d_event=after_h2d,
                after_preprocess_event=after_preprocess,
                after_encode_event=after_encode,
                after_vision_preprocess_event=after_vision_preprocess,
                after_vlm_preprocess_event=after_vlm_preprocess,
                after_vision_encode_event=after_vision_encode if split_frozen_encoders else None,
                after_vlm_encode_event=after_vlm_encode if split_frozen_encoders else None,
            )
        )

    def pop_oldest(self) -> EncodedFeatureBatch:
        if not self._queue:
            raise RuntimeError("feature queue is empty.")
        return self._queue.popleft()

    def encode_next(self, batch: dict[str, Any]) -> EncodedFeatureBatch:
        self.submit(batch)
        return self.pop_oldest()


class AsyncGpuFeatureProducer:
    def __init__(
        self,
        *,
        dataloader: Iterable[dict[str, Any]],
        feature_producer: Any,
        feature_queue_depth: int,
        cpu_prefetch_depth: int,
        max_steps: int,
        batch_metadata_fn: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        warmup_dataloader: Iterable[dict[str, Any]] | None = None,
        vlm_warmup_max_batches: int = 0,
        vlm_signature_registry: Any | None = None,
        on_new_vlm_signature: str = "warn_and_warmup",
        log_vlm_signatures: bool = True,
        overlap_train_reader_during_warmup: bool = False,
    ) -> None:
        self.dataloader = dataloader
        self.warmup_dataloader = warmup_dataloader
        self.feature_producer = feature_producer
        self.max_steps = max(int(max_steps), 0)
        self.batch_metadata_fn = batch_metadata_fn
        self._vlm_warmup_max_batches = max(int(vlm_warmup_max_batches), 0)
        self._warmup_discard_batches = self._vlm_warmup_max_batches if self.max_steps > 0 else 0
        self._uses_separate_warmup_dataloader = (
            self.warmup_dataloader is not None and self._warmup_discard_batches > 0
        )
        self._reader_max_batches = self.max_steps + (
            0 if self._uses_separate_warmup_dataloader else self._warmup_discard_batches
        )
        self.vlm_signature_registry = vlm_signature_registry
        self.on_new_vlm_signature = str(on_new_vlm_signature)
        self.log_vlm_signatures = bool(log_vlm_signatures)
        self._vlm_signature_micro_batch_size = int(
            getattr(feature_producer, "vlm_encoder_micro_batch_size", 0) or 0
        )
        self.overlap_train_reader_during_warmup = bool(overlap_train_reader_during_warmup)
        self._cpu_queue: queue.Queue[tuple[dict[str, Any], dict[str, Any]] | None] = queue.Queue(
            maxsize=max(int(cpu_prefetch_depth), 1)
        )
        self._feature_queue_depth = max(int(feature_queue_depth), 1)
        self._feature_queue: queue.Queue[EncodedFeatureBatch] = queue.Queue(
            maxsize=self._feature_queue_depth
        )
        self._warmup_complete = threading.Event()
        if self._warmup_discard_batches == 0:
            self._warmup_complete.set()
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._reader = threading.Thread(
            target=self._reader_main, name="servovla-cpu-batch-reader", daemon=True
        )
        self._encoder = threading.Thread(
            target=self._encoder_main, name="servovla-gpu-feature-producer", daemon=True
        )

    def start(self) -> None:
        self._reader.start()
        self._encoder.start()

    def _set_error(self, error: BaseException) -> None:
        self._error = error
        self._stop.set()
        self._warmup_complete.set()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _mark_warmup_complete(self) -> None:
        if self._warmup_discard_batches > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._warmup_complete.set()

    def wait_warmup_complete(self, *, timeout: float | None = None) -> None:
        deadline = time.perf_counter() + float(timeout) if timeout is not None else None
        while not self._warmup_complete.wait(timeout=0.05):
            self._raise_if_error()
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError("Timed out waiting for VLM compile warmup")
        self._raise_if_error()

    def _read_batches_into_cpu_queue(
        self,
        *,
        source: Iterable[dict[str, Any]],
        count: int,
        source_name: str,
    ) -> int:
        produced = 0
        data_iter = iter(source)
        while not self._stop.is_set() and produced < int(count):
            data_t0 = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(source)
                data_t0 = time.perf_counter()
                try:
                    batch = next(data_iter)
                except StopIteration:
                    raise RuntimeError(
                        f"AsyncGpuFeatureProducer {source_name} dataloader produced no batches"
                    )
            data_time = time.perf_counter() - data_t0
            metadata: dict[str, Any] = {
                "data_time": data_time,
                "producer_timing.reader_data_time": data_time,
                "producer_state.reader_source": source_name,
            }
            if self.batch_metadata_fn is not None:
                metadata.update(dict(self.batch_metadata_fn(batch)))
            put_t0 = time.perf_counter()
            while not self._stop.is_set():
                metadata["producer_timing.cpu_queue_put_wait"] = time.perf_counter() - put_t0
                try:
                    self._cpu_queue.put_nowait((batch, metadata))
                    break
                except queue.Full:
                    time.sleep(0.001)
            else:
                return produced
            produced += 1
        return produced

    def _reader_main(self) -> None:
        try:
            if self._uses_separate_warmup_dataloader:
                assert self.warmup_dataloader is not None
                self._read_batches_into_cpu_queue(
                    source=self.warmup_dataloader,
                    count=self._warmup_discard_batches,
                    source_name="warmup",
                )
                if not self.overlap_train_reader_during_warmup:
                    while not self._stop.is_set() and not self._warmup_complete.wait(timeout=0.05):
                        pass
                if self._stop.is_set():
                    return
                self._read_batches_into_cpu_queue(
                    source=self.dataloader,
                    count=self.max_steps,
                    source_name="train",
                )
            else:
                self._read_batches_into_cpu_queue(
                    source=self.dataloader,
                    count=self._reader_max_batches,
                    source_name="train",
                )
            self._cpu_queue.put(None)
        except BaseException as exc:
            self._set_error(exc)
            try:
                self._cpu_queue.put_nowait(None)
            except queue.Full:
                pass

    def _encoder_main(self) -> None:
        encoded_count = 0
        pending_contexts: deque[tuple[bool, tuple[Any, ...], dict[str, Any]]] = deque()
        input_exhausted = False
        try:
            while not self._stop.is_set():
                while (
                    not self._stop.is_set()
                    and not input_exhausted
                    and len(pending_contexts) + self._feature_queue.qsize()
                    < self._feature_queue_depth
                    and (
                        not callable(getattr(self.feature_producer, "can_submit", None))
                        or self.feature_producer.can_submit()
                    )
                ):
                    queue_wait_t0 = time.perf_counter()
                    if pending_contexts:
                        try:
                            item = self._cpu_queue.get_nowait()
                        except queue.Empty:
                            break
                    else:
                        item = self._cpu_queue.get()
                    cpu_queue_wait = time.perf_counter() - queue_wait_t0
                    if item is None:
                        input_exhausted = True
                        break
                    batch, metadata = item
                    metadata["producer_timing.encoder_cpu_queue_wait"] = cpu_queue_wait
                    is_warmup_batch = (
                        encoded_count + len(pending_contexts) < self._warmup_discard_batches
                    )
                    signatures: tuple[Any, ...] = ()
                    if self.vlm_signature_registry is not None:
                        if is_warmup_batch:
                            signatures = make_vlm_input_signatures(
                                batch,
                                micro_batch_size=self._vlm_signature_micro_batch_size,
                            )
                            for signature in signatures:
                                is_new_signature = self.vlm_signature_registry.mark_seen(signature)
                                if is_new_signature and self.log_vlm_signatures:
                                    logger.info(
                                        "VLM compile warmup signature %d | %s",
                                        self.vlm_signature_registry.seen_count,
                                        signature,
                                    )
                        else:
                            handle_runtime_vlm_signatures(
                                batch,
                                registry=self.vlm_signature_registry,
                                on_new_signature=self.on_new_vlm_signature,
                                logger_name=__name__,
                                micro_batch_size=self._vlm_signature_micro_batch_size,
                            )
                    submit_t0 = time.perf_counter()
                    self.feature_producer.submit(batch, metadata=metadata)
                    metadata["producer_timing.submit_time"] = time.perf_counter() - submit_t0
                    pending_contexts.append((is_warmup_batch, signatures, metadata))

                if not pending_contexts:
                    if input_exhausted:
                        self._mark_warmup_complete()
                        return
                    time.sleep(0.001)
                    continue

                encoded = self.feature_producer.pop_oldest()
                is_warmup_batch, signatures, metadata = pending_contexts.popleft()
                encoded.metadata.update(metadata)
                encoded_count += 1
                if is_warmup_batch:
                    if signatures:
                        for signature in signatures:
                            self.vlm_signature_registry.mark_warmed(signature)
                    del encoded
                    if encoded_count >= self._warmup_discard_batches:
                        self._mark_warmup_complete()
                else:
                    feature_queue_put_t0 = time.perf_counter()
                    while not self._stop.is_set():
                        encoded.metadata["producer_timing.feature_queue_put_wait"] = (
                            time.perf_counter() - feature_queue_put_t0
                        )
                        try:
                            self._feature_queue.put_nowait(encoded)
                            break
                        except queue.Full:
                            time.sleep(0.001)
        except BaseException as exc:
            self._set_error(exc)

    def pop_oldest(self) -> EncodedFeatureBatch:
        while True:
            self._raise_if_error()
            try:
                return self._feature_queue.get(timeout=0.05)
            except queue.Empty:
                if not self._reader.is_alive() and not self._encoder.is_alive():
                    self._raise_if_error()
                    raise StopIteration("AsyncGpuFeatureProducer exhausted")

    def __len__(self) -> int:
        return int(self._feature_queue.qsize())

    def stop(self) -> None:
        self._stop.set()
        try:
            self._cpu_queue.put_nowait(None)
        except queue.Full:
            pass
        for thread in (self._reader, self._encoder):
            if thread.is_alive():
                thread.join(timeout=5.0)
