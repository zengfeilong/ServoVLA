from __future__ import annotations

import importlib.util
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

log = logging.getLogger(__name__)

SERVOVLA_PYAV_CUDA_BACKEND = "servovla_pyav_cuda"
SERVOVLA_TORCHCODEC_CUDA_BACKEND = "servovla_torchcodec_cuda"
_LEROBOT_NATIVE_BACKENDS = {"torchcodec", "pyav", "video_reader"}
_PATCHED = False
_ORIGINAL_DECODE_VIDEO_FRAMES: Callable[..., torch.Tensor] | None = None
_LOGGED_SUCCESS_BACKENDS: set[str] = set()
_LOGGED_FALLBACK_BACKENDS: set[str] = set()
_PYAV_CUDA_SEQUENTIAL_MAX_GAP_S = 2.0
_PYAV_CUDA_RECENT_FRAME_CACHE_MAX_SECONDS = 4.0
_PYAV_CUDA_RECENT_FRAME_CACHE_MAX_FRAMES = 96


@dataclass(frozen=True)
class VideoDecodeSettings:
    backend: str | None
    device: str | None = None
    fallback_backend: str = "torchcodec"
    allow_software_fallback: bool = True
    num_workers: int | None = None
    decoder_cache_size: int = 8
    output_format: str = "rgb24"
    thread_count: int | None = 1
    thread_type: str | None = "none"
    intra_batch_parallelism: int = 1

    @property
    def requires_adapter(self) -> bool:
        return self.backend in {
            SERVOVLA_PYAV_CUDA_BACKEND,
            SERVOVLA_TORCHCODEC_CUDA_BACKEND,
        }


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _normalize_backend(value: Any) -> str | None:
    if value is None:
        return None
    backend = str(value).strip().lower()
    if backend in {"", "none", "null", "default"}:
        return None
    if backend in {"pyav_cuda", "cuda_pyav", SERVOVLA_PYAV_CUDA_BACKEND}:
        return SERVOVLA_PYAV_CUDA_BACKEND
    if backend in {"torchcodec_cuda", "cuda_torchcodec", SERVOVLA_TORCHCODEC_CUDA_BACKEND}:
        return SERVOVLA_TORCHCODEC_CUDA_BACKEND
    return backend


def _cuda_device_index(device: str | None) -> str | None:
    if device is None:
        return None
    normalized = str(device).strip().lower()
    if not normalized.startswith("cuda"):
        return None
    if ":" not in normalized:
        return None
    index = normalized.split(":", 1)[1]
    return index if index else None


def torchcodec_decode_available() -> bool:
    return importlib.util.find_spec("torchcodec") is not None


def pyav_cuda_decode_available() -> bool:
    try:
        import av
        from av.codec.hwaccel import hwdevices_available

        if "cuda" not in hwdevices_available():
            return False
        av.codec.Codec("h264_cuvid", "r")
        return True
    except Exception:
        return False


def resolve_video_decode_settings(cfg: Any) -> VideoDecodeSettings:
    if cfg is None:
        return VideoDecodeSettings(backend=None)

    requested_backend = _normalize_backend(
        _cfg_get(cfg, "video_backend", _cfg_get(cfg, "backend", "auto"))
    )
    device = _cfg_get(cfg, "device", None)
    fallback_backend = (
        _normalize_backend(_cfg_get(cfg, "fallback_backend", "torchcodec")) or "torchcodec"
    )
    allow_software_fallback = bool(_cfg_get(cfg, "allow_software_fallback", True))
    num_workers_value = _cfg_get(cfg, "num_workers", None)
    num_workers = None if num_workers_value is None else int(num_workers_value)
    if num_workers is not None and num_workers < 0:
        raise ValueError(f"training.decode.num_workers must be >= 0, got {num_workers}")
    decoder_cache_size = int(_cfg_get(cfg, "decoder_cache_size", 8))
    if decoder_cache_size < 1:
        raise ValueError(
            f"training.decode.decoder_cache_size must be >= 1, got {decoder_cache_size}"
        )
    thread_count_value = _cfg_get(cfg, "thread_count", 1)
    thread_count = None if thread_count_value is None else int(thread_count_value)
    if thread_count is not None and thread_count < 0:
        raise ValueError(f"training.decode.thread_count must be >= 0, got {thread_count}")
    thread_type_value = _cfg_get(cfg, "thread_type", "none")
    thread_type = None if thread_type_value is None else str(thread_type_value).strip().lower()
    if thread_type in {"", "default"}:
        thread_type = None
    if thread_type is not None and thread_type not in {"auto", "none", "slice", "frame"}:
        raise ValueError(
            "Unsupported training.decode.thread_type="
            f"{thread_type!r}; supported: ['auto', 'none', 'slice', 'frame']"
        )
    intra_batch_parallelism = int(_cfg_get(cfg, "intra_batch_parallelism", 1))
    if intra_batch_parallelism < 1:
        raise ValueError(
            f"training.decode.intra_batch_parallelism must be >= 1, got {intra_batch_parallelism}"
        )

    if requested_backend == "auto":
        device_name = str(device or "").strip().lower()
        if device_name.startswith("cuda") and pyav_cuda_decode_available():
            backend = SERVOVLA_PYAV_CUDA_BACKEND
        elif torchcodec_decode_available():
            backend = "torchcodec"
        else:
            backend = "pyav"
    else:
        backend = requested_backend

    if backend is not None and backend not in _LEROBOT_NATIVE_BACKENDS | {
        SERVOVLA_PYAV_CUDA_BACKEND,
        SERVOVLA_TORCHCODEC_CUDA_BACKEND,
    }:
        supported = sorted(_LEROBOT_NATIVE_BACKENDS | {"auto", "pyav_cuda", "torchcodec_cuda"})
        raise ValueError(
            f"Unsupported training.decode.video_backend={backend!r}; supported: {supported}"
        )

    output_format = str(_cfg_get(cfg, "output_format", "auto")).strip().lower()
    if output_format in {"", "auto"}:
        output_format = "nv12" if backend == SERVOVLA_PYAV_CUDA_BACKEND else "rgb24"
    if output_format not in {"rgb24", "nv12"}:
        raise ValueError(
            f"Unsupported training.decode.output_format={output_format!r}; supported: ['auto', 'rgb24', 'nv12']"
        )

    return VideoDecodeSettings(
        backend=backend,
        device=str(device) if device is not None else None,
        fallback_backend=fallback_backend,
        allow_software_fallback=allow_software_fallback,
        num_workers=num_workers,
        decoder_cache_size=decoder_cache_size,
        output_format=output_format,
        thread_count=thread_count,
        thread_type=thread_type,
        intra_batch_parallelism=intra_batch_parallelism,
    )


def _frame_time_seconds(frame: Any, stream: Any) -> float:
    if frame.pts is not None:
        return float(frame.pts * stream.time_base)
    return float(frame.time)


def _frame_to_uint8_tensor(frame: Any, *, output_format: str) -> torch.Tensor:
    if output_format == "nv12":
        frame_array = frame.to_ndarray(format="nv12")
        return torch.from_numpy(frame_array).contiguous()
    if output_format != "rgb24":
        raise ValueError(f"Unsupported PyAV CUDA output_format={output_format!r}")
    frame_array = frame.to_ndarray(format="rgb24")
    return torch.from_numpy(frame_array).permute(2, 0, 1).contiguous()


class _PyAvCudaSequentialDecoder:
    def __init__(
        self,
        video_path: Path | str,
        *,
        device: str | None,
        allow_software_fallback: bool,
        thread_count: int | None = 1,
        thread_type: str | None = "none",
    ):
        import av
        from av.codec.hwaccel import HWAccel

        self.video_path = str(video_path)
        hwaccel = HWAccel(
            "cuda",
            device=_cuda_device_index(device),
            allow_software_fallback=bool(allow_software_fallback),
        )
        self.container = av.open(self.video_path, hwaccel=hwaccel)
        self.stream = self.container.streams.video[0]
        self._configure_codec_context(
            self.stream.codec_context,
            thread_count=thread_count,
            thread_type=thread_type,
        )
        self._frame_iter = None
        self._last_ts: float | None = None
        self._last_frame: torch.Tensor | None = None
        self._recent_frames: list[tuple[float, torch.Tensor]] = []

    @staticmethod
    def _configure_codec_context(
        codec_context: Any,
        *,
        thread_count: int | None,
        thread_type: str | None,
    ) -> None:
        if thread_count is not None:
            codec_context.thread_count = int(thread_count)
        if thread_type is None or str(thread_type).lower() == "auto":
            return
        codec_context.thread_type = str(thread_type).upper()

    def close(self) -> None:
        self.container.close()

    def _seek(self, timestamp_s: float) -> None:
        seek_offset = int(float(timestamp_s) / float(self.stream.time_base))
        self.container.seek(seek_offset, stream=self.stream, any_frame=False, backward=True)
        self._frame_iter = self.container.decode(self.stream)
        self._last_ts = None
        self._last_frame = None
        self._recent_frames = []

    def _ensure_recent_frames(self) -> list[tuple[float, torch.Tensor]]:
        recent = getattr(self, "_recent_frames", None)
        if recent is None:
            recent = []
            last_ts = getattr(self, "_last_ts", None)
            last_frame = getattr(self, "_last_frame", None)
            if last_ts is not None and last_frame is not None:
                recent.append((float(last_ts), last_frame))
            self._recent_frames = recent
        return recent

    def _remember_frame(self, timestamp_s: float, frame: torch.Tensor) -> None:
        recent = self._ensure_recent_frames()
        timestamp = float(timestamp_s)
        for idx, (cached_ts, _cached_frame) in enumerate(recent):
            if abs(float(cached_ts) - timestamp) < 1e-7:
                recent[idx] = (timestamp, frame)
                break
        else:
            recent.append((timestamp, frame))
            if len(recent) > 1 and recent[-2][0] > timestamp:
                recent.sort(key=lambda item: item[0])

        if not recent:
            return
        latest_ts = float(recent[-1][0])
        min_ts = latest_ts - _PYAV_CUDA_RECENT_FRAME_CACHE_MAX_SECONDS
        while (
            len(recent) > _PYAV_CUDA_RECENT_FRAME_CACHE_MAX_FRAMES or float(recent[0][0]) < min_ts
        ):
            recent.pop(0)

    def _recent_frame_matches(self, timestamp_s: float, tolerance_s: float) -> bool:
        target = float(timestamp_s)
        tolerance = float(tolerance_s)
        return any(
            abs(float(cached_ts) - target) < tolerance
            for cached_ts, _frame in self._ensure_recent_frames()
        )

    def _recent_frames_for_range(
        self,
        first_ts: float,
        last_ts: float,
        tolerance_s: float,
    ) -> list[tuple[float, torch.Tensor]]:
        lower = float(first_ts) - float(tolerance_s)
        upper = float(last_ts) + float(tolerance_s)
        return [
            (float(timestamp_s), frame)
            for timestamp_s, frame in self._ensure_recent_frames()
            if lower <= float(timestamp_s) <= upper
        ]

    def _should_seek(self, first_ts: float, tolerance_s: float) -> bool:
        if self._frame_iter is None or self._last_ts is None:
            return True
        if first_ts < self._last_ts - float(tolerance_s) and not self._recent_frame_matches(
            first_ts, tolerance_s
        ):
            return True
        return (first_ts - self._last_ts) > _PYAV_CUDA_SEQUENTIAL_MAX_GAP_S

    def decode(
        self, timestamps: list[float], tolerance_s: float, *, output_format: str = "rgb24"
    ) -> torch.Tensor:
        from lerobot.datasets.video_utils import FrameTimestampError

        first_ts = min(float(ts) for ts in timestamps)
        last_ts = max(float(ts) for ts in timestamps)
        self._ensure_recent_frames()
        if self._should_seek(first_ts, tolerance_s):
            self._seek(first_ts)

        loaded_pairs = self._recent_frames_for_range(first_ts, last_ts, tolerance_s)
        loaded_ts = [timestamp_s for timestamp_s, _frame in loaded_pairs]
        loaded_frames = [frame for _timestamp_s, frame in loaded_pairs]

        while not loaded_ts or loaded_ts[-1] < last_ts:
            try:
                frame = next(self._frame_iter)
            except StopIteration as exc:
                if loaded_ts:
                    break
                raise FrameTimestampError(
                    f"Reached end of video before timestamps={timestamps}"
                ) from exc
            current_ts = _frame_time_seconds(frame, self.stream)
            current_frame = _frame_to_uint8_tensor(frame, output_format=output_format)
            loaded_frames.append(current_frame)
            loaded_ts.append(current_ts)
            self._last_ts = current_ts
            self._last_frame = current_frame
            self._remember_frame(current_ts, current_frame)

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        decoded_ts = torch.tensor(loaded_ts, dtype=torch.float32)
        distances = torch.cdist(query_ts[:, None], decoded_ts[:, None], p=1)
        min_dist, argmin = distances.min(1)
        is_within_tol = min_dist < float(tolerance_s)
        if not bool(is_within_tol.all()):
            raise FrameTimestampError(
                f"One or several query timestamps unexpectedly violate the tolerance "
                f"({min_dist[~is_within_tol]} > tolerance_s={tolerance_s}). "
                f"\nqueried timestamps: {query_ts}"
                f"\nloaded timestamps: {decoded_ts}"
                f"\nvideo: {self.video_path}"
            )

        return torch.stack([loaded_frames[int(idx)] for idx in argmin])


class _PyAvCudaDecoderCache:
    def __init__(self, max_size: int = 64):
        self.max_size = max(int(max_size), 1)
        self._cache: OrderedDict[
            tuple[str, str | None, bool, str, int | None, str | None],
            _PyAvCudaSequentialDecoder,
        ] = OrderedDict()
        self._active: dict[tuple[str, str | None, bool, str, int | None, str | None], int] = {}
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "decoder_creates": 0,
            "decoder_evictions": 0,
            "fallbacks": 0,
        }
        self._lock = threading.RLock()

    def _key(
        self,
        video_path: Path | str,
        *,
        device: str | None,
        allow_software_fallback: bool,
        output_format: str,
        thread_count: int | None,
        thread_type: str | None,
    ) -> tuple[str, str | None, bool, str, int | None, str | None]:
        return (
            str(video_path),
            str(device) if device is not None else None,
            bool(allow_software_fallback),
            str(output_format),
            None if thread_count is None else int(thread_count),
            None if thread_type is None else str(thread_type),
        )

    def _evict_inactive_locked(self) -> None:
        while len(self._cache) > self.max_size:
            evicted = False
            for key in list(self._cache.keys()):
                if self._active.get(key, 0) > 0:
                    continue
                old_decoder = self._cache.pop(key)
                old_decoder.close()
                self._stats["decoder_evictions"] += 1
                evicted = True
                break
            if not evicted:
                break

    def set_max_size(self, max_size: int) -> None:
        with self._lock:
            self.max_size = max(int(max_size), 1)
            self._evict_inactive_locked()

    def get(
        self,
        video_path: Path | str,
        *,
        device: str | None,
        allow_software_fallback: bool,
        output_format: str,
        thread_count: int | None,
        thread_type: str | None,
    ) -> _PyAvCudaSequentialDecoder:
        key = self._key(
            video_path,
            device=device,
            allow_software_fallback=allow_software_fallback,
            output_format=output_format,
            thread_count=thread_count,
            thread_type=thread_type,
        )
        with self._lock:
            decoder = self._cache.get(key)
            if decoder is not None:
                self._stats["cache_hits"] += 1
                self._cache.move_to_end(key)
                return decoder

            self._stats["cache_misses"] += 1
            decoder = _PyAvCudaSequentialDecoder(
                video_path,
                device=device,
                allow_software_fallback=allow_software_fallback,
                thread_count=thread_count,
                thread_type=thread_type,
            )
            self._stats["decoder_creates"] += 1
            self._cache[key] = decoder
            self._cache.move_to_end(key)
            self._evict_inactive_locked()
            return decoder

    @contextmanager
    def borrow(
        self,
        video_path: Path | str,
        *,
        device: str | None,
        allow_software_fallback: bool,
        output_format: str,
        thread_count: int | None,
        thread_type: str | None,
    ):
        key = self._key(
            video_path,
            device=device,
            allow_software_fallback=allow_software_fallback,
            output_format=output_format,
            thread_count=thread_count,
            thread_type=thread_type,
        )
        with self._lock:
            decoder = self.get(
                video_path,
                device=device,
                allow_software_fallback=allow_software_fallback,
                output_format=output_format,
                thread_count=thread_count,
                thread_type=thread_type,
            )
            self._active[key] = self._active.get(key, 0) + 1
        try:
            yield decoder
        finally:
            with self._lock:
                remaining = self._active.get(key, 0) - 1
                if remaining > 0:
                    self._active[key] = remaining
                else:
                    self._active.pop(key, None)
                self._evict_inactive_locked()

    def clear(self) -> None:
        with self._lock:
            for decoder in self._cache.values():
                decoder.close()
            self._cache.clear()
            self._active.clear()

    def record_fallback(self) -> None:
        with self._lock:
            self._stats["fallbacks"] += 1

    def snapshot_stats(self) -> dict[str, int]:
        with self._lock:
            return {
                **self._stats,
                "cache_size": len(self._cache),
                "cache_active": sum(int(value) for value in self._active.values()),
            }


_DEFAULT_PYAV_CUDA_DECODER_CACHE = _PyAvCudaDecoderCache()


def get_decode_stats_snapshot() -> dict[str, int]:
    return _DEFAULT_PYAV_CUDA_DECODER_CACHE.snapshot_stats()


def diff_decode_stats(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    cumulative_keys = {
        "cache_hits",
        "cache_misses",
        "decoder_creates",
        "decoder_evictions",
        "fallbacks",
    }
    gauge_keys = {"cache_size", "cache_active"}
    result: dict[str, int] = {}
    for key in cumulative_keys:
        result[key] = int(after.get(key, 0)) - int(before.get(key, 0))
    for key in gauge_keys:
        result[key] = int(after.get(key, 0))
    return result


def decode_video_frames_pyav_cuda(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    *,
    device: str | None,
    allow_software_fallback: bool,
    decoder_cache_size: int = 8,
    output_format: str = "nv12",
    thread_count: int | None = 1,
    thread_type: str | None = "none",
) -> torch.Tensor:
    if not timestamps:
        raise ValueError("timestamps must contain at least one timestamp")

    _DEFAULT_PYAV_CUDA_DECODER_CACHE.set_max_size(decoder_cache_size)
    with _DEFAULT_PYAV_CUDA_DECODER_CACHE.borrow(
        video_path,
        device=device,
        allow_software_fallback=allow_software_fallback,
        output_format=output_format,
        thread_count=thread_count,
        thread_type=thread_type,
    ) as decoder:
        return decoder.decode(timestamps, tolerance_s, output_format=output_format)


def decode_video_frames_torchcodec_cuda(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    *,
    device: str | None,
) -> torch.Tensor:
    from lerobot.datasets.video_utils import FrameTimestampError
    from torchcodec.decoders import VideoDecoder

    if not timestamps:
        raise ValueError("timestamps must contain at least one timestamp")

    decoder = VideoDecoder(
        str(video_path),
        device=str(device or "cuda"),
        seek_mode="approximate",
        dimension_order="NCHW",
    )
    average_fps = decoder.metadata.average_fps
    frame_indices = [round(float(ts) * average_fps) for ts in timestamps]
    frames_batch = decoder.get_frames_at(indices=frame_indices)
    query_ts = torch.tensor(timestamps, device=frames_batch.pts_seconds.device)
    decoded_ts = frames_batch.pts_seconds.to(dtype=query_ts.dtype)
    distances = torch.cdist(query_ts[:, None], decoded_ts[:, None], p=1)
    min_dist, _argmin = distances.min(1)
    if not bool((min_dist < float(tolerance_s)).all()):
        raise FrameTimestampError(
            f"One or several query timestamps unexpectedly violate the tolerance "
            f"({min_dist} > tolerance_s={tolerance_s}). video: {video_path}"
        )
    return frames_batch.data.to(dtype=torch.uint8)


def make_lerobot_decode_adapter(
    settings: VideoDecodeSettings,
    *,
    original_decode: Callable[..., torch.Tensor],
) -> Callable[..., torch.Tensor]:
    def _decode(video_path, timestamps, tolerance_s, backend=None):
        if backend != settings.backend:
            return original_decode(video_path, timestamps, tolerance_s, backend)
        try:
            if settings.backend == SERVOVLA_PYAV_CUDA_BACKEND:
                output = decode_video_frames_pyav_cuda(
                    video_path,
                    timestamps,
                    tolerance_s,
                    device=settings.device,
                    allow_software_fallback=settings.allow_software_fallback,
                    decoder_cache_size=settings.decoder_cache_size,
                    output_format=settings.output_format,
                    thread_count=settings.thread_count,
                    thread_type=settings.thread_type,
                )
            elif settings.backend == SERVOVLA_TORCHCODEC_CUDA_BACKEND:
                output = decode_video_frames_torchcodec_cuda(
                    video_path,
                    timestamps,
                    tolerance_s,
                    device=settings.device,
                )
            else:
                return original_decode(video_path, timestamps, tolerance_s, backend)
            if settings.backend not in _LOGGED_SUCCESS_BACKENDS:
                _LOGGED_SUCCESS_BACKENDS.add(str(settings.backend))
                log.info(
                    "ServoVLA video decode backend active: %s device=%s output_format=%s thread_count=%s thread_type=%s",
                    settings.backend,
                    settings.device,
                    settings.output_format,
                    settings.thread_count,
                    settings.thread_type,
                )
            return output
        except Exception as exc:
            fallback_backend = settings.fallback_backend
            if not fallback_backend:
                raise
            if settings.backend == SERVOVLA_PYAV_CUDA_BACKEND:
                _DEFAULT_PYAV_CUDA_DECODER_CACHE.record_fallback()
            if settings.backend not in _LOGGED_FALLBACK_BACKENDS:
                _LOGGED_FALLBACK_BACKENDS.add(str(settings.backend))
                log.warning(
                    "ServoVLA video decode backend %s failed on device=%s; falling back to %s. error=%r",
                    settings.backend,
                    settings.device,
                    fallback_backend,
                    exc,
                )
            return original_decode(video_path, timestamps, tolerance_s, fallback_backend)

    return _decode


def install_lerobot_video_decode_adapter(settings: VideoDecodeSettings) -> str | None:
    global _ORIGINAL_DECODE_VIDEO_FRAMES, _PATCHED
    if settings.backend is None:
        return None
    if not settings.requires_adapter:
        return settings.backend

    from lerobot.datasets import dataset_reader, video_utils

    if _ORIGINAL_DECODE_VIDEO_FRAMES is None:
        _ORIGINAL_DECODE_VIDEO_FRAMES = video_utils.decode_video_frames
    adapter = make_lerobot_decode_adapter(settings, original_decode=_ORIGINAL_DECODE_VIDEO_FRAMES)
    video_utils.decode_video_frames = adapter
    dataset_reader.decode_video_frames = adapter
    _PATCHED = True
    log.info(
        "Configured LeRobot video decode backend: backend=%s device=%s fallback=%s decoder_cache_size=%d output_format=%s thread_count=%s thread_type=%s",
        settings.backend,
        settings.device,
        settings.fallback_backend,
        settings.decoder_cache_size,
        settings.output_format,
        settings.thread_count,
        settings.thread_type,
    )
    return settings.backend
