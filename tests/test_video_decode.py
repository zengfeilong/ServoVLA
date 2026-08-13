from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from servovla.data import video_decode


def test_auto_video_decode_prefers_pyav_cuda_when_available(monkeypatch):
    monkeypatch.setattr(video_decode, "pyav_cuda_decode_available", lambda: True)
    cfg = OmegaConf.create({"device": "cuda:2", "video_backend": "auto"})

    settings = video_decode.resolve_video_decode_settings(cfg)

    assert settings.backend == video_decode.SERVOVLA_PYAV_CUDA_BACKEND
    assert settings.device == "cuda:2"


def test_auto_video_decode_falls_back_to_torchcodec_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(video_decode, "pyav_cuda_decode_available", lambda: False)
    monkeypatch.setattr(video_decode, "torchcodec_decode_available", lambda: True)
    cfg = OmegaConf.create({"device": "cuda:2", "video_backend": "auto"})

    settings = video_decode.resolve_video_decode_settings(cfg)

    assert settings.backend == "torchcodec"


def test_video_decode_settings_parse_worker_cap():
    cfg = OmegaConf.create(
        {
            "device": "cuda:2",
            "video_backend": "pyav_cuda",
            "num_workers": 1,
            "decoder_cache_size": 8,
        }
    )

    settings = video_decode.resolve_video_decode_settings(cfg)

    assert settings.backend == video_decode.SERVOVLA_PYAV_CUDA_BACKEND
    assert settings.device == "cuda:2"
    assert settings.num_workers == 1
    assert settings.decoder_cache_size == 8
    assert settings.output_format == "nv12"
    assert settings.thread_count == 1
    assert settings.thread_type == "none"


def test_video_decode_settings_allow_explicit_pyav_thread_options():
    cfg = OmegaConf.create(
        {
            "device": "cuda:2",
            "video_backend": "pyav_cuda",
            "thread_count": 2,
            "thread_type": "slice",
        }
    )

    settings = video_decode.resolve_video_decode_settings(cfg)

    assert settings.thread_count == 2
    assert settings.thread_type == "slice"


def test_video_decode_settings_parse_intra_batch_parallelism():
    cfg = OmegaConf.create(
        {
            "device": "cuda:2",
            "video_backend": "pyav_cuda",
            "intra_batch_parallelism": 3,
        }
    )

    settings = video_decode.resolve_video_decode_settings(cfg)

    assert settings.intra_batch_parallelism == 3


def test_pyav_cuda_frame_conversion_can_keep_nv12_without_rgb_cpu_conversion():
    calls = []

    class _Frame:
        def to_ndarray(self, *, format):
            calls.append(format)
            if format == "nv12":
                return np.arange(6 * 4, dtype=np.uint8).reshape(6, 4)
            return np.zeros((4, 4, 3), dtype=np.uint8)

    tensor = video_decode._frame_to_uint8_tensor(_Frame(), output_format="nv12")

    assert calls == ["nv12"]
    assert tensor.shape == (6, 4)
    assert tensor.dtype == torch.uint8


def test_pyav_cuda_decoder_cache_size_is_bounded(monkeypatch):
    closed = []
    created = []

    class _Decoder:
        def __init__(
            self, video_path, *, device, allow_software_fallback, thread_count, thread_type
        ):
            self.video_path = str(video_path)
            created.append(
                (self.video_path, device, allow_software_fallback, thread_count, thread_type)
            )

        def close(self):
            closed.append(self.video_path)

    monkeypatch.setattr(video_decode, "_PyAvCudaSequentialDecoder", _Decoder)
    cache = video_decode._PyAvCudaDecoderCache(max_size=2)

    cache.get(
        "a.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    cache.get(
        "b.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    cache.get(
        "c.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )

    assert [item[0] for item in created] == ["a.mp4", "b.mp4", "c.mp4"]
    assert closed == ["a.mp4"]
    assert len(cache._cache) == 2


def test_pyav_cuda_decoder_cache_evicts_inactive_entries_after_active_oldest(monkeypatch):
    closed = []

    class _Decoder:
        def __init__(
            self, video_path, *, device, allow_software_fallback, thread_count, thread_type
        ):
            self.video_path = str(video_path)

        def close(self):
            closed.append(self.video_path)

    monkeypatch.setattr(video_decode, "_PyAvCudaSequentialDecoder", _Decoder)
    cache = video_decode._PyAvCudaDecoderCache(max_size=2)

    borrow = cache.borrow(
        "a.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    borrow.__enter__()
    try:
        cache.get(
            "b.mp4",
            device="cuda:2",
            allow_software_fallback=True,
            output_format="nv12",
            thread_count=1,
            thread_type="none",
        )
        cache.get(
            "c.mp4",
            device="cuda:2",
            allow_software_fallback=True,
            output_format="nv12",
            thread_count=1,
            thread_type="none",
        )

        assert closed == ["b.mp4"]
        assert list(cache._cache.keys()) == [
            ("a.mp4", "cuda:2", True, "nv12", 1, "none"),
            ("c.mp4", "cuda:2", True, "nv12", 1, "none"),
        ]
    finally:
        borrow.__exit__(None, None, None)


def test_pyav_cuda_decoder_cache_reports_counter_deltas(monkeypatch):
    class _Decoder:
        def __init__(
            self, video_path, *, device, allow_software_fallback, thread_count, thread_type
        ):
            self.video_path = str(video_path)

        def close(self):
            pass

    monkeypatch.setattr(video_decode, "_PyAvCudaSequentialDecoder", _Decoder)
    cache = video_decode._PyAvCudaDecoderCache(max_size=2)

    before = cache.snapshot_stats()
    cache.get(
        "a.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    cache.get(
        "a.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    cache.get(
        "b.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )
    cache.get(
        "c.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        output_format="nv12",
        thread_count=1,
        thread_type="none",
    )

    delta = video_decode.diff_decode_stats(before, cache.snapshot_stats())

    assert delta == {
        "cache_hits": 1,
        "cache_misses": 3,
        "decoder_creates": 3,
        "decoder_evictions": 1,
        "cache_size": 2,
        "cache_active": 0,
        "fallbacks": 0,
    }


def test_pyav_cuda_decoder_limits_ffmpeg_threads_before_decode(monkeypatch):
    opened = []

    class _CodecContext:
        def __init__(self):
            self.thread_count = 0
            self.thread_type = "SLICE"

    class _Stream:
        time_base = 1.0

        def __init__(self):
            self.codec_context = _CodecContext()

    class _Streams:
        def __init__(self):
            self.video = [_Stream()]

    class _Container:
        def __init__(self, path, hwaccel):
            self.path = str(path)
            self.hwaccel = hwaccel
            self.streams = _Streams()

        def close(self):
            pass

    class _HWAccel:
        def __init__(self, device_type, *, device, allow_software_fallback):
            self.device_type = device_type
            self.device = device
            self.allow_software_fallback = allow_software_fallback

    class _Av:
        @staticmethod
        def open(path, *, hwaccel):
            container = _Container(path, hwaccel)
            opened.append(container)
            return container

    monkeypatch.setitem(__import__("sys").modules, "av", _Av)
    monkeypatch.setitem(
        __import__("sys").modules, "av.codec.hwaccel", type("_HwMod", (), {"HWAccel": _HWAccel})
    )

    decoder = video_decode._PyAvCudaSequentialDecoder(
        "video.mp4",
        device="cuda:2",
        allow_software_fallback=True,
        thread_count=1,
        thread_type="none",
    )

    assert decoder.stream.codec_context.thread_count == 1
    assert decoder.stream.codec_context.thread_type == "NONE"
    assert opened[0].hwaccel.device == "2"


def test_pyav_cuda_decoder_uses_cached_tail_frame_when_eof_is_within_tolerance():
    decoder = object.__new__(video_decode._PyAvCudaSequentialDecoder)
    decoder.video_path = "video.mp4"
    decoder.stream = object()
    decoder._frame_iter = iter(())
    decoder._last_ts = 76.13330
    decoder._last_frame = torch.full((3, 2, 2), 128, dtype=torch.uint8)

    frames = decoder.decode([76.13333396911621], tolerance_s=1e-4)

    assert frames.shape == (1, 3, 2, 2)
    assert frames.dtype == torch.uint8
    assert torch.equal(frames[0], torch.full((3, 2, 2), 128, dtype=torch.uint8))


def test_pyav_cuda_decoder_reuses_recent_frames_before_continuing_forward(monkeypatch):
    class _Frame:
        def __init__(self, timestamp, value):
            self.timestamp = float(timestamp)
            self.value = int(value)

    monkeypatch.setattr(
        video_decode,
        "_frame_time_seconds",
        lambda frame, stream: float(frame.timestamp),
    )
    monkeypatch.setattr(
        video_decode,
        "_frame_to_uint8_tensor",
        lambda frame, *, output_format: torch.full((3, 2, 2), int(frame.value), dtype=torch.uint8),
    )

    decoder = object.__new__(video_decode._PyAvCudaSequentialDecoder)
    decoder.video_path = "video.mp4"
    decoder.stream = object()
    decoder._frame_iter = iter(
        [
            _Frame(1.0, 10),
            _Frame(1.1, 11),
            _Frame(1.2, 12),
            _Frame(1.3, 13),
        ]
    )
    decoder._last_ts = 0.9
    decoder._last_frame = torch.full((3, 2, 2), 9, dtype=torch.uint8)
    seek_calls = []

    def _unexpected_seek(timestamp_s):
        seek_calls.append(float(timestamp_s))
        raise AssertionError("recent cached frames should avoid seeking backwards")

    decoder._seek = _unexpected_seek

    first = decoder.decode([1.0, 1.2], tolerance_s=1e-4)
    second = decoder.decode([1.1, 1.3], tolerance_s=1e-4)

    assert seek_calls == []
    assert [int(frame[0, 0, 0].item()) for frame in first] == [10, 12]
    assert [int(frame[0, 0, 0].item()) for frame in second] == [11, 13]


def test_decode_adapter_routes_pyav_cuda_backend_and_falls_back(monkeypatch):
    settings = video_decode.VideoDecodeSettings(
        backend=video_decode.SERVOVLA_PYAV_CUDA_BACKEND,
        device="cuda:2",
        fallback_backend="torchcodec",
        allow_software_fallback=True,
    )
    calls = []

    def _failing_hw_decode(
        video_path,
        timestamps,
        tolerance_s,
        *,
        device,
        allow_software_fallback,
        decoder_cache_size,
        output_format,
        thread_count,
        thread_type,
    ):
        calls.append(
            (
                "hw",
                video_path,
                timestamps,
                tolerance_s,
                device,
                allow_software_fallback,
                decoder_cache_size,
                output_format,
                thread_count,
                thread_type,
            )
        )
        raise RuntimeError("no cuda decoder")

    def _fallback_decode(video_path, timestamps, tolerance_s, backend=None):
        calls.append(("fallback", video_path, timestamps, tolerance_s, backend))
        return torch.zeros(1, 3, 2, 2)

    monkeypatch.setattr(video_decode, "decode_video_frames_pyav_cuda", _failing_hw_decode)
    adapter = video_decode.make_lerobot_decode_adapter(settings, original_decode=_fallback_decode)

    output = adapter("video.mp4", [0.0], 1e-4, video_decode.SERVOVLA_PYAV_CUDA_BACKEND)

    assert output.shape == (1, 3, 2, 2)
    assert calls == [
        ("hw", "video.mp4", [0.0], 1e-4, "cuda:2", True, 8, "rgb24", 1, "none"),
        ("fallback", "video.mp4", [0.0], 1e-4, "torchcodec"),
    ]
