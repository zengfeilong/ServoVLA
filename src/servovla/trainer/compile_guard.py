from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import Any

import torch

logger = logging.getLogger(__name__)

_TORCH_COMPILE_CONCURRENCY_LOCK = RLock()


@contextmanager
def torch_compile_concurrency_guard() -> Iterator[None]:
    """Protect torch.compile/FX trace phases from concurrent module calls."""
    with _TORCH_COMPILE_CONCURRENCY_LOCK:
        yield


def apply_torch_compile_nested_fx_trace_fallback() -> bool:
    """Allow eager fallback when FX tracing reaches a compiled callable.

    Some Qwen/FA2 paths still enter FX tracing while a surrounding module has
    already been wrapped by Dynamo.  PyTorch can run the original callable in
    that nested situation if `error_on_nested_fx_trace` is disabled; this keeps
    compilation usable without changing model-facing tensors.
    """

    config = getattr(getattr(torch, "_dynamo", None), "config", None)
    if config is None or not hasattr(config, "error_on_nested_fx_trace"):
        return False
    if getattr(config, "error_on_nested_fx_trace") is False:
        return False
    setattr(config, "error_on_nested_fx_trace", False)
    return True


def apply_transformers_flash_attention_compile_graph_break() -> bool:
    """Keep Transformers FlashAttention eager when compiling Qwen-style VLMs.

    Current torch/flash-attn combinations can repeatedly fail fake-tensor tracing
    for varlen FlashAttention.  Marking the Transformers wrapper as a graph break
    keeps the surrounding module eligible for torch.compile while leaving the FA2
    kernel call itself in eager mode.
    """

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:
        disable = torch._dynamo.disable

    replacements: dict[int, Any] = {}
    patched = False

    def _disabled(fn):
        nonlocal patched
        if fn is None or bool(getattr(fn, "_servovla_compile_graph_break", False)):
            return fn
        fn_id = id(fn)
        if fn_id in replacements:
            return replacements[fn_id]
        wrapped = disable(fn, recursive=True)
        setattr(wrapped, "_servovla_compile_graph_break", True)
        setattr(wrapped, "_servovla_original", fn)
        replacements[fn_id] = wrapped
        patched = True
        return wrapped

    def _patch_module_attr(module_name: str, attr_name: str) -> None:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug(
                "Transformers flash attention module patch skipped for %s: %s", module_name, exc
            )
            return
        fn = getattr(module, attr_name, None)
        wrapped = _disabled(fn)
        if wrapped is not fn:
            setattr(module, attr_name, wrapped)

    _patch_module_attr("transformers.integrations.flash_attention", "flash_attention_forward")
    _patch_module_attr("transformers.modeling_flash_attention_utils", "_flash_attention_forward")

    try:
        modeling_utils = importlib.import_module("transformers.modeling_utils")
    except Exception as exc:
        logger.debug("Transformers attention registry patch skipped: %s", exc)
        return patched

    attention_functions = getattr(modeling_utils, "ALL_ATTENTION_FUNCTIONS", None)
    for mapping_name in ("_global_mapping", "_local_mapping"):
        mapping = getattr(attention_functions, mapping_name, None)
        if not isinstance(mapping, dict):
            continue
        for key, fn in list(mapping.items()):
            if "flash_attention" not in str(key):
                continue
            wrapped = _disabled(fn)
            if wrapped is not fn:
                mapping[key] = wrapped

    return patched


def apply_transformers_qwen_visual_position_compile_graph_break() -> bool:
    """Keep Qwen visual position helpers eager under torch.compile.

    Qwen3/3.5 VL visual encoders build RoPE tables from `grid_thw.tolist()` and
    `torch.arange(max_hw)`.  Dynamo can repeatedly fail on the resulting
    symbolic `Max(...)` expression during fake-tensor tracing; these helpers are
    small and shape-driven, so treating them as eager islands avoids compile
    retries without changing model-facing tensors.
    """

    disable = getattr(getattr(torch, "compiler", None), "disable", None)
    if disable is None:
        disable = torch._dynamo.disable

    module_names = (
        "transformers.models.qwen3_5.modeling_qwen3_5",
        "transformers.models.qwen3_vl.modeling_qwen3_vl",
        "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
        "transformers.models.qwen2_vl.modeling_qwen2_vl",
    )
    method_names = ("rot_pos_emb", "fast_pos_embed_interpolate")
    replacements: dict[int, Any] = {}
    patched = False

    def _disabled(fn):
        nonlocal patched
        if fn is None or bool(getattr(fn, "_servovla_compile_graph_break", False)):
            return fn
        fn_id = id(fn)
        if fn_id in replacements:
            return replacements[fn_id]
        wrapped = disable(fn, recursive=True)
        setattr(wrapped, "_servovla_compile_graph_break", True)
        setattr(wrapped, "_servovla_original", fn)
        replacements[fn_id] = wrapped
        patched = True
        return wrapped

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            logger.debug("Qwen visual position compile patch skipped for %s: %s", module_name, exc)
            continue
        for obj in vars(module).values():
            if not isinstance(obj, type):
                continue
            for method_name in method_names:
                fn = obj.__dict__.get(method_name)
                if fn is None:
                    continue
                wrapped = _disabled(fn)
                if wrapped is not fn:
                    setattr(obj, method_name, wrapped)

    return patched
