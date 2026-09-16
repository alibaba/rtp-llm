"""DSV4 Torch profiler ranges."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Callable, ContextManager

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    _NOOP_RECORD_FUNCTION_RANGE,
    disable_record_function_ranges,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_ranges_enabled as _generic_ranges_enabled,
)

_RANGES_ENABLED = os.environ.get("DSV4_RECORD_FUNCTION_RANGES", "1") != "0"

LayerForwardRange = Callable[[int], ContextManager[Any]]


def record_function_ranges_enabled() -> bool:
    return _RANGES_ENABLED and _generic_ranges_enabled()


def record_function_range(name: str):
    if not record_function_ranges_enabled():
        return _NOOP_RECORD_FUNCTION_RANGE
    return torch.profiler.record_function(name)


def _noop_layer_forward_range(_layer_idx: int) -> ContextManager[Any]:
    return _NOOP_RECORD_FUNCTION_RANGE


def _active_layer_forward_range(layer_idx: int) -> ContextManager[Any]:
    # FUNCTION scope retains CPU layer attribution without projecting a user
    # annotation onto every GPU stream covered by the range.
    return torch._C._profiler._RecordFunctionFast(f"forward(layer={layer_idx})")


def make_layer_forward_range() -> LayerForwardRange:
    """Capture profiler state before the fast path suppresses nested ranges."""
    if not record_function_ranges_enabled():
        return _noop_layer_forward_range
    profiler_enabled = getattr(torch.autograd, "_profiler_enabled", None)
    if profiler_enabled is not None and not profiler_enabled():
        return _noop_layer_forward_range
    return _active_layer_forward_range


@contextmanager
def moe_record_function_scope():
    """Propagate the DSV4 profiler switch into the generic MoE call."""
    if _RANGES_ENABLED:
        yield
        return
    with disable_record_function_ranges():
        yield
