"""Profiler ranges and suppression shared by model implementations."""

from __future__ import annotations

import contextvars
import sys
from contextlib import contextmanager
from typing import Optional, Type

import torch

_DISABLED_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "fused_moe_record_function_ranges_disabled",
    default=0,
)


class _NoopRecordFunctionRange:
    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb,
    ) -> bool:
        return False


_NOOP_RECORD_FUNCTION_RANGE = _NoopRecordFunctionRange()


def is_light_timeline() -> bool:
    # Standalone models can also run under a Python-owned profiler.
    probe = getattr(sys.modules.get("libth_transformer"), "is_light_timeline", None)
    return probe is not None and probe()


def active_layer_forward_range():
    # Choose the range type once per forward. Full traces avoid GPU projections
    # of each layer by using FUNCTION instead of USER_SCOPE.
    record = (
        torch.profiler.record_function
        if is_light_timeline()
        else torch._C._profiler._RecordFunctionFast
    )
    return lambda layer_idx: record(f"forward(layer={layer_idx})")


def record_function_ranges_enabled() -> bool:
    return _DISABLED_DEPTH.get() <= 0


def record_function_range(name: str):
    if not record_function_ranges_enabled():
        return _NOOP_RECORD_FUNCTION_RANGE
    return torch.profiler.record_function(name)


@contextmanager
def disable_record_function_ranges():
    token = _DISABLED_DEPTH.set(_DISABLED_DEPTH.get() + 1)
    try:
        yield
    finally:
        _DISABLED_DEPTH.reset(token)
