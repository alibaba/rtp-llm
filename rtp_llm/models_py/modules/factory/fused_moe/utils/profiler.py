"""Profiler ranges shared by generic fused-MoE implementations."""

from __future__ import annotations

import contextvars
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
