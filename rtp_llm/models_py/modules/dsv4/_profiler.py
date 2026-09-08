"""DSV4 Torch profiler ranges."""

from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from typing import Optional, Type

import torch


_RANGES_ENABLED = os.environ.get("DSV4_RECORD_FUNCTION_RANGES", "1") != "0"
_DISABLED_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "dsv4_record_function_ranges_disabled", default=0
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
    return _RANGES_ENABLED and _DISABLED_DEPTH.get() <= 0


def record_function_range(name: str):
    if not record_function_ranges_enabled():
        return _NOOP_RECORD_FUNCTION_RANGE
    return torch.profiler.record_function(name)


@contextmanager
def disable_record_function_ranges():
    depth = _DISABLED_DEPTH.get()
    token = _DISABLED_DEPTH.set(depth + 1)
    try:
        yield
    finally:
        _DISABLED_DEPTH.reset(token)
