"""DSV4 Torch profiler ranges."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def record_function_range(name: str):
    is_compiling = getattr(getattr(torch, "compiler", None), "is_compiling", None)
    if is_compiling is not None and is_compiling():
        yield
        return
    with torch.profiler.record_function(name):
        yield
