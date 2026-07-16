"""DSV4 Torch profiler ranges."""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def record_function_range(name: str):
    with torch.profiler.record_function(name):
        yield
