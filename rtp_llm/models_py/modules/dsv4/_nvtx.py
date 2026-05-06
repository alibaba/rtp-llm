"""DSV4 profiling ranges.

``record_function`` annotations are always emitted so Chrome timeline JSONs
carry coarse DSV4 attribution by default.  ``DSV4_NVTX=1`` only controls
CUDA NVTX push/pop markers for external profilers.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch


def enabled() -> bool:
    return os.environ.get("DSV4_NVTX", "0") == "1"


@contextmanager
def nvtx_range(name: str):
    with torch.autograd.profiler.record_function(name):
        pushed = False
        if enabled() and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
            pushed = True
        try:
            yield
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()
