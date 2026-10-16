"""DSV4 Torch profiler ranges."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def record_function_range(name: str):
    # These ranges have no work when neither Kineto nor the autograd/NVTX
    # profiler is active. Avoid hundreds of Python/C++ context entries per
    # forward, while checking dynamically so later profiling still works.
    if torch.autograd._profiler_enabled():
        return torch.profiler.record_function(name)
    return nullcontext()
