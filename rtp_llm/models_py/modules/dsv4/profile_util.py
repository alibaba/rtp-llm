"""Torch-profiler scope decorator for DSV4 modules.

When torch.profiler is not active the decorator is a near-zero-cost
no-op (record_function only emits through at::RecordFunction callbacks
when at::hasCallbacks() is true).

Annotated points show up in the Chrome/Perfetto timeline as named
ranges so engineers can correlate GPU kernel bursts with logical
stages (indexer / compressor / attention / MoE).
"""

import functools

import torch


def record(name: str):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with torch.profiler.record_function(name):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
