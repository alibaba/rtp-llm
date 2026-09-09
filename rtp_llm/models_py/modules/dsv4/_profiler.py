"""DSV4 Torch profiler ranges."""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    _NOOP_RECORD_FUNCTION_RANGE,
    disable_record_function_ranges,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.profiler import (
    record_function_ranges_enabled as _generic_ranges_enabled,
)

_RANGES_ENABLED = os.environ.get("DSV4_RECORD_FUNCTION_RANGES", "1") != "0"


def record_function_ranges_enabled() -> bool:
    return _RANGES_ENABLED and _generic_ranges_enabled()


def record_function_range(name: str):
    if not record_function_ranges_enabled():
        return _NOOP_RECORD_FUNCTION_RANGE
    return torch.profiler.record_function(name)


@contextmanager
def moe_record_function_scope():
    """Propagate the DSV4 profiler switch into the generic MoE call."""
    if _RANGES_ENABLED:
        yield
        return
    with disable_record_function_ranges():
        yield
