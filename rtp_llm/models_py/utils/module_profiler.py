"""Optional model ranges, with no record_function work in unprofiled runs."""

import os
from contextlib import nullcontext

import torch

_ENABLED = os.environ.get("RTP_LLM_PROFILE_MODEL_MODULES", "0") == "1"
_NO_SCOPE = nullcontext()


def profile_scope(name):
    return (
        torch.profiler.record_function(name)
        if _ENABLED and torch.autograd._profiler_enabled()
        else _NO_SCOPE
    )
