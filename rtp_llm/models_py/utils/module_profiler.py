"""Optional model ranges, with no record_function work in unprofiled runs."""

from contextlib import nullcontext

import torch

_NO_SCOPE = nullcontext()


def profile_scope(name):
    return (
        torch.profiler.record_function(name)
        if torch.autograd._profiler_enabled()
        else _NO_SCOPE
    )
