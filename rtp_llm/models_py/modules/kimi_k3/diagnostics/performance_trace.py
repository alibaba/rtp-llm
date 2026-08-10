"""Performance-analysis ranges for Kimi K3.

Production serving keeps the legacy fine-grained ranges disabled.  The cuLA
range remains active so external profilers can attribute the third-party call.
"""

from contextlib import nullcontext
from typing import Optional

import torch


def performance_profile(
    name: str, tensor: Optional[torch.Tensor] = None
):
    del name, tensor
    return nullcontext()


def active_performance_profile(name: str):
    return torch.autograd.profiler.record_function(name)


__all__ = ["active_performance_profile", "performance_profile"]
