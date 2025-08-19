from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import triton
from torch import nn
from enum import Enum

# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y

# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y

def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))

def is_cuda():
    return torch.cuda.is_available() and torch.version.cuda

# https://pytorch.org/docs/stable/notes/hip.html#checking-for-hip
def is_hip() -> bool:
    return torch.version.hip is not None


class DeepEPMode(Enum):
    normal = "normal"
    low_latency = "low_latency"
    auto = "auto"

    def enable_normal(self):
        return self in [DeepEPMode.normal, DeepEPMode.auto]

    def enable_low_latency(self):
        return self in [DeepEPMode.low_latency, DeepEPMode.auto]

    def resolve(self, forward_mode):
        if self != DeepEPMode.auto:
            return self

        if forward_mode.is_decode():
            return DeepEPMode.low_latency
        else:
            return DeepEPMode.normal
        

def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: Optional[Dict[str, Any]],
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


def get_compiler_backend() -> str:
    if hasattr(torch, "hpu") and torch.hpu.is_available():
        return "hpu_backend"

    if hasattr(torch, "npu") and torch.npu.is_available():
        import torchair

        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        return npu_backend

    return "inductor"