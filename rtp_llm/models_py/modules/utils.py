from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.ops import DeviceExporter, DeviceType, get_device


# COPIED FROM DeepGEMM
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM DeepGEMM
def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))


def is_cuda():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.Cuda:
        return True
    else:
        return False


def is_hip():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.ROCm:
        return True
    else:
        return False


def is_ppu():
    device_type = get_device().get_device_type()
    if device_type == DeviceType.Ppu:
        return True
    else:
        return False
