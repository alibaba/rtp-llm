from __future__ import annotations

import torch

from rtp_llm.ops.compute_ops import DeviceType, get_device


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


def dispose_tensor(x: torch.Tensor):
    x.set_(torch.empty((0,), device=x.device, dtype=x.dtype))  # type: ignore


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
