import os
from enum import IntEnum

import torch


class DeviceType(IntEnum):
    Cpu = 0
    Cuda = 1
    Yitian = 2
    ArmCpu = 3
    ROCm = 4
    Ppu = 5
    Ascend = 6


def get_device_type() -> DeviceType:
    if torch.cuda.is_available():
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return DeviceType.ROCm
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return DeviceType.Ppu
        return DeviceType.Cuda
    try:
        import torch_npu
        if torch.npu.is_available():
            return DeviceType.Ascend
    except ImportError:
        pass
    return DeviceType.Cpu


def is_cuda() -> bool:
    return get_device_type() == DeviceType.Cuda


def is_hip() -> bool:
    return get_device_type() == DeviceType.ROCm


def is_ascend() -> bool:
    return get_device_type() == DeviceType.Ascend
