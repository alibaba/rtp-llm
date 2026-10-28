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
    return DeviceType.Cpu


def is_cuda() -> bool:
    return get_device_type() == DeviceType.Cuda


def is_hip() -> bool:
    return get_device_type() == DeviceType.ROCm


def is_ppu() -> bool:
    return get_device_type() == DeviceType.Ppu


def is_sm90(device=None) -> bool:
    """Gate on CUDA SM90 capability across all device SKUs.

    Whole-token name matching never matched H20-3e GPUs ("NVIDIA H20-3e"
    has no "H20" token), leaving the H20-gated decode optimizations inert
    on the target fleet. Gate by compute capability instead; every SM90
    part (H20, H20-3e, H100, H800, H200) matches here.
    """
    return is_cuda() and torch.cuda.get_device_capability(device) == (9, 0)


def is_h20(device=None) -> bool:
    """Compatibility alias for callers that historically meant SM90."""
    return is_sm90(device)
