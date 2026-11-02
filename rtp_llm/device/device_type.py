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
    Xpu = 6


def get_device_type() -> DeviceType:
    # Intel GPUs live under torch.xpu, a namespace separate from torch.cuda.
    # Unlike ROCm -- where HIP masquerades as cuda, which is why the ROCm check
    # below is nested inside torch.cuda.is_available() -- XPU does not. So
    # torch.cuda.is_available() is False on an XPU host and the chain would fall
    # through to Cpu, silently running on CPU on a machine with a GPU. Test XPU
    # first so the device is reported, not inferred.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return DeviceType.Xpu
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


def is_xpu() -> bool:
    return get_device_type() == DeviceType.Xpu


def torch_device_type() -> str:
    """The torch device-type string for the compute device: "xpu" or "cuda".

    Use this instead of a hardcoded "cuda". PyTorch-ROCm aliases "cuda" to HIP,
    so ROCm keeps working with "cuda" -- which is exactly why hardcoded "cuda"
    strings went unnoticed until a backend without that alias (Intel XPU)
    arrived.
    """
    return "xpu" if is_xpu() else "cuda"
