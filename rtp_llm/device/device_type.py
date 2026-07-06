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


_DEVICE_TYPE_OVERRIDE = {
    "cpu": DeviceType.Cpu,
    "cuda": DeviceType.Cuda,
    "yitian": DeviceType.Yitian,
    "armcpu": DeviceType.ArmCpu,
    "rocm": DeviceType.ROCm,
    "ppu": DeviceType.Ppu,
    "xpu": DeviceType.Xpu,
}


_DEVICE_TYPE_CACHE = {}


def get_device_type() -> DeviceType:
    # Cache the resolved type per RTP_LLM_DEVICE_TYPE value so the torch
    # xpu/cuda probing and the mixed-device warning run once per process
    # (RTP_LLM_DEVICE_TYPE only needs to be set before startup).  Keying on the
    # override value still lets a process that changes the env re-resolve.
    override = os.environ.get("RTP_LLM_DEVICE_TYPE", "").strip().lower()
    cached = _DEVICE_TYPE_CACHE.get(override)
    if cached is not None:
        return cached
    resolved = _resolve_device_type(override)
    _DEVICE_TYPE_CACHE[override] = resolved
    return resolved


def _resolve_device_type(override: str) -> DeviceType:
    # Explicit override wins so a mixed XPU+CUDA host is never silently
    # resolved by detection-order alone. Set RTP_LLM_DEVICE_TYPE=cuda|xpu|...
    if override:
        if override in _DEVICE_TYPE_OVERRIDE:
            return _DEVICE_TYPE_OVERRIDE[override]
        raise ValueError(
            f"Unknown RTP_LLM_DEVICE_TYPE={override!r}; valid values: "
            f"{sorted(_DEVICE_TYPE_OVERRIDE.keys())}")

    xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
    cuda_available = torch.cuda.is_available()
    if xpu_available and cuda_available:
        # Both backends visible and no explicit override: prefer CUDA so an
        # existing CUDA deployment that merely has XPU libraries present is not
        # silently switched to XPU. Set RTP_LLM_DEVICE_TYPE=xpu to force XPU.
        import logging
        logging.getLogger(__name__).warning(
            "Both XPU and CUDA are available; selecting CUDA by default. "
            "Set RTP_LLM_DEVICE_TYPE=xpu to force XPU.")
    if cuda_available:
        if hasattr(torch.version, "hip") and torch.version.hip is not None:
            return DeviceType.ROCm
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return DeviceType.Ppu
        return DeviceType.Cuda
    if xpu_available:
        return DeviceType.Xpu
    return DeviceType.Cpu


def is_cuda() -> bool:
    return get_device_type() == DeviceType.Cuda


def is_hip() -> bool:
    return get_device_type() == DeviceType.ROCm


def is_xpu() -> bool:
    return get_device_type() == DeviceType.Xpu
