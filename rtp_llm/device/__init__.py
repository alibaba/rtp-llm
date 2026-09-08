from typing import TYPE_CHECKING, Optional, Type

from rtp_llm.device.device_type import DeviceType, get_device_type

if TYPE_CHECKING:
    from rtp_llm.device.device_base import DeviceBase

_current_device: Optional["DeviceBase"] = None
_current_device_type: Optional[DeviceType] = None


def __getattr__(name):
    # Preserve public imports without loading every backend with DeviceType.
    if name == "DeviceBase":
        from rtp_llm.device.device_base import DeviceBase

        return DeviceBase
    if name in {"ArmCpuImpl", "CpuImpl", "CudaImpl", "PpuImpl", "RocmImpl"}:
        from rtp_llm.device import device_impl

        return getattr(device_impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_device_cls(type: DeviceType) -> Type:
    from rtp_llm.device.device_impl import (
        ArmCpuImpl,
        CpuImpl,
        CudaImpl,
        PpuImpl,
        RocmImpl,
    )

    if type == DeviceType.Cpu:
        return CpuImpl
    elif type == DeviceType.ArmCpu:
        return ArmCpuImpl
    elif type == DeviceType.Cuda:
        return CudaImpl
    elif type == DeviceType.Ppu:
        return PpuImpl
    elif type == DeviceType.ROCm:
        return RocmImpl
    else:
        raise ValueError(f"Invalid device type {type}")


def get_current_device() -> "DeviceBase":
    global _current_device, _current_device_type

    if _current_device != None:
        return _current_device

    device_type = get_device_type()
    device_cls = get_device_cls(device_type)

    _current_device = device_cls()
    if not _current_device:
        raise ValueError(f"Failed to create device of type {device_type}")
    _current_device_type = device_type

    return _current_device


def get_cached_device_type() -> Optional[DeviceType]:
    """Read the cached object's construction identity without creating a device."""
    if _current_device is None:
        return None
    if _current_device_type is None:
        raise RuntimeError("Cached device has no recorded construction identity")
    return _current_device_type
