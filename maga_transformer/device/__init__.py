import logging
from typing import Optional, Type
from maga_transformer.device.device_base import DeviceType, DeviceBase
from maga_transformer.device.device_impl import CpuImpl, ArmCpuImpl, CudaImpl, PpuImpl, RocmImpl
from maga_transformer.ops import get_device, DeviceType, DeviceExporter

_current_device: Optional[DeviceBase] = None

def get_device_cls(type: DeviceType) -> Type:
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

def get_current_device() -> DeviceBase:
    global _current_device

    if _current_device != None:
        return _current_device

    exported_device: DeviceExporter = get_device()
    device_type = exported_device.get_device_type()
    device_cls = get_device_cls(device_type)

    _current_device = device_cls(exported_device)
    if (not _current_device):
        raise ValueError(f"Failed to create device of type {device_type}")

    return _current_device
