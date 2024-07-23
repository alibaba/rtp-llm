import logging
from typing import Optional
from maga_transformer.device.device_base import DeviceType, DeviceBase
from maga_transformer.device.device_impl import CpuImpl, CudaImpl, RocmImpl

_current_device: Optional[DeviceBase] = None


def get_current_device() -> DeviceBase:
    global _current_device

    if _current_device != None:
        return _current_device

    try:
        _current_device = CudaImpl()
        return _current_device
    except:
        pass

    try:
        _current_device = RocmImpl()
        return _current_device
    except:
        pass

    _current_device = CpuImpl()
    return _current_device
