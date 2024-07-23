from enum import Enum
from typing import Optional, Union, Dict, Any, List, Set
import logging

# TODO(wangyin): export device types and initialization from cpp DeviceFactory.

class DeviceType(Enum):
    CPU = "cpu"
    CUDA_GPU = "cuda"
    ROCM_GPU = "rocm"

class MemInfo:
    used: int = 0
    free: int = 0

    def __init__(self, used: int, free: int):
        self.used = used
        self.free = free

class DeviceBase:
    def __init__(self, device_type: DeviceType):
        self.device_type = device_type

    def get_device_type(self) -> DeviceType:
        return self.device_type

    def get_device_id(self) -> int:
        return 0

    def get_mem_info(self) -> MemInfo:
        raise NotImplementedError("get_mem_info is not implemented")
