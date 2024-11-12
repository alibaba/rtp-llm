from enum import Enum
from typing import Optional, Union, Dict, Any, List, Set
import logging

from maga_transformer.ops import DeviceType, DeviceExporter


class MemInfo:
    used: int = 0
    free: int = 0

    def __init__(self, used: int, free: int):
        self.used = used
        self.free = free

class DeviceBase:
    def __init__(self, exported_device: DeviceExporter):
        self.exported_device = exported_device

    def get_device_type(self) -> DeviceType:
        return self.exported_device.get_device_type()

    def get_device_id(self) -> int:
        return self.exported_device.get_device_id()

    def get_mem_info(self) -> MemInfo:
        raise NotImplementedError("get_mem_info is not implemented")
