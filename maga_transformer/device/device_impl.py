from maga_transformer.device.device_base import DeviceBase, DeviceType, MemInfo
from maga_transformer.ops import DeviceType, DeviceExporter

import torch
import psutil

class CpuImpl(DeviceBase):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

    def get_mem_info(self) -> MemInfo:
        vmem = psutil.virtual_memory()
        return MemInfo(vmem.used, vmem.free)

class ArmCpuImpl(CpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

class GpuImpl(DeviceBase):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

class CudaImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        import pynvml
        pynvml.nvmlInit()

    def get_mem_info(self) -> MemInfo:
        import pynvml
        # TODO(wangyin): change this to current id when old async model is dropped.
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda._parse_visible_devices()[0])
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return MemInfo(meminfo.used, meminfo.free)

class RocmImpl(GpuImpl):
    def __init__(self, exported_device: DeviceExporter):
        super().__init__(exported_device)
        from pyrsmi import rocml
        rocml.smi_initialize()

    def get_mem_info(self) -> MemInfo:
        from pyrsmi import rocml
        id = self.get_device_id()
        used = rocml.smi_get_device_memory_used(id)
        total = rocml.smi_get_device_memory_total(id)
        return MemInfo(total - used, used)
