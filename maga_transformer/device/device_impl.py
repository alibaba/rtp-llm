from maga_transformer.device.device_base import DeviceBase, DeviceType, MemInfo

import torch
import psutil

class CpuImpl(DeviceBase):
    def __init__(self):
        super(CpuImpl, self).__init__(DeviceType.CPU)

    def get_mem_info(self) -> MemInfo:
        vmem = psutil.virtual_memory()
        return MemInfo(vmem.used, vmem.free)

class CudaImpl(DeviceBase):
    def __init__(self):
        super(CudaImpl, self).__init__(DeviceType.CUDA_GPU)
        import pynvml
        pynvml.nvmlInit()

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

    def get_mem_info(self) -> MemInfo:
        import pynvml
        # TODO(wangyin): change this to current id when old async model is dropped.
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda._parse_visible_devices()[0])
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return MemInfo(meminfo.used, meminfo.free)

class RocmImpl(DeviceBase):
    def __init__(self):
        super(RocmImpl, self).__init__(DeviceType.ROCM_GPU)
        from pyrsmi import rocml
        rocml.smi_initialize()

    def get_device_id(self) -> int:
        return torch.cuda.current_device()

    def get_mem_info(self) -> MemInfo:
        from pyrsmi import rocml
        id = self.get_device_id()
        used = rocml.smi_get_device_memory_used(id)
        total = rocml.smi_get_device_memory_total(id)
        return MemInfo(total - used, used)
