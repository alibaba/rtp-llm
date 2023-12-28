import logging
from pynvml import *
from typing import NamedTuple, Optional, List, Dict, Any


class GpuInfo(NamedTuple):
    tag: Dict[str, Any]
    util_nvml: float
    memory_used: int
    # pcie_tx_bytes: int
    # pcie_rx_byets: int
    # decoder_util: int

class GpuUtil(object):
    def __init__(self, *args: Any):
        super(GpuUtil, self).__init__(*args)
        self.has_gpu = self.check_gpu_available()
        if self.has_gpu:
            self.device_count = nvmlDeviceGetCount()
            self.device_names = [
                self.get_device_name(i) for i in range(self.device_count)
            ]
            self.nvml_handles = [
                nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)
            ]
            self.gpu_tags = [
                {'gpu_device': f'{self.device_names[i]}__{i}'} for i in range(self.device_count)
            ]

            logging.info(f'detected [{self.device_count}] gpus')
        else:
            logging.info('no GPU detected on this machine.')

    def get_device_name(self, device_id: int) -> str:
        name = nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(device_id))
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        name = name.replace(' ', '_')
        return name

    def get_gpu_info(self) -> Optional[List[GpuInfo]]:
        if not self.has_gpu:
            return None
        try:
            gpu_info_list = []
            for i in range(self.device_count):
                nvml_util_rate = nvmlDeviceGetUtilizationRates(self.nvml_handles[i])
                nvml_mem_info = nvmlDeviceGetMemoryInfo(self.nvml_handles[i])
                # pcie_tx_bytes = nvmlDeviceGetPcieThroughput(self.nvml_handles[i], NVML_PCIE_UTIL_TX_BYTES)
                # pcie_rx_bytes = nvmlDeviceGetPcieThroughput(self.nvml_handles[i], NVML_PCIE_UTIL_RX_BYTES)
                # decoder_util = nvmlDeviceGetDecoderUtilization(self.nvml_handles[i])[0]
                gpu_info_list.append(
                    GpuInfo(self.gpu_tags[i], nvml_util_rate.gpu, nvml_mem_info.used,
                            # pcie_tx_bytes, pcie_rx_bytes, decoder_util
                    )
                )
            return gpu_info_list
        except Exception as e:
            logging.warn('failed to get nvml info: [%s]', e)
            return None

    def check_gpu_available(self) -> bool:
        try:
            nvmlInit()
            return (nvmlDeviceGetCount() > 0)
        except Exception:
            return False
