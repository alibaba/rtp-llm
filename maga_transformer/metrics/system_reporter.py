import os
import time
import psutil
import logging
import traceback
from time import sleep
from threading import Thread
from psutil._common import sdiskusage
from typing import Optional, List, Any

from maga_transformer.utils.gpu_util import GpuUtil, GpuInfo

UPDATE_INTERVAL_SECOND = 10

class SystemReporter(object):
    def __init__(self, kmon: Any):
        self._kmon = kmon
        self.report_thread: Optional[Thread] = None
        self.process = psutil.Process(os.getpid())

        self.cpu_percent_metric = self._kmon.register_gauge_metric('py_rtp_cpu_percent')

        self.physical_memory_metric = self._kmon.register_gauge_metric('py_rtp_phys_mem')
        self.virtual_memory_metric = self._kmon.register_gauge_metric('py_rtp_virt_mem')
        self.shared_memory_metric = self._kmon.register_gauge_metric('py_rtp_shared_mem')

        self.gpu_util = GpuUtil()
        if self.gpu_util.has_gpu:
            self.gpu_util_metric = self._kmon.register_gauge_metric('py_rtp_gpu_util_nvml')
            self.gpu_memory_used_metric = self._kmon.register_gauge_metric('py_rtp_gpu_mem_used')
            # self.gpu_pcie_tx_metric = kmon_metrics.register_gauge_metric('py_rtp_gpu_tx', gpu_tags)
            # self.gpu_pcie_rx_metric = kmon_metrics.register_gauge_metric('py_rtp_gpu_rx', gpu_tags)
            # self.gpu_decoder_util_metric = kmon_metrics.register_gauge_metric('py_rtp_gpu_decoder_util', gpu_tags)

        self.thread_num_metric = self._kmon.register_gauge_metric('py_rtp_thread_num')

        self.used_disk_metric = self._kmon.register_gauge_metric('py_rtp_used_disk')
        self.disk_usage_metric = self._kmon.register_gauge_metric('py_rtp_disk_usage')

        self.last_report_time: float = time.time()

    def update_system_metrics(self) -> None:
        self.cpu_percent_metric.report(self.process.cpu_percent())

        memory_info = self.process.memory_info()
        self.physical_memory_metric.report(memory_info.rss)
        self.virtual_memory_metric.report(memory_info.vms)
        self.shared_memory_metric.report(memory_info.shared)
        self.thread_num_metric.report(self.process.num_threads())

        gpu_info_list: Optional[List[GpuInfo]] = self.gpu_util.get_gpu_info()
        if gpu_info_list != None:
            for gpu_info in gpu_info_list:
                self.gpu_util_metric.report(gpu_info.util_nvml, gpu_info.tag)
                self.gpu_memory_used_metric.report(gpu_info.memory_used, gpu_info.tag)
                # self.gpu_pcie_tx_metric.report(gpu_info.pcie_tx_bytes)
                # self.gpu_pcie_rx_metric.report(gpu_info.pcie_rx_byets)
                # self.gpu_decoder_util_metric.report(gpu_info.decoder_util)

        disk_usage: sdiskusage = psutil.disk_usage('/')
        self.used_disk_metric.report(disk_usage.used)
        self.disk_usage_metric.report(disk_usage.percent)

        current_time = time.time()
        self.last_report_time = current_time

    def update_forever(self) -> None:
        while True:
            try:
                self.update_system_metrics()
            except BaseException as e:
                logging.error(f'update system metrics failed: {e}, {traceback.format_exc()}')
            sleep(UPDATE_INTERVAL_SECOND)

    def start(self) -> None:
        if self.report_thread == None:
            self.report_thread = Thread(target=self.update_forever, args=())
            self.report_thread.daemon = True
            self.report_thread.start()
        else:
            logging.warn('system metric reporter process already started.')

