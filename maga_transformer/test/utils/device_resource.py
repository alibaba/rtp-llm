import os
import time
import torch
import logging
import traceback
from filelock import FileLock, Timeout
from typing import Tuple, Any, List
from contextlib import ExitStack
import platform

class DeviceResource:
    def __init__(self, required_gpu_count: int):
        self.required_gpu_count = required_gpu_count

        if platform.processor() != 'aarch64':
            self.total_gpus = list(range(torch.cuda.device_count()))
            gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
            if gpus is not None:
                self.total_gpus = [int(id) for id in gpus.split(',')]
        else:
            # TODO: for arm cpu device, how to simulate gpu ?
            self.total_gpus = [0, 1]
        logging.info(f"total gpu: {self.total_gpus}")
        self.gpu_ids = []
        self.gpu_locks = ExitStack()
        self.global_lock_file = "/tmp/maga_transformer/smoke/test/gpu_status_lock"
        self.gpu_status_root_path = "/tmp/maga_transformer/smoke/test/gpu_status"

    def _lock_gpus(self):
        with ExitStack() as stack:
            gpu_ids = []
            for id in self.total_gpus:
                lock_device = FileLock(f"{self.gpu_status_root_path}/{id}")
                try:
                    stack.enter_context(lock_device.acquire(timeout=1))
                except Timeout as _:
                    logging.info(f"lock device {id} failed")
                    continue
                gpu_ids.append(str(id))
                logging.info(f"lock device {id} done")
                if len(gpu_ids) >= self.required_gpu_count:
                    logging.info(f"use gpus:[{gpu_ids}]")
                    self.gpu_locks = stack.pop_all()
                    self.gpu_ids = gpu_ids
                    return True
        return False

    def __enter__(self):
        logging.info(f"waiting for gpu count:[{self.required_gpu_count}]")
        while True:
            with FileLock(self.global_lock_file):
                try:
                    if self._lock_gpus():
                        return self
                except Exception as e:
                    logging.warn(f"{traceback.format_exc()}")
            time.sleep(1)

    def __exit__(self, *args: Any):
        with FileLock(self.global_lock_file):
            logging.info(f"release gpu:{self.gpu_ids}")
            self.gpu_ids = []
            self.gpu_locks.close()
            logging.info("release done")
