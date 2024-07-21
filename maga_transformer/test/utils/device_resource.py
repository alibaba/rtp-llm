import os
import time
import torch
import logging
from filelock import FileLock, Timeout
from typing import Tuple, Any, List


class DeviceResource:
    def __init__(self, required_gpu_count: int):
        self.required_gpu_count = required_gpu_count
        self.total_gpus = list(range(torch.cuda.device_count()))
        gpus = os.environ.get('CUDA_VISIBLE_DEVICES')
        if gpus is not None:
            self.total_gpus = [int(id) for id in gpus.split(',')]
        logging.info(f"total gpu: {self.total_gpus}")
        self.gpu_id_with_lock: List[Tuple[int, Any]] = []
        self.global_lock_file = "/tmp/maga_transformer/smoke/test/gpu_status_lock"
        self.gpu_status_root_path = "/tmp/maga_transformer/smoke/test/gpu_status"

    @property
    def gpu_ids(self) -> List[str]:
        return [str(_[0]) for _ in self.gpu_id_with_lock]

    def _lock_gpus(self):
        for id in self.total_gpus:
            lock_device = FileLock(f"{self.gpu_status_root_path}/{id}")
            try:
                lock_device.acquire(timeout=0.1)
            except Timeout as _:
                logging.info(f"lock device {id} failed")
                continue
            logging.info(f"lock device {id} done")
            self.gpu_id_with_lock.append((id, lock_device))
            if len(self.gpu_id_with_lock) >= self.required_gpu_count:
                logging.info(f"use gpus:[{self.gpu_ids}]")
                return True
        return False

    def __enter__(self):
        logging.info(f"waiting for gpu count:[{self.required_gpu_count}]")
        lock = FileLock(self.global_lock_file)
        while True:
            with lock:
                try:
                    if self._lock_gpus():
                        return self
                except:
                    pass
                [_[1].release() for _ in self.gpu_id_with_lock]
                self.gpu_id_with_lock = []
            time.sleep(1)

    def __exit__(self, *args: Any):
        lock = FileLock(self.global_lock_file)
        with lock:
            [_[1].release() for _ in self.gpu_id_with_lock]
            logging.info(f"release gpu:{self.gpu_ids}")
