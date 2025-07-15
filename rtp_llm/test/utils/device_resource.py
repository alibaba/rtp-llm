import logging
import os
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from typing import Any, List, Tuple

import torch
from filelock import FileLock, Timeout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DeviceResource:
    def __init__(self, required_gpu_count: int):
        self.required_gpu_count = required_gpu_count
        if not torch.cuda.is_available():
            self.total_gpus = list(range(128))
        else:
            self.total_gpus = list(range(torch.cuda.device_count()))
            gpus = os.environ.get(
                "CUDA_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES")
            )
            if gpus is not None:
                self.total_gpus = [int(id) for id in gpus.split(",")]
                logging.info(f"{torch.cuda.get_device_name()}: {self.total_gpus}")
        if required_gpu_count > len(self.total_gpus):
            raise ValueError(
                f"required gpu count {required_gpu_count} is greater than total gpu count {len(self.total_gpus)}"
            )
        self.gpu_ids: List[int] = []
        self.gpu_locks = ExitStack()
        self.global_lock_file = "/tmp/rtp_llm/smoke/test/gpu_status_lock"
        self.gpu_status_root_path = "/tmp/rtp_llm/smoke/test/gpu_status"

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


if __name__ == "__main__":
    if not torch.cuda.is_available():
        logging.info("no gpu, continue")
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)
    else:
        require_count = int(
            os.environ.get("WORLD_SIZE", os.environ.get("GPU_COUNT", "1"))
        )
        with DeviceResource(require_count) as gpu_resource:
            if "308" in torch.cuda.get_device_name():
                env_name = "HIP_VISIBLE_DEVICES"
            else:
                env_name = "CUDA_VISIBLE_DEVICES"
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            result = subprocess.run(sys.argv[1:])
            sys.exit(result.returncode)
