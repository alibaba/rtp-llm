import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from typing import Any, List

from filelock import FileLock, Timeout

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

python_code = """
import torch
import json
import sys

try:
    if torch.cuda.is_available():
        device_info = {torch.cuda.get_device_name(0): torch.cuda.device_count()}
    else:
        device_info = {}
    print(json.dumps(device_info))
except Exception as e:
    # 捕获并打印内部错误，避免子进程静默失败
    print(json.dumps({"error": str(e), "note": "Failed to get CUDA info"}), file=sys.stderr)
    sys.exit(1) # 告知外部进程执行失败
"""


def get_cuda_info():
    result = subprocess.run(
        [sys.executable, "-c", python_code], capture_output=True, text=True, check=False
    )
    logging.info(f"cuda info result: {result.stdout}, stderr: {result.stderr} return code: {result.returncode}")
    if result.returncode != 0:
        raise Exception(f"get cuda info returncode error, self ip: {get_ip()}")
    cuda_info = json.loads(result.stdout)
    if not cuda_info:
        return None
    name = list(cuda_info.keys())[0]
    count = cuda_info[name]
    return name, count


def get_ip():
    return socket.gethostbyname(socket.gethostname())


def get_gpu_ids():
    cuda_info = get_cuda_info()
    logging.info(f"{cuda_info}")

    if not cuda_info:
        return list(range(128))
    device_name = cuda_info[0]
    total_gpus = range(cuda_info[1])

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))
    if gpus is not None:
        total_gpus = [int(id) for id in gpus.split(",")]
    logging.info(f"{get_ip()} {device_name}: {total_gpus}")
    return total_gpus


class DeviceResource:
    def __init__(self, required_gpu_count: int):
        self.required_gpu_count = required_gpu_count
        self.total_gpus = get_gpu_ids()
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
                logging.info(f"{get_ip()} lock device {id} done")
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


def main(argv: List[str]):
    cuda_info = get_cuda_info()
    if not cuda_info:
        logging.info("no gpu, continue")
        result = subprocess.run(argv)
        logging.info("exitcode: %d", result.returncode)

        sys.exit(result.returncode)
    else:
        from .jit_sys_path_setup import setup_jit_cache

        setup_jit_cache()

        device_name, _ = cuda_info
        require_count = int(
            os.environ.get("WORLD_SIZE", os.environ.get("GPU_COUNT", "1"))
        )
        with DeviceResource(require_count) as gpu_resource:
            if "308" in device_name:
                env_name = "HIP_VISIBLE_DEVICES"
            else:
                env_name = "CUDA_VISIBLE_DEVICES"
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            result = subprocess.run(argv)
            logging.info("exitcode: %d", result.returncode)
            sys.exit(result.returncode)

if __name__ == "__main__":
    main(sys.argv[1:])