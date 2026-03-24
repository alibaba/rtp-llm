import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from typing import Any, List, Set

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
    logging.info(f"cuda info result: {result.stdout}, return code: {result.returncode}")

    try:
        if result.returncode != 0:
            raise Exception(f"get cuda info returncode error, self ip: {get_ip()}")
        cuda_info = json.loads(result.stdout)
        if not cuda_info:
            return None
    except Exception as e:
        # fallback to get device count when subprocess execute failed
        import torch

        if torch.cuda.is_available():
            device_info = {torch.cuda.get_device_name(0): torch.cuda.device_count()}
        else:
            device_info = {}
        cuda_info = device_info

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

    def _get_gpu_pids(self, gpu_id: str) -> List[int]:
        """Return PIDs of compute processes on a physical GPU via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader", f"--id={gpu_id}"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return [int(p.strip()) for p in result.stdout.strip().splitlines()
                        if p.strip()]
        except Exception:
            pass
        return []

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check if a process is alive by probing /proc/<pid>."""
        return os.path.exists(f"/proc/{pid}")

    def _has_zombie_gpu_contexts(self, gpu_id: str) -> bool:
        """Check if a GPU has zombie CUDA contexts (dead processes still holding memory).

        When CUDA processes are SIGKILLed, the driver may fail to reclaim
        GPU memory, leaving permanent zombie contexts. These GPUs are unusable
        until a GPU reset or reboot.
        """
        pids = self._get_gpu_pids(gpu_id)
        if not pids:
            return False
        return all(not self._pid_alive(p) for p in pids)

    def _ensure_gpus_released(self, timeout: int = 30):
        """Wait until acquired GPUs have no stale compute processes.

        Uses SIGTERM first to allow graceful CUDA cleanup, then SIGKILL
        as a last resort. Detects zombie GPU contexts (dead processes that
        still hold GPU memory) which indicate unrecoverable state.

        Returns True if GPUs are clean, False if zombie contexts detected.
        """
        my_pid = os.getpid()
        sigterm_sent: Set[int] = set()
        sigkill_sent: Set[int] = set()
        deadline = time.time() + timeout

        while time.time() < deadline:
            all_clear = True
            for gpu_id in self.gpu_ids:
                stale = [p for p in self._get_gpu_pids(gpu_id) if p != my_pid]
                live_stale = [p for p in stale if self._pid_alive(p)]

                if not stale:
                    continue

                if not live_stale:
                    # All nvidia-smi PIDs are dead → zombie GPU contexts
                    logging.warning(
                        f"GPU {gpu_id} has zombie CUDA contexts (dead PIDs: {stale}). "
                        f"Memory is permanently leaked until GPU reset."
                    )
                    return False

                all_clear = False
                for pid in live_stale:
                    if pid not in sigterm_sent:
                        try:
                            os.kill(pid, signal.SIGTERM)
                            logging.info(f"SIGTERM pid {pid} on GPU {gpu_id}")
                            sigterm_sent.add(pid)
                        except (ProcessLookupError, PermissionError):
                            pass
                    elif pid not in sigkill_sent:
                        try:
                            os.kill(pid, signal.SIGKILL)
                            logging.info(f"SIGKILL pid {pid} on GPU {gpu_id}")
                            sigkill_sent.add(pid)
                        except (ProcessLookupError, PermissionError):
                            pass
                break

            if all_clear:
                return True
            time.sleep(1)

        logging.warning(f"GPU cleanup timed out after {timeout}s for GPUs {self.gpu_ids}")
        return False

    def _lock_gpus(self):
        with ExitStack() as stack:
            gpu_ids = []
            for id in self.total_gpus:
                if self._has_zombie_gpu_contexts(str(id)):
                    logging.info(f"skip GPU {id}: zombie CUDA contexts detected")
                    continue
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
                        gpus_clean = self._ensure_gpus_released()
                        if gpus_clean:
                            break
                        # Zombie contexts found — release these GPUs and retry
                        logging.warning(f"GPUs {self.gpu_ids} have zombie contexts, retrying")
                        self.gpu_ids = []
                        self.gpu_locks.close()
                except Exception as e:
                    logging.warn(f"{traceback.format_exc()}")
            time.sleep(1)
        return self

    def __exit__(self, *args: Any):
        self._ensure_gpus_released()
        with FileLock(self.global_lock_file):
            logging.info(f"release gpu:{self.gpu_ids}")
            self.gpu_ids = []
            self.gpu_locks.close()
            logging.info("release done")


if __name__ == "__main__":
    cuda_info = get_cuda_info()
    if not cuda_info:
        logging.info("no gpu, continue")
        result = subprocess.run(sys.argv[1:])
        logging.info("exitcode: %d", result.returncode)

        sys.exit(result.returncode)
    else:
        from jit_sys_path_setup import setup_jit_cache

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
            result = subprocess.run(sys.argv[1:])
            logging.info("exitcode: %d", result.returncode)
            sys.exit(result.returncode)
