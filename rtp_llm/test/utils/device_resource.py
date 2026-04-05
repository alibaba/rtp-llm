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

_NVIDIA_SMI_PATHS = [
    "nvidia-smi",
    "/usr/local/cuda/bin/nvidia-smi",
    "/usr/local/nvidia/bin/nvidia-smi",
    "/usr/bin/nvidia-smi",
]


def _detect_nvidia():
    """Detect NVIDIA GPUs via nvidia-smi (driver-level, no torch needed).

    Tries multiple known paths because Bazel sandbox / PPU workers may not
    have nvidia-smi on PATH.  Rejects output containing "error" to avoid
    false positives from PPU's wrapper when the driver is absent.
    """
    for smi in _NVIDIA_SMI_PATHS:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=False,
            )
            if result.returncode != 0:
                continue
            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if not lines:
                continue
            if "error" in lines[0].lower():
                logging.info(f"nvidia-smi returned error string: {lines[0]}")
                continue
            return lines[0], len(lines)
        except FileNotFoundError:
            continue
    return None


def _detect_rocm():
    """Detect AMD GPUs via rocm-smi / rocminfo."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            gpu_names = []
            for line in result.stdout.strip().splitlines():
                if "GPU[" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 3 and parts[-1].strip():
                        gpu_names.append(parts[-1].strip())
            if gpu_names:
                return gpu_names[0], len(gpu_names)
    except FileNotFoundError:
        pass
    try:
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10, check=False,
        )
        if result.returncode == 0:
            names = [
                line.split(":")[-1].strip()
                for line in result.stdout.splitlines()
                if "Marketing Name" in line and line.split(":")[-1].strip() != "Host"
            ]
            if names:
                return names[0], len(names)
    except FileNotFoundError:
        pass
    return None


def get_device_info():
    """Detect GPU/accelerator name and count. NVIDIA/PPU via nvidia-smi, ROCm via rocm-smi.

    PPU exposes a CUDA-compatible driver so nvidia-smi works identically.
    No torch dependency — works in Bazel sandbox without pip packages.

    Returns: (device_name, device_count) or None if no device found.
    """
    for detector in [_detect_nvidia, _detect_rocm]:
        result = detector()
        if result:
            name, count = result
            logging.info(f"Detected device: {name} x{count} (via {detector.__name__})")
            return name, count
    logging.info("No GPU/accelerator detected")
    return None


get_cuda_info = get_device_info


def get_ip():
    return socket.gethostbyname(socket.gethostname())


def get_gpu_ids():
    device_info = get_device_info()
    logging.info(f"{device_info}")

    if not device_info:
        return list(range(128))
    device_name = device_info[0]
    total_gpus = range(device_info[1])

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
        """Return PIDs of compute processes on a physical GPU.

        nvidia-smi provides per-GPU PID queries; for ROCm/PPU we degrade
        gracefully to empty (file-based locking still provides isolation).
        """
        for smi in _NVIDIA_SMI_PATHS:
            try:
                result = subprocess.run(
                    [smi, "--query-compute-apps=pid",
                     "--format=csv,noheader", f"--id={gpu_id}"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return [int(p.strip()) for p in result.stdout.strip().splitlines()
                            if p.strip() and p.strip().isdigit()]
                if result.returncode == 0:
                    return []
            except FileNotFoundError:
                continue
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


def _get_visible_devices_env(device_name: str) -> str:
    """Return the env var name for device visibility control."""
    if "308" in device_name or "MI3" in device_name:
        return "HIP_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


if __name__ == "__main__":
    device_info = get_device_info()
    if not device_info:
        logging.info("no device detected, running without GPU isolation")
        result = subprocess.run(sys.argv[1:])
        logging.info("exitcode: %d", result.returncode)
        sys.exit(result.returncode)
    else:
        try:
            from jit_sys_path_setup import setup_jit_cache
            setup_jit_cache()
        except Exception as e:
            logging.warning(f"JIT cache setup skipped: {e}")

        device_name, _ = device_info
        require_count = int(
            os.environ.get("WORLD_SIZE", os.environ.get("GPU_COUNT", "1"))
        )
        with DeviceResource(require_count) as gpu_resource:
            env_name = _get_visible_devices_env(device_name)
            os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
            result = subprocess.run(sys.argv[1:])
            logging.info("exitcode: %d", result.returncode)
            sys.exit(result.returncode)
