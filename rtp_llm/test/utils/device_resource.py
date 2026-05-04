import json
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from contextlib import ExitStack
from typing import Any, Dict, List, Optional

from filelock import FileLock, Timeout

GPU_LOCK_TIMEOUT_ENV = "RTP_GPU_LOCK_TIMEOUT"
GPU_LOCK_DEFAULT_TIMEOUT = 120
GPU_STATUS_ROOT = "/tmp/rtp_llm/smoke/test/gpu_status"
GPU_GLOBAL_LOCK_FILE = "/tmp/rtp_llm/smoke/test/gpu_status_lock"


class GpuLockError(RuntimeError):
    """Raised when GPU lock acquisition fails (timeout or insufficient GPUs)."""

    pass


class GpuLockTimeoutError(GpuLockError):
    """Raised when GPU lock acquisition times out (GPUs are busy)."""

    pass


class GpuInsufficientError(GpuLockError):
    """Raised when the visible GPU count cannot satisfy the request at all."""

    pass


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
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
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
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
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
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
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
        raise RuntimeError(
            "get_gpu_ids(): no GPU/accelerator detected — "
            "nvidia-smi / rocm-smi not found or returned no devices. "
            "Cannot allocate GPUs."
        )
    device_name = device_info[0]
    total_gpus = range(device_info[1])

    gpus = os.environ.get("CUDA_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))
    if gpus is not None:
        total_gpus = [int(id) for id in gpus.split(",")]
    logging.info(f"{get_ip()} {device_name}: {total_gpus}")
    return total_gpus


class DeviceResource:
    def __init__(self, required_gpu_count: int, timeout: Optional[int] = None):
        """
        Args:
            timeout: seconds to wait for GPU locks.
                     None (default) = wait forever (for Bazel / standalone usage).
                     Pytest fixtures should pass an explicit timeout.
        """
        self.required_gpu_count = required_gpu_count
        self.total_gpus = get_gpu_ids()
        if required_gpu_count > len(self.total_gpus):
            raise GpuInsufficientError(
                f"Need {required_gpu_count} GPUs but only {len(self.total_gpus)} visible "
                f"(CUDA/HIP_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', os.environ.get('HIP_VISIBLE_DEVICES', 'unset'))})"
            )
        os.makedirs(GPU_STATUS_ROOT, exist_ok=True)
        env_timeout = os.environ.get(GPU_LOCK_TIMEOUT_ENV)
        if timeout is not None:
            self.timeout = timeout
        elif env_timeout is not None:
            self.timeout = int(env_timeout)
        else:
            self.timeout = None  # wait forever (Bazel / standalone)
        self.gpu_ids: List[str] = []
        self.gpu_locks = ExitStack()
        self.global_lock_file = GPU_GLOBAL_LOCK_FILE
        self.gpu_status_root_path = GPU_STATUS_ROOT
        self._gpu_bad_until: Dict[str, float] = {}
        self.bad_gpu_cooldown_s = 30
        self._lock_start_idx = 0

    def _get_gpu_pids(self, gpu_id: str) -> List[int]:
        """Return PIDs of compute processes on a physical GPU.

        nvidia-smi provides per-GPU PID queries; for ROCm/PPU we degrade
        gracefully to empty (file-based locking still provides isolation).
        """
        for smi in _NVIDIA_SMI_PATHS:
            try:
                result = subprocess.run(
                    [
                        smi,
                        "--query-compute-apps=pid",
                        "--format=csv,noheader",
                        f"--id={gpu_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return [
                        int(p.strip())
                        for p in result.stdout.strip().splitlines()
                        if p.strip() and p.strip().isdigit()
                    ]
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

    def _check_gpu_usable(self) -> bool:
        """Return False if any locked GPU has zombie CUDA contexts. Never kills processes.

        NOTE on Docker PID namespaces: nvidia-smi reports host-namespace PIDs,
        while /proc/<pid> and os.kill() operate in the container namespace.
        This means _pid_alive() may return False for a host PID that is actually
        alive (just invisible from inside the container), or True for a
        completely unrelated container-local process.  We only use this check
        to detect the worst case — zombie GPU contexts where memory is
        permanently leaked.  False positives (skipping a usable GPU) are
        harmless; false negatives are caught by file locks.
        """
        for gpu_id in self.gpu_ids:
            if self._has_zombie_gpu_contexts(str(gpu_id)):
                logging.warning(
                    "GPU %s has zombie contexts (dead PIDs still hold memory), skipping",
                    gpu_id,
                )
                return False
        return True

    def _iter_gpu_ids(self):
        """Yield GPU IDs starting from _lock_start_idx, wrapping around.

        This distributes lock attempts across GPUs so multiple concurrent
        DeviceResource instances don't all contend for GPU 0 first.
        """
        n = len(self.total_gpus)
        for i in range(n):
            yield self.total_gpus[(self._lock_start_idx + i) % n]
        self._lock_start_idx = (self._lock_start_idx + 1) % n

    def _lock_gpus(self):
        with ExitStack() as stack:
            gpu_ids = []
            now = time.time()
            for id in self._iter_gpu_ids():
                if self._gpu_bad_until.get(str(id), 0) > now:
                    continue
                lock_device = FileLock(f"{self.gpu_status_root_path}/{id}")
                try:
                    stack.enter_context(lock_device.acquire(timeout=0))
                except Timeout:
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
        timeout_desc = f"{self.timeout}s" if self.timeout is not None else "infinite"
        logging.info(
            f"waiting for gpu count:[{self.required_gpu_count}] timeout={timeout_desc}"
        )
        deadline = (time.time() + self.timeout) if self.timeout is not None else None
        while True:
            locked = False
            with FileLock(self.global_lock_file):
                try:
                    locked = self._lock_gpus()
                except (GpuLockError, GpuInsufficientError):
                    raise
                except Exception:
                    logging.warning(f"{traceback.format_exc()}")
            if locked:
                if self._check_gpu_usable():
                    break
                logging.warning("GPUs %s have zombie contexts, retrying", self.gpu_ids)
                for gid in self.gpu_ids:
                    self._gpu_bad_until[gid] = time.time() + self.bad_gpu_cooldown_s
                self.gpu_ids = []
                self.gpu_locks.close()
            if deadline is not None and time.time() >= deadline:
                raise GpuLockTimeoutError(
                    f"GPU lock timed out after {self.timeout}s: "
                    f"need {self.required_gpu_count} GPUs from {self.total_gpus}"
                )
            time.sleep(1)
        return self

    def __exit__(self, *args: Any):
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
    # Only treat the GPU requirement as hard when the caller explicitly asks
    # for a non-zero count via GPU_COUNT/WORLD_SIZE. Without an explicit ask
    # we default to "no GPU needed" — this lets CPU-only cc_tests (utils,
    # api_server http_client, etc.) that bazel dispatches to the cpu sub-pool
    # of the cuda12_9 worker pool succeed under `--run_under=gpu_lock`.
    # Tests that genuinely need a GPU either declare it via env / args or
    # fail clearly inside the binary when CUDA is missing.
    gpu_count_env = os.environ.get("GPU_COUNT", os.environ.get("WORLD_SIZE"))
    require_count = int(gpu_count_env) if gpu_count_env is not None else 0

    if not device_info:
        if require_count > 0:
            raise RuntimeError(
                f"[device_resource] GPU_COUNT={require_count} requested but no GPU detected "
                f"(nvidia-smi / rocm-smi not found or returned no devices)"
            )
        logging.warning(
            "[device_resource] no GPU detected, running without GPU isolation"
        )
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)

    if require_count == 0:
        # Have GPU(s) but caller didn't ask for any — skip locking entirely.
        logging.info("[device_resource] GPU present but GPU_COUNT not set; skipping lock")
        result = subprocess.run(sys.argv[1:])
        sys.exit(result.returncode)

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)

    with DeviceResource(require_count) as gpu_resource:
        os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
        sys.stderr.write(
            f"[device_resource] {env_name}={os.environ[env_name]} "
            f"locked={gpu_resource.gpu_ids} pid={os.getpid()}\n"
        )
        sys.stderr.flush()
        result = subprocess.run(sys.argv[1:])
        logging.info("exitcode: %d", result.returncode)
        sys.exit(result.returncode)
