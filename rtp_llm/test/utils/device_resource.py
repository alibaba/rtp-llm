import json
import logging
import os
import shutil
import signal
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
GPU_MAX_PREEXISTING_MEMORY_MB_ENV = "RTP_GPU_MAX_PREEXISTING_MEMORY_MB"
GPU_MAX_PREEXISTING_MEMORY_MB_DEFAULT = 1024
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


def _remote_session_id() -> str:
    return os.environ.get("RTP_REMOTE_SESSION_ID", "")


def _read_proc_environ(pid: int) -> bytes:
    try:
        with open(f"/proc/{pid}/environ", "rb") as f:
            return f.read()
    except OSError:
        return b""


def _pid_session_id(pid: int) -> Optional[str]:
    return _pid_env_value(pid, "RTP_REMOTE_SESSION_ID")


def _pid_env_value(pid: int, key: str) -> Optional[str]:
    prefix = f"{key}=".encode()
    for entry in _read_proc_environ(pid).split(b"\0"):
        if entry.startswith(prefix):
            return entry.split(b"=", 1)[1].decode("utf-8", errors="replace")
    return None


def _session_pids(session_id: str, owner_pid: Optional[int] = None) -> List[int]:
    if not session_id:
        return []
    pids: List[int] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        if _pid_session_id(pid) == session_id:
            if owner_pid is not None and _pid_env_value(
                pid, "RTP_DEVICE_RESOURCE_OWNER_PID"
            ) != str(owner_pid):
                continue
            pids.append(pid)
    return pids


def _kill_process_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _kill_pids(pids: List[int], sig: int) -> None:
    for pid in sorted(set(pids)):
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def _run_child(argv: List[str]) -> int:
    """Run the wrapped command.

    Remote sessions use Popen with inherited stdout/stderr instead of PIPE.
    That avoids deadlocking device_resource if a child emits more output than
    a pipe buffer can hold, while still letting REAPI capture the process logs.
    """
    if not _remote_session_id():
        return subprocess.run(argv).returncode

    owner_pid = os.getpid()
    env = dict(os.environ)
    env["RTP_DEVICE_RESOURCE_OWNER_PID"] = str(owner_pid)
    child = subprocess.Popen(argv, start_new_session=True, env=env)
    old_term = signal.getsignal(signal.SIGTERM)
    old_int = signal.getsignal(signal.SIGINT)

    def _handler(signum, frame):
        _kill_process_group(child.pid, signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    try:
        return child.wait()
    finally:
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)
        if child.poll() is None:
            _kill_process_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(child.pid, signal.SIGKILL)
        escaped = _session_pids(_remote_session_id(), owner_pid=owner_pid)
        if escaped:
            logging.warning(
                "[device_resource] cleaning escaped session pids: %s", escaped
            )
            _kill_pids(escaped, signal.SIGTERM)
            time.sleep(1)
            _kill_pids(
                _session_pids(_remote_session_id(), owner_pid=owner_pid),
                signal.SIGKILL,
            )


def _detect_nvidia():
    """Detect NVIDIA GPUs via nvidia-smi (driver-level, no torch needed).

    Tries multiple known paths because Bazel sandbox workers may not
    have nvidia-smi on PATH.  Rejects output containing "error" to avoid
    false positives from driver wrappers when the driver is absent.
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
    """Detect GPU/accelerator name and count. NVIDIA-compatible via nvidia-smi, ROCm via rocm-smi.

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


get_smoke_gpu_pool = get_gpu_ids


_nvidia_smi_path: Optional[str] = None
_nvidia_smi_resolved = False


def _nvidia_smi() -> Optional[str]:
    """Path to nvidia-smi, or None on machines that do not have it.

    PPU and ROCm workers have no nvidia-smi at all, so a failed query there says
    nothing about device health -- unlike an NVIDIA box, where a failing query is
    itself a symptom.
    """
    global _nvidia_smi_path, _nvidia_smi_resolved
    if not _nvidia_smi_resolved:
        _nvidia_smi_path = next(
            (resolved for path in _NVIDIA_SMI_PATHS if (resolved := shutil.which(path))),
            None,
        )
        _nvidia_smi_resolved = True
        if _nvidia_smi_path is None:
            logging.info(
                "nvidia-smi not present; skipping NVIDIA zombie-context checks"
            )
    return _nvidia_smi_path


class DeviceResource:
    def __init__(self, required_gpu_count: int, timeout: Optional[int] = None):
        """
        Args:
            timeout: seconds to wait for GPU locks.
                     None uses the environment or the bounded default.
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
        elif "RTP_GPU_ACQUIRE_TIMEOUT" in os.environ:
            self.timeout = int(os.environ["RTP_GPU_ACQUIRE_TIMEOUT"])
        else:
            self.timeout = GPU_LOCK_DEFAULT_TIMEOUT if _remote_session_id() else 1800
        self.gpu_ids: List[str] = []
        self.candidate_groups: Optional[List[List[int]]] = None
        self.gpu_locks = ExitStack()
        self.global_lock_file = GPU_GLOBAL_LOCK_FILE
        self.gpu_status_root_path = GPU_STATUS_ROOT
        self._gpu_bad_until: Dict[str, float] = {}
        self.bad_gpu_cooldown_s = 30
        self._lock_start_idx = 0

    def _get_gpu_pids(self, gpu_id: str) -> Optional[List[int]]:
        """PIDs of compute processes on a physical GPU, or None if unknowable.

        None means nvidia-smi could not answer -- it errored, or timed out. That
        must not be confused with "no processes": a wedged or degraded device is
        precisely the case where the query fails, and reporting it as idle keeps
        handing it to tests.
        """
        try:
            result = subprocess.run(
                [
                    _nvidia_smi() or "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                    f"--id={gpu_id}",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logging.warning(
                    "nvidia-smi failed for gpu %s (rc=%s): %s",
                    gpu_id,
                    result.returncode,
                    (result.stderr or "").strip()[:200],
                )
                return None
            return [
                int(p.strip())
                for p in result.stdout.strip().splitlines()
                if p.strip()
            ]
        except subprocess.TimeoutExpired:
            logging.warning("nvidia-smi timed out querying gpu %s", gpu_id)
            return None
        except Exception as e:
            logging.warning("nvidia-smi query failed for gpu %s: %s", gpu_id, e)
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Check if a process is alive by probing /proc/<pid>."""
        return os.path.exists(f"/proc/{pid}")

    def _has_zombie_gpu_contexts(self, gpu_id: str) -> bool:
        """Whether this GPU should be considered unusable.

        Covers two cases: dead processes still holding memory (the driver
        sometimes fails to reclaim after a SIGKILL, leaving contexts that survive
        until a reset), and a device we cannot query at all -- treated as bad, so
        a degraded GPU is skipped rather than handed out repeatedly.

        "Cannot query" only means anything where nvidia-smi exists. On PPU and
        ROCm workers it never does, and treating that as a zombie rejected all 16
        PPUs in run 69752178, so no device could be locked and every PPU smoke
        test failed at the acquisition bound with nothing actually held.
        """
        if _nvidia_smi() is None:
            return False
        pids = self._get_gpu_pids(gpu_id)
        if pids is None:
            logging.warning("gpu %s is not queryable; treating as unusable", gpu_id)
            return True
        if not pids:
            return False
        return all(not self._pid_alive(p) for p in pids)

    def _has_non_session_live_cuda_pids(self, gpu_id: str) -> bool:
        """Return True when a remotely locked GPU already has other live CUDA PIDs."""
        session_id = _remote_session_id()
        if not session_id:
            return False
        pids = self._get_gpu_pids(gpu_id)
        if pids is None:
            return _nvidia_smi() is not None
        for pid in pids:
            if not self._pid_alive(pid):
                continue
            pid_session = _pid_session_id(pid)
            if pid_session != session_id:
                logging.error(
                    "GPU %s has non-session live CUDA pid=%s session=%s while current session=%s",
                    gpu_id,
                    pid,
                    pid_session or "unset",
                    session_id,
                )
                return True
        return False

    def _has_excess_preexisting_memory(self, gpu_id: str) -> bool:
        """Detect remote GPU users hidden by the worker's PID namespace."""
        if not _remote_session_id():
            return False
        limit_mb = int(
            os.environ.get(
                GPU_MAX_PREEXISTING_MEMORY_MB_ENV,
                str(GPU_MAX_PREEXISTING_MEMORY_MB_DEFAULT),
            )
        )
        for smi in _NVIDIA_SMI_PATHS:
            try:
                result = subprocess.run(
                    [
                        smi,
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={gpu_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
            except FileNotFoundError:
                continue
            if result.returncode != 0:
                continue
            try:
                used_mb = int(result.stdout.strip().splitlines()[0])
            except (IndexError, ValueError):
                return False
            if used_mb > limit_mb:
                logging.error(
                    "GPU %s already uses %s MiB before session start (limit=%s MiB)",
                    gpu_id,
                    used_mb,
                    limit_mb,
                )
                return True
            return False
        return False

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

    def _locked_gpu_ids(self) -> List[int]:
        """Visible GPUs currently held by someone else, for diagnostics only."""
        held = []
        for id in self.total_gpus:
            probe = FileLock(f"{self.gpu_status_root_path}/{id}")
            try:
                with probe.acquire(timeout=0):
                    pass
            except Timeout:
                held.append(id)
            except Exception:
                pass
        return held

    def _lock_gpus(self):
        candidate_groups = self._candidate_gpu_groups()
        preexisting_memory_seen = False
        with ExitStack() as stack:
            now = time.time()
            for group in candidate_groups:
                gpu_ids = []
                with ExitStack() as group_stack:
                    for id in group:
                        if self._gpu_bad_until.get(str(id), 0) > now:
                            logging.info(f"skip GPU {id}: temporary cooldown")
                            break
                        if self._has_zombie_gpu_contexts(str(id)):
                            logging.info(f"skip GPU {id}: zombie CUDA contexts detected")
                            break
                        lock_device = FileLock(f"{self.gpu_status_root_path}/{id}")
                        try:
                            group_stack.enter_context(lock_device.acquire(timeout=0))
                        except Timeout as _:
                            logging.info(f"lock device {id} failed")
                            break
                        if self._has_non_session_live_cuda_pids(str(id)):
                            logging.info(
                                "skip GPU %s: non-session CUDA process detected", id
                            )
                            break
                        if self._has_excess_preexisting_memory(str(id)):
                            preexisting_memory_seen = True
                            logging.info(
                                "skip GPU %s: pre-existing memory exceeds limit", id
                            )
                            break
                        if self._has_zombie_gpu_contexts(str(id)):
                            logging.info(f"skip GPU {id}: zombie CUDA contexts detected")
                            break
                        gpu_ids.append(str(id))
                        logging.info(f"{get_ip()} lock device {id} done")
                    if len(gpu_ids) == self.required_gpu_count:
                        logging.info(f"use gpus:[{gpu_ids}]")
                        stack.enter_context(group_stack.pop_all())
                        self.gpu_locks = stack.pop_all()
                        self.gpu_ids = gpu_ids
                        return True
        if preexisting_memory_seen:
            raise GpuLockTimeoutError(
                "GPU lock unavailable: all candidate groups have pre-existing "
                "memory allocations"
            )
        return False

    def _candidate_gpu_groups(self) -> List[List[int]]:
        if self.candidate_groups is not None:
            return self.candidate_groups
        if self.required_gpu_count <= 1:
            self.candidate_groups = [[id] for id in self.total_gpus]
            return self.candidate_groups

        numa_groups = self._get_topology_numa_groups()
        candidates: List[List[int]] = []
        seen = set()
        total_gpu_set = set(self.total_gpus)
        for group in numa_groups:
            visible_group = [id for id in group if id in total_gpu_set]
            for start in range(0, len(visible_group) - self.required_gpu_count + 1):
                candidate = visible_group[start : start + self.required_gpu_count]
                key = tuple(candidate)
                if key not in seen:
                    seen.add(key)
                    candidates.append(candidate)

        # NUMA grouping is only a preference: fall back to every window over the
        # visible GPUs so a busy NUMA node can never starve the whole request.
        for start in range(0, len(self.total_gpus) - self.required_gpu_count + 1):
            candidate = self.total_gpus[start : start + self.required_gpu_count]
            key = tuple(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)
        if candidates:
            logging.info(f"candidate gpu groups: {candidates}")
        self.candidate_groups = candidates
        return candidates

    def _get_topology_numa_groups(self) -> List[List[int]]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []
            numa_column = -1
            groups: Dict[str, List[int]] = {}
            for raw_line in result.stdout.splitlines():
                columns = [column.strip() for column in raw_line.split("\t")]
                if numa_column < 0:
                    if "NUMA Affinity" in columns:
                        numa_column = columns.index("NUMA Affinity")
                    continue
                name = columns[0]
                if not name.startswith("GPU") or not name[3:].isdigit():
                    continue
                if numa_column >= len(columns):
                    continue
                numa_id = columns[numa_column]
                if not numa_id.isdigit():
                    continue
                groups.setdefault(numa_id, []).append(int(name[3:]))
            return [sorted(group) for _, group in sorted(groups.items())]
        except Exception:
            return []

    def __enter__(self):
        started = time.time()
        deadline = started + self.timeout if self.timeout is not None else None
        next_report = started + 60
        logging.info(
            "waiting for %s of %s visible GPUs, timeout=%s",
            self.required_gpu_count, len(self.total_gpus), self.timeout,
        )
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
                    f"need {self.required_gpu_count} GPUs from {self.total_gpus}; "
                    f"currently held: {self._locked_gpu_ids()}"
                )
            now = time.time()
            if now >= next_report:
                logging.info(
                    "still waiting for %s GPUs after %ss; held: %s",
                    self.required_gpu_count, int(now - started), self._locked_gpu_ids(),
                )
                next_report = now + 60
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


def _get_required_gpu_count(device_available: bool = True) -> int:
    gpu_count_env = os.environ.get("GPU_COUNT", os.environ.get("WORLD_SIZE"))
    if gpu_count_env is not None:
        return int(gpu_count_env)
    return 1 if device_available else 0


if __name__ == "__main__":
    device_info = get_device_info()
    # `--run_under=gpu_lock` wraps the entire Bazel C++ suite, including tests
    # dispatched to CPU workers. Lock one GPU by default when a device exists;
    # a caller that explicitly requests GPUs still fails on a CPU-only worker.
    require_count = _get_required_gpu_count(device_available=device_info is not None)

    if not device_info:
        if require_count > 0:
            raise RuntimeError(
                f"[device_resource] GPU_COUNT={require_count} requested but no GPU detected "
                f"(nvidia-smi / rocm-smi not found or returned no devices)"
            )
        logging.warning(
            "[device_resource] no GPU detected, running without GPU isolation"
        )
        sys.exit(_run_child(sys.argv[1:]))

    if require_count == 0:
        logging.info("[device_resource] GPU_COUNT=0; skipping GPU lock")
        sys.exit(_run_child(sys.argv[1:]))

    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)

    with DeviceResource(require_count) as gpu_resource:
        os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
        sys.stderr.write(
            f"[device_resource] {env_name}={os.environ[env_name]} "
            f"locked={gpu_resource.gpu_ids} pid={os.getpid()}\n"
        )
        sys.stderr.flush()
        # Keep JIT-backed dependencies on the same stable package paths used by
        # the legacy Bazel GPU wrapper. setup_jit_cache() injects those paths
        # into the native pytest child via PYTHONPATH when runfiles are absent.
        from jit_sys_path_setup import setup_jit_cache

        setup_jit_cache()
        exit_code = _run_child(sys.argv[1:])
        logging.info("exitcode: %d", exit_code)
        sys.exit(exit_code)
