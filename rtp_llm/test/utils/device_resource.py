import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Set, Tuple

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


class GpuLockTimeoutError(GpuLockError, TimeoutError):
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


def _run_child(argv: List[str], device_resource=None) -> int:
    """Run with inherited streams, bounded lifetime and exact session ownership."""
    if not _IS_POSIX:
        return subprocess.run(argv).returncode
    session_id = _remote_session_id()
    owner_pid = os.getpid()
    kwargs = {"start_new_session": True}
    if session_id:
        env = dict(os.environ)
        env["RTP_DEVICE_RESOURCE_OWNER_PID"] = str(owner_pid)
        kwargs["env"] = env
    child = subprocess.Popen(argv, **kwargs)
    if device_resource is not None:
        device_resource.own_test_process_group = child.pid
    try:
        code = _wait_for_test(child, child.pid, _watchdog_seconds())
        return _WATCHDOG_EXIT_CODE if code is None else code
    finally:
        if child.poll() is None:
            _kill_process_group(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kill_process_group(child.pid, signal.SIGKILL)
        if session_id:
            escaped = _session_pids(session_id, owner_pid=owner_pid)
            if escaped:
                logging.warning(
                    "[device_resource] cleaning escaped session pids: %s", escaped
                )
                _kill_pids(escaped, signal.SIGTERM)
                time.sleep(1)
                _kill_pids(
                    _session_pids(session_id, owner_pid=owner_pid), signal.SIGKILL
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
            gpu_names = {}
            for line in result.stdout.strip().splitlines():
                gpu = re.match(r"\s*GPU\[(\d+)\]\s*:", line)
                if gpu:
                    parts = line.split(":")
                    if len(parts) >= 3 and parts[-1].strip():
                        # --showproductname emits several properties per GPU.
                        gpu_names.setdefault(gpu.group(1), parts[-1].strip())
            if gpu_names:
                return next(iter(gpu_names.values())), len(gpu_names)
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


python_code = """
import torch
import json
import os
import sys

def report_driver_details():
    details = {"torch_cuda": torch.version.cuda}
    try:
        details["cuda_root"] = os.path.realpath("/usr/local/cuda")
        with open("/proc/driver/nvidia/version") as f:
            details["kernel_driver"] = f.read().strip()
        with open("/proc/self/maps") as f:
            details["libcuda"] = sorted({
                line.split(maxsplit=5)[-1].strip() for line in f if "/libcuda.so" in line
            })
    except OSError as e:
        details["driver_details_error"] = str(e)
    print("CUDA probe diagnostics: " + json.dumps(details), file=sys.stderr)

try:
    if torch.cuda.is_available():
        device_info = {torch.cuda.get_device_name(0): torch.cuda.device_count()}
    else:
        device_info = {}
        report_driver_details()
    print(json.dumps(device_info))
except Exception as e:
    report_driver_details()
    # 捕获并打印内部错误，避免子进程静默失败
    print(json.dumps({"error": str(e), "note": "Failed to get CUDA info"}), file=sys.stderr)
    sys.exit(1) # 告知外部进程执行失败
"""


def _retry_native_cuda_driver(result):
    # An obsolete compat libcuda can shadow the worker's valid host driver.
    # Probe in a fresh process: an initialized CUDA runtime cannot switch libs.
    library_path = os.environ.get("LD_LIBRARY_PATH", "")
    native_path = ":".join(
        part
        for part in library_path.split(":")
        if not (
            "cuda" in part
            and (part.rstrip("/").endswith("/compat") or "/compat/" in part)
        )
    )
    if native_path == library_path:
        raise RuntimeError(
            "CUDA driver mismatch (803): align the worker's libcuda libraries "
            f"with its host driver. Host: {get_ip()}. Probe: {result.stderr.strip()}"
        )
    native = subprocess.run(
        [sys.executable, "-c", python_code],
        env={**os.environ, "LD_LIBRARY_PATH": native_path},
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        native_info = json.loads(native.stdout) if native.returncode == 0 else {}
    except ValueError:
        native_info = {}
    if not native_info or not all(
        isinstance(count, int) and count > 0 for count in native_info.values()
    ):
        raise RuntimeError(
            "CUDA driver mismatch (803): both configured compat and native "
            "driver probes failed. Align the worker's libcuda libraries "
            f"with its host driver. Host: {get_ip()}. "
            f"Configured probe: {result.stderr.strip()}. Native probe: {native.stderr.strip()}"
        )
    os.environ["LD_LIBRARY_PATH"] = native_path
    logging.warning(
        "CUDA compat probe failed with 803; using the working native driver"
    )
    return native


_CUDA_PROBE_LOCK = threading.Lock()


def get_cuda_info():
    with _CUDA_PROBE_LOCK:
        result = subprocess.run(
            [sys.executable, "-c", python_code],
            capture_output=True,
            text=True,
            check=False,
        )
        if (
            "Error 803:" in result.stderr
            or "system has unsupported display driver" in result.stderr
        ):
            result = _retry_native_cuda_driver(result)
        logging.info(
            f"cuda info result: {result.stdout}, return code: {result.returncode}"
        )

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
            (
                resolved
                for path in _NVIDIA_SMI_PATHS
                if (resolved := shutil.which(path))
            ),
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
        self.own_test_process_group: Optional[int] = None

    @property
    def acquire_timeout(self):
        return self.timeout

    @acquire_timeout.setter
    def acquire_timeout(self, value):
        self.timeout = value

    def _get_gpu_pids(self, gpu_id: str) -> Optional[List[int]]:
        """PIDs of compute processes on a physical GPU, or None if unknowable."""
        rows = self._query_compute_apps(gpu_id, "pid")
        if rows is None:
            return None
        try:
            return [int(row[0]) for row in rows if row]
        except (ValueError, IndexError):
            logging.warning("unexpected nvidia-smi pid output for gpu %s", gpu_id)
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """Probe process existence without signalling it."""
        if sys.platform.startswith("linux"):
            return os.path.exists(f"/proc/{pid}")
        try:
            os.getpgid(pid)
            return True
        except (OSError, AttributeError):
            return False

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

    def _query_compute_apps(
        self, gpu_id: str, columns: str
    ) -> Optional[List[List[str]]]:
        """Cell values of each compute process on a GPU, or None if unknowable.

        None means nvidia-smi could not answer -- it errored, or timed out. That
        must not be confused with "no processes": a wedged or degraded device is
        precisely the case where the query fails, and reporting it as idle keeps
        handing it to tests.
        """
        try:
            result = subprocess.run(
                [
                    _nvidia_smi() or "nvidia-smi",
                    f"--query-compute-apps={columns}",
                    "--format=csv,noheader,nounits",
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
                [cell.strip() for cell in line.split(",")]
                for line in (result.stdout or "").strip().splitlines()
                if line.strip()
            ]
        except subprocess.TimeoutExpired:
            logging.warning("nvidia-smi timed out querying gpu %s", gpu_id)
            return None
        except Exception as e:
            logging.warning("nvidia-smi query failed for gpu %s: %s", gpu_id, e)
            return None

    def _get_gpu_compute_apps(
        self, gpu_id: str
    ) -> Optional[List[Tuple[int, Optional[int]]]]:
        """(pid, used MiB) of each compute process on a GPU, or None.

        One query carries both facts so a log line can say who holds the device
        and how much: a bare PID list cannot tell "the neighbour's test is
        mid-run" from "a dead process left 252 GiB behind". Used memory is best
        effort -- a driver that cannot report it falls back to the PID-only
        query with memory None instead of declaring the device unqueryable.
        """
        rows = self._query_compute_apps(gpu_id, "pid,used_memory")
        if rows is None:
            rows = self._query_compute_apps(gpu_id, "pid")
        if rows is None:
            return None
        apps: List[Tuple[int, Optional[int]]] = []
        for row in rows:
            if not row:
                continue
            try:
                pid = int(row[0])
            except ValueError:
                logging.warning(
                    "unexpected nvidia-smi process output for gpu %s: %r", gpu_id, row
                )
                return None
            apps.append((pid, _parse_mib(row[1]) if len(row) > 1 else None))
        return apps

    def _get_gpu_used_memory_mib(self, gpu_id: str) -> Optional[int]:
        """`memory.used` of a GPU in MiB, or None if nvidia-smi cannot say."""
        try:
            result = subprocess.run(
                [
                    _nvidia_smi() or "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_id),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logging.warning(
                    "nvidia-smi memory query failed for gpu %s (rc=%s): %s",
                    gpu_id,
                    result.returncode,
                    (result.stderr or "").strip()[:200],
                )
                return None
            lines = [
                line for line in (result.stdout or "").splitlines() if line.strip()
            ]
            if not lines:
                return None
            return int(lines[0].strip().split()[0])
        except subprocess.TimeoutExpired:
            logging.warning("nvidia-smi timed out querying memory for gpu %s", gpu_id)
            return None
        except Exception as e:
            logging.warning("nvidia-smi memory query failed for gpu %s: %s", gpu_id, e)
            return None

    def _pid_in_test_process_group(self, pid: int) -> bool:
        """Whether pid is the test process or shares its group/session.

        main() starts the test with start_new_session=True, so the child's pid
        is also its pgid and sid, and everything it spawns inherits both unless
        it creates its own session. Processes outside that group belong to
        someone else, even when they sit on a device we hold the lock for.
        """
        group = self.own_test_process_group
        if group is None:
            return False
        if pid == group:
            return True
        try:
            with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as stat:
                contents = stat.read()
        except OSError:
            try:
                return os.getpgid(pid) == group or os.getsid(pid) == group
            except (OSError, AttributeError):
                return False
        # comm may itself contain spaces or parentheses; rsplit skips to the
        # fields that follow the closing one. After comm: state(3) ppid(4)
        # pgrp(5) session(6).
        after_comm = contents.rsplit(")", 1)
        if len(after_comm) < 2:
            return False
        fields = after_comm[1].split()
        if len(fields) < 4:
            return False
        return fields[2] == str(group) or fields[3] == str(group)

    def _gpu_idle_problem(self, gpu_id: str) -> Optional[str]:
        """Why this GPU is not truly idle, or None when it is.

        A lock only says no *other wrapper* claims the device; it cannot prove
        the device is free. Ask nvidia-smi instead: memory left behind by a
        killed context, a process that ignores the lock protocol, or a zombie
        context all hand a test a device that dies with CUDA OOM a moment
        later. During acquisition none of these can be "cleaned up" -- whatever
        holds the memory is a live process of someone else.
        """
        limit = _gpu_free_memory_mib()
        problems: List[str] = []
        if limit >= 0:
            used = self._get_gpu_used_memory_mib(gpu_id)
            if used is None:
                problems.append(f"gpu={gpu_id} memory.used=unknown")
            elif used > limit:
                problems.append(
                    f"gpu={gpu_id} memory.used={used}MiB > "
                    f"RTP_GPU_FREE_MEMORY_MIB={limit}"
                )
        apps = self._get_gpu_compute_apps(gpu_id)
        if apps is None:
            problems.append(f"gpu={gpu_id} compute-apps query failed")
            return "; ".join(problems)
        for pid, memory_mib in apps:
            session_id = _remote_session_id()
            if session_id and _pid_session_id(pid) == session_id:
                continue
            if not self._pid_alive(pid):
                logging.warning("zombie context: gpu=%s pid=%s", gpu_id, pid)
                problems.append(f"gpu={gpu_id} zombie-context pid={pid}")
                continue
            logging.warning(
                "busy: gpu=%s pid=%s mem=%s", gpu_id, pid, _memory_text(memory_mib)
            )
            problems.append(
                f"gpu={gpu_id} busy pid={pid} mem={_memory_text(memory_mib)}"
            )
        return "; ".join(problems) if problems else None

    def _gpu_group_idle_problems(self, gpu_ids: List[str]) -> List[str]:
        """Reasons the whole candidate group is not idle (empty list == idle)."""
        if _nvidia_smi() is None:
            # PPU/ROCm workers have no nvidia-smi; the lock protocol is all the
            # evidence available there, exactly as before.
            return []
        problems: List[str] = []
        for gpu_id in gpu_ids:
            problem = self._gpu_idle_problem(str(gpu_id))
            if problem is not None:
                problems.append(problem)
        return problems

    def _gpu_state_snapshot(self) -> str:
        """Per-GPU memory and compute processes, for acquisition timeouts.

        A lock-holder list alone misleads -- a wrapper can hold the lock while
        using nothing, and a device can be busy while its lock looks free -- so
        report what nvidia-smi actually shows. Never raises: diagnostics must
        not mask the real error.
        """
        parts: List[str] = []
        for gpu_id in self.total_gpus:
            try:
                used = self._get_gpu_used_memory_mib(str(gpu_id))
                apps = self._get_gpu_compute_apps(str(gpu_id))
                if apps is None:
                    pids = "n/a" if _nvidia_smi() is None else "unknown"
                else:
                    pids = ",".join(
                        f"{pid}:{'alive' if self._pid_alive(pid) else 'zombie'}"
                        for pid, _ in apps
                    )
                    if not pids:
                        pids = "none"
                used_text = f"{used}MiB" if used is not None else "unknown"
                parts.append(f"gpu{gpu_id}: used={used_text} pids=[{pids}]")
            except Exception as e:
                parts.append(f"gpu{gpu_id}: unreadable ({e})")
        return "; ".join(parts)

    @staticmethod
    def _signal_own_process(
        pid: int, gpu_id: str, sigterm_sent: Set[int], sigkill_sent: Set[int]
    ) -> None:
        """SIGTERM first to allow graceful CUDA cleanup, SIGKILL on a later pass."""
        try:
            if pid not in sigterm_sent:
                os.kill(pid, signal.SIGTERM)
                logging.info("SIGTERM pid %s on GPU %s", pid, gpu_id)
                sigterm_sent.add(pid)
            elif pid not in sigkill_sent:
                os.kill(pid, signal.SIGKILL)
                logging.info("SIGKILL pid %s on GPU %s", pid, gpu_id)
                sigkill_sent.add(pid)
        except (ProcessLookupError, PermissionError):
            pass

    def _ensure_gpus_released(self, timeout: int = 30) -> bool:
        """Wait until acquired GPUs have no compute processes of our test left.

        Only processes of the test's own process group may be signalled (see
        `own_test_process_group`). A live foreign process is someone else's
        work: killing it either fails the co-tenant's run mid-flight or leaves
        its memory to be reclaimed later, which is exactly what poisons the
        next test. It is reported instead, and the device counts as unreleased.
        Zombie contexts (dead PIDs still holding memory) cannot be reclaimed at
        all. Nothing here may kill a neighbour.

        Returns True if the GPUs are clean, False otherwise.
        """
        sigterm_sent: Set[int] = set()
        sigkill_sent: Set[int] = set()
        deadline = time.time() + timeout

        while time.time() < deadline:
            all_clear = True
            own_live = False
            for gpu_id in self.gpu_ids:
                apps = self._get_gpu_compute_apps(gpu_id)
                if apps is None:
                    # Unknowable: no nvidia-smi, or it failed. Nothing to clean
                    # up and nothing to conclude. Iterating None here used to
                    # raise.
                    continue
                for pid, memory_mib in apps:
                    if pid == os.getpid():
                        continue
                    if not self._pid_alive(pid):
                        logging.warning(
                            "unreclaimable zombie context gpu=%s pid=%s", gpu_id, pid
                        )
                        all_clear = False
                    elif not self._pid_in_test_process_group(pid):
                        logging.warning(
                            "device held by live foreign pid=%s mem=%s; not killing",
                            pid,
                            _memory_text(memory_mib),
                        )
                        all_clear = False
                    else:
                        all_clear = False
                        own_live = True
                        self._signal_own_process(
                            pid, gpu_id, sigterm_sent, sigkill_sent
                        )
            if all_clear:
                return True
            if not own_live:
                # Only foreign or dead PIDs remain: this wrapper has nothing
                # left to do, and waiting past this point just delays teardown.
                logging.warning(
                    "GPUs %s not released; no test process left to clean up",
                    self.gpu_ids,
                )
                return False
            time.sleep(1)

        logging.warning(
            f"GPU cleanup timed out after {timeout}s for GPUs {self.gpu_ids}"
        )
        return False

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
                            logging.info(
                                f"skip GPU {id}: zombie CUDA contexts detected"
                            )
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
                            logging.info(
                                f"skip GPU {id}: zombie CUDA contexts detected"
                            )
                            break
                        gpu_ids.append(str(id))
                        logging.info(f"{get_ip()} lock device {id} done")
                    if len(gpu_ids) == self.required_gpu_count:
                        problems = self._gpu_group_idle_problems(gpu_ids)
                        if problems:
                            logging.warning(
                                "GPUs %s are not truly idle (%s); trying next group",
                                gpu_ids,
                                "; ".join(problems),
                            )
                            continue
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
            self.required_gpu_count,
            len(self.total_gpus),
            self.timeout,
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
                problems = self._gpu_group_idle_problems(self.gpu_ids)
                if not problems:
                    break
                logging.warning(
                    "GPUs %s are not truly idle (%s); releasing and retrying",
                    self.gpu_ids,
                    "; ".join(problems),
                )
                for gid in self.gpu_ids:
                    self._gpu_bad_until[gid] = time.time() + self.bad_gpu_cooldown_s
                self.gpu_ids = []
                self.gpu_locks.close()
            if deadline is not None and time.time() >= deadline:
                raise GpuLockTimeoutError(
                    f"GPU lock timed out after {self.timeout}s: "
                    f"need {self.required_gpu_count} GPUs from {self.total_gpus}; "
                    f"gpu state: {self._gpu_state_snapshot()}"
                )
            now = time.time()
            if now >= next_report:
                logging.info(
                    "still waiting for %s GPUs after %ss; held: %s",
                    self.required_gpu_count,
                    int(now - started),
                    self._locked_gpu_ids(),
                )
                next_report = now + 60
            time.sleep(1)
        return self

    def __exit__(self, *args: Any):
        with FileLock(self.global_lock_file):
            logging.info(f"release gpu:{self.gpu_ids}")
            try:
                if self.own_test_process_group is not None:
                    self._ensure_gpus_released()
            finally:
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


_WATCHDOG_EXIT_CODE = 70
_WATCHDOG_SIGNAL_GRACE_SECONDS = 30
_WATCHDOG_POLL_SECONDS = 0.5
_IS_POSIX = os.name == "posix"


def _parse_mib(text: str) -> Optional[int]:
    """MiB value from an nvidia-smi cell, or None when it is not a number."""
    cleaned = text.strip().replace(",", "").replace("MiB", "").strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _memory_text(memory_mib: Optional[int]) -> str:
    """Render a MiB count for logs, preserving "we could not tell"."""
    return f"{memory_mib}" if memory_mib is not None else "unknown"


def _gpu_free_memory_mib() -> int:
    """How much GPU memory (MiB) may still be in use and count as idle.

    RTP_GPU_FREE_MEMORY_MIB defaults to 1024. A negative value relaxes
    acquisition to "no active compute processes" only, for platforms or loads
    where the driver itself accounts for memory on an otherwise free device.
    """
    raw = os.environ.get("RTP_GPU_FREE_MEMORY_MIB", "1024")
    try:
        return int(raw)
    except ValueError:
        logging.warning("invalid RTP_GPU_FREE_MEMORY_MIB=%r; using 1024", raw)
        return 1024


def _watchdog_seconds() -> int:
    """Runtime budget for a started test; defaults to 1800s, <=0 disables it."""
    raw = os.environ.get("RTP_TEST_WATCHDOG_SECONDS", "1800")
    try:
        return int(raw)
    except ValueError:
        logging.warning("invalid RTP_TEST_WATCHDOG_SECONDS=%r; using 1800", raw)
        return 1800


def _visible_gpus(env_name: str = "CUDA_VISIBLE_DEVICES") -> str:
    """Device list this test runs on, for log lines (NVIDIA or ROCm)."""
    if env_name == "HIP_VISIBLE_DEVICES":
        return os.environ.get(env_name, "")
    return os.environ.get(env_name) or os.environ.get("HIP_VISIBLE_DEVICES", "")


def _log_test_placement() -> None:
    """Pin node and cards into the log, so any later failure is locatable.

    Never raises: diagnostics must not fail a test run.
    """
    gpus = _visible_gpus()
    uuids = "<unavailable>"
    smi = _nvidia_smi()
    if smi is not None and gpus.strip():
        try:
            result = subprocess.run(
                [smi, "-i", gpus, "--query-gpu=uuid", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                found = [
                    line.strip()
                    for line in (result.stdout or "").splitlines()
                    if line.strip()
                ]
                uuids = ",".join(found) if found else "<unavailable>"
            else:
                logging.warning(
                    "nvidia-smi uuid query failed (rc=%s): %s",
                    result.returncode,
                    (result.stderr or "").strip()[:200],
                )
        except Exception as e:
            logging.warning("nvidia-smi uuid query failed: %s", e)
    logging.info(f"host={socket.gethostname()} gpus={gpus} uuids={uuids}")


def _signal_process_group(pgid: int, signum: int, reason: str) -> bool:
    try:
        os.killpg(pgid, signum)
        logging.warning(
            "%s: sent %s to test process group %s",
            reason,
            signal.Signals(signum).name,
            pgid,
        )
        return True
    except ProcessLookupError:
        logging.info("%s: test process group %s already gone", reason, pgid)
        return False
    except OSError as e:
        logging.warning("%s: cannot signal test process group %s: %s", reason, pgid, e)
        return False


def _process_group_exists(pgid: int) -> bool:
    """Whether the test process group still has any member."""
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        # EPERM: the group exists, we just may not signal it.
        return True


def _wait_for_group_exit(proc, pgid: int, timeout: float) -> bool:
    """Wait until the test process is reaped and its group has no members."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None and not _process_group_exists(pgid):
            return True
        time.sleep(_WATCHDOG_POLL_SECONDS)
    proc.poll()
    return not _process_group_exists(pgid)


def _kill_hung_test(proc, pgid: int, elapsed: int, watchdog: int) -> int:
    """Kill a hung test group, preserving a stack dump when the test can give one.

    SIGUSR1 first: a test that registered faulthandler (see
    rtp_llm.test.utils.hang_watchdog) prints every thread's stack, turning "it
    hung" into "it hung here". SIGKILL follows after a short grace, so the test
    and everything it spawned cannot outlive the wrapper and keep holding the
    cards.

    The grace is spent on the group, not just the direct child: SIGUSR1's
    default disposition can take the direct child down immediately while the
    workers it spawned are the ones that still have to print their stacks.
    """
    _signal_process_group(pgid, signal.SIGUSR1, "watchdog stack dump request")
    _wait_for_group_exit(proc, pgid, _WATCHDOG_SIGNAL_GRACE_SECONDS)
    _signal_process_group(pgid, signal.SIGKILL, "watchdog")
    _wait_for_group_exit(proc, pgid, _WATCHDOG_SIGNAL_GRACE_SECONDS)
    logging.error(
        "watchdog: hung test killed by watchdog host=%s gpus=%s elapsed=%ss "
        "watchdog=%ss",
        socket.gethostname(),
        _visible_gpus(),
        elapsed,
        watchdog,
    )
    return _WATCHDOG_EXIT_CODE


def _escalate_forwarded_signal(
    pgid: int, forwarded_at: Optional[float], grace: float
) -> bool:
    """True once a forwarded signal has been escalated to SIGKILL.

    Bazel SIGKILLs the wrapper after its own termination grace; a test that
    ignores the forwarded signal would be orphaned again, so the group is
    killed if it has not exited in time.
    """
    if forwarded_at is None or time.monotonic() - forwarded_at < grace:
        return False
    _signal_process_group(pgid, signal.SIGKILL, "wrapper termination grace expired")
    return True


def _wait_for_test(proc, pgid: int, watchdog: int) -> Optional[int]:
    """Wait for a test process; None means the watchdog killed it."""
    forwarded: List[int] = []
    forwarded_at: Optional[float] = None
    escalated = False

    def _handle_termination(signum, _frame):
        nonlocal forwarded_at
        forwarded.append(signum)
        if forwarded_at is None:
            forwarded_at = time.monotonic()
        _signal_process_group(pgid, signum, "wrapper received signal")
        if len(forwarded) > 1:
            _signal_process_group(
                pgid, signal.SIGKILL, "wrapper received repeated signal"
            )

    previous_handlers = {
        sig: signal.signal(sig, _handle_termination)
        for sig in (signal.SIGTERM, signal.SIGINT)
    }
    started = time.monotonic()
    deadline = started + watchdog if watchdog > 0 else None
    try:
        while True:
            code = proc.poll()
            if code is not None:
                if forwarded and not escalated:
                    # The direct child died from the forwarded signal, but the
                    # workers/helpers it spawned may still be alive; take the
                    # whole group down before returning.
                    _signal_process_group(
                        pgid, signal.SIGKILL, "wrapper group cleanup after signal"
                    )
                break
            if not escalated:
                escalated = _escalate_forwarded_signal(
                    pgid, forwarded_at, _WATCHDOG_SIGNAL_GRACE_SECONDS
                )
            if deadline is not None and time.monotonic() >= deadline:
                _kill_hung_test(proc, pgid, int(time.monotonic() - started), watchdog)
                return None
            time.sleep(_WATCHDOG_POLL_SECONDS)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    if forwarded:
        logging.warning(
            "wrapper received signal(s) %s; test exited with code %s", forwarded, code
        )
    if code < 0:
        # Killed by a signal: report the shell convention (128+N) instead of a
        # negative Python exit status.
        code = 128 - code
    return code


def _run_test_command(device_resource: Optional[DeviceResource] = None) -> int:
    if os.environ.get("GPU_COUNT") != "0":
        _log_test_placement()
    return _run_child(sys.argv[1:], device_resource)


def main():
    # Explicit CPU targets must not probe devices or initialize a GPU runtime.
    if os.environ.get("GPU_COUNT") == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["HIP_VISIBLE_DEVICES"] = ""
        return _run_test_command()
    device_info = get_device_info()
    require_count = _get_required_gpu_count(device_available=device_info is not None)
    if device_info is None:
        if require_count > 0:
            raise RuntimeError(
                f"[device_resource] GPU_COUNT={require_count} requested but no GPU detected "
                "(nvidia-smi / rocm-smi not found or returned no devices)"
            )
        logging.warning(
            "[device_resource] no GPU detected, running without GPU isolation"
        )
        return _run_test_command()
    if require_count == 0:
        return _run_test_command()
    device_name, _ = device_info
    env_name = _get_visible_devices_env(device_name)
    with DeviceResource(require_count) as gpu_resource:
        os.environ[env_name] = ",".join(gpu_resource.gpu_ids)
        sys.stderr.write(
            f"[device_resource] {env_name}={os.environ[env_name]} "
            f"locked={gpu_resource.gpu_ids} pid={os.getpid()}\n"
        )
        sys.stderr.flush()
        try:
            from jit_sys_path_setup import setup_jit_cache
        except ModuleNotFoundError:
            from rtp_llm.test.utils.jit_sys_path_setup import setup_jit_cache
        setup_jit_cache()
        return _run_test_command(gpu_resource)


if __name__ == "__main__":
    sys.exit(main())
