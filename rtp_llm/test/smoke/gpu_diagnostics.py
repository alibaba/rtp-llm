"""
GPU error classification and system state diagnostics for smoke tests.

Two separate classifiers for two separate domains:
- ExceptionType: classifies Python exceptions (test process)
- ProcessFailureType: classifies server subprocess exit codes
"""

import logging
import os
import re
import signal
import subprocess
import sys
from enum import Enum
from typing import List, Optional, Tuple

_LOG = logging.getLogger(__name__)

_SIGNAL_NAMES = {
    signal.SIGSEGV: "SIGSEGV",
    signal.SIGABRT: "SIGABRT",
    signal.SIGKILL: "SIGKILL",
    signal.SIGTERM: "SIGTERM",
    signal.SIGBUS: "SIGBUS",
    signal.SIGFPE: "SIGFPE",
    signal.SIGILL: "SIGILL",
}


class ExceptionType(Enum):
    """Classifies Python exceptions thrown during smoke test execution."""

    CUDA_OOM = "cuda_oom"
    CUDA_RUNTIME_ERROR = "cuda_runtime_error"
    NOT_GPU_ERROR = "not_gpu_error"


class ProcessFailureType(Enum):
    """Classifies server process exit status."""

    SIGNAL_KILLED = "signal_killed"
    NONZERO_EXIT = "nonzero_exit"
    HEALTH_TIMEOUT = "health_timeout"


def classify_exception(exc: Exception) -> ExceptionType:
    """Classify exception: OOM vs other CUDA error vs non-GPU error."""
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return ExceptionType.CUDA_OOM
    except ImportError:
        pass
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        if "out of memory" in msg:
            return ExceptionType.CUDA_OOM
        if "cuda" in msg:
            return ExceptionType.CUDA_RUNTIME_ERROR
    return ExceptionType.NOT_GPU_ERROR


def classify_process_exit(returncode: Optional[int]) -> Tuple[ProcessFailureType, str]:
    """Classify server process exit code. Returns (type, human-readable description)."""
    if returncode is None:
        return ProcessFailureType.HEALTH_TIMEOUT, "process still alive, health check timed out"
    if returncode < 0:
        sig = -returncode
        sig_name = _SIGNAL_NAMES.get(sig, f"signal {sig}")
        return ProcessFailureType.SIGNAL_KILLED, f"killed by {sig_name} (exit code {returncode})"
    return ProcessFailureType.NONZERO_EXIT, f"exited with code {returncode}"


# ---------------------------------------------------------------------------
# dmesg snapshot / delta
# ---------------------------------------------------------------------------

def snapshot_dmesg() -> int:
    """Capture current dmesg line count as baseline. Call at case start."""
    try:
        out = subprocess.run(
            ["dmesg"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.count("\n")
    except Exception:
        pass
    return 0


_DMESG_ERROR_PATTERNS = re.compile(
    r"oom|killed process|segfault|traps:|general protection|"
    r"out of memory|sigkill|sigsegv|sigabrt|invoked oom-killer|"
    r"page allocation failure|memory cgroup",
    re.IGNORECASE,
)


def dump_dmesg_errors(baseline: int = 0) -> str:
    """Read dmesg lines generated after baseline, filter for crash/OOM patterns."""
    try:
        out = subprocess.run(
            ["dmesg", "-T"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return f"dmesg failed (code {out.returncode})"
        lines = out.stdout.splitlines()
        new_lines = lines[baseline:]
        matched = [l for l in new_lines if _DMESG_ERROR_PATTERNS.search(l)]
        if not matched:
            return f"(no crash/OOM messages in {len(new_lines)} new dmesg lines)"
        return "\n".join(matched[-30:])
    except FileNotFoundError:
        return "dmesg not found"
    except subprocess.TimeoutExpired:
        return "dmesg timeout"
    except PermissionError:
        return "dmesg permission denied"
    except Exception as e:
        return f"dmesg error: {e}"


# ---------------------------------------------------------------------------
# GPU / system state collectors
# ---------------------------------------------------------------------------

import shutil

_SMI_SEARCH_PATHS = [
    "/usr/local/cuda/bin",
    "/usr/local/corex/bin",
    "/usr/bin",
    "/usr/local/bin",
]


def _detect_smi_command() -> str:
    """Detect GPU SMI tool: nvidia-smi (CUDA/PPU) or ppusmi."""
    for name in ("nvidia-smi", "ppusmi"):
        path = shutil.which(name)
        if path:
            return path
    for search_dir in _SMI_SEARCH_PATHS:
        for name in ("nvidia-smi", "ppusmi"):
            full = os.path.join(search_dir, name)
            if os.path.isfile(full) and os.access(full, os.X_OK):
                return full
    return ""


def _gpu_smi_memory() -> str:
    """Run GPU SMI tool to get memory usage (auto-detects nvidia-smi or ppusmi)."""
    smi = _detect_smi_command()
    if not smi:
        return "no GPU SMI tool found (tried nvidia-smi, ppusmi in PATH and /usr/local/cuda/bin)"
    try:
        out = subprocess.run(
            [smi, "-q", "-d", "MEMORY"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout
        return f"{smi} failed: {out.stderr or out.stdout}"
    except subprocess.TimeoutExpired:
        return f"{smi} timeout"
    except Exception as e:
        return f"{smi} error: {e}"


def _gpu_smi_processes() -> str:
    """Run GPU SMI tool for per-process usage."""
    smi = _detect_smi_command()
    if not smi:
        return ""
    try:
        out = subprocess.run(
            [smi, "pmon", "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout
        return ""
    except Exception:
        return ""


def _torch_cuda_memory_summary() -> str:
    """Get torch.cuda memory stats for the test process."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "torch.cuda not available"
        lines = ["torch.cuda (current process):"]
        for i in range(torch.cuda.device_count()):
            try:
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                lines.append(
                    f"  device {i}: allocated={allocated:.3f} GiB reserved={reserved:.3f} GiB"
                )
            except Exception as e:
                lines.append(f"  device {i}: error {e}")
        return "\n".join(lines)
    except ImportError:
        return "torch not imported"
    except Exception as e:
        return f"torch.cuda summary error: {e}"


def _proc_status(pid: Optional[int] = None) -> str:
    """Read /proc/<pid>/status for memory info (RSS, etc.)."""
    path = f"/proc/{pid}/status" if pid else "/proc/self/status"
    label = f"pid={pid}" if pid else "self"
    try:
        with open(path, "r") as f:
            return f"({label})\n{f.read()}"
    except Exception as e:
        return f"/proc/{label}/status error: {e}"


# ---------------------------------------------------------------------------
# process.log scanning
# ---------------------------------------------------------------------------

_PROCESS_LOG_PATTERNS = re.compile(
    r"Error|Exception|OOM|out of memory|CUDA|NCCL|"
    r"Segmentation|Aborted|killed|SIGSE|SIGAB|"
    r"RuntimeError|FileNotFoundError|KeyError|AttributeError|"
    r"core dumped|exit code|failed to",
    re.IGNORECASE,
)


def scan_process_log(log_path: Optional[str], max_lines: int = 50) -> List[str]:
    """Extract error-relevant lines from server process.log."""
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "r") as f:
            all_lines = f.readlines()
        matched = [l.rstrip() for l in all_lines if _PROCESS_LOG_PATTERNS.search(l)]
        tail = all_lines[-20:]
        tail_lines = [f"[TAIL] {l.rstrip()}" for l in tail]
        result = matched[-max_lines:] + ["--- last 20 lines ---"] + tail_lines
        return result
    except Exception as e:
        return [f"Failed to scan {log_path}: {e}"]


# ---------------------------------------------------------------------------
# Main state dump
# ---------------------------------------------------------------------------

def dump_gpu_state(
    exc: Optional[Exception] = None,
    failure_context: str = "unknown failure",
    log_path: Optional[str] = None,
    server_pid: Optional[int] = None,
    server_proc_status: Optional[str] = None,
    dmesg_baseline: int = 0,
) -> None:
    """
    Dump GPU and system state for any smoke test failure.

    Args:
        exc: The exception that triggered the dump (if any)
        failure_context: Human-readable description of the failure type
        log_path: Optional path to write the dump report
        server_pid: Optional server process PID for /proc/<pid>/status
        server_proc_status: Pre-captured /proc/<pid>/status (use when server already exited)
        dmesg_baseline: dmesg line count from snapshot_dmesg() at case start
    """
    pid = os.getpid()
    header = (
        f"\n{'=' * 60}\n"
        f"GPU/System state dump (test pid={pid})\n"
        f"Failure context: {failure_context}\n"
        f"{'=' * 60}\n"
    )
    if exc is not None:
        header += f"Exception: {type(exc).__name__}: {exc}\n"

    smi_path = _detect_smi_command()
    smi_name = os.path.basename(smi_path) if smi_path else "gpu-smi"
    sections = [
        (f"{smi_name} -q (memory)", _gpu_smi_memory()),
        (f"{smi_name} pmon (per-process)", _gpu_smi_processes()),
        ("torch.cuda (test process)", _torch_cuda_memory_summary()),
        ("/proc/self/status (test process)", _proc_status()),
        ("dmesg errors (during case run)", dump_dmesg_errors(dmesg_baseline)),
    ]

    if server_proc_status:
        label = f"pid={server_pid}" if server_pid else "server"
        sections.append(
            (f"/proc/{label}/status (server, pre-captured)", f"({label})\n{server_proc_status}")
        )
    elif server_pid:
        sections.append(
            (f"/proc/{server_pid}/status (server process)", _proc_status(server_pid))
        )

    body = "\n".join(f"--- {name} ---\n{content}\n" for name, content in sections)
    report = header + body + "=" * 60 + "\n"

    sys.stderr.write(report)
    sys.stderr.flush()
    _LOG.error(report)

    if log_path:
        try:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "w") as f:
                f.write(report)
            _LOG.info("GPU state dump written to %s", log_path)
        except Exception as e:
            _LOG.warning("Failed to write GPU state dump to %s: %s", log_path, e)
