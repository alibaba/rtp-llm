"""Bounded parent-side waits for leases owned by managed workers or descendants."""

import json
import logging
import os
import socket
import stat
import time
from typing import Dict, Iterable, NamedTuple, Optional

CUDACORE_LEASE_PREFIX = "cudacore_lease."
_CUDACORE_DIAGNOSTICS_DIRNAME = "cudacore_diagnostics"
_CUDACORE_LEASE_MAX_WAIT_SECONDS = 35.0
_MAX_WINDOW_MS = 30000
_MAX_LEASE_BYTES = 16384
_MAX_ANCESTORS = 64
_POLL_INTERVAL_SECONDS = 0.5


class _ProcessIdentity(NamedTuple):
    parent_pid: int
    start_ticks: int


def cudacore_diagnostics_dir() -> str:
    template = os.environ.get("CUDA_COREDUMP_FILE", "")
    if template and "|" not in template:
        try:
            is_fifo = stat.S_ISFIFO(os.stat(template).st_mode)
        except OSError:
            is_fifo = False
        if not is_fifo:
            return os.path.dirname(os.path.abspath(template))
    return os.path.join(os.getcwd(), _CUDACORE_DIAGNOSTICS_DIRNAME)


def _process_identity(pid: int) -> Optional[_ProcessIdentity]:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as proc_file:
            line = proc_file.read(4096)
        # comm may contain spaces and parentheses; field 3 follows its final ')'.
        fields = line[line.rindex(")") + 1 :].split()
        return _ProcessIdentity(int(fields[1]), int(fields[19]))
    except (OSError, ValueError, IndexError):
        return None


def _managed_roots(pids: Iterable[int]) -> Dict[int, _ProcessIdentity]:
    roots = {}
    for pid in pids:
        if type(pid) is not int or pid <= 0:
            continue
        identity = _process_identity(pid)
        if identity is not None:
            roots[pid] = identity
    return roots


def _belongs_to_managed_tree(
    pid: int, start_id: str, roots: Dict[int, _ProcessIdentity]
) -> bool:
    current = _process_identity(pid)
    if current is None or start_id != f"{pid}-{current.start_ticks}":
        return False
    seen = set()
    for _ in range(_MAX_ANCESTORS):
        if pid in seen:
            return False
        seen.add(pid)
        if pid in roots:
            return current.start_ticks == roots[pid].start_ticks
        parent_pid = current.parent_pid
        if parent_pid <= 0 or parent_pid == pid:
            return False
        parent = _process_identity(parent_pid)
        if parent is None or parent.start_ticks > current.start_ticks:
            return False
        pid, current = parent_pid, parent
    return False


def _read_lease_remaining(
    path: str, roots: Dict[int, _ProcessIdentity], wall_now: float, mono_now: float
) -> float:
    try:
        with open(path, "r", encoding="utf-8") as lease_file:
            payload = lease_file.read(_MAX_LEASE_BYTES + 1)
        if len(payload) > _MAX_LEASE_BYTES:
            return 0.0
        lease = json.loads(payload)
        if not isinstance(lease, dict):
            return 0.0
        if lease.get("schema_version") != "rtp_llm.cudacore_lease.v1":
            return 0.0
        if lease.get("host") != socket.gethostname():
            return 0.0
        pid = lease.get("pid")
        created = lease.get("created_epoch_ms")
        deadline = lease.get("deadline_epoch_ms")
        window = lease.get("window_ms")
        if any(type(value) is not int for value in (pid, created, deadline, window)):
            return 0.0
        if pid <= 0 or created <= 0 or not 0 < window <= _MAX_WINDOW_MS:
            return 0.0
        if deadline - created != window:
            return 0.0
        if not _belongs_to_managed_tree(pid, lease.get("worker_start_id"), roots):
            return 0.0
        if "deadline_mono_ms" in lease or "created_mono_ms" in lease:
            created_mono = lease.get("created_mono_ms")
            deadline_mono = lease.get("deadline_mono_ms")
            if type(created_mono) is not int or type(deadline_mono) is not int:
                return 0.0
            if created_mono <= 0 or deadline_mono - created_mono != window:
                return 0.0
            remaining = deadline_mono / 1000.0 - mono_now
        else:
            # Compatibility with the first version of the temporary patch.
            remaining = deadline / 1000.0 - wall_now
        if remaining > window / 1000.0:
            return 0.0
        return max(0.0, remaining)
    except (OSError, ValueError, TypeError, OverflowError):
        return 0.0


def _remaining_for_roots(
    roots: Dict[int, _ProcessIdentity], directory: str, wall_now: float, mono_now: float
) -> float:
    if not roots:
        return 0.0
    try:
        entries = os.listdir(directory)
    except OSError:
        return 0.0
    return max(
        (
            _read_lease_remaining(os.path.join(directory, name), roots, wall_now, mono_now)
            for name in entries
            if name.startswith(CUDACORE_LEASE_PREFIX) and ".tmp." not in name
        ),
        default=0.0,
    )


def remaining_collection_seconds(
    pids: Iterable[int],
    diagnostics_dir: Optional[str] = None,
    now: Optional[float] = None,
) -> float:
    """Remaining window for live managed children AND their validated descendants."""
    return _remaining_for_roots(
        _managed_roots(pids),
        diagnostics_dir or cudacore_diagnostics_dir(),
        time.time() if now is None else now,
        time.monotonic(),
    )


def wait_for_collection_leases(
    pids: Iterable[int],
    diagnostics_dir: Optional[str] = None,
    reason: str = "",
) -> float:
    """Keep original worker deadlines; never extend for a repeated error or clock step."""
    roots = _managed_roots(pids)
    directory = diagnostics_dir or cudacore_diagnostics_dir()
    started = time.monotonic()
    wall_started = time.time()
    hard_deadline = started + _CUDACORE_LEASE_MAX_WAIT_SECONDS
    slept = False
    while True:
        mono_now = time.monotonic()
        remaining = _remaining_for_roots(
            roots, directory, wall_started + (mono_now - started), mono_now
        )
        if remaining <= 0:
            break
        budget = hard_deadline - time.monotonic()
        if budget <= 0:
            logging.warning("Cudacore lease wait reached hard deadline (%s)", reason)
            break
        if not slept:
            logging.warning(
                "Waiting up to %.1fs for cudacore collection before %s",
                min(remaining, budget), reason or "cleanup",
            )
        slept = True
        time.sleep(min(remaining, budget, _POLL_INTERVAL_SECONDS))
    return (time.monotonic() - started) if slept else 0.0
