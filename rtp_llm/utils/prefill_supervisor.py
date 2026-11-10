"""Keep a resident Engram owner across local Prefill worker restarts."""

import ctypes
import logging
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODE_ENV = "RTP_LLM_IN_CONTAINER_RESTART"
OWNER_ARGS_ENV = "RTP_LLM_RESIDENT_OWNER_ARGS"
WORKER_CGROUP_ENV = "RTP_LLM_WORKER_CGROUP"
STOP_TIMEOUT_ENV = "RTP_LLM_SUPERVISOR_STOP_TIMEOUT_S"
READY_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE"
OWNER_MODULE = "rtp_llm.models_py.modules.dsv41.engram_distributed.resident"
_LOGGER = logging.getLogger(__name__)
_POLL_S = 0.1
_REAP_TIMEOUT_S = 10
_STABLE_S = 600
_REAP_TIMEOUT_ERRORS = (TimeoutError, subprocess.TimeoutExpired)


def enter_worker_cgroup():
    """Join the ASI-delegated worker leaf before importing torch."""
    path = os.environ.get(WORKER_CGROUP_ENV)
    if not path:
        return
    group = Path(path).resolve(strict=True)
    if Path("/sys/fs/cgroup") not in group.parents:
        raise ValueError(f"{WORKER_CGROUP_ENV} must name a cgroup v2 leaf")
    if (
        (group / "memory.max").read_text().strip() == "max"
        or (group / "memory.oom.group").read_text().strip() != "1"
    ):
        raise ValueError("worker cgroup requires memory.max and memory.oom.group=1")
    current = Path("/proc/self/cgroup")
    previous = current.read_text()
    (group / "cgroup.procs").write_text("0\n")
    if current.read_text() == previous:
        raise ValueError("worker cgroup must be separate from supervisor and owner")


def _enable_subreaper():
    if sys.platform != "linux":
        raise RuntimeError("in-container Prefill restart requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    # Adopt orphaned ranks even when the container shares its PID namespace.
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _exited(process):
    # Keep the session leader unreaped until killpg has targeted its group.
    return os.waitid(
        os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
    ) is not None


def _kill_and_reap(process):
    deadline = time.monotonic() + _REAP_TIMEOUT_S
    if process.returncode is None:
        # The unreaped leader pins the group ID. After reaping it, only
        # wait for children already targeted by the original SIGKILL.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=_REAP_TIMEOUT_S)
    while True:
        try:
            pid, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Prefill process group {process.pid} did not exit")
            time.sleep(_POLL_S)


def _stop(process, grace):
    if not _exited(process) and grace:
        try:
            os.kill(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + grace
        while not _exited(process) and time.monotonic() < deadline:
            time.sleep(_POLL_S)
    _kill_and_reap(process)


class PrefillSupervisor:
    def __init__(self, worker_cmd, worker_env, owner, stop_timeout):
        self.worker_cmd = worker_cmd
        self.worker_env = worker_env
        self.owner = owner
        self.stop_timeout = stop_timeout
        self.stopping = False

    def request_stop(self, *_):
        self.stopping = True

    def _owner_exited(self):
        return self.owner is not None and _exited(self.owner)

    def _wait(self, delay):
        deadline = time.monotonic() + delay
        while not self.stopping and not self._owner_exited():
            if time.monotonic() >= deadline:
                return True
            time.sleep(_POLL_S)
        return False

    def _stop_worker(self, worker):
        try:
            _stop(worker, self.stop_timeout if self.stopping else 0)
            return
        except _REAP_TIMEOUT_ERRORS:
            if self.stopping or self._owner_exited():
                raise
            _LOGGER.warning(
                "prefill event=worker_cleanup_pending pid=%s; holding readiness",
                worker.pid,
            )
        while not self.stopping and not self._owner_exited():
            try:
                _kill_and_reap(worker)
                return
            except _REAP_TIMEOUT_ERRORS:
                continue

    def _run_worker(self, gate):
        try:
            worker = subprocess.Popen(
                self.worker_cmd,
                env={**self.worker_env, READY_FILE_ENV: str(gate)},
                start_new_session=True,
            )
        except OSError:
            _LOGGER.exception("prefill event=worker_spawn_failed; retrying")
            return
        _LOGGER.info("prefill event=worker_started pid=%s", worker.pid)
        try:
            while not self.stopping and not self._owner_exited() and not _exited(worker):
                time.sleep(_POLL_S)
        finally:
            try:
                gate.unlink(missing_ok=True)
            finally:
                self._stop_worker(worker)
        _LOGGER.info(
            "prefill event=worker_exited pid=%s rc=%s", worker.pid, worker.returncode
        )

    def run(self, directory):
        backoff = 1
        gate = directory / "ready"
        try:
            while not self.stopping and not self._owner_exited():
                started = time.monotonic()
                gate.unlink(missing_ok=True)
                self._run_worker(gate)
                if time.monotonic() - started >= _STABLE_S:
                    backoff = 1
                if not self._wait(backoff):
                    break
                backoff = min(backoff * 2, 60)
            if not self.stopping and self._owner_exited():
                owner = self.owner
                self.owner = None
                _kill_and_reap(owner)
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            _LOGGER.exception("prefill event=worker_supervision_failed; holding container")
        if not self.stopping:
            # A dead owner cannot recreate the same Fabric generation.
            _LOGGER.error("prefill event=recovery_blocked; awaiting platform stop")
            while not self.stopping:
                time.sleep(_POLL_S)


def supervise_prefill(args):
    logging.basicConfig(level=logging.INFO)
    _enable_subreaper()
    worker_env = {**os.environ, MODE_ENV: "worker"}
    worker_env.setdefault("RTP_LLM_PREFILL_WATCHDOG", "1")
    owner_args = shlex.split(worker_env.pop(OWNER_ARGS_ENV, ""))
    stop_timeout = int(os.environ.get(STOP_TIMEOUT_ENV, "300"))
    if not 1 <= stop_timeout <= 3600:
        raise ValueError(f"{STOP_TIMEOUT_ENV} must be between 1 and 3600")
    if not all(
        worker_env.get(name)
        for name in ("RTP_ENGRAM_RESIDENT_DIR", "RTP_ENGRAM_RESIDENT_GENERATION")
    ):
        raise ValueError("in-container Prefill restart requires resident Engram")
    worker_cmd = [sys.executable, "-m", "rtp_llm.start_server", *args]
    supervisor = PrefillSupervisor(worker_cmd, worker_env, None, stop_timeout)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, supervisor.request_stop)
    try:
        if owner_args and not supervisor.stopping:
            supervisor.owner = subprocess.Popen(
                [sys.executable, "-m", OWNER_MODULE, *owner_args],
                start_new_session=True,
            )
        with tempfile.TemporaryDirectory(prefix="rtp-prefill-") as directory:
            supervisor.run(Path(directory))
    finally:
        if supervisor.owner is not None:
            _stop(supervisor.owner, _REAP_TIMEOUT_S)
