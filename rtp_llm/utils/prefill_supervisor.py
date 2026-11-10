"""Keep a resident Engram owner across local Prefill worker restarts."""

import ctypes
import hashlib
import logging
import os
import shlex
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from uuid import UUID

MODE_ENV = "RTP_LLM_IN_CONTAINER_RESTART"
OWNER_ARGS_ENV = "RTP_LLM_RESIDENT_OWNER_ARGS"
WORKER_CGROUP_ENV = "RTP_LLM_WORKER_CGROUP"
STOP_TIMEOUT_ENV = "RTP_LLM_SUPERVISOR_STOP_TIMEOUT_S"
READY_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE"
OWNER_MODULE = "rtp_llm.models_py.modules.dsv41.engram_distributed.resident"
DEFAULT_OWNER_KEY_FILE = "/mnt/fuse/.rtp-engram-local/owner.key"
_MODEL_METADATA_LIMIT = 16 * 1024 * 1024
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
    if any(entry.is_dir() for entry in group.iterdir()):
        raise ValueError("worker cgroup must be a leaf")
    processes = group / "cgroup.procs"
    if processes.read_text().strip():
        raise ValueError("worker cgroup must be empty before starting the worker")
    if (group / "memory.max").read_text().strip() == "max" or (
        group / "memory.oom.group"
    ).read_text().strip() != "1":
        raise ValueError("worker cgroup requires memory.max and memory.oom.group=1")
    current = Path("/proc/self/cgroup")
    previous = current.read_text()
    processes.write_text("0\n")
    if current.read_text() == previous:
        raise ValueError("worker cgroup must be separate from supervisor and owner")


def _worker_oom_kills(path):
    if not path:
        return None
    try:
        events = (Path(path) / "memory.events").read_text().splitlines()
        return int(dict(event.split() for event in events)["oom_kill"])
    except (OSError, KeyError, ValueError):
        _LOGGER.warning("prefill event=worker_oom_counter_unavailable")
        return None


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
    return (
        os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        is not None
    )


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
        if not _exited(process):
            _LOGGER.warning(
                "prefill event=stop_grace_expired pid=%s grace_s=%s",
                process.pid,
                grace,
            )
    _kill_and_reap(process)


class PrefillSupervisor:
    def __init__(self, worker_cmd, worker_env, owner, stop_timeout):
        self.worker_cmd = worker_cmd
        self.worker_env = worker_env
        self.owner = owner
        self.stop_timeout = stop_timeout
        self.stopping = False
        self.stop_signal = None

    def request_stop(self, signum, _frame):
        self.stop_signal = signum
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
        started = time.monotonic()
        cgroup = self.worker_env.get(WORKER_CGROUP_ENV)
        oom_before = _worker_oom_kills(cgroup)
        try:
            worker = subprocess.Popen(
                self.worker_cmd,
                env={**self.worker_env, READY_FILE_ENV: str(gate)},
                start_new_session=True,
            )
        except OSError:
            _LOGGER.exception("prefill event=worker_spawn_failed; retrying")
            return
        _LOGGER.info(
            "prefill event=worker_started pid=%s generation=%s",
            worker.pid,
            self.worker_env.get("RTP_ENGRAM_RESIDENT_GENERATION", "unset"),
        )
        reason = "supervision_error"
        ready = False
        try:
            while (
                not self.stopping and not self._owner_exited() and not _exited(worker)
            ):
                current_ready = gate.is_file()
                if current_ready != ready:
                    _LOGGER.info(
                        "prefill event=%s pid=%d elapsed_s=%.3f",
                        "worker_ready" if current_ready else "worker_unready",
                        worker.pid,
                        time.monotonic() - started,
                    )
                    ready = current_ready
                time.sleep(_POLL_S)
            if self.stopping:
                reason = "supervisor_stop"
            elif self._owner_exited():
                reason = "owner_exit"
            else:
                reason = "worker_exit"
        finally:
            _LOGGER.info(
                "prefill event=worker_stopping pid=%s reason=%s signal=%s",
                worker.pid,
                reason,
                self.stop_signal,
            )
            try:
                gate.unlink(missing_ok=True)
            finally:
                self._stop_worker(worker)
        oom_after = _worker_oom_kills(cgroup) if oom_before is not None else None
        _LOGGER.info(
            "prefill event=worker_exited pid=%s rc=%s oom_kill_delta=%s",
            worker.pid,
            worker.returncode,
            oom_after - oom_before if oom_after is not None else None,
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
                if not self.stopping and not self._owner_exited():
                    _LOGGER.info(
                        "prefill event=worker_restart_wait delay_s=%s", backoff
                    )
                if not self._wait(backoff):
                    break
                backoff = min(backoff * 2, 60)
            if not self.stopping and self._owner_exited():
                owner = self.owner
                self.owner = None
                _kill_and_reap(owner)
                _LOGGER.error(
                    "prefill event=owner_exited pid=%s rc=%s",
                    owner.pid,
                    owner.returncode,
                )
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            _LOGGER.exception(
                "prefill event=worker_supervision_failed; holding container"
            )
        if not self.stopping:
            # A dead owner cannot recreate the same Fabric generation.
            _LOGGER.error("prefill event=recovery_blocked; awaiting platform stop")
            while not self.stopping:
                time.sleep(_POLL_S)


def _prepare_model_mounts(env):
    """Keep shared model mounts owned by the supervisor, outside worker cleanup.

    All model paths handed to children are local. Children therefore never
    register these mounts in their process-local Fuser cleanup map.
    """
    from rtp_llm.utils.fuser import fetch_remote_file_to_local

    mounted = {}
    result = dict(env)
    for name in (
        "CHECKPOINT_PATH",
        "TOKENIZER_PATH",
        "SP_CHECKPOINT_PATH",
        "RTP_ENGRAM_CHECKPOINT",
    ):
        source = env.get(name)
        if not source:
            continue
        if source not in mounted:
            mounted[source] = fetch_remote_file_to_local(source)
        result[name] = mounted[source]
    result.setdefault("RTP_ENGRAM_CHECKPOINT", result.get("CHECKPOINT_PATH", ""))
    if not result["RTP_ENGRAM_CHECKPOINT"]:
        raise ValueError("resident Owner requires a checkpoint")
    _LOGGER.info("prefill event=model_mounts_ready unique_paths=%d", len(mounted))
    return result


def _model_manifest_revision(checkpoint):
    """Hash bounded model metadata, not the full weight tensor contents."""
    digest = hashlib.sha256(b"rtp-engram-model-manifest-v1\0")
    for name in ("config.json", "model.safetensors.index.json"):
        with os.fdopen(
            os.open(Path(checkpoint) / name, os.O_RDONLY | os.O_NONBLOCK), "rb"
        ) as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MODEL_METADATA_LIMIT:
                raise ValueError("invalid or oversized Engram model metadata")
            data = source.read(_MODEL_METADATA_LIMIT + 1)
        if len(data) > _MODEL_METADATA_LIMIT:
            raise ValueError("oversized Engram model metadata")
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    return "manifest@" + digest.hexdigest()


def _prepare_network_identity(env, directory):
    """Resolve opt-in defaults once; workers only see the immutable local key."""
    from rtp_llm.models_py.modules.dsv41.engram_distributed.resident_network import (
        KEY_ENV,
        STATE_ENV,
        read_key_file,
    )

    result = dict(env)
    if "RTP_ENGRAM_RESIDENT_DIR" in result:
        raise ValueError("tcp resident transport does not use a shared directory")
    state = Path(result.get(STATE_ENV, str(directory / "owner-state")))
    if not state.is_absolute():
        raise ValueError("resident local state requires an absolute path")
    state.mkdir(mode=0o700, exist_ok=True)
    info = state.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.geteuid()
        or info.st_mode & 0o077
    ):
        raise ValueError("resident local state must be private and owned by this user")
    result[STATE_ENV] = str(state)
    key = read_key_file(result.get(KEY_ENV, DEFAULT_OWNER_KEY_FILE))
    snapshot = directory / "owner.key"
    with os.fdopen(
        os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "wb",
    ) as target:
        target.write(key)
    result[KEY_ENV] = str(snapshot)
    if "RTP_ENGRAM_RESIDENT_GENERATION" not in result:
        identity = hashlib.sha256(
            b"rtp-engram-resident-generation-v1\0" + key
        ).digest()[:16]
        result["RTP_ENGRAM_RESIDENT_GENERATION"] = str(UUID(bytes=identity, version=5))
    generation = result["RTP_ENGRAM_RESIDENT_GENERATION"]
    if str(UUID(generation)) != generation:
        raise ValueError("invalid resident generation")
    if "RTP_ENGRAM_MODEL_REVISION" not in result:
        result["RTP_ENGRAM_MODEL_REVISION"] = _model_manifest_revision(
            result["RTP_ENGRAM_CHECKPOINT"]
        )
    revision = result["RTP_ENGRAM_MODEL_REVISION"]
    if not 0 < len(revision) <= 256:
        raise ValueError("invalid resident model revision")
    count = result.get("ENGRAM_SHARD_COUNT", "")
    if (
        "RTP_ENGRAM_STORAGE" not in result
        and count.isascii()
        and count.isdecimal()
        and int(count) > 1
    ):
        result["RTP_ENGRAM_STORAGE"] = "host_shared"
    _LOGGER.info(
        "prefill event=resident_identity_ready generation=%s revision=%s identity_source=%s",
        generation,
        revision,
        "explicit" if "RTP_ENGRAM_RESIDENT_GENERATION" in env else "group_key",
    )
    return result


def supervise_prefill(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _enable_subreaper()
    worker_env = {**os.environ, MODE_ENV: "worker"}
    worker_env.setdefault("RTP_LLM_PREFILL_WATCHDOG", "1")
    stop_timeout = int(os.environ.get(STOP_TIMEOUT_ENV, "300"))
    if not 1 <= stop_timeout <= 3600:
        raise ValueError(f"{STOP_TIMEOUT_ENV} must be between 1 and 3600")
    transport = worker_env.setdefault(
        "RTP_ENGRAM_RESIDENT_TRANSPORT",
        "file" if "RTP_ENGRAM_RESIDENT_DIR" in worker_env else "tcp",
    )
    if transport not in ("file", "tcp"):
        raise ValueError("invalid resident transport")
    if transport == "file" and (
        not worker_env.get("RTP_ENGRAM_RESIDENT_GENERATION")
        or not worker_env.get("RTP_ENGRAM_RESIDENT_DIR")
    ):
        raise ValueError("in-container Prefill restart requires resident Engram")
    owner_args = shlex.split(
        worker_env.pop(
            OWNER_ARGS_ENV, "--discover --device 0" if transport == "tcp" else ""
        )
    )
    worker_cmd = [sys.executable, "-m", "rtp_llm.start_server", *args]
    supervisor = PrefillSupervisor(worker_cmd, worker_env, None, stop_timeout)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, supervisor.request_stop)
    _LOGGER.info(
        "prefill event=supervisor_started stop_timeout_s=%s generation=%s instance=%s pid=%d transport=%s",
        stop_timeout,
        worker_env.get("RTP_ENGRAM_RESIDENT_GENERATION", "auto"),
        socket.gethostname(),
        os.getpid(),
        transport,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="rtp-prefill-") as directory:
            try:
                if transport == "tcp":
                    worker_env = _prepare_model_mounts(worker_env)
                    worker_env = _prepare_network_identity(worker_env, Path(directory))
                    supervisor.worker_env = worker_env
                if owner_args and not supervisor.stopping:
                    supervisor.owner = subprocess.Popen(
                        [sys.executable, "-m", OWNER_MODULE, *owner_args],
                        env=worker_env,
                        start_new_session=True,
                    )
                    _LOGGER.info(
                        "prefill event=owner_started pid=%s", supervisor.owner.pid
                    )
                supervisor.run(Path(directory))
            except Exception as error:
                # Keep the platform entry process alive when bootstrap fails.
                # No worker may advertise readiness; an allocated Owner stays
                # alive until an explicit platform stop. Never log env/secrets.
                _LOGGER.error(
                    "prefill event=bootstrap_failed error_class=%s; holding container",
                    type(error).__name__,
                )
                while not supervisor.stopping:
                    time.sleep(_POLL_S)
            finally:
                if supervisor.owner is not None:
                    _stop(supervisor.owner, _REAP_TIMEOUT_S)
                    _LOGGER.info(
                        "prefill event=owner_stopped pid=%s rc=%s",
                        supervisor.owner.pid,
                        supervisor.owner.returncode,
                    )
    finally:
        _LOGGER.info(
            "prefill event=supervisor_stopped signal=%s", supervisor.stop_signal
        )
        if transport == "tcp":
            from rtp_llm.utils.fuser import umount_all

            umount_all()
