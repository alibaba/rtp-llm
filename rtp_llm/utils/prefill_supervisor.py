"""Keep a resident Engram owner across local Prefill worker restarts."""

import argparse
import ctypes
import hashlib
import json
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

MODE_ENV = "RTP_LLM_IN_CONTAINER_RESTART"
OWNER_ARGS_ENV = "RTP_LLM_RESIDENT_OWNER_ARGS"
WORKER_CGROUP_ENV = "RTP_LLM_WORKER_CGROUP"
STOP_TIMEOUT_ENV = "RTP_LLM_SUPERVISOR_STOP_TIMEOUT_S"
STARTUP_TIMEOUT_ENV = "RTP_LLM_PREFILL_STARTUP_TIMEOUT_S"
READY_FILE_ENV = "RTP_LLM_STARTUP_WARMUP_HEALTH_GATE_FILE"
OWNER_MODULE = "rtp_llm.models_py.modules.dsv41.engram_distributed.resident"
_MODEL_METADATA_LIMIT = 16 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)
_POLL_S = 0.1
_REAP_TIMEOUT_S = 10
_STABLE_S = 600
_REAP_TIMEOUT_ERRORS = (TimeoutError, subprocess.TimeoutExpired)
_CANONICALIZE_SENTINEL = "__RTP_CANONICAL_OPTIONS__"
_CANONICALIZE_ERROR = "__RTP_CANONICAL_ERROR__"
_CANONICALIZE_CODE = f"""
import json
import sys

from rtp_llm.server.server_args.server_args import canonicalize_server_options

payload = json.load(sys.stdin)
try:
    result = canonicalize_server_options(payload["args"], payload["options"])
except ValueError as error:
    print("{_CANONICALIZE_ERROR}" + str(error))
    raise SystemExit(2)
print("{_CANONICALIZE_SENTINEL}" + json.dumps(result))
"""


class BootstrapError(ValueError):
    """Bounded bootstrap failure with a safe, loggable code."""

    def __init__(self, code, message=None):
        self.code = code
        super().__init__(message or code)


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


def _reap_adopted_zombies(*managed):
    """Reap exited direct children, including helpers that created a session."""
    if sys.platform != "linux":
        return
    protected = {process.pid for process in managed if process is not None}
    reaped = 0
    # Bound each pass so a stream of exiting helpers cannot starve supervision.
    for _ in range(256):
        try:
            child = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            break
        if child is None or child.si_pid in protected:
            # Their exit status and unreaped process-group leader are still
            # needed by _exited() and _kill_and_reap(). The supervision loop
            # handles them before attempting another adopted-child sweep.
            break
        try:
            exited_pid, _ = os.waitpid(child.si_pid, os.WNOHANG)
            reaped += bool(exited_pid)
        except ChildProcessError:
            continue
    if reaped:
        _LOGGER.info("prefill event=adopted_children_reaped count=%d", reaped)


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


def _kill_adopted_worker_children():
    """Empty a worker-only subreaper, including descendants that called setsid."""
    count = 0
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if pid:
            continue
        # A direct child cannot reuse its PID before this process reaps it.
        # Repeat the snapshot because reparenting can expose more children.
        for child in children_file.read_text().split():
            try:
                os.kill(int(child), signal.SIGKILL)
                count += 1
            except ProcessLookupError:
                pass
        time.sleep(_POLL_S)
    if count:
        _LOGGER.info("prefill event=detached_worker_children_killed signals=%d", count)


def _guard_worker(command):
    """Contain one worker tree; the Owner and FUSE helpers are siblings."""
    _enable_subreaper()
    requested = None

    def request_stop(signum, _frame):
        nonlocal requested
        if requested != signal.SIGUSR1:
            requested = signum

    signals = {signal.SIGTERM, signal.SIGUSR1}
    for sig in signals:
        signal.signal(sig, request_stop)
    signal.pthread_sigmask(signal.SIG_UNBLOCK, signals)
    if requested is not None:
        return

    gate = Path(os.environ[READY_FILE_ENV])
    try:
        worker = subprocess.Popen(command, start_new_session=True)
    except OSError:
        _LOGGER.exception("prefill event=worker_spawn_failed; retrying")
        return
    _LOGGER.info(
        "prefill event=worker_started pid=%d guard_pid=%d", worker.pid, os.getpid()
    )
    forwarded_term = False
    try:
        while not _exited(worker) and requested != signal.SIGUSR1:
            if requested == signal.SIGTERM and not forwarded_term:
                try:
                    os.kill(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                forwarded_term = True
            _reap_adopted_zombies(worker)
            time.sleep(_POLL_S)
    finally:
        try:
            gate.unlink(missing_ok=True)
        finally:
            while True:
                try:
                    _kill_and_reap(worker)
                    break
                except _REAP_TIMEOUT_ERRORS:
                    _LOGGER.warning(
                        "prefill event=worker_cleanup_pending pid=%d", worker.pid
                    )
            _kill_adopted_worker_children()
            _LOGGER.info(
                "prefill event=worker_exited pid=%d rc=%s",
                worker.pid,
                worker.returncode,
            )


def _spawn_worker_guard(command, env):
    signals = {signal.SIGTERM, signal.SIGUSR1}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, signals)
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "rtp_llm.utils.prefill_supervisor", *command],
            env=env,
            start_new_session=True,
        )
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


class PrefillSupervisor:
    def __init__(
        self, worker_cmd, worker_env, owner, stop_timeout, startup_timeout=7200
    ):
        self.worker_cmd = worker_cmd
        self.worker_env = worker_env
        self.owner = owner
        self.stop_timeout = stop_timeout
        self.startup_timeout = startup_timeout
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
            _reap_adopted_zombies(self.owner)
            if time.monotonic() >= deadline:
                return True
            time.sleep(_POLL_S)
        return False

    def _stop_worker(self, guard):
        if not _exited(guard):
            if self.stopping:
                os.kill(guard.pid, signal.SIGTERM)
                try:
                    guard.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    _LOGGER.warning("prefill event=worker_stop_grace_expired")
            if guard.returncode is None and not _exited(guard):
                os.kill(guard.pid, signal.SIGUSR1)
        while guard.returncode is None:
            try:
                guard.wait(timeout=_REAP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _LOGGER.warning(
                    "prefill event=worker_cleanup_pending; holding readiness"
                )
                if self.stopping or self._owner_exited():
                    raise
        if guard.returncode:
            raise OSError(
                f"worker guard exited before confirming cleanup: {guard.returncode}"
            )

    def _run_worker(self, gate):
        started = time.monotonic()
        cgroup = self.worker_env.get(WORKER_CGROUP_ENV)
        oom_before = _worker_oom_kills(cgroup)
        try:
            guard = _spawn_worker_guard(
                self.worker_cmd, {**self.worker_env, READY_FILE_ENV: str(gate)}
            )
        except OSError:
            _LOGGER.exception("prefill event=worker_spawn_failed; retrying")
            return
        _LOGGER.info(
            "prefill event=worker_guard_started guard_pid=%s generation=%s",
            guard.pid,
            self.worker_env.get("RTP_ENGRAM_RESIDENT_GENERATION", "unset"),
        )
        reason = "supervision_error"
        ready = False
        was_ready = False
        try:
            while not self.stopping and not self._owner_exited() and not _exited(guard):
                _reap_adopted_zombies(self.owner, guard)
                current_ready = gate.is_file()
                was_ready = was_ready or current_ready
                if current_ready != ready:
                    _LOGGER.info(
                        "prefill event=%s guard_pid=%d elapsed_s=%.3f",
                        "worker_ready" if current_ready else "worker_unready",
                        guard.pid,
                        time.monotonic() - started,
                    )
                    ready = current_ready
                if not was_ready and time.monotonic() - started >= self.startup_timeout:
                    reason = "startup_timeout"
                    break
                time.sleep(_POLL_S)
            if self.stopping:
                reason = "supervisor_stop"
            elif self._owner_exited():
                reason = "owner_exit"
            elif reason != "startup_timeout":
                reason = "worker_exit"
        finally:
            _LOGGER.info(
                "prefill event=worker_guard_stopping guard_pid=%s reason=%s signal=%s",
                guard.pid,
                reason,
                self.stop_signal,
            )
            try:
                gate.unlink(missing_ok=True)
            finally:
                self._stop_worker(guard)
        oom_after = _worker_oom_kills(cgroup) if oom_before is not None else None
        _LOGGER.info(
            "prefill event=worker_guard_exited guard_pid=%s rc=%s oom_kill_delta=%s",
            guard.pid,
            guard.returncode,
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


def _canonicalize_server_options(args, options):
    """Resolve abbreviations in a child; the supervisor must not import torch."""
    payload = json.dumps({"args": list(args), "options": list(options)})
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CANONICALIZE_CODE],
            input=payload,
            text=True,
            capture_output=True,
            timeout=120,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BootstrapError(
            "SERVER_OPTION_CANONICALIZATION_FAILED",
            "failed to resolve abbreviated server options",
        ) from error
    for line in completed.stdout.splitlines():
        if line.startswith(_CANONICALIZE_ERROR):
            raise BootstrapError(
                "AMBIGUOUS_SERVER_OPTION", line[len(_CANONICALIZE_ERROR) :]
            )
        if line.startswith(_CANONICALIZE_SENTINEL):
            return json.loads(line[len(_CANONICALIZE_SENTINEL) :])
    raise BootstrapError(
        "SERVER_OPTION_CANONICALIZATION_FAILED",
        f"failed to resolve abbreviated server options (rc={completed.returncode})",
    )


def _resolve_worker_args(env, args):
    """Move the launcher's mount and GPU-count options to canonical env keys."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mount_options = ("--checkpoint_path", "--tokenizer_path", "--sp_checkpoint_path")
    gpu_options = ("--world_size", "--local_world_size")
    for option in mount_options:
        parser.add_argument(option)
    for option in gpu_options:
        parser.add_argument(option, type=int)
    resident_options = mount_options + gpu_options
    if any(
        option.startswith("--")
        and option not in resident_options
        and any(full.startswith(option) for full in resident_options)
        for option in (arg.partition("=")[0] for arg in args)
    ):
        args = _canonicalize_server_options(args, resident_options)
    options, remaining = parser.parse_known_args(args)
    return {
        **env,
        **{
            name.upper(): str(value)
            for name, value in vars(options).items()
            if value is not None
        },
    }, remaining


def _positive_int(value):
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise BootstrapError("INVALID_WORKER_WORLD_SIZE") from None
    if number < 1:
        raise BootstrapError("INVALID_WORKER_WORLD_SIZE")
    return number


def _validate_worker_preconditions(env):
    """Reject the full-world misconfiguration before allocating the Owner."""
    world_size = _positive_int(env.get("WORLD_SIZE"))
    local_world_size = _positive_int(env.get("LOCAL_WORLD_SIZE"))
    # LOCAL_WORLD_SIZE=1 is the worker's "derive from visible GPUs" sentinel.
    # The supervisor cannot see the device count without importing torch, so
    # leave that case to the worker's final topology check.
    if (
        world_size is not None
        and local_world_size is not None
        and local_world_size > 1
        and world_size != local_world_size
    ):
        raise BootstrapError(
            "IN_CONTAINER_RESTART_REQUIRES_FULL_WORLD",
            "in-container restart requires the full Prefill world",
        )


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
    """Resolve startup identity once for the Owner and all Worker attempts."""
    from rtp_llm.models_py.modules.dsv41.engram_distributed.resident_network import (
        STATE_ENV,
        TRANSPORT_ENV,
        c2_topology,
    )

    result = dict(env)
    if "RTP_ENGRAM_RESIDENT_DIR" in result or result.get(TRANSPORT_ENV, "tcp") != "tcp":
        raise BootstrapError("RESIDENT_C2_TCP_REQUIRED")
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
    generation_env = "RTP_ENGRAM_RESIDENT_GENERATION"
    generation, peers, _ = c2_topology(result)
    if result.setdefault(generation_env, generation) != generation:
        raise BootstrapError("RESIDENT_C2_GANG_IDENTITY_MISMATCH")
    if "RTP_ENGRAM_MODEL_REVISION" not in result:
        result["RTP_ENGRAM_MODEL_REVISION"] = _model_manifest_revision(
            result["RTP_ENGRAM_CHECKPOINT"]
        )
    revision = result["RTP_ENGRAM_MODEL_REVISION"]
    if not 0 < len(revision) <= 256:
        raise ValueError("invalid resident model revision")
    if len(peers) > 1:
        result.setdefault("RTP_ENGRAM_STORAGE", "host_shared")
    _LOGGER.info(
        "prefill event=resident_identity_ready generation=%s revision=%s identity_source=c2_gang",
        generation,
        revision,
    )
    return result


def _supervisor_tmp_root():
    shm = Path("/dev/shm")
    try:
        info = shm.lstat()
    except OSError:
        return None
    if stat.S_ISDIR(info.st_mode) and os.access(shm, os.W_OK):
        return str(shm)
    return None


def _hold_after_bootstrap_failure(supervisor, error):
    """Keep the entry process alive and unready after a bootstrap failure."""
    _LOGGER.error(
        "prefill event=bootstrap_failed error_class=%s code=%s errno=%s; holding container",
        type(error).__name__,
        getattr(error, "code", type(error).__name__),
        getattr(error, "errno", None),
    )
    while not supervisor.stopping:
        time.sleep(_POLL_S)


def supervise_prefill(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s pid=%(process)d %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _enable_subreaper()
    worker_env = {**os.environ, MODE_ENV: "worker"}
    stop_timeout = int(os.environ.get(STOP_TIMEOUT_ENV, "300"))
    if not 1 <= stop_timeout <= 3600:
        raise ValueError(f"{STOP_TIMEOUT_ENV} must be between 1 and 3600")
    startup_timeout = int(os.environ.get(STARTUP_TIMEOUT_ENV, "7200"))
    if not 1 <= startup_timeout <= 86400:
        raise ValueError(f"{STARTUP_TIMEOUT_ENV} must be between 1 and 86400")
    worker_env.setdefault("RTP_ENGRAM_RESIDENT_TRANSPORT", "tcp")
    owner_args = shlex.split(worker_env.pop(OWNER_ARGS_ENV, "--discover --device 0"))
    worker_cmd = [sys.executable, "-m", "rtp_llm.start_server", *args]
    supervisor = PrefillSupervisor(
        worker_cmd, worker_env, None, stop_timeout, startup_timeout
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, supervisor.request_stop)
    _LOGGER.info(
        "prefill event=supervisor_started stop_timeout_s=%s generation=%s instance=%s pid=%d transport=tcp",
        stop_timeout,
        worker_env.get("RTP_ENGRAM_RESIDENT_GENERATION", "auto"),
        socket.gethostname(),
        os.getpid(),
    )
    try:
        try:
            tmp_root = _supervisor_tmp_root()
            if tmp_root is None:
                _LOGGER.warning(
                    "prefill event=supervisor_tmpdir_fallback; use a container-local TMPDIR"
                )
            with tempfile.TemporaryDirectory(
                prefix="rtp-prefill-", dir=tmp_root
            ) as directory:
                try:
                    worker_env, remaining = _resolve_worker_args(worker_env, args)
                    worker_env = _prepare_model_mounts(worker_env)
                    _validate_worker_preconditions(worker_env)
                    worker_env = _prepare_network_identity(worker_env, Path(directory))
                    supervisor.worker_env = worker_env
                    supervisor.worker_cmd = [*worker_cmd[:3], *remaining]
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
                    # Keep the entry process alive and unready when bootstrap
                    # fails. Never log env/secrets; only the safe error code.
                    _hold_after_bootstrap_failure(supervisor, error)
                finally:
                    if supervisor.owner is not None:
                        _stop(supervisor.owner, _REAP_TIMEOUT_S)
                        _LOGGER.info(
                            "prefill event=owner_stopped pid=%s rc=%s",
                            supervisor.owner.pid,
                            supervisor.owner.returncode,
                        )
        except Exception as error:
            # Temp-dir validation/creation is part of bootstrap: a failure
            # must not terminate the entry process and trigger a gang recycle.
            _hold_after_bootstrap_failure(supervisor, error)
    finally:
        _LOGGER.info(
            "prefill event=supervisor_stopped signal=%s", supervisor.stop_signal
        )
        from rtp_llm.utils.fuser import umount_all

        umount_all()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s pid=%(process)d %(message)s",
    )
    _guard_worker(sys.argv[1:])
