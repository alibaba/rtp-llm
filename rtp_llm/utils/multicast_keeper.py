"""Node-local lifecycle manager for the Level3 multicast keeper.

The holder is a CUDA-free service process.  It must outlive checkpointed
backend ranks, but it must never be part of their checkpoint process tree.
This module intentionally launches the holder ELF directly; no shell launcher
is part of the runtime contract.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

_LOGGER = logging.getLogger(__name__)

ENABLE_ENV = "RTP_LLM_CUDA_CKPT_MULTICAST_KEEPER"
HOLDER_ENV = "RTP_LLM_MC_HOLDER_BIN"
CREATOR_ENV = "RTP_LLM_MC_CREATOR_BIN"
SHIM_ENV = "RTP_LLM_MC_SHIM"
BIN_DIR_ENV = "RTP_LLM_MC_KEEPER_BIN_DIR"
GPU_ENV = "RTP_LLM_MC_KEEPER_GPUS"
LOCAL_GPU_ENV = "RTP_LLM_MC_LOCAL_GPUS"
FABRIC_TEAM_ENV = "RTP_LLM_MC_FABRIC_TEAM_SIZE"
SOCKET_ENV = "RTP_LLM_CUDA_CKPT_MULTICAST_SOCKET"
KEEPER_DIR_ENV = "NEKYIA_KEEPER_DIR"

_ARTIFACT_SUBDIR = Path("cpp/cuda_checkpoint/multicast_keeper")
_BAZEL_ARTIFACT_SUBDIR = Path("rtp_llm") / _ARTIFACT_SUBDIR
_ARTIFACT_NAMES = {
    "holder": "keeper_lite_holder",
    "creator": "keeper_lite_creator",
    "shim": "mc_shim_unified.so",
}

_PROTOCOL_MAGIC = 0x3250434D505452
_PROTOCOL_VERSION = 3
_PING_OPCODE = 1
_STATUS_OK = 0
_REQUEST = struct.Struct("<QHHIQQQQIIQ")
_RESPONSE = struct.Struct("<QHHIiIQQQQQIIQ")
_UNIX_SOCKET_PATH_LIMIT = 107


class MulticastKeeperError(RuntimeError):
    """Base exception for keeper configuration and lifecycle failures."""


class MulticastKeeperConfigError(MulticastKeeperError):
    """The opt-in keeper configuration is invalid."""


class MulticastKeeperHealthError(MulticastKeeperError):
    """The holder process or its protocol identity is unhealthy."""


class MulticastKeeperMode(str, Enum):
    """Transport selected from the global and node-local worker topology."""

    SINGLE_NODE_POSIX = "single_node_posix"
    CROSS_NODE_FABRIC = "cross_node_fabric"


@dataclass(frozen=True)
class KeeperArtifacts:
    holder: Path
    creator: Path
    shim: Path


@dataclass(frozen=True)
class KeeperHealth:
    instance_hi: int
    instance_lo: int
    entries: int
    local_device_count: int

    @property
    def instance(self) -> Tuple[int, int]:
        return (self.instance_hi, self.instance_lo)


def is_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Return whether the existing explicit Level3 marker is exactly ``1``."""

    source = os.environ if env is None else env
    return source.get(ENABLE_ENV) == "1"


def parse_gpu_list(value: str, *, source: str = GPU_ENV) -> Tuple[int, ...]:
    """Parse a physical CUDA ordinal list, rejecting UUIDs and duplicates."""

    if not value or not value.strip():
        raise MulticastKeeperConfigError(f"{source} must be a non-empty GPU list")
    fields = value.split(",")
    if any(not re.fullmatch(r"[0-9]+", field.strip()) for field in fields):
        raise MulticastKeeperConfigError(
            f"{source} must contain only comma-separated integer CUDA ordinals: {value!r}"
        )
    gpus = tuple(int(field.strip()) for field in fields)
    if len(set(gpus)) != len(gpus):
        raise MulticastKeeperConfigError(
            f"{source} contains duplicate CUDA ordinals: {value!r}"
        )
    return gpus


def _artifact_ok(kind: str, path: Path) -> bool:
    if not path.is_file():
        return False
    if kind in ("holder", "creator"):
        return os.access(path, os.X_OK)
    return True


def _checked_artifacts(paths: Mapping[str, Path], origin: str) -> KeeperArtifacts:
    invalid = [
        f"{kind}={path}" for kind, path in paths.items() if not _artifact_ok(kind, path)
    ]
    if invalid:
        raise MulticastKeeperConfigError(
            f"invalid multicast keeper artifacts from {origin}: {', '.join(invalid)}"
        )
    return KeeperArtifacts(
        holder=paths["holder"].resolve(),
        creator=paths["creator"].resolve(),
        shim=paths["shim"].resolve(),
    )


def discover_artifacts(
    env: Optional[Mapping[str, str]] = None,
    *,
    package_root: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> KeeperArtifacts:
    """Resolve holder, creator, and shim without depending on a shell launcher.

    Explicit per-artifact paths take highest priority.  An explicit bin dir is
    the next production override.  Otherwise packaged files under ``rtp_llm``
    are preferred, with source-tree ``bazel-bin`` kept as a developer fallback.
    Explicit configuration is fail-closed and never silently falls back.
    """

    source = dict(os.environ if env is None else env)
    explicit = {
        "holder": source.get(HOLDER_ENV),
        "creator": source.get(CREATOR_ENV),
        "shim": source.get(SHIM_ENV),
    }
    explicit_dir = source.get(BIN_DIR_ENV)

    if any(explicit.values()):
        fallback_dir = Path(explicit_dir) if explicit_dir else None
        missing = [
            kind for kind, value in explicit.items() if not value and not fallback_dir
        ]
        if missing:
            raise MulticastKeeperConfigError(
                "explicit multicast artifact paths must specify all artifacts or "
                f"set {BIN_DIR_ENV}; missing: {', '.join(missing)}"
            )
        paths = {
            kind: Path(value) if value else fallback_dir / _ARTIFACT_NAMES[kind]  # type: ignore[operator]
            for kind, value in explicit.items()
        }
        return _checked_artifacts(paths, "explicit paths")

    if explicit_dir:
        directory = Path(explicit_dir)
        paths = {kind: directory / name for kind, name in _ARTIFACT_NAMES.items()}
        return _checked_artifacts(paths, BIN_DIR_ENV)

    resolved_package_root = (
        Path(package_root)
        if package_root is not None
        else Path(__file__).resolve().parents[1]
    )
    resolved_repo_root = (
        Path(repo_root) if repo_root is not None else resolved_package_root.parent
    )
    directories = (
        ("installed package", resolved_package_root / _ARTIFACT_SUBDIR),
        ("source bazel-bin", resolved_repo_root / "bazel-bin" / _BAZEL_ARTIFACT_SUBDIR),
    )
    attempted = []
    for origin, directory in directories:
        paths = {kind: directory / name for kind, name in _ARTIFACT_NAMES.items()}
        attempted.extend(str(path) for path in paths.values())
        if all(_artifact_ok(kind, path) for kind, path in paths.items()):
            return _checked_artifacts(paths, origin)
    raise MulticastKeeperConfigError(
        "multicast keeper runtime artifacts were not found; tried: "
        + ", ".join(attempted)
    )


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise MulticastKeeperConfigError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise MulticastKeeperConfigError(f"{name} must be a positive integer")
    return value


def _role_name(role: Any) -> str:
    name = getattr(role, "name", None)
    if name is not None:
        return str(name)
    value = getattr(role, "value", role)
    return str(value)


def _preload_entries(value: str) -> Sequence[str]:
    return tuple(entry for entry in re.split(r"[:\s]+", value) if entry)


def _append_preload(value: str, shim: Path) -> str:
    entries = list(_preload_entries(value))
    shim_text = str(shim)
    entries = [entry for entry in entries if entry != shim_text]
    entries.append(shim_text)
    return ":".join(entries)


class MulticastKeeperRuntime:
    """Own exactly one node-local holder for the lifetime of backend ranks."""

    def __init__(
        self,
        world_size: int,
        local_world_size: int,
        role: Any,
        *,
        env: Optional[Mapping[str, str]] = None,
        artifacts: Optional[KeeperArtifacts] = None,
        state_root: Optional[Path] = None,
    ) -> None:
        self._env: Dict[str, str] = dict(os.environ if env is None else env)
        if not is_enabled(self._env):
            raise MulticastKeeperConfigError(
                f"constructing a multicast keeper requires {ENABLE_ENV}=1"
            )
        if world_size <= 0 or local_world_size <= 0:
            raise MulticastKeeperConfigError(
                "world_size and local_world_size must be positive"
            )
        if world_size < local_world_size:
            raise MulticastKeeperConfigError(
                "world_size must be greater than or equal to local_world_size"
            )
        gpu_source = GPU_ENV if self._env.get(GPU_ENV) else "CUDA_VISIBLE_DEVICES"
        gpu_value = self._env.get(GPU_ENV) or self._env.get("CUDA_VISIBLE_DEVICES", "")
        self.gpus = parse_gpu_list(gpu_value, source=gpu_source)
        if len(self.gpus) != local_world_size:
            raise MulticastKeeperConfigError(
                f"{gpu_source} has {len(self.gpus)} GPUs, but local_world_size="
                f"{local_world_size}"
            )

        self.world_size = int(world_size)
        self.local_world_size = int(local_world_size)
        self.role = role
        self.mode = (
            MulticastKeeperMode.CROSS_NODE_FABRIC
            if self.world_size > self.local_world_size
            else MulticastKeeperMode.SINGLE_NODE_POSIX
        )
        self.fabric_team_size = (
            self.world_size
            if self.mode == MulticastKeeperMode.CROSS_NODE_FABRIC
            else None
        )
        self.artifacts = artifacts or discover_artifacts(self._env)
        self._state_root = Path(state_root) if state_root is not None else None

        self.client_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_HOLDER_IO_TIMEOUT_MS", 1000
        )
        self.creator_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_CREATOR_TIMEOUT_MS", 120000
        )
        self.start_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_KEEPER_START_TIMEOUT_MS", 10000
        )
        self.stop_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_KEEPER_STOP_TIMEOUT_MS", 10000
        )
        self.request_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_REQUEST_TIMEOUT_MS", 5000
        )
        self.create_timeout_ms = _positive_int(
            self._env, "RTP_LLM_MC_CREATE_TIMEOUT_MS", 125000
        )

        self.process: Optional[subprocess.Popen[bytes]] = None
        self.state_dir: Optional[Path] = None
        self.socket_path: Optional[Path] = None
        self.ready_path: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self._log_handle: Optional[Any] = None
        self._instance: Optional[Tuple[int, int]] = None
        self._started = False
        self._stopped = False

    @classmethod
    def from_config(
        cls,
        py_env_configs: Any,
        *,
        env: Optional[Mapping[str, str]] = None,
        **kwargs: Any,
    ) -> Optional["MulticastKeeperRuntime"]:
        """Build from server config, returning ``None`` when not opted in.

        The constructor remains independent from RTP-LLM config types; this
        adapter only reads the three values needed by the node-local runtime.
        """

        source = os.environ if env is None else env
        if not is_enabled(source):
            return None
        parallelism = py_env_configs.parallelism_config
        world_size = int(parallelism.world_size)
        configured_local_world_size = getattr(parallelism, "local_world_size", None)
        if (
            configured_local_world_size is not None
            and int(configured_local_world_size) > 0
        ):
            local_world_size = int(configured_local_world_size)
        else:
            local_world_size = int(source.get("LOCAL_WORLD_SIZE", world_size))
        role = py_env_configs.role_config.role_type
        if _role_name(role).upper().split(".")[-1] == "FRONTEND":
            return None
        return cls(
            world_size,
            local_world_size,
            role,
            env=source,
            **kwargs,
        )

    @property
    def instance(self) -> Optional[Tuple[int, int]]:
        return self._instance

    def _create_state_dir(self) -> None:
        role = re.sub(r"[^A-Za-z0-9_.-]+", "_", _role_name(self.role)).lower()
        directory = Path(
            tempfile.mkdtemp(prefix=f"rtp-llm-mc-{role}-", dir=self._state_root)
        )
        directory.chmod(0o700)
        socket_path = directory / "mcsk.sock"
        if len(os.fsencode(socket_path)) > _UNIX_SOCKET_PATH_LIMIT:
            shutil.rmtree(directory, ignore_errors=True)
            raise MulticastKeeperConfigError(
                f"multicast keeper socket path is too long: {socket_path}"
            )
        self.state_dir = directory
        self.socket_path = socket_path
        self.ready_path = directory / "holder.ready"
        self.log_path = directory / "holder.log"

    def _holder_command(self) -> Sequence[str]:
        if self.socket_path is None or self.ready_path is None:
            raise MulticastKeeperError("keeper state directory has not been created")
        command = [
            str(self.artifacts.holder),
            "--socket",
            str(self.socket_path),
            "--ready-file",
            str(self.ready_path),
            "--parent-pid",
            str(os.getpid()),
            "--creator",
            str(self.artifacts.creator),
            "--client-timeout-ms",
            str(self.client_timeout_ms),
            "--creator-timeout-ms",
            str(self.creator_timeout_ms),
            "--gpus",
            ",".join(str(gpu) for gpu in self.gpus),
        ]
        if self.fabric_team_size is not None:
            command.extend(["--fabric-team-size", str(self.fabric_team_size)])
        return command

    def _holder_env(self) -> Dict[str, str]:
        holder_env = dict(self._env)
        holder_env.pop("LD_PRELOAD", None)
        holder_env.pop("CUDA_VISIBLE_DEVICES", None)
        return holder_env

    def start(self) -> "MulticastKeeperRuntime":
        """Start the holder once and wait for a protocol-verified identity."""

        if self._started or self._stopped:
            raise MulticastKeeperError(
                "a multicast keeper runtime cannot be started more than once"
            )
        self._started = True
        try:
            self._create_state_dir()
            assert self.log_path is not None
            _LOGGER.info(
                "multicast keeper start begin: mode=%s role=%s world_size=%d "
                "local_world_size=%d gpus=%s fabric_team_size=%s holder=%s "
                "creator=%s shim=%s socket=%s",
                self.mode.value,
                _role_name(self.role),
                self.world_size,
                self.local_world_size,
                self.gpus,
                self.fabric_team_size,
                self.artifacts.holder,
                self.artifacts.creator,
                self.artifacts.shim,
                self.socket_path,
            )
            self._log_handle = self.log_path.open("ab", buffering=0)
            self.log_path.chmod(0o600)
            self.process = subprocess.Popen(
                self._holder_command(),
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                env=self._holder_env(),
                close_fds=True,
                start_new_session=True,
            )

            deadline = time.monotonic() + self.start_timeout_ms / 1000.0
            last_error: Optional[BaseException] = None
            while time.monotonic() < deadline:
                return_code = self.process.poll()
                if return_code is not None:
                    raise MulticastKeeperError(
                        f"multicast holder exited during startup with code {return_code}"
                    )
                if self.ready_path is not None and self.ready_path.is_file():
                    try:
                        health = self._ping()
                        self._instance = health.instance
                        _LOGGER.info(
                            "multicast keeper ready pid=%s instance=%016x%016x "
                            "gpus=%s fabric_team_size=%s socket=%s",
                            self.process.pid,
                            health.instance_hi,
                            health.instance_lo,
                            self.gpus,
                            self.fabric_team_size,
                            self.socket_path,
                        )
                        return self
                    except (OSError, MulticastKeeperHealthError) as exc:
                        last_error = exc
                time.sleep(0.05)
            suffix = f": {last_error}" if last_error is not None else ""
            raise MulticastKeeperError(
                f"multicast holder did not become ready within {self.start_timeout_ms}ms{suffix}"
            )
        except BaseException as exc:
            log_tail = self._read_log_tail()
            self._terminate_process()
            self._close_log()
            self._remove_state_dir()
            if isinstance(exc, MulticastKeeperError):
                detail = f"\nholder log tail:\n{log_tail}" if log_tail else ""
                raise type(exc)(f"{exc}{detail}") from exc
            raise

    def _ping(self) -> KeeperHealth:
        if self.socket_path is None:
            raise MulticastKeeperHealthError("multicast keeper has no socket")
        request = _REQUEST.pack(
            _PROTOCOL_MAGIC,
            _PROTOCOL_VERSION,
            _PING_OPCODE,
            _REQUEST.size,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        with socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET) as client:
            client.settimeout(self.client_timeout_ms / 1000.0)
            client.connect(str(self.socket_path))
            sent = client.send(request)
            if sent != len(request):
                raise MulticastKeeperHealthError(
                    f"short PING write: sent {sent} of {len(request)} bytes"
                )
            response = client.recv(_RESPONSE.size + 1)
        if len(response) != _RESPONSE.size:
            raise MulticastKeeperHealthError(
                f"invalid PING response size: {len(response)}"
            )
        values = _RESPONSE.unpack(response)
        (
            magic,
            version,
            opcode,
            struct_size,
            status,
            local_device_count,
            instance_hi,
            instance_lo,
            entries,
            _requested_size,
            _served_size,
            _num_devices,
            _handle_types,
            _flags,
        ) = values
        if (
            magic != _PROTOCOL_MAGIC
            or version != _PROTOCOL_VERSION
            or opcode != _PING_OPCODE
            or struct_size != _RESPONSE.size
            or status != _STATUS_OK
        ):
            raise MulticastKeeperHealthError(
                "multicast holder returned an invalid PING response"
            )
        if instance_hi == 0 and instance_lo == 0:
            raise MulticastKeeperHealthError(
                "multicast holder returned a zero identity"
            )
        if local_device_count != len(self.gpus):
            raise MulticastKeeperHealthError(
                "multicast holder local GPU count changed: "
                f"expected {len(self.gpus)}, got {local_device_count}"
            )
        return KeeperHealth(
            instance_hi=instance_hi,
            instance_lo=instance_lo,
            entries=entries,
            local_device_count=local_device_count,
        )

    def health(self) -> KeeperHealth:
        """PING the holder and require the exact identity recorded at startup."""

        if not self._started or self._stopped or self.process is None:
            raise MulticastKeeperHealthError("multicast keeper is not running")
        return_code = self.process.poll()
        if return_code is not None:
            raise MulticastKeeperHealthError(
                f"multicast holder exited with code {return_code}"
            )
        health = self._ping()
        if self._instance is None or health.instance != self._instance:
            expected = self._instance
            raise MulticastKeeperHealthError(
                f"multicast holder identity changed: expected {expected}, "
                f"got {health.instance}"
            )
        return health

    def is_alive(self) -> bool:
        try:
            self.health()
            return True
        except (OSError, MulticastKeeperHealthError):
            return False

    def diagnostics(self) -> Dict[str, Any]:
        """Return compact, non-secret state suitable for incident logs."""

        return {
            "mode": self.mode.value,
            "role": _role_name(self.role),
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "gpus": self.gpus,
            "fabric_team_size": self.fabric_team_size,
            "pid": self.process.pid if self.process is not None else None,
            "returncode": self.process.poll() if self.process is not None else None,
            "instance": self._instance,
            "socket": str(self.socket_path) if self.socket_path is not None else "",
            "log": str(self.log_path) if self.log_path is not None else "",
        }

    def log_tail(self, limit: int = 40) -> str:
        """Return the holder log tail before private runtime state is removed."""

        return self._read_log_tail(limit)

    def subprocess_env(
        self, base_env: Optional[Mapping[str, str]] = None
    ) -> Dict[str, str]:
        """Return a complete environment for checkpointed backend children."""

        self.health()
        child_env = dict(os.environ if base_env is None else base_env)
        assert self.state_dir is not None and self.socket_path is not None
        child_env[ENABLE_ENV] = "1"
        child_env[KEEPER_DIR_ENV] = str(self.state_dir)
        child_env[SOCKET_ENV] = str(self.socket_path)
        child_env[LOCAL_GPU_ENV] = ",".join(str(gpu) for gpu in self.gpus)
        if self.fabric_team_size is None:
            child_env.pop(FABRIC_TEAM_ENV, None)
        else:
            child_env[FABRIC_TEAM_ENV] = str(self.fabric_team_size)
        child_env.setdefault("NCCL_NVLS_ENABLE", "1")
        child_env.setdefault("TORCH_SYMM_MEM_DISABLE_MULTICAST", "0")
        child_env.setdefault(
            "RTP_LLM_MC_REQUEST_TIMEOUT_MS", str(self.request_timeout_ms)
        )
        child_env.setdefault(
            "RTP_LLM_MC_CREATE_TIMEOUT_MS", str(self.create_timeout_ms)
        )
        child_env["LD_PRELOAD"] = _append_preload(
            child_env.get("LD_PRELOAD", ""), self.artifacts.shim
        )
        return child_env

    @contextlib.contextmanager
    def configure_subprocess(self) -> Iterator[Mapping[str, str]]:
        """Temporarily configure ``os.environ`` while a backend is spawned."""

        original = dict(os.environ)
        configured = self.subprocess_env(original)
        changed_keys = {
            key
            for key in original.keys() | configured.keys()
            if original.get(key) != configured.get(key)
            or (key in original) != (key in configured)
        }
        for key in changed_keys:
            if key in configured:
                os.environ[key] = configured[key]
            else:
                os.environ.pop(key, None)
        try:
            yield configured
        finally:
            for key in changed_keys:
                if key in original:
                    os.environ[key] = original[key]
                else:
                    os.environ.pop(key, None)

    def _read_log_tail(self, limit: int = 40) -> str:
        if self.log_path is None or not self.log_path.is_file():
            return ""
        try:
            lines = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(lines[-limit:])
        except OSError:
            return ""

    def _terminate_process(self) -> None:
        process = self.process
        if process is None:
            return
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.stop_timeout_ms / 1000.0)
            except subprocess.TimeoutExpired:
                _LOGGER.warning(
                    "multicast holder pid=%s did not stop after SIGTERM; sending SIGKILL",
                    process.pid,
                )
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(1.0, self.stop_timeout_ms / 1000.0))
                except subprocess.TimeoutExpired:
                    _LOGGER.error("failed to reap multicast holder pid=%s", process.pid)
        else:
            process.wait()

    def _close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _remove_state_dir(self) -> None:
        if self.state_dir is not None:
            shutil.rmtree(self.state_dir, ignore_errors=True)

    def stop(self) -> None:
        """Stop the holder with bounded TERM/KILL waits and remove private state."""

        if self._stopped:
            return
        self._stopped = True
        self._terminate_process()
        diagnostics = self.diagnostics()
        holder_log_tail = self.log_tail()
        self._close_log()
        self._remove_state_dir()
        if self.process is not None:
            _LOGGER.info(
                "multicast keeper stopped: diagnostics=%s holder_log_tail=%s",
                diagnostics,
                holder_log_tail,
            )

    def __enter__(self) -> "MulticastKeeperRuntime":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()


__all__ = [
    "ENABLE_ENV",
    "KeeperArtifacts",
    "KeeperHealth",
    "MulticastKeeperConfigError",
    "MulticastKeeperError",
    "MulticastKeeperHealthError",
    "MulticastKeeperMode",
    "MulticastKeeperRuntime",
    "discover_artifacts",
    "is_enabled",
    "parse_gpu_list",
]
