import fcntl
import functools
import importlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Iterator, Optional

import torch

_LOGGER = logging.getLogger(__name__)
_DISABLE_ENV = "RTP_LLM_DISABLE_AITER_JIT_PATCH"
_AITER_AOT_IMPORT_ENV = "AITER_AOT_IMPORT"
_FILE_BATON_ORIGINAL_ATTR = "_rtp_llm_original_file_baton"
_AITER_CODEGEN_SCRIPTS = {
    "codegen.py",
    "gen_instances.py",
    "gen_instances_cktile.py",
    "generate.py",
    "generate_binaryop.py",
}
_SHELL_CONTROL_CHARS = frozenset(";&|<>")
_PATCH_LOCK = threading.RLock()
_AITER_BUILD_LOCK_TIMEOUT_SECONDS = 600.0
_AITER_BUILD_LOCK_STALE_SECONDS = 30.0
_AITER_BUILD_LOCK_POLL_SECONDS = 0.05


class _ModuleProxy:
    """Delegate module attributes while overriding a small, explicit subset."""

    def __init__(self, target: object, overrides: dict[str, object]):
        self._target = target
        self._overrides = overrides

    def __getattr__(self, name: str) -> object:
        if name in self._overrides:
            return self._overrides[name]
        return getattr(self._target, name)


@dataclass(frozen=True)
class _PatchState:
    core: types.ModuleType
    original_os: object
    original_importlib: object
    os_proxy: _ModuleProxy
    importlib_proxy: _ModuleProxy
    file_baton_bindings: tuple["_FileBatonBinding", ...]


@dataclass(frozen=True)
class _FileBatonBinding:
    namespace: dict[str, object]
    name: str
    original: type
    patched: type
    label: str


@dataclass(frozen=True)
class _BuildLockOwner:
    token: str
    pid: int
    hostname: str
    boot_id: Optional[str]
    pid_namespace: Optional[str]


@dataclass(frozen=True)
class _OwnedBuildLock:
    fd: int
    device: int
    inode: int
    owner: _BuildLockOwner


class _AiterBuildLockTimeout(RuntimeError):
    """AITER's file baton did not become available before the deadline."""


_PATCH_STATE: Optional[_PatchState] = None


class _StaleModuleRecovery(Enum):
    NOT_HANDLED = auto()
    REBUILD = auto()
    RETRY_IMPORT = auto()


def _is_rocm_runtime() -> bool:
    return torch.version.hip is not None


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def _aiter_package_roots() -> tuple[str, ...]:
    roots = []
    for module_name in ("aiter", "aiter_meta"):
        try:
            spec = importlib.util.find_spec(module_name)
        except (ImportError, ValueError):
            continue
        if spec is None or spec.submodule_search_locations is None:
            continue
        roots.extend(os.path.abspath(path) for path in spec.submodule_search_locations)
    return tuple(roots)


def _aiter_jit_roots() -> tuple[str, ...]:
    roots = [os.path.join(root, "jit") for root in _aiter_package_roots()]
    configured_root = os.environ.get("AITER_JIT_DIR")
    if configured_root:
        roots.append(os.path.abspath(os.path.expanduser(configured_root)))
    roots.append(os.path.abspath(os.path.expanduser("~/.aiter/jit")))
    return tuple(dict.fromkeys(roots))


def _is_within_root(path: str, root: str) -> bool:
    try:
        logical_path = os.path.abspath(path)
        logical_root = os.path.abspath(root)
        if os.path.commonpath((logical_path, logical_root)) == logical_root:
            return True
        real_path = os.path.realpath(path)
        real_root = os.path.realpath(root)
        return os.path.commonpath((real_path, real_root)) == real_root
    except ValueError:
        return False


def _split_aiter_codegen_command(command: object) -> Optional[list[str]]:
    if not isinstance(command, str):
        return None
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        args = list(lexer)
    except ValueError:
        return None
    if not args or any(
        any(char in arg for char in _SHELL_CONTROL_CHARS) for arg in args
    ):
        return None

    package_roots = _aiter_package_roots()
    matching_script_indices = [
        index
        for index, arg in enumerate(args)
        if os.path.basename(arg) in _AITER_CODEGEN_SCRIPTS
        and any(_is_within_root(arg, root) for root in package_roots)
    ]
    # AITER constructs every codegen command as ``<python> <script> ...``.
    # Requiring the script in argv[1] prevents an unrelated command from being
    # intercepted merely because one of its data arguments names a codegen file.
    return args if matching_script_indices == [1] else None


def _is_aiter_codegen_command(command: object) -> bool:
    return _split_aiter_codegen_command(command) is not None


def _run_aiter_codegen_command(command: str) -> int:
    args = _split_aiter_codegen_command(command)
    if args is None:
        _LOGGER.error("refusing to run malformed or compound aiter JIT codegen command")
        return 1
    try:
        return subprocess.run(args, check=False).returncode
    except OSError as error:
        _LOGGER.error("failed to run aiter JIT codegen command: %s", error)
        return 1


def _extract_shared_object_path(error_message: str) -> Optional[str]:
    match = re.search(
        r"(/[^\s:]+\.so):\s+"
        r"(?:undefined symbol:|file too short(?:\s*$)|invalid ELF header(?:\s*$))",
        error_message,
    )
    if match is None:
        return None
    return match.group(1)


def _valid_stale_module_path(module_name: str, so_path: str) -> bool:
    module_leaf = module_name.rsplit(".", 1)[-1]
    if os.path.basename(so_path) != f"{module_leaf}.so":
        return False
    return any(_is_within_root(so_path, root) for root in _aiter_jit_roots())


def _read_pid_namespace() -> Optional[str]:
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return None


def _read_boot_id() -> Optional[str]:
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="utf-8") as file:
            return file.read().strip() or None
    except OSError:
        return None


def _new_build_lock_owner() -> _BuildLockOwner:
    return _BuildLockOwner(
        token=os.urandom(16).hex(),
        pid=os.getpid(),
        hostname=socket.gethostname(),
        boot_id=_read_boot_id(),
        pid_namespace=_read_pid_namespace(),
    )


def _build_lock_owner_payload(owner: _BuildLockOwner) -> bytes:
    return json.dumps(
        {
            "token": owner.token,
            "pid": owner.pid,
            "hostname": owner.hostname,
            "boot_id": owner.boot_id,
            "pid_namespace": owner.pid_namespace,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_build_lock_owner(payload: bytes) -> Optional[_BuildLockOwner]:
    try:
        record = json.loads(payload.decode("utf-8"))
        owner = _BuildLockOwner(
            token=record["token"],
            pid=record["pid"],
            hostname=record["hostname"],
            boot_id=record["boot_id"],
            pid_namespace=record["pid_namespace"],
        )
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(owner.token, str)
        or not owner.token
        or not isinstance(owner.pid, int)
        or isinstance(owner.pid, bool)
        or owner.pid <= 0
        or not isinstance(owner.hostname, str)
        or not owner.hostname
        or (owner.boot_id is not None and not isinstance(owner.boot_id, str))
        or (
            owner.pid_namespace is not None and not isinstance(owner.pid_namespace, str)
        )
    ):
        return None
    return owner


def _read_build_lock_owner(lock_fd: int) -> Optional[_BuildLockOwner]:
    try:
        os.lseek(lock_fd, 0, os.SEEK_SET)
        payload = os.read(lock_fd, 4097)
    except OSError:
        return None
    if not payload or len(payload) > 4096:
        return None
    return _parse_build_lock_owner(payload)


def _same_lock_inode(lock_path: str, device: int, inode: int) -> bool:
    try:
        current = os.stat(lock_path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return current.st_dev == device and current.st_ino == inode


def _owner_process_is_dead(owner: _BuildLockOwner) -> Optional[bool]:
    if (
        owner.hostname != socket.gethostname()
        or owner.boot_id is None
        or owner.boot_id != _read_boot_id()
        or owner.pid_namespace is None
        or owner.pid_namespace != _read_pid_namespace()
    ):
        return None

    try:
        os.kill(owner.pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return None
    return False


def _try_create_owned_build_lock(lock_path: str) -> Optional[_OwnedBuildLock]:
    try:
        lock_fd = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_RDWR,
            0o644,
        )
    except FileExistsError:
        return None

    lock_stat = None
    try:
        lock_stat = os.fstat(lock_fd)
        owner = _new_build_lock_owner()
        payload = _build_lock_owner_payload(owner)
        if os.write(lock_fd, payload) != len(payload):
            raise OSError(f"short write while recording AITER lock owner: {lock_path}")
        os.fsync(lock_fd)
        return _OwnedBuildLock(
            fd=lock_fd,
            device=lock_stat.st_dev,
            inode=lock_stat.st_ino,
            owner=owner,
        )
    except BaseException:
        try:
            if lock_stat is not None and _same_lock_inode(
                lock_path, lock_stat.st_dev, lock_stat.st_ino
            ):
                os.remove(lock_path)
        finally:
            os.close(lock_fd)
        raise


def _try_reclaim_stale_build_lock(lock_path: str, stale_lock_seconds: float) -> str:
    # The sidecar path may persist, but its kernel lock is released on process
    # exit. It serializes reclaimers without becoming another stale-file baton.
    guard_fd = os.open(f"{lock_path}.rtp_reclaim", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(guard_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return "another process is checking the lock owner"
        try:
            lock_fd = os.open(lock_path, os.O_RDONLY)
        except FileNotFoundError:
            return "lock was released"
        except OSError as error:
            return f"lock metadata is unreadable: {error}"
        try:
            lock_stat = os.fstat(lock_fd)
            owner = _read_build_lock_owner(lock_fd)
        finally:
            os.close(lock_fd)
        if owner is None:
            return "owner is unknown (legacy or malformed lock)"

        age_seconds = max(0.0, (time.time_ns() - lock_stat.st_mtime_ns) / 1e9)
        if age_seconds < stale_lock_seconds:
            return f"owner pid={owner.pid} lock age={age_seconds:.1f}s"

        owner_is_dead = _owner_process_is_dead(owner)
        if owner_is_dead is not True:
            state = "alive" if owner_is_dead is False else "not locally verifiable"
            return f"owner pid={owner.pid} host={owner.hostname} is {state}"

        if not _same_lock_inode(lock_path, lock_stat.st_dev, lock_stat.st_ino):
            return "lock owner changed while checking staleness"
        try:
            os.remove(lock_path)
        except FileNotFoundError:
            return "lock was released while checking staleness"
        _LOGGER.warning(
            "reclaimed stale aiter JIT build lock %s from dead owner pid=%d age=%.1fs",
            lock_path,
            owner.pid,
            age_seconds,
        )
        return f"reclaimed dead owner pid={owner.pid}"
    finally:
        os.close(guard_fd)


def _release_owned_build_lock(lock_path: str, lock: _OwnedBuildLock) -> None:
    try:
        current_owner = None
        try:
            current_fd = os.open(lock_path, os.O_RDONLY)
        except FileNotFoundError:
            current_fd = None
        if current_fd is not None:
            try:
                current_owner = _read_build_lock_owner(current_fd)
            finally:
                os.close(current_fd)
        if (
            _same_lock_inode(lock_path, lock.device, lock.inode)
            and current_owner == lock.owner
        ):
            os.remove(lock_path)
        elif os.path.lexists(lock_path):
            _LOGGER.warning(
                "AITER build lock owner changed before release; preserving peer lock: %s",
                lock_path,
            )
    except FileNotFoundError:
        pass
    finally:
        os.close(lock.fd)


@contextmanager
def _aiter_build_lock(
    so_path: str,
    *,
    timeout_seconds: Optional[float] = None,
    stale_lock_seconds: Optional[float] = None,
    poll_seconds: Optional[float] = None,
) -> Iterator[None]:
    """Acquire AITER's file-baton path with bounded, owner-safe recovery."""

    module_base = os.path.splitext(os.path.basename(so_path))[0]
    build_root = os.path.join(os.path.dirname(so_path), "build")
    os.makedirs(build_root, exist_ok=True)
    lock_path = os.path.join(build_root, f"lock_{module_base}")
    timeout_seconds = (
        _AITER_BUILD_LOCK_TIMEOUT_SECONDS
        if timeout_seconds is None
        else timeout_seconds
    )
    stale_lock_seconds = (
        _AITER_BUILD_LOCK_STALE_SECONDS
        if stale_lock_seconds is None
        else stale_lock_seconds
    )
    poll_seconds = (
        _AITER_BUILD_LOCK_POLL_SECONDS if poll_seconds is None else poll_seconds
    )
    if timeout_seconds <= 0 or stale_lock_seconds < 0 or poll_seconds <= 0:
        raise ValueError(
            "AITER build lock timeout/poll must be positive and stale age non-negative"
        )

    deadline = time.monotonic() + timeout_seconds
    lock = None
    last_status = "not inspected"
    logged_wait = False
    try:
        while lock is None:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise _AiterBuildLockTimeout(
                    f"timed out after {timeout_seconds:.1f}s waiting for AITER JIT "
                    f"build lock {lock_path}: {last_status}. The lock was preserved "
                    "because no stale owner could be proven safe to reclaim."
                )

            lock = _try_create_owned_build_lock(lock_path)
            if lock is not None:
                break

            last_status = _try_reclaim_stale_build_lock(lock_path, stale_lock_seconds)
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                continue
            if not logged_wait:
                _LOGGER.info(
                    "waiting up to %.1fs for AITER JIT build lock %s: %s",
                    timeout_seconds,
                    lock_path,
                    last_status,
                )
                logged_wait = True
            time.sleep(min(poll_seconds, remaining_seconds))
        yield
    finally:
        if lock is not None:
            _release_owned_build_lock(lock_path, lock)


def _make_bounded_file_baton(original_file_baton: type) -> type:
    """Add owner-safe recovery and a deadline to AITER's file baton."""

    class _BoundedFileBaton(original_file_baton):
        def try_acquire(self) -> bool:
            lock = _try_create_owned_build_lock(self.lock_file_path)
            if lock is None:
                return False
            self.fd = lock.fd
            self._rtp_owned_lock = lock
            return True

        def wait(self) -> None:
            timeout_seconds = _AITER_BUILD_LOCK_TIMEOUT_SECONDS
            stale_lock_seconds = _AITER_BUILD_LOCK_STALE_SECONDS
            try:
                poll_seconds = float(self.wait_seconds)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "AITER file-baton poll interval must be numeric"
                ) from error
            if timeout_seconds <= 0 or stale_lock_seconds < 0 or poll_seconds <= 0:
                raise ValueError(
                    "AITER file-baton timeout/poll must be positive and stale age "
                    "non-negative"
                )

            deadline = time.monotonic() + timeout_seconds
            last_status = "not inspected"
            _LOGGER.info(
                "waiting up to %.1fs for AITER JIT file baton %s",
                timeout_seconds,
                self.lock_file_path,
            )
            while os.path.exists(self.lock_file_path):
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise _AiterBuildLockTimeout(
                        f"timed out after {timeout_seconds:.1f}s waiting for AITER "
                        f"JIT file baton {self.lock_file_path}: {last_status}. The "
                        "lock was preserved because no stale owner could be proven "
                        "safe to reclaim."
                    )

                last_status = _try_reclaim_stale_build_lock(
                    self.lock_file_path, stale_lock_seconds
                )
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    continue
                time.sleep(min(poll_seconds, remaining_seconds))

        def release(self) -> None:
            lock = getattr(self, "_rtp_owned_lock", None)
            if lock is None:
                super().release()
                return
            try:
                _release_owned_build_lock(self.lock_file_path, lock)
            finally:
                self.fd = None
                self._rtp_owned_lock = None

    _BoundedFileBaton.__name__ = f"RtpBounded{original_file_baton.__name__}"
    _BoundedFileBaton.__qualname__ = _BoundedFileBaton.__name__
    setattr(_BoundedFileBaton, _FILE_BATON_ORIGINAL_ATTR, original_file_baton)
    return _BoundedFileBaton


def _maybe_remove_stale_aiter_jit_module(
    module_name: str,
    error: ImportError,
    import_started_ns: Optional[int] = None,
) -> _StaleModuleRecovery:
    so_path = _extract_shared_object_path(str(error))
    if so_path is None or not _valid_stale_module_path(module_name, so_path):
        return _StaleModuleRecovery.NOT_HANDLED

    try:
        with _aiter_build_lock(so_path):
            # A peer may already have removed the same stale file. Treat that as
            # a successful cleanup so every rank proceeds to AITER's build lock.
            removed_here = False
            if os.path.exists(so_path):
                # A peer may have rebuilt while this import was failing. Retrying
                # the import avoids deleting or redundantly rebuilding its result.
                if (
                    import_started_ns is not None
                    and os.stat(so_path).st_mtime_ns > import_started_ns
                ):
                    return _StaleModuleRecovery.RETRY_IMPORT
                os.remove(so_path)
                removed_here = True
            module_base = os.path.splitext(os.path.basename(so_path))[0]
            build_dir = os.path.join(os.path.dirname(so_path), "build", module_base)
            # Only the rank that removed the stale .so clears its build tree.
            # Peers that arrive later must not delete an in-progress rebuild.
            if removed_here and os.path.exists(build_dir):
                shutil.rmtree(build_dir)
        if removed_here:
            _LOGGER.warning(
                "removed stale aiter JIT module after import failure: %s", so_path
            )
        else:
            _LOGGER.info(
                "stale aiter JIT module was already removed by a peer: %s",
                so_path,
            )
        return _StaleModuleRecovery.REBUILD
    except OSError as remove_error:
        _LOGGER.warning(
            "failed to remove stale aiter JIT module %s: %s",
            so_path,
            remove_error,
        )
        return _StaleModuleRecovery.NOT_HANDLED


def _make_patched_system(original_system: Callable[..., int]) -> Callable[..., int]:
    def patched_system(command: object) -> int:
        if not _is_aiter_codegen_command(command):
            return original_system(command)
        ret = _run_aiter_codegen_command(command)
        if ret != 0:
            raise RuntimeError(f"aiter JIT codegen failed with exit code {ret}")
        return ret

    return patched_system


def _make_patched_import_module(
    original_import_module: Callable[..., types.ModuleType],
) -> Callable[..., types.ModuleType]:
    def patched_import_module(
        name: str, package: Optional[str] = None
    ) -> types.ModuleType:
        import_started_ns = time.time_ns()
        try:
            return original_import_module(name, package)
        except ImportError as error:
            recovery = _maybe_remove_stale_aiter_jit_module(
                name, error, import_started_ns
            )
            if recovery is _StaleModuleRecovery.RETRY_IMPORT:
                return original_import_module(name, package)
            if recovery is _StaleModuleRecovery.REBUILD:
                raise ModuleNotFoundError(
                    f"removed stale aiter JIT module {name}; rebuild required"
                ) from error
            raise

    return patched_import_module


@contextmanager
def _temporary_stdlib_hooks() -> Iterator[None]:
    """Patch only while AITER's eager import runs, then restore in ``finally``."""

    original_system = os.system
    original_import_module = importlib.import_module
    patched_system = _make_patched_system(original_system)
    patched_import_module = _make_patched_import_module(original_import_module)
    os.system = patched_system
    importlib.import_module = patched_import_module
    try:
        yield
    finally:
        if os.system is patched_system:
            os.system = original_system
        else:
            _LOGGER.warning(
                "os.system changed during AITER bootstrap; "
                "leaving the newer value intact"
            )
        if importlib.import_module is patched_import_module:
            importlib.import_module = original_import_module
        else:
            _LOGGER.warning(
                "importlib.import_module changed during AITER bootstrap; "
                "leaving the newer value intact"
            )


def _get_aiter_core(aiter_module: types.ModuleType) -> types.ModuleType:
    core = getattr(aiter_module, "core", None)
    if core is None:
        core = sys.modules.get("aiter.jit.core")
    if not isinstance(core, types.ModuleType):
        raise RuntimeError("AITER JIT core was not initialized by import aiter")
    return core


def _import_aiter_file_baton_module() -> None:
    module_name = "aiter.jit.utils.file_baton"
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise


def _bootstrap_aiter_core() -> types.ModuleType:
    # AITER's normal top-level import invokes module_aiter_core while importing
    # aiter.ops.enum. Preload only its JIT core in the supported AOT mode, patch
    # both file batons, then reload with the caller's original mode. This closes
    # the orphan-lock window before the first eager JIT build starts.
    needs_bounded_preload = (
        "aiter" not in sys.modules and os.environ.get(_AITER_AOT_IMPORT_ENV) != "1"
    )
    if not needs_bounded_preload:
        with _temporary_stdlib_hooks():
            aiter_module = importlib.import_module("aiter")
            _import_aiter_file_baton_module()
        return _get_aiter_core(aiter_module)

    missing = object()
    original_aot_import = os.environ.get(_AITER_AOT_IMPORT_ENV, missing)
    os.environ[_AITER_AOT_IMPORT_ENV] = "1"
    try:
        with _temporary_stdlib_hooks():
            aiter_module = importlib.import_module("aiter")
            _import_aiter_file_baton_module()
    finally:
        if os.environ.get(_AITER_AOT_IMPORT_ENV) != "1":
            _LOGGER.warning(
                "%s changed during AITER bootstrap; leaving the newer value intact",
                _AITER_AOT_IMPORT_ENV,
            )
        elif original_aot_import is missing:
            os.environ.pop(_AITER_AOT_IMPORT_ENV, None)
        else:
            os.environ[_AITER_AOT_IMPORT_ENV] = original_aot_import

    bootstrap_bindings: tuple[_FileBatonBinding, ...] = ()
    try:
        core = _get_aiter_core(aiter_module)
        bootstrap_bindings = _file_baton_bindings(core)
        _apply_file_baton_bindings(bootstrap_bindings)
        with _temporary_stdlib_hooks():
            reloaded_aiter = importlib.reload(aiter_module)
        reloaded_core = _get_aiter_core(reloaded_aiter)
        if reloaded_core is not core:
            raise RuntimeError("AITER JIT core changed during bounded bootstrap")
        return core
    except BaseException:
        if bootstrap_bindings:
            _restore_file_baton_bindings(bootstrap_bindings, warn_on_change=False)
        if sys.modules.get("aiter") is aiter_module:
            sys.modules.pop("aiter", None)
        raise


def _file_baton_bindings(core: types.ModuleType) -> tuple[_FileBatonBinding, ...]:
    namespaces: list[tuple[str, dict[str, object]]] = [
        ("aiter.jit.core.FileBaton", core.__dict__)
    ]
    jit_compile = getattr(core, "_jit_compile", None)
    jit_compile_globals = getattr(jit_compile, "__globals__", None)
    if isinstance(jit_compile_globals, dict):
        namespaces.append(
            ("aiter.jit.utils.cpp_extension.FileBaton", jit_compile_globals)
        )

    current_core_file_baton = core.__dict__.get("FileBaton")
    if not isinstance(current_core_file_baton, type):
        raise RuntimeError("AITER JIT core does not expose a FileBaton class")
    core_file_baton = getattr(
        current_core_file_baton,
        _FILE_BATON_ORIGINAL_ATTR,
        current_core_file_baton,
    )
    if not isinstance(core_file_baton, type):
        raise RuntimeError("AITER JIT FileBaton original binding is invalid")
    file_baton_module_names = (
        core_file_baton.__module__,
        "file_baton",
        "aiter.jit.utils.file_baton",
    )
    for module_name in dict.fromkeys(file_baton_module_names):
        file_baton_module = sys.modules.get(module_name)
        if isinstance(file_baton_module, types.ModuleType) and isinstance(
            file_baton_module.__dict__.get("FileBaton"), type
        ):
            namespaces.append((f"{module_name}.FileBaton", file_baton_module.__dict__))

    bindings = []
    patched_classes: dict[type, type] = {}
    if current_core_file_baton is not core_file_baton:
        patched_classes[core_file_baton] = current_core_file_baton
    seen_namespaces = set()
    for label, namespace in namespaces:
        namespace_key = (id(namespace), "FileBaton")
        if namespace_key in seen_namespaces:
            continue
        seen_namespaces.add(namespace_key)
        current_file_baton = namespace.get("FileBaton")
        if not isinstance(current_file_baton, type):
            continue
        original_file_baton = getattr(
            current_file_baton,
            _FILE_BATON_ORIGINAL_ATTR,
            current_file_baton,
        )
        if not isinstance(original_file_baton, type):
            continue
        if current_file_baton is not original_file_baton:
            patched_classes.setdefault(original_file_baton, current_file_baton)
        patched_file_baton = patched_classes.get(original_file_baton)
        if patched_file_baton is None:
            patched_file_baton = _make_bounded_file_baton(original_file_baton)
            patched_classes[original_file_baton] = patched_file_baton
        bindings.append(
            _FileBatonBinding(
                namespace=namespace,
                name="FileBaton",
                original=original_file_baton,
                patched=patched_file_baton,
                label=label,
            )
        )
    return tuple(bindings)


def _apply_file_baton_bindings(
    bindings: tuple[_FileBatonBinding, ...],
) -> None:
    for binding in bindings:
        binding.namespace[binding.name] = binding.patched


def _restore_file_baton_bindings(
    bindings: tuple[_FileBatonBinding, ...], *, warn_on_change: bool
) -> None:
    for binding in bindings:
        if binding.namespace.get(binding.name) is binding.patched:
            binding.namespace[binding.name] = binding.original
        elif warn_on_change:
            _LOGGER.warning(
                "%s changed after installation; not overwriting it",
                binding.label,
            )


def _install_core_patch(core: types.ModuleType) -> _PatchState:
    original_os = core.os
    original_importlib = core.importlib
    os_proxy = _ModuleProxy(
        original_os,
        {"system": _make_patched_system(getattr(original_os, "system"))},
    )
    importlib_proxy = _ModuleProxy(
        original_importlib,
        {
            "import_module": _make_patched_import_module(
                getattr(original_importlib, "import_module")
            )
        },
    )
    file_baton_bindings = _file_baton_bindings(core)
    core.os = os_proxy
    core.importlib = importlib_proxy
    _apply_file_baton_bindings(file_baton_bindings)
    return _PatchState(
        core=core,
        original_os=original_os,
        original_importlib=original_importlib,
        os_proxy=os_proxy,
        importlib_proxy=importlib_proxy,
        file_baton_bindings=file_baton_bindings,
    )


def _uninstall_locked() -> bool:
    global _PATCH_STATE
    state = _PATCH_STATE
    if state is None:
        return False
    if state.core.os is state.os_proxy:
        state.core.os = state.original_os
    else:
        _LOGGER.warning(
            "AITER core os binding changed after installation; not overwriting it"
        )
    if state.core.importlib is state.importlib_proxy:
        state.core.importlib = state.original_importlib
    else:
        _LOGGER.warning(
            "AITER core importlib binding changed after installation; "
            "not overwriting it"
        )
    _restore_file_baton_bindings(state.file_baton_bindings, warn_on_change=True)
    _PATCH_STATE = None
    return True


def install_aiter_jit_patch(*, enabled: Optional[bool] = None) -> bool:
    """Initialize AITER and install process-local compatibility at its JIT boundary.

    Standard-library hooks exist only during AITER's eager import and are always
    restored. Lazy JIT calls are handled through proxies stored on
    ``aiter.jit.core`` itself. Set ``RTP_LLM_DISABLE_AITER_JIT_PATCH=1`` (or pass
    ``enabled=False``) to use upstream AITER behavior unchanged.
    """

    global _PATCH_STATE
    disabled = enabled is False or (enabled is None and _env_flag_enabled(_DISABLE_ENV))
    if disabled:
        with _PATCH_LOCK:
            _uninstall_locked()
        return False
    if enabled is not True and not _is_rocm_runtime():
        return False

    with _PATCH_LOCK:
        if _PATCH_STATE is not None:
            state = _PATCH_STATE
            if (
                state.core.os is state.os_proxy
                and state.core.importlib is state.importlib_proxy
                and all(
                    binding.namespace.get(binding.name) is binding.patched
                    for binding in state.file_baton_bindings
                )
            ):
                return True
            _uninstall_locked()
        core = _bootstrap_aiter_core()
        try:
            _PATCH_STATE = _install_core_patch(core)
        except BaseException:
            _restore_file_baton_bindings(
                _file_baton_bindings(core), warn_on_change=False
            )
            raise
        return True


def load_aiter() -> types.ModuleType:
    """Load AITER through the scoped JIT compatibility boundary."""

    install_aiter_jit_patch()
    return importlib.import_module("aiter")


def uninstall_aiter_jit_patch() -> bool:
    """Restore AITER core bindings installed by :func:`install_aiter_jit_patch`."""

    with _PATCH_LOCK:
        return _uninstall_locked()
