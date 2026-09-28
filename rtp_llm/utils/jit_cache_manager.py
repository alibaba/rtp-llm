import importlib.metadata
import logging
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from collections import namedtuple
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from stat import S_ISREG
from threading import Event, Thread
from time import monotonic

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError:
    # Some accelerator images intentionally omit watchdog. Snapshot polling in
    # _sync_loop preserves correctness; only event-driven publication is lost.
    class FileSystemEventHandler:
        pass

    Observer = None

from rtp_llm.utils import jit_cache_store as store

SYNC_POLL_S, STOP_TIMEOUT_S = 120.0, 10.0
RTP_JIT_VERSION, CUDA, ROCM = "v1", "cuda", "rocm"
LOCKS_DIR, STAGING_DIR = ".locks", ".staging"
# Fixed path: build artifacts embed absolute paths, so relocating voids snapshots; opt out via --manage_jit_cache.
LOCAL_JIT_ROOT = Path("/tmp/rtp-llm/.jit_cache")
GPU_PROBE = """import torch
a={str(torch.cuda.get_device_properties(i).gcnArchName).split(":")[0] if torch.version.hip else "sm_{}{}".format(*torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())}
if len(a)!=1: raise SystemExit(f"mixed GPU architectures: {sorted(a)}")
print(a.pop())"""


def _run(args, timeout: float) -> str:
    return subprocess.check_output(
        args, text=True, timeout=timeout, stderr=subprocess.PIPE
    ).strip()


def _accelerator_scope(backend: str, version: str) -> str | None:
    try:
        arch = _run([sys.executable, "-c", GPU_PROBE], 60)
        return f"{backend}-{version}-{arch}" if arch else None
    except (OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or error
        logging.warning("JIT_CACHE_FAIL_OPEN: GPU probe failed: %s", detail)


def _toolkit_scope(backend: str) -> str | None:
    try:
        from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

        name, home = ("nvcc", CUDA_HOME) if backend == CUDA else ("hipcc", ROCM_HOME)
        compiler = Path(home) / "bin" / name if home else shutil.which(name)
        if not compiler or (home and not compiler.is_file()):
            raise OSError(f"{name} not found")
        v = _run([str(compiler), "--version"], 10)
        return f"{name}-{sha256(v.encode()).hexdigest()[:16]}" if v else None
    except (ImportError, OSError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or error
        logging.warning("JIT_CACHE_FAIL_OPEN: GPU toolkit probe failed: %s", detail)


def _cpp_runtime_scope() -> str | None:
    try:
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        version = _run([*cxx, "--version"], 5).splitlines()[0]
        library = Path(_run([*cxx, "-print-file-name=libstdc++.so.6"], 5))
        if not library.is_file():
            raise OSError(f"unresolved libstdc++: {library}")
        digest = sha256(version.encode() + b"\0" + library.read_bytes())
        return f"cxx-{digest.hexdigest()[:16]}"
    except (IndexError, OSError, ValueError, subprocess.SubprocessError) as error:
        detail = getattr(error, "stderr", "") or error
        logging.warning("JIT_CACHE_FAIL_OPEN: C++ probe failed: %s", detail)


# fmt: off
@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    rules: tuple
    scopes: tuple[str, ...] = ()
    backend: str | None = None
    local_dir: Path = Path()

    def should_sync(self, rel: str, event_type: str | None = None) -> bool:
        events = next((e for suffixes, e in self.rules if rel.endswith(suffixes)), ())
        parts = rel.split("/")
        return ((event_type in events if event_type else bool(events)) and ".." not in parts
                and not any(p == "tmp" or p.startswith("tmp.pid_") for p in parts))
def rule(events: frozenset[str], *suffixes: str):
    return suffixes, events
NINJA = (".so", ".o", "build.ninja", ".ninja_log", ".ninja_deps")
MOVED = frozenset({"moved"})
CLOSED = MOVED | {"closed"}
CREATED = CLOSED | {"created"}
COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE", (rule(CREATED, ".cu", ".inc", ".h"), rule(CLOSED, *NINJA)), ("torch", "@flashinfer-python"), CUDA),
    Component("deep_gemm", "DG_JIT_CACHE_DIR", (rule(CREATED, "kernel.cu", "kernel.cubin"),), ("accelerator", "@deep_gemm"), CUDA),
    Component("trtllm_deep_gemm", "TRTLLM_DG_CACHE_DIR", (rule(CREATED, "nvcc_kernel.cubin"),), ("accelerator", "@flashinfer-python"), CUDA),
    Component("tilelang", "TILELANG_CACHE_DIR", (rule(CLOSED, ".so", ".pkl", ".cu", ".json", ".cubin", ".py"),), ("torch", "@tilelang"), CUDA),
    # rtp_kernel is the only producer here whose outputs are not self-keyed (TIPC content-hashes its subdir).
    Component("torch_extensions", "TORCH_EXTENSIONS_DIR", (rule(CLOSED, *NINJA, ".cpp", ".cu"),), ("torch", "@rtp_kernel")),
    Component("aiter", "AITER_JIT_DIR", (rule(CLOSED, *NINJA, ".cu", ".cpp", ".hip", ".h"),), ("torch", "@aiter"), ROCM),
    Component("flydsl", "FLYDSL_RUNTIME_CACHE_DIR", (rule(MOVED, ".pkl"),), ("accelerator", "@flydsl"), ROCM),
    Component("tvm_ffi", "TVM_FFI_CACHE_DIR", (rule(CREATED, ".so"),), ("torch", "@apache-tvm-ffi"), CUDA),
    Component("cute_dsl", "CUTE_DSL_CACHE_DIR", (rule(MOVED, ".mlir"),), ("accelerator", "@nvidia-cutlass-dsl"), CUDA),
    # First match wins: autotune results remain local.
    Component("triton", "TRITON_CACHE_DIR", (rule(frozenset(), ".autotune.json"), rule(CREATED, ".json", ".cubin", ".hsaco", ".so")), ("cxx",)),
)
Scope = namedtuple("Scope", "scope_id root components")
# fmt: on


def resolve_scope(local_root: Path) -> Scope | None:
    import torch

    from rtp_llm.utils.util import COMPILE_FLAG_ENVS, torch_abi_fingerprint

    if not (version := torch.version.hip or torch.version.cuda):
        return None
    backend = ROCM if torch.version.hip else CUDA
    accelerator, toolkit = _accelerator_scope(backend, version), _toolkit_scope(backend)
    cpp, abi = _cpp_runtime_scope(), torch_abi_fingerprint()
    probes = {"gpu": accelerator, "toolkit": toolkit, "cxx": cpp, "abi": abi}
    if failed := [name for name, value in probes.items() if not value]:
        logging.warning("JIT_CACHE_FAIL_OPEN: build scope probes failed: %s", failed)
        return None
    libc = "-".join(platform.libc_ver())
    soabi = sysconfig.get_config_var("SOABI") or platform.machine()
    torch_scope = "-".join(map(str, (soabi, libc, accelerator, cpp, *abi)))
    scopes = {"accelerator": accelerator, "torch": torch_scope, "cxx": cpp}
    selected, keys = [], [f"root-{local_root}", toolkit, accelerator, torch_scope]
    # Preserve the established zstd scope. Only fallback archives need a new
    # namespace so a host without zstandard never opens an incompatible .zst.
    if store.zstd is None:
        keys.append("archive-gzip")
    flags = [f"{k}={v}" for k in COMPILE_FLAG_ENVS if (v := os.environ.get(k))]
    if flags:  # toolchain overrides change codegen: they belong in the key
        keys.append("flags-" + sha256("\0".join(flags).encode()).hexdigest()[:12])
    for item in COMPONENTS:
        if item.backend not in (None, backend) or item.env_name in os.environ:
            continue
        with suppress(importlib.metadata.PackageNotFoundError):
            parts = tuple(
                importlib.metadata.version(p[1:]) if p.startswith("@") else scopes[p]
                for p in item.scopes
            )
            if parts and all(parts):
                selected.append(item)
                keys.append("-".join((item.name, *parts)))
    if not selected:
        logging.warning("JIT_CACHE_FAIL_OPEN: no managed components")
        return None
    scope_id = sha256("\0".join(keys).encode()).hexdigest()[:16]
    names = ",".join(x.name for x in selected)
    logging.info("JIT scope %s at %s: %s", scope_id, local_root, names)
    root = local_root / RTP_JIT_VERSION / scope_id
    components = tuple(replace(item, local_dir=root / item.name) for item in selected)
    return Scope(scope_id, root, components)


# All UIDs may write/load this tree; enable only in a trusted container.
def _make_shared_dir(path: Path) -> None:
    if path.is_symlink() or path.parent.is_symlink():
        raise OSError(f"untrusted JIT dir {path}")
    path.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        path.chmod(0o1777)


def _prepare_shared_root(root: Path) -> None:
    _make_shared_dir(root)
    # Only roots are sticky; the ACL keeps managed descendants peer-replaceable.
    # It lands descendants at 0777/0666 without a process-wide umask(0).
    try:  # a co-tenant's root is not ours to setfacl; settle for an equivalent one
        _run(["setfacl", "-m", "g::rwx,o::rwx,d:g::rwx,d:o::rwx", root], 5)
    except (OSError, subprocess.SubprocessError):
        acl = set(_run(["getfacl", "-cp", root], 5).splitlines())
        if not {"other::rwx", "default:other::rwx"} <= acl:
            raise OSError(f"shared ACL unavailable for {root}")


@lru_cache(maxsize=1)  # a second call must not read the env this one exports
def setup_jit_cache_env() -> Scope | None:
    try:
        # Test-only override: isolated roots intentionally skip the shared parent.
        override = os.getenv("TEST_JIT_LOCAL_DIR", "").strip()
        local_root = Path(override) if override else LOCAL_JIT_ROOT
        if not (scope := resolve_scope(local_root)):
            return None
        if not override:
            _make_shared_dir(local_root.parent)
        _prepare_shared_root(local_root)
        mode = os.stat(local_root).st_mode & 0o7777
        logging.info("JIT shared root %s mode %o", local_root, mode)
        if scope.root.exists() and not os.access(scope.root, os.W_OK):
            raise OSError(f"scope root not shared by its owner: {scope.root}")
        for item in scope.components:
            os.environ[item.env_name] = str(item.local_dir)
            # Only torch/aiter use existence batons; tvm_ffi's same-named file is flocked.
            if item.name in ("torch_extensions", "aiter"):
                store.reap_stale_batons(item.local_dir)
        return scope
    except Exception:
        logging.warning("JIT_CACHE_FAIL_OPEN: env setup failed", exc_info=True)


class JitCacheManager(FileSystemEventHandler):
    def __init__(self, scope: Scope, remote_value: str):
        self.scope, self._remote_value, self.store = scope, remote_value, None
        self._observer = self._worker = self._prepare_thread = None
        self._seen = self._restored = None
        self._dirty, self._stop = Event(), Event()

    def on_any_event(self, event) -> None:
        if event.is_directory or self._stop.is_set():
            return
        path = Path(event.dest_path if event.event_type == "moved" else event.src_path)
        for item in self.scope.components:
            if item.local_dir in path.parents:
                with suppress(OSError, ValueError):
                    rel = path.relative_to(item.local_dir).as_posix()
                    if item.should_sync(rel, event.event_type) and path.stat().st_size:
                        self._dirty.set()
                return

    def _lock_path(self, kind: str) -> Path:
        return self.scope.root.parent / LOCKS_DIR / f"{self.scope.scope_id}.{kind}.lock"

    def _prepare(self) -> None:
        """Own the whole restore: its lock, its mount, and its staging tree."""
        root, restore_fd = self.scope.root, None
        try:
            if not store.scope_root_usable(root):
                restore_fd = store.acquire_flock(self._lock_path("restore"))
            if self._stop.is_set():  # abandoned: do not mount a remote nobody reads
                return
            self.store = store.resolve_remote(
                self._remote_value, RTP_JIT_VERSION, self.scope.scope_id
            )
            cold = self.store and restore_fd is not None
            if cold and not store.scope_root_usable(root):
                staging_root = root.parent / STAGING_DIR / self.scope.scope_id
                shutil.rmtree(staging_root, ignore_errors=True)
                if prepared := self.store.prepare_restore(staging_root):
                    # Abandoned; past here commit_restore still guards a live root.
                    if self._stop.is_set():
                        shutil.rmtree(prepared.staging, ignore_errors=True)
                    elif store.commit_restore(prepared.staging, root):
                        self._restored = prepared.snapshot.name
        except Exception:
            logging.exception("JIT_CACHE_FAIL_OPEN: restore failed")
        finally:
            if restore_fd is not None:
                os.close(restore_fd)

    def bootstrap(self, timeout_s: float) -> bool:
        """Restore under a deadline, then watch; -1 waits without limit."""
        self._prepare_thread = Thread(target=self._prepare, daemon=True)
        self._prepare_thread.start()
        self._prepare_thread.join(None if timeout_s == -1 else timeout_s)
        if self._prepare_thread.is_alive():
            self._stop.set()  # a late restore drops its own staging, best effort
            logging.warning("JIT_CACHE_FAIL_OPEN: setup timed out after %ss", timeout_s)
            return False
        if not self.store:
            return False
        if self._restored:
            logging.info("JIT_CACHE_RESTORED: %s", self._restored)
        else:
            with suppress(OSError):  # seed an empty remote once; restarts stay quiet
                if not any(self.store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}")):
                    self._dirty.set()
        return self._start_watch()

    def _start_watch(self) -> bool:
        if Observer is None:
            logging.warning(
                "JIT_CACHE_FAIL_OPEN: watchdog unavailable; polling fallback active"
            )
        else:
            try:
                self._observer = Observer()
                for item in self.scope.components:
                    item.local_dir.mkdir(parents=True, exist_ok=True)
                    self._observer.schedule(self, str(item.local_dir), recursive=True)
                self._observer.start()
            except Exception:
                logging.exception(
                    "JIT_CACHE_FAIL_OPEN: inotify watch setup failed; polling fallback active"
                )
        self._worker = Thread(target=self._sync_loop, daemon=True)
        self._worker.start()
        return True

    def _snapshot_files(self) -> dict[str, Path]:
        files = {}
        for item in self.scope.components:
            for path in item.local_dir.rglob("*"):
                with suppress(OSError):
                    st, rel = path.lstat(), path.relative_to(item.local_dir).as_posix()
                    packable = S_ISREG(st.st_mode) and os.access(path, os.R_OK)
                    if st.st_size and packable and item.should_sync(rel):
                        files[f"{item.name}/{rel}"] = path
        return files

    def publish_pending_snapshot(self) -> None:
        files, state = {}, {}
        for name, path in self._snapshot_files().items():
            with suppress(FileNotFoundError):
                state[name], files[name] = store.file_sig(path), path
        seen, self._seen = self._seen, state
        if seen is not None and seen != state:  # inotify can go blind
            self._dirty.set()
        lock, dirty = self._lock_path("publish"), self._dirty.is_set()
        if (fd := store.acquire_flock(lock, False) if dirty else None) is None:
            return  # a peer holds the publish lock; our next poll picks the work up
        try:
            self._dirty.clear()
            self.store.publish_snapshot(files, self._snapshot_files)
        except Exception as error:
            self._dirty.set()
            if not isinstance(error, store.SnapshotRaced):
                raise
            logging.info("JIT snapshot deferred; builder active during pack: %s", error)
        finally:
            os.close(fd)

    def _sync_loop(self) -> None:
        while True:
            stopping = self._stop.wait(SYNC_POLL_S)
            try:
                self.publish_pending_snapshot()
            except Exception:
                logging.exception("JIT_CACHE_FAIL_OPEN: snapshot sync failed")
            if stopping:
                return

    def _cleanup(self, deadline: float) -> None:
        for worker in (self._observer, self._worker, self._prepare_thread):
            if worker:
                with suppress(RuntimeError):
                    worker.join(max(0.0, deadline - monotonic()))
        if self.store:  # unmount even if a worker overran; close() locks out publish
            self.store.close()

    def stop(self) -> None:
        """Idempotent and retryable, because the store's own close() is both."""
        self._stop.set()
        if self._observer:
            self._observer.stop()
        cleanup = Thread(
            target=self._cleanup, args=(monotonic() + STOP_TIMEOUT_S,), daemon=True
        )
        cleanup.start()
        cleanup.join(STOP_TIMEOUT_S)
        if cleanup.is_alive():
            logging.warning("JIT cleanup continues in background")


def start_from_config(config):
    if not config.manage_jit_cache:
        logging.info("JIT cache management disabled by configuration")
        return
    remote = str(config.remote_jit_dir or "").strip()
    if (scope := setup_jit_cache_env()) is None or not remote:
        return
    if store.zstd is None:
        logging.warning(
            "JIT_CACHE_FALLBACK: zstandard unavailable; gzip snapshot fallback active"
        )
    manager = JitCacheManager(scope, remote)
    try:
        if manager.bootstrap(config.jit_cache_setup_timeout_s):
            return manager
    except BaseException:  # the caller logs it and falls back to a cold start
        manager.stop()
        raise
    manager.stop()
