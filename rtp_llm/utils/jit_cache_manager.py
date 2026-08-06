import hashlib
import importlib.metadata
import logging
import os
import platform
import queue
import shlex
import shutil
import stat
import subprocess
import sys
import sysconfig
import threading
import time
from collections import namedtuple
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.utils import jit_cache_store as store

SYNC_POLL_S, STOP_TIMEOUT_S = 120.0, 10.0
RTP_JIT_VERSION, CUDA, ROCM = "v1", "cuda", "rocm"
LOCAL_JIT_ROOT = Path("/tmp/rtp-llm/.jit_cache")  # JIT artifacts embed this path
GPU_PROBE = (
    "import torch;a={str(torch.cuda.get_device_properties(i).gcnArchName).split(':')[0] if torch.version.hip "
    "else 'sm_{}{}'.format(*torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())};"
    "assert len(a)==1,a;print(a.pop())"
)


def _run(args, timeout: float, stderr=subprocess.STDOUT) -> str:
    output = subprocess.check_output(args, text=True, timeout=timeout, stderr=stderr)
    return output.strip()


def _accelerator_scope(backend: str, version: str) -> str | None:
    try:
        arch = _run([sys.executable, "-c", GPU_PROBE], 60)
        return f"{backend}-{version}-{arch}" if arch else None
    except (OSError, ValueError, subprocess.SubprocessError):
        logging.warning("JIT_CACHE_FAIL_OPEN: GPU probe failed", exc_info=True)


def _toolkit_scope(backend: str) -> str | None:
    try:
        from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

        name, home = ("nvcc", CUDA_HOME) if backend == CUDA else ("hipcc", ROCM_HOME)
        compiler = Path(home) / "bin" / name if home else shutil.which(name)
        if not compiler or (home and not compiler.is_file()):
            raise OSError(f"{name} not found")
        return f"{name}-{v}" if (v := _run([str(compiler), "--version"], 10)) else None
    except (ImportError, OSError, ValueError, subprocess.SubprocessError):
        logging.warning("JIT_CACHE_FAIL_OPEN: GPU toolkit probe failed", exc_info=True)


def _cpp_runtime_scope() -> str | None:
    try:
        cxx = shlex.split(os.environ.get("CXX", "c++"))
        version = _run([*cxx, "--version"], 5, stderr=None).splitlines()[0]
        library = Path(_run([*cxx, "-print-file-name=libstdc++.so.6"], 5, stderr=None))
        if not library.is_file():
            raise OSError(f"unresolved libstdc++: {library}")
        digest = hashlib.sha256(version.encode() + b"\0" + library.read_bytes())
        return f"cxx-{digest.hexdigest()[:16]}"
    except (IndexError, OSError, ValueError, subprocess.SubprocessError):
        logging.warning("JIT_CACHE_FAIL_OPEN: C++ probe failed", exc_info=True)


def _torch_scope(accelerator: str, cpp: str) -> str | None:
    from rtp_llm.utils.util import torch_abi_fingerprint

    if not (fingerprint := torch_abi_fingerprint()):
        logging.warning("JIT_CACHE_FAIL_OPEN: torch ABI flag unavailable")
        return None
    machine = f"{platform.system()}-{platform.machine()}"
    soabi = sysconfig.get_config_var("SOABI") or machine
    parts = (soabi, "-".join(platform.libc_ver()), accelerator, fingerprint[0], cpp)
    return "-".join(map(str, (*parts, fingerprint[1])))


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

    from rtp_llm.utils.util import COMPILE_FLAG_ENVS

    version = torch.version.hip or torch.version.cuda
    if not version:
        return None
    backend = ROCM if torch.version.hip else CUDA
    accelerator = _accelerator_scope(backend, version)
    toolkit, cpp = _toolkit_scope(backend), _cpp_runtime_scope()
    torch_scope = accelerator and toolkit and cpp and _torch_scope(accelerator, cpp)
    if not torch_scope:
        return None
    scopes = {"accelerator": accelerator, "torch": torch_scope, "cxx": cpp}
    selected, keys = [], [f"toolkit-{toolkit}", accelerator, torch_scope]
    flags = "\0".join(os.environ.get(name, "") for name in COMPILE_FLAG_ENVS)
    if flags.strip("\0"):
        keys.append("flags-" + hashlib.sha256(flags.encode()).hexdigest()[:12])
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
    scope_id = hashlib.sha256("\0".join(keys).encode()).hexdigest()[:16]
    names = ",".join(x.name for x in selected)
    logging.info("JIT scope %s at %s: %s", scope_id, local_root, names)
    root = local_root / RTP_JIT_VERSION / scope_id
    components = tuple(replace(item, local_dir=root / item.name) for item in selected)
    return Scope(scope_id, root, components)


def setup_jit_cache_env() -> Scope | None:
    try:
        local_root = Path(os.getenv("RTP_JIT_LOCAL_ROOT", "").strip() or LOCAL_JIT_ROOT)
        os.umask(0)  # all local JIT producers must create files for shared users
        for path in (local_root.parent, local_root):
            path.mkdir(parents=True, exist_ok=True)
            if not stat.S_ISDIR(os.lstat(path).st_mode):  # a symlink retargets chmod
                raise OSError(f"untrusted JIT root {path}")
            with suppress(OSError):
                os.chmod(path, 0o1777)  # sticky: co-tenants add, never swap, the tree
        with suppress(OSError):
            for path in (p for p in local_root.rglob("*") if not p.is_symlink()):
                with suppress(OSError):
                    os.chmod(path, 0o777 if path.is_dir() else 0o666)
        scope = resolve_scope(local_root)
        for item in scope.components if scope else ():
            os.environ[item.env_name] = str(item.local_dir)
        return scope
    except OSError as error:
        logging.warning("JIT_CACHE_FAIL_OPEN: local root unavailable: %s", error)
    except Exception:
        logging.exception("JIT_CACHE_FAIL_OPEN: env setup failed")


class JitCacheManager(FileSystemEventHandler):
    def __init__(self, scope: Scope, remote_value: str):
        self.scope, self._remote_value, self.store = scope, remote_value, None
        self._observer = self._worker = None
        self._prepare_thread = self._cleanup_thread = None
        self._dirty, self._stop = threading.Event(), threading.Event()
        self._lock, self._prepared = threading.Lock(), queue.Queue(maxsize=1)

    def _discard_prepared(self) -> None:
        with suppress(queue.Empty):
            remote_store, prepared, restore_fd = self._prepared.get_nowait()
            if prepared:
                shutil.rmtree(prepared.staging, ignore_errors=True)
            if remote_store:
                remote_store.close()
            if restore_fd is not None:
                os.close(restore_fd)

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
        return self.scope.root.parent / ".locks" / f"{self.scope.scope_id}.{kind}.lock"

    def _prepare(self, staging_root: Path) -> None:
        remote_store, prepared, restore_fd = None, None, None
        root = self.scope.root
        try:
            if not store.scope_root_usable(root):
                restore_fd = store.acquire_flock(self._lock_path("restore"))
            if not self._stop.is_set():
                remote_store = store.resolve_remote(
                    self._remote_value, RTP_JIT_VERSION, self.scope.scope_id
                )
                cold = remote_store and restore_fd is not None
                if cold and not store.scope_root_usable(root):
                    shutil.rmtree(staging_root, ignore_errors=True)
                    prepared = remote_store.prepare_restore(staging_root)
        except Exception:
            logging.exception("JIT_CACHE_FAIL_OPEN: restore failed")
        self._prepared.put((remote_store, prepared, restore_fd))
        if self._stop.is_set():
            self._discard_prepared()

    def bootstrap(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        staging_root = self.scope.root.parent / ".staging" / self.scope.scope_id
        self._prepare_thread = threading.Thread(
            target=self._prepare, args=(staging_root,), daemon=True
        )
        self._prepare_thread.start()
        try:
            result = self._prepared.get(timeout=max(0, deadline - time.monotonic()))
        except queue.Empty:
            self._stop.set()
            threading.Thread(target=self._discard_prepared, daemon=True).start()
            logging.warning("JIT_CACHE_FAIL_OPEN: setup timed out")
            return False
        self.store, prepared, restore_fd = result
        try:
            if not self.store:
                self._stop.set()
                return False
            root = self.scope.root
            restored = bool(prepared and store.commit_restore(prepared.staging, root))
            if restored:
                logging.info("JIT_CACHE_RESTORED: %s", prepared.snapshot.name)
        finally:
            if restore_fd is not None:
                os.close(restore_fd)
        if not restored:
            with suppress(OSError):  # seed an empty remote once; restarts stay quiet
                if not any(self.store.remote_root.glob(f"*{store.SNAPSHOT_SUFFIX}")):
                    self._dirty.set()
            # torch/aiter FileBatons deadlock if left; flock "*.lock" self-releases.
            for item in self.scope.components:
                if item.name in ("torch_extensions", "aiter"):
                    store.reap_stale_batons(item.local_dir)
        return self._start_watch()

    def _start_watch(self) -> bool:
        self._observer = Observer()
        try:
            for item in self.scope.components:
                item.local_dir.mkdir(parents=True, exist_ok=True)
                self._observer.schedule(self, str(item.local_dir), recursive=True)
            self._observer.start()
        except Exception:
            logging.exception("JIT_CACHE_FAIL_OPEN: inotify watch setup failed")
            return False
        self._worker = threading.Thread(target=self._sync_loop, daemon=True)
        self._worker.start()
        return True

    def _snapshot_files(self) -> dict[str, Path]:
        files = {}
        for item in self.scope.components:
            for path in item.local_dir.rglob("*"):
                with suppress(OSError):
                    st, rel = path.lstat(), path.relative_to(item.local_dir).as_posix()
                    if (
                        st.st_size
                        and stat.S_ISREG(st.st_mode)
                        and item.should_sync(rel)
                    ):
                        files[f"{item.name}/{rel}"] = path
        return files

    def publish_pending_snapshot(self) -> None:
        if not self._dirty.is_set():
            return
        publisher_fd = store.acquire_flock(self._lock_path("publish"), blocking=False)
        if publisher_fd is None:
            return
        try:
            self._dirty.clear()
            self.store.publish_snapshot(self._snapshot_files())
        except Exception as error:
            self._dirty.set()
            if not isinstance(error, store.SnapshotRaced):
                raise
            logging.info("JIT snapshot deferred; builder active during pack: %s", error)
        finally:
            os.close(publisher_fd)

    def _sync_loop(self) -> None:
        while True:
            stopping = self._stop.wait(SYNC_POLL_S)
            try:
                self.publish_pending_snapshot()
            except Exception:
                logging.exception("JIT_CACHE_FAIL_OPEN: snapshot sync failed")
            if stopping:
                return

    def _cleanup(self) -> None:
        try:
            for worker in (self._observer, self._worker, self._prepare_thread):
                if worker:
                    with suppress(RuntimeError):
                        worker.join()
        finally:
            self._discard_prepared()
            if self.store:
                self.store.close()
            with self._lock:
                self._cleanup_thread = None

    def stop(self) -> None:
        self._stop.set()
        if self._observer:
            self._observer.stop()
        with self._lock:
            if self._cleanup_thread is None:
                cleanup_thread = threading.Thread(target=self._cleanup, daemon=True)
                self._cleanup_thread = cleanup_thread
                cleanup_thread.start()
            cleanup_thread = self._cleanup_thread
        cleanup_thread.join(STOP_TIMEOUT_S)
        if cleanup_thread.is_alive():
            logging.warning("JIT cleanup continues in background")
            if self.store:
                threading.Thread(target=self.store.close, daemon=True).start()


def start_from_config(config):
    remote = str(config.remote_jit_dir or "").strip()
    if (scope := setup_jit_cache_env()) is None or not remote:
        return
    manager = JitCacheManager(scope, remote)
    try:
        if manager.bootstrap(config.jit_cache_setup_timeout_s):
            return manager
    except Exception:
        logging.exception("JIT_CACHE_FAIL_OPEN: bootstrap failed; cold start")
    except BaseException:
        manager.stop()
        raise
    manager.stop()
