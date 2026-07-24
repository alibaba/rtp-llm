import hashlib
import importlib.metadata
import logging
import os
import platform
import shlex
import subprocess
import sysconfig
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.utils.jit_cache_store import (
    RemoteSnapshotStore,
    is_lock_file,
    reap_dead_lock,
    sanitize,
)

# Quiet period before publishing, so a burst of artifacts coalesces into one upload.
SYNC_POLL_S = 120.0
# Bounded shutdown; an unfinished upload is dropped and republished on next cold start.
SHUTDOWN_TIMEOUT_S = 10
RTP_JIT_VERSION = "v1"
# One shared tree per host; restore()'s flock + .ready marker let the first cold start
# fill it and the rest reuse it (artifacts are content-addressed; builders self-coordinate).
LOCAL_JIT_DIR = f"/tmp/rtp-llm/.jit_cache/{RTP_JIT_VERSION}"
CUDA, ROCM = "cuda", "rocm"


def resolve_remote_root(value) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

        text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW, True)
    base = Path(text).expanduser().absolute()
    if not base.is_dir():
        logging.warning("JIT remote cache disabled: invalid directory %s", base)
        return None
    root = base / RTP_JIT_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def _token(text: str) -> str | None:
    return sanitize(text, "") or None


def _pkg_version(name: str) -> str | None:
    try:
        return _token(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError:
        return None


def _accelerator_scope(backend: str) -> str | None:
    import torch

    try:
        device = torch.cuda.current_device()
        if backend == CUDA:
            prefix = "sm_{}{}".format(*torch.cuda.get_device_capability(device))
            version = torch.version.cuda
        else:
            arch = str(torch.cuda.get_device_properties(device).gcnArchName)
            prefix = _token(arch.split(":", 1)[0])
            version = torch.version.hip
        if not version or not prefix:
            return None
        return f"{backend}-{str(version).replace('.', '_')}-{prefix}"
    except Exception:
        logging.warning("failed to detect GPU architecture", exc_info=True)
        return None


def _cpp_runtime_scope() -> str | None:
    try:
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        version = subprocess.check_output(
            [*compiler, "--version"], text=True, timeout=5
        ).splitlines()[0]
        library = Path(
            subprocess.check_output(
                [*compiler, "-print-file-name=libstdc++.so.6"], text=True, timeout=5
            ).strip()
        )
        if not library.is_file():
            raise OSError(f"unresolved libstdc++: {library}")
        digest = hashlib.sha256(version.encode() + b"\0" + library.read_bytes())
        return f"cxx-{digest.hexdigest()[:16]}"
    except (IndexError, OSError, ValueError, subprocess.SubprocessError):
        logging.warning("failed to identify C++ runtime for JIT cache", exc_info=True)
        return None


def _torch_scope(accelerator: str | None, cpp: str | None) -> str | None:
    import torch

    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    host = f"{platform.system().lower()}-{platform.machine().lower()}"
    soabi = _token(sysconfig.get_config_var("SOABI") or host)
    libc = _token("-".join(platform.libc_ver()))
    version = _token(torch.__version__)
    if not isinstance(abi, bool) or not all((accelerator, cpp, soabi, libc, version)):
        return None
    return f"{soabi}-{libc}-{accelerator}-torch-{version}-cxxabi-{int(abi)}-{cpp}"


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    rules: tuple
    scopes: tuple[str, ...] = ()
    backend: str | None = None
    local_dir: Path = Path()

    def should_sync(self, rel: str, event_type: str | None = None) -> bool:
        events = next(
            (events for suffixes, events in self.rules if rel.endswith(suffixes)), ()
        )
        parts = rel.split("/")
        return (
            (event_type in events if event_type else bool(events))
            and ".." not in parts
            and not any(p == "tmp" or p.startswith("tmp.pid_") for p in parts)
        )


def rule(events: frozenset[str], *suffixes: str):
    return suffixes, events


NINJA = (".so", ".o", "build.ninja", ".ninja_log", ".ninja_deps")
MOVED = frozenset({"moved"})
CLOSED = MOVED | {"closed"}
CREATED = CLOSED | {"created"}

# fmt: off
COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE",
              (rule(CREATED, ".cu", ".inc", ".h"), rule(CLOSED, *NINJA)),
              ("torch", "@flashinfer-python"), CUDA),
    Component("deep_gemm", "DG_JIT_CACHE_DIR",
              (rule(CREATED, "kernel.cu", "kernel.cubin"),),
              ("accelerator", "@deep_gemm"), CUDA),
    Component("trtllm_deep_gemm", "TRTLLM_DG_CACHE_DIR",
              (rule(CREATED, "nvcc_kernel.cubin"),),
              ("accelerator", "@flashinfer-python"), CUDA),
    Component("tilelang", "TILELANG_CACHE_DIR",
              (rule(CLOSED, ".so", ".pkl", ".cu", ".json", ".cubin", ".py"),),
              ("torch", "@tilelang"), CUDA),
    Component("torch_extensions", "TORCH_EXTENSIONS_DIR",
              (rule(CLOSED, *NINJA, ".cpp", ".cu"),), ("torch",)),
    Component("aiter", "AITER_JIT_DIR",
              (rule(CLOSED, *NINJA, ".cu", ".cpp", ".hip", ".h"),),
              ("torch", "@aiter"), ROCM),
    Component("flydsl", "FLYDSL_RUNTIME_CACHE_DIR", (rule(MOVED, ".pkl"),),
              ("accelerator", "@flydsl"), ROCM),
    Component("tvm_ffi", "TVM_FFI_CACHE_DIR", (rule(CREATED, ".so"),),
              ("torch", "@apache-tvm-ffi"), CUDA),
    Component("cute_dsl", "CUTE_DSL_CACHE_DIR", (rule(MOVED, ".mlir"),),
              ("accelerator", "@nvidia-cutlass-dsl"), CUDA),
    # triton keys on src+machine only; cxx scope guards against incompatible .so.
    Component("triton", "TRITON_CACHE_DIR",
              (rule(frozenset(), ".autotune.json"), rule(CREATED, ".json", ".cubin", ".hsaco", ".so")),
              ("cxx",)),
)
# fmt: on


def _resolve_components() -> tuple[Component, ...]:
    import torch

    root = Path(LOCAL_JIT_DIR)
    backend = ROCM if torch.version.hip else CUDA if torch.version.cuda else None
    if not backend:
        return ()
    accelerator, cpp = _accelerator_scope(backend), _cpp_runtime_scope()
    scopes = {
        "accelerator": accelerator,
        "torch": _torch_scope(accelerator, cpp),
        "cxx": cpp,
    }
    result = []
    for item in COMPONENTS:
        if item.backend not in (None, backend):
            continue
        parts = tuple(
            _pkg_version(part[1:]) if part.startswith("@") else scopes[part]
            for part in item.scopes
        )
        if parts and all(parts):
            local = root / item.name / "-".join((item.name, *parts))
            result.append(replace(item, local_dir=local))
    return tuple(result)


def clear_jit_locks() -> None:
    # Reap dead builder locks so the next load() doesn't hang on a corpse left by
    # a killed build; reap_dead_lock spares any a live process still holds.
    try:
        for lock in Path(LOCAL_JIT_DIR).rglob("*lock"):
            if is_lock_file(lock.name):
                reap_dead_lock(lock)
    except OSError:
        logging.warning("JIT lock cleanup skipped", exc_info=True)


def setup_jit_cache_env() -> tuple[tuple[Component, ...], bool]:
    try:
        components = _resolve_components()
    except Exception:
        logging.exception("JIT cache environment setup failed; using upstream defaults")
        return (), False
    clear_jit_locks()
    managed = []
    for item in components:
        local = str(item.local_dir)
        resolved = os.environ.setdefault(item.env_name, local)
        if resolved == local:
            managed.append(item)
        else:
            logging.warning(
                "JIT %s uses preset %s=%s; not managed",
                item.name,
                item.env_name,
                resolved,
            )
    return tuple(managed), any("torch" in item.scopes for item in managed)


class _EventHandler(FileSystemEventHandler):
    def __init__(self, manager: "JitCacheManager"):
        self.manager = manager

    def on_any_event(self, event) -> None:
        if event.is_directory or self.manager._stop.is_set():
            return
        path = Path(event.dest_path if event.event_type == "moved" else event.src_path)
        for component in self.manager.components:
            if component.local_dir not in path.parents:
                continue
            with suppress(OSError, ValueError):
                rel = path.relative_to(component.local_dir).as_posix()
                if component.should_sync(rel, event.event_type) and path.stat().st_size:
                    self.manager._last_event_at = time.monotonic()
                    self.manager._dirty.set()
                    return


class JitCacheManager:
    def __init__(self, remote_root: Path, components: tuple[Component, ...]):
        self.local_root = Path(LOCAL_JIT_DIR)
        self.components = components
        self.store = RemoteSnapshotStore(remote_root)
        self._dirty, self._stop = threading.Event(), threading.Event()
        self._last_event_at = 0.0
        self._observer = self._sync_thread = None

    def start_background_sync(self, cancel=None, commit=None) -> None:
        if self._observer:
            return
        try:
            if self.store.restore(self.local_root, cancel, commit):
                logging.info("loaded JIT cache from remote snapshot")
        except Exception:
            logging.exception("JIT cache restore failed")
        if cancel and cancel.is_set():
            return

        observer = Observer()
        try:
            for component in self.components:
                component.local_dir.mkdir(parents=True, exist_ok=True)
            observer.schedule(_EventHandler(self), str(self.local_root), recursive=True)
            observer.start()
        except Exception:
            observer.stop()
            raise
        self._observer = observer
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def _snapshot_files(self) -> dict[str, Path]:
        files = {}
        # All known components, not self.components: an archive is one complete
        # generation of the shared tree, which co-located processes with a
        # different managed set may also populate.
        for component in COMPONENTS:
            root = self.local_root / component.name
            for path in root.rglob("*"):
                with suppress(OSError):  # file may vanish between walk and stat
                    if path.is_symlink() or not path.is_file():
                        continue
                    rel = path.relative_to(root).as_posix()
                    if component.should_sync(rel) and path.stat().st_size:
                        files[path.relative_to(self.local_root).as_posix()] = path
        return files

    def publish_pending_snapshot(self) -> None:
        if not self._dirty.is_set():
            return
        self._dirty.clear()
        try:
            if not self.store.publish_snapshot(self._snapshot_files):
                self._dirty.set()  # deferred: retry after the next quiet period
        except Exception:
            self._dirty.set()
            raise

    def _sync_loop(self) -> None:
        while True:
            stopping = self._stop.wait(SYNC_POLL_S)
            if not stopping and time.monotonic() - self._last_event_at < SYNC_POLL_S:
                continue
            try:
                self.publish_pending_snapshot()
            except Exception:
                logging.exception("JIT cache sync failed")
            if stopping:
                return

    def stop(self) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        self._stop.set()
        if self._observer:
            self._observer.stop()
            self._observer.join(max(0, deadline - time.monotonic()))
            self._observer = None
        if self._sync_thread:
            self._sync_thread.join(max(0, deadline - time.monotonic()))
            self._sync_thread = None
