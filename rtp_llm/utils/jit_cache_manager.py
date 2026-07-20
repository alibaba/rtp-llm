import hashlib
import logging
import os
import platform
import queue
import re
import shlex
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.utils.jit_cache_store import RemoteSnapshotStore

# Publish only after JIT output has been quiet this long, so a burst of
# artifacts coalesces into one full-snapshot upload instead of many.
SYNC_POLL_S = 120.0
# Bounded shutdown: the final flush gets only this long, then the process
# hard-exits. An unfinished upload is dropped; the next cold start republishes.
SHUTDOWN_TIMEOUT_S = 10
# A builder's *_lock a SIGKILL couldn't release is stale once untouched this long.
STALE_LOCK_TIMEOUT_S = 3600.0
RTP_JIT_VERSION = "v1"
LOCAL_JIT_DIR = (
    f"{os.environ.get('TEST_TMPDIR', '/tmp')}/rtp_llm/.jit_cache/{RTP_JIT_VERSION}"
)
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


def _safe(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or "unknown"


def _pkg_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return _safe(metadata.version(name))
    except metadata.PackageNotFoundError:
        return None


def _backend() -> str | None:
    import torch

    return ROCM if torch.version.hip else CUDA if torch.version.cuda else None


def _accelerator_scope(backend: str) -> str | None:
    import torch

    if backend not in (CUDA, ROCM):
        return None
    try:
        device = torch.cuda.current_device()
        if backend == CUDA:
            arch = "{}{}".format(*torch.cuda.get_device_capability(device))
            version = torch.version.cuda
            prefix = f"sm_{arch}"
        else:
            props = torch.cuda.get_device_properties(device)
            prefix = _safe(str(props.gcnArchName).split(":", 1)[0])
            version = torch.version.hip
    except Exception:
        logging.warning("failed to detect GPU architecture", exc_info=True)
        return None
    return f"{backend}-{str(version).replace('.', '_')}-{prefix}" if version else None


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


def _torch_scope(accelerator: str | None) -> str | None:
    import sysconfig

    import torch

    cpp = _cpp_runtime_scope()
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    if not accelerator or not cpp or not isinstance(abi, bool):
        return None
    host = f"{platform.system().lower()}-{platform.machine().lower()}"
    soabi = _safe(sysconfig.get_config_var("SOABI") or host)
    libc = _safe("-".join(platform.libc_ver()))
    return (
        f"{soabi}-{libc}-{accelerator}-torch-{_safe(torch.__version__)}-"
        f"cxxabi-{int(abi)}-{cpp}"
    )


Rule = tuple[tuple[str, ...], frozenset[str]]


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    rules: tuple[Rule, ...]
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
            and not any(
                p in {"tmp", ".staging"} or p.startswith("tmp.pid_") for p in parts
            )
        )


def rule(events: frozenset[str], *suffixes: str) -> Rule:
    return suffixes, events


NINJA = (".so", ".o", "build.ninja", ".ninja_log", ".ninja_deps")
MOVED = frozenset({"moved"})
CLOSED = MOVED | {"closed"}
CREATED = CLOSED | {"created"}
IGNORED = frozenset()

# Created-only producers are copied on delayed flush.
# Scope tokens form the compatibility-keyed local directory.
# fmt: off
COMPONENTS = (
    Component("flashinfer", "FLASHINFER_WORKSPACE_BASE",
              (rule(CREATED, ".cu", ".inc", ".h"), rule(CLOSED, *NINJA)),
              ("torch", "flashinfer", "@flashinfer-python"), CUDA),
    Component("deep_gemm", "DG_JIT_CACHE_DIR",
              (rule(CREATED, "kernel.cu", "kernel.cubin"),),
              ("accelerator", "deep_gemm", "@deep_gemm"), CUDA),
    Component("trtllm_deep_gemm", "TRTLLM_DG_CACHE_DIR",
              (rule(CREATED, "nvcc_kernel.cubin"),),
              ("accelerator", "flashinfer", "@flashinfer-python"), CUDA),
    Component("tilelang", "TILELANG_CACHE_DIR",
              (rule(CLOSED, ".so", ".pkl", ".cu", ".json", ".cubin", ".py"),),
              ("torch", "tilelang", "@tilelang"), CUDA),
    Component("torch_extensions", "TORCH_EXTENSIONS_DIR",
              (rule(CLOSED, *NINJA, ".cpp", ".cu"),), ("torch",)),
    Component("aiter", "AITER_JIT_DIR",
              (rule(CLOSED, *NINJA, ".cu", ".cpp", ".hip", ".h"),),
              ("torch", "aiter", "@aiter"), ROCM),
    Component("flydsl", "FLYDSL_RUNTIME_CACHE_DIR", (rule(MOVED, ".pkl"),),
              ("flydsl", "@flydsl", "accelerator"), ROCM),
    Component("tvm_ffi", "TVM_FFI_CACHE_DIR", (rule(CREATED, ".so"),),
              ("torch", "tvm-ffi", "@apache-tvm-ffi"), CUDA),
    Component("cute_dsl", "CUTE_DSL_CACHE_DIR", (rule(MOVED, ".mlir"),),
              ("accelerator", "cutlass-dsl", "@nvidia-cutlass-dsl"), CUDA),
    Component("triton", "TRITON_CACHE_DIR",
              (rule(IGNORED, ".autotune.json"), rule(CREATED, ".json", ".cubin", ".hsaco", ".so"))),
)
# fmt: on


def _resolve_components(root: Path | str | None = None) -> tuple[Component, ...]:
    root, backend = Path(root or LOCAL_JIT_DIR), _backend()
    if not backend:
        return ()
    accelerator = _accelerator_scope(backend)
    scope_values = {"accelerator": accelerator, "torch": _torch_scope(accelerator)}

    def resolve(part: str) -> str | None:
        return (
            scope_values.get(part)
            if part in scope_values
            else _pkg_version(part[1:]) if part.startswith("@") else part
        )

    result = []
    for item in COMPONENTS:
        if item.backend not in (None, backend):
            continue
        parts = tuple(resolve(part) for part in item.scopes)
        if any(part is None for part in parts):
            continue
        local = root / item.name
        result.append(
            replace(item, local_dir=local / "-".join(parts) if parts else local)
        )
    return tuple(result)


def clear_stale_jit_locks() -> None:
    # Builders (deep_gemm/torch/ninja) leave *_lock files a SIGKILL can't release;
    # reap only ones idle past the cutoff so a live builder's lock is never deleted.
    cutoff = time.time() - STALE_LOCK_TIMEOUT_S
    for lock in Path(LOCAL_JIT_DIR).rglob("*"):
        if lock.name.endswith(("_lock", ".lock")):
            with suppress(OSError):
                if lock.is_file() and lock.stat().st_mtime < cutoff:
                    lock.unlink()


def setup_jit_cache_env() -> tuple[tuple[Component, ...], bool]:
    # Local management is always on; REMOTE_JIT_DIR only adds restore/sync. Returns
    # (managed components, remote-compatible?) — compatibility judged on the full
    # set before filtering preset-env components, so a preset path can't disable it.
    try:
        components = _resolve_components()
    except Exception:
        logging.exception("JIT cache environment setup failed; using upstream defaults")
        return (), False
    clear_stale_jit_locks()
    compatible = any("torch" in item.scopes for item in components)
    managed = tuple(
        item
        for item in components
        if os.environ.setdefault(item.env_name, str(item.local_dir))
        == str(item.local_dir)
    )
    return managed, compatible


class _EventHandler(FileSystemEventHandler):
    def __init__(self, component: Component, manager: "JitCacheManager"):
        self.component, self.manager = component, manager

    def on_any_event(self, event) -> None:
        if event.is_directory or self.manager._stop.is_set():
            return
        path = Path(event.dest_path if event.event_type == "moved" else event.src_path)
        with suppress(OSError, ValueError):
            rel = path.relative_to(self.component.local_dir).as_posix()
            if (
                self.component.should_sync(rel, event.event_type)
                and path.is_file()
                and path.stat().st_size
            ):
                name = path.relative_to(self.manager.local_root).as_posix()
                self.manager._last_event_at = time.monotonic()
                self.manager.events.put((name, path))


class JitCacheManager:
    def __init__(self, remote_root: Path | None, components: tuple[Component, ...]):
        self.local_root, self.components = Path(LOCAL_JIT_DIR), components
        self.store = RemoteSnapshotStore(remote_root) if remote_root else None
        self.events = queue.SimpleQueue()
        self._stop = threading.Event()
        self._last_event_at = 0.0
        self._observer = self._sync_thread = None

    def start_background_sync(self) -> None:
        if not self.store or self._observer:
            return
        for component in self.components:
            component.local_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.store.restore(self.local_root):
                logging.info("loaded JIT cache from remote snapshot")
        except Exception:
            logging.exception("JIT cache restore failed")
        self._observer = Observer()
        for component in self.components:
            self._observer.schedule(
                _EventHandler(component, self), str(component.local_dir), recursive=True
            )
        self._observer.start()
        # Event-only best effort: missed events only reduce remote hit rate.
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def flush_delta2remote(self) -> None:
        if not self.store:
            return
        pending = {}
        while True:
            try:
                name, path = self.events.get_nowait()
                pending[name] = path
            except queue.Empty:
                break
        if not pending:
            return
        # Some producers atomically move a completed directory into place. inotify
        # may report only the directory move, so a pure per-file event delta would
        # omit every stable file below it. Once any managed artifact makes the cache
        # dirty, rescan all component roots and publish the complete stable delta.
        for component in self.components:
            for path in component.local_dir.rglob("*"):
                with suppress(OSError, ValueError):
                    rel = path.relative_to(component.local_dir).as_posix()
                    if (
                        component.should_sync(rel)
                        and path.is_file()
                        and path.stat().st_size
                    ):
                        name = path.relative_to(self.local_root).as_posix()
                        pending[name] = path
        try:
            self.store.publish_snapshot(pending)
        except Exception:
            for name, path in pending.items():
                if path.is_file():
                    self.events.put((name, path))
            raise

    def _sync_loop(self) -> None:
        while True:
            stopping = self._stop.wait(SYNC_POLL_S)
            if not stopping and time.monotonic() - self._last_event_at < SYNC_POLL_S:
                continue
            try:
                self.flush_delta2remote()
            except Exception:
                logging.exception("JIT cache sync failed")
            if stopping:
                return

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=SHUTDOWN_TIMEOUT_S)
            self._observer = None
        self._stop.set()
        if self._sync_thread:
            # A snapshot still uploading past this deadline is dropped; its orphaned
            # .upload.*.tmp is reclaimed by a later publish once idle past
            # IDLE_REAP_S, never mid-upload of a live peer.
            self._sync_thread.join(timeout=SHUTDOWN_TIMEOUT_S)
            self._sync_thread = None
