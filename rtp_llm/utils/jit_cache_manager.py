import logging
import os
import queue
import re
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.utils.jit_cache_store import RemoteSnapshotStore

LOCAL_JIT_DIR = ".jit_cache"
SYNC_POLL_S = 60.0


def _safe_part(value: str, fallback: str = "unknown") -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or fallback


def _dist_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _cuda_scope() -> str:
    import torch

    return "cuda-" + _safe_part(str(torch.version.cuda or "unknown"))


def _torch_extensions_scope() -> str:
    import torch

    py = f"py{sys.version_info[0]}{sys.version_info[1]}{getattr(sys, 'abiflags', '')}"
    cuda = "cu" + torch.version.cuda.replace(".", "") if torch.version.cuda else "cpu"
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    abi = str(int(abi)) if isinstance(abi, bool) else "unknown"
    return f"torch-{_safe_part(_dist_version('torch'))}-{_safe_part(f'{py}_{cuda}')}-cxxabi-{_safe_part(abi)}"


@dataclass(frozen=True)
class Component:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...]
    upload_events: frozenset[str]
    scope_func: Callable[[], str | None] | None = None
    local_dir: Path = Path()

    def resolve(self, root: Path) -> "Component":
        # Scope subdir (torch/cuda/abi) so caches never restore onto an incompatible toolchain.
        scope = self.scope_func() if self.scope_func else None
        local_dir = root / self.name
        return replace(self, local_dir=local_dir / scope if scope else local_dir)

    def should_sync(self, rel: str) -> bool:
        parts = rel.split("/")
        return (
            rel.endswith(self.sync_suffixes)
            and not (self.name == "triton" and rel.endswith(".autotune.json"))
            and rel == rel.lstrip("/")
            and ".." not in parts
            and not any(p == "tmp" or p.startswith("tmp.pid_") for p in parts)
        )


COMPONENTS = (
    Component(
        "flashinfer",
        "FLASHINFER_WORKSPACE_BASE",
        (".so", ".o", "build.ninja", ".ninja_log", ".ninja_deps")
        + (".cu", ".inc", ".h"),
        frozenset({"closed", "moved"}),
        _cuda_scope,
    ),
    Component(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        ("kernel.cu", "kernel.cubin"),
        frozenset({"created", "moved"}),
        lambda: "deep_gemm-" + _safe_part(_dist_version("deep_gemm")),
    ),
    Component(
        "tensorrt_llm_deep_gemm",
        "TRTLLM_DG_CACHE_DIR",
        ("nvcc_kernel.cubin",),
        frozenset({"created", "moved"}),
        lambda: f"{_cuda_scope()}-flashinfer-{_safe_part(_dist_version('flashinfer-python'))}",
    ),
    Component(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        (".so", ".o", "build.ninja", ".ninja_log", ".ninja_deps"),
        frozenset({"closed", "moved"}),
        _torch_extensions_scope,
    ),
    Component(
        "triton",
        "TRITON_CACHE_DIR",
        (".json", ".cubin", ".hsaco", ".so"),
        frozenset({"created", "moved"}),
    ),
)


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

        text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW, True)
    path = Path(text).expanduser().absolute()
    if not path.is_dir():
        logging.warning(
            "JIT remote cache disabled: %s does not exist or is not a directory", path
        )
        return None
    return path


def setup_jit_cache_env(local_root: Path | str = LOCAL_JIT_DIR) -> Path:
    root = Path(local_root).expanduser().absolute()
    for component in COMPONENTS:
        if component.env_name not in os.environ:
            os.environ[component.env_name] = str(component.resolve(root).local_dir)
    return root


class _JitFileEventHandler(FileSystemEventHandler):
    def __init__(self, component: Component, stage):
        self.component, self.stage = component, stage
        self.prefix = str(component.local_dir) + os.sep

    def on_any_event(self, event: Any) -> None:
        if event.is_directory:
            return
        path = event.dest_path if event.event_type == "moved" else event.src_path
        if path.startswith(self.prefix):
            rel = path[len(self.prefix) :].replace(os.sep, "/")
            if event.event_type in self.component.upload_events:
                with suppress(OSError):
                    if os.path.getsize(path):
                        self.stage(self.component, rel)


class JitCacheManager:
    def __init__(self, jit_config=None):
        from rtp_llm.config.py_config_modules import JITConfig

        jit_config = jit_config or JITConfig()
        remote = resolve_remote_root(jit_config.remote_jit_dir)
        self.local_root = Path(LOCAL_JIT_DIR).expanduser().absolute()
        self.store = RemoteSnapshotStore(remote) if remote else None
        self.components = tuple(c.resolve(self.local_root) for c in COMPONENTS)
        self.events = queue.SimpleQueue()
        self._stop = threading.Event()
        self._observer: Any | None = None
        self._sync_thread: threading.Thread | None = None

    def bootstrap(self) -> None:
        for component in self.components:
            component.local_dir.mkdir(parents=True, exist_ok=True)

    def start_background_sync(self) -> None:
        if self.store is None or self._observer is not None:
            return
        try:
            if self.store.restore(self.local_root):
                logging.info("loaded JIT cache from remote snapshot")
        except Exception:
            logging.exception("JIT cache restore failed")

        observer = Observer()
        for component in self.components:
            configured_dir = os.environ.get(component.env_name, "")
            if Path(configured_dir).expanduser().absolute() != component.local_dir:
                continue
            observer.schedule(
                _JitFileEventHandler(component, self.stage_delta_file),
                str(component.local_dir),
                recursive=True,
            )
        observer.start()
        self._observer = observer
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def stage_delta_file(self, component: Component, rel_path: str) -> None:
        if not self._stop.is_set() and component.should_sync(rel_path):
            self.events.put((component, rel_path))

    def flush_delta2remote(self) -> None:
        if self.store is None:
            return
        pending: dict[str, tuple[Component, str]] = {}
        while True:
            try:
                component, rel = self.events.get_nowait()
            except queue.Empty:
                break
            name = (component.local_dir / rel).relative_to(self.local_root).as_posix()
            pending[name] = (component, rel)
        if not pending:
            return
        try:
            self.store.publish_snapshot(
                {name: comp.local_dir / rel for name, (comp, rel) in pending.items()}
            )
        except Exception:
            for comp, rel in pending.values():
                self.events.put((comp, rel))
            raise

    def _sync_loop(self) -> None:
        while True:
            stopping = self._stop.wait(SYNC_POLL_S)
            try:
                self.flush_delta2remote()
            except Exception:
                logging.exception("JIT cache sync failed")
            if stopping:
                return

    def stop(self) -> None:
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=2.0)
        self._stop.set()
        thread, self._sync_thread = self._sync_thread, None
        if thread is not None:
            thread.join(timeout=10.0)  # wait for stop flush
