from __future__ import annotations

import logging
import os
import sys
import tarfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import zstandard as zstd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.models_py.triton_kernels.autotune_cache import get_gpu_info
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.time_util import current_time_ms

SNAPSHOT_PREFIX = ".jit_snapshot_"
SNAPSHOT_SUFFIX = ".tar.zst"
SNAPSHOT_KEEP_COUNT = 2
SNAPSHOT_PUBLISH_LEASE_PREFIX = ".jit_snapshot_publish_lease."
SNAPSHOT_PUBLISH_INTERVAL_S = 20 * 60
LOCAL_LOCK_NAME = ".rtp_jit_local.lock"
SNAPSHOT_COMPLETE_NAME = ".rtp_jit_snapshot_complete"

DEFAULT_JIT_SYNC_WORKERS = 8
COPY_CHUNK_SIZE = 4 * 1024 * 1024
STARTUP_WAIT_POLL_S = 0.3
WATCHDOG_JOIN_TIMEOUT_S = 2.0
PERIODIC_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    env_name: str
    # .ptx / .ttir / .ttgir / .llir are triton IR intermediates not needed at runtime.
    sync_suffixes: tuple[str, ...] = (".so", ".cubin", ".json")
    upload_events: frozenset[str] = frozenset({"closed"})
    gpu_scoped: bool = False
    import_time_sensitive: bool = False


COMPONENT_SPECS = (
    ComponentSpec(
        "flashinfer", "FLASHINFER_WORKSPACE_BASE", import_time_sensitive=True
    ),
    ComponentSpec(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        upload_events=frozenset({"created"}),
        import_time_sensitive=True,
    ),
    ComponentSpec("torch_extensions", "TORCH_EXTENSIONS_DIR"),
    ComponentSpec("triton", "TRITON_CACHE_DIR", upload_events=frozenset({"moved"})),
    ComponentSpec(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        sync_suffixes=(".json", ".pkl", ".pickle"),
        gpu_scoped=True,
    ),
)
COMPONENT_BY_NAME = {c.name: c for c in COMPONENT_SPECS}
_DEEP_GEMM_STALE_LOCK_GLOB = "**/*_lock"
_TRITON_AUTOTUNE_EXTRA_ENV = (("TRITON_AUTOTUNE_CACHE_MODE", "cached"),)


@dataclass
class SyncStats:
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    skipped_files: int = 0
    failed_files: int = 0

    def as_summary(self) -> dict[str, int]:
        return {
            "candidate_files": self.uploaded_files
            + self.skipped_files
            + self.failed_files,
            "uploaded_files": self.uploaded_files,
            "uploaded_bytes": self.uploaded_bytes,
            "skipped_files": self.skipped_files,
            "failed_files": self.failed_files,
        }


def aggregate_sync_stats(stats: dict[str, SyncStats]) -> dict[str, Any]:
    total = SyncStats()
    components: dict[str, dict[str, int]] = {}
    for name, s in stats.items():
        row = s.as_summary()
        if any(row.values()):
            components[name] = row
        total.uploaded_files += s.uploaded_files
        total.uploaded_bytes += s.uploaded_bytes
        total.skipped_files += s.skipped_files
        total.failed_files += s.failed_files
    return {"components": components, **total.as_summary()}


@dataclass
class JitCacheConfig:
    remote_root: Path | None
    local_root: Path
    remote_sync_timeout_s: float
    sync_workers: int = DEFAULT_JIT_SYNC_WORKERS

    @classmethod
    def from_config(cls, jit_config: JITConfig | None = None) -> JitCacheConfig:
        if jit_config is None:
            jit_config = JITConfig()
        return cls(
            remote_root=resolve_remote_root(jit_config.remote_jit_dir),
            local_root=normalize_local_path(jit_config.local_jit_dir),
            remote_sync_timeout_s=jit_config.remote_sync_timeout_s,
        )

    @property
    def remote_sync_longer_timeout_s(self) -> float:
        # 5x is for longer-running ops: snapshot pull / periodic sync / snapshot publish.
        return 5 * self.remote_sync_timeout_s


def normalize_local_path(p: str | Path) -> Path:
    return Path(p).expanduser().absolute()


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        try:
            mounted_path = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
            if not mounted_path:
                raise ValueError(f"failed to mount remote_jit_dir: {text!r}")
            text = mounted_path
        except Exception as e:
            logging.warning(
                "failed to mount remote_jit_dir %r: %s; remote JIT cache is disabled",
                text,
                e,
            )
            return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"remote_jit_dir must be an absolute path, got {text!r}")
    if not path.is_dir():
        logging.warning(
            "remote_jit_dir %r is not an existing directory; remote JIT cache is disabled",
            text,
        )
        return None
    return path


def component_cache_dir(root: Path, component: ComponentSpec) -> Path:
    return (
        root / component.name / get_gpu_info()
        if component.gpu_scoped
        else root / component.name
    )


def strip_gpu_prefix(component: ComponentSpec, rel: str) -> str:
    """Turn a tar/remote path (post `<name>/`, includes `<gpu>/` for gpu_scoped
    components) into the path relative to component_dirs[name][0], which
    already contains the gpu subdir."""
    if component.gpu_scoped and "/" in rel:
        return rel.split("/", 1)[1]
    return rel


def apply_jit_cache_env(local_root: Path | str) -> None:
    """Must be called BEFORE importing deep_gemm/flashinfer.

    Both freeze their cache dir at import time (deep_gemm in C++ init,
    flashinfer in a module-level Path); triton/torch_extensions read env dynamically.
    """
    root = normalize_local_path(local_root)
    for component in COMPONENT_SPECS:
        if component.import_time_sensitive and component.name in sys.modules:
            logging.warning(
                "%s already imported before apply_jit_cache_env, "
                "cache dir env var may not take effect",
                component.name,
            )
        managed = str(component_cache_dir(root, component))
        preset = os.environ.get(component.env_name)
        # Respect caller-preset env; JCM bootstrap adapts component_dirs to follow it.
        if preset and preset != managed:
            logging.info(
                "%s preset to %r (JCM will follow the preset instead of managed %r)",
                component.env_name,
                preset,
                managed,
            )
        os.environ.setdefault(component.env_name, managed)
    for env_key, env_value in _TRITON_AUTOTUNE_EXTRA_ENV:
        os.environ.setdefault(env_key, env_value)


def clear_component_startup_files(
    component: ComponentSpec, component_dir: Path
) -> None:
    # Only deep_gemm leaves lock files (from its C++ FileLock) — the rest have no
    # startup-time cleanup to do.
    if component.name != "deep_gemm":
        return
    cleared = 0
    for path in component_dir.glob(_DEEP_GEMM_STALE_LOCK_GLOB):
        if not path.is_file():
            continue
        try:
            path.unlink(missing_ok=True)
            cleared += 1
        except OSError:
            logging.warning(
                "failed to remove deep_gemm startup file: %s",
                path,
                exc_info=True,
            )
    if cleared:
        logging.info(
            "removed %d deep_gemm startup files from %s", cleared, component_dir
        )


def tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


@contextmanager
def atomic_write_path(path: Path) -> Iterator[Path]:
    tmp = tmp_sibling(path)
    replaced = False
    try:
        yield tmp
        try:
            tmp.replace(path)
        except PermissionError:
            # FUSE rejects rename-over-existing with EPERM.
            path.unlink(missing_ok=True)
            tmp.replace(path)
        replaced = True
    finally:
        # Skip unlink after successful replace — avoids wasted FUSE RPC.
        if not replaced:
            with suppress(OSError):
                tmp.unlink()


def should_sync_file(component: ComponentSpec, rel: str) -> bool:
    if not rel.endswith(component.sync_suffixes):
        return False
    return not any(part.startswith("tmp.pid_") for part in rel.split("/"))


def iter_component_sync_files(
    root: Path,
    component: ComponentSpec,
    *,
    log_errors: bool = False,
) -> Iterator[tuple[Path, str]]:
    try:
        if not root.is_dir():
            return
    except OSError:
        if log_errors:
            logging.warning(
                "failed to probe component dir %s; skipping in snapshot",
                component.name,
                exc_info=True,
            )
        return

    def onerror(error: OSError) -> None:
        if log_errors:
            logging.warning("os.walk error during JIT snapshot scan: %s", error)

    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            if not should_sync_file(component, rel):
                continue
            # Skip 0-byte artifacts: partial/broken writes. .so/.cubin/.json can't
            # be legitimately empty, and every downstream caller had to filter these.
            try:
                if path.stat(follow_symlinks=False).st_size <= 0:
                    continue
            except OSError:
                continue
            yield path, rel


def is_snapshot_name(name: str) -> bool:
    return name.startswith(SNAPSHOT_PREFIX) and name.endswith(SNAPSHOT_SUFFIX)


def new_snapshot_name() -> str:
    return f"{SNAPSHOT_PREFIX}{int(current_time_ms())}_{uuid.uuid4().hex[:8]}{SNAPSHOT_SUFFIX}"


def find_latest_snapshot(remote_root: Path) -> Path | None:
    # Names are timestamped, so lexical max IS the latest. atomic_write_path +
    # cleanup_remote_root guarantee any listed snapshot is either fully written
    # or a stale tmp (filtered by is_snapshot_name), so no per-entry stat is needed.
    try:
        latest = max(
            (e for e in remote_root.iterdir() if is_snapshot_name(e.name)),
            key=lambda p: p.name,
            default=None,
        )
    except OSError:
        return None
    return latest


def cleanup_remote_root(
    remote_root: Path,
    keep_snapshots: int = SNAPSHOT_KEEP_COUNT,
    current_lease_bucket: int | None = None,
) -> None:
    tmp_prefix = f".{SNAPSHOT_PREFIX}"
    stale_cutoff = (
        current_lease_bucket - 2 if current_lease_bucket is not None else None
    )
    snapshots: list[Path] = []
    try:
        for entry in remote_root.iterdir():
            name = entry.name
            if is_snapshot_name(name):
                snapshots.append(entry)
            elif (
                name.startswith(tmp_prefix)
                and f"{SNAPSHOT_SUFFIX}." in name
                and name.endswith(".tmp")
            ):
                with suppress(OSError):
                    entry.unlink()
            elif stale_cutoff is not None and name.startswith(
                SNAPSHOT_PUBLISH_LEASE_PREFIX
            ):
                try:
                    bucket = int(name.removeprefix(SNAPSHOT_PUBLISH_LEASE_PREFIX))
                except ValueError:
                    continue
                if bucket < stale_cutoff:
                    with suppress(OSError):
                        entry.rmdir()
    except OSError:
        return
    for old in sorted(snapshots)[:-keep_snapshots]:
        with suppress(OSError):
            old.unlink()


def stream_copy(source: Any, out: Any, deadline_s: float, buffer: bytearray) -> None:
    view = memoryview(buffer)
    while True:
        if time.monotonic() >= deadline_s:
            raise TimeoutError("JIT cache file copy exceeded deadline")
        read_size = source.readinto(buffer)
        if not read_size:
            return
        out.write(view[:read_size])


def copy_with_deadline(src: Path, dst: Path, deadline_s: float) -> None:
    buffer = bytearray(COPY_CHUNK_SIZE)
    with src.open("rb") as source, dst.open("wb") as out:
        stream_copy(source, out, deadline_s, buffer=buffer)


@contextmanager
def open_snapshot_reader(path: Path) -> Iterator[tarfile.TarFile]:
    with (
        path.open("rb") as compressed,
        zstd.ZstdDecompressor().stream_reader(compressed) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        yield tar


@contextmanager
def open_snapshot_writer(path: Path) -> Iterator[tarfile.TarFile]:
    with (
        path.open("wb") as raw,
        zstd.ZstdCompressor(level=3).stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        yield tar


class _JitFileEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        component: ComponentSpec,
        root_prefix: str,
        enqueue: Callable[[ComponentSpec, str], bool],
    ):
        self.component = component
        self.root_prefix = root_prefix
        self.enqueue = enqueue

    def on_any_event(self, event: Any) -> None:
        if event.is_directory or event.event_type not in self.component.upload_events:
            return
        src = event.dest_path if event.event_type == "moved" else event.src_path
        if not src.startswith(self.root_prefix):
            return
        rel = src[len(self.root_prefix) :].replace(os.sep, "/")
        if not should_sync_file(self.component, rel):
            return
        if event.event_type == "created":
            try:
                if os.stat(src).st_size == 0:
                    return
            except OSError:
                return
        self.enqueue(self.component, rel)


class JitDirtyTracker:
    def __init__(
        self,
        component_dirs: dict[str, tuple[Path, Path]],
        enqueue_upload: Callable[[ComponentSpec, str], bool],
    ):
        self.component_dirs = component_dirs
        self.enqueue_upload = enqueue_upload
        self.observer: Any | None = None

    def _make_handler(
        self, component: ComponentSpec, root: Path
    ) -> _JitFileEventHandler:
        return _JitFileEventHandler(component, str(root) + os.sep, self.enqueue_upload)

    def start(self) -> bool:
        if self.observer is not None:
            return True
        observer = Observer()
        try:
            for name, (local_dir, _) in self.component_dirs.items():
                handler = self._make_handler(COMPONENT_BY_NAME[name], local_dir)
                observer.schedule(handler, str(local_dir), recursive=True)
            observer.start()
            self.observer = observer
            return True
        except Exception:
            logging.warning(
                "failed to start JIT cache dirty watcher; remote writeback is disabled",
                exc_info=True,
            )
            with suppress(Exception):
                observer.stop()
                observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
            return False

    def stop(self) -> None:
        observer = self.observer
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
        self.observer = None
