from __future__ import annotations

import json
import logging
import os
import stat
import sys
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import zstandard as zstd
from filelock import FileLock, Timeout
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.models_py.triton_kernels.autotune_cache import get_gpu_info
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local
from rtp_llm.utils.time_util import current_time_ms

SNAPSHOT_PREFIX = ".jit_snapshot_"
SNAPSHOT_SUFFIX = ".tar.zst"
SNAPSHOT_KEEP_COUNT = 2
SNAPSHOT_PUBLISH_LEASE_PREFIX = ".jit_snapshot_publish_lease."
SNAPSHOT_PUBLISH_INTERVAL_S = 20 * 60
PERIODIC_SYNC_INTERVAL_S = 5 * 60
LOCAL_LOCK_NAME = ".rtp_jit_local.lock"
SNAPSHOT_COMPLETE_NAME = ".rtp_jit_snapshot_complete"
SUMMARY_NAME = ".rtp_jit_summary.json"

DEFAULT_JIT_SYNC_WORKERS = 8
COPY_CHUNK_SIZE = 4 * 1024 * 1024
STARTUP_WAIT_POLL_S = 0.3
WATCHDOG_JOIN_TIMEOUT_S = 2.0
DEFAULT_SYNC_SUFFIXES = (".so", ".cubin", ".json")


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...]
    upload_events: frozenset[str]
    gpu_scoped: bool = False
    import_time_sensitive: bool = False
    startup_cleanup_globs: tuple[str, ...] = ()


COMPONENT_SPECS = (
    ComponentSpec(
        "flashinfer",
        "FLASHINFER_WORKSPACE_BASE",
        DEFAULT_SYNC_SUFFIXES,
        frozenset({"closed"}),
        import_time_sensitive=True,
    ),
    ComponentSpec(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        DEFAULT_SYNC_SUFFIXES,
        frozenset({"created"}),
        import_time_sensitive=True,
        startup_cleanup_globs=("**/*_lock",),
    ),
    ComponentSpec(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        DEFAULT_SYNC_SUFFIXES,
        frozenset({"closed"}),
    ),
    ComponentSpec(
        "triton", "TRITON_CACHE_DIR", DEFAULT_SYNC_SUFFIXES, frozenset({"moved"})
    ),
    ComponentSpec(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        (".json", ".pkl", ".pickle"),
        frozenset({"closed"}),
        gpu_scoped=True,
    ),
)
COMPONENT_BY_NAME = {c.name: c for c in COMPONENT_SPECS}


@dataclass
class SyncStats:
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    skipped_files: int = 0
    failed_files: int = 0


@dataclass
class JitCacheConfig:
    remote_root: Path | None
    local_root: Path
    remote_sync_timeout_s: float
    remote_sync_longer_timeout_s: float = 0.0
    sync_workers: int = DEFAULT_JIT_SYNC_WORKERS

    @classmethod
    def from_config(cls, jit_config: JITConfig | None = None) -> JitCacheConfig:
        if jit_config is None:
            jit_config = JITConfig()
        timeout = float(jit_config.remote_sync_timeout_s)
        return cls(
            remote_root=resolve_remote_root(jit_config.remote_jit_cache_dir),
            local_root=Path(jit_config.local_jit_cache_dir).expanduser().absolute(),
            remote_sync_timeout_s=timeout,
            remote_sync_longer_timeout_s=5 * timeout,
        )


def elapsed_ms(start_s: float) -> int:
    return int((time.monotonic() - start_s) * 1000)


def new_jit_cache_run_id() -> str:
    return f"{int(current_time_ms())}-{os.getpid()}-{uuid.uuid4().hex}"


def resolve_remote_root(remote_jit_cache_dir: Any) -> Path | None:
    text = str(remote_jit_cache_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        try:
            mounted_path = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
            if not mounted_path:
                raise ValueError(f"failed to mount remote_jit_cache_dir: {text!r}")
            text = mounted_path
        except Exception as e:
            logging.warning(
                "failed to mount remote_jit_cache_dir %r: %s; remote JIT cache is disabled",
                text,
                e,
            )
            return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"remote_jit_cache_dir must be an absolute path, got {text!r}")
    if not path.is_dir():
        logging.warning(
            "remote_jit_cache_dir %r is not an existing directory; remote JIT cache is disabled",
            text,
        )
        return None
    return path


def component_cache_dir(root: Path, component: ComponentSpec) -> Path:
    component_root = root / component.name
    if component.gpu_scoped:
        return component_root / get_gpu_info()
    return component_root


def apply_jit_cache_env(local_root: Path | str) -> None:
    """Must be called BEFORE importing deep_gemm/flashinfer.

    deep_gemm reads DG_JIT_CACHE_DIR once in C++ init() at import time.
    flashinfer resolves FLASHINFER_WORKSPACE_BASE into a module-level Path at import time.
    torch_extensions, triton, and triton_autotune read their env vars dynamically at
    each compilation/invocation, so they are safe to call after import.
    """
    root = Path(local_root).expanduser().absolute()
    for component in COMPONENT_SPECS:
        if component.import_time_sensitive and component.name in sys.modules:
            logging.warning(
                "%s already imported before apply_jit_cache_env, "
                "cache dir env var may not take effect",
                component.name,
            )
        os.environ.setdefault(
            component.env_name, str(component_cache_dir(root, component))
        )
    os.environ.setdefault("TRITON_AUTOTUNE_CACHE_MODE", "cached")


def _clear_component_startup_files(
    component: ComponentSpec, component_dir: Path
) -> int:
    if not component.startup_cleanup_globs or not component_dir.exists():
        return 0
    cleared = 0
    for pattern in component.startup_cleanup_globs:
        for path in component_dir.glob(pattern):
            try:
                if not path.is_file():
                    continue
                path.unlink()
                cleared += 1
            except FileNotFoundError:
                pass
            except OSError:
                logging.warning(
                    "failed to remove %s startup file: %s",
                    component.name,
                    path,
                    exc_info=True,
                )
    if cleared:
        logging.info(
            "removed %d %s startup files from %s",
            cleared,
            component.name,
            component_dir,
        )
    return cleared


def _tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


@contextmanager
def atomic_write_path(path: Path, ensure_parent: bool = True) -> Iterator[Path]:
    if ensure_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_sibling(path)
    try:
        yield tmp
        try:
            tmp.replace(path)
        except PermissionError:
            # FUSE mounts reject rename-over-existing (EPERM).
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            tmp.replace(path)
    finally:
        with suppress(OSError):
            tmp.unlink()


def should_sync_file(component: ComponentSpec, rel: str) -> bool:
    if not rel.rsplit("/", 1)[-1].endswith(component.sync_suffixes):
        return False
    return not any(part.startswith("tmp.pid_") for part in rel.split("/"))


def _stream_copy(
    source: Any, out: Any, buffer: bytearray, view: memoryview, deadline_s: float
) -> None:
    while True:
        if time.monotonic() >= deadline_s:
            raise TimeoutError("JIT cache file copy exceeded deadline")
        read_size = source.readinto(buffer)
        if not read_size:
            return
        out.write(view[:read_size])


def _copy_with_deadline(src: Path, dst: Path, deadline_s: float) -> None:
    buffer = bytearray(COPY_CHUNK_SIZE)
    view = memoryview(buffer)
    with src.open("rb") as source, dst.open("wb") as out:
        _stream_copy(source, out, buffer, view, deadline_s)


_USAGE_LOCATIONS = (
    ("local", GaugeMetrics.JIT_CACHE_LOCAL_USAGE_METRIC),
    ("remote", GaugeMetrics.JIT_CACHE_REMOTE_USAGE_METRIC),
)


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
    ) -> FileSystemEventHandler:
        root_prefix = str(root) + os.sep
        upload_events = component.upload_events
        enqueue = self.enqueue_upload

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event: Any) -> None:
                if event.is_directory or event.event_type not in upload_events:
                    return
                src = event.dest_path if event.event_type == "moved" else event.src_path
                if not src.startswith(root_prefix):
                    return
                rel = src[len(root_prefix) :].replace(os.sep, "/")
                if not should_sync_file(component, rel):
                    return
                if event.event_type == "created":
                    try:
                        if os.stat(src).st_size == 0:
                            return
                    except OSError:
                        return
                enqueue(component, rel)

        return Handler()

    def start(self) -> bool:
        if self.observer is not None:
            return True
        observer = Observer()
        try:
            for name, (local_dir, _) in self.component_dirs.items():
                component = COMPONENT_BY_NAME[name]
                observer.schedule(
                    self._make_handler(component, local_dir),
                    str(local_dir),
                    recursive=True,
                )
            observer.start()
            self.observer = observer
            return True
        except Exception:
            logging.warning(
                "failed to start JIT cache dirty watcher; remote writeback is disabled",
                exc_info=True,
            )
            try:
                observer.stop()
                observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
            except Exception:
                pass
            return False

    def stop(self) -> None:
        observer = self.observer
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
        self.observer = None


class JitCacheManager:
    def __init__(self, jit_config: JITConfig | None = None, run_id: str | None = None):
        self.config = JitCacheConfig.from_config(jit_config)
        self.run_id = run_id or new_jit_cache_run_id()
        self.component_dirs: dict[str, tuple[Path, Path]] = {}
        self.dirty_tracker: JitDirtyTracker | None = None

        self._lock = threading.Lock()
        self.sync_stats: dict[str, SyncStats] = {}
        self._pending_count = 0
        self._pending_empty = threading.Event()
        self._pending_empty.set()
        self._sync_lock = threading.Lock()
        self._snapshot_publishing = False

        self._periodic_sync_stop = threading.Event()
        self._periodic_sync_thread: threading.Thread | None = None

        self._usage: dict[str, dict[str, dict[str, int]]] = {
            loc: {c.name: {"bytes": 0, "files": 0} for c in COMPONENT_SPECS}
            for loc, _ in _USAGE_LOCATIONS
        }

        self._prepare_result: dict[str, Any] | None = None
        self.startup_lock: FileLock | None = None
        self.remote_cache_available = False
        self._sync_executor: ThreadPoolExecutor | None = None
        # avoid redundant mkdir RPCs on remote jit cache dir
        self._known_remote_dirs: set[Path] = set()

    @property
    def enabled(self) -> bool:
        return self.config.remote_root is not None

    @property
    def sync_executor(self) -> ThreadPoolExecutor:
        if self._sync_executor is None:
            self._sync_executor = ThreadPoolExecutor(
                max_workers=self.config.sync_workers,
                thread_name_prefix="jit-cache-sync",
            )
        return self._sync_executor

    @property
    def owns_startup_lock(self) -> bool:
        return bool(self.startup_lock and self.startup_lock.is_locked)

    @property
    def snapshot_complete_path(self) -> Path:
        return self.config.local_root / SNAPSHOT_COMPLETE_NAME

    def _add_usage(self, location: str, component_name: str, size: int) -> None:
        with self._lock:
            b = self._usage[location][component_name]
            b["bytes"] += size
            b["files"] += 1

    def _usage_snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                loc: {
                    "bytes": sum(b["bytes"] for b in buckets.values()),
                    "files": sum(b["files"] for b in buckets.values()),
                    "components": {n: dict(b) for n, b in buckets.items()},
                }
                for loc, buckets in self._usage.items()
            }

    def _report_usage_metrics(
        self, snapshot: dict[str, dict[str, Any]] | None = None
    ) -> None:
        if snapshot is None:
            snapshot = self._usage_snapshot()
        try:
            for loc, metric in _USAGE_LOCATIONS:
                data = snapshot[loc]
                for name, comp in data["components"].items():
                    kmonitor.report(
                        metric, comp["bytes"], {"module": name, "value": "bytes"}
                    )
                    kmonitor.report(
                        metric, comp["files"], {"module": name, "value": "files"}
                    )
                kmonitor.report(
                    metric, data["bytes"], {"module": "total", "value": "bytes"}
                )
                kmonitor.report(
                    metric, data["files"], {"module": "total", "value": "files"}
                )
        except Exception:
            logging.warning("failed to report JIT cache usage metrics", exc_info=True)

    @staticmethod
    def _usage_summary(
        snapshot: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            f"{loc}_cache": snapshot[loc]
            for loc, _ in _USAGE_LOCATIONS
            if snapshot[loc]["files"]
        }

    def _make_summary(
        self,
        mode: str,
        result: str,
        start_s: float,
        *,
        stats: dict[str, SyncStats] | None = None,
        drain_timed_out: bool = False,
        usage: dict[str, Any] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "timestamp_ms": int(current_time_ms()),
            "mode": mode,
            "result": result,
            "total_cost_ms": elapsed_ms(start_s),
        }
        if stats is not None:
            agg = SyncStats()
            components: dict[str, dict[str, int]] = {}
            for name, st in stats.items():
                row = {
                    "candidate_files": st.uploaded_files
                    + st.skipped_files
                    + st.failed_files,
                    "uploaded_files": st.uploaded_files,
                    "uploaded_bytes": st.uploaded_bytes,
                    "skipped_files": st.skipped_files,
                    "failed_files": st.failed_files,
                }
                agg.uploaded_files += st.uploaded_files
                agg.uploaded_bytes += st.uploaded_bytes
                agg.skipped_files += st.skipped_files
                agg.failed_files += st.failed_files
                if any(row.values()):
                    components[name] = row
            event["components"] = components
            event.update(
                {
                    "candidate_files": agg.uploaded_files
                    + agg.skipped_files
                    + agg.failed_files,
                    "uploaded_files": agg.uploaded_files,
                    "uploaded_bytes": agg.uploaded_bytes,
                    "skipped_files": agg.skipped_files,
                    "failed_files": agg.failed_files,
                }
            )
            if drain_timed_out:
                event["drain_timed_out"] = True
        event.update(
            {k: v for k, v in extra.items() if v is not None and v != "" and v != 0}
        )
        if usage:
            event.update(usage)
        self._write_summary(event)
        return event

    def _write_summary(self, event: dict[str, Any]) -> None:
        with atomic_write_path(self.config.local_root / SUMMARY_NAME) as tmp:
            tmp.write_bytes(
                json.dumps(
                    {
                        "event": "jit_cache_summary",
                        "updated_at_ms": int(current_time_ms()),
                        "run_id": self.run_id,
                        "remote_root": str(self.config.remote_root or ""),
                        "local_root": str(self.config.local_root),
                        **event,
                    },
                    indent=2,
                    sort_keys=True,
                ).encode()
            )

    def bootstrap(self) -> None:
        self.config.local_root.mkdir(parents=True, exist_ok=True)
        for component in COMPONENT_SPECS:
            local_dir = component_cache_dir(self.config.local_root, component)
            local_dir.mkdir(parents=True, exist_ok=True)
            _clear_component_startup_files(component, local_dir)
            if self.config.remote_root is not None:
                remote_dir = component_cache_dir(self.config.remote_root, component)
                self.component_dirs[component.name] = (local_dir, remote_dir)
        apply_jit_cache_env(self.config.local_root)
        logging.info(
            "JIT cache local=%s remote=%s",
            self.config.local_root,
            self.config.remote_root,
        )

    def prepare(self) -> dict[str, Any]:
        start_s = time.monotonic()
        if not self.enabled:
            return self._make_summary(
                "snapshot_download", "skipped", start_s, cache_state="disabled"
            )
        if self._prepare_result is not None:
            return self._prepare_result
        deadline_s = start_s + self.config.remote_sync_longer_timeout_s
        if self.acquire_startup_lock(0.0):
            summary = self._prepare_as_leader(deadline_s)
        else:
            summary = self._wait_ready_or_takeover(start_s, deadline_s)
        self.remote_cache_available = summary.get("result") not in {"failed", "timeout"}
        if not self.remote_cache_available:
            self.release_startup_lock()
        self._prepare_result = summary
        return summary

    def acquire_startup_lock(self, timeout_s: float = 0.0) -> bool:
        if self.owns_startup_lock:
            return True
        lock = FileLock(
            str(self.config.local_root / LOCAL_LOCK_NAME), thread_local=False
        )
        try:
            lock.acquire(timeout=max(timeout_s, 0.0))
        except Timeout:
            return False
        with self._lock:
            self.startup_lock = lock
        return True

    def _prepare_as_leader(self, deadline_s: float) -> dict[str, Any]:
        self.snapshot_complete_path.unlink(missing_ok=True)
        try:
            return self._pull_snapshot(deadline_s)
        finally:
            # By design: write marker even on failure so followers don't block or retry indefinitely.
            self.snapshot_complete_path.touch()

    def _wait_ready_or_takeover(
        self, start_s: float, deadline_s: float
    ) -> dict[str, Any]:
        while True:
            if self.snapshot_complete_path.exists():
                return self._make_summary(
                    "snapshot_download",
                    "success",
                    start_s,
                    cache_state="leader_completed",
                )
            if self.acquire_startup_lock(0.0):
                return self._prepare_as_leader(deadline_s)
            if time.monotonic() >= deadline_s:
                return self._make_summary(
                    "snapshot_download", "timeout", start_s, cache_state="timeout"
                )
            time.sleep(
                min(STARTUP_WAIT_POLL_S, max(0.0, deadline_s - time.monotonic()))
            )

    def _pull_snapshot(self, deadline_s: float) -> dict[str, Any]:
        start_s = time.monotonic()
        remote_root = self.config.remote_root
        snapshot_path = self._find_latest_snapshot(remote_root)
        snapshot_bytes = 0
        try:
            if snapshot_path is None:
                return self._make_summary(
                    "snapshot_download", "skipped", start_s, cache_state="snapshot_miss"
                )
            try:
                snapshot_bytes = snapshot_path.stat().st_size
            except OSError:
                pass
            ext_bytes, ext_files = self._extract_snapshot(snapshot_path, deadline_s)
            return self._make_summary(
                "snapshot_download",
                "success",
                start_s,
                cache_state="snapshot_hit",
                snapshot_bytes=snapshot_bytes,
                extracted_files=ext_files,
                extracted_bytes=ext_bytes,
            )
        except TimeoutError as e:
            return self._make_summary(
                "snapshot_download",
                "timeout",
                start_s,
                cache_state="timeout",
                message=str(e),
                snapshot_bytes=snapshot_bytes,
            )
        except Exception as e:
            logging.exception("failed to download/extract JIT cache snapshot")
            return self._make_summary(
                "snapshot_download",
                "failed",
                start_s,
                cache_state="snapshot_error",
                message=str(e),
            )

    def _extract_snapshot(self, archive: Path, deadline_s: float) -> tuple[int, int]:
        extracted_bytes = 0
        extracted_files = 0
        dctx = zstd.ZstdDecompressor()
        gpu_scope = get_gpu_info()
        buffer = bytearray(COPY_CHUNK_SIZE)
        view = memoryview(buffer)
        with archive.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        parts = member.name.split("/")
                        if not parts or member.name.startswith("/") or ".." in parts:
                            raise RuntimeError(f"unsafe snapshot path: {member.name}")
                        component = COMPONENT_BY_NAME.get(parts[0])
                        if component is None or not member.isfile():
                            continue
                        if component.gpu_scoped and (
                            len(parts) < 2 or parts[1] != gpu_scope
                        ):
                            continue
                        if time.monotonic() >= deadline_s:
                            raise TimeoutError(
                                "snapshot extraction exceeded prepare timeout"
                            )
                        target = self.config.local_root.joinpath(*parts)
                        try:
                            st = target.stat()
                            if st.st_size == member.size and int(st.st_mtime) == int(
                                member.mtime
                            ):
                                continue
                        except OSError:
                            pass
                        source = tar.extractfile(member)
                        if source is None:
                            continue
                        with source, atomic_write_path(target) as tmp:
                            with tmp.open("wb") as out:
                                _stream_copy(source, out, buffer, view, deadline_s)
                            os.utime(tmp, (member.mtime, member.mtime))
                        rel = "/".join(parts[1:])
                        if should_sync_file(component, rel):
                            extracted_files += 1
                            extracted_bytes += member.size
                            self._add_usage("local", component.name, member.size)
        return extracted_bytes, extracted_files

    @property
    def _sync_eligible(self) -> bool:
        return self.enabled and self.owns_startup_lock and self.remote_cache_available

    def start_background_sync(self) -> None:
        if not self._sync_eligible:
            return
        if self.dirty_tracker is not None:
            return
        tracker = JitDirtyTracker(self.component_dirs, self.enqueue_upload)
        if not tracker.start():
            return
        self.dirty_tracker = tracker
        self._start_periodic_sync()

    def _start_periodic_sync(self) -> None:
        if self._periodic_sync_thread is not None:
            return
        self._periodic_sync_stop.clear()
        t = threading.Thread(
            target=self._periodic_sync_loop,
            name=f"jit-periodic-sync-{self.run_id}",
            daemon=True,
        )
        self._periodic_sync_thread = t
        t.start()

    def _periodic_sync_loop(self) -> None:
        while not self._periodic_sync_stop.wait(PERIODIC_SYNC_INTERVAL_S):
            try:
                self.sync_once("periodic_flush")
            except Exception:
                logging.warning("periodic JIT cache sync failed", exc_info=True)

    def _stop_periodic_sync(self) -> None:
        t = self._periodic_sync_thread
        if t is None:
            return
        # Signal the loop to exit; short join since we don't drain remaining uploads.
        self._periodic_sync_stop.set()
        if t is not threading.current_thread():
            t.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
        self._periodic_sync_thread = None

    def enqueue_upload(self, component: ComponentSpec, rel_path: str) -> bool:
        with self._lock:
            self._pending_count += 1
            self._pending_empty.clear()
        try:
            self.sync_executor.submit(self._upload_task, component, rel_path)
            return True
        except RuntimeError:
            self._complete_pending()
            return False

    def _complete_pending(self) -> None:
        with self._lock:
            self._pending_count = max(0, self._pending_count - 1)
            if self._pending_count == 0:
                self._pending_empty.set()

    def _upload_task(self, component: ComponentSpec, rel_path: str) -> None:
        try:
            uploaded = self._do_upload(component, rel_path)
            with self._lock:
                st = self.sync_stats.setdefault(component.name, SyncStats())
                if uploaded > 0:
                    st.uploaded_files += 1
                    st.uploaded_bytes += uploaded
                else:
                    st.skipped_files += 1
            if uploaded > 0:
                self._add_usage("remote", component.name, uploaded)
        except Exception:
            logging.warning(
                "failed to upload JIT cache file %s/%s",
                component.name,
                rel_path,
                exc_info=True,
            )
            with self._lock:
                self.sync_stats.setdefault(
                    component.name, SyncStats()
                ).failed_files += 1
        finally:
            self._complete_pending()

    def _do_upload(self, component: ComponentSpec, rel_path: str) -> int:
        if not should_sync_file(component, rel_path):
            return 0
        src_root, dst_root = self.component_dirs[component.name]
        src = src_root / rel_path
        try:
            return self._copy_atomic(
                src,
                dst_root / rel_path,
                time.monotonic() + self.config.remote_sync_timeout_s,
            )
        except FileNotFoundError:
            return 0

    def _copy_atomic(self, src: Path, dst: Path, deadline_s: float) -> int:
        try:
            if dst.exists():
                return 0
        except OSError:
            pass
        src_stat = src.stat()
        if src_stat.st_size <= 0:
            return 0
        parent = dst.parent
        if parent not in self._known_remote_dirs:
            parent.mkdir(parents=True, exist_ok=True)
            self._known_remote_dirs.add(parent)
        try:
            with atomic_write_path(dst, ensure_parent=False) as tmp:
                _copy_with_deadline(src, tmp, deadline_s)
        except PermissionError:
            # FUSE mounts may reject overwrite-rename (EPERM); if the
            # destination already landed from another rank, treat as skip.
            try:
                if dst.exists():
                    return 0
            except OSError:
                pass
            raise
        return src_stat.st_size

    def sync_once(
        self, mode: str = "manual_sync", drain_timeout_s: float | None = None
    ) -> dict[str, Any]:
        if not self._sync_lock.acquire(blocking=False):
            return {"mode": mode, "result": "skipped", "reason": "sync in progress"}
        try:
            return self._sync_once_impl(mode, drain_timeout_s)
        finally:
            self._sync_lock.release()

    def _sync_once_impl(
        self, mode: str, drain_timeout_s: float | None = None
    ) -> dict[str, Any]:
        start_s = time.monotonic()
        if not self.enabled:
            return self._make_summary(
                mode,
                "skipped",
                start_s,
                cache_state="disabled",
            )
        usage_snap = self._usage_snapshot()
        drain_timed_out = False
        try:
            stats, drain_timed_out = self._drain_upload_queue(drain_timeout_s)
        except Exception:
            logging.warning("failed to sync remote JIT cache", exc_info=True)
            stats = {}
            drain_timed_out = True
        self._report_usage_metrics(usage_snap)
        has_failures = drain_timed_out or any(s.failed_files for s in stats.values())
        summary = self._make_summary(
            mode,
            "failed" if has_failures else "success",
            start_s,
            cache_state="remote_hit",
            stats=stats,
            drain_timed_out=drain_timed_out,
            message="remote JIT cache sync failed" if has_failures else "",
            usage=self._usage_summary(usage_snap),
        )
        if not has_failures:
            try:
                self._publish_snapshot_if_due()
            except Exception:
                logging.warning("failed to publish JIT cache snapshot", exc_info=True)
        return summary

    def _drain_upload_queue(
        self, drain_timeout_s: float | None = None
    ) -> tuple[dict[str, SyncStats], bool]:
        timeout = (
            drain_timeout_s
            if drain_timeout_s is not None
            else self.config.remote_sync_timeout_s
        )
        timed_out = not self._pending_empty.wait(timeout=max(timeout, 0.0))
        with self._lock:
            stats = dict(self.sync_stats)
            self.sync_stats = {}
            pending = self._pending_count if timed_out else 0
        if timed_out:
            logging.warning(
                "JIT cache upload queue drain timed out with %d pending", pending
            )
        return stats, timed_out

    def _publish_snapshot_if_due(self) -> None:
        if not self._sync_eligible:
            return
        with self._lock:
            if self._snapshot_publishing:
                return
            self._snapshot_publishing = True
        lease_dir = self._try_acquire_publish_lease()
        if lease_dir is None:
            with self._lock:
                self._snapshot_publishing = False
            return
        try:
            self.sync_executor.submit(self._snapshot_publish_task, lease_dir)
        except RuntimeError:
            with self._lock:
                self._snapshot_publishing = False
            self._release_publish_lease(lease_dir)

    def _snapshot_publish_task(self, lease_dir: Path) -> None:
        try:
            if not self._publish_snapshot():
                self._release_publish_lease(lease_dir)
        except Exception:
            self._release_publish_lease(lease_dir)
            logging.warning("failed to publish JIT cache snapshot", exc_info=True)
        finally:
            with self._lock:
                self._snapshot_publishing = False

    def _try_acquire_publish_lease(self) -> Path | None:
        remote_root = self.config.remote_root
        if remote_root is None:
            return None
        bucket = int(time.time()) // SNAPSHOT_PUBLISH_INTERVAL_S
        self._cleanup_stale_leases(remote_root, bucket)
        lease_dir = remote_root / f"{SNAPSHOT_PUBLISH_LEASE_PREFIX}{bucket}"
        try:
            lease_dir.mkdir()
            return lease_dir
        except FileExistsError:
            return None
        except OSError:
            logging.warning(
                "failed to acquire JIT snapshot publish lease", exc_info=True
            )
            return None

    @staticmethod
    def _cleanup_stale_leases(remote_root: Path, current_bucket: int) -> None:
        try:
            for entry in remote_root.iterdir():
                if not entry.name.startswith(SNAPSHOT_PUBLISH_LEASE_PREFIX):
                    continue
                try:
                    entry_bucket = int(entry.name[len(SNAPSHOT_PUBLISH_LEASE_PREFIX) :])
                except ValueError:
                    continue
                if entry_bucket < current_bucket - 2:
                    with suppress(OSError):
                        entry.rmdir()
        except OSError:
            pass

    @staticmethod
    def _release_publish_lease(lease_dir: Path) -> None:
        try:
            lease_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            logging.warning(
                "failed to release JIT snapshot publish lease: %s",
                lease_dir,
                exc_info=True,
            )

    @staticmethod
    def _new_snapshot_name() -> str:
        epoch_ms = int(current_time_ms())
        short_uuid = uuid.uuid4().hex[:8]
        return f"{SNAPSHOT_PREFIX}{epoch_ms}_{short_uuid}{SNAPSHOT_SUFFIX}"

    @staticmethod
    def _list_versioned_snapshots(remote_root: Path) -> list[Path]:
        result: list[Path] = []
        try:
            for entry in remote_root.iterdir():
                if entry.name.startswith(SNAPSHOT_PREFIX) and entry.name.endswith(
                    SNAPSHOT_SUFFIX
                ):
                    result.append(entry)
        except OSError:
            pass
        result.sort(key=lambda p: p.name)
        return result

    @staticmethod
    def _find_latest_snapshot(remote_root: Path) -> Path | None:
        for entry in reversed(JitCacheManager._list_versioned_snapshots(remote_root)):
            try:
                if entry.stat().st_size > 0:
                    return entry
            except OSError:
                continue
        return None

    @staticmethod
    def _cleanup_remote_root(
        remote_root: Path,
        keep_snapshots: int = SNAPSHOT_KEEP_COUNT,
        tmp_max_age_s: float = 3600,
    ) -> None:
        snapshots: list[Path] = []
        now = time.time()
        try:
            for entry in remote_root.iterdir():
                name = entry.name
                if name.startswith(SNAPSHOT_PREFIX) and name.endswith(SNAPSHOT_SUFFIX):
                    snapshots.append(entry)
                elif name.endswith(".tmp"):
                    try:
                        if now - entry.stat().st_mtime > tmp_max_age_s:
                            with suppress(OSError):
                                entry.unlink()
                    except OSError:
                        pass
        except OSError:
            pass
        snapshots.sort(key=lambda p: p.name)
        for old in snapshots[:-keep_snapshots]:
            with suppress(OSError):
                old.unlink()

    def _publish_snapshot(self) -> bool:
        remote_root = self.config.remote_root
        if remote_root is None:
            return False
        snapshot_name = self._new_snapshot_name()
        snapshot_path = remote_root / snapshot_name
        local_tmp = _tmp_sibling(self.config.local_root / "snapshot_build.tar.zst")
        start_s = time.monotonic()
        publish_deadline = start_s + self.config.remote_sync_longer_timeout_s
        try:
            files, bytes_ = self._create_snapshot_archive(local_tmp, publish_deadline)
            if files <= 0:
                logging.info("skip publishing empty JIT cache snapshot")
                return False
            _copy_with_deadline(local_tmp, snapshot_path, publish_deadline)
            self._cleanup_remote_root(remote_root)
            logging.info(
                "published JIT cache snapshot %s files=%d bytes=%d cost_ms=%d",
                snapshot_path,
                files,
                bytes_,
                elapsed_ms(start_s),
            )
            return True
        finally:
            with suppress(OSError):
                local_tmp.unlink()

    def _create_snapshot_archive(
        self, archive: Path, deadline_s: float = float("inf")
    ) -> tuple[int, int]:
        files = 0
        bytes_ = 0
        remote_root = self.config.remote_root
        assert remote_root is not None
        local_root = self.config.local_root
        cctx = zstd.ZstdCompressor(level=3)
        with archive.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for component, remote_path, rel in self._iter_snapshot_files(
                        remote_root
                    ):
                        if time.monotonic() >= deadline_s:
                            logging.warning(
                                "snapshot archive creation exceeded deadline after %d files",
                                files,
                            )
                            break
                        # File list from remote (authoritative); prefer local copy to avoid FUSE overhead.
                        local_path = local_root / component.name / rel
                        source_path = (
                            local_path if local_path.is_file() else remote_path
                        )
                        try:
                            with source_path.open("rb") as source:
                                source_stat = os.fstat(source.fileno())
                                if source_stat.st_size <= 0 or not stat.S_ISREG(
                                    source_stat.st_mode
                                ):
                                    continue
                                info = tarfile.TarInfo(f"{component.name}/{rel}")
                                info.size = source_stat.st_size
                                info.mtime = source_stat.st_mtime
                                info.mode = stat.S_IMODE(source_stat.st_mode)
                                tar.addfile(info, source)
                                files += 1
                                bytes_ += source_stat.st_size
                        except FileNotFoundError:
                            continue
                        except OSError:
                            logging.warning(
                                "failed to add JIT cache file to snapshot: %s",
                                remote_path,
                                exc_info=True,
                            )
        return files, bytes_

    @staticmethod
    def _iter_snapshot_files(root: Path) -> Iterator[tuple[ComponentSpec, Path, str]]:
        for component in COMPONENT_SPECS:
            component_root = root / component.name
            try:
                if not component_root.is_dir():
                    continue
            except OSError:
                logging.warning(
                    "failed to probe component dir %s; skipping in snapshot",
                    component.name,
                    exc_info=True,
                )
                continue
            prefix = str(component_root) + os.sep
            for dirpath, dirnames, filenames in os.walk(
                component_root,
                onerror=lambda e: logging.warning(
                    "os.walk error during JIT snapshot scan: %s", e
                ),
            ):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for filename in filenames:
                    full = os.path.join(dirpath, filename)
                    rel = full[len(prefix) :].replace(os.sep, "/")
                    if not should_sync_file(component, rel):
                        continue
                    yield component, Path(full), rel

    def stop(self) -> None:
        # Stop watchdog first so no new uploads are enqueued.
        if self.dirty_tracker:
            self.dirty_tracker.stop()
            self.dirty_tracker = None
        self._stop_periodic_sync()
        # By design: JIT cache does not need strong consistency; drop pending uploads.
        if self._sync_executor is not None:
            self._sync_executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            # reset the counter to unblock any _drain_upload_queue still waiting.
            self._pending_count = 0
            self._pending_empty.set()
            self._snapshot_publishing = False
        self.release_startup_lock()

    def release_startup_lock(self) -> None:
        with self._lock:
            lock = self.startup_lock
            self.startup_lock = None
        if lock and lock.is_locked:
            lock.release()
