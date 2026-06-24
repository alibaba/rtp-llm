from __future__ import annotations

import logging
import os
import shutil
import stat
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, NamedTuple
from urllib.parse import urlparse

import orjson
import torch
import zstandard as zstd
from filelock import FileLock, Timeout
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.models_py.triton_kernels.autotune_cache import sanitize_gpu_name
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
SNAPSHOT_PUBLISH_LEASE_PREFIX = ".jit_snapshot_publish_lease."
SNAPSHOT_PUBLISH_INTERVAL_S = 20 * 60
LOCAL_LOCK_NAME = ".rtp_jit_local.lock"
SNAPSHOT_COMPLETE_NAME = ".rtp_jit_snapshot_complete"
SUMMARY_NAME = ".rtp_jit_summary.json"

DEFAULT_JIT_SYNC_WORKERS = 32
MAX_UPLOAD_ATTEMPTS = 3
COPY_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB chunks for deadline-aware copy
STARTUP_WAIT_POLL_S = 0.3
SNAPSHOT_EXTRACT_TIMEOUT_MESSAGE = (
    "snapshot extraction exceeded prepare timeout; remote bootstrap skipped"
)


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...]
    upload_events: frozenset[str]
    gpu_scoped: bool = False


COMPONENT_SPECS = (
    ComponentSpec(
        "flashinfer",
        "FLASHINFER_WORKSPACE_BASE",
        (".so", ".cubin", ".json"),
        frozenset({"closed"}),
    ),
    ComponentSpec(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        (".so", ".cubin", ".json"),
        frozenset({"created"}),
    ),
    ComponentSpec(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        (".so", ".cubin", ".json"),
        frozenset({"closed"}),
    ),
    ComponentSpec(
        "triton",
        "TRITON_CACHE_DIR",
        (".so", ".cubin", ".json"),
        frozenset({"moved"}),
    ),
    ComponentSpec(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        (".json", ".pkl", ".pickle"),
        frozenset({"closed"}),
        gpu_scoped=True,
    ),
)
COMPONENT_BY_NAME = {component.name: component for component in COMPONENT_SPECS}


class FileMeta(NamedTuple):
    size: int
    mtime_ns: int


@dataclass
class SyncStats:
    candidate_files: int = 0
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    skipped_files: int = 0
    failed_files: int = 0
    retried_files: int = 0
    drain_timed_out: int = 0


@dataclass(frozen=True)
class UploadTask:
    component: ComponentSpec
    rel_path: str
    attempts: int = 0


@dataclass(frozen=True)
class SyncResult:
    status: str
    uploaded_bytes: int = 0
    reason: str = ""


@dataclass
class SummaryEvent:
    mode: str
    result: str
    total_cost_ms: int
    cache_state: str = ""
    stats: dict[str, SyncStats] | None = None
    message: str = ""
    snapshot_bytes: int | None = None
    extracted_files: int | None = None
    extracted_bytes: int | None = None


@dataclass
class _UsageBucket:
    bytes: int = 0
    files: int = 0


@dataclass
class JitCacheConfig:
    remote_root: Path | None
    local_root: Path
    remote_timeout_s: float
    sync_workers: int = DEFAULT_JIT_SYNC_WORKERS

    @classmethod
    def from_config(cls, jit_config: JITConfig | None = None) -> JitCacheConfig:
        if jit_config is None:
            jit_config = JITConfig()
        return cls(
            remote_root=resolve_remote_root(jit_config.remote_jit_dir),
            local_root=Path(jit_config.local_jit_cache_dir).expanduser().absolute(),
            remote_timeout_s=float(jit_config.jit_remote_timeout_s),
        )


def now_ms() -> int:
    return int(time.time() * 1000)


def elapsed_ms(start_s: float) -> int:
    return int((time.monotonic() - start_s) * 1000)


def new_jit_cache_run_id() -> str:
    return f"{int(time.time() * 1000)}-{os.getpid()}-{uuid.uuid4().hex}"


@lru_cache(maxsize=1)
def get_gpu_scope() -> str:
    gpu_name = os.environ.get("TRITON_AUTOTUNE_GPU_NAME")
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return sanitize_gpu_name(gpu_name) if gpu_name else "unknown_gpu"


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
    component_root = root / component.name
    if component.gpu_scoped:
        return component_root / get_gpu_scope()
    return component_root


def apply_jit_cache_env(local_root: Path | str) -> None:
    root = Path(local_root).expanduser().absolute()
    for component in COMPONENT_SPECS:
        os.environ[component.env_name] = str(component_cache_dir(root, component))
    os.environ["TRITON_AUTOTUNE_CACHE_MODE"] = "cached"


def file_meta(path: Path, *, stat: os.stat_result | None = None) -> FileMeta:
    stat = stat or path.stat()
    return FileMeta(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


@contextmanager
def atomic_write_path(path: Path) -> Iterator[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(path)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        with suppress(OSError):
            tmp.unlink()


def is_syncable_file(component: ComponentSpec, rel: str) -> bool:
    return rel.rsplit("/", 1)[-1].endswith(component.sync_suffixes)


def is_tmp_jit_path(rel: str) -> bool:
    return any(part.startswith("tmp.pid_") for part in rel.split("/"))


_USAGE_LOCATIONS = (
    ("local", GaugeMetrics.JIT_CACHE_LOCAL_USAGE_METRIC),
    ("remote", GaugeMetrics.JIT_CACHE_REMOTE_USAGE_METRIC),
)


class CacheUsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._buckets = {
            loc: {c.name: _UsageBucket() for c in COMPONENT_SPECS}
            for loc, _ in _USAGE_LOCATIONS
        }

    def add(self, location: str, component_name: str, size: int) -> None:
        with self._lock:
            b = self._buckets[location][component_name]
            b.bytes += size
            b.files += 1

    def report_metrics(self) -> None:
        if not getattr(kmonitor, "_inited", False):
            return
        with self._lock:
            snapshot = {
                loc: {n: (b.bytes, b.files) for n, b in bkts.items()}
                for loc, bkts in self._buckets.items()
            }
        try:
            for loc, metric in _USAGE_LOCATIONS:
                total_bytes = total_files = 0
                for name, (byt, fil) in snapshot[loc].items():
                    total_bytes += byt
                    total_files += fil
                    kmonitor.report(metric, byt, {"module": name, "value": "bytes"})
                    kmonitor.report(metric, fil, {"module": name, "value": "files"})
                kmonitor.report(
                    metric, total_bytes, {"module": "total", "value": "bytes"}
                )
                kmonitor.report(
                    metric, total_files, {"module": "total", "value": "files"}
                )
        except Exception:
            logging.warning("failed to report JIT cache usage metrics", exc_info=True)

    def to_summary_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        with self._lock:
            for loc, _ in _USAGE_LOCATIONS:
                buckets = self._buckets[loc]
                total_bytes = sum(b.bytes for b in buckets.values())
                total_files = sum(b.files for b in buckets.values())
                if total_files:
                    result[f"{loc}_cache"] = {
                        "bytes": total_bytes,
                        "files": total_files,
                        "components": {
                            name: {"bytes": b.bytes, "files": b.files}
                            for name, b in buckets.items()
                        },
                    }
        return result


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
                if not rel.endswith(component.sync_suffixes):
                    return
                if is_tmp_jit_path(rel):
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
            for component_name, (local_dir, _) in self.component_dirs.items():
                component = COMPONENT_BY_NAME[component_name]
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
                observer.join(timeout=1)
            except Exception:
                pass
            return False

    def stop(self) -> None:
        observer = self.observer
        if observer is None:
            return
        observer.stop()
        observer.join(timeout=5)
        self.observer = None


class JitCacheManager:
    def __init__(self, jit_config: JITConfig | None = None, run_id: str | None = None):
        self.config = JitCacheConfig.from_config(jit_config)
        self.run_id = run_id or new_jit_cache_run_id()
        self.component_dirs: dict[str, tuple[Path, Path]] = {}
        self.dirty_tracker: JitDirtyTracker | None = None
        self._lock = threading.Condition()
        self.sync_stats: dict[str, SyncStats] = {}
        self.pending_uploads: set[tuple[str, str]] = set()
        self.usage_tracker = CacheUsageTracker()
        self._prepare_result: dict[str, Any] | None = None
        self.startup_lock: FileLock | None = None
        self.remote_cache_available = False
        self._snapshot_publish_lock = threading.Lock()
        self._snapshot_publish_thread: threading.Thread | None = None
        self.sync_executor = ThreadPoolExecutor(
            max_workers=self.config.sync_workers,
            thread_name_prefix="jit-cache-upload",
        )

    @property
    def enabled(self) -> bool:
        return self.config.remote_root is not None

    @property
    def owns_startup_lock(self) -> bool:
        return bool(self.startup_lock and self.startup_lock.is_locked)

    @property
    def snapshot_complete_path(self) -> Path:
        return self.config.local_root / SNAPSHOT_COMPLETE_NAME

    def bootstrap(self) -> None:
        self.config.local_root.mkdir(parents=True, exist_ok=True)
        for component in COMPONENT_SPECS:
            local_dir = component_cache_dir(self.config.local_root, component)
            local_dir.mkdir(parents=True, exist_ok=True)
            if self.config.remote_root is not None:
                self.component_dirs[component.name] = (
                    local_dir,
                    component_cache_dir(self.config.remote_root, component),
                )
        apply_jit_cache_env(self.config.local_root)
        logging.info(
            "JIT cache local=%s remote=%s",
            self.config.local_root,
            self.config.remote_root,
        )

    def prepare(self) -> dict[str, Any]:
        start_s = time.monotonic()
        if not self.enabled:
            return self._emit_summary(
                SummaryEvent(
                    mode="snapshot_download",
                    result="skipped",
                    total_cost_ms=0,
                    cache_state="disabled",
                )
            )
        deadline_s = start_s + max(self.config.remote_timeout_s, 0.0)
        if self._prepare_result is not None:
            return self._prepare_result
        if self.acquire_startup_lock(0.0):
            summary = self._prepare_as_startup_owner(deadline_s)
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
        self.startup_lock = lock
        return True

    def _prepare_as_startup_owner(self, deadline_s: float) -> dict[str, Any]:
        if self.snapshot_complete_path.exists():
            return self._emit_summary(
                self._snapshot_event(
                    time.monotonic(),
                    "skipped",
                    "local_hit",
                    message="local JIT cache snapshot complete marker exists; remote bootstrap skipped",
                )
            )
        return self._emit_summary(self._pull_snapshot(deadline_s))

    def _wait_ready_or_takeover(
        self, start_s: float, deadline_s: float
    ) -> dict[str, Any]:
        while True:
            if self.snapshot_complete_path.exists():
                return self._emit_summary(
                    self._snapshot_event(
                        start_s,
                        "success",
                        "leader_completed",
                        message="leader completed snapshot; follower proceeding",
                    )
                )
            if self.acquire_startup_lock(0.0):
                return self._prepare_as_startup_owner(deadline_s)
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                break
            time.sleep(min(STARTUP_WAIT_POLL_S, remaining_s))
        return self._emit_summary(self._snapshot_event(start_s, "timeout", "timeout"))

    def _pull_snapshot(self, deadline_s: float) -> SummaryEvent:
        start_s = time.monotonic()
        snapshot_path = self.config.remote_root / SNAPSHOT_NAME
        if not snapshot_path.exists():
            # Mark this local cache initialized to avoid repeated waits on a miss.
            self._write_snapshot_complete_marker(0, 0, 0)
            return self._snapshot_event(start_s, "skipped", "snapshot_miss")

        extract_root: Path | None = None
        snapshot_bytes = 0
        try:
            extract_root = tmp_sibling(self.config.local_root / ".jit_extract")
            snapshot_bytes = snapshot_path.stat().st_size
            ext_bytes, ext_files = self._extract_snapshot(
                snapshot_path,
                extract_root,
                deadline_s,
            )
            self._apply_extract(extract_root)
            self._write_snapshot_complete_marker(snapshot_bytes, ext_files, ext_bytes)
            return self._snapshot_event(
                start_s,
                "success",
                "snapshot_hit",
                snapshot_bytes=snapshot_bytes,
                extracted_files=ext_files,
                extracted_bytes=ext_bytes,
            )
        except TimeoutError as e:
            return self._snapshot_event(
                start_s,
                "timeout",
                "timeout",
                message=str(e),
                snapshot_bytes=snapshot_bytes,
            )
        except Exception as e:
            logging.exception("failed to download/extract JIT cache snapshot")
            return self._snapshot_event(
                start_s,
                "failed",
                "snapshot_error",
                message=str(e),
            )
        finally:
            if extract_root is not None and extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)

    def _extract_snapshot(
        self,
        archive: Path,
        dst_root: Path,
        deadline_s: float,
    ) -> tuple[int, int]:
        extracted_bytes = 0
        extracted_files = 0
        dctx = zstd.ZstdDecompressor()
        gpu_scope = get_gpu_scope()
        buffer = bytearray(COPY_CHUNK_SIZE)
        view = memoryview(buffer)
        with archive.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        if time.monotonic() >= deadline_s:
                            raise TimeoutError(SNAPSHOT_EXTRACT_TIMEOUT_MESSAGE)
                        member_path = PurePosixPath(member.name)
                        if (
                            not member_path.parts
                            or member_path.is_absolute()
                            or ".." in member_path.parts
                        ):
                            raise RuntimeError(f"unsafe snapshot path: {member.name}")
                        component = COMPONENT_BY_NAME.get(member_path.parts[0])
                        if component is None or not member.isfile():
                            continue
                        if component.gpu_scoped and (
                            len(member_path.parts) < 2
                            or member_path.parts[1] != gpu_scope
                        ):
                            continue
                        source = tar.extractfile(member)
                        if source is None:
                            continue
                        target = dst_root.joinpath(*member_path.parts)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source, target.open("wb") as out:
                            while True:
                                if time.monotonic() >= deadline_s:
                                    raise TimeoutError(SNAPSHOT_EXTRACT_TIMEOUT_MESSAGE)
                                read_size = source.readinto(buffer)
                                if not read_size:
                                    break
                                out.write(view[:read_size])
                        os.utime(target, (member.mtime, member.mtime))
                        rel = "/".join(member_path.parts[1:])
                        if is_syncable_file(component, rel):
                            extracted_files += 1
                            extracted_bytes += member.size
                            self.usage_tracker.add("local", component.name, member.size)
        return extracted_bytes, extracted_files

    def _apply_extract(self, extract_root: Path) -> None:
        if not extract_root.exists():
            return
        for src_file in extract_root.rglob("*"):
            if not src_file.is_file():
                continue
            dst = self.config.local_root / src_file.relative_to(extract_root)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_file), str(dst))

    def start_background_sync(self) -> None:
        if not (
            self.enabled and self.owns_startup_lock and self.remote_cache_available
        ):
            return
        tracker = JitDirtyTracker(self.component_dirs, self.enqueue_upload)
        if not tracker.start():
            return
        self.dirty_tracker = tracker

    def enqueue_upload(self, component: ComponentSpec, rel_path: str) -> bool:
        pending_key = (component.name, rel_path)
        with self._lock:
            if pending_key in self.pending_uploads:
                return False
            self.pending_uploads.add(pending_key)
        if not self._submit_upload_task(UploadTask(component, rel_path)):
            self._discard_pending_upload(pending_key)
            return False
        return True

    def _submit_upload_task(self, task: UploadTask) -> bool:
        try:
            self.sync_executor.submit(self._process_upload_task, task)
        except RuntimeError:
            return False
        return True

    def _discard_pending_upload(self, pending_key: tuple[str, str]) -> None:
        with self._lock:
            was_pending = pending_key in self.pending_uploads
            self.pending_uploads.discard(pending_key)
            if was_pending:
                self._lock.notify_all()

    def _process_upload_task(self, task: UploadTask) -> SyncResult:
        upload_error: BaseException | None = None
        try:
            result = self._try_upload(task.component, task.rel_path)
        except Exception as e:
            upload_error = e
            result = SyncResult("retry", reason=str(e))
        if result.status == "retry" and task.attempts + 1 < MAX_UPLOAD_ATTEMPTS:
            if self._submit_upload_task(
                UploadTask(task.component, task.rel_path, attempts=task.attempts + 1)
            ):
                with self._lock:
                    self.sync_stats.setdefault(
                        task.component.name, SyncStats()
                    ).retried_files += 1
                return result
        if result.status == "retry":
            logging.warning(
                "failed to upload JIT cache file %s after %d attempts: %s",
                task,
                task.attempts + 1,
                result.reason or "retry exhausted",
                exc_info=upload_error,
            )
        with self._lock:
            self._record_sync_stats(task.component, result)
            self.pending_uploads.discard((task.component.name, task.rel_path))
            self._lock.notify_all()
        if result.status == "uploaded":
            self.usage_tracker.add("remote", task.component.name, result.uploaded_bytes)
        return result

    def _try_upload(
        self,
        component: ComponentSpec,
        rel_path: str,
    ) -> SyncResult:
        if is_tmp_jit_path(rel_path) or not is_syncable_file(component, rel_path):
            return SyncResult("skipped")
        src_root, dst_root = self.component_dirs[component.name]
        src = src_root / rel_path
        try:
            src_meta = file_meta(src)
        except FileNotFoundError:
            return SyncResult("skipped")
        except OSError as e:
            return SyncResult("retry", reason=f"failed to stat source: {e}")
        if src_meta.size <= 0:
            return SyncResult("skipped")
        uploaded_bytes = self._copy_verified(src, dst_root / rel_path, src_meta)
        return SyncResult("uploaded", uploaded_bytes)

    def _copy_verified(self, src: Path, dst: Path, src_meta: FileMeta) -> int:
        with atomic_write_path(dst) as tmp:
            shutil.copyfile(src, tmp)
            post_meta = file_meta(src)
            if post_meta != src_meta:
                raise RuntimeError(
                    f"source changed during upload: {src} "
                    f"before={src_meta} after={post_meta}"
                )
            os.utime(tmp, ns=(src_meta.mtime_ns, src_meta.mtime_ns))
        return src_meta.size

    def _record_sync_stats(self, component: ComponentSpec, result: SyncResult) -> None:
        stats = self.sync_stats.setdefault(component.name, SyncStats())
        stats.candidate_files += 1
        if result.status == "uploaded":
            stats.uploaded_files += 1
            stats.uploaded_bytes += result.uploaded_bytes
        elif result.status == "skipped":
            stats.skipped_files += 1
        elif result.status == "retry":
            stats.failed_files += 1

    def _publish_snapshot_if_due(self) -> None:
        if not (
            self.enabled and self.owns_startup_lock and self.remote_cache_available
        ):
            return
        with self._snapshot_publish_lock:
            publish_thread = self._snapshot_publish_thread
            if publish_thread is not None and publish_thread.is_alive():
                return
            lease_dir = self._try_acquire_snapshot_publish_lease()
            if lease_dir is None:
                return

            def publish_snapshot() -> None:
                published = False
                try:
                    published = self._publish_snapshot()
                except Exception:
                    logging.warning(
                        "failed to publish JIT cache snapshot", exc_info=True
                    )
                finally:
                    if not published:
                        self._release_snapshot_publish_lease(lease_dir)
                    with self._snapshot_publish_lock:
                        if self._snapshot_publish_thread is threading.current_thread():
                            self._snapshot_publish_thread = None

            publish_thread = threading.Thread(
                target=publish_snapshot,
                name=f"jit-cache-snapshot-{self.run_id}",
                daemon=True,
            )
            self._snapshot_publish_thread = publish_thread
        try:
            publish_thread.start()
        except Exception:
            with self._snapshot_publish_lock:
                if self._snapshot_publish_thread is publish_thread:
                    self._snapshot_publish_thread = None
            self._release_snapshot_publish_lease(lease_dir)
            raise

    def _try_acquire_snapshot_publish_lease(self) -> Path | None:
        remote_root = self.config.remote_root
        if remote_root is None:
            return None
        bucket = int(time.time()) // SNAPSHOT_PUBLISH_INTERVAL_S
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

    def _release_snapshot_publish_lease(self, lease_dir: Path) -> None:
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

    def _publish_snapshot(self) -> bool:
        remote_root = self.config.remote_root
        if remote_root is None:
            return False
        snapshot_path = remote_root / SNAPSHOT_NAME
        remote_tmp_path = remote_root / f"{SNAPSHOT_NAME}.{self.run_id}.tmp"
        local_tmp_path = tmp_sibling(self.config.local_root / SNAPSHOT_NAME)
        start_s = time.monotonic()
        try:
            files, bytes_ = self._create_snapshot_archive(local_tmp_path)
            if files <= 0:
                logging.info("skip publishing empty JIT cache snapshot")
                return False
            remote_tmp_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_tmp_path, remote_tmp_path)
            remote_tmp_path.replace(snapshot_path)
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
                local_tmp_path.unlink()
            with suppress(OSError):
                remote_tmp_path.unlink()

    def _create_snapshot_archive(self, archive: Path) -> tuple[int, int]:
        archive.parent.mkdir(parents=True, exist_ok=True)
        files = 0
        bytes_ = 0
        cctx = zstd.ZstdCompressor()
        with archive.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for component, path, rel in self._iter_remote_snapshot_files():
                        try:
                            with path.open("rb") as source:
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
                                path,
                                exc_info=True,
                            )
        return files, bytes_

    def _iter_remote_snapshot_files(self) -> Iterator[tuple[ComponentSpec, Path, str]]:
        remote_root = self.config.remote_root
        if remote_root is None:
            return
        for component in COMPONENT_SPECS:
            component_root = remote_root / component.name
            if not component_root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(component_root):
                dirnames[:] = [d for d in dirnames if d != "__pycache__"]
                for filename in filenames:
                    path = Path(dirpath) / filename
                    try:
                        rel = path.relative_to(component_root).as_posix()
                    except ValueError:
                        continue
                    if is_tmp_jit_path(rel) or not is_syncable_file(component, rel):
                        continue
                    yield component, path, rel

    def stop(self) -> None:
        self.stop_dirty_tracker()
        if self.enabled and self.owns_startup_lock and self.remote_cache_available:
            with suppress(Exception):
                self.sync_once("periodic_flush")
        self.sync_executor.shutdown(wait=True)
        self.release_startup_lock()

    def stop_dirty_tracker(self) -> None:
        tracker = self.dirty_tracker
        if tracker is None:
            return
        tracker.stop()
        self.dirty_tracker = None

    def sync_once(self, mode: str = "manual_sync") -> dict[str, Any]:
        if not self.enabled:
            self.usage_tracker.report_metrics()
            return self._emit_summary(
                SummaryEvent(
                    mode=mode,
                    result="skipped",
                    total_cost_ms=0,
                    cache_state="disabled",
                ),
                usage=self.usage_tracker.to_summary_dict(),
            )
        start_s = time.monotonic()
        try:
            stats = self._drain_upload_queue()
        except Exception:
            logging.warning("failed to sync remote JIT cache", exc_info=True)
            stats = {name: SyncStats(drain_timed_out=1) for name in self.component_dirs}
        self.usage_tracker.report_metrics()
        has_failures = any(
            stat.drain_timed_out or stat.failed_files for stat in stats.values()
        )
        summary = self._emit_summary(
            SummaryEvent(
                mode=mode,
                result="failed" if has_failures else "success",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="remote_hit",
                stats=stats,
                message="remote JIT cache sync failed" if has_failures else "",
            ),
            usage=self.usage_tracker.to_summary_dict(),
        )
        if not has_failures:
            try:
                self._publish_snapshot_if_due()
            except Exception:
                logging.warning("failed to publish JIT cache snapshot", exc_info=True)
        return summary

    def _drain_upload_queue(self) -> dict[str, SyncStats]:
        timed_out = self._wait_pending_uploads(self.config.remote_timeout_s)
        with self._lock:
            stats = dict(self.sync_stats)
            self.sync_stats = {}
        if timed_out:
            logging.warning(
                "JIT cache upload queue drain timed out with %d pending uploads",
                len(timed_out),
            )
            for component_name, _ in timed_out:
                stats.setdefault(component_name, SyncStats()).drain_timed_out += 1
        return stats

    def _wait_pending_uploads(self, timeout_s: float) -> set[tuple[str, str]]:
        deadline_s = time.monotonic() + max(timeout_s, 0.0)
        with self._lock:
            while self.pending_uploads:
                remaining_s = deadline_s - time.monotonic()
                if remaining_s <= 0:
                    return set(self.pending_uploads)
                self._lock.wait(timeout=remaining_s)
            return set()

    def release_startup_lock(self) -> None:
        lock = self.startup_lock
        self.startup_lock = None
        if lock and lock.is_locked:
            lock.release()

    @staticmethod
    def _snapshot_event(
        start_s: float, result: str, cache_state: str, **kw: Any
    ) -> SummaryEvent:
        return SummaryEvent(
            mode="snapshot_download",
            result=result,
            total_cost_ms=elapsed_ms(start_s),
            cache_state=cache_state,
            **kw,
        )

    def _emit_summary(
        self,
        summary_event: SummaryEvent,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {"timestamp_ms": now_ms()}
        if summary_event.stats is not None:
            aggregate = {
                field.name: sum(
                    getattr(stat, field.name) for stat in summary_event.stats.values()
                )
                for field in fields(SyncStats)
            }
            components = {
                name: data
                for name, stat in summary_event.stats.items()
                if any((data := asdict(stat)).values())
            }
            event["components"] = components
            event.update(aggregate)
        event.update(
            {
                k: v
                for k, v in asdict(summary_event).items()
                if k != "stats" and v is not None and v != ""
            }
        )
        if usage:
            event.update(usage)
        self._write_summary(event)
        return event

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        with atomic_write_path(path) as tmp:
            tmp.write_bytes(
                orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
            )

    def _write_summary(self, event: dict[str, Any]) -> None:
        self._write_json(
            self.config.local_root / SUMMARY_NAME,
            {
                "event": "jit_cache_summary",
                "updated_at_ms": now_ms(),
                "run_id": self.run_id,
                "remote_root": str(self.config.remote_root or ""),
                "local_root": str(self.config.local_root),
                **event,
            },
        )

    def _write_snapshot_complete_marker(
        self,
        snapshot_bytes: int,
        extracted_files: int,
        extracted_bytes: int,
    ) -> None:
        self._write_json(
            self.snapshot_complete_path,
            {
                "event": "jit_cache_snapshot_complete",
                "completed_at_ms": now_ms(),
                "run_id": self.run_id,
                "snapshot_bytes": snapshot_bytes,
                "extracted_files": extracted_files,
                "extracted_bytes": extracted_bytes,
            },
        )
