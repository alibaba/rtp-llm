from __future__ import annotations

import logging
import os
import queue
import shutil
import tarfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, fields
from itertools import chain
from pathlib import Path, PurePath, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import orjson
import zstandard as zstd
from filelock import FileLock, Timeout
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
PAX_MTIME_NS = "RTP.mtime_ns"
LOCAL_LOCK_NAME = ".rtp_jit_local.lock"
LOCAL_LOCK_DIR_NAME = ".rtp_jit_locks"
READY_DIR_NAME = ".rtp_jit_ready"
SNAPSHOT_COMPLETE_NAME = ".rtp_jit_snapshot_complete"
SUMMARY_NAME = "summary.json"

REMOTE_JIT_DIR_ENV = "REMOTE_JIT_DIR"
LOCAL_JIT_CACHE_DIR_ENV = "LOCAL_JIT_CACHE_DIR"
JIT_PREPARE_TIMEOUT_ENV = "JIT_PREPARE_TIMEOUT_S"
JIT_SYNC_WORKERS_ENV = "JIT_SYNC_WORKERS"
TRITON_AUTOTUNE_CACHE_MODE_ENV = "TRITON_AUTOTUNE_CACHE_MODE"
RUN_ID_ENV = "RTP_JIT_CACHE_RUN_ID"
DEFAULT_JIT_PREPARE_TIMEOUT_S = 30.0
DEFAULT_JIT_SYNC_WORKERS = 32
MAX_UPLOAD_ATTEMPTS = 3
UPLOAD_QUEUE_DRAIN_POLL_S = 0.05
# H20 JIT workloads cover all components: DeepGEMM may only emit created,
# Triton publishes final cache files via moved, while FlashInfer,
# torch_extensions, and triton_autotune direct writes are caught on closed.
WATCHDOG_UPLOAD_EVENTS = {"closed", "moved", "created"}


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    env_name: str
    sync_suffixes: Tuple[str, ...]


# Only completed runtime artifacts enter the remote cache.
COMPONENT_SPECS = (
    ComponentSpec(
        "flashinfer", "FLASHINFER_WORKSPACE_BASE", (".so", ".cubin", ".json")
    ),
    ComponentSpec("triton", "TRITON_CACHE_DIR", (".so", ".cubin", ".json")),
    ComponentSpec(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        (".json", ".pkl", ".pickle"),
    ),
    ComponentSpec("deep_gemm", "DG_JIT_CACHE_DIR", (".so", ".cubin", ".json")),
    ComponentSpec(
        "torch_extensions", "TORCH_EXTENSIONS_DIR", (".so", ".cubin", ".json")
    ),
)
COMPONENT_BY_NAME = {component.name: component for component in COMPONENT_SPECS}


class FileMeta(NamedTuple):
    size: int
    mtime_ns: int


@dataclass
class SyncStats:
    candidate_files: int = 0
    dirty_files: int = 0
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    failed_files: int = 0
    failed_components: int = 0


@dataclass(frozen=True)
class CacheUsage:
    bytes: int = 0
    files: int = 0


@dataclass(frozen=True)
class JitCacheUsageSnapshot:
    local_total: Optional[CacheUsage] = None
    local_components: Optional[Dict[str, CacheUsage]] = None
    remote_total: Optional[CacheUsage] = None
    remote_components: Optional[Dict[str, CacheUsage]] = None

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.local_total:
            result["local_cache"] = _usage_to_dict(
                self.local_total, self.local_components
            )
        if self.remote_total:
            result["remote_cache"] = _usage_to_dict(
                self.remote_total, self.remote_components
            )
        return result


def _usage_to_dict(
    total: CacheUsage, components: Optional[Dict[str, CacheUsage]]
) -> Dict[str, Any]:
    usage: Dict[str, Any] = {"bytes": total.bytes, "files": total.files}
    if components:
        usage["components"] = {
            name: {"bytes": component_usage.bytes, "files": component_usage.files}
            for name, component_usage in components.items()
        }
    return usage


@dataclass(frozen=True)
class UploadTask:
    component: ComponentSpec
    key: str
    attempts: int = 0


@dataclass(frozen=True)
class SyncResult:
    status: str
    uploaded_bytes: int = 0


@dataclass
class SummaryEvent:
    mode: str
    result: str
    total_cost_ms: int
    cache_state: str = ""
    stats: Optional[Dict[str, SyncStats]] = None
    message: str = ""
    snapshot_result: Optional[str] = None
    snapshot_message: str = ""
    snapshot_bytes: Optional[int] = None
    extracted_files: Optional[int] = None
    extracted_bytes: Optional[int] = None


def summarize_component_stats(
    stats: Dict[str, SyncStats],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    stat_fields = fields(SyncStats)
    aggregate = {
        field.name: sum(getattr(stat, field.name) for stat in stats.values())
        for field in stat_fields
    }
    components = {}
    for component in COMPONENT_SPECS:
        stat = stats.get(component.name)
        if stat is None:
            continue
        stat_dict = asdict(stat)
        if any(stat_dict[field.name] for field in stat_fields):
            components[component.name] = stat_dict
    return aggregate, components


@dataclass
class JitCacheConfig:
    remote_root: Optional[Path]
    local_root: Path
    prepare_timeout_s: float
    sync_workers: int

    @classmethod
    def from_env_and_config(cls, jit_config: Any = None) -> "JitCacheConfig":
        def resolve(
            attr: str, env: str, default: Any, *, env_first: bool = False
        ) -> Any:
            cfg_value = (
                getattr(jit_config, attr, None) if jit_config is not None else None
            )
            env_value = os.environ.get(env)
            cfg_ok = cfg_value not in (None, "")
            env_ok = env_value not in (None, "")
            if env_first:
                return env_value if env_ok else (cfg_value if cfg_ok else default)
            return cfg_value if cfg_ok else (env_value if env_ok else default)

        remote_dir = resolve("remote_jit_dir", REMOTE_JIT_DIR_ENV, "")
        local_dir = resolve(
            "local_jit_cache_dir", LOCAL_JIT_CACHE_DIR_ENV, "./jit_cache"
        )

        return cls(
            remote_root=resolve_remote_root(remote_dir),
            local_root=Path(str(local_dir)).expanduser().absolute(),
            prepare_timeout_s=float(
                resolve(
                    "jit_prepare_timeout_s",
                    JIT_PREPARE_TIMEOUT_ENV,
                    DEFAULT_JIT_PREPARE_TIMEOUT_S,
                )
            ),
            sync_workers=max(
                1,
                int(
                    resolve(
                        "jit_sync_workers",
                        JIT_SYNC_WORKERS_ENV,
                        DEFAULT_JIT_SYNC_WORKERS,
                        env_first=True,
                    )
                ),
            ),
        )


def now_ms() -> int:
    return int(time.time() * 1000)


def elapsed_ms(start_s: float) -> int:
    return int((time.monotonic() - start_s) * 1000)


def _get_process_run_id() -> str:
    return f"{int(time.time() * 1000)}-{os.getpid()}-{uuid.uuid4().hex}"


def ensure_jit_cache_run_id() -> str:
    run_id = os.environ.get(RUN_ID_ENV)
    if not run_id:
        run_id = _get_process_run_id()
        os.environ[RUN_ID_ENV] = run_id
    return run_id


def resolve_remote_root(remote_jit_dir: Any) -> Optional[Path]:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"remote_jit_dir must be an absolute directory, got {text!r}")
    path = path.absolute()
    if not path.is_dir():
        raise ValueError(
            f"remote_jit_dir must be an existing absolute directory, got {text!r}"
        )
    return path


def file_meta(path: Path, *, stat: Optional[os.stat_result] = None) -> FileMeta:
    stat = stat or path.stat()
    return FileMeta(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


@contextmanager
def atomic_write_path(path: Path) -> Iterable[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(path)
    try:
        yield tmp
        tmp.replace(path)
    finally:
        with suppress(OSError):
            tmp.unlink()


def read_json_dict(path: Path) -> Dict[str, Any]:
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: Dict[str, Any]) -> None:
    with atomic_write_path(path) as tmp:
        tmp.write_bytes(
            orjson.dumps(value, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )


def is_cache_file_static(component: ComponentSpec, rel: str | PurePath) -> bool:
    name = rel.name if isinstance(rel, PurePath) else rel.rsplit("/", 1)[-1]
    return name.endswith(component.sync_suffixes)


def cache_usage(root: Path) -> CacheUsage:
    total_bytes = 0
    files = 0
    if not root.is_dir():
        return CacheUsage()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not dirname.startswith(".rtp_jit_") and dirname != "__pycache__"
        ]
        for filename in filenames:
            try:
                stat = os.lstat(os.path.join(dirpath, filename))
            except OSError:
                continue
            total_bytes += stat.st_size
            files += 1
    return CacheUsage(bytes=total_bytes, files=files)


def component_cache_usage(root: Path) -> Tuple[CacheUsage, Dict[str, CacheUsage]]:
    components: Dict[str, CacheUsage] = {}
    total_bytes = 0
    total_files = 0
    for component in COMPONENT_SPECS:
        usage = cache_usage(root / component.name)
        components[component.name] = usage
        total_bytes += usage.bytes
        total_files += usage.files
    return CacheUsage(bytes=total_bytes, files=total_files), components


def bootstrap_cache_env(config: JitCacheConfig) -> Dict[str, Tuple[Path, Path]]:
    layouts: Dict[str, Tuple[Path, Path]] = {}
    config.local_root.mkdir(parents=True, exist_ok=True)
    for component in COMPONENT_SPECS:
        local_dir = config.local_root / component.name
        local_dir.mkdir(parents=True, exist_ok=True)
        os.environ[component.env_name] = str(local_dir)
        if config.remote_root is not None:
            layouts[component.name] = (local_dir, config.remote_root / component.name)
    os.environ[LOCAL_JIT_CACHE_DIR_ENV] = str(config.local_root)
    os.environ[TRITON_AUTOTUNE_CACHE_MODE_ENV] = "cached"
    if config.remote_root is not None or os.environ.get(REMOTE_JIT_DIR_ENV):
        # Child ranks inherit concrete component cache dirs; clear the generic
        # remote knob so they do not try to become remote cache managers again.
        os.environ[REMOTE_JIT_DIR_ENV] = ""
    logging.info("JIT cache local=%s remote=%s", config.local_root, config.remote_root)
    return layouts


def _report_cache_usage_metric(
    kmonitor: Any,
    metric: Any,
    total: CacheUsage,
    components: Dict[str, CacheUsage],
) -> None:
    for module, usage in chain([("total", total)], components.items()):
        kmonitor_tags = {"module": module}
        kmonitor.report(metric, usage.bytes, {**kmonitor_tags, "value": "bytes"})
        kmonitor.report(metric, usage.files, {**kmonitor_tags, "value": "files"})


def report_jit_cache_usage(
    local_root: Path,
    remote_root: Optional[Path],
) -> Optional[JitCacheUsageSnapshot]:
    try:
        from rtp_llm.metrics import GaugeMetrics, kmonitor

        if not getattr(kmonitor, "_inited", False):
            return None

        local_total, local_components = component_cache_usage(local_root)
        _report_cache_usage_metric(
            kmonitor,
            GaugeMetrics.JIT_CACHE_LOCAL_USAGE_METRIC,
            local_total,
            local_components,
        )

        remote_total = None
        remote_components = None
        if remote_root is not None:
            remote_total, remote_components = component_cache_usage(remote_root)
            _report_cache_usage_metric(
                kmonitor,
                GaugeMetrics.JIT_CACHE_REMOTE_USAGE_METRIC,
                remote_total,
                remote_components,
            )

        return JitCacheUsageSnapshot(
            local_total=local_total,
            local_components=local_components,
            remote_total=remote_total,
            remote_components=remote_components,
        )
    except Exception:
        logging.warning("failed to report JIT cache usage", exc_info=True)
        return None


class JitDirtyTracker:
    def __init__(
        self,
        layouts: Dict[str, Tuple[Path, Path]],
        enqueue_upload: Callable[[ComponentSpec, str], bool],
    ):
        self.layouts = layouts
        self.enqueue_upload = enqueue_upload
        self.observer: Optional[Any] = None

    def start(self) -> bool:
        if self.observer is not None:
            return True
        observer = Observer()
        try:
            for component_name, (local_dir, _) in self.layouts.items():
                observer.schedule(
                    _JitCacheEventHandler(
                        self,
                        COMPONENT_BY_NAME[component_name],
                        local_dir,
                    ),
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

    def add_key(self, component: str, key: str) -> None:
        self.enqueue_upload(COMPONENT_BY_NAME[component], key)


class _JitCacheEventHandler(FileSystemEventHandler):
    def __init__(self, tracker: JitDirtyTracker, component: ComponentSpec, root: Path):
        self.tracker = tracker
        self.component = component
        self.root = root

    def on_any_event(self, event: Any) -> None:
        if event.is_directory or event.event_type not in WATCHDOG_UPLOAD_EVENTS:
            return
        path = Path(event.dest_path if event.event_type == "moved" else event.src_path)
        try:
            rel = path.relative_to(self.root).as_posix()
        except ValueError:
            return
        if is_cache_file_static(self.component, rel):
            self.tracker.add_key(self.component.name, rel)


class JitCacheManager:
    def __init__(self, jit_config: Any = None):
        self.config = JitCacheConfig.from_env_and_config(jit_config)
        self.run_id = ensure_jit_cache_run_id()
        self.remote_root_text = str(self.config.remote_root or "")
        self.ready_dir = self.config.local_root / READY_DIR_NAME
        self.layouts: Dict[str, Tuple[Path, Path]] = {}
        self.dirty_tracker: Optional[JitDirtyTracker] = None
        self.stop_event = threading.Event()
        self.upload_queue: queue.Queue[UploadTask] = queue.Queue()
        self.worker_futures = []
        self.summary_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        self.pending_lock = threading.Lock()
        self.sync_stats: Dict[str, SyncStats] = {}
        self.pending_uploads: Set[Tuple[str, str]] = set()
        self.ready_path = self.ready_dir / f"{self.run_id}.json"
        self.snapshot_complete_path = self.ready_dir / SNAPSHOT_COMPLETE_NAME
        self.startup_lock_path = (
            self.config.local_root / LOCAL_LOCK_DIR_NAME / LOCAL_LOCK_NAME
        )
        self.startup_lock: Optional[FileLock] = None
        self.remote_cache_available = False
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

    def acquire_startup_lock(self, timeout_s: float = 0.0) -> bool:
        if self.owns_startup_lock:
            return True
        self.startup_lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(self.startup_lock_path), thread_local=False)
        try:
            lock.acquire(timeout=max(timeout_s, 0.0))
        except Timeout:
            return False
        self.startup_lock = lock
        return True

    def bootstrap_env(self) -> None:
        self.layouts = bootstrap_cache_env(self.config)

    def _emit_summary(
        self,
        summary_event: SummaryEvent,
        usage: Optional[JitCacheUsageSnapshot] = None,
    ) -> Dict[str, Any]:
        event = {
            "timestamp_ms": now_ms(),
            "mode": summary_event.mode,
            "result": summary_event.result,
            "total_cost_ms": summary_event.total_cost_ms,
        }
        if summary_event.stats:
            aggregate, components = summarize_component_stats(summary_event.stats)
            event["components"] = components
            event.update(aggregate)
        base_fields = {"mode", "result", "total_cost_ms", "stats"}
        for field in fields(SummaryEvent):
            if field.name in base_fields:
                continue
            value = getattr(summary_event, field.name)
            if value is not None and value != "":
                event[field.name] = value
        if usage is not None:
            event.update(usage.to_dict())
        self._write_summary(event)
        return event

    def _write_summary(self, event: Dict[str, Any]) -> None:
        with self.summary_lock:
            path = self.ready_dir / SUMMARY_NAME
            summary = {
                "event": "jit_cache_summary",
                "updated_at_ms": now_ms(),
                "run_id": self.run_id,
                "remote_root": self.remote_root_text,
                "local_root": str(self.config.local_root),
                **event,
            }
            write_json_atomic(path, summary)

    def _write_ready(self, summary: Dict[str, Any]) -> None:
        ready = {
            "event": "jit_cache_ready",
            "timestamp_ms": now_ms(),
            "run_id": self.run_id,
            "remote_root": self.remote_root_text,
            "local_root": str(self.config.local_root),
            "summary": summary,
        }
        write_json_atomic(self.ready_path, ready)

    def _read_ready_summary(self) -> Optional[Dict[str, Any]]:
        if not self.ready_path.exists():
            return None
        ready = read_json_dict(self.ready_path)
        summary = ready.get("summary", {})
        if (
            ready.get("run_id") == self.run_id
            and ready.get("remote_root") == self.remote_root_text
            and ready.get("local_root") == str(self.config.local_root)
            and isinstance(summary, dict)
        ):
            return summary
        return None

    def _local_snapshot_complete(self) -> bool:
        return self.snapshot_complete_path.exists()

    def _pull_snapshot(self, deadline_s: Optional[float]) -> SummaryEvent:
        start_s = time.monotonic()
        snapshot_path = self.config.remote_root / SNAPSHOT_NAME
        if not snapshot_path.exists():
            return SummaryEvent(
                mode="snapshot_download",
                result="skipped",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="snapshot_miss",
            )

        tmp: Optional[Path] = None
        extract_root: Optional[Path] = None
        extracted_metas: List[FileMeta] = []
        snapshot_bytes = 0
        try:
            tmp = tmp_sibling(self.config.local_root / SNAPSHOT_NAME)
            extract_root = tmp_sibling(self.config.local_root / ".jit_extract")
            snapshot_bytes = snapshot_path.stat().st_size
            self.config.local_root.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot_path, tmp)
            if deadline_s is not None and time.monotonic() >= deadline_s:
                raise TimeoutError("snapshot copy exceeded prepare timeout")
            self._extract_snapshot(tmp, extract_root, deadline_s, extracted_metas)
            self._commit_extract(extract_root)
            extracted_files = len(extracted_metas)
            extracted_bytes = sum(meta.size for meta in extracted_metas)
            self.snapshot_complete_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_complete_path.touch()
            return SummaryEvent(
                mode="snapshot_download",
                result="success",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="snapshot_hit",
                snapshot_bytes=snapshot_bytes if snapshot_bytes > 0 else None,
                extracted_files=extracted_files if extracted_files > 0 else None,
                extracted_bytes=extracted_bytes if extracted_bytes > 0 else None,
            )
        except TimeoutError as e:
            return SummaryEvent(
                mode="snapshot_download",
                result="timeout",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="timeout",
                message=str(e),
                snapshot_bytes=snapshot_bytes if snapshot_bytes > 0 else None,
            )
        except Exception as e:
            logging.exception("failed to download/extract JIT cache snapshot")
            return SummaryEvent(
                mode="snapshot_download",
                result="failed",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="snapshot_error",
                message=str(e),
            )
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
            if extract_root is not None and extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)

    def _extract_snapshot(
        self,
        archive: Path,
        dst_root: Path,
        deadline_s: Optional[float],
        extracted_metas: Optional[List[FileMeta]] = None,
    ) -> None:
        dctx = zstd.ZstdDecompressor()
        with archive.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        if deadline_s is not None and time.monotonic() >= deadline_s:
                            raise TimeoutError(
                                "snapshot extraction exceeded prepare timeout; "
                                "remote bootstrap skipped"
                            )
                        member_path = PurePosixPath(member.name)
                        if (
                            not member_path.parts
                            or member_path.is_absolute()
                            or ".." in member_path.parts
                        ):
                            raise RuntimeError(f"unsafe snapshot path: {member.name}")
                        component_name = member_path.parts[0]
                        component = COMPONENT_BY_NAME.get(component_name)
                        if component is None:
                            continue
                        target = dst_root.joinpath(*member_path.parts)
                        if member.isdir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        if not member.isfile():
                            continue
                        source = tar.extractfile(member)
                        if source is None:
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open("wb") as out:
                            shutil.copyfileobj(source, out)
                        mtime_ns_text = member.pax_headers.get(PAX_MTIME_NS)
                        if mtime_ns_text is not None:
                            mtime_ns = int(mtime_ns_text)
                            os.utime(target, ns=(mtime_ns, mtime_ns))
                        else:
                            os.utime(target, (member.mtime, member.mtime))
                        if extracted_metas is None:
                            continue
                        rel = PurePosixPath(*member_path.parts[1:])
                        if is_cache_file_static(component, rel):
                            extracted_metas.append(file_meta(target))

    def _commit_extract(self, extract_root: Path) -> None:
        if not extract_root.exists():
            return

        def move_into(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir() and dst.is_dir():
                for child in src.iterdir():
                    move_into(child, dst / child.name)
                src.rmdir()
                return
            if dst.exists() or dst.is_symlink():
                if dst.is_dir() and not dst.is_symlink():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            shutil.move(str(src), str(dst))

        for child in extract_root.iterdir():
            move_into(child, self.config.local_root / child.name)

    def prepare(self) -> Dict[str, Any]:
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
        deadline_s = start_s + max(self.config.prepare_timeout_s, 0.0)
        ready_summary = self._read_ready_summary()
        if ready_summary is not None:
            summary = ready_summary
        elif self.acquire_startup_lock(0.0):
            summary = self._prepare_as_startup_owner(deadline_s)
        else:
            summary = self._wait_ready_or_takeover(start_s, deadline_s)
        self.remote_cache_available = self.enabled and summary.get("result") not in {
            "failed",
            "timeout",
        }
        if not self.remote_cache_available:
            self.release_startup_lock()
        return summary

    def _wait_ready_or_takeover(
        self, start_s: float, deadline_s: float
    ) -> Dict[str, Any]:
        while True:
            try:
                ready_summary = self._read_ready_summary()
            except Exception:
                ready_summary = None
            if ready_summary is not None:
                return ready_summary
            if self.acquire_startup_lock(0.0):
                return self._prepare_as_startup_owner(deadline_s)
            remaining_s = deadline_s - time.monotonic()
            if remaining_s <= 0:
                break
            time.sleep(min(0.05, remaining_s))
        return self._emit_summary(
            SummaryEvent(
                mode="snapshot_download",
                result="timeout",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="timeout",
            )
        )

    def _prepare_as_startup_owner(self, deadline_s: float) -> Dict[str, Any]:
        ready_summary = self._read_ready_summary()
        if ready_summary is not None:
            return ready_summary
        if self._local_snapshot_complete():
            summary = self._emit_summary(
                SummaryEvent(
                    mode="snapshot_download",
                    result="skipped",
                    total_cost_ms=0,
                    cache_state="local_hit",
                    message=(
                        "local JIT cache snapshot complete marker exists; "
                        "remote bootstrap skipped"
                    ),
                )
            )
        else:
            summary = self._emit_summary(self._pull_snapshot(deadline_s))
        self._write_ready(summary)
        return summary

    def start_background_sync(self) -> None:
        if not (
            self.enabled and self.owns_startup_lock and self.remote_cache_available
        ):
            return
        self.start_dirty_tracker()
        if self.dirty_tracker is None:
            return
        self.start_upload_workers()

    def stop(self) -> None:
        try:
            if self.enabled and self.owns_startup_lock and self.remote_cache_available:
                self.sync_once("periodic_flush")
        finally:
            try:
                self.stop_dirty_tracker()
            finally:
                try:
                    self.stop_event.set()
                    self.sync_executor.shutdown(wait=True, cancel_futures=True)
                finally:
                    self.release_startup_lock()

    def release_startup_lock(self) -> None:
        lock = self.startup_lock
        self.startup_lock = None
        if lock and lock.is_locked:
            lock.release()

    def start_upload_workers(self) -> None:
        if self.worker_futures:
            return
        self.stop_event.clear()
        self.worker_futures = [
            self.sync_executor.submit(self._upload_worker)
            for _ in range(self.config.sync_workers)
        ]

    def _upload_worker(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.upload_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_upload_task(task, retry=True)
            finally:
                self.upload_queue.task_done()

    def _snapshot_stats(self) -> Dict[str, SyncStats]:
        with self.stats_lock:
            return {
                component: SyncStats(**asdict(stats))
                for component, stats in self.sync_stats.items()
            }

    def _diff_stats(
        self, before: Dict[str, SyncStats], after: Dict[str, SyncStats]
    ) -> Dict[str, SyncStats]:
        result = {}
        field_names = [field.name for field in fields(SyncStats)]
        component_names = set(before.keys()) | set(self.layouts.keys())
        for component in component_names:
            old = before.get(component, SyncStats())
            new = after.get(component, SyncStats())
            result[component] = SyncStats(
                **{
                    name: getattr(new, name) - getattr(old, name)
                    for name in field_names
                }
            )
        return result

    def _record_sync_result(self, component: ComponentSpec, result: SyncResult) -> None:
        with self.stats_lock:
            stats = self.sync_stats.setdefault(component.name, SyncStats())
            stats.candidate_files += 1
            if result.status == "uploaded":
                stats.dirty_files += 1
                stats.uploaded_files += 1
                stats.uploaded_bytes += result.uploaded_bytes
            elif result.status == "retry":
                stats.failed_files += 1

    def enqueue_upload(self, component: ComponentSpec, key: str) -> bool:
        normalized_key = PurePosixPath(key).as_posix()
        pending_key = (component.name, normalized_key)
        with self.pending_lock:
            if pending_key in self.pending_uploads:
                return False
            self.pending_uploads.add(pending_key)
        self.upload_queue.put(UploadTask(component, normalized_key))
        return True

    def _release_pending_upload(self, task: UploadTask) -> None:
        with self.pending_lock:
            self.pending_uploads.discard((task.component.name, task.key))

    def _process_upload_task(self, task: UploadTask, retry: bool = True) -> SyncResult:
        upload_error: Optional[BaseException] = None
        try:
            result = self.upload_candidate(task.component, task.key)
        except Exception as e:
            upload_error = e
            result = SyncResult("retry")
        should_retry = (
            retry
            and result.status == "retry"
            and task.attempts + 1 < MAX_UPLOAD_ATTEMPTS
            and not self.stop_event.is_set()
        )
        if should_retry:
            self.upload_queue.put(
                UploadTask(task.component, task.key, attempts=task.attempts + 1)
            )
            return result
        if upload_error is not None:
            logging.warning(
                "failed to upload JIT cache file %s",
                task,
                exc_info=(
                    type(upload_error),
                    upload_error,
                    upload_error.__traceback__,
                ),
            )
        self._record_sync_result(task.component, result)
        self._release_pending_upload(task)
        return result

    def _drain_upload_queue(self) -> Dict[str, SyncStats]:
        before = self._snapshot_stats()
        warned = False
        while True:
            pending = self.upload_queue.unfinished_tasks
            workers_alive = bool(self.worker_futures) and any(
                not future.done() for future in self.worker_futures
            )
            if pending == 0:
                break
            if workers_alive:
                time.sleep(UPLOAD_QUEUE_DRAIN_POLL_S)
                continue
            if self.worker_futures and not warned:
                logging.warning(
                    "JIT cache upload workers stopped with %d unfinished tasks; "
                    "draining uploads inline",
                    pending,
                )
                warned = True
            try:
                task = self.upload_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self._process_upload_task(task, retry=True)
            finally:
                self.upload_queue.task_done()
        return self._diff_stats(before, self._snapshot_stats())

    def _emit_sync_summary(
        self,
        mode: str,
        start_s: float,
        stats: Dict[str, SyncStats],
        usage: Optional[JitCacheUsageSnapshot],
        message: str = "",
    ) -> Dict[str, Any]:
        has_failures = any(
            stat.failed_components or stat.failed_files for stat in stats.values()
        )
        return self._emit_summary(
            SummaryEvent(
                mode=mode,
                result="failed" if has_failures else "success",
                total_cost_ms=elapsed_ms(start_s),
                cache_state="remote_hit",
                stats=stats,
                message=message,
            ),
            usage=usage,
        )

    def sync_once(self, mode: str = "manual_sync") -> Dict[str, Any]:
        if not self.enabled:
            usage = report_jit_cache_usage(self.config.local_root, None)
            return self._emit_summary(
                SummaryEvent(
                    mode=mode,
                    result="skipped",
                    total_cost_ms=0,
                    cache_state="disabled",
                ),
                usage=usage,
            )
        start_s = time.monotonic()
        try:
            stats = self._drain_upload_queue()
            usage = report_jit_cache_usage(
                self.config.local_root, self.config.remote_root
            )
            return self._emit_sync_summary(mode, start_s, stats, usage)
        except Exception:
            logging.warning("failed to sync remote JIT cache", exc_info=True)
            stats = {name: SyncStats(failed_components=1) for name in self.layouts}
            return self._emit_sync_summary(
                mode,
                start_s,
                stats,
                None,
                message="remote JIT cache sync failed",
            )

    def start_dirty_tracker(self) -> None:
        if not self.enabled or self.dirty_tracker is not None:
            return
        tracker = JitDirtyTracker(self.layouts, self.enqueue_upload)
        if tracker.start():
            self.dirty_tracker = tracker

    def stop_dirty_tracker(self) -> None:
        tracker = self.dirty_tracker
        if tracker is None:
            return
        tracker.stop()
        self.dirty_tracker = None

    def upload_candidate(
        self,
        component: ComponentSpec,
        key: str,
    ) -> SyncResult:
        rel = PurePosixPath(key)
        if not is_cache_file_static(component, rel):
            return SyncResult("skipped")
        src_root, dst_root = self.layouts[component.name]
        src = src_root / rel
        try:
            src_meta = file_meta(src)
        except FileNotFoundError:
            return SyncResult("skipped")
        except OSError:
            return SyncResult("retry")
        if src_meta.size <= 0:
            return SyncResult("retry")
        uploaded_bytes = self.copy_final(src, dst_root / rel, src_meta)
        return SyncResult("uploaded", uploaded_bytes)

    def copy_final(self, src: Path, dst: Path, src_meta: FileMeta) -> int:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_sibling(dst)
        try:
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return src_meta.size
