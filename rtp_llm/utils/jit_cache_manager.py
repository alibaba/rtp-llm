from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import tarfile
import threading
import time
import uuid
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import zstandard as zstd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

COMPONENTS = (
    "flashinfer",
    "triton",
    "triton_autotune",
    "deep_gemm",
    "torch_extensions",
)

# Local cache roots managed by each JIT subsystem.
ENV_BY_COMPONENT = {
    "flashinfer": "FLASHINFER_WORKSPACE_BASE",
    "triton": "TRITON_CACHE_DIR",
    "triton_autotune": "TRITON_AUTOTUNE_CONFIG_DIR",
    "deep_gemm": "DG_JIT_CACHE_DIR",
    "torch_extensions": "TORCH_EXTENSIONS_DIR",
}

# Only completed runtime artifacts enter the remote cache.
SYNC_SUFFIXES = {
    "flashinfer": (".so", ".cubin", ".json"),
    "triton": (".so", ".cubin", ".json"),
    "triton_autotune": (".json", ".pkl", ".pickle"),
    "deep_gemm": (".so", ".cubin", ".json"),
    "torch_extensions": (".so", ".cubin", ".json"),
}

SKIP_SUFFIXES = (
    ".tmp",
    ".lock",
    ".part",
    ".partial",
    ".incomplete",
    ".o",
    ".cu",
    ".ptx",
    ".log",
)

SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
LOCK_NAME = ".rtp_jit_worker.lock"
READY_DIR_NAME = ".rtp_jit_ready"
SUMMARY_NAME = "summary.json"
SNAPSHOT_COMPLETE_NAME = ".rtp_jit_snapshot_complete"
PAX_MTIME_NS = "RTP.mtime_ns"

SKIP_NAMES = {
    "build.ninja",
    SUMMARY_NAME,
    ".rtp_jit_manifest.json",
    ".rtp_jit_synced_manifest.json",
    SNAPSHOT_NAME,
}

REMOTE_JIT_DIR_ENV = "REMOTE_JIT_DIR"
LOCAL_JIT_CACHE_DIR_ENV = "LOCAL_JIT_CACHE_DIR"
JIT_PREPARE_TIMEOUT_ENV = "JIT_PREPARE_TIMEOUT_S"
JIT_SYNC_INTERVAL_ENV = "JIT_SYNC_INTERVAL_S"
TRITON_AUTOTUNE_CACHE_MODE_ENV = "TRITON_AUTOTUNE_CACHE_MODE"
LOG_PATH_ENV = "LOG_PATH"
RUN_ID_ENV = "RTP_JIT_CACHE_RUN_ID"
FileMeta = Tuple[int, int]


@dataclass
class ComponentLayout:
    local_dir: Path
    remote_dir: Path


@dataclass
class ComponentStat:
    candidate_files: int = 0
    dirty_files: int = 0
    uploaded_files: int = 0
    uploaded_bytes: int = 0
    failed_files: int = 0
    failed_components: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {field: getattr(self, field) for field in STAT_FIELDS}


STAT_FIELDS = tuple(field.name for field in fields(ComponentStat))


def now_ms() -> int:
    return int(time.time() * 1000)


def cost_ms(start: float) -> int:
    return int((time.time() - start) * 1000)


def ensure_jit_cache_run_id() -> str:
    run_id = os.environ.get(RUN_ID_ENV)
    if not run_id:
        run_id = f"{now_ms()}-{os.getpid()}"
        os.environ[RUN_ID_ENV] = run_id
    return run_id


def remote_root(remote_jit_dir: Any) -> Tuple[Optional[Path], str]:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None, ""
    if "://" in text or not Path(text).is_absolute():
        return (
            None,
            "remote_jit_dir must be an absolute mounted path, got "
            f"{text!r}; remote sync is disabled",
        )
    path = Path(text)
    if not path.exists() or not path.is_dir():
        return (
            None,
            "remote_jit_dir must be an existing mounted directory, got "
            f"{text!r}; remote sync is disabled",
        )
    return path, ""


def file_meta(path: Path) -> FileMeta:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def tmp_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")


def tar_member_mtime_ns(member: tarfile.TarInfo) -> Optional[int]:
    mtime_ns = member.pax_headers.get(PAX_MTIME_NS)
    if mtime_ns is not None:
        try:
            return int(mtime_ns)
        except ValueError:
            pass
    try:
        return int(Decimal(str(member.mtime)) * Decimal(1_000_000_000))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if (
            path.is_file()
            and "__pycache__" not in path.parts
            and path.name not in SKIP_NAMES
        ):
            yield path


def is_cache_file_static(component: str, rel: Path) -> bool:
    name = rel.name
    if name.startswith(".nfs") or name in SKIP_NAMES:
        return False
    if name.endswith(SKIP_SUFFIXES):
        return False
    return name.endswith(SYNC_SUFFIXES[component])


def iter_static_cache_files(
    component: str, root: Path
) -> Iterable[Tuple[Path, Path, FileMeta]]:
    if not root.exists():
        return
    for path in iter_files(root):
        rel = path.relative_to(root)
        if is_cache_file_static(component, rel):
            meta = file_meta(path)
            if meta[0] > 0:
                yield path, rel, meta


class JitDirtyTracker:
    def __init__(self, layouts: Dict[str, ComponentLayout]):
        self.layouts = layouts
        self.lock = threading.Lock()
        self.dirty: Dict[str, Set[str]] = {name: set() for name in layouts}
        self.observer: Optional[Any] = None

    def start(self) -> bool:
        if self.observer is not None:
            return True
        observer = Observer()
        try:
            for component, layout in self.layouts.items():
                observer.schedule(
                    _JitCacheEventHandler(self, component, layout.local_dir),
                    str(layout.local_dir),
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

    def add_candidate(self, component: str, path: Path, root: Path) -> None:
        try:
            rel = path.relative_to(root)
        except ValueError:
            return
        if is_cache_file_static(component, rel):
            with self.lock:
                self.dirty.setdefault(component, set()).add(rel.as_posix())

    def add_keys(self, component: str, keys: Iterable[str]) -> None:
        with self.lock:
            dirty = self.dirty.setdefault(component, set())
            dirty.update(keys)

    def pop_candidates(self, component: str) -> Set[str]:
        with self.lock:
            candidates = self.dirty.setdefault(component, set())
            self.dirty[component] = set()
            return set(candidates)


class _JitCacheEventHandler(FileSystemEventHandler):
    def __init__(self, tracker: JitDirtyTracker, component: str, root: Path):
        self.tracker = tracker
        self.component = component
        self.root = root

    def dispatch(self, event: Any) -> None:
        if event.is_directory:
            return
        if event.event_type not in {"closed", "moved"}:
            return
        path = event.dest_path if event.event_type == "moved" else event.src_path
        self.tracker.add_candidate(self.component, Path(path), self.root)


class JitCacheManager:
    def __init__(self, jit_config: Any = None):
        remote_jit_dir = (
            getattr(jit_config, "remote_jit_dir", None)
            if jit_config is not None
            else None
        ) or os.environ.get(REMOTE_JIT_DIR_ENV, "")
        self.remote_root, self.invalid_remote_reason = remote_root(remote_jit_dir)
        local_dir = (
            getattr(jit_config, "local_jit_cache_dir", None)
            if jit_config is not None
            else None
        ) or os.environ.get(LOCAL_JIT_CACHE_DIR_ENV, "./jit_cache")
        self.local_root = Path(os.path.expanduser(local_dir))
        self.prepare_timeout_s = float(
            (
                getattr(jit_config, "jit_prepare_timeout_s", None)
                if jit_config is not None
                else None
            )
            or os.environ.get(JIT_PREPARE_TIMEOUT_ENV)
            or 30
        )
        self.sync_interval_s = float(
            (
                getattr(jit_config, "jit_sync_interval_s", None)
                if jit_config is not None
                else None
            )
            or os.environ.get(JIT_SYNC_INTERVAL_ENV)
            or 300
        )
        self.run_id = ensure_jit_cache_run_id()
        self.ready_path = self.local_root / READY_DIR_NAME / f"{self.run_id}.json"
        self.snapshot_complete_path = (
            self.local_root / READY_DIR_NAME / SNAPSHOT_COMPLETE_NAME
        )
        self.layouts: Dict[str, ComponentLayout] = {}
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.dirty_tracker: Optional[JitDirtyTracker] = None
        self.summary_lock = threading.Lock()
        self.lock_file = None
        self.owns_lock = False
        self.remote_cache_available = False

    @property
    def enabled(self) -> bool:
        return self.remote_root is not None

    def bootstrap_env(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        for component in COMPONENTS:
            local_dir = self.local_root / component
            local_dir.mkdir(parents=True, exist_ok=True)
            os.environ[ENV_BY_COMPONENT[component]] = str(local_dir)
            if self.remote_root is not None:
                self.layouts[component] = ComponentLayout(
                    local_dir=local_dir,
                    remote_dir=self.remote_root / component,
                )
        os.environ[LOCAL_JIT_CACHE_DIR_ENV] = str(self.local_root)
        os.environ[TRITON_AUTOTUNE_CACHE_MODE_ENV] = "cached"
        if self.enabled or self.invalid_remote_reason:
            os.environ[REMOTE_JIT_DIR_ENV] = ""
        if self.invalid_remote_reason:
            # Keep lower-level JIT libraries on the local cache path; do not let
            # URI/relative values create fake remote cache directories.
            logging.debug(self.invalid_remote_reason)
        logging.info("JIT cache local=%s remote=%s", self.local_root, self.remote_root)

    def prepare(self) -> Dict[str, Any]:
        start = time.time()
        if self.invalid_remote_reason:
            return self.summary(
                "snapshot_download",
                "disabled",
                "invalid_config",
                {},
                cost_ms(start),
                self.invalid_remote_reason,
            )
        if not self.enabled:
            return self.summary("snapshot_download", "disabled", "skipped", {}, 0)
        if not self.acquire_lock():
            summary = self.wait_ready(start)
            self.remote_cache_available = self.can_use_remote_cache(summary)
            return summary

        summary = self.prepare_as_owner(start, self.prepare_timeout_s)
        self.remote_cache_available = self.can_use_remote_cache(summary)
        return summary

    def prepare_as_owner(self, start: float, timeout_s: float) -> Dict[str, Any]:
        self.clear_startup_state()
        snapshot_copy_deadline = time.time() + max(timeout_s, 0.0)
        summary = self.bootstrap_from_remote(snapshot_copy_deadline)
        self.write_ready(summary)
        return summary

    def can_use_remote_cache(self, summary: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        return summary.get("result") not in {"failed", "timeout", "invalid_config"}

    def start_background_sync(self) -> None:
        if not (self.enabled and self.owns_lock and self.remote_cache_available):
            return
        self.start_dirty_tracker()
        if self.dirty_tracker is None:
            return
        if self.sync_interval_s > 0:
            self.start_worker()

    def stop(self) -> None:
        try:
            self.stop_event.set()
            worker = self.worker
            if (
                worker is None
                and self.enabled
                and self.owns_lock
                and self.remote_cache_available
            ):
                self.sync_once("shutdown_flush")
            if worker and worker.is_alive():
                worker.join()
            self.stop_dirty_tracker()
        finally:
            self.release_lock()

    def sync_once(self, mode: str = "manual_sync") -> Dict[str, Any]:
        if not self.enabled:
            return self.summary(mode, "disabled", "skipped", {}, 0)
        start = time.time()
        stats = self.sync_layouts()
        result = (
            "failed"
            if any(
                stat.failed_components or stat.failed_files for stat in stats.values()
            )
            else "success"
        )
        return self.summary(mode, "remote_hit", result, stats, cost_ms(start))

    def start_worker(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.worker = threading.Thread(
            target=self.worker_loop,
            name="jit-cache-worker",
            daemon=True,
        )
        self.worker.start()

    def worker_loop(self) -> None:
        try:
            try:
                if self.sync_interval_s <= 0:
                    self.stop_event.wait()
                else:
                    while self.enabled:
                        if self.stop_event.wait(self.sync_interval_s):
                            break
                        self.sync_once("periodic_flush")
            finally:
                if self.enabled:
                    self.sync_once("shutdown_flush")
                self.stop_dirty_tracker()
        except Exception:
            logging.exception("JIT cache worker loop failed")

    def start_dirty_tracker(self) -> None:
        if not self.enabled or self.dirty_tracker is not None:
            return
        tracker = JitDirtyTracker(self.layouts)
        if tracker.start():
            self.dirty_tracker = tracker

    def stop_dirty_tracker(self) -> None:
        tracker = self.dirty_tracker
        if tracker is None:
            return
        tracker.stop()
        self.dirty_tracker = None

    def bootstrap_from_remote(
        self, snapshot_copy_deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        if self.local_cache_has_stable_files():
            return self.summary(
                "snapshot_download",
                "local_hit",
                "skipped",
                {},
                0,
                "local JIT cache already has stable files; remote bootstrap skipped",
            )
        return self.pull_snapshot(snapshot_copy_deadline)

    def pull_snapshot(
        self, snapshot_copy_deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        start = time.time()
        snapshot_path = self.remote_root / SNAPSHOT_NAME
        if not snapshot_path.exists():
            return self.summary(
                "snapshot_download",
                "snapshot_miss",
                "skipped",
                {},
                cost_ms(start),
            )
        tmp = self.local_root / f".{SNAPSHOT_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        extract_root = (
            self.local_root / f".jit_extract.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            snapshot_bytes = snapshot_path.stat().st_size
            self.local_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snapshot_path, tmp)
            if (
                snapshot_copy_deadline is not None
                and time.time() > snapshot_copy_deadline
            ):
                return self.summary(
                    "snapshot_download",
                    "timeout",
                    "timeout",
                    {},
                    cost_ms(start),
                    "snapshot copy exceeded prepare timeout; extraction skipped",
                    extra={"snapshot_bytes": snapshot_bytes},
                )
            self.extract_snapshot(tmp, extract_root, snapshot_copy_deadline)
            self.commit_extracted_snapshot(extract_root)
            extracted_files, extracted_bytes = self.count_stable_files(self.local_root)
            self.write_snapshot_complete()
            return self.summary(
                "snapshot_download",
                "snapshot_hit",
                "success",
                {},
                cost_ms(start),
                extra={
                    "snapshot_bytes": snapshot_bytes,
                    "extracted_files": extracted_files,
                    "extracted_bytes": extracted_bytes,
                },
            )
        except TimeoutError as e:
            return self.summary(
                "snapshot_download",
                "timeout",
                "timeout",
                {},
                cost_ms(start),
                str(e),
                extra={"snapshot_bytes": snapshot_bytes},
            )
        except Exception as e:
            logging.exception("failed to download/extract JIT cache snapshot")
            return self.summary(
                "snapshot_download",
                "snapshot_hit",
                "failed",
                {},
                cost_ms(start),
                str(e),
            )
        finally:
            if tmp.exists():
                tmp.unlink()
            if extract_root.exists():
                shutil.rmtree(extract_root, ignore_errors=True)

    def extract_snapshot(
        self, archive: Path, dst_root: Path, deadline: Optional[float] = None
    ) -> None:
        dctx = zstd.ZstdDecompressor()
        with archive.open("rb") as compressed:
            with dctx.stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tar:
                    for member in tar:
                        if deadline is not None and time.time() > deadline:
                            raise TimeoutError(
                                "snapshot extraction exceeded prepare timeout; "
                                "remote bootstrap skipped"
                            )
                        self.extract_tar_member(tar, member, dst_root)

    def commit_extracted_snapshot(self, extract_root: Path) -> None:
        for child in extract_root.iterdir():
            dst = self.local_root / child.name
            if dst.exists():
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            os.replace(child, dst)

    def extract_tar_member(
        self, tar: tarfile.TarFile, member: tarfile.TarInfo, dst_root: Path
    ) -> None:
        member_path = PurePosixPath(member.name)
        if (
            not member_path.parts
            or member_path.is_absolute()
            or ".." in member_path.parts
        ):
            raise RuntimeError(f"unsafe snapshot path: {member.name}")
        target = dst_root.joinpath(*member_path.parts)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            return
        if not member.isfile():
            return
        source = tar.extractfile(member)
        if source is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_sibling(target)
        try:
            with tmp.open("wb") as out:
                shutil.copyfileobj(source, out)
            os.replace(tmp, target)
            mtime_ns = tar_member_mtime_ns(member)
            if mtime_ns is not None:
                os.utime(target, ns=(mtime_ns, mtime_ns))
        finally:
            if tmp.exists():
                tmp.unlink()

    def sync_layouts(self) -> Dict[str, ComponentStat]:
        return {
            name: self.sync_component(name, layout)
            for name, layout in self.layouts.items()
        }

    def sync_component(self, component: str, layout: ComponentLayout) -> ComponentStat:
        stat = ComponentStat()
        tracker = self.dirty_tracker
        if tracker is None:
            return stat
        candidates = tracker.pop_candidates(component)
        retry_candidates: Set[str] = set()
        stat.candidate_files = len(candidates)
        if not candidates:
            return stat
        src_root = layout.local_dir
        dst_root = layout.remote_dir
        try:
            for key in sorted(candidates):
                rel = Path(key)
                src = src_root / rel
                if not is_cache_file_static(component, rel):
                    continue
                try:
                    src_meta = file_meta(src)
                except FileNotFoundError:
                    continue
                except OSError:
                    stat.failed_files += 1
                    retry_candidates.add(key)
                    continue
                if src_meta[0] <= 0:
                    retry_candidates.add(key)
                    continue
                dst = dst_root / rel
                stat.dirty_files += 1
                try:
                    uploaded_bytes = self.copy_final(src, dst, src_meta)
                except Exception:
                    stat.failed_files += 1
                    retry_candidates.add(key)
                    continue
                if uploaded_bytes > 0:
                    stat.uploaded_files += 1
                    stat.uploaded_bytes += uploaded_bytes
        except Exception:
            stat.failed_components += 1
            retry_candidates.update(candidates)
            logging.warning(
                "failed to upload JIT cache component %s",
                component,
            )
        if retry_candidates:
            tracker.add_keys(component, retry_candidates)
        if stat.failed_files:
            logging.warning(
                "failed to copy %s JIT cache files for component %s",
                stat.failed_files,
                component,
            )
        return stat

    def copy_final(self, src: Path, dst: Path, src_meta: FileMeta) -> int:
        try:
            if dst.exists() and file_meta(dst) == src_meta:
                return 0
        except OSError:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_sibling(dst)
        try:
            shutil.copy2(src, tmp)
            if file_meta(src) != src_meta:
                raise OSError(f"source changed during copy: {src}")
            os.replace(tmp, dst)
            return src_meta[0]
        finally:
            if tmp.exists():
                tmp.unlink()

    def summary(
        self,
        mode: str,
        cache_state: str,
        result: str,
        stats: Dict[str, ComponentStat],
        total_cost_ms: int,
        message: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        aggregate, components = self.summarize_stats(stats)
        event = {
            "timestamp_ms": now_ms(),
            "mode": mode,
            "cache_state": cache_state,
            "result": result,
            "total_cost_ms": total_cost_ms,
        }
        if stats:
            event["components"] = components
            event.update(aggregate)
        if extra:
            event.update(extra)
        if message:
            event["message"] = message
        self.write_summary(event)
        return event

    def summarize_stats(
        self, stats: Dict[str, ComponentStat]
    ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
        aggregate = {
            field: sum(getattr(stat, field) for stat in stats.values())
            for field in STAT_FIELDS
        }
        components = {}
        for name in COMPONENTS:
            stat = stats.get(name)
            if stat is None:
                continue
            stat_dict = stat.to_dict()
            if any(stat_dict[field] for field in STAT_FIELDS):
                components[name] = stat_dict
        return aggregate, components

    def local_cache_has_stable_files(self) -> bool:
        if not self.snapshot_complete_path.exists():
            return False
        files, _ = self.count_stable_files(self.local_root)
        return files > 0

    def count_stable_files(self, root: Path) -> Tuple[int, int]:
        files = 0
        total_bytes = 0
        for component in COMPONENTS:
            for _, _, meta in iter_static_cache_files(component, root / component):
                files += 1
                total_bytes += meta[0]
        return files, total_bytes

    def wait_ready(self, start: float) -> Dict[str, Any]:
        deadline = time.time() + max(self.prepare_timeout_s, 0.0)
        while True:
            try:
                ready = json.loads(self.ready_path.read_text(encoding="utf-8"))
                summary = ready.get("summary", {})
                if (
                    ready.get("run_id") == self.run_id
                    and ready.get("remote_root") == str(self.remote_root)
                    and ready.get("local_root") == str(self.local_root)
                    and isinstance(summary, dict)
                ):
                    return summary
            except Exception:
                pass
            if self.acquire_lock():
                return self.prepare_as_owner(start, max(0.0, deadline - time.time()))
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            time.sleep(max(0.0, min(0.05, remaining)))
        return self.summary(
            "snapshot_download", "timeout", "timeout", {}, cost_ms(start)
        )

    def clear_startup_state(self) -> None:
        try:
            self.ready_path.unlink(missing_ok=True)
        except Exception:
            logging.debug("failed to clear JIT cache startup state")

    def write_snapshot_complete(self) -> None:
        try:
            self.snapshot_complete_path.parent.mkdir(parents=True, exist_ok=True)
            self.snapshot_complete_path.touch()
        except Exception:
            logging.warning("failed to write JIT cache snapshot-complete marker")

    def write_ready(self, summary: Dict[str, Any]) -> None:
        try:
            ready = {
                "event": "jit_cache_ready",
                "timestamp_ms": now_ms(),
                "run_id": self.run_id,
                "remote_root": str(self.remote_root),
                "local_root": str(self.local_root),
                "summary": summary,
            }
            self.write_json(self.ready_path, ready)
        except Exception:
            logging.warning("failed to write JIT cache ready marker")

    def write_summary(self, event: Dict[str, Any]) -> None:
        try:
            with self.summary_lock:
                path = self.summary_dir() / SUMMARY_NAME
                summary = self.load_summary(path)
                summary["updated_at_ms"] = now_ms()
                summary.setdefault("events", {})[event["mode"]] = event
                self.write_json(path, summary)
        except Exception:
            logging.debug("failed to write JIT cache summary")

    def load_summary(self, path: Path) -> Dict[str, Any]:
        expected = {
            "event": "jit_cache_summary",
            "run_id": self.run_id,
            "remote_root": str(self.remote_root or ""),
            "local_root": str(self.local_root),
        }
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            if (
                isinstance(summary, dict)
                and summary.get("run_id") == expected["run_id"]
                and summary.get("remote_root") == expected["remote_root"]
                and summary.get("local_root") == expected["local_root"]
                and isinstance(summary.get("events"), dict)
            ):
                summary["event"] = expected["event"]
                return summary
        except Exception:
            pass
        return {**expected, "updated_at_ms": now_ms(), "events": {}}

    def summary_dir(self) -> Path:
        return Path(os.environ.get(LOG_PATH_ENV, "logs")) / "jit_cache"

    def write_json(
        self, path: Path, value: Dict[str, Any], indent: Optional[int] = 2
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_sibling(path)
        try:
            tmp.write_text(
                json.dumps(value, indent=indent, sort_keys=True), encoding="utf-8"
            )
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def acquire_lock(self) -> bool:
        if self.owns_lock:
            return True
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.lock_file = open(self.local_root / LOCK_NAME, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.owns_lock = True
            return True
        except BlockingIOError:
            self.lock_file.close()
            self.lock_file = None
            return False

    def release_lock(self) -> None:
        if self.lock_file is None:
            return
        try:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()
        finally:
            self.lock_file = None
            self.owns_lock = False


class JitSnapshotBuilder:
    def __init__(self, remote_jit_dir: str):
        self.remote_root, self.invalid_remote_reason = remote_root(remote_jit_dir)

    def build(self) -> Dict[str, Any]:
        if self.invalid_remote_reason or self.remote_root is None:
            raise ValueError(self.invalid_remote_reason or "remote_jit_dir is required")
        if not self.remote_root.exists():
            raise FileNotFoundError(
                f"remote_jit_dir does not exist: {self.remote_root}"
            )

        tmp = self.remote_root / f"{SNAPSHOT_NAME}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        snapshot = self.remote_root / SNAPSHOT_NAME
        included: List[Tuple[Path, str]] = []

        try:
            for component in COMPONENTS:
                component_root = self.remote_root / component
                for path, rel, _ in iter_static_cache_files(component, component_root):
                    included.append((path, str(Path(component) / rel)))

            self.write_snapshot(tmp, included)
            os.replace(tmp, snapshot)
            return {
                "result": "success",
                "snapshot_path": str(snapshot),
            }
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    def write_snapshot(self, tmp: Path, included: List[Tuple[Path, str]]) -> None:
        cctx = zstd.ZstdCompressor()
        with tmp.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for path, arcname in included:
                        tar_info = tar.gettarinfo(str(path), arcname=arcname)
                        tar_info.pax_headers = dict(tar_info.pax_headers)
                        tar_info.pax_headers[PAX_MTIME_NS] = str(
                            path.stat().st_mtime_ns
                        )
                        with path.open("rb") as source:
                            tar.addfile(tar_info, source)
