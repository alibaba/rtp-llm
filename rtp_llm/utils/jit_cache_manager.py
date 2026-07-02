from __future__ import annotations

import logging
import os
import stat
import tarfile
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from rtp_llm.config.py_config_modules import JITConfig
from rtp_llm.metrics import GaugeMetrics, kmonitor
from rtp_llm.utils.jit_cache_common import (
    COMPONENT_BY_NAME,
    COMPONENT_SPECS,
    COPY_CHUNK_SIZE,
    LOCAL_LOCK_NAME,
    PERIODIC_JOIN_TIMEOUT_S,
    SNAPSHOT_COMPLETE_NAME,
    SNAPSHOT_PUBLISH_INTERVAL_S,
    SNAPSHOT_PUBLISH_LEASE_PREFIX,
    STARTUP_WAIT_POLL_S,
    ComponentSpec,
    JitCacheConfig,
    JitDirtyTracker,
    SyncStats,
    aggregate_sync_stats,
    apply_jit_cache_env,
    atomic_write_path,
    cleanup_remote_root,
    clear_component_startup_files,
    component_cache_dir,
    copy_with_deadline,
    find_latest_snapshot,
    get_gpu_info,
    iter_component_sync_files,
    new_snapshot_name,
    normalize_local_path,
    open_snapshot_reader,
    open_snapshot_writer,
    should_sync_file,
    stream_copy,
    strip_gpu_prefix,
    tmp_sibling,
)
from rtp_llm.utils.time_util import current_time_ms, elapsed_ms


def _path_exists(path: Path) -> bool:
    with suppress(OSError):
        return path.exists()
    return False


_USAGE_LOCATIONS = ("local", "remote")
_USAGE_METRICS = (
    ("bytes", GaugeMetrics.JIT_CACHE_USAGE_BYTES_METRIC),
    ("files", GaugeMetrics.JIT_CACHE_USAGE_FILES_METRIC),
)
_USAGE_VALUES = tuple(v for v, _ in _USAGE_METRICS)


class JitCacheManager:
    def __init__(self, jit_config: JITConfig | None = None):
        self.config = JitCacheConfig.from_config(jit_config)
        self.component_dirs: dict[str, tuple[Path, Path]] = {}
        self.dirty_tracker: JitDirtyTracker | None = None

        self._lock = threading.Lock()
        self.sync_stats: dict[str, SyncStats] = {}
        self._pending_count = 0
        # notify_all when pending drains OR when stop() sets _stopping.
        self._pending_cv = threading.Condition(self._lock)
        self._sync_lock = threading.Lock()
        self._snapshot_publish_done = threading.Event()
        self._snapshot_publish_done.set()
        self._stopping = False

        self._periodic_sync_stop = threading.Event()
        self._periodic_sync_thread: threading.Thread | None = None

        self._usage: dict[str, dict[str, dict[str, int]]] = {
            loc: {c.name: {"bytes": 0, "files": 0} for c in COMPONENT_SPECS}
            for loc in _USAGE_LOCATIONS
        }

        self.startup_lock: FileLock | None = None
        self.remote_cache_available = False
        self._sync_executor: ThreadPoolExecutor | None = None
        # Cache mkdir'd remote dirs to skip redundant FUSE RPCs.
        self._known_remote_dirs: set[Path] = set()
        self.snapshot_complete_path = self.config.local_root / SNAPSHOT_COMPLETE_NAME

    @property
    def enabled(self) -> bool:
        return self.config.remote_root is not None

    def _get_or_create_executor_locked(self) -> ThreadPoolExecutor | None:
        """Caller must hold self._lock."""
        if self._stopping:
            return None
        if self._sync_executor is None:
            self._sync_executor = ThreadPoolExecutor(
                max_workers=self.config.sync_workers,
                thread_name_prefix="jit-cache-sync",
            )
        return self._sync_executor

    def ensure_sync_executor(self) -> ThreadPoolExecutor | None:
        with self._lock:
            return self._get_or_create_executor_locked()

    @property
    def owns_startup_lock(self) -> bool:
        return bool(self.startup_lock and self.startup_lock.is_locked)

    def usage_snapshot(self) -> dict[str, dict[str, dict[str, int]]]:
        with self._lock:
            return {
                loc: {name: dict(bucket) for name, bucket in buckets.items()}
                for loc, buckets in self._usage.items()
            }

    @staticmethod
    def usage_totals(
        snapshot: dict[str, dict[str, dict[str, int]]]
    ) -> dict[str, dict[str, int]]:
        return {
            loc: {
                v: sum(bucket[v] for bucket in buckets.values()) for v in _USAGE_VALUES
            }
            for loc, buckets in snapshot.items()
        }

    def _report_usage_metrics(
        self,
        snapshot: dict[str, dict[str, dict[str, int]]],
        totals: dict[str, dict[str, int]] | None = None,
    ) -> None:
        if not kmonitor.is_inited:
            return
        if totals is None:
            totals = self.usage_totals(snapshot)
        for loc in _USAGE_LOCATIONS:
            buckets = {**snapshot[loc], "total": totals[loc]}
            for name, comp in buckets.items():
                for value, metric in _USAGE_METRICS:
                    with suppress(Exception):
                        kmonitor.report(
                            metric, comp[value], {"location": loc, "module": name}
                        )

    def make_summary(
        self,
        mode: str,
        result: str,
        start_s: float,
        *,
        stats: dict[str, SyncStats] | None = None,
        drain_timed_out: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "timestamp_ms": int(current_time_ms()),
            "mode": mode,
            "result": result,
            "total_cost_ms": elapsed_ms(start_s),
        }
        if stats is not None:
            event.update(aggregate_sync_stats(stats))
        if drain_timed_out:
            event["drain_timed_out"] = True
        event.update(extra)
        return event

    def bootstrap(self) -> None:
        local_root = self.config.local_root
        remote_root = self.config.remote_root
        local_root.mkdir(parents=True, exist_ok=True)
        # Env first so component_dirs follow the effective JIT read/write paths.
        apply_jit_cache_env(local_root)
        for component in COMPONENT_SPECS:
            # TRITON_AUTOTUNE_CONFIG_DIR=__builtin__ opts out (use packaged fallback).
            # Must match BUILTIN_CONFIG_SENTINEL in autotune_cache/cache.py.
            if os.environ.get(component.env_name) == "__builtin__":
                continue
            local_dir = normalize_local_path(os.environ[component.env_name])
            local_dir.mkdir(parents=True, exist_ok=True)
            clear_component_startup_files(component, local_dir)
            if remote_root is not None:
                self.component_dirs[component.name] = (
                    local_dir,
                    component_cache_dir(remote_root, component),
                )
        logging.info("JIT cache local=%s remote=%s", local_root, remote_root)

    def prepare(self) -> dict[str, Any]:
        start_s = time.monotonic()
        if not self.enabled:
            return self.make_summary(
                "snapshot_download",
                "skipped",
                start_s,
                cache_state="disabled",
            )
        deadline_s = start_s + self.config.remote_sync_longer_timeout_s
        if self.acquire_startup_lock(0.0):
            summary = self._prepare_as_leader(deadline_s)
        else:
            summary = self._wait_ready_or_takeover(start_s, deadline_s)
        self.remote_cache_available = summary.get("result") not in {
            "failed",
            "timeout",
        }
        if not self.remote_cache_available:
            self.release_startup_lock()
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
        # Marker means a prior boot already pulled — skip re-extract.
        # Clear local_root to force re-pull after ABI-breaking upgrades.
        if self.snapshot_complete_path.exists():
            return self.make_summary(
                "snapshot_download",
                "success",
                time.monotonic(),
                cache_state="local_ready",
            )
        summary = self._pull_snapshot(deadline_s)
        # Only mark on a real hit; a miss must let future boots retry.
        if summary.get("cache_state") == "snapshot_hit":
            self.snapshot_complete_path.touch()
        return summary

    def _wait_ready_or_takeover(
        self, start_s: float, deadline_s: float
    ) -> dict[str, Any]:
        while time.monotonic() < deadline_s:
            if self.snapshot_complete_path.exists():
                return self.make_summary(
                    "snapshot_download",
                    "success",
                    start_s,
                    cache_state="local_ready",
                )
            if self.acquire_startup_lock(0.0):
                return self._prepare_as_leader(deadline_s)
            time.sleep(
                min(STARTUP_WAIT_POLL_S, max(0.0, deadline_s - time.monotonic()))
            )
        return self.make_summary(
            "snapshot_download", "timeout", start_s, cache_state="timeout"
        )

    def _pull_snapshot(self, deadline_s: float) -> dict[str, Any]:
        start_s = time.monotonic()
        summary = partial(self.make_summary, "snapshot_download", start_s=start_s)
        snapshot_path = find_latest_snapshot(self.config.remote_root)
        if snapshot_path is None:
            return summary("skipped", cache_state="snapshot_miss")
        try:
            ext_bytes, ext_files = self._extract_snapshot(snapshot_path, deadline_s)
        except TimeoutError as e:
            return summary("timeout", cache_state="timeout", message=str(e))
        except Exception as e:
            logging.exception("failed to download/extract JIT cache snapshot")
            return summary("failed", cache_state="snapshot_error", message=str(e))
        return summary(
            "success",
            cache_state="snapshot_hit",
            extracted_files=ext_files,
            extracted_bytes=ext_bytes,
        )

    def _extract_snapshot(self, archive: Path, deadline_s: float) -> tuple[int, int]:
        per_component: dict[str, dict[str, int]] = defaultdict(
            lambda: {"bytes": 0, "files": 0}
        )
        gpu_scope = get_gpu_info()
        # Dedup parent mkdirs across all extracted files: a snapshot has hundreds
        # of files sharing a handful of parent dirs, and atomic_write_path would
        # otherwise re-mkdir them every time.
        mkdirs_done: set[Path] = set()
        # Reuse one 4 MB buffer across all members instead of alloc-per-file.
        copy_buffer = bytearray(COPY_CHUNK_SIZE)
        with open_snapshot_reader(archive) as tar:
            for member in tar:
                parts = member.name.split("/")
                if member.name.startswith("/") or ".." in parts:
                    raise RuntimeError(f"unsafe snapshot path: {member.name}")
                # component_dirs is the managed set (opt-outs excluded).
                if parts[0] not in self.component_dirs or not member.isfile():
                    continue
                component = COMPONENT_BY_NAME[parts[0]]
                if component.gpu_scoped and (len(parts) < 2 or parts[1] != gpu_scope):
                    continue
                local_rel = strip_gpu_prefix(component, "/".join(parts[1:]))
                if not should_sync_file(component, local_rel):
                    continue
                if time.monotonic() >= deadline_s:
                    raise TimeoutError("snapshot extraction exceeded prepare timeout")
                source = tar.extractfile(member)
                if source is None:
                    continue
                target = self.component_dirs[component.name][0] / local_rel
                if target.parent not in mkdirs_done:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    mkdirs_done.add(target.parent)
                with source, atomic_write_path(target) as tmp:
                    with tmp.open("wb") as out:
                        stream_copy(source, out, deadline_s, buffer=copy_buffer)
                    os.utime(tmp, (member.mtime, member.mtime))
                bucket = per_component[component.name]
                bucket["bytes"] += member.size
                bucket["files"] += 1
        if per_component:
            with self._lock:
                for name, agg in per_component.items():
                    dst = self._usage["local"][name]
                    dst["bytes"] += agg["bytes"]
                    dst["files"] += agg["files"]
        return (
            sum(a["bytes"] for a in per_component.values()),
            sum(a["files"] for a in per_component.values()),
        )

    @property
    def _can_sync_remote(self) -> bool:
        # remote_cache_available is only set by a successful prepare(),
        return (
            self.remote_cache_available
            and self.owns_startup_lock
            and not self._stopping
        )

    def start_background_sync(self) -> None:
        if not self._can_sync_remote or self.dirty_tracker is not None:
            return
        tracker = JitDirtyTracker(self.component_dirs, self.enqueue_upload)
        if not tracker.start():
            return
        self.dirty_tracker = tracker
        # Reaching here implies dirty_tracker was None (outer guard),
        # and thread/tracker are set+cleared together — so thread is also None.
        self._periodic_sync_thread = threading.Thread(
            target=self._periodic_sync_loop,
            name="jit-periodic-sync",
            daemon=True,
        )
        self._periodic_sync_thread.start()

    def _periodic_sync_loop(self) -> None:
        interval_s = max(self.config.remote_sync_longer_timeout_s, STARTUP_WAIT_POLL_S)
        while not self._periodic_sync_stop.wait(interval_s):
            try:
                self.sync_once("periodic_flush")
            except Exception:
                logging.exception("periodic JIT cache sync failed")

    def enqueue_upload(self, component: ComponentSpec, rel_path: str) -> bool:
        # Executor check/create + pending++ under one lock (halves lock traffic per enqueue).
        with self._lock:
            executor = self._get_or_create_executor_locked()
            if executor is None:
                return False
            self._pending_count += 1
        try:
            executor.submit(self._upload_task, component, rel_path)
            return True
        except RuntimeError:
            with self._lock:
                self._decrement_pending_locked()
            return False

    def _decrement_pending_locked(self) -> None:
        """Caller must hold self._lock."""
        self._pending_count -= 1
        if self._pending_count == 0:
            self._pending_cv.notify_all()

    def _upload_task(self, component: ComponentSpec, rel_path: str) -> None:
        local_dir, remote_dir = self.component_dirs[component.name]
        deadline_s = time.monotonic() + self.config.remote_sync_timeout_s
        try:
            uploaded: int | None = self._copy_atomic(
                local_dir / rel_path, remote_dir / rel_path, deadline_s
            )
        except Exception:
            logging.exception(
                "failed to upload JIT cache file %s/%s", component.name, rel_path
            )
            uploaded = None
        # One lock pass covers stats, usage, and pending decrement.
        with self._lock:
            stats = self.sync_stats.setdefault(component.name, SyncStats())
            if uploaded is None:
                stats.failed_files += 1
            elif uploaded > 0:
                stats.uploaded_files += 1
                stats.uploaded_bytes += uploaded
                bucket = self._usage["remote"][component.name]
                bucket["bytes"] += uploaded
                bucket["files"] += 1
            else:
                stats.skipped_files += 1
            self._decrement_pending_locked()

    def _copy_atomic(self, local: Path, remote: Path, deadline_s: float) -> int:
        if _path_exists(remote):
            return 0
        try:
            local_stat = local.stat()
        except FileNotFoundError:
            return 0
        if local_stat.st_size <= 0:
            return 0
        self._ensure_remote_parent(remote.parent)
        try:
            with atomic_write_path(remote) as tmp:
                copy_with_deadline(local, tmp, deadline_s)
        except PermissionError:
            if _path_exists(remote):
                return 0
            raise
        return local_stat.st_size

    def _ensure_remote_parent(self, parent: Path) -> None:
        with self._lock:
            if parent in self._known_remote_dirs:
                return
            self._known_remote_dirs.add(parent)
        parent.mkdir(parents=True, exist_ok=True)

    def sync_once(self, mode: str = "manual_sync") -> dict[str, Any]:
        start_s = time.monotonic()
        if self._stopping:
            return self.make_summary(mode, "skipped", start_s, reason="stopping")
        if not self.enabled:
            return self.make_summary(mode, "skipped", start_s, cache_state="disabled")
        if not self._sync_lock.acquire(blocking=False):
            return self.make_summary(
                mode, "skipped", start_s, reason="sync in progress"
            )
        try:
            return self._sync_once_impl(mode)
        finally:
            self._sync_lock.release()

    def _sync_once_impl(self, mode: str) -> dict[str, Any]:
        start_s = time.monotonic()
        stats, drain_timed_out = self._drain_upload_queue()
        usage_snap = self.usage_snapshot()
        usage_totals = self.usage_totals(usage_snap)
        self._report_usage_metrics(usage_snap, usage_totals)
        has_failures = drain_timed_out or any(s.failed_files for s in stats.values())
        extras: dict[str, Any] = {
            f"{loc}_cache": {**usage_totals[loc], "components": usage_snap[loc]}
            for loc in _USAGE_LOCATIONS
            if usage_totals[loc]["files"]
        }
        if has_failures:
            extras["message"] = "remote JIT cache sync failed"
        summary = self.make_summary(
            mode,
            "failed" if has_failures else "success",
            start_s,
            cache_state="remote_hit",
            stats=stats,
            drain_timed_out=drain_timed_out,
            **extras,
        )
        if not has_failures:
            self._publish_snapshot_if_due()
        return summary

    def _drain_upload_queue(self) -> tuple[dict[str, SyncStats], bool]:
        timeout = max(self.config.remote_sync_timeout_s, 0.0)
        with self._lock:
            # wait_for returns False on timeout; stop() satisfies the predicate
            # via _stopping without needing to force _pending_count to 0.
            drained = self._pending_cv.wait_for(
                lambda: self._pending_count == 0 or self._stopping,
                timeout=timeout,
            )
            stats = self.sync_stats
            self.sync_stats = {}
            pending = self._pending_count
        if not drained:
            logging.warning(
                "JIT cache upload queue drain timed out with %d pending", pending
            )
        return stats, not drained

    def _publish_snapshot_if_due(self) -> None:
        if not self._can_sync_remote:
            return
        with self._lock:
            if self._stopping or not self._snapshot_publish_done.is_set():
                return
            self._snapshot_publish_done.clear()

        lease = self._try_acquire_publish_lease()
        if lease is None:
            self._snapshot_publish_done.set()
            return
        lease_dir, lease_bucket = lease

        future = None
        executor = self.ensure_sync_executor()
        if executor is not None:
            with suppress(RuntimeError):
                future = executor.submit(self._snapshot_publish_task, lease_bucket)

        if future is None:
            self._release_publish_lease(lease_dir)
            self._snapshot_publish_done.set()
            return

        def _on_done(
            fut: Any, lease_dir: Path = lease_dir, lease_bucket: int = lease_bucket
        ) -> None:
            # Single decision point for lease lifecycle: keep on success (bucket marker),
            # release+cleanup on cancel / exception / empty-archive returned False.
            published = (
                not fut.cancelled() and fut.exception() is None and fut.result() is True
            )
            if not published:
                cleanup_remote_root(lease_dir.parent, current_lease_bucket=lease_bucket)
                self._release_publish_lease(lease_dir)
            self._snapshot_publish_done.set()

        future.add_done_callback(_on_done)

    def _snapshot_publish_task(self, lease_bucket: int) -> bool:
        try:
            return self._publish_snapshot(lease_bucket)
        except Exception:
            logging.exception("failed to publish JIT cache snapshot")
            return False

    def _try_acquire_publish_lease(self) -> tuple[Path, int] | None:
        remote_root = self.config.remote_root
        bucket = int(time.time()) // SNAPSHOT_PUBLISH_INTERVAL_S
        lease_dir = remote_root / f"{SNAPSHOT_PUBLISH_LEASE_PREFIX}{bucket}"
        try:
            lease_dir.mkdir()
        except FileExistsError:
            return None
        except OSError:
            logging.exception("failed to acquire JIT snapshot publish lease")
            return None
        return lease_dir, bucket

    @staticmethod
    def _release_publish_lease(lease_dir: Path) -> None:
        try:
            lease_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            logging.exception(
                "failed to release JIT snapshot publish lease: %s", lease_dir
            )

    def _publish_snapshot(self, current_lease_bucket: int) -> bool:
        remote_root = self.config.remote_root
        snapshot_path = remote_root / new_snapshot_name()
        local_tmp = tmp_sibling(self.config.local_root / "snapshot_build.tar.zst")
        start_s = time.monotonic()
        publish_deadline = start_s + self.config.remote_sync_longer_timeout_s
        try:
            files, bytes_ = self._create_snapshot_archive(local_tmp, publish_deadline)
            if files <= 0:
                logging.info("skip publishing empty JIT cache snapshot")
                return False
            # remote_root already exists (lease mkdir succeeded above), skip parent mkdir
            with atomic_write_path(snapshot_path) as snapshot_tmp:
                copy_with_deadline(local_tmp, snapshot_tmp, publish_deadline)
            cleanup_remote_root(remote_root, current_lease_bucket=current_lease_bucket)
            logging.info(
                "published JIT cache snapshot %s files=%d bytes=%d cost_ms=%d",
                snapshot_path,
                files,
                bytes_,
                elapsed_ms(start_s),
            )
            return True
        finally:
            local_tmp.unlink(missing_ok=True)

    def _create_snapshot_archive(
        self, archive: Path, deadline_s: float
    ) -> tuple[int, int]:
        files = 0
        bytes_ = 0
        remote_root = self.config.remote_root
        if remote_root is None:
            return files, bytes_
        # Enumerate remote union; prefer local reads to avoid FUSE small-file I/O.
        # component_dirs is the effective managed set (opt-outs excluded).
        deadline_hit = False
        with open_snapshot_writer(archive) as tar:
            for name, (local_dir, _) in self.component_dirs.items():
                if deadline_hit:
                    break
                component = COMPONENT_BY_NAME[name]
                for _, rel in iter_component_sync_files(
                    remote_root / name, component, log_errors=True
                ):
                    if time.monotonic() >= deadline_s:
                        logging.warning(
                            "snapshot archive creation exceeded deadline after %d files",
                            files,
                        )
                        deadline_hit = True
                        break
                    local_path = local_dir / strip_gpu_prefix(component, rel)
                    remote_path = remote_root / name / rel
                    try:
                        try:
                            source = local_path.open("rb")
                        except FileNotFoundError:
                            source = remote_path.open("rb")
                        with source:
                            source_stat = os.fstat(source.fileno())
                            if source_stat.st_size <= 0 or not stat.S_ISREG(
                                source_stat.st_mode
                            ):
                                continue
                            info = tarfile.TarInfo(f"{name}/{rel}")
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
                            "failed to add JIT cache file to snapshot: %s (local=%s)",
                            remote_path,
                            local_path,
                        )
        return files, bytes_

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
            executor = self._sync_executor
            self._sync_executor = None
            # Wake in-flight drain waiters — predicate now sees _stopping.
            self._pending_cv.notify_all()
        try:
            tracker = self.dirty_tracker
            self.dirty_tracker = None
            if tracker is not None:
                tracker.stop()

            self._periodic_sync_stop.set()  # idempotent even if thread was never started
            t = self._periodic_sync_thread
            if t is not None and t is not threading.current_thread():
                t.join(timeout=PERIODIC_JOIN_TIMEOUT_S)

            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)
        finally:
            # Snapshot publish runs in the sync executor — use the sync-side
            # timeout, not the watchdog observer's.
            self._snapshot_publish_done.wait(timeout=PERIODIC_JOIN_TIMEOUT_S)
            self.release_startup_lock()

    def release_startup_lock(self) -> None:
        with self._lock:
            lock = self.startup_lock
            self.startup_lock = None
        if lock:
            lock.release()
