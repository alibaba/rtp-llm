import errno
import json
import os
import tarfile
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from io import BytesIO
from pathlib import Path
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit_cache_module
from rtp_llm.utils.jit_cache_manager import (
    COMPONENT_SPECS,
    SNAPSHOT_NAME,
    JitCacheManager,
)


class FakeFileEvent:
    def __init__(self, event_type: str, src_path: str, dest_path: str = ""):
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = False


class Config:
    def __init__(self, local_root: str, timeout_s: float = 5):
        self.local_jit_cache_dir = local_root
        self.jit_prepare_timeout_s = timeout_s


def iter_component_files(root: Path, component):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [dirname for dirname in dirnames if dirname != "__pycache__"]
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            try:
                stat = path.stat(follow_symlinks=False)
            except OSError:
                continue
            rel = path.relative_to(root)
            if stat.st_size <= 0 or not jit_cache_module.is_cache_file_static(
                component, rel
            ):
                continue
            yield path, rel, jit_cache_module.FileMeta(
                size=stat.st_size, mtime_ns=stat.st_mtime_ns
            )


def add_path_to_tracker(tracker, component, path: Path, root: Path) -> None:
    rel = path.relative_to(root)
    if jit_cache_module.is_cache_file_static(component, rel):
        tracker.add_key(component.name, rel.as_posix())


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        for env_name in (
            "REMOTE_JIT_DIR",
            "JIT_PREPARE_TIMEOUT_S",
            "RTP_JIT_CACHE_RUN_ID",
        ):
            os.environ.pop(env_name, None)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self,
        remote=None,
        timeout_s: float = 5,
        create_remote: bool = True,
    ) -> JitCacheManager:
        if remote is not None:
            if remote:
                remote_path = Path(remote)
                if create_remote and remote_path.is_absolute():
                    remote_path.mkdir(parents=True, exist_ok=True)
                os.environ["REMOTE_JIT_DIR"] = remote
            else:
                os.environ.pop("REMOTE_JIT_DIR", None)
        config = Config(str(self.root / "local"), timeout_s=timeout_s)
        manager = JitCacheManager(config)
        manager.bootstrap_env()
        return manager

    def write_snapshot(self, remote: Path):
        snapshot = remote / SNAPSHOT_NAME
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for component in COMPONENT_SPECS:
                        component_root = remote / component.name
                        for path, rel, _ in iter_component_files(
                            component_root, component
                        ):
                            info = tar.gettarinfo(
                                str(path),
                                arcname=f"{component.name}/{rel.as_posix()}",
                            )
                            with path.open("rb") as source:
                                tar.addfile(info, source)
        return snapshot

    def read_summary(self):
        return json.loads(
            (self.root / "local" / ".rtp_jit_ready" / "summary.json").read_text(
                encoding="utf-8"
            )
        )

    def component(self, name: str):
        return jit_cache_module.COMPONENT_BY_NAME[name]

    def enqueue_file(self, manager: JitCacheManager, component_name: str, key: str):
        component = self.component(component_name)
        root = self.root / "local" / component_name
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(component_name, encoding="utf-8")
        tracker = manager.dirty_tracker or jit_cache_module.JitDirtyTracker(
            manager.layouts, manager.enqueue_upload
        )
        manager.dirty_tracker = tracker
        add_path_to_tracker(tracker, component, path, root)
        return path

    def wait_for_queue_empty(self, manager: JitCacheManager):
        deadline = time.time() + 5
        while time.time() < deadline:
            if (
                manager.upload_queue.unfinished_tasks == 0
                and manager.upload_queue.empty()
            ):
                return
            time.sleep(0.01)
        self.fail("timed out waiting for upload queue to drain")

    def test_run_id_comes_from_env_for_spawned_ranks(self):
        os.environ["RTP_JIT_CACHE_RUN_ID"] = "shared-run"

        manager = self.make_manager()

        self.assertEqual(jit_cache_module.ensure_jit_cache_run_id(), "shared-run")
        self.assertEqual(manager.run_id, "shared-run")
        self.assertEqual(manager.ready_path.name, "shared-run.json")

    def test_default_run_id_includes_current_pid(self):
        with mock.patch.object(jit_cache_module.time, "time", return_value=1.0):
            with mock.patch.object(
                jit_cache_module.uuid,
                "uuid4",
                return_value=mock.Mock(hex="child-run"),
            ):
                run_id = jit_cache_module.ensure_jit_cache_run_id()

        self.assertEqual(run_id, f"1000-{os.getpid()}-child-run")
        self.assertEqual(os.environ["RTP_JIT_CACHE_RUN_ID"], run_id)

    def test_bootstrap_sets_local_env_and_hides_remote_env(self):
        manager = self.make_manager(str(self.root / "remote"))

        self.assertTrue(manager.enabled)
        self.assertEqual(os.environ["REMOTE_JIT_DIR"], "")
        self.assertEqual(os.environ["LOCAL_JIT_CACHE_DIR"], str(self.root / "local"))
        self.assertEqual(
            set(manager.layouts), {component.name for component in COMPONENT_SPECS}
        )
        for component in COMPONENT_SPECS:
            self.assertEqual(
                os.environ[component.env_name],
                str(self.root / "local" / component.name),
            )

    def test_prepare_without_remote_returns_disabled_summary(self):
        manager = self.make_manager()

        summary = manager.prepare()

        self.assertFalse(manager.enabled)
        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "disabled")
        self.assertNotIn("components", summary)

    def test_remote_config_falls_back_to_env_and_fails_fast(self):
        remote = self.root / "remote_from_env"
        remote.mkdir()
        os.environ["REMOTE_JIT_DIR"] = str(remote)
        config = Config(str(self.root / "local"))
        config.remote_jit_dir = ""

        manager = JitCacheManager(config)

        self.assertTrue(manager.enabled)
        self.assertEqual(manager.config.remote_root, remote)

        with self.assertRaisesRegex(ValueError, "absolute directory"):
            self.make_manager("relative/jit_cache")
        with self.assertRaisesRegex(ValueError, "existing absolute directory"):
            self.make_manager(str(self.root / "missing_remote"), create_remote=False)

    def test_file_meta_requires_exact_metadata_match(self):
        left = jit_cache_module.FileMeta(size=16, mtime_ns=10_000)

        self.assertNotEqual(left, jit_cache_module.FileMeta(size=16, mtime_ns=10_999))
        self.assertNotEqual(left, jit_cache_module.FileMeta(size=17, mtime_ns=10_000))
        self.assertEqual(left, left)

    def test_prepare_snapshot_miss_does_not_scan_remote_files(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertEqual(self.read_summary()["mode"], "periodic_flush")

    def test_prepare_uses_external_snapshot_and_marks_memory_cache(self):
        remote = self.root / "remote"
        remote_file = remote / "triton" / "kernel" / "a.so"
        remote_file.parent.mkdir(parents=True)
        remote_file.write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        sync = manager.sync_once("periodic_flush")
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["extracted_files"], 1)
        self.assertTrue((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertEqual(sync["uploaded_files"], 0)
        self.assertEqual(sync["components"], {})

    def test_snapshot_rejects_path_traversal_before_writing(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME
        payload = b"escaped"
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    info = tarfile.TarInfo("../../escaped.txt")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_error")
        self.assertEqual(summary["result"], "failed")
        self.assertFalse((self.root / "escaped.txt").exists())

    def test_snapshot_rejects_path_traversal_directory_before_writing(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    info = tarfile.TarInfo("../../escaped_dir")
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_error")
        self.assertEqual(summary["result"], "failed")
        self.assertFalse((self.root / "escaped_dir").exists())

    def test_snapshot_skips_unknown_component_entries(self):
        remote = self.root / "remote"
        remote.mkdir()
        snapshot = remote / SNAPSHOT_NAME
        payload = b"legacy"
        cctx = jit_cache_module.zstd.ZstdCompressor()
        with snapshot.open("wb") as raw:
            with cctx.stream_writer(raw) as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    info = tarfile.TarInfo("legacy_component/kernel/a.so")
                    info.size = len(payload)
                    tar.addfile(info, BytesIO(payload))
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertNotIn("extracted_files", summary)
        self.assertFalse(
            (self.root / "local" / "legacy_component" / "kernel" / "a.so").exists()
        )

    def test_commit_extract_falls_back_when_move_crosses_devices(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        extract_root = self.root / "extract"
        extracted_file = extract_root / "new_component" / "kernel" / "a.so"
        extracted_file.parent.mkdir(parents=True)
        extracted_file.write_text("so", encoding="utf-8")

        def raise_exdev(src, dst):
            raise OSError(errno.EXDEV, "Invalid cross-device link")

        with mock.patch.object(
            jit_cache_module.shutil.os, "rename", side_effect=raise_exdev
        ):
            manager._commit_extract(extract_root)

        self.assertEqual(list(extract_root.iterdir()), [])
        self.assertEqual(
            (self.root / "local" / "new_component" / "kernel" / "a.so").read_text(
                encoding="utf-8"
            ),
            "so",
        )

    def test_upload_queue_reuploads_same_size_file_without_remote_stat(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        local_file = self.enqueue_file(manager, "flashinfer", "kernel/a.cubin")

        first = manager.sync_once()
        local_file.write_text("same_size!", encoding="utf-8")
        add_path_to_tracker(
            manager.dirty_tracker,
            self.component("flashinfer"),
            local_file,
            self.root / "local" / "flashinfer",
        )
        second = manager.sync_once()

        self.assertEqual(first["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(second["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(second["components"]["flashinfer"]["dirty_files"], 1)
        self.assertEqual(
            (remote / "flashinfer" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "same_size!",
        )

    def test_upload_covers_all_jit_components(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        filenames = {
            "flashinfer": "kernel/a.cubin",
            "triton": "kernel/a.so",
            "triton_autotune": "configs/a.pkl",
            "deep_gemm": "kernel/a.cubin",
            "torch_extensions": "extension/a.so",
        }

        for component, key in filenames.items():
            self.enqueue_file(manager, component, key)

        summary = manager.sync_once("periodic_flush")

        self.assertEqual(summary["uploaded_files"], len(COMPONENT_SPECS))
        for component, key in filenames.items():
            self.assertEqual(
                (remote / component / key).read_text(encoding="utf-8"), component
            )

    def test_sync_reports_jit_cache_usage_metrics_by_module(self):
        from rtp_llm.metrics import GaugeMetrics, kmonitor

        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        self.enqueue_file(manager, "triton", "kernel/a.so")

        with mock.patch.object(kmonitor, "_inited", True), mock.patch.object(
            kmonitor, "report"
        ) as report:
            summary = manager.sync_once("periodic_flush")

        local_total, local_components = jit_cache_module.component_cache_usage(
            self.root / "local"
        )
        remote_total, remote_components = jit_cache_module.component_cache_usage(remote)
        self.assertEqual(summary["local_cache"]["bytes"], local_total.bytes)
        self.assertEqual(summary["local_cache"]["files"], local_total.files)
        self.assertEqual(summary["remote_cache"]["bytes"], remote_total.bytes)
        self.assertEqual(summary["remote_cache"]["files"], remote_total.files)
        self.assertNotIn("snapshot_result", summary)
        self.assertNotIn("snapshot_message", summary)
        self.assertEqual(
            summary["local_cache"]["components"],
            {
                name: {"bytes": usage.bytes, "files": usage.files}
                for name, usage in local_components.items()
            },
        )
        self.assertEqual(
            summary["remote_cache"]["components"],
            {
                name: {"bytes": usage.bytes, "files": usage.files}
                for name, usage in remote_components.items()
            },
        )

        def expected_calls(metric, total, components):
            calls = [
                mock.call(metric, total.bytes, {"module": "total", "value": "bytes"}),
                mock.call(metric, total.files, {"module": "total", "value": "files"}),
            ]
            for module, usage in components.items():
                calls.extend(
                    [
                        mock.call(
                            metric,
                            usage.bytes,
                            {"module": module, "value": "bytes"},
                        ),
                        mock.call(
                            metric,
                            usage.files,
                            {"module": module, "value": "files"},
                        ),
                    ]
                )
            return calls

        report.assert_has_calls(
            expected_calls(
                GaugeMetrics.JIT_CACHE_LOCAL_USAGE_METRIC,
                local_total,
                local_components,
            )
            + expected_calls(
                GaugeMetrics.JIT_CACHE_REMOTE_USAGE_METRIC,
                remote_total,
                remote_components,
            ),
            any_order=True,
        )

    def test_cache_usage_ignores_jit_management_dirs(self):
        root = self.root / "local"
        cache_file = root / "triton" / "kernel" / "a.so"
        ready_file = root / ".rtp_jit_ready" / "summary.json"
        lock_file = root / ".rtp_jit_locks" / ".rtp_jit_local.lock"
        pycache_file = root / "triton" / "__pycache__" / "ignored.pyc"
        for path, text in (
            (cache_file, "cache"),
            (ready_file, "ready"),
            (lock_file, "lock"),
            (pycache_file, "pyc"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

        usage = jit_cache_module.cache_usage(root)

        self.assertEqual(usage.files, 1)
        self.assertEqual(usage.bytes, len("cache"))

    def test_upload_does_not_restat_source_after_copy(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        local_file = self.enqueue_file(manager, "triton", "kernel/a.cubin")
        original_copyfile = jit_cache_module.shutil.copyfile

        def mutating_copy(src, dst):
            result = original_copyfile(src, dst)
            local_file.write_text("new-content", encoding="utf-8")
            return result

        with mock.patch.object(
            jit_cache_module.shutil, "copyfile", side_effect=mutating_copy
        ):
            first = manager.sync_once()
        second = manager.sync_once()

        self.assertEqual(first["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(second["uploaded_files"], 0)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "triton",
        )

    def test_copy_final_preserves_existing_file_when_copy_fails(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        src = self.root / "local" / "triton" / "kernel" / "a.cubin"
        src.parent.mkdir(parents=True)
        src.write_text("new", encoding="utf-8")
        dst = remote / "triton" / "kernel" / "a.cubin"
        dst.parent.mkdir(parents=True)
        dst.write_text("old", encoding="utf-8")

        def failing_copy(src_path, dst_path):
            Path(dst_path).write_text("partial", encoding="utf-8")
            raise OSError("copy failed")

        with mock.patch.object(
            jit_cache_module.shutil, "copyfile", side_effect=failing_copy
        ):
            with self.assertRaises(OSError):
                manager.copy_final(src, dst, jit_cache_module.file_meta(src))

        self.assertEqual(dst.read_text(encoding="utf-8"), "old")
        self.assertEqual(list(dst.parent.glob(f".{dst.name}.*.tmp")), [])

    def test_watcher_enqueue_filters_events(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.layouts, manager.enqueue_upload
        )
        root = self.root / "local" / "triton"
        kernel = root / "kernel"
        kernel.mkdir(parents=True)
        valid_file = kernel / "a.cubin"
        valid_file.write_text("cubin", encoding="utf-8")

        add_path_to_tracker(tracker, self.component("triton"), valid_file, root)
        add_path_to_tracker(tracker, self.component("triton"), valid_file, root)
        for name in ("a.cubin.tmp", "a.o", "a.cu", "compile.log", "build.ninja"):
            skipped_file = kernel / name
            skipped_file.write_text("skip", encoding="utf-8")
            add_path_to_tracker(tracker, self.component("triton"), skipped_file, root)

        self.assertEqual(manager.upload_queue.qsize(), 1)
        self.assertEqual(manager.upload_queue.get_nowait().key, "kernel/a.cubin")

    def test_watcher_ignores_moved_out_paths(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.layouts, manager.enqueue_upload
        )
        handler = jit_cache_module._JitCacheEventHandler(
            tracker,
            self.component("triton"),
            self.root / "local" / "triton",
        )

        handler.on_any_event(
            FakeFileEvent(
                "moved",
                str(self.root / "local" / "triton" / "kernel" / "a.so"),
                str(self.root / "outside" / "a.so"),
            )
        )

        self.assertTrue(manager.upload_queue.empty())

    def test_watcher_only_enqueues_selected_file_events(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        tracker = jit_cache_module.JitDirtyTracker(
            manager.layouts, manager.enqueue_upload
        )
        root = self.root / "local" / "triton"
        kernel = root / "kernel"
        kernel.mkdir(parents=True)
        handler = jit_cache_module._JitCacheEventHandler(
            tracker,
            self.component("triton"),
            root,
        )

        ignored_file = kernel / "ignored.so"
        ignored_file.write_text("ignored", encoding="utf-8")
        for event_type in ("opened", "deleted", "modified"):
            handler.on_any_event(FakeFileEvent(event_type, str(ignored_file)))
        self.assertTrue(manager.upload_queue.empty())

        created_file = kernel / "created.so"
        created_file.write_text("so", encoding="utf-8")
        handler.on_any_event(FakeFileEvent("created", str(created_file)))
        self.assertEqual(manager.upload_queue.get_nowait().key, "kernel/created.so")

        closed_file = kernel / "closed.so"
        closed_file.write_text("so", encoding="utf-8")
        handler.on_any_event(FakeFileEvent("closed", str(closed_file)))
        self.assertEqual(manager.upload_queue.get_nowait().key, "kernel/closed.so")

        moved_file = kernel / "moved.so"
        moved_file.write_text("so", encoding="utf-8")
        handler.on_any_event(
            FakeFileEvent("moved", str(kernel / "tmp.so"), str(moved_file))
        )
        self.assertEqual(manager.upload_queue.get_nowait().key, "kernel/moved.so")

    def test_upload_failure_retries_then_releases_pending_candidate(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        self.enqueue_file(manager, "triton", "kernel/a.cubin")
        original_copy = manager.copy_final
        failures = 0

        def raise_copy(src, dst, src_meta):
            nonlocal failures
            failures += 1
            raise OSError("remote write failed")

        manager.copy_final = raise_copy
        with self.assertLogs(level="WARNING"):
            first = manager.sync_once()
        self.assertEqual(first["result"], "failed")
        self.assertEqual(first["components"]["triton"]["failed_files"], 1)
        self.assertEqual(failures, jit_cache_module.MAX_UPLOAD_ATTEMPTS)
        self.assertEqual(manager.upload_queue.qsize(), 0)

        manager.copy_final = original_copy
        self.enqueue_file(manager, "triton", "kernel/a.cubin")
        second = manager.sync_once()

        self.assertEqual(second["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "triton",
        )

    def test_background_worker_retries_transient_upload_failure(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.prepare()
        manager.start_background_sync()
        local_file = self.root / "local" / "triton" / "kernel" / "a.so"
        local_file.parent.mkdir(parents=True)
        local_file.write_text("so", encoding="utf-8")
        original_copy = manager.copy_final
        attempts = 0

        def flaky_copy(src, dst, src_meta):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("transient remote write failed")
            return original_copy(src, dst, src_meta)

        manager.copy_final = flaky_copy
        add_path_to_tracker(
            manager.dirty_tracker,
            self.component("triton"),
            local_file,
            self.root / "local" / "triton",
        )
        self.wait_for_queue_empty(manager)
        manager.stop()

        self.assertEqual(attempts, 2)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.so").read_text(encoding="utf-8"),
            "so",
        )

    def test_sync_drains_inline_when_workers_have_exited(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        self.enqueue_file(manager, "triton", "kernel/a.so")
        failed_worker = Future()
        failed_worker.set_exception(RuntimeError("worker stopped"))
        manager.worker_futures = [failed_worker]

        with self.assertLogs(level="WARNING"):
            summary = manager.sync_once()

        self.assertEqual(summary["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.so").read_text(encoding="utf-8"),
            "triton",
        )

    def test_start_background_sync_uploads_files_closed_after_prepare(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.start_background_sync()
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        time.sleep(0.05)
        final_file = local_kernel / "a.cubin"
        final_file.write_text("cubin", encoding="utf-8")
        handler = jit_cache_module._JitCacheEventHandler(
            manager.dirty_tracker,
            self.component("triton"),
            self.root / "local" / "triton",
        )
        handler.on_any_event(FakeFileEvent("closed", str(final_file)))
        self.wait_for_queue_empty(manager)
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )
        stored = self.read_summary()
        self.assertEqual(stored["mode"], "periodic_flush")

    def test_non_owner_takes_over_after_owner_lock_is_released(self):
        remote = self.root / "remote"
        remote.mkdir()
        first = self.make_manager(str(remote))
        self.assertTrue(first.acquire_startup_lock())
        second = self.make_manager(str(remote), timeout_s=1)
        release_owner = threading.Timer(0.05, first.stop)

        release_owner.start()
        try:
            summary = second.prepare()
        finally:
            release_owner.join()
            second.stop()
            first.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")

    def test_non_owner_reads_ready_while_owner_holds_lock(self):
        remote = self.root / "remote"
        first = self.make_manager(str(remote), timeout_s=2)
        first_summary = first.prepare()
        self.assertTrue(first.owns_startup_lock)
        second = self.make_manager(str(remote), timeout_s=2)

        start_s = time.monotonic()
        try:
            summary = second.prepare()
        finally:
            second.stop()
            first.stop()

        self.assertEqual(first_summary["cache_state"], "snapshot_miss")
        self.assertEqual(first_summary["result"], "skipped")
        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")
        self.assertLess(time.monotonic() - start_s, 0.5)

    def test_prepare_timeout_writes_summary_event(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote), timeout_s=0)

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertEqual(self.read_summary()["cache_state"], "timeout")

    def test_local_marker_skips_remote_snapshot(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("remote", encoding="utf-8")
        self.write_snapshot(remote)
        manager = self.make_manager(str(remote))
        manager.snapshot_complete_path.parent.mkdir(parents=True, exist_ok=True)
        manager.snapshot_complete_path.touch()

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "local_hit")
        self.assertEqual(summary["result"], "skipped")
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())


if __name__ == "__main__":
    unittest.main()
