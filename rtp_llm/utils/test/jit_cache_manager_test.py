import io
import json
import os
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jit_cache_manager as jit_cache_module
from jit_cache_manager import (
    COMPONENTS,
    SNAPSHOT_NAME,
    JitCacheManager,
    JitSnapshotBuilder,
)


class FakeFileEvent:
    def __init__(self, event_type: str, src_path: str, dest_path: str = ""):
        self.event_type = event_type
        self.src_path = src_path
        self.dest_path = dest_path
        self.is_directory = False


class Config:
    def __init__(self, local_root: str):
        self.local_jit_cache_dir = local_root


class JitCacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old_env = os.environ.copy()
        for env_name in (
            "REMOTE_JIT_DIR",
            "JIT_PREPARE_TIMEOUT_S",
            "JIT_SYNC_INTERVAL_S",
        ):
            os.environ.pop(env_name, None)
        os.environ["LOG_PATH"] = str(self.root / "logs")
        os.environ["RTP_JIT_CACHE_RUN_ID"] = "test-run"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)
        self.tmp.cleanup()

    def make_manager(
        self, remote=None, timeout_s: float = 5, create_remote: bool = True
    ) -> JitCacheManager:
        if remote is not None:
            if remote:
                remote_path = Path(remote)
                if create_remote and remote_path.is_absolute():
                    remote_path.mkdir(parents=True, exist_ok=True)
                os.environ["REMOTE_JIT_DIR"] = remote
            else:
                os.environ.pop("REMOTE_JIT_DIR", None)
        os.environ["JIT_PREPARE_TIMEOUT_S"] = str(timeout_s)
        os.environ["JIT_SYNC_INTERVAL_S"] = "0"
        manager = JitCacheManager(Config(str(self.root / "local")))
        manager.bootstrap_env()
        return manager

    def read_summary(self):
        return json.loads(
            (self.root / "logs" / "jit_cache" / "summary.json").read_text(
                encoding="utf-8"
            )
        )

    def wait_for_candidates(
        self, manager: JitCacheManager, component: str, count: int = 1
    ):
        deadline = time.time() + 5
        while time.time() < deadline:
            tracker = manager.dirty_tracker
            if tracker is not None:
                with tracker.lock:
                    if len(tracker.dirty.get(component, set())) >= count:
                        return
            time.sleep(0.01)
        self.fail(f"timed out waiting for {count} {component} JIT cache candidates")

    def test_bootstrap_sets_local_env_and_hides_remote_env(self):
        manager = self.make_manager(str(self.root / "remote"))

        self.assertTrue(manager.enabled)
        self.assertEqual(os.environ["REMOTE_JIT_DIR"], "")
        self.assertEqual(os.environ["LOCAL_JIT_CACHE_DIR"], str(self.root / "local"))
        self.assertEqual(set(manager.layouts), set(COMPONENTS))
        self.assertEqual(
            os.environ["FLASHINFER_WORKSPACE_BASE"],
            str(self.root / "local" / "flashinfer"),
        )
        self.assertEqual(
            os.environ["TRITON_CACHE_DIR"], str(self.root / "local" / "triton")
        )
        self.assertEqual(
            os.environ["DG_JIT_CACHE_DIR"], str(self.root / "local" / "deep_gemm")
        )

    def test_prepare_without_remote_returns_disabled_summary(self):
        manager = self.make_manager()

        summary = manager.prepare()

        self.assertFalse(manager.enabled)
        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "disabled")
        self.assertNotIn("components", summary)

    def test_remote_and_timeouts_can_read_config(self):
        config = Config(str(self.root / "local"))
        remote = self.root / "remote_from_config"
        remote.mkdir()
        config.remote_jit_dir = str(remote)
        config.jit_prepare_timeout_s = 123
        config.jit_sync_interval_s = 456

        manager = JitCacheManager(config)

        self.assertTrue(manager.enabled)
        self.assertEqual(manager.remote_root, remote)
        self.assertEqual(manager.prepare_timeout_s, 123)
        self.assertEqual(manager.sync_interval_s, 456)

    def test_invalid_remote_paths_disable_sync(self):
        missing_remote = self.root / "missing_remote"
        cases = (
            ("oss://bucket/jit_cache", "absolute mounted path", True),
            ("relative/jit_cache", "absolute mounted path", True),
            (str(missing_remote), "existing mounted directory", False),
        )

        for remote, message, create_remote in cases:
            with self.subTest(remote=remote):
                manager = self.make_manager(remote, create_remote=create_remote)

                with mock.patch.object(jit_cache_module.logging, "exception") as log:
                    summary = manager.prepare()

                self.assertFalse(manager.enabled)
                self.assertEqual(os.environ["REMOTE_JIT_DIR"], "")
                self.assertEqual(summary["result"], "invalid_config")
                self.assertIn(message, summary["message"])
                log.assert_not_called()
        self.assertFalse((Path.cwd() / "relative" / "jit_cache").exists())
        self.assertFalse(missing_remote.exists())

    def test_prepare_snapshot_miss_does_not_scan_remote_files(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        (remote_kernel / "a.cu").write_text("cu", encoding="utf-8")
        manager = self.make_manager(str(remote))

        with mock.patch.object(jit_cache_module.logging, "exception") as log:
            summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["result"], "skipped")
        self.assertNotIn("components", summary)
        log.assert_not_called()
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.cu").exists())
        stored = self.read_summary()
        self.assertIn("snapshot_download", stored["events"])
        self.assertTrue(
            (self.root / "local" / ".rtp_jit_ready" / "test-run.json").exists()
        )

    def test_upload_writes_and_skips_unchanged_files(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.dirty_tracker = jit_cache_module.JitDirtyTracker(manager.layouts)
        local_kernel = self.root / "local" / "flashinfer" / "kernel"
        local_kernel.mkdir(parents=True)
        local_file = local_kernel / "a.cubin"
        local_file.write_text("cubin", encoding="utf-8")
        (local_kernel / "build.ninja").write_text("skip", encoding="utf-8")
        assert manager.dirty_tracker is not None
        manager.dirty_tracker.add_candidate(
            "flashinfer", local_file, self.root / "local" / "flashinfer"
        )

        self.wait_for_candidates(manager, "flashinfer")
        first = manager.sync_once()
        remote_file = remote / "flashinfer" / "kernel" / "a.cubin"
        manager.dirty_tracker.add_candidate(
            "flashinfer", local_file, self.root / "local" / "flashinfer"
        )
        second = manager.sync_once()
        another_file = local_kernel / "b.cubin"
        another_file.write_text("new-cubin", encoding="utf-8")
        manager.dirty_tracker.add_candidate(
            "flashinfer", another_file, self.root / "local" / "flashinfer"
        )
        self.wait_for_candidates(manager, "flashinfer")
        third = manager.sync_once()
        manager.stop_dirty_tracker()

        self.assertEqual(first["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(second["uploaded_files"], 0)
        self.assertEqual(third["components"]["flashinfer"]["uploaded_files"], 1)
        self.assertEqual(remote_file.read_text(encoding="utf-8"), "cubin")
        self.assertEqual(
            (remote / "flashinfer" / "kernel" / "b.cubin").read_text(encoding="utf-8"),
            "new-cubin",
        )
        self.assertFalse((remote / "flashinfer" / "kernel" / "build.ninja").exists())
        self.assertFalse((remote / "flashinfer" / "summary.json").exists())

    def test_upload_requeues_if_source_changes_during_copy(self):
        remote = self.root / "remote"
        writer = self.make_manager(str(remote))
        writer.dirty_tracker = jit_cache_module.JitDirtyTracker(writer.layouts)
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        local_file = local_kernel / "a.cubin"
        local_file.write_text("old", encoding="utf-8")
        writer.dirty_tracker.add_candidate(
            "triton", local_file, self.root / "local" / "triton"
        )
        original_copy2 = jit_cache_module.shutil.copy2

        def mutating_copy(src, dst):
            result = original_copy2(src, dst)
            local_file.write_text("new", encoding="utf-8")
            return result

        with self.assertLogs(level="WARNING"), mock.patch.object(
            jit_cache_module.shutil, "copy2", side_effect=mutating_copy
        ):
            first = writer.sync_once()
        with writer.dirty_tracker.lock:
            self.assertIn("kernel/a.cubin", writer.dirty_tracker.dirty["triton"])

        second = writer.sync_once()

        self.assertEqual(first["result"], "failed")
        self.assertEqual(first["components"]["triton"]["failed_files"], 1)
        self.assertEqual(second["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "new",
        )

    def test_upload_tracks_renamed_final_files(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.start_dirty_tracker()
        local_kernel = self.root / "local" / "deep_gemm" / "kernel"
        local_kernel.mkdir(parents=True)
        tmp_file = local_kernel / "a.cubin.tmp"
        final_file = local_kernel / "a.cubin"
        tmp_file.write_text("cubin", encoding="utf-8")
        os.replace(tmp_file, final_file)

        self.wait_for_candidates(manager, "deep_gemm")
        summary = manager.sync_once()
        manager.stop_dirty_tracker()

        self.assertEqual(summary["components"]["deep_gemm"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "deep_gemm" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )

    def test_upload_tracks_close_write_final_files(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        manager.start_dirty_tracker()
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        final_file = local_kernel / "a.cubin"
        final_file.write_text("cubin", encoding="utf-8")
        assert manager.dirty_tracker is not None

        handler = jit_cache_module._JitCacheEventHandler(
            manager.dirty_tracker, "triton", self.root / "local" / "triton"
        )
        handler.dispatch(FakeFileEvent("closed", str(final_file)))
        summary = manager.sync_once()
        manager.stop_dirty_tracker()

        self.assertEqual(summary["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )

    def test_upload_failure_requeues_dirty_candidate(self):
        remote = self.root / "remote"
        writer = self.make_manager(str(remote))
        writer.start_dirty_tracker()
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        (local_kernel / "a.cubin").write_text("cubin", encoding="utf-8")
        self.wait_for_candidates(writer, "triton")
        original_copy = writer.copy_final

        def raise_copy(src, dst, src_meta):
            raise OSError("remote write failed")

        writer.copy_final = raise_copy
        with self.assertLogs(level="WARNING"):
            first = writer.sync_once()
        with writer.dirty_tracker.lock:
            self.assertIn("kernel/a.cubin", writer.dirty_tracker.dirty["triton"])

        writer.copy_final = original_copy
        second = writer.sync_once()
        writer.stop_dirty_tracker()

        self.assertEqual(first["result"], "failed")
        self.assertEqual(first["components"]["triton"]["failed_files"], 1)
        self.assertEqual(second["components"]["triton"]["uploaded_files"], 1)
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )

    def test_shutdown_sync_uploads_files_closed_after_prepare(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        summary = manager.prepare()
        manager.start_background_sync()
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        time.sleep(0.05)
        (local_kernel / "a.cubin").write_text("cubin", encoding="utf-8")
        (local_kernel / "a.ptx").write_text("skip", encoding="utf-8")
        self.wait_for_candidates(manager, "triton")
        manager.stop()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(
            (remote / "triton" / "kernel" / "a.cubin").read_text(encoding="utf-8"),
            "cubin",
        )
        self.assertFalse((remote / "triton" / "kernel" / "a.ptx").exists())
        stored = self.read_summary()
        self.assertIn("shutdown_flush", stored["events"])
        self.assertGreater(
            stored["events"]["shutdown_flush"]["components"]["triton"][
                "uploaded_files"
            ],
            0,
        )

    def test_non_owner_waits_ready_marker_not_stale_summary(self):
        remote = self.root / "remote"
        first = self.make_manager(str(remote))
        self.assertTrue(first.acquire_lock())
        summary_dir = self.root / "logs" / "jit_cache"
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps(
                {
                    "events": {
                        "download": {
                            "cache_state": "remote_hit",
                            "result": "success",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

        second = self.make_manager(str(remote), timeout_s=0.1)
        summary = second.prepare()

        self.assertEqual(summary["result"], "timeout")
        first.stop()

    def test_non_owner_takes_over_after_owner_lock_is_released(self):
        remote = self.root / "remote"
        remote.mkdir()
        first = self.make_manager(str(remote))
        self.assertTrue(first.acquire_lock())
        second = self.make_manager(str(remote), timeout_s=1)
        release_owner = threading.Timer(0.05, first.stop)

        release_owner.start()
        try:
            summary = second.prepare()
        finally:
            release_owner.join()
            second.stop()
            first.stop()

        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")

    def test_prepare_timeout_writes_summary_event(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        JitSnapshotBuilder(str(remote)).build()
        manager = self.make_manager(str(remote), timeout_s=0)

        with mock.patch.object(jit_cache_module.logging, "exception") as log:
            summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertNotIn("components", summary)
        log.assert_not_called()
        stored = self.read_summary()
        self.assertEqual(
            stored["events"]["snapshot_download"]["cache_state"], "timeout"
        )

    def test_prepare_timeout_does_not_use_background_extract(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        JitSnapshotBuilder(str(remote)).build()

        manager = self.make_manager(str(remote), timeout_s=0.02)
        original_extract = manager.extract_snapshot

        def slow_extract(archive, dst_root, deadline=None):
            time.sleep(0.05)
            return original_extract(archive, dst_root, deadline)

        manager.extract_snapshot = slow_extract

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())

    def test_prepare_timeout_after_snapshot_copy_is_bounded(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        JitSnapshotBuilder(str(remote)).build()

        manager = self.make_manager(str(remote), timeout_s=0.01)
        original_extract = manager.extract_snapshot

        def slow_extract(archive, dst_root, deadline=None):
            time.sleep(0.05)
            return original_extract(archive, dst_root, deadline)

        manager.extract_snapshot = slow_extract

        summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        self.assertIn("snapshot extraction exceeded", summary["message"])

    def test_prepare_timeout_skips_extract_when_snapshot_copy_exceeds_deadline(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        JitSnapshotBuilder(str(remote)).build()

        manager = self.make_manager(str(remote), timeout_s=0.01)
        original_copy2 = jit_cache_module.shutil.copy2

        def slow_copy(src, dst):
            time.sleep(0.05)
            return original_copy2(src, dst)

        with mock.patch.object(
            jit_cache_module.shutil, "copy2", side_effect=slow_copy
        ), mock.patch.object(
            manager, "extract_snapshot", wraps=manager.extract_snapshot
        ) as extract:
            summary = manager.prepare()
            manager.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")
        extract.assert_not_called()
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.so").exists())

    def test_non_owner_waits_ready_marker_within_timeout(self):
        remote = self.root / "remote"
        owner = self.make_manager(str(remote))
        self.assertTrue(owner.acquire_lock())
        second = self.make_manager(str(remote), timeout_s=1)
        ready_summary = {
            "mode": "snapshot_download",
            "cache_state": "snapshot_hit",
            "result": "success",
            "total_cost_ms": 50,
        }
        ready = threading.Timer(0.05, lambda: owner.write_ready(ready_summary))

        ready.start()
        try:
            summary = second.prepare()
        finally:
            ready.join()
            second.stop()
            owner.stop()

        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")

    def test_non_owner_times_out_without_ready_marker(self):
        remote = self.root / "remote"
        owner = self.make_manager(str(remote))
        self.assertTrue(owner.acquire_lock())
        second = self.make_manager(str(remote), timeout_s=0.01)

        try:
            summary = second.prepare()
        finally:
            second.stop()
            owner.stop()

        self.assertEqual(summary["cache_state"], "timeout")
        self.assertEqual(summary["result"], "timeout")

    def test_stop_waits_for_worker_before_releasing_lock(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        self.assertTrue(manager.acquire_lock())

        class SlowWorker:
            def __init__(self):
                self.join_timeout = "not-called"
                self.alive = True

            def is_alive(self):
                return self.alive

            def join(self, timeout=None):
                self.join_timeout = timeout
                self.alive = False

        worker = SlowWorker()
        manager.worker = worker

        manager.stop()

        self.assertIsNone(worker.join_timeout)
        self.assertFalse(manager.owns_lock)

    def test_watcher_start_failure_does_not_block_prepare(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))

        with mock.patch.object(
            jit_cache_module.JitDirtyTracker, "start", return_value=False
        ):
            summary = manager.prepare()
        manager.stop()

        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")
        self.assertIsNone(manager.dirty_tracker)

    def test_local_files_without_snapshot_complete_marker_do_not_skip_remote(self):
        remote = self.root / "remote"
        manager = self.make_manager(str(remote))
        local_kernel = self.root / "local" / "triton" / "kernel"
        local_kernel.mkdir(parents=True)
        (local_kernel / "a.so").write_text("partial", encoding="utf-8")

        summary = manager.bootstrap_from_remote()

        self.assertEqual(summary["cache_state"], "snapshot_miss")
        self.assertEqual(summary["result"], "skipped")

    def test_prepare_uses_snapshot_without_remote_file_scan(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.so").write_text("so", encoding="utf-8")
        (remote_kernel / "a.ptx").write_text("skip", encoding="utf-8")
        build = JitSnapshotBuilder(str(remote)).build()
        self.assertEqual(build["result"], "success")

        manager = self.make_manager(str(remote))
        summary = manager.bootstrap_from_remote()

        self.assertEqual(summary["mode"], "snapshot_download")
        self.assertEqual(summary["cache_state"], "snapshot_hit")
        self.assertEqual(summary["result"], "success")
        self.assertEqual(summary["extracted_files"], 1)
        self.assertTrue((self.root / "local" / "triton" / "kernel" / "a.so").exists())
        self.assertFalse((self.root / "local" / "triton" / "kernel" / "a.ptx").exists())
        self.assertTrue(manager.snapshot_complete_path.exists())
        sync = manager.sync_once("periodic_flush")
        self.assertEqual(sync["dirty_files"], 0)
        self.assertEqual(sync["uploaded_files"], 0)
        self.assertEqual(sync["components"], {})
        stored = self.read_summary()
        self.assertIn("snapshot_download", stored["events"])

    def test_snapshot_preserves_mtime_and_skips_blind_reupload(self):
        remote = self.root / "remote"
        remote_kernel = remote / "triton" / "kernel"
        remote_kernel.mkdir(parents=True)
        remote_file = remote_kernel / "a.so"
        remote_file.write_text("so", encoding="utf-8")
        mtime_ns = 1_700_000_000_123_456_789
        os.utime(remote_file, ns=(mtime_ns, mtime_ns))
        JitSnapshotBuilder(str(remote)).build()

        manager = self.make_manager(str(remote))
        summary = manager.bootstrap_from_remote()
        local_file = self.root / "local" / "triton" / "kernel" / "a.so"
        manager.start_dirty_tracker()
        assert manager.dirty_tracker is not None
        with manager.dirty_tracker.lock:
            manager.dirty_tracker.dirty["triton"].add("kernel/a.so")
        sync = manager.sync_once("periodic_flush")
        manager.stop()

        self.assertEqual(summary["result"], "success")
        self.assertEqual(
            jit_cache_module.file_meta(local_file),
            jit_cache_module.file_meta(remote_file),
        )
        self.assertEqual(sync["dirty_files"], 1)
        self.assertEqual(sync["uploaded_files"], 0)

    def test_extract_rejects_unsafe_snapshot_paths(self):
        manager = self.make_manager()
        manager.local_root.mkdir(parents=True, exist_ok=True)

        for name in ("../escape.so", "/escape.so", "triton/../escape.so"):
            with self.subTest(name=name):
                archive = self.root / f"malicious_{abs(hash(name))}.tar.zst"
                raw_tar = io.BytesIO()
                with tarfile.open(fileobj=raw_tar, mode="w") as tar:
                    info = tarfile.TarInfo(name=name)
                    info.size = 4
                    tar.addfile(info, io.BytesIO(b"test"))
                archive.write_bytes(
                    jit_cache_module.zstd.ZstdCompressor().compress(raw_tar.getvalue())
                )

                with self.assertRaises(RuntimeError):
                    manager.extract_snapshot(archive, manager.local_root, None)

    def test_snapshot_builder_excludes_unstable_files_and_preserves_old_on_failure(
        self,
    ):
        remote = self.root / "remote"
        remote_kernel = remote / "deep_gemm" / "kernel"
        remote_kernel.mkdir(parents=True)
        (remote_kernel / "a.cubin").write_text("cubin", encoding="utf-8")
        (remote_kernel / "a.tmp").write_text("tmp", encoding="utf-8")
        first = JitSnapshotBuilder(str(remote)).build()

        self.assertEqual(first["result"], "success")
        snapshot_path = remote / SNAPSHOT_NAME
        self.assertEqual(first["snapshot_path"], str(snapshot_path))
        snapshot_bytes = snapshot_path.read_bytes()

        builder = JitSnapshotBuilder(str(remote))
        builder.write_snapshot = lambda tmp, included: (_ for _ in ()).throw(
            RuntimeError("boom")
        )
        with self.assertRaises(RuntimeError):
            builder.build()

        self.assertEqual(snapshot_path.read_bytes(), snapshot_bytes)
        self.assertFalse(list(remote.glob(f"{SNAPSHOT_NAME}.*.tmp")))


if __name__ == "__main__":
    unittest.main()
