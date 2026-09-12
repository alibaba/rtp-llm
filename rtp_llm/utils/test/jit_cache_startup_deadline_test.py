"""Actual flock contention and late restores must not corrupt a running cache."""

import contextlib
import multiprocessing
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm import start_backend_server as backend
from rtp_llm.utils import jit_cache_manager as jit
from rtp_llm.utils import jit_cache_store as store


def hold_restore_lock(target, entered, release):
    with store.restore_lock(Path(target)):
        entered.set()
        if not release.wait(10):
            raise TimeoutError("test did not release its live flock owner")


class StartupDeadlineTest(unittest.TestCase):
    def test_live_lock_owner_is_inside_startup_timeout(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "local"
            ctx = multiprocessing.get_context("spawn")
            entered, release = ctx.Event(), ctx.Event()
            holder = ctx.Process(
                target=hold_restore_lock, args=(str(target), entered, release)
            )
            holder.start()
            finished, ready = threading.Event(), threading.Event()
            results = []
            manager = mock.Mock()

            def setup():
                results.append(backend._setup_jit_cache("/remote", 0, ready))
                finished.set()

            worker = None
            try:
                self.assertTrue(entered.wait(10))
                with mock.patch.object(
                    backend, "JIT_CACHE_SETUP_TIMEOUT_S", 0.05
                ), mock.patch.object(
                    jit, "setup_jit_cache_env", return_value=((), True)
                ), mock.patch.object(
                    jit, "LOCAL_JIT_DIR", str(target)
                ), mock.patch.object(
                    jit, "resolve_remote_root", return_value=Path(temporary) / "remote"
                ) as resolve, mock.patch.object(
                    jit, "JitCacheManager", return_value=manager
                ):
                    worker = threading.Thread(target=setup, daemon=True)
                    started = time.monotonic()
                    worker.start()
                    returned_before_release = finished.wait(0.5)
                    elapsed = time.monotonic() - started
                    ready_before_release = ready.is_set()
                    release.set()
                    holder.join(10)
                    worker.join(5)
                    # Give a timed-out setup worker a chance to acquire the now
                    # free real lock. It must observe cancellation before I/O.
                    with store.restore_lock(target):
                        pass
                    self.assertTrue(
                        returned_before_release,
                        "rank0 blocked behind a live restore owner",
                    )
                    self.assertTrue(ready_before_release)
                    self.assertLess(elapsed, 0.5)
                    self.assertEqual(results, [None])
                    resolve.assert_not_called()
                    manager.start_background_sync.assert_not_called()
                    self.assertFalse(target.with_name("local.ready").exists())
                self.assertEqual(holder.exitcode, 0)
            finally:
                release.set()
                holder.join(10)
                if holder.is_alive():
                    holder.terminate()
                    holder.join(5)
                if worker is not None:
                    worker.join(5)

    def test_cancel_during_warm_scan_does_not_claim_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, remote = root / "local", root / "remote"
            target.mkdir()
            remote.mkdir()
            (target / "live.cubin").write_bytes(b"locally compiled")
            cancel = threading.Event()

            def scan(_):
                cancel.set()
                return True

            with mock.patch.object(store, "_tree_is_warm", side_effect=scan):
                self.assertFalse(
                    store.RemoteSnapshotStore(remote).restore(
                        target, cancel, threading.Lock()
                    )
                )
            self.assertFalse(target.with_name("local.ready").exists())
            self.assertEqual((target / "live.cubin").read_bytes(), b"locally compiled")

    def test_new_local_artifact_survives_late_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target, remote, source = root / "local", root / "remote", root / "source"
            target.mkdir()
            remote.mkdir()
            source.mkdir()
            (source / "old.cubin").write_bytes(b"remote generation")
            archive = remote / ("old" + store.SNAPSHOT_SUFFIX)
            store.pack_zstd_tar(archive, source)
            original = store.extract_zstd_tar

            def extract(archive, staging):
                original(archive, staging)
                (target / "live.cubin").write_bytes(b"locally compiled")

            with mock.patch.object(store, "extract_zstd_tar", side_effect=extract):
                self.assertFalse(store.RemoteSnapshotStore(remote).restore(target))
            self.assertEqual((target / "live.cubin").read_bytes(), b"locally compiled")
            self.assertFalse((target / "old.cubin").exists())
            self.assertTrue(target.with_name("local.ready").exists())
            self.assertFalse(list(root.glob("local.stage.*")))


if __name__ == "__main__":
    unittest.main()
