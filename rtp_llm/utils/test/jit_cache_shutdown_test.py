import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.utils import jit_cache_manager as jit


class JitCacheShutdownTest(unittest.TestCase):
    def make_manager(self):
        manager = jit.JitCacheManager(mock.Mock(), "")
        manager.store = mock.Mock()
        return manager

    def test_stop_before_sync_does_not_publish_dirty_cache(self):
        manager = self.make_manager()
        manager._dirty.set()
        manager._stop.set()
        with mock.patch.object(manager, "publish_pending_snapshot") as publish:
            manager._sync_loop()
        publish.assert_not_called()
        manager.store.publish_snapshot.assert_not_called()

    def test_stop_wakes_idle_sync_without_final_publish(self):
        manager = self.make_manager()
        waiting = threading.Event()
        wait = manager._stop.wait

        def wait_until_stop(timeout):
            waiting.set()
            return wait(timeout)

        with mock.patch.object(
            manager._stop, "wait", side_effect=wait_until_stop
        ), mock.patch.object(manager, "publish_pending_snapshot") as publish:
            manager._worker = threading.Thread(target=manager._sync_loop, daemon=True)
            manager._worker.start()
            try:
                self.assertTrue(waiting.wait(1))
                manager.stop()
                self.assertFalse(manager._worker.is_alive())
                publish.assert_not_called()
                manager.store.close.assert_called_once_with()
            finally:
                manager._stop.set()
                manager._worker.join(1)

    def test_periodic_publish_continues_after_failure(self):
        manager = self.make_manager()
        with mock.patch.object(
            manager._stop, "wait", side_effect=[False, False, True]
        ), mock.patch.object(
            manager, "publish_pending_snapshot", side_effect=[OSError("busy"), None]
        ) as publish, self.assertLogs(
            level="ERROR"
        ):
            manager._sync_loop()
        self.assertEqual(publish.call_count, 2)

    def test_cleanup_leaves_store_open_while_any_worker_is_alive(self):
        for name in ("_observer", "_worker", "_prepare_thread"):
            with self.subTest(worker=name):
                manager = self.make_manager()
                workers = dict.fromkeys(("_observer", "_worker", "_prepare_thread"))
                worker = workers[name] = mock.Mock()
                worker.is_alive.return_value = True
                manager._observer = workers["_observer"]
                manager._worker = workers["_worker"]
                manager._prepare_thread = workers["_prepare_thread"]
                with self.assertLogs(level="WARNING") as logs, mock.patch.object(
                    jit, "CLEANUP_GRACE_S", 0.0
                ):
                    manager._cleanup(time.monotonic() - 1)
                self.assertEqual(worker.join.call_args_list, [mock.call(0.0)] * 2)
                manager.store.close.assert_not_called()
                self.assertTrue(
                    any("leaving store open" in line for line in logs.output)
                )

    def test_single_stop_closes_store_after_a_slow_publish(self):
        manager = self.make_manager()
        publishing, release, cleaned = (threading.Event() for _ in range(3))
        cleanup = manager._cleanup

        def publish():
            publishing.set()
            release.wait(5)

        def cleanup_and_signal(deadline):
            try:
                cleanup(deadline)
            finally:
                cleaned.set()

        with mock.patch.object(
            manager, "publish_pending_snapshot", side_effect=publish
        ) as publish_mock, mock.patch.object(
            manager, "_cleanup", side_effect=cleanup_and_signal
        ), mock.patch.object(
            jit, "SYNC_POLL_S", 0.001
        ), mock.patch.object(
            jit, "STOP_TIMEOUT_S", 0.05
        ):
            manager._worker = threading.Thread(target=manager._sync_loop, daemon=True)
            manager._worker.start()
            try:
                self.assertTrue(publishing.wait(1))
                started = time.monotonic()
                with self.assertLogs(level="WARNING") as logs:
                    manager.stop()
                    self.assertLess(time.monotonic() - started, 1.0)

                    def waiting():
                        return any(
                            "closing store once it exits" in line
                            for line in logs.output
                        )

                    while not waiting() and time.monotonic() - started < 2.0:
                        time.sleep(0.01)
                    self.assertTrue(waiting())
                self.assertTrue(manager._worker.is_alive())
                self.assertFalse(cleaned.is_set())
                manager.store.close.assert_not_called()
                release.set()  # no second stop(): background cleanup finishes the job
                self.assertTrue(cleaned.wait(5))
                self.assertFalse(manager._worker.is_alive())
                manager.store.close.assert_called_once_with()
                manager.stop()  # still harmless
                manager.store.close.assert_called_with()
                publish_mock.assert_called_once_with()
            finally:
                release.set()
                manager._stop.set()
                manager._worker.join(1)

    def test_stop_logs_dropped_dirty_snapshot(self):
        manager = self.make_manager()
        manager._dirty.set()
        manager._stop.set()
        with self.assertLogs(jit.logger, level="INFO") as logs:
            manager._sync_loop()
        self.assertTrue(
            any("pending snapshot is not published" in line for line in logs.output)
        )

    def test_cleanup_without_store_or_workers(self):
        manager = jit.JitCacheManager(mock.Mock(), "")
        manager.stop()
        manager.stop()

    def test_concurrent_stop_shares_cleanup_and_closes_once(self):
        manager = self.make_manager()
        closing, release = threading.Event(), threading.Event()

        def close():
            closing.set()
            release.wait(5)

        manager.store.close.side_effect = close
        with mock.patch.object(jit, "STOP_TIMEOUT_S", 0.05):
            first = threading.Thread(target=manager.stop)
            try:
                with self.assertLogs(jit.logger, level="WARNING"):
                    first.start()
                    self.assertTrue(closing.wait(1))
                    cleanup = manager._cleanup_thread
                    started = time.monotonic()
                    manager.stop()
                    self.assertLess(time.monotonic() - started, 1.0)
                    self.assertIs(manager._cleanup_thread, cleanup)
                    first.join(1)
                    self.assertFalse(first.is_alive())
                manager.store.close.assert_called_once_with()
            finally:
                release.set()
                first.join(1)
                manager._cleanup_thread.join(1)
        self.assertFalse(manager._cleanup_thread.is_alive())
        manager.store.close.assert_called_once_with()

    def test_repeated_stop_unmounts_only_once(self):
        manager = self.make_manager()
        manager.store = jit.store.RemoteSnapshotStore(Path("/unused"), "/mnt/fake")
        unmount = mock.Mock()
        with mock.patch.dict(
            sys.modules,
            {"rtp_llm.utils.fuser": types.SimpleNamespace(umount_file=unmount)},
        ):
            manager.stop()
            manager.stop()
        unmount.assert_called_once_with("/mnt/fake")

    def test_later_stop_retries_failed_unmount(self):
        manager = self.make_manager()
        manager.store = jit.store.RemoteSnapshotStore(Path("/unused"), "/mnt/fake")
        unmount = mock.Mock(side_effect=[OSError("busy"), None])
        with mock.patch.dict(
            sys.modules,
            {"rtp_llm.utils.fuser": types.SimpleNamespace(umount_file=unmount)},
        ):
            with self.assertLogs(level="WARNING"):
                manager.stop()
            manager.stop()
        self.assertEqual(unmount.call_args_list, [mock.call("/mnt/fake")] * 2)


if __name__ == "__main__":
    unittest.main()
