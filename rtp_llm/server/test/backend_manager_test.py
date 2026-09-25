import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.server.backend_manager import BackendManager


class _FakeEngine:
    def __init__(self, exc=None):
        self.exc = exc
        self.stopped = False

    def stop(self):
        self.stopped = True
        if self.exc is not None:
            raise self.exc


class _FakeNfsManager:
    def __init__(self, exc=None):
        self.exc = exc
        self.unmounted = False

    def unmount_all(self):
        self.unmounted = True
        if self.exc is not None:
            raise self.exc


class BackendManagerStopTest(unittest.TestCase):
    def make_manager(self, engine):
        manager = BackendManager.__new__(BackendManager)
        manager.engine = engine
        manager.thread_lock_ = threading.Lock()
        manager._stopped = False
        manager._stop_error = None
        manager._shutdown_control = None
        return manager

    def test_stop_unmounts_nfs_after_engine_stop(self):
        engine = _FakeEngine()
        nfs_manager = _FakeNfsManager()
        manager = self.make_manager(engine)

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_even_when_engine_stop_raises(self):
        engine_error = RuntimeError("engine stop failed")
        engine = _FakeEngine(engine_error)
        nfs_manager = _FakeNfsManager()
        manager = self.make_manager(engine)

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def coordinated_manager(self):
        manager = self.make_manager(Mock())
        manager._shutdown_control = Mock()
        manager._shutdown_incarnation = "current-worker"
        manager._distributed_server = SimpleNamespace(store=Mock())
        manager.py_env_configs = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_rank=1, world_size=4),
            server_config=SimpleNamespace(shutdown_timeout=23),
        )
        return manager

    def test_coordination_finishes_before_stop_and_unmount(self):
        manager = self.coordinated_manager()
        nfs_manager = Mock()
        events = []
        manager.engine.stop.side_effect = lambda: events.append("engine_stop")
        nfs_manager.unmount_all.side_effect = lambda: events.append("unmount")
        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager), patch(
            "rtp_llm.utils.backend_shutdown.graceful_backend_shutdown",
            side_effect=lambda *args: events.append("coordinated"),
        ) as coordinated:
            manager.stop()
            manager.stop()
        self.assertEqual(events, ["coordinated", "engine_stop", "unmount"])
        coordinated.assert_called_once_with(
            manager._shutdown_control,
            manager._distributed_server.store,
            1,
            4,
            "current-worker",
            23,
        )
        self.assertFalse(manager.engine.started)

    def test_failed_coordination_does_not_release_resources(self):
        manager = self.coordinated_manager()
        nfs_manager = Mock()
        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager), patch(
            "rtp_llm.utils.backend_shutdown.graceful_backend_shutdown",
            side_effect=RuntimeError("missing peer"),
        ) as coordinated:
            with self.assertRaisesRegex(RuntimeError, "missing peer"):
                manager.stop()
            with self.assertRaisesRegex(RuntimeError, "previously failed"):
                manager.stop()
            coordinated.assert_called_once()
        manager.engine.stop.assert_not_called()
        nfs_manager.unmount_all.assert_not_called()
        self.assertFalse(manager._stopped)


if __name__ == "__main__":
    unittest.main()
