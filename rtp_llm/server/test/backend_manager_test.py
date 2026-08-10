import unittest
from unittest.mock import patch

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
    def test_stop_unmounts_nfs_after_engine_stop(self):
        manager = BackendManager.__new__(BackendManager)
        engine = _FakeEngine()
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_even_when_engine_stop_raises(self):
        manager = BackendManager.__new__(BackendManager)
        engine_error = RuntimeError("engine stop failed")
        engine = _FakeEngine(engine_error)
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)


if __name__ == "__main__":
    unittest.main()
