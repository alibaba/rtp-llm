import threading
import unittest
from unittest.mock import patch

from rtp_llm.distribute.distributed_server import BackendStopConsensusError
from rtp_llm.server.backend_manager import BackendManager


class _FakeEngine:
    def __init__(self, exc=None):
        self.exc = exc
        self.prepared = False
        self.coordinated = None
        self.target_step = None
        self.stopped = False
        self.armed_target = None
        self.armed_stop_cancelled = False

    def prepare_stop(self, coordinated=True, target_step=-1):
        self.prepared = True
        self.coordinated = coordinated
        self.target_step = target_step

    def completed_steps(self):
        return 64

    def arm_stop(self, target_step):
        self.armed_target = target_step

    def cancel_armed_stop(self):
        self.armed_stop_cancelled = True

    def stop(self):
        self.stopped = True
        if self.exc is not None:
            raise self.exc

    def onflight_request_num(self):
        return 0


class _FakeDistributedServer:
    def __init__(self):
        self.waited = []
        self.shutdown_requested = False

    def wait_for_backend_shutdown(self, timeout, phase):
        self.waited.append(phase)

    def request_backend_shutdown(self):
        self.shutdown_requested = True

    def is_backend_shutdown_requested(self):
        return self.shutdown_requested

    def choose_backend_stop_step(
        self, timeout, local_step, arm_stop, cancel_armed_stop
    ):
        arm_stop(local_step + 256)
        return local_step + 256


class _FakeServerConfig:
    shutdown_timeout = 1


class _FakePyEnvConfigs:
    server_config = _FakeServerConfig()

    class parallelism_config:
        world_size = 2


class _FakeNfsManager:
    def __init__(self, exc=None):
        self.exc = exc
        self.unmounted = False

    def unmount_all(self):
        self.unmounted = True
        if self.exc is not None:
            raise self.exc


class BackendManagerStopTest(unittest.TestCase):
    def _manager(self, engine):
        manager = BackendManager.__new__(BackendManager)
        manager.engine = engine
        manager._stopped = threading.Event()
        manager._distributed_server = _FakeDistributedServer()
        manager.py_env_configs = _FakePyEnvConfigs()
        return manager

    def test_stop_unmounts_nfs_after_engine_stop(self):
        engine = _FakeEngine()
        manager = self._manager(engine)
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(engine.prepared)
        self.assertTrue(nfs_manager.unmounted)
        self.assertEqual(
            manager._distributed_server.waited, ["drained", "engine_stopped"]
        )

    def test_stop_without_global_controller_still_stops_engine(self):
        engine = _FakeEngine()
        manager = self._manager(engine)
        manager._global_controller = None
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)
        self.assertEqual(
            manager._distributed_server.waited, ["drained", "engine_stopped"]
        )

    def test_rendezvous_failure_still_stops_engine(self):
        engine = _FakeEngine()
        manager = self._manager(engine)

        def fail_rendezvous(timeout, phase):
            raise TimeoutError("rendezvous failed")

        manager._distributed_server.wait_for_backend_shutdown = fail_rendezvous
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_rendezvous_store_failure_still_stops_engine(self):
        engine = _FakeEngine()
        manager = self._manager(engine)

        def fail_rendezvous(timeout, phase):
            raise RuntimeError("store unavailable")

        manager._distributed_server.wait_for_backend_shutdown = fail_rendezvous
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_indeterminate_stop_consensus_parks_without_cleanup(self):
        engine = _FakeEngine()
        manager = self._manager(engine)
        manager._choose_backend_stop_step = lambda _engine: (_ for _ in ()).throw(
            BackendStopConsensusError("indeterminate")
        )
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ), patch(
            "rtp_llm.server.backend_manager.time.sleep",
            side_effect=SystemExit("parked"),
        ):
            with self.assertRaisesRegex(SystemExit, "parked"):
                manager.stop()

        self.assertFalse(engine.prepared)
        self.assertFalse(engine.stopped)
        self.assertFalse(nfs_manager.unmounted)

    def test_drain_failure_is_raised_after_engine_cleanup(self):
        engine = _FakeEngine()
        engine.onflight_request_num = lambda: (_ for _ in ()).throw(
            RuntimeError("drain failed")
        )
        manager = self._manager(engine)
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            with self.assertRaisesRegex(RuntimeError, "drain failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_drain_timeout_is_raised_after_engine_cleanup(self):
        engine = _FakeEngine()
        engine.onflight_request_num = lambda: (_ for _ in ()).throw(
            TimeoutError("native drain timed out")
        )
        manager = self._manager(engine)
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            with self.assertRaisesRegex(TimeoutError, "native drain timed out"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_prepare_stop_failure_does_not_publish_engine_stopped(self):
        engine = _FakeEngine()

        def fail_prepare(**kwargs):
            raise RuntimeError("prepare failed")

        engine.prepare_stop = fail_prepare
        manager = self._manager(engine)
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            with self.assertRaisesRegex(RuntimeError, "prepare failed"):
                manager.stop()

        self.assertEqual(manager._distributed_server.waited, ["drained"])
        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_even_when_engine_stop_raises(self):
        engine_error = RuntimeError("engine stop failed")
        engine = _FakeEngine(engine_error)
        manager = self._manager(engine)
        nfs_manager = _FakeNfsManager()

        with patch("rtp_llm.server.backend_manager._nfs_manager", nfs_manager), patch(
            "rtp_llm.server.backend_manager.BaseEngine", _FakeEngine
        ):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)


if __name__ == "__main__":
    unittest.main()
