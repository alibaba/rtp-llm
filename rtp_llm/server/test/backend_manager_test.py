import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.distribute.distributed_server import BackendStopConsensusError
from rtp_llm.ops import TaskType
from rtp_llm.server.backend_manager import BackendManager


class _FakeEngine:
    def __init__(self, exc=None):
        self.exc = exc
        self.prepared = False
        self.coordinated = None
        self.target_step = None
        self.stopped = False
        self.serving_states = []
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

    def set_serving(self, serving):
        self.serving_states.append(serving)


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
        manager._shutdown_requested = threading.Event()
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
        self.assertEqual(engine.serving_states[0], False)

    def test_draining_notification_only_marks_metrics_and_keeps_engine_running(self):
        engine = _FakeEngine()
        manager = self._manager(engine)
        manager._service_draining = threading.Event()
        with patch("rtp_llm.server.backend_manager.kmonitor.set_serving") as report:
            listener = threading.Thread(target=manager._wait_for_service_draining)
            listener.start()
            try:
                self.assertEqual(engine.serving_states, [])
            finally:
                manager._service_draining.set()
                listener.join(5)
            self.assertFalse(listener.is_alive())
            report.assert_called_once_with(False)
        self.assertEqual(engine.serving_states, [False])
        self.assertFalse(engine.stopped)
        self.assertFalse(engine.prepared)
        self.assertFalse(manager._shutdown_requested.is_set())
        self.assertFalse(manager._distributed_server.shutdown_requested)

    def test_request_shutdown_marks_metrics_and_native_engine_not_serving(self):
        engine = _FakeEngine()
        manager = self._manager(engine)

        with patch(
            "rtp_llm.server.backend_manager.kmonitor.set_serving"
        ) as set_python_serving:
            manager.request_shutdown()

        set_python_serving.assert_called_once_with(False)
        self.assertEqual(engine.serving_states, [False])
        self.assertTrue(manager._shutdown_requested.is_set())
        self.assertTrue(manager._distributed_server.shutdown_requested)

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


class BackendManagerServiceDrainingWatcherTest(unittest.TestCase):
    def _config(self):
        return SimpleNamespace(
            profiling_debug_logging_config=SimpleNamespace(log_file_backup_count=1),
            server_config=SimpleNamespace(rank_id=0, frontend_server_id=0),
            parallelism_config=SimpleNamespace(world_rank=1),
        )

    def test_scr_startup_defers_watcher_construction(self):
        event = threading.Event()
        with patch(
            "rtp_llm.server.backend_manager.AccessLogger"
        ), patch(
            "rtp_llm.server.backend_manager.DistributedServer"
        ), patch(
            "rtp_llm.server.backend_manager.get_global_controller"
        ), patch(
            "rtp_llm.server.backend_manager.get_log_path", return_value="/tmp"
        ), patch(
            "rtp_llm.server.backend_manager.kmonitor.bind_service_draining"
        ), patch.object(
            BackendManager, "start_service_draining_watcher"
        ) as start_watcher:
            manager = BackendManager(
                self._config(),
                event,
                defer_service_draining_watcher=True,
            )

        start_watcher.assert_not_called()
        self.assertIsNone(manager._service_draining_thread)

    def test_watcher_starts_once_after_barrier(self):
        event = threading.Event()
        manager = BackendManager.__new__(BackendManager)
        manager._service_draining = event
        manager._service_draining_thread = None
        manager.thread_lock_ = threading.Lock()
        watcher = Mock()

        with patch(
            "rtp_llm.server.backend_manager.threading.Thread",
            return_value=watcher,
        ) as thread:
            manager.start_service_draining_watcher()
            manager.start_service_draining_watcher()

        thread.assert_called_once_with(
            target=manager._wait_for_service_draining,
            name="service_draining",
            daemon=True,
        )
        watcher.start.assert_called_once_with()
        self.assertIs(manager._service_draining_thread, watcher)

    def test_watcher_is_inert_without_shared_event(self):
        manager = BackendManager.__new__(BackendManager)
        manager._service_draining = None
        manager._service_draining_thread = None
        manager.thread_lock_ = threading.Lock()

        with patch("rtp_llm.server.backend_manager.threading.Thread") as thread:
            manager.start_service_draining_watcher()

        thread.assert_not_called()


class BackendManagerServiceStartTest(unittest.TestCase):
    """The SCR gate must be an opt-in flag, not a new default startup path."""

    def _manager_and_config(self):
        manager = BackendManager.__new__(BackendManager)
        manager._service_draining = None
        manager._distributed_server = Mock()
        manager._distributed_server.get_nccl_comm_config.return_value = None

        profiling = SimpleNamespace(log_file_backup_count=1)
        py_config = SimpleNamespace(
            profiling_debug_logging_config=profiling,
            server_config=SimpleNamespace(),
            parallelism_config=SimpleNamespace(world_size=1, local_rank=0),
            distribute_config=SimpleNamespace(dist_comm_timeout=1),
            model_args=object(),
            lora_config=SimpleNamespace(merge_lora=False),
            generate_env_config=object(),
            embedding_config=object(),
            quantization_config=object(),
            render_config=object(),
            eplb_config=object(),
            vit_config=object(),
        )

        engine_config = SimpleNamespace(
            parallelism_config=SimpleNamespace(world_size=1, local_rank=0),
            runtime_config=object(),
            kv_cache_config=object(),
            profiling_debug_logging_config=profiling,
            moe_config=SimpleNamespace(
                moe_strategy="", use_deepep_moe=False, use_all_gather=False
            ),
            sp_config=object(),
        )
        model_config = SimpleNamespace(expert_num=0, task_type=TaskType.LANGUAGE_MODEL)
        engine = SimpleNamespace(task_type=TaskType.LANGUAGE_MODEL)
        return manager, py_config, engine_config, model_config, engine

    def test_start_forwards_defer_service_start_without_changing_default(self):
        """Normal startup still asks the engine to start its listeners immediately."""
        manager, py_config, engine_config, model_config, engine = (
            self._manager_and_config()
        )
        with patch(
            "rtp_llm.server.backend_manager.EngineConfig.create",
            return_value=engine_config,
        ), patch(
            "rtp_llm.server.backend_manager.get_world_info", return_value=None
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_model_config",
            return_value=model_config,
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.update_engine_config_from_model_config"
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_propose_model_config",
            return_value=None,
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.from_model_configs",
            return_value=engine,
        ) as create_engine:
            manager.py_env_configs = py_config
            manager.start()
            manager.start(defer_service_start=True)

        self.assertEqual(
            [call.kwargs["defer_service_start"] for call in create_engine.call_args_list],
            [False, True],
        )

    def test_start_rejects_embedding_in_deferred_template_mode(self):
        manager, py_config, engine_config, model_config, engine = (
            self._manager_and_config()
        )
        model_config.task_type = TaskType.DENSE_EMBEDDING
        with patch(
            "rtp_llm.server.backend_manager.EngineConfig.create",
            return_value=engine_config,
        ), patch(
            "rtp_llm.server.backend_manager.get_world_info", return_value=None
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_model_config",
            return_value=model_config,
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.update_engine_config_from_model_config"
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.create_propose_model_config",
            return_value=None,
        ), patch(
            "rtp_llm.server.backend_manager.ModelFactory.from_model_configs"
        ) as create_engine:
            manager.py_env_configs = py_config
            with self.assertRaisesRegex(RuntimeError, "does not support embedding"):
                manager.start(defer_service_start=True)

        create_engine.assert_not_called()

if __name__ == "__main__":
    unittest.main()
