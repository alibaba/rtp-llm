import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.server.backend_manager import BackendManager


class _FakeEngine:
    def __init__(self, exc=None):
        self.exc = exc
        self.stopped = False
        self.set_serving = Mock()
        self.stop_count = 0

    def stop(self):
        self.stopped = True
        self.stop_count += 1
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
    def setUp(self):
        self.metric_patch = patch("rtp_llm.server.backend_manager.kmonitor")
        self.metrics = self.metric_patch.start()
        self.addCleanup(self.metric_patch.stop)

    @staticmethod
    def make_manager(event=None):
        manager = BackendManager.__new__(BackendManager)
        manager.engine = None
        manager._shutdown_requested = threading.Event()
        manager._service_draining = event
        manager._metrics_state_lock = threading.RLock()
        manager._stop_lock = threading.Lock()
        manager._stopped = False
        manager._status_watcher_stop = threading.Event()
        return manager

    def test_drain_updates_native_tags_without_stopping_engine(self):
        event = threading.Event()
        manager = self.make_manager(event)
        manager.engine = _FakeEngine()
        watcher = threading.Thread(target=manager._watch_service_draining)
        watcher.start()
        event.set()
        watcher.join(2)
        self.assertFalse(watcher.is_alive())
        manager.engine.set_serving.assert_called_once_with(False)
        self.metrics.set_serving.assert_called_with(False)
        self.assertFalse(manager.engine.stopped)
        self.assertFalse(manager._shutdown_requested.is_set())

    def test_drain_before_engine_created_is_replayed_when_ready(self):
        event = threading.Event()
        event.set()
        manager = self.make_manager(event)
        manager._watch_service_draining()
        manager.engine = _FakeEngine()
        manager._mark_serving_when_ready()
        manager.engine.set_serving.assert_called_once_with(False)
        self.metrics.start_serving_when_ready.assert_not_called()

    def test_ready_starts_native_and_python_metrics(self):
        manager = self.make_manager()
        manager.engine = _FakeEngine()
        manager._mark_serving_when_ready()
        manager.engine.set_serving.assert_called_once_with(True)
        self.metrics.start_serving_when_ready.assert_called_once_with()

    def test_shutdown_before_ready_cannot_reopen_metrics(self):
        manager = self.make_manager()
        manager.request_shutdown()
        manager.engine = _FakeEngine()
        manager._mark_serving_when_ready()
        manager.engine.set_serving.assert_called_once_with(False)
        self.metrics.start_serving_when_ready.assert_not_called()

    def test_shutdown_can_reenter_a_metric_transition_from_signal_handler(self):
        manager = self.make_manager()
        manager.engine = _FakeEngine()
        with manager._metrics_state_lock:
            manager.request_shutdown()
        manager.engine.set_serving.assert_called_once_with(False)

    def test_repeated_stop_cleans_up_only_once(self):
        manager = self.make_manager()
        manager.engine = _FakeEngine()
        nfs_manager = _FakeNfsManager()
        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()
            manager.stop()
        self.assertEqual(manager.engine.stop_count, 1)
        self.assertTrue(manager._status_watcher_stop.is_set())

    def test_serving_update_failure_does_not_skip_engine_cleanup(self):
        manager = self.make_manager()
        manager.engine = _FakeEngine()
        manager.engine.set_serving.side_effect = RuntimeError("status unavailable")
        nfs_manager = _FakeNfsManager()
        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()
        self.assertTrue(manager.engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_after_engine_stop(self):
        manager = self.make_manager()
        engine = _FakeEngine()
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)

    def test_stop_unmounts_nfs_even_when_engine_stop_raises(self):
        manager = self.make_manager()
        engine_error = RuntimeError("engine stop failed")
        engine = _FakeEngine(engine_error)
        nfs_manager = _FakeNfsManager()
        manager.engine = engine

        with patch("rtp_llm.utils.fuser._nfs_manager", nfs_manager):
            with self.assertRaisesRegex(RuntimeError, "engine stop failed"):
                manager.stop()

        self.assertTrue(engine.stopped)
        self.assertTrue(nfs_manager.unmounted)


class BackendManagerDeepEPTest(unittest.TestCase):
    def test_mega_moe_skips_deepep_but_other_strategies_initialize_it(self):
        for strategy in (
            "mega_moe",
            "mega_moe_se",
            "mega_moe_fp8",
            "mega_moe_fp8_se",
            "mega_moe_fused",
            "auto",
        ):
            with self.subTest(strategy=strategy):
                manager = BackendManager.__new__(BackendManager)
                manager.py_env_configs = Mock()
                manager._distributed_server = Mock()
                manager._mark_serving_when_ready = Mock()
                config = Mock()
                config.parallelism_config.world_size = 4
                config.moe_config.use_all_gather = False
                config.moe_config.use_deepep_moe = True
                config.moe_config.use_mori_ep = False
                config.moe_config.moe_strategy = strategy
                factory = Mock()
                factory.create_model_config.return_value.expert_num = 8
                init_deepep = Mock()
                with patch.multiple(
                    "rtp_llm.server.backend_manager",
                    EngineConfig=Mock(create=Mock(return_value=config)),
                    ModelFactory=factory,
                    init_distributed_environment=Mock(),
                    get_world_info=Mock(),
                    update_worker_addrs=Mock(),
                    validate_hw_kernel_group_args=Mock(),
                ), patch.dict(
                    "sys.modules",
                    {
                        "rtp_llm.models_py.distributed.deepep_wrapper": SimpleNamespace(
                            init_deepep_wrapper=init_deepep
                        )
                    },
                ):
                    manager.start()
                if strategy == "auto":
                    init_deepep.assert_called_once_with(
                        config, factory.create_model_config.return_value
                    )
                else:
                    init_deepep.assert_not_called()
                factory.from_model_configs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
