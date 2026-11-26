import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.ops import NcclCommConfig
from rtp_llm.server import backend_manager
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


class BackendManagerStartupTest(unittest.TestCase):
    def test_direct_start_forwards_collective_overrides_without_launcher(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.tp_size = 2
        configs.parallelism_config.ep_size = 2
        configs.parallelism_config.world_size = 2
        configs.parallelism_config.local_world_size = 2
        set_parallelism_config(configs.parallelism_config)
        configs.py_hw_kernel_config.glm5_cp_opt = True
        configs.all_reduce_backend = "nccl"
        configs.all_reduce_large_prefill = False
        server = MagicMock()
        comm_config = NcclCommConfig()
        server.get_nccl_comm_config.return_value = comm_config
        server.get_nccl_init_port.return_value = 29501
        from rtp_llm.models_py.distributed import pynccl_cp

        def check_cp_configuration(*args, **kwargs):
            self.assertTrue(pynccl_cp.enabled())

        with (
            patch.object(pynccl_cp, "_OPT_ENABLED", False),
            patch.object(backend_manager, "AccessLogger"),
            patch.object(backend_manager, "DistributedServer", return_value=server),
            patch.object(backend_manager.kmonitor, "init"),
            patch.object(
                backend_manager,
                "init_distributed_environment",
                side_effect=check_cp_configuration,
            ) as initialize,
            patch.object(
                backend_manager,
                "get_world_info",
                side_effect=RuntimeError("stop before model loading"),
            ),
        ):
            manager = BackendManager(configs)
            with self.assertRaisesRegex(RuntimeError, "stop before model loading"):
                manager.start()
        initialize.assert_called_once_with(
            configs.parallelism_config,
            nccl_comm_config=comm_config,
            nccl_init_port=29501,
            backend="nccl",
            collective_config=CollectiveConfig("nccl", False),
            timeout=configs.distribute_config.dist_comm_timeout,
        )

    def test_moe_ep_consistency_failure_blocks_deepep_and_unsupported_raises(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.tp_size = 1
        configs.parallelism_config.dp_size = 4
        configs.parallelism_config.ep_size = 4
        configs.parallelism_config.world_size = 4
        configs.parallelism_config.local_world_size = 2
        set_parallelism_config(configs.parallelism_config)
        configs.moe_config.use_deepep_moe = True

        server = MagicMock()
        server.validate_moe_ep_consistency.side_effect = ValueError("EP mismatch")
        model_config = SimpleNamespace(expert_num=256, task_type=None)

        with (
            patch.object(backend_manager, "AccessLogger"),
            patch.object(backend_manager, "DistributedServer", return_value=server),
            patch.object(backend_manager.kmonitor, "init"),
            patch.object(backend_manager, "init_distributed_environment"),
            patch.object(backend_manager, "get_world_info"),
            patch.object(
                backend_manager.ModelFactory,
                "create_model_config",
                return_value=model_config,
            ),
            patch.object(
                backend_manager.ModelFactory, "update_engine_config_from_model_config"
            ),
            patch(
                "rtp_llm.models_py.distributed.deepep_wrapper.init_deepep_wrapper"
            ) as init_deepep,
        ):
            manager = BackendManager(configs)
            with self.assertRaisesRegex(ValueError, "EP mismatch"):
                manager.start()
        server.validate_moe_ep_consistency.assert_called_once()
        init_deepep.assert_not_called()

        from rtp_llm.models_py.distributed import deepep_wrapper

        with patch.object(
            deepep_wrapper.DeepEPWrapper, "supported", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "not supported"):
                deepep_wrapper.init_deepep_wrapper(MagicMock(), model_config)


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
                with (
                    patch.multiple(
                        "rtp_llm.server.backend_manager",
                        EngineConfig=Mock(create=Mock(return_value=config)),
                        ModelFactory=factory,
                        init_distributed_environment=Mock(),
                        get_world_info=Mock(),
                        update_worker_addrs=Mock(),
                        validate_hw_kernel_group_args=Mock(),
                    ),
                    patch.dict(
                        "sys.modules",
                        {
                            "rtp_llm.models_py.distributed.deepep_wrapper": SimpleNamespace(
                                init_deepep_wrapper=init_deepep
                            )
                        },
                    ),
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
