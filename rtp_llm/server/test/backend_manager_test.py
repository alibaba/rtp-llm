import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.ops import (
    NcclCommConfig,
    RoleType,
    SpeculativeType,
    TaskType,
)
from rtp_llm.server import backend_manager
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


class BackendManagerStartupTest(unittest.TestCase):
    def test_direct_start_forwards_collective_overrides_without_launcher(self):
        configs = PyEnvConfigs()
        configs.parallelism_config.tp_size = 2
        configs.parallelism_config.ep_size = 2
        configs.parallelism_config.world_size = 2
        configs.parallelism_config.local_world_size = 2
        set_parallelism_config(configs.parallelism_config)
        configs.all_reduce_backend = "nccl"
        configs.all_reduce_large_prefill = False
        server = MagicMock()
        comm_config = NcclCommConfig()
        server.get_nccl_comm_config.return_value = comm_config
        server.get_nccl_init_port.return_value = 29501
        with (
            patch.object(backend_manager, "AccessLogger"),
            patch.object(backend_manager, "DistributedServer", return_value=server),
            patch.object(backend_manager.kmonitor, "init"),
            patch.object(backend_manager, "init_distributed_environment") as initialize,
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


class BackendManagerServiceStartTest(unittest.TestCase):
    """The SCR gate must be an opt-in flag, not a new default startup path."""

    def _manager_and_config(self):
        manager = BackendManager.__new__(BackendManager)
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
            parallelism_config=SimpleNamespace(
                world_size=1, local_rank=0, role_type=RoleType.PDFUSION
            ),
            runtime_config=SimpleNamespace(
                fifo_scheduler_config=SimpleNamespace(max_context_batch_size=1)
            ),
            kv_cache_config=object(),
            profiling_debug_logging_config=profiling,
            moe_config=SimpleNamespace(
                use_deepep_moe=False, use_all_gather=False
            ),
            sp_config=SimpleNamespace(type=SpeculativeType.MTP),
            # Graph validation returns early when the master switch is off.
            hw_kernel_config=SimpleNamespace(
                enable_cuda_graph=False,
                generation_prefill_capture_token_buckets=[],
            ),
            concurrency_config=SimpleNamespace(concurrency_limit=1),
        )
        model_config = SimpleNamespace(expert_num=0, task_type=TaskType.LANGUAGE_MODEL)
        engine = SimpleNamespace(task_type=TaskType.LANGUAGE_MODEL)
        return manager, py_config, engine_config, model_config, engine

    def test_start_prepares_then_start_service_binds_listeners(self):
        """Every deployment is two-phase: start() prepares, start_service() binds."""
        manager, py_config, engine_config, model_config, engine = (
            self._manager_and_config()
        )
        engine.start_service = Mock()
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
            engine.start_service.assert_not_called()
            manager.start_service()
            manager.start_service()

        create_engine.assert_called_once()
        # BackendManager forwards every call; LanguageCppEngine owns the
        # idempotence guard.
        self.assertEqual(engine.start_service.call_count, 2)

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
        ) as create_engine, patch(
            "rtp_llm.server.backend_manager.template_phase_active",
            return_value=True,
        ):
            manager.py_env_configs = py_config
            with self.assertRaisesRegex(RuntimeError, "does not support embedding"):
                manager.start()

        create_engine.assert_not_called()

if __name__ == "__main__":
    unittest.main()
