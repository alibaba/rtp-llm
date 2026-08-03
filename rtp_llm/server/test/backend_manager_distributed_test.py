import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import MagicMock, call, create_autospec, patch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import DistributedServer
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.server import backend_manager


class BackendManagerDistributedInitializationTest(unittest.TestCase):
    def _engine_config(self, world_size: int, tp_size: int, cuda_graph: bool):
        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = world_size
        parallelism_config.tp_size = tp_size
        hw_kernel_config = HWKernelConfig()
        hw_kernel_config.enable_cuda_graph = cuda_graph
        values = {
            field.name: MagicMock(name=field.name) for field in fields(EngineConfig)
        }
        values.update(
            parallelism_config=parallelism_config,
            hw_kernel_config=hw_kernel_config,
        )
        return EngineConfig(**values)

    def test_graph_requirement_matrix_and_full_initialization_contract(self):
        for world_size, cuda_graph, timeout in (
            (1, False, 77),
            (1, True, None),
            (2, False, 77),
            (2, True, None),
        ):
            tp_size = world_size
            with self.subTest(
                world_size=world_size,
                tp_size=tp_size,
                cuda_graph=cuda_graph,
                timeout=timeout,
            ):
                engine_config = self._engine_config(world_size, tp_size, cuda_graph)
                server = create_autospec(DistributedServer, instance=True)
                server.get_nccl_comm_config.return_value = "nccl-config"
                server.get_nccl_init_port.return_value = 12345
                py_config = PyEnvConfigs()
                py_config.distribute_config.dist_comm_timeout = timeout
                with patch.object(
                    backend_manager,
                    "init_distributed_environment",
                    autospec=True,
                ) as init:
                    backend_manager._init_distributed_environment_for_backend(
                        engine_config, py_config, server
                    )
                if world_size == 1:
                    init.assert_not_called()
                    continue
                init.assert_called_once_with(
                    engine_config.parallelism_config,
                    nccl_comm_config="nccl-config",
                    nccl_init_port=12345,
                    backend="nccl",
                    timeout=timeout,
                    graph_required=cuda_graph,
                )

    def test_graph_communication_prepare_failure_is_not_downgraded(self):
        engine_config = self._engine_config(2, 2, True)
        server = create_autospec(DistributedServer, instance=True)
        py_config = PyEnvConfigs()
        with patch.object(
            backend_manager,
            "init_distributed_environment",
            side_effect=RuntimeError("graph prepare failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "graph prepare failed"):
                backend_manager._init_distributed_environment_for_backend(
                    engine_config, py_config, server
                )

    def test_start_stops_before_model_construction_when_graph_prepare_fails(self):
        engine_config = self._engine_config(2, 2, True)
        py_config = PyEnvConfigs()
        server = create_autospec(DistributedServer, instance=True)
        with (
            patch.object(backend_manager, "AccessLogger"),
            patch.object(backend_manager, "DistributedServer", return_value=server),
            patch.object(backend_manager, "get_global_controller"),
            patch.object(backend_manager.kmonitor, "init"),
            patch.object(
                backend_manager.EngineConfig, "create", return_value=engine_config
            ),
            patch.object(
                backend_manager,
                "init_distributed_environment",
                side_effect=RuntimeError("graph prepare failed"),
            ),
            patch.object(
                backend_manager.ModelFactory, "create_model_config"
            ) as create_model_config,
        ):
            manager = backend_manager.BackendManager(py_config)
            with self.assertRaisesRegex(RuntimeError, "graph prepare failed"):
                manager.start()
        create_model_config.assert_not_called()

    def test_start_prepares_distributed_graph_before_engine_construction(self):
        engine_config = self._engine_config(2, 1, True)
        py_config = PyEnvConfigs()
        server = create_autospec(DistributedServer, instance=True)
        server.get_nccl_comm_config.return_value = "nccl-config"
        server.get_nccl_init_port.return_value = 12345
        model_config = SimpleNamespace(expert_num=0)
        engine = SimpleNamespace(task_type="language-model")
        order = MagicMock()

        def create_engine(**kwargs):
            order.engine()
            return engine

        with (
            patch.object(backend_manager, "AccessLogger"),
            patch.object(backend_manager, "DistributedServer", return_value=server),
            patch.object(backend_manager, "get_global_controller"),
            patch.object(backend_manager.kmonitor, "init"),
            patch.object(
                backend_manager.EngineConfig,
                "create",
                return_value=engine_config,
            ),
            patch.object(
                backend_manager,
                "init_distributed_environment",
                autospec=True,
                side_effect=lambda *args, **kwargs: order.distributed(),
            ) as init_distributed,
            patch.object(backend_manager, "get_world_info", return_value=object()),
            patch.object(backend_manager, "update_worker_addrs"),
            patch.object(
                backend_manager.ModelFactory,
                "create_model_config",
                return_value=model_config,
            ),
            patch.object(
                backend_manager.ModelFactory,
                "update_engine_config_from_model_config",
            ),
            patch.object(
                backend_manager.ModelFactory,
                "create_propose_model_config",
                return_value=None,
            ),
            patch.object(
                backend_manager.ModelFactory,
                "from_model_configs",
                side_effect=create_engine,
            ),
        ):
            manager = backend_manager.BackendManager(py_config)
            manager.start()

        init_distributed.assert_called_once_with(
            engine_config.parallelism_config,
            nccl_comm_config="nccl-config",
            nccl_init_port=12345,
            backend="nccl",
            timeout=py_config.distribute_config.dist_comm_timeout,
            graph_required=True,
        )
        self.assertEqual(order.mock_calls, [call.distributed(), call.engine()])
        self.assertIs(manager.engine, engine)


if __name__ == "__main__":
    unittest.main()
