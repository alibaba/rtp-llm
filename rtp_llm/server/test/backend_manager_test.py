import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
