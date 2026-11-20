"""Exercise address and identity fixups with the production configuration schema."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.utils.scr.restore_context import RestoreContext


class ScrEndpointConfigTest(unittest.TestCase):

    def setUp(self):
        env = patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "restore"})
        env.start()
        self.addCleanup(env.stop)
        self.configs = PyEnvConfigs()
        self.configs.role_config.role_type = "PREFILL"
        self.configs.parallelism_config.world_size = 1
        self.configs.parallelism_config.local_world_size = 1
        member = WorkerInfo("192.0.2.10", 0, 0, "seed", 9000, 9)
        self.world = WorldInfo([member], member, member, 1, True)
        self.context = RestoreContext("192.0.2.20")

    def test_backend_refreshes_engine_endpoints(self):
        backend = BackendManager.__new__(BackendManager)
        backend.py_env_configs = self.configs
        backend._distributed_server = Mock()
        backend._engine_config = SimpleNamespace(
            runtime_config=SimpleNamespace(),
            parallelism_config=self.configs.parallelism_config,
        )
        backend._world_info = self.world
        backend.engine = Mock()
        with patch(
            "rtp_llm.server.backend_manager.get_world_info", return_value=self.world
        ), patch(
            "rtp_llm.utils.scr.runtime_fixup.get_restore_runtime_identity",
            return_value=self.context,
        ):
            backend.restore_fixup(self.context)
        config = backend._engine_config.runtime_config
        self.assertEqual(config.worker_grpc_addrs, ["127.0.0.1:9001"])
        self.assertEqual(config.worker_addrs, ["192.0.2.20:9002:9004"])
        self.assertEqual(self.configs.server_config.ip, self.context.pod_ip)
        backend.engine.update_runtime_endpoints.assert_called_once_with(
            config, backend._world_info
        )


if __name__ == "__main__":
    unittest.main()
