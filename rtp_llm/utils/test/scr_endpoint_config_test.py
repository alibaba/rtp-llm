"""Exercise endpoint restore hooks with the production configuration schema."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.utils.scr_template_utils import _BackendVisitorTemplateHook


class ScrEndpointConfigTest(unittest.TestCase):
    def test_visitor_uses_real_config_and_preserves_endpoint_requirements(self):
        for role, nodes, phase, required in [
            ("PREFILL", 1, "restore", True),
            ("DECODE", 1, "restore", True),
            ("PDFUSION", 2, "restore", True),
            ("PDFUSION", 1, "restore", False),
            ("PREFILL", 1, "checkpoint", False),
        ]:
            with self.subTest(role=role, nodes=nodes, phase=phase):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                current = SimpleNamespace(num_nodes=nodes)
                restored = object()
                visitor = Mock()
                hook = _BackendVisitorTemplateHook(visitor, configs)
                with patch.dict(os.environ, {"SCR_PHASE": phase}), patch(
                    "rtp_llm.distribute.distributed_server.get_world_info", return_value=current
                ), patch(
                    "rtp_llm.distribute.distributed_server.get_dp_addrs_from_world_info",
                    return_value=["192.0.2.20:9001"],
                ), patch(
                    "rtp_llm.utils.scr_endpoint_provider.resolve_world_info", return_value=restored
                ) as resolve:
                    hook.restore_fixup("generation-2")
                resolve.assert_called_once_with(
                    current, generation="generation-2", require_manifest=required,
                    require_transport=required,
                )
                visitor.update_addresses.assert_called_once_with(["192.0.2.20:9001"])

    def test_backend_uses_real_config_and_refreshes_engine_endpoints(self):
        for role, nodes, phase, required in [
            ("PREFILL", 1, "restore", True),
            ("DECODE", 1, "restore", True),
            ("PDFUSION", 2, "restore", True),
            ("PDFUSION", 1, "restore", False),
            ("PREFILL", 1, "checkpoint", False),
        ]:
            with self.subTest(role=role, nodes=nodes, phase=phase):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                backend = BackendManager.__new__(BackendManager)
                backend.py_env_configs = configs
                backend._distributed_server = Mock()
                backend._engine_config = SimpleNamespace(
                    runtime_config=object(), parallelism_config=configs.parallelism_config
                )
                backend._world_info = object()
                backend.engine = Mock()
                current = SimpleNamespace(num_nodes=nodes)
                restored = object()
                with patch.dict(os.environ, {"SCR_PHASE": phase}), patch(
                    "rtp_llm.server.backend_manager.get_world_info", return_value=current
                ), patch(
                    "rtp_llm.server.backend_manager.resolve_world_info", return_value=restored
                ) as resolve, patch(
                    "rtp_llm.server.backend_manager.update_worker_addrs"
                ) as update:
                    backend.restore_fixup("generation-2")
                resolve.assert_called_once_with(
                    current, generation="generation-2", require_manifest=required,
                    require_transport=required,
                )
                update.assert_called_once_with(
                    backend._engine_config.runtime_config, configs.parallelism_config, restored
                )
                self.assertIs(backend._world_info, restored)
                backend.engine.update_runtime_endpoints.assert_called_once_with(
                    backend._engine_config.runtime_config, restored
                )


if __name__ == "__main__":
    unittest.main()
