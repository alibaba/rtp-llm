"""Exercise endpoint restore hooks with the production configuration schema."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.server.backend_manager import BackendManager
from rtp_llm.utils.scr_restore_context import RestoreContext
from rtp_llm.utils.scr_template_utils import _BackendVisitorTemplateHook


class ScrEndpointConfigTest(unittest.TestCase):

    def test_restored_manifest_gates_backend_before_publishing_real_endpoints(self):
        configs = PyEnvConfigs()
        configs.role_config.role_type = "PREFILL"
        member = WorkerInfo("192.0.2.10", 0, 0, "seed", 9000, 9)
        peer = WorkerInfo("192.0.2.11", 0, 1, "peer", 9000, 9)
        configs.parallelism_config.world_size = 2
        configs.parallelism_config.local_world_size = 1
        current = WorldInfo([member, peer], member, member, 2, True)
        for ready in (False, True):
            with self.subTest(ready=ready):
                backend = BackendManager.__new__(BackendManager)
                backend.py_env_configs = configs
                backend._distributed_server = Mock()
                backend._world_info = current
                backend._engine_config = SimpleNamespace(
                    runtime_config=SimpleNamespace(),
                    parallelism_config=configs.parallelism_config,
                )
                backend.engine = Mock()
                manifest = {
                    "generation": "g1",
                    "phase": "restore",
                    "num_nodes": 2,
                    "transport": {"ready": ready},
                    "members": [
                        {
                            "world_rank": 0,
                            "local_rank": 0,
                            "ip": "192.0.2.50",
                            "server_port": 9000,
                        },
                        {
                            "world_rank": 1,
                            "local_rank": 0,
                            "ip": "192.0.2.51",
                            "server_port": 9000,
                        },
                    ],
                }
                with patch.dict(
                    os.environ,
                    {
                        "SCR_PHASE": "checkpoint",
                        "RTP_LLM_SCR_ENDPOINT_MANIFEST": json.dumps(manifest),
                    },
                ), patch(
                    "rtp_llm.server.backend_manager.get_world_info",
                    return_value=current,
                ):
                    if ready:
                        backend.restore_fixup(RestoreContext("g1", "192.0.2.50"))
                        self.assertEqual(
                            backend._engine_config.runtime_config.worker_addrs,
                            ["192.0.2.50:9002:9004", "192.0.2.51:9002:9004"],
                        )
                        self.assertEqual(
                            backend._engine_config.runtime_config.worker_grpc_addrs,
                            ["192.0.2.50:9001", "192.0.2.51:9001"],
                        )
                        backend.engine.update_runtime_endpoints.assert_called_once()
                        self.assertEqual(configs.server_config.ip, "192.0.2.50")
                    else:
                        with self.assertRaisesRegex(RuntimeError, "transport.ready"):
                            backend.restore_fixup(RestoreContext("g1", "192.0.2.50"))
                        backend.engine.update_runtime_endpoints.assert_not_called()
                        self.assertIs(backend._world_info, current)
                backend._distributed_server.assert_not_called()
                self.assertEqual(backend._distributed_server.method_calls, [])

    def test_visitor_uses_real_config_and_preserves_endpoint_requirements(self):
        for role, nodes, phase, manifest_phase, required in [
            ("PREFILL", 1, "restore", "", False),
            ("DECODE", 1, "restore", "", False),
            ("PDFUSION", 2, "restore", "", True),
            ("PDFUSION", 1, "restore", "", False),
            ("PREFILL", 1, "checkpoint", "", False),
            ("PREFILL", 1, "checkpoint", "restore", False),
            ("PDFUSION", 2, "checkpoint", "restore", True),
        ]:
            with self.subTest(role=role, nodes=nodes, phase=phase):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                current = SimpleNamespace(num_nodes=nodes)
                restored = SimpleNamespace(num_nodes=nodes)
                visitor = Mock()
                hook = _BackendVisitorTemplateHook(visitor, configs)
                manifest = (
                    {"generation": "generation-2", "phase": manifest_phase}
                    if manifest_phase
                    else None
                )
                with patch.dict(
                    os.environ,
                    {
                        "SCR_PHASE": phase,
                        "RTP_LLM_SCR_ENDPOINT_MANIFEST": (
                            json.dumps(manifest) if manifest else ""
                        ),
                    },
                ), patch(
                    "rtp_llm.distribute.distributed_server.get_world_info",
                    return_value=current,
                ), patch(
                    "rtp_llm.distribute.distributed_server.get_dp_addrs_from_world_info",
                    return_value=["192.0.2.20:9001"],
                ), patch(
                    "rtp_llm.utils.scr_endpoint_provider.resolve_world_info",
                    return_value=restored,
                ) as resolve:
                    hook.restore_fixup(RestoreContext("generation-2", "192.0.2.20"))
                resolve.assert_called_once_with(
                    current,
                    generation="generation-2",
                    require_manifest=required,
                    require_transport=required,
                    expected_world_size=configs.parallelism_config.world_size,
                    manifest=manifest,
                )
                visitor.update_addresses.assert_called_once_with(["192.0.2.20:9001"])

    def test_backend_uses_real_config_and_refreshes_engine_endpoints(self):
        for role, nodes, phase, manifest_phase, required in [
            ("PREFILL", 1, "restore", "", False),
            ("DECODE", 1, "restore", "", False),
            ("PDFUSION", 2, "restore", "", True),
            ("PDFUSION", 1, "restore", "", False),
            ("PREFILL", 1, "checkpoint", "", False),
            ("PREFILL", 1, "checkpoint", "restore", False),
            ("PDFUSION", 2, "checkpoint", "restore", True),
        ]:
            with self.subTest(role=role, nodes=nodes, phase=phase):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                backend = BackendManager.__new__(BackendManager)
                backend.py_env_configs = configs
                backend._distributed_server = Mock()
                backend._engine_config = SimpleNamespace(
                    runtime_config=object(),
                    parallelism_config=configs.parallelism_config,
                )
                backend._world_info = object()
                backend.engine = Mock()
                current = SimpleNamespace(num_nodes=nodes)
                restored = SimpleNamespace(num_nodes=nodes)
                manifest = (
                    {"generation": "generation-2", "phase": manifest_phase}
                    if manifest_phase
                    else None
                )
                with patch.dict(
                    os.environ,
                    {
                        "SCR_PHASE": phase,
                        "RTP_LLM_SCR_ENDPOINT_MANIFEST": (
                            json.dumps(manifest) if manifest else ""
                        ),
                    },
                ), patch(
                    "rtp_llm.server.backend_manager.get_world_info",
                    return_value=current,
                ), patch(
                    "rtp_llm.utils.scr_endpoint_provider.resolve_world_info",
                    return_value=restored,
                ) as resolve, patch(
                    "rtp_llm.server.backend_manager.update_worker_addrs"
                ) as update:
                    backend.restore_fixup(RestoreContext("generation-2", "192.0.2.20"))
                resolve.assert_called_once_with(
                    current,
                    generation="generation-2",
                    require_manifest=required,
                    require_transport=required,
                    expected_world_size=configs.parallelism_config.world_size,
                    manifest=manifest,
                )
                update.assert_called_once_with(
                    backend._engine_config.runtime_config,
                    configs.parallelism_config,
                    restored,
                )
                self.assertIs(backend._world_info, restored)
                backend.engine.update_runtime_endpoints.assert_called_once_with(
                    backend._engine_config.runtime_config, restored
                )


if __name__ == "__main__":
    unittest.main()
