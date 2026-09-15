import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.utils.scr_endpoint_provider import (
    is_restore_phase,
    read_restore_manifest,
    resolve_world_info,
)


def world():
    members = [
        WorkerInfo("10.0.0.%d" % (i + 1), i, i, "rank_%d" % i, 8088, 9)
        for i in range(2)
    ]
    return WorldInfo(members, members[0], members[0], 2, True)


class ScrEndpointProviderTest(unittest.TestCase):

    def test_manifest_phase_overrides_snapshotted_checkpoint_environment(self):
        manifest = {"generation": "g1", "phase": "restore"}
        with patch.dict(
            os.environ,
            {
                "SCR_PHASE": "checkpoint",
                "RTP_LLM_SCR_ENDPOINT_MANIFEST": json.dumps(manifest),
            },
        ):
            self.assertTrue(is_restore_phase(read_restore_manifest("g1")))
        with patch.dict(os.environ, {"SCR_PHASE": "restore"}):
            self.assertTrue(is_restore_phase({"phase": "checkpoint"}))

    def test_global_manifest_accepts_frontend_local_world_view(self):
        current = world()
        current.members = current.members[:1]
        manifest = {
            "generation": "g1",
            "num_nodes": 2,
            "members": [
                {
                    "world_rank": 0,
                    "local_rank": 0,
                    "ip": "192.0.2.10",
                    "server_port": 9000,
                },
                {
                    "world_rank": 1,
                    "local_rank": 1,
                    "ip": "192.0.2.11",
                    "server_port": 9000,
                },
            ],
        }
        restored = resolve_world_info(
            current, generation="g1", expected_world_size=2, manifest=manifest
        )
        self.assertEqual(len(restored.members), 2)
        self.assertEqual(restored.self.world_rank, current.self.world_rank)

    def test_loopback_is_automatic_only_for_a_complete_single_node_template(self):
        current = world()
        current.members = current.members[:1]
        current.self = current.master = current.members[0]
        for enabled, phase, nodes, expected_size, expected_ip in [
            ("1", "checkpoint", 1, 1, "127.0.0.1"),
            ("1", "restore", 1, 1, "127.0.0.1"),
            ("1", "normal", 1, 1, "10.0.0.1"),
            ("0", "checkpoint", 1, 1, "10.0.0.1"),
            ("1", "checkpoint", 2, 2, "10.0.0.1"),
            ("1", "checkpoint", 1, 2, "10.0.0.1"),
        ]:
            with self.subTest(
                enabled=enabled, phase=phase, nodes=nodes, expected_size=expected_size
            ), patch.dict(
                os.environ,
                {
                    "RTPLLM_ENABLE_SCR": enabled,
                    "SCR_PHASE": phase,
                    "RTP_LLM_SCR_ENDPOINT_MANIFEST": "",
                },
                clear=True,
            ):
                current.num_nodes = nodes
                restored = resolve_world_info(
                    current, generation="g1", expected_world_size=expected_size
                )
                self.assertEqual(restored.members[0].ip, expected_ip)
                self.assertEqual(current.members[0].ip, "10.0.0.1")

    def test_restore_manifest_replaces_transport_hosts(self):
        manifest = {
            "generation": "g1",
            "num_nodes": 2,
            "transport": {"ready": True},
            "members": [
                {"world_rank": 0, "local_rank": 0, "ip": "192.0.2.10", "server_port": 9000},
                {"world_rank": 1, "local_rank": 1, "ip": "192.0.2.11", "server_port": 9000},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(manifest, file)
            file.flush()
            with patch.dict(
                os.environ,
                {"RTP_LLM_SCR_ENDPOINT_MANIFEST": file.name},
                clear=False,
            ):
                restored = resolve_world_info(world(), generation="g1", require_manifest=True)
        self.assertEqual([member.ip for member in restored.members], ["192.0.2.10", "192.0.2.11"])
        self.assertEqual(restored.members[1].rpc_server_port, 9010)

    def test_cross_host_restore_requires_manifest(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_SCR_ENDPOINT_MANIFEST", None)
            with self.assertRaises(RuntimeError):
                resolve_world_info(world(), generation="g1", require_manifest=True)

    def test_manifest_generation_is_checked(self):
        manifest = {
            "generation": "other",
            "num_nodes": 2,
            "transport": {"ready": True},
            "members": [
                {"world_rank": 0, "local_rank": 0, "ip": "192.0.2.10", "server_port": 9000},
                {"world_rank": 1, "local_rank": 1, "ip": "192.0.2.11", "server_port": 9000},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(manifest, file)
            file.flush()
            with patch.dict(os.environ, {"RTP_LLM_SCR_ENDPOINT_MANIFEST": file.name}):
                with self.assertRaises(RuntimeError):
                    resolve_world_info(world(), generation="g1", require_manifest=True)

    def test_manifest_restore_requires_generation(self):
        manifest = {
            "num_nodes": 2,
            "transport": {"ready": True},
            "members": [
                {"world_rank": 0, "local_rank": 0, "ip": "192.0.2.10", "server_port": 9000},
                {"world_rank": 1, "local_rank": 1, "ip": "192.0.2.11", "server_port": 9000},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(manifest, file)
            file.flush()
            with patch.dict(os.environ, {"RTP_LLM_SCR_ENDPOINT_MANIFEST": file.name}):
                with self.assertRaises(ValueError):
                    resolve_world_info(world(), generation="g1", require_manifest=True)

    def test_cross_host_restore_requires_transport_provider_ready(self):
        manifest = {
            "generation": "g1",
            "num_nodes": 2,
            "members": [
                {"world_rank": 0, "local_rank": 0, "ip": "192.0.2.10", "server_port": 9000},
                {"world_rank": 1, "local_rank": 1, "ip": "192.0.2.11", "server_port": 9000},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump(manifest, file)
            file.flush()
            with patch.dict(os.environ, {"RTP_LLM_SCR_ENDPOINT_MANIFEST": file.name}):
                with self.assertRaises(RuntimeError):
                    resolve_world_info(
                        world(), generation="g1", require_manifest=True, require_transport=True
                    )

    def test_inline_manifest_does_not_get_treated_as_a_path(self):
        manifest = {
            "generation": "g1",
            "num_nodes": 2,
            "transport": {"ready": True},
            "members": [
                {
                    "world_rank": 0,
                    "local_rank": 0,
                    "ip": "192.0.2.10",
                    "server_port": 9000,
                },
                {
                    "world_rank": 1,
                    "local_rank": 1,
                    "ip": "192.0.2.11",
                    "server_port": 9000,
                },
            ],
        }
        with patch.dict(
            os.environ, {"RTP_LLM_SCR_ENDPOINT_MANIFEST": json.dumps(manifest)}
        ):
            restored = resolve_world_info(
                world(), generation="g1", require_manifest=True
            )
        self.assertEqual(restored.members[0].ip, "192.0.2.10")


if __name__ == "__main__":
    unittest.main()
