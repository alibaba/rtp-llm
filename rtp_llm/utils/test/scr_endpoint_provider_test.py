import json
import os
import tempfile
import unittest
from unittest.mock import patch

from rtp_llm.distribute.distributed_server import WorldInfo
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.utils.scr_endpoint_provider import resolve_world_info


def world():
    members = [
        WorkerInfo("10.0.0.%d" % (i + 1), i, i, "rank_%d" % i, 8088, 9)
        for i in range(2)
    ]
    return WorldInfo(members, members[0], members[0], 2, True)


class ScrEndpointProviderTest(unittest.TestCase):
    def test_local_template_can_rebind_to_loopback_explicitly(self):
        current = world()
        current.num_nodes = 1
        current.members = current.members[:1]
        current.self = current.members[0]
        current.master = current.members[0]
        with patch.dict(os.environ, {"RTP_LLM_SCR_LOCAL_COMM": "1"}, clear=False):
            restored = resolve_world_info(current, generation="g1")
        self.assertEqual(restored.members[0].ip, "127.0.0.1")

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


if __name__ == "__main__":
    unittest.main()
