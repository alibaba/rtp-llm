import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.utils.scr_restore_context import RestoreContext


class RestoreContextTest(unittest.TestCase):
    def test_changed_local_listener_ports_fail_before_endpoint_publication(self):
        from rtp_llm.distribute.distributed_server import WorldInfo
        from rtp_llm.distribute.worker_info import WorkerInfo

        member = WorkerInfo("192.0.2.10", 0, 0, "seed", 9000, 9)
        world = WorldInfo([member], member, member, 1, True)
        manifest = {
            "generation": "seed",
            "phase": "restore",
            "transport": {"ready": True},
            "members": [
                {
                    "world_rank": 0,
                    "local_rank": 0,
                    "ip": "192.0.2.20",
                    "server_port": 19000,
                }
            ],
        }
        with patch.dict(os.environ, {"SCR_PHASE": "restore"}), patch(
            "rtp_llm.utils.scr_endpoint_provider.read_restore_manifest",
            return_value=manifest,
        ):
            with self.assertRaisesRegex(RuntimeError, "local listener port layout"):
                RestoreContext("seed", "192.0.2.20").resolve_world_info(
                    world, SimpleNamespace(world_size=1)
                )
        self.assertEqual(world.self.ip, "192.0.2.10")
        self.assertEqual(world.self.cache_store_listen_port, 9002)

    def test_manifest_is_read_once_per_attempt_even_for_same_seed(self):
        first = {"generation": "seed", "phase": "restore"}
        second = {"generation": "seed", "phase": "checkpoint"}
        reader = Mock(side_effect=[first, second])
        with patch("rtp_llm.utils.scr_endpoint_provider.read_restore_manifest", reader):
            for expected in (first, second):
                context = RestoreContext("seed", "192.0.2.20")
                self.assertIs(context.endpoint_manifest, expected)
                self.assertIs(context.endpoint_manifest, expected)
        self.assertEqual(reader.call_count, 2)

    def test_absent_manifest_is_shared_without_a_second_provider_read(self):
        world = SimpleNamespace(num_nodes=1, members=[])
        pc = SimpleNamespace(world_size=1)
        with patch(
            "rtp_llm.utils.scr_endpoint_provider.read_restore_manifest",
            return_value=None,
        ) as reader, patch(
            "rtp_llm.utils.scr_endpoint_provider.is_restore_phase", return_value=False
        ), patch(
            "rtp_llm.utils.scr_endpoint_provider.local_comm_enabled", return_value=False
        ):
            context = RestoreContext("seed", "192.0.2.20")
            self.assertIs(context.resolve_world_info(world, pc), world)
            self.assertIs(context.resolve_world_info(world, pc), world)
        reader.assert_called_once_with("seed")


if __name__ == "__main__":
    unittest.main()
