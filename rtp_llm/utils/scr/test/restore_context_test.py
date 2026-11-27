import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils.gang_info import GangInfoReader
from rtp_llm.utils.scr import network


class RestoreContextTest(unittest.TestCase):

    def test_single_pod_restore_changes_only_control_address(self):
        from rtp_llm.distribute.distributed_server import WorldInfo
        from rtp_llm.distribute.worker_info import WorkerInfo

        member = WorkerInfo("192.0.2.10", 0, 0, "seed", 9000, 9)
        world = WorldInfo([member], member, member, 1, True)
        pc = SimpleNamespace(world_size=1, local_world_size=1)
        with patch.dict(os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": "restore"}):
            restored = network.resolve_world_info(
                world, pc, GangInfoReader("/unused/annotations")
            )
        self.assertEqual(restored.self.ip, "127.0.0.1")
        self.assertEqual(world.self.ip, "192.0.2.10")
        self.assertIs(restored.self, restored.master)
        for attr in (
            "server_port",
            "rpc_server_port",
            "cache_store_listen_port",
            "cache_store_rdma_listen_port",
        ):
            self.assertEqual(getattr(restored.self, attr), getattr(world.self, attr))


if __name__ == "__main__":
    unittest.main()
