import unittest
from types import SimpleNamespace

from rtp_llm.config.engine_config import update_worker_addrs
from rtp_llm.ops import ParallelismConfig, RuntimeConfig


class WorldWorkerAddrsTest(unittest.TestCase):
    def test_world_members_include_all_dp_ranks_in_world_order(self):
        parallelism = ParallelismConfig()
        parallelism.dp_size = 16
        parallelism.tp_size = 1
        parallelism.dp_rank = 3
        runtime = RuntimeConfig()
        members = [
            SimpleNamespace(
                world_rank=rank,
                ip=f"host{rank // 4}",
                rpc_server_port=8001 + (rank % 4) * 10,
                cache_store_listen_port=8002 + (rank % 4) * 10,
                cache_store_rdma_listen_port=8003 + (rank % 4) * 10,
            )
            for rank in range(16)
        ]
        update_worker_addrs(
            runtime, parallelism, SimpleNamespace(members=members[::-1])
        )
        self.assertEqual(runtime.worker_grpc_addrs, ["host0:8031"])
        self.assertEqual(
            runtime.world_worker_grpc_addrs,
            [f"{member.ip}:{member.rpc_server_port}" for member in members],
        )

    def test_world_members_include_other_tp_groups(self):
        parallelism = ParallelismConfig()
        parallelism.dp_size = 2
        parallelism.tp_size = 2
        parallelism.dp_rank = 1
        runtime = RuntimeConfig()
        members = [
            SimpleNamespace(
                world_rank=rank,
                ip=f"host{rank // 2}",
                rpc_server_port=8001 + (rank % 2) * 10,
                cache_store_listen_port=8002 + (rank % 2) * 10,
                cache_store_rdma_listen_port=8003 + (rank % 2) * 10,
            )
            for rank in range(4)
        ]
        update_worker_addrs(
            runtime, parallelism, SimpleNamespace(members=members[::-1])
        )
        self.assertEqual(runtime.worker_grpc_addrs, ["host1:8011", "host1:8001"])
        self.assertEqual(
            runtime.world_worker_grpc_addrs,
            ["host0:8001", "host0:8011", "host1:8001", "host1:8011"],
        )


if __name__ == "__main__":
    unittest.main()
