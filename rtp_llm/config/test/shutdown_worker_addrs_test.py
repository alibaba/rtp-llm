import unittest
from types import SimpleNamespace

from rtp_llm.config.engine_config import update_worker_addrs
from rtp_llm.ops import ParallelismConfig, RuntimeConfig


class ShutdownWorkerAddrsTest(unittest.TestCase):
    def test_shutdown_members_include_all_dp_ranks_in_world_order(self):
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
            runtime.shutdown_worker_grpc_addrs,
            [f"{member.ip}:{member.rpc_server_port}" for member in members],
        )


if __name__ == "__main__":
    unittest.main()
