from types import SimpleNamespace
from unittest import TestCase, main

from rtp_llm.config.engine_config import update_worker_addrs
from rtp_llm.ops import ParallelismConfig, RuntimeConfig


class EngineConfigTest(TestCase):
    def _parallelism_config(self) -> ParallelismConfig:
        config = ParallelismConfig()
        config.local_rank = 0
        config.tp_size = 2
        config.dp_size = 1
        config.dp_rank = 0
        return config

    def _world_info(self):
        members = [
            SimpleNamespace(
                world_rank=0,
                ip="127.0.0.1",
                cache_store_listen_port=12000,
                cache_store_rdma_listen_port=12100,
                rpc_server_port=13000,
            ),
            SimpleNamespace(
                world_rank=1,
                ip="127.0.0.2",
                cache_store_listen_port=12010,
                cache_store_rdma_listen_port=12110,
                rpc_server_port=13010,
            ),
        ]
        return SimpleNamespace(members=members)

    def test_update_worker_addrs_keeps_legacy_cache_store_format(self):
        runtime_config = RuntimeConfig()

        update_worker_addrs(
            runtime_config,
            self._parallelism_config(),
            self._world_info(),
            decode_entrance=False,
        )

        self.assertEqual(
            runtime_config.worker_addrs,
            [
                "127.0.0.1:12000:12100",
                "127.0.0.2:12010:12110",
            ],
        )
        self.assertEqual(
            runtime_config.worker_grpc_addrs,
            [
                "127.0.0.1:13000",
                "127.0.0.2:13010",
            ],
        )
        self.assertEqual(runtime_config.p2p_worker_addrs, [])

    def test_update_worker_addrs_uses_p2p_format_for_decode_entrance(self):
        runtime_config = RuntimeConfig()

        update_worker_addrs(
            runtime_config,
            self._parallelism_config(),
            self._world_info(),
            decode_entrance=True,
        )

        self.assertEqual(
            runtime_config.worker_addrs,
            [
                "127.0.0.1:12001:13000",
                "127.0.0.2:12011:13010",
            ],
        )
        self.assertEqual(
            runtime_config.worker_grpc_addrs,
            [
                "127.0.0.1:13000",
                "127.0.0.2:13010",
            ],
        )
        self.assertEqual(
            runtime_config.p2p_worker_addrs,
            [
                "127.0.0.1:12001:13000",
                "127.0.0.2:12011:13010",
            ],
        )


if __name__ == "__main__":
    main()
