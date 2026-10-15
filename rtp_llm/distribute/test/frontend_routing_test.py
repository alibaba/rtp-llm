import unittest
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

from torch.distributed import TCPStore

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    get_dp_rpc_addresses,
)
from rtp_llm.test.utils.port_util import PortsContext


class FrontendRoutingTest(unittest.TestCase):
    def config(self, *, pp=2, dp=1, tp=2, local_size=2, world_rank=2):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.pp_size, pc.dp_size, pc.tp_size = pp, dp, tp
        pc.ep_size = 0
        pc.world_size = pp * dp * tp
        pc.local_world_size = local_size
        set_parallelism_config(pc, world_rank)
        config.server_config.start_port = 8000
        config.server_config.worker_info_port_num = 10
        config.distribute_config.dist_comm_timeout = 5
        return config

    def test_remote_stage_zero_uses_backend_registry(self):
        cases = [
            # The original two-node PP=2, TP=2 failure: no local stage-0 rank.
            (self.config(), ["198.51.100.1:8001"]),
            # Two DP lanes on the remote node; rank 2's port is already offset.
            (
                self.config(dp=2, local_size=4, world_rank=4),
                ["198.51.100.1:8001", "198.51.100.1:8021"],
            ),
            # This node straddles stages: rank 4 serves, rank 6 does not.
            (
                self.config(dp=3, local_size=4, world_rank=4),
                [
                    "198.51.100.1:8001",
                    "198.51.100.1:8021",
                    "127.0.0.1:8001",
                ],
            ),
        ]
        with PortsContext(num_ports=1) as ports:
            store = TCPStore(
                host_name="127.0.0.1",
                port=ports[0],
                is_master=True,
                wait_for_workers=False,
                timeout=timedelta(seconds=5),
            )
            try:
                store.set(
                    DistributedServer.REGISTRY_RANK_ADDRESS_KEY + "0",
                    "198.51.100.1:8000",
                )
                store.set(
                    DistributedServer.REGISTRY_RANK_ADDRESS_KEY + "2",
                    "198.51.100.1:8020",
                )
                for config, expected in cases:
                    config.distribute_config.gang_config_string = (
                        f"name:test_part0,ip:127.0.0.1,port:{ports[0] + 1}"
                    )
                    with self.subTest(dp=config.parallelism_config.dp_size), patch(
                        "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                        return_value="127.0.0.1",
                    ):
                        self.assertEqual(get_dp_rpc_addresses(config), expected)
            finally:
                del store

    def test_local_stage_zero_and_non_pp_do_not_need_registry(self):
        for config in (
            self.config(dp=2, local_size=8, world_rank=4),
            self.config(pp=1, dp=4, local_size=4, world_rank=4),
        ):
            with self.subTest(pp=config.parallelism_config.pp_size), patch(
                "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                return_value="127.0.0.1",
            ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
                self.assertEqual(
                    get_dp_rpc_addresses(config),
                    ["127.0.0.1:8001", "127.0.0.1:8021"],
                )
                store.assert_not_called()

    def test_non_pp_ffn_disaggregate_keeps_existing_serving_ranks(self):
        config = self.config(pp=1, dp=1, tp=1, local_size=3, world_rank=0)
        config.parallelism_config.world_size = 3
        ffn = config.parallelism_config.ffn_disaggregate_config
        ffn.enable_ffn_disaggregate = True
        ffn.attention_tp_size = 1
        ffn.attention_dp_size = 2
        with patch(
            "rtp_llm.distribute.distributed_server.socket.gethostbyname",
            return_value="127.0.0.1",
        ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
            self.assertEqual(
                get_dp_rpc_addresses(config),
                ["127.0.0.1:8001", "127.0.0.1:8011"],
            )
            store.assert_not_called()

    def test_registry_failure_propagates_with_backend_startup_timeout(self):
        for timeout_s, expected in ((None, None), (37, timedelta(seconds=37))):
            config = self.config()
            config.distribute_config.dist_comm_timeout = timeout_s
            with self.subTest(timeout=timeout_s), patch(
                "rtp_llm.distribute.distributed_server.get_master",
                return_value=("127.0.0.1", 8000),
            ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
                store.return_value.get.side_effect = RuntimeError("rank not registered")
                with self.assertRaisesRegex(RuntimeError, "rank not registered"):
                    get_dp_rpc_addresses(config)
                self.assertEqual(store.call_args.kwargs["timeout"], expected)
                self.assertFalse(store.call_args.kwargs["is_master"])
                self.assertFalse(store.call_args.kwargs["wait_for_workers"])
                self.assertIsNone(store.call_args.kwargs.get("world_size"))

    def test_serving_clients_use_remote_stage_zero(self):
        from rtp_llm.frontend.frontend_app import FrontendApp
        from rtp_llm.frontend.frontend_worker import FrontendWorker
        from rtp_llm.server.backend_rpc_server_visitor import (
            create_backend_rpc_server_visitor,
        )
        from rtp_llm.start_server import _get_startup_real_warmup_grpc_addresses

        config = self.config()
        model_config = SimpleNamespace(
            ckpt_path="unused",
            tokenizer_path="unused",
            model_type="unused",
            max_seq_len=1024,
            attn_config=SimpleNamespace(tokens_per_block=16),
        )
        expected = ["198.51.100.1:8001"]
        with (
            patch(
                "rtp_llm.distribute.distributed_server.get_master",
                return_value=("127.0.0.1", 8000),
            ),
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            patch("rtp_llm.frontend.frontend_worker.TokenizerFactory.create"),
            patch("rtp_llm.frontend.frontend_worker.Pipeline") as pipeline,
            patch("rtp_llm.frontend.frontend_app.FrontendServer"),
            patch(
                "rtp_llm.server.backend_rpc_server_visitor.BackendRPCServerVisitor"
            ) as visitor,
        ):
            store.return_value.get.return_value = b"198.51.100.1:8000"
            FrontendWorker(config, model_config, special_tokens=None)
            self.assertEqual(pipeline.call_args.kwargs["addresses"], expected)
            app = FrontendApp(config)
            self.assertEqual(app.grpc_client.dp_addresses, expected)
            create_backend_rpc_server_visitor(config, model_config)
            self.assertEqual(visitor.call_args.kwargs["addresses"], expected)
            self.assertEqual(_get_startup_real_warmup_grpc_addresses(config), expected)


if __name__ == "__main__":
    unittest.main()
