import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.distribute.distributed_server import get_dp_rpc_addresses


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
        return config

    def test_pp_selects_only_local_stage_zero_dp_roots(self):
        cases = [
            # Both stage-0 DP lanes are local.
            (
                self.config(dp=2, local_size=8, world_rank=4),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            # Stage 0 spans two nodes; each frontend keeps only its local lanes.
            (
                self.config(dp=4, local_size=4, world_rank=0),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            (
                self.config(dp=4, local_size=4, world_rank=4),
                ["127.0.0.1:8001", "127.0.0.1:8021"],
            ),
            # This node straddles stages: rank 4 serves, rank 6 does not.
            (
                self.config(dp=3, local_size=4, world_rank=4),
                ["127.0.0.1:8001"],
            ),
        ]
        for config, expected in cases:
            pc = config.parallelism_config
            with (
                self.subTest(dp=pc.dp_size, world_rank=pc.world_rank),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
                self.assertEqual(get_dp_rpc_addresses(config), expected)
                store.assert_not_called()

    def test_no_local_serving_rank_does_not_discover_remote_backends(self):
        for config in (
            self.config(),
            self.config(dp=2, local_size=4, world_rank=4),
            self.config(dp=4, local_size=4, world_rank=8),
            self.config(pp=1, dp=1, tp=4, local_size=2, world_rank=2),
        ):
            pc = config.parallelism_config
            with (
                self.subTest(pp=pc.pp_size, dp=pc.dp_size, world_rank=pc.world_rank),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
                self.assertEqual(get_dp_rpc_addresses(config), [])
                store.assert_not_called()

    def test_non_pp_keeps_local_dp_addresses(self):
        for config in (
            self.config(pp=1, dp=4, local_size=4, world_rank=0),
            self.config(pp=1, dp=4, local_size=4, world_rank=4),
        ):
            with (
                self.subTest(pp=config.parallelism_config.pp_size),
                patch(
                    "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                    return_value="127.0.0.1",
                ),
                patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            ):
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
        with (
            patch(
                "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                return_value="127.0.0.1",
            ),
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
        ):
            self.assertEqual(
                get_dp_rpc_addresses(config),
                ["127.0.0.1:8001", "127.0.0.1:8011"],
            )
            store.assert_not_called()

    def test_warmup_without_local_serving_rank_keeps_resolution_error(self):
        from rtp_llm.start_server import (
            StartupRealWarmupAddressResolutionError,
            _get_startup_real_warmup_grpc_addresses,
        )

        with (
            patch(
                "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                return_value="127.0.0.1",
            ),
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            self.assertRaises(StartupRealWarmupAddressResolutionError),
        ):
            _get_startup_real_warmup_grpc_addresses(self.config())
        store.assert_not_called()

    def test_serving_clients_keep_local_stage_zero_addresses(self):
        from rtp_llm.frontend.frontend_app import FrontendApp
        from rtp_llm.frontend.frontend_worker import FrontendWorker
        from rtp_llm.server.backend_rpc_server_visitor import (
            create_backend_rpc_server_visitor,
        )
        from rtp_llm.start_server import _get_startup_real_warmup_grpc_addresses

        config = self.config(dp=4, local_size=4, world_rank=4)
        model_config = SimpleNamespace(
            ckpt_path="unused",
            tokenizer_path="unused",
            model_type="unused",
            max_seq_len=1024,
            attn_config=SimpleNamespace(tokens_per_block=16),
        )
        expected = ["127.0.0.1:8001", "127.0.0.1:8021"]
        with (
            patch(
                "rtp_llm.distribute.distributed_server.socket.gethostbyname",
                return_value="127.0.0.1",
            ),
            patch("rtp_llm.distribute.distributed_server.TCPStore") as store,
            patch("rtp_llm.frontend.frontend_worker.TokenizerFactory.create"),
            patch("rtp_llm.frontend.frontend_worker.Pipeline") as pipeline,
            patch("rtp_llm.frontend.frontend_app.FrontendServer"),
            patch(
                "rtp_llm.frontend.frontend_app.get_eplb_stage_root_addresses",
                return_value=["stage0:8001", "stage1:8001"],
            ) as resolve_roots,
            patch(
                "rtp_llm.server.backend_rpc_server_visitor.BackendRPCServerVisitor"
            ) as visitor,
        ):
            FrontendWorker(config, model_config, special_tokens=None)
            self.assertEqual(pipeline.call_args.kwargs["addresses"], expected)
            for pp_size in (1, 2):
                with self.subTest(pp=pp_size):
                    app_config = self.config(
                        pp=pp_size, dp=4, local_size=4, world_rank=4
                    )
                    client = FrontendApp(app_config).grpc_client
                    self.assertEqual(client.dp_addresses, expected)
                    resolve_roots.assert_not_called()
                    if pp_size == 1:
                        self.assertIsNone(client._eplb_address_resolver)
                    else:
                        self.assertEqual(
                            client._eplb_address_resolver(), resolve_roots.return_value
                        )
                        resolve_roots.assert_called_once_with(app_config)
            create_backend_rpc_server_visitor(config, model_config)
            self.assertEqual(visitor.call_args.kwargs["addresses"], expected)
            self.assertEqual(_get_startup_real_warmup_grpc_addresses(config), expected)
            store.assert_not_called()


if __name__ == "__main__":
    unittest.main()
