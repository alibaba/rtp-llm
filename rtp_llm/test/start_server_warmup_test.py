import multiprocessing
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from rtp_llm import start_backend_server, start_server
from rtp_llm.ops import RoleType, SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
    def test_backend_ranks_report_resolved_model_length(self):
        from unittest.mock import Mock

        readers = []
        for _ in range(2):
            reader, writer = multiprocessing.Pipe(duplex=False)
            self.addCleanup(reader.close)
            readers.append(reader)
            start_backend_server._send_pipe_status(
                writer, "success", "ready", max_seq_len=4096
            )
        process = Mock(exitcode=None)
        process.is_alive.return_value = True
        self.assertEqual(
            start_backend_server._wait_for_ranks_startup(
                [process, process], readers, 2
            ),
            4096,
        )

    def test_warmup_uses_resolved_model_length(self):
        config = SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=None),
            sp_config=SimpleNamespace(type=SpeculativeType.NONE, gen_num_per_cycle=0),
            grpc_config=None,
        )
        requests = []

        async def enqueue(request):
            requests.append(request)
            yield SimpleNamespace(generate_outputs=[])

        client = SimpleNamespace(enqueue=enqueue, close=AsyncMock())
        with patch(
            "rtp_llm.model_factory.ModelFactory.create_model_config",
            side_effect=AssertionError("warmup must reuse the backend model length"),
        ), patch(
            "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient", return_value=client
        ), patch.object(
            start_server,
            "_get_startup_real_warmup_grpc_addresses",
            return_value=["localhost:12345"],
        ):
            for cli_length in (None, 8192):
                with self.subTest(cli_length=cli_length):
                    config.model_args.max_seq_len = cli_length
                    requests.clear()
                    start_server._run_startup_real_warmup_async(
                        start_server._run_startup_real_warmup_grpc(config, 4096)
                    )
                    self.assertEqual(
                        [request.token_ids.numel() for request in requests],
                        [2**power for power in range(1, 12)] + [4095],
                    )
        self.assertEqual(client.close.await_count, 2)

    def test_warmup_rejects_invalid_resolved_length(self):
        for length in (0, -1):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "should be at least"):
                    start_server._run_startup_real_warmup_async(
                        start_server._run_startup_real_warmup_grpc(None, length)
                    )

    def test_single_stage_dp_ranks_pass_startup_warmup_gate(self):
        for world_size in (2, 3, 4):
            for world_rank in range(world_size):
                with self.subTest(world_size=world_size, world_rank=world_rank):
                    config = SimpleNamespace(
                        parallelism_config=SimpleNamespace(
                            pp_size=1,
                            dp_size=world_size,
                            tp_size=1,
                            world_size=world_size,
                            world_rank=world_rank,
                        ),
                        runtime_config=SimpleNamespace(
                            warm_up=True, model_warm_up=True
                        ),
                        role_config=SimpleNamespace(role_type=RoleType.PREFILL),
                        model_args=SimpleNamespace(model_type="qwen_3"),
                    )
                    self.assertTrue(
                        start_server._is_startup_real_warmup_entry_rank(config)
                    )
                    self.assertFalse(
                        start_server._should_run_startup_real_warmup(config)
                    )

    def test_warmup_entry_rank_preserves_pp_and_tp_boundaries(self):
        for pp_size, world_rank, expected in (
            (1, 0, True),
            (1, 1, False),
            (1, 2, True),
            (2, 0, True),
            (2, 2, True),
            (2, 4, False),
            (2, 5, False),
            (2, 6, False),
        ):
            with self.subTest(pp_size=pp_size, world_rank=world_rank):
                config = SimpleNamespace(
                    parallelism_config=SimpleNamespace(
                        pp_size=pp_size,
                        dp_size=2,
                        tp_size=2,
                        world_size=pp_size * 4,
                        world_rank=world_rank,
                    )
                )
                self.assertEqual(
                    start_server._is_startup_real_warmup_entry_rank(config),
                    expected,
                )

    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    def test_speculative_reserve_matches_engine(self):
        mtp = self._configs(SpeculativeType.MTP, 3)
        dspark = self._configs(SpeculativeType.DSPARK, 3)

        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "0"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                4,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(mtp),
                7,
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                9,
            )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048567,
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=4,
            ),
            1048571,
        )

    def test_k3_prefill_entry_only_and_both_switches_required(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(
                world_rank=0, world_size=8, pp_size=1, dp_size=2, tp_size=4
            ),
            model_args=SimpleNamespace(model_type="kimi_k3"),
        )
        self.assertTrue(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 1
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.parallelism_config.world_rank = 0
        config.role_config.role_type = RoleType.DECODE
        self.assertFalse(start_server._should_run_startup_real_warmup(config))
        config.role_config.role_type = RoleType.PREFILL
        config.runtime_config.model_warm_up = False
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_real_warmup_only_enables_dsv4_and_k3_family_targets(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=1, tp_size=1),
            model_args=SimpleNamespace(model_type="deepseek_v4"),
        )
        for model_type, expected in (
            ("deepseek_v4", True),
            ("kimi_k3", True),
            ("kimi_k31", True),
            ("deepseek_v4_mtp", False),
            ("kimi_k3_mtp", False),
            ("kimi_k31_mtp", False),
            ("custom_model", False),
            ("", False),
        ):
            with self.subTest(model_type=model_type):
                config.model_args.model_type = model_type
                self.assertEqual(
                    start_server._should_run_startup_real_warmup(config), expected
                )


if __name__ == "__main__":
    unittest.main()
