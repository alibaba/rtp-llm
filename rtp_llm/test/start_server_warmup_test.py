import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
    def setUp(self):
        # Import bindings normally, but fail rather than accidentally create a
        # CUDA context from a CPU-only startup-policy test.
        no_cuda = patch.object(
            torch.cuda, "_lazy_init", side_effect=AssertionError("GPU in CPU test")
        )
        no_cuda.start()
        self.addCleanup(no_cuda.stop)

    @staticmethod
    def _configs(sp_type, gamma):
        return SimpleNamespace(
            sp_config=SimpleNamespace(
                type=sp_type,
                gen_num_per_cycle=gamma,
            )
        )

    @staticmethod
    def _startup_config(model_max=32, token_budget=0):
        return SimpleNamespace(
            model_args=SimpleNamespace(
                model_type="deepseek_v41", max_seq_len=model_max
            ),
            runtime_config=SimpleNamespace(
                warm_up=True,
                model_warm_up=True,
                max_generate_batch_size=16,
                fifo_scheduler_config=SimpleNamespace(
                    max_batch_tokens_size=token_budget
                ),
            ),
            concurrency_config=SimpleNamespace(concurrency_limit=16),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=4, tp_size=4),
            grpc_config=None,
        )

    def test_v4_and_v41_preserve_role_rank_and_both_warmup_gates(self):
        for model_type in ("deepseek_v4", "deepseek_v41"):
            config = self._startup_config()
            config.model_args.model_type = model_type
            with self.subTest(model_type=model_type):
                self.assertTrue(start_server._should_run_startup_real_warmup(config))
                for flag in ("warm_up", "model_warm_up"):
                    with patch.object(config.runtime_config, flag, False):
                        self.assertFalse(
                            start_server._should_run_startup_real_warmup(config)
                        )
                with patch.object(config.role_config, "role_type", RoleType.DECODE):
                    self.assertFalse(
                        start_server._should_run_startup_real_warmup(config)
                    )
                with patch.object(config.parallelism_config, "world_rank", 1):
                    self.assertFalse(
                        start_server._should_run_startup_real_warmup(config)
                    )
                with patch.object(
                    config.parallelism_config, "world_size", 8
                ), patch.object(config.parallelism_config, "world_rank", 4):
                    self.assertTrue(
                        start_server._should_run_startup_real_warmup(config)
                    )
        config.model_args.model_type = "qwen_3"
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_token_budget_caps_inputs_and_retains_non_power_of_two_endpoint(self):
        cases = (
            (17, 1, [1]),
            (17, 2, [2]),
            (17, 8, [2, 4, 8]),
            (17, 9, [2, 4, 8, 9]),
            (17, 17, [2, 4, 8, 16, 17]),
            (17, 128, [2, 4, 8, 16, 17]),
            (17, 0, [2, 4, 8, 16, 17]),
            (17, -1, [2, 4, 8, 16, 17]),
            (17, None, [2, 4, 8, 16, 17]),
        )
        for model_max, budget, expected in cases:
            with self.subTest(model_max=model_max, budget=budget):
                config = self._startup_config(model_max, budget)
                self.assertEqual(
                    start_server._get_startup_real_warmup_token_lens(config), expected
                )
                self.assertEqual(
                    start_server._get_startup_real_warmup_max_len(config), model_max
                )

    def test_missing_scheduler_or_budget_keeps_model_limit(self):
        config = self._startup_config(model_max=16)
        del config.runtime_config.fifo_scheduler_config.max_batch_tokens_size
        self.assertEqual(
            start_server._get_startup_real_warmup_token_lens(config), [2, 4, 8, 16]
        )
        del config.runtime_config.fifo_scheduler_config
        self.assertEqual(
            start_server._get_startup_real_warmup_token_lens(config), [2, 4, 8, 16]
        )

    def test_invalid_model_limit_is_not_hidden_by_scheduler_budget(self):
        for model_max in (None, 0, -1):
            with self.subTest(model_max=model_max), self.assertRaisesRegex(
                ValueError, "max_seq_len should be positive"
            ):
                start_server._get_startup_real_warmup_token_lens(
                    self._startup_config(model_max=model_max, token_budget=4096)
                )

    def test_reserve_reduces_model_limit_not_input_budget(self):
        config = self._startup_config(model_max=64, token_budget=8)
        config.sp_config = self._configs(SpeculativeType.DSPARK, 5).sp_config
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            reserve = start_server._get_startup_real_warmup_speculative_reserve_step(
                config
            )
        self.assertEqual(reserve, 11)
        model_max = start_server._get_startup_real_warmup_max_len(config)
        token_lens = start_server._get_startup_real_warmup_token_lens(config)
        self.assertEqual(
            [
                start_server._get_startup_real_warmup_request_token_len(
                    tokens, model_max, reserve
                )
                for tokens in token_lens
            ],
            [2, 4, 8],
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                64, model_max, reserve
            ),
            53,
        )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(64, model_max, 0),
            63,
        )

    def test_full_forward_requests_stay_serial_and_do_not_expand_with_concurrency(self):
        for concurrency in (1, 16):
            with self.subTest(concurrency=concurrency):
                config = self._startup_config(model_max=64, token_budget=8)
                config.runtime_config.max_generate_batch_size = concurrency
                config.concurrency_config.concurrency_limit = concurrency
                config.sp_config = self._configs(SpeculativeType.DSPARK, 5).sp_config
                requests = []
                active = 0
                peak_active = 0
                closed = []

                class Client:
                    def __init__(self, **kwargs):
                        self.addresses = kwargs["addresses"]

                    async def enqueue(self, request):
                        nonlocal active, peak_active
                        active += 1
                        peak_active = max(peak_active, active)
                        requests.append(request)
                        try:
                            await asyncio.sleep(0)
                            yield SimpleNamespace(generate_outputs=[])
                        finally:
                            active -= 1

                    async def close(self):
                        closed.append(self.addresses)

                with patch(
                    "rtp_llm.cpp.model_rpc.model_rpc_client.ModelRpcClient", Client
                ), patch.object(
                    start_server,
                    "_get_startup_real_warmup_grpc_addresses",
                    return_value=["127.0.0.1:1"],
                ), patch.dict(
                    "os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}
                ):
                    asyncio.run(start_server._run_startup_real_warmup_grpc(config))
                self.assertEqual(peak_active, 1)
                self.assertEqual(active, 0)
                self.assertEqual(
                    [request.token_ids.numel() for request in requests], [2, 4, 8]
                )
                self.assertEqual(closed, [["127.0.0.1:1"]])
                for request in requests:
                    self.assertEqual(request.token_ids.device.type, "cpu")
                    generation = request.generate_config
                    self.assertEqual(generation.max_new_tokens, 1)
                    self.assertEqual(generation.top_k, 1)
                    self.assertEqual(generation.top_p, 1.0)
                    self.assertEqual(generation.temperature, 0.0)
                    self.assertTrue(generation.do_sample)
                    self.assertFalse(generation.can_use_pd_separation)
                    self.assertFalse(generation.reuse_cache)
                    self.assertFalse(generation.enable_device_cache)
                    self.assertFalse(generation.enable_memory_cache)
                    self.assertFalse(generation.enable_remote_cache)

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
                4,
            )
        with patch.dict("os.environ", {"RTP_LLM_STREAM_ASYNC": "1"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_speculative_reserve_step(dspark),
                7,
            )
        self.assertEqual(
            start_server._get_startup_real_warmup_request_token_len(
                token_len=1048576,
                max_len=1048576,
                reserve_step=8,
            ),
            1048568,
        )


if __name__ == "__main__":
    unittest.main()
