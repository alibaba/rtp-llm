import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
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
            1048568,
        )

    def test_glm_auto_enable_and_explicit_disable(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            model_args=SimpleNamespace(model_type="glm_5"),
        )
        with patch.object(
            start_server, "_role_is_prefill", return_value=True
        ), patch.object(
            start_server, "_is_startup_real_warmup_entry_rank", return_value=True
        ):
            with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "auto"}):
                self.assertTrue(start_server._should_run_startup_real_warmup(config))
            with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "off"}):
                self.assertFalse(start_server._should_run_startup_real_warmup(config))
            with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "force"}):
                config.runtime_config.model_warm_up = False
                self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_decode_and_non_entry_ranks_skip_warmup(self):
        config = SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            model_args=SimpleNamespace(model_type="glm_5"),
            parallelism_config=SimpleNamespace(world_rank=1, tp_size=8, world_size=8),
        )
        with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "force"}):
            with patch.object(start_server, "_role_is_prefill", return_value=False):
                self.assertFalse(start_server._should_run_startup_real_warmup(config))
            with patch.object(
                start_server, "_role_is_prefill", return_value=True
            ), patch.object(
                start_server, "_is_startup_real_warmup_entry_rank", return_value=False
            ):
                self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_lens_follow_model_limit_despite_legacy_environment(self):
        for max_len, expected in (
            (2, [2]),
            (64, [2, 4, 8, 16, 32, 64]),
            (70, [2, 4, 8, 16, 32, 64, 70]),
        ):
            config = SimpleNamespace(model_args=SimpleNamespace(max_seq_len=max_len))
            for value in ("2,7,64", "abc"):
                with self.subTest(max_len=max_len, value=value), patch.dict(
                    "os.environ", {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": value}
                ):
                    self.assertEqual(
                        start_server._get_startup_real_warmup_token_lens(config),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
