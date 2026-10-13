import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import SpeculativeType


class StartupRealWarmupTest(unittest.TestCase):
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

    def test_configurable_lens_and_validation(self):
        config = SimpleNamespace(model_args=SimpleNamespace(max_seq_len=64))
        with patch.dict(
            "os.environ", {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": "2, 7,64"}
        ):
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(config), [2, 7, 64]
            )
        for value in ("1", "65", "2,,8", "abc", "-1"):
            with self.subTest(value=value), patch.dict(
                "os.environ", {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": value}
            ):
                with self.assertRaises(ValueError):
                    start_server._get_startup_real_warmup_token_lens(config)
        with patch.dict("os.environ", {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": ""}):
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(config),
                [2, 4, 8, 16, 32, 64],
            )


if __name__ == "__main__":
    unittest.main()
