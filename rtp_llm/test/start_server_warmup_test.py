import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server
from rtp_llm.ops import RoleType, SpeculativeType


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

    @staticmethod
    def _warmup_config():
        return SimpleNamespace(
            runtime_config=SimpleNamespace(warm_up=True, model_warm_up=True),
            role_config=SimpleNamespace(role_type=RoleType.PREFILL),
            parallelism_config=SimpleNamespace(world_rank=0, world_size=1, tp_size=1),
            model_args=SimpleNamespace(model_type="deepseek_v4"),
        )

    def test_startup_real_warmup_bool_env_is_strict(self):
        config = self._warmup_config()
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(start_server._should_run_startup_real_warmup(config))
        with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "0"}, clear=True):
            self.assertFalse(start_server._should_run_startup_real_warmup(config))
        with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "off"}, clear=True):
            self.assertFalse(start_server._should_run_startup_real_warmup(config))
        with patch.dict("os.environ", {"DSV4_STARTUP_REAL_WARMUP": "on"}, clear=True):
            self.assertTrue(start_server._should_run_startup_real_warmup(config))
        with patch.dict(
            "os.environ", {"DSV4_STARTUP_REAL_WARMUP": "maybe"}, clear=True
        ):
            with self.assertRaises(ValueError):
                start_server._should_run_startup_real_warmup(config)


if __name__ == "__main__":
    unittest.main()
