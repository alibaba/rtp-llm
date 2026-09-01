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
            misc_config=SimpleNamespace(dsv4_startup_real_warmup=True),
        )

    def test_startup_real_warmup_uses_parsed_config(self):
        config = self._warmup_config()
        self.assertTrue(start_server._should_run_startup_real_warmup(config))
        config.misc_config.dsv4_startup_real_warmup = False
        self.assertFalse(start_server._should_run_startup_real_warmup(config))

    def test_irrelevant_roles_ignore_malformed_warmup_value(self):
        cases = (
            (RoleType.FRONTEND, "deepseek_v4"),
            (RoleType.PREFILL, "qwen_2"),
        )
        for role_type, model_type in cases:
            with self.subTest(role_type=role_type, model_type=model_type):
                config = self._warmup_config()
                config.role_config.role_type = role_type
                config.model_args.model_type = model_type
                self.assertFalse(start_server._should_run_startup_real_warmup(config))


if __name__ == "__main__":
    unittest.main()
