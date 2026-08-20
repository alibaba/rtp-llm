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

    @staticmethod
    def _py_env_configs(
        role_type=RoleType.PDFUSION,
        model_type="qwen_3_moe",
        max_seq_len=8192,
        warm_up=True,
        model_warm_up=True,
        world_size=1,
    ):
        return SimpleNamespace(
            runtime_config=SimpleNamespace(
                warm_up=warm_up,
                model_warm_up=model_warm_up,
            ),
            role_config=SimpleNamespace(role_type=role_type),
            parallelism_config=SimpleNamespace(
                world_rank=0,
                world_size=world_size,
                tp_size=1,
            ),
            model_args=SimpleNamespace(
                model_type=model_type,
                max_seq_len=max_seq_len,
            ),
        )

    def test_gate_covers_pdfusion_and_prefill_roles(self):
        for role_type in (RoleType.PDFUSION, RoleType.PREFILL):
            configs = self._py_env_configs(role_type=role_type)
            self.assertTrue(
                start_server._should_run_startup_real_warmup(configs),
                f"role_type={role_type} should run startup real warmup",
            )

    def test_gate_skips_decode_only_role(self):
        configs = self._py_env_configs(role_type=RoleType.DECODE)
        self.assertFalse(start_server._should_run_startup_real_warmup(configs))

    def test_gate_model_type_allowlist(self):
        for model_type in (
            "deepseek_v4",
            "qwen_3",
            "qwen_3_tool",
            "qwen_3_moe",
            "qwen_3_moe_eagle3",
            "qwen3_next",
        ):
            configs = self._py_env_configs(model_type=model_type)
            self.assertTrue(
                start_server._should_run_startup_real_warmup(configs),
                f"model_type={model_type} should run startup real warmup",
            )
        configs = self._py_env_configs(model_type="qwen_2")
        self.assertFalse(start_server._should_run_startup_real_warmup(configs))

    def test_gate_env_override(self):
        configs = self._py_env_configs(model_type="qwen_2")
        with patch.dict("os.environ", {"STARTUP_REAL_WARMUP": "1"}):
            self.assertTrue(start_server._should_run_startup_real_warmup(configs))
        configs = self._py_env_configs(model_type="qwen_3_moe")
        with patch.dict("os.environ", {"STARTUP_REAL_WARMUP": "0"}):
            self.assertFalse(start_server._should_run_startup_real_warmup(configs))

    def test_gate_respects_warmup_switches(self):
        configs = self._py_env_configs(warm_up=False)
        self.assertFalse(start_server._should_run_startup_real_warmup(configs))
        configs = self._py_env_configs(model_warm_up=False)
        self.assertFalse(start_server._should_run_startup_real_warmup(configs))

    def test_max_len_defaults_to_model_max_seq_len(self):
        configs = self._py_env_configs(max_seq_len=81921)
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("STARTUP_REAL_WARMUP_MAX_TOKEN_LEN", None)
            self.assertEqual(
                start_server._get_startup_real_warmup_max_len(configs), 81921
            )

    def test_max_len_env_caps_warmup_len(self):
        configs = self._py_env_configs(max_seq_len=81921)
        with patch.dict("os.environ", {"STARTUP_REAL_WARMUP_MAX_TOKEN_LEN": "4096"}):
            self.assertEqual(
                start_server._get_startup_real_warmup_max_len(configs), 4096
            )
            self.assertEqual(
                start_server._get_startup_real_warmup_pow2_lens(4096)[-1], 4096
            )
        with patch.dict(
            "os.environ", {"STARTUP_REAL_WARMUP_MAX_TOKEN_LEN": "1048576"}
        ):
            self.assertEqual(
                start_server._get_startup_real_warmup_max_len(configs),
                81921,
                "cap above model max_seq_len should keep model max_seq_len",
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


if __name__ == "__main__":
    unittest.main()
