import os
import unittest
from unittest import mock

from rtp_llm.server.startup_warmup_policy import (
    startup_real_warmup_auto_enabled,
    startup_real_warmup_enabled,
    startup_real_warmup_flag_env,
    startup_real_warmup_timeout_env,
)


class StartupWarmupPolicyTest(unittest.TestCase):
    def test_kimi_k3_uses_its_own_controls_and_is_auto_enabled(self):
        self.assertTrue(startup_real_warmup_auto_enabled("kimi_k3"))
        self.assertEqual(
            startup_real_warmup_flag_env("kimi_k3"),
            "KIMI_K3_STARTUP_REAL_WARMUP",
        )
        self.assertEqual(
            startup_real_warmup_timeout_env("kimi_k3"),
            "KIMI_K3_STARTUP_REAL_WARMUP_TIMEOUT_S",
        )

    def test_deepseek_v4_compatibility_is_preserved(self):
        self.assertTrue(startup_real_warmup_auto_enabled("deepseek_v4"))
        self.assertEqual(
            startup_real_warmup_flag_env("deepseek_v4"),
            "DSV4_STARTUP_REAL_WARMUP",
        )
        self.assertEqual(
            startup_real_warmup_timeout_env("deepseek_v4"),
            "DSV4_STARTUP_REAL_WARMUP_TIMEOUT_S",
        )

    def test_other_models_remain_opt_in(self):
        self.assertFalse(startup_real_warmup_auto_enabled("qwen_3"))

    def test_one_kimi_k3_flag_controls_all_startup_warmup(self):
        for value in ("0", "false", "off", "no"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"KIMI_K3_STARTUP_REAL_WARMUP": value}, clear=True
            ):
                self.assertFalse(startup_real_warmup_enabled("kimi_k3"))
        for value in ("1", "true", "on", "yes", "force"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {"KIMI_K3_STARTUP_REAL_WARMUP": value}, clear=True
            ):
                self.assertTrue(startup_real_warmup_enabled("kimi_k3"))

    def test_kimi_k3_remains_auto_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(startup_real_warmup_enabled("kimi_k3"))


if __name__ == "__main__":
    unittest.main()
