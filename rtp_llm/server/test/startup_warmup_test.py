import os
import unittest
from unittest import mock

from rtp_llm.server.startup_warmup import (
    STARTUP_REAL_WARMUP_ENV,
    STARTUP_REAL_WARMUP_TIMEOUT_ENV,
    STARTUP_REAL_WARMUP_TIMEOUT_S,
    _get_startup_real_warmup_pow2_lens,
    _get_startup_real_warmup_request_token_len,
    _get_startup_real_warmup_timeout_s,
    startup_real_warmup_enabled,
)


class StartupWarmupTest(unittest.TestCase):
    def test_all_models_use_shared_controls(self):
        self.assertEqual(STARTUP_REAL_WARMUP_ENV, "STARTUP_REAL_WARMUP")
        self.assertEqual(
            STARTUP_REAL_WARMUP_TIMEOUT_ENV,
            "STARTUP_REAL_WARMUP_TIMEOUT_S",
        )

    def test_supported_models_are_auto_enabled(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(startup_real_warmup_enabled("kimi_k3"))
            self.assertTrue(startup_real_warmup_enabled("deepseek_v4"))

    def test_other_models_remain_opt_in(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(startup_real_warmup_enabled("qwen_3"))

    def test_shared_flag_disables_all_startup_warmup(self):
        for value in ("0", "false", "off", "no"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {STARTUP_REAL_WARMUP_ENV: value}, clear=True
            ):
                self.assertFalse(startup_real_warmup_enabled("kimi_k3"))
                self.assertFalse(startup_real_warmup_enabled("deepseek_v4"))

    def test_shared_flag_forces_all_models(self):
        for value in ("1", "true", "on", "yes", "force"):
            with self.subTest(value=value), mock.patch.dict(
                os.environ, {STARTUP_REAL_WARMUP_ENV: value}, clear=True
            ):
                self.assertTrue(startup_real_warmup_enabled("kimi_k3"))
                self.assertTrue(startup_real_warmup_enabled("qwen_3"))

    def test_pow2_token_lens_include_exact_max_len(self):
        self.assertEqual(_get_startup_real_warmup_pow2_lens(8), [2, 4, 8])
        self.assertEqual(_get_startup_real_warmup_pow2_lens(10), [2, 4, 8, 10])
        with self.assertRaises(ValueError):
            _get_startup_real_warmup_pow2_lens(1)

    def test_request_token_len_reserves_generation_and_speculative_steps(self):
        self.assertEqual(_get_startup_real_warmup_request_token_len(16, 16), 15)
        self.assertEqual(
            _get_startup_real_warmup_request_token_len(16, 16, reserve_step=5),
            11,
        )
        self.assertEqual(
            _get_startup_real_warmup_request_token_len(2, 16, reserve_step=5),
            2,
        )
        with self.assertRaises(ValueError):
            _get_startup_real_warmup_request_token_len(5, 5, reserve_step=5)

    def test_timeout_uses_shared_control(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _get_startup_real_warmup_timeout_s(),
                STARTUP_REAL_WARMUP_TIMEOUT_S,
            )
        with mock.patch.dict(
            os.environ,
            {STARTUP_REAL_WARMUP_TIMEOUT_ENV: "12.5"},
            clear=True,
        ):
            self.assertEqual(_get_startup_real_warmup_timeout_s(), 12.5)
        with mock.patch.dict(
            os.environ,
            {STARTUP_REAL_WARMUP_TIMEOUT_ENV: "0"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                _get_startup_real_warmup_timeout_s()


if __name__ == "__main__":
    unittest.main()
