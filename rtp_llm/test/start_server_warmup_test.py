import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm import start_server


class StartupRealWarmupTokenLensTest(unittest.TestCase):
    @staticmethod
    def _configs(max_seq_len=262144):
        return SimpleNamespace(
            model_args=SimpleNamespace(max_seq_len=max_seq_len),
        )

    def test_default_uses_pow2_token_lens(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS", None)
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(
                    self._configs(max_seq_len=16)
                ),
                [2, 4, 8, 16],
            )

    def test_configured_token_lens(self):
        with patch.dict(
            os.environ,
            {
                "RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": (
                    "4096,131072,262144"
                )
            },
        ):
            self.assertEqual(
                start_server._get_startup_real_warmup_token_lens(self._configs()),
                [4096, 131072, 262144],
            )

    def test_configured_token_lens_must_fit_model_max(self):
        with patch.dict(
            os.environ,
            {"RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": "4096,262144"},
        ):
            with self.assertRaises(ValueError):
                start_server._get_startup_real_warmup_token_lens(
                    self._configs(max_seq_len=131072)
                )


if __name__ == "__main__":
    unittest.main()
