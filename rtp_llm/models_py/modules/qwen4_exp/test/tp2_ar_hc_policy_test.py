import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.models_py.modules.qwen4_exp.tp2_ar_hc_fused import (
    prepare_tp2_ar_hc_workspace,
)


class Tp2ArHcPolicyTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(hc_mult=4, hidden_size=2560)
        self.parallelism = SimpleNamespace(tp_size=2, dp_size=1, local_world_size=2)

    def test_default_enables_fused_tp2_ar_hc(self):
        for env in ({}, {"FT_DISABLE_CUSTOM_AR": "0"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with patch("torch.cuda.is_available", return_value=False) as available:
                    self.assertIsNone(
                        prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                    )
                    available.assert_called_once_with()

    def test_explicit_disable_skips_device_validation(self):
        env = {"FT_DISABLE_CUSTOM_AR": "0", "RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED": "1"}
        with patch.dict(os.environ, env, clear=True):
            with patch("torch.cuda.is_available") as available:
                self.assertIsNone(
                    prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                )
                available.assert_not_called()

    def test_custom_allreduce_disable_overrides_default(self):
        env = {"FT_DISABLE_CUSTOM_AR": "1", "RTP_QWEN4_DISABLE_TP2_AR_HC_FUSED": "0"}
        with patch.dict(os.environ, env, clear=True):
            with patch("torch.cuda.is_available") as available:
                self.assertIsNone(
                    prepare_tp2_ar_hc_workspace(self.config, self.parallelism)
                )
                available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
