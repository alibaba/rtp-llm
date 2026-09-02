import os
import unittest
from unittest import mock

from rtp_llm.models.kimi_k3.kimi_k3 import (
    KimiK3,
    KimiK3ModelConfig,
    _mla_prefill_expanded_kv_budget_bytes,
)


class KimiK3MLAWorkspaceConfigTest(unittest.TestCase):
    _BUDGET_ENV = "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_BYTES"

    @staticmethod
    def _parse_budget_bytes() -> int:
        config = KimiK3ModelConfig()
        KimiK3._parse_attention_config(
            {
                "num_attention_heads": 96,
                "num_key_value_heads": 96,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "linear_attn_config": {"num_heads": 96, "head_dim": 128},
            },
            config,
        )
        return config.attn_config.mla_prefill_expanded_kv_budget_bytes

    def test_k3_defaults_to_disabled_expanded_kv_planner(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._BUDGET_ENV, None)
            self.assertEqual(_mla_prefill_expanded_kv_budget_bytes(), 0)
            self.assertEqual(self._parse_budget_bytes(), 0)

    def test_explicit_budget_and_zero_disable_are_forwarded(self) -> None:
        for raw, expected in (("1073741824", 1024**3), ("0", 0)):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    self.assertEqual(self._parse_budget_bytes(), expected)

    def test_invalid_budget_is_rejected(self) -> None:
        for raw in ("-1", "invalid"):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        _mla_prefill_expanded_kv_budget_bytes()


if __name__ == "__main__":
    unittest.main()
