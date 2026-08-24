import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rtp_llm.models.kimi_k3.kimi_k3 import (
    KimiK3,
    KimiK3Eagle3,
    KimiK3ModelConfig,
    _mla_prefill_kv_chunk_tokens,
)


class KimiK3MLAChunkConfigTest(unittest.TestCase):
    _ENV_NAME = "KIMI_K3_MLA_PREFILL_KV_CHUNK_TOKENS"

    @staticmethod
    def _parse_chunk_tokens() -> int:
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
        return config.attn_config.mla_prefill_kv_chunk_tokens

    @staticmethod
    def _parse_eagle3_chunk_tokens() -> int:
        raw_config = {
            "model_type": "deepseek_v3_swa",
            "num_hidden_layers": 1,
            "hidden_size": 7168,
            "vocab_size": 163840,
            "max_position_embeddings": 1048576,
            "intermediate_size": 18432,
            "num_attention_heads": 96,
            "num_key_value_heads": 96,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "sliding_window": 4096,
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            Path(checkpoint_dir, "config.json").write_text(
                json.dumps(raw_config), encoding="utf-8"
            )
            config = KimiK3Eagle3._create_config(checkpoint_dir)
        return config.attn_config.mla_prefill_kv_chunk_tokens

    def test_k3_chunking_is_opt_in_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._ENV_NAME, None)
            self.assertEqual(_mla_prefill_kv_chunk_tokens(), 0)
            self.assertEqual(self._parse_chunk_tokens(), 0)
            self.assertEqual(self._parse_eagle3_chunk_tokens(), 0)

    def test_explicit_chunk_capacity_is_forwarded_to_attention_config(self) -> None:
        with mock.patch.dict(os.environ, {self._ENV_NAME: "65536"}, clear=False):
            self.assertEqual(self._parse_chunk_tokens(), 65536)
        with mock.patch.dict(os.environ, {self._ENV_NAME: "0"}, clear=False):
            self.assertEqual(self._parse_chunk_tokens(), 0)

    def test_explicit_chunk_capacity_is_forwarded_to_eagle3_config(self) -> None:
        with mock.patch.dict(os.environ, {self._ENV_NAME: "16384"}, clear=False):
            self.assertEqual(self._parse_eagle3_chunk_tokens(), 16384)

    def test_invalid_override(self) -> None:
        for parser in (self._parse_chunk_tokens, self._parse_eagle3_chunk_tokens):
            with self.subTest(parser=parser.__name__):
                with mock.patch.dict(os.environ, {self._ENV_NAME: "-1"}, clear=False):
                    with self.assertRaisesRegex(ValueError, "must be non-negative"):
                        parser()
                with mock.patch.dict(
                    os.environ, {self._ENV_NAME: "invalid"}, clear=False
                ):
                    with self.assertRaisesRegex(ValueError, "non-negative integer"):
                        parser()


if __name__ == "__main__":
    unittest.main()
