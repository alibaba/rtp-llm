import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.qwen_v2 import QWenV2


_BASE_CONFIG = {
    "intermediate_size": 12288,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
}


def _parse(extra):
    config = ModelConfig()
    config_json = dict(_BASE_CONFIG)
    config_json.update(extra)
    QWenV2._from_config_json(config, config_json)
    return config


class QwenConfigParseTest(unittest.TestCase):
    def test_transformers_5_2_nested_rope_and_dtype(self):
        config = _parse(
            {
                "rope_parameters": {
                    "rope_theta": 10_000_000,
                    "rope_type": "default",
                },
                "dtype": "bfloat16",
            }
        )
        self.assertEqual(config.attn_config.rope_config.base, 10_000_000)
        self.assertEqual(config.config_dtype, "bfloat16")

    def test_legacy_flat_rope_and_torch_dtype(self):
        config = _parse(
            {
                "rope_theta": 1_000_000,
                "torch_dtype": "bfloat16",
            }
        )
        self.assertEqual(config.attn_config.rope_config.base, 1_000_000)
        self.assertEqual(config.config_dtype, "bfloat16")

    def test_top_level_rope_theta_takes_precedence(self):
        config = _parse(
            {
                "rope_theta": 500_000,
                "rope_parameters": {"rope_theta": 1_000_000},
            }
        )
        self.assertEqual(config.attn_config.rope_config.base, 500_000)


if __name__ == "__main__":
    unittest.main()
