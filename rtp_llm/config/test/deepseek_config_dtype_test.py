import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v2 import DeepSeekV2


class DeepSeekConfigDtypeTest(unittest.TestCase):

    def _parse(self, **dtype_fields):
        config_json = {
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "vocab_size": 32000,
            "hidden_size": 2048,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "routed_scaling_factor": 1.0,
            "num_experts_per_tok": 2,
            "n_routed_experts": 8,
            "moe_intermediate_size": 1024,
            "n_shared_experts": 1,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            **dtype_fields,
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(
                json.dumps(config_json), encoding="utf-8"
            )
            config = ModelConfig()
            DeepSeekV2._from_hf(config, directory)
        return config

    def test_dtype_is_used_when_torch_dtype_is_absent(self):
        config = self._parse(dtype="bfloat16")
        self.assertEqual(config.config_dtype, "bfloat16")

    def test_torch_dtype_takes_precedence(self):
        config = self._parse(dtype="bfloat16", torch_dtype="float16")
        self.assertEqual(config.config_dtype, "float16")


if __name__ == "__main__":
    unittest.main()
