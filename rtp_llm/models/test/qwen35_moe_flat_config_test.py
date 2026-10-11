import json
import os
import tempfile
import unittest

from rtp_llm.models.qwen3_next.qwen3_next import Qwen35Moe
from rtp_llm.ops import DataType, HybridAttentionType


def _decider_flat_text_config() -> dict:
    """Flat (no text_config wrapper) qwen3_5_moe_text config, mirroring
    decider-35b-a3b/config.json."""
    return {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "model_type": "qwen3_5_moe_text",
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_value_head_dim": 128,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 262144,
        "moe_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "num_key_value_heads": 2,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "shared_expert_intermediate_size": 512,
        "tie_word_embeddings": False,
        "vocab_size": 248320,
    }


class Qwen35MoeFlatConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(cls._tmp.name, "config.json"), "w") as writer:
            json.dump(_decider_flat_text_config(), writer)
        cls.config = Qwen35Moe._create_config(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_basic_config(self):
        config = self.config
        self.assertEqual(config.num_layers, 40)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.vocab_size, 248320)
        self.assertEqual(config.max_seq_len, 262144)
        self.assertFalse(config.tie_word_embeddings)
        self.assertEqual(config.attn_config.head_num, 16)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 256)

    def test_moe_config(self):
        config = self.config
        self.assertEqual(config.expert_num, 256)
        self.assertEqual(config.moe_k, 8)
        self.assertEqual(config.moe_inter_size, 512)
        # Independently-sized single shared expert, not DeepSeek-style counting.
        self.assertEqual(config.inter_size, 512)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        self.assertEqual(config.moe_layer_index, list(range(40)))

    def test_hybrid_attention_pattern(self):
        config = self.config
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)
        types = config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(len(types), 40)
        for i in range(40):
            with self.subTest(layer=i):
                if (i + 1) % 4 == 0:
                    self.assertEqual(types[i], HybridAttentionType.NONE)
                else:
                    self.assertEqual(types[i], HybridAttentionType.LINEAR)
        full_layers = [i for i in range(40) if types[i] == HybridAttentionType.NONE]
        self.assertEqual(full_layers, [3, 7, 11, 15, 19, 23, 27, 31, 35, 39])

    def test_linear_attention_config(self):
        lac = self.config.linear_attention_config
        self.assertEqual(lac.linear_conv_kernel_dim, 4)
        self.assertEqual(lac.linear_key_head_dim, 128)
        self.assertEqual(lac.linear_num_key_heads, 16)
        self.assertEqual(lac.linear_num_value_heads, 32)
        self.assertEqual(lac.linear_value_head_dim, 128)
        self.assertEqual(lac.ssm_state_dtype, DataType.TYPE_FP32)

    def test_rope_config_style7_mrope(self):
        rope = self.config.attn_config.rope_config
        self.assertEqual(int(rope.style), 7)
        self.assertEqual(rope.base, 10000000)
        self.assertEqual(self.config.partial_rotary_factor, 0.25)
        # dim = size_per_head * partial_rotary_factor = 256 * 0.25
        self.assertEqual(rope.dim, 64)
        # mrope_section [11, 11, 10] applied via apply_mrope_section.
        self.assertEqual(rope.index_factor, 3)
        self.assertEqual(rope.mrope_dim1, 11)
        self.assertEqual(rope.mrope_dim2, 11)
        self.assertEqual(rope.mrope_dim3, 10)
        self.assertTrue(rope.mrope_interleaved)
        self.assertEqual(self.config.mm_model_config.mm_position_ids_style, 2)

    def test_text_only_is_not_multimodal(self):
        # Flat config has no vision_start_token_id: mm parsing returns early.
        self.assertFalse(self.config.mm_model_config.is_multimodal)
        self.assertFalse(self.config.is_multimodal())
        self.assertEqual(self.config.mm_model_config.mm_sep_tokens, [])


if __name__ == "__main__":
    unittest.main()
