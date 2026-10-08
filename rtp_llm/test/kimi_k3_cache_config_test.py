import unittest

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType


class KimiK3MainCacheConfigTest(unittest.TestCase):
    def test_kda_replay_cache_uses_per_channel_gate(self):
        config = KimiK3ModelConfig()
        KimiK3._parse_attention_config(
            {
                "num_attention_heads": 96,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "kv_lora_rank": 512,
                "v_head_dim": 128,
                "linear_attn_config": {"num_heads": 96, "head_dim": 128},
            },
            config,
        )

        linear = config.linear_attention_config
        self.assertEqual(linear.linear_key_head_dim, 128)
        self.assertEqual(linear.linear_value_head_dim, 128)
        self.assertEqual(linear.linear_num_value_heads, 96)
        # KDA's forget gate has one value per key channel, not per head.
        self.assertTrue(linear.replay_vector_gate)

    def test_hybrid_schedule_builds_separate_main_cache_groups(self):
        config = KimiK3ModelConfig()
        config.num_layers = 4
        KimiK3._parse_hybrid_attention_config(
            {"linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}},
            config,
        )
        KimiK3._post_build_model_config(config)

        self.assertEqual(config.hybrid_attention_config.hybrid_attention_types, [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ])
        self.assertEqual([descs[0].tag for descs in config.kv_cache_spec_descs], [
            "linear", "linear", "linear", "full"
        ])
        self.assertEqual([descs[0].cache_type for descs in config.kv_cache_spec_descs], [
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.LINEAR,
            KVCacheSpecType.MLA,
        ])


if __name__ == "__main__":
    unittest.main()
