import unittest

from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType


class KimiK3MainCacheConfigTest(unittest.TestCase):
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
