from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.ops import HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


class KimiLinearKVCacheSpecTest(TestCase):
    def _build_model_config(self, layer_types):
        config = ModelConfig()
        config.num_layers = len(layer_types)
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = layer_types
        return config

    def _post_build_tags(self, layer_types):
        config = self._build_model_config(layer_types)
        KimiLinear._post_build_model_config(config)
        return [layer_descs[0].tag for layer_descs in config.kv_cache_spec_descs]

    def test_kimi_linear_reuses_phase_tags_across_hybrid_cycles(self):
        tags = self._post_build_tags(
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
            ]
        )
        self.assertEqual(
            tags,
            [
                "linear0",
                "linear1",
                "linear2",
                "full",
                "linear0",
                "linear1",
                "linear2",
                "full",
            ],
        )

    def test_kimi_linear_keeps_single_linear_tag_without_full_layers(self):
        tags = self._post_build_tags(
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
                HybridAttentionType.LINEAR,
            ]
        )
        self.assertEqual(tags, ["linear", "linear", "linear"])

    def test_kimi_linear_full_desc_uses_mla_when_enabled(self):
        config = self._build_model_config([HybridAttentionType.NONE])
        config.attn_config.use_mla = True
        config.mla_ops_type = "FLASH_MLA"

        KimiLinear._post_build_model_config(config)

        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "full")
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].cache_type, KVCacheSpecType.MLA
        )

    def test_kimi_linear_does_not_override_existing_descs(self):
        config = self._build_model_config([HybridAttentionType.LINEAR])
        desc = KVCacheSpecDesc()
        desc.tag = "sentinel"
        desc.cache_type = KVCacheSpecType.LINEAR
        config.kv_cache_spec_descs = [[desc]]

        KimiLinear._post_build_model_config(config)

        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "sentinel")


if __name__ == "__main__":
    main()
