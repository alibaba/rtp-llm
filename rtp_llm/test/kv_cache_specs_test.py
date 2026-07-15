from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.deepseek_v2 import DeepSeekV3Mtp
from rtp_llm.models.hybrid_kv_cache import calculate_hybrid_group_layer_num
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next
from rtp_llm.models.qwen3_next.qwen3_next_mtp import Qwen3NextMTP
from rtp_llm.models.qwen_v2 import QwenV2MTP
from rtp_llm.ops import HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType


class HybridKVCacheSpecTest(TestCase):
    def _build_model_config(self, layer_types):
        config = ModelConfig()
        config.num_layers = len(layer_types)
        config.hybrid_attention_config.enable_hybrid_attention = True
        config.hybrid_attention_config.hybrid_attention_types = layer_types
        return config

    def _kimi_post_build_tags(self, layer_types):
        config = self._build_model_config(layer_types)
        KimiLinear._post_build_model_config(config)
        return [layer_descs[0].tag for layer_descs in config.kv_cache_spec_descs]

    def test_qwen_v2_mtp_default_desc_matches_model_layers(self):
        config = ModelConfig()
        config.num_layers = 32
        config.is_mtp = True

        QwenV2MTP._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)
        for layer_descs in config.kv_cache_spec_descs:
            self.assertEqual(layer_descs[0].tag, "default")
            self.assertEqual(layer_descs[0].cache_type, KVCacheSpecType.MHA)

    def test_deepseek_v3_mtp_default_desc_matches_model_layers(self):
        config = ModelConfig()
        config.num_layers = 61
        config.is_mtp = True
        config.attn_config.use_mla = True
        config.mla_ops_type = "FLASH_MLA"

        DeepSeekV3Mtp._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_spec_descs), config.num_layers)
        for layer_descs in config.kv_cache_spec_descs:
            self.assertEqual(layer_descs[0].tag, "default")
            self.assertEqual(layer_descs[0].cache_type, KVCacheSpecType.MLA)

    def test_mtp_single_layer_models_keep_one_descriptor(self):
        for model_cls in (QwenV2MTP, DeepSeekV3Mtp):
            config = ModelConfig()
            config.num_layers = 1
            config.is_mtp = True

            model_cls._post_build_model_config(config)

            self.assertEqual(len(config.kv_cache_spec_descs), 1)

    def test_qwen3_next_mtp_desc_has_one_full_layer(self):
        config = self._build_model_config([HybridAttentionType.NONE])
        config.is_mtp = True

        Qwen3NextMTP._post_build_model_config(config)

        self.assertEqual(len(config.kv_cache_spec_descs), 1)
        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "full")
        self.assertEqual(config.kv_cache_spec_descs[0][0].cache_type, KVCacheSpecType.MHA)

    def test_calculate_group_layer_num_uses_full_count_fallback(self):
        self.assertEqual(calculate_hybrid_group_layer_num(30, 10), 10)
        self.assertEqual(calculate_hybrid_group_layer_num(4, 6), 6)
        self.assertEqual(calculate_hybrid_group_layer_num(3, 0), 3)
        self.assertEqual(calculate_hybrid_group_layer_num(0, 3), 3)

    def test_qwen3_next_40_layers_uses_contiguous_linear_split(self):
        layer_types = [
            HybridAttentionType.NONE if (i + 1) % 4 == 0 else HybridAttentionType.LINEAR
            for i in range(40)
        ]
        config = self._build_model_config(layer_types)

        Qwen3Next._post_build_model_config(config)

        tags = [layer_descs[0].tag for layer_descs in config.kv_cache_spec_descs]
        self.assertEqual(tags.count("full"), 10)
        self.assertEqual(tags.count("linear0"), 10)
        self.assertEqual(tags.count("linear1"), 10)
        self.assertEqual(tags.count("linear2"), 10)
        self.assertEqual(tags[11], "full")
        self.assertEqual(tags[12], "linear0")
        self.assertEqual(tags[13], "linear1")

    def test_kimi_linear_uses_contiguous_tags_across_hybrid_cycles(self):
        tags = self._kimi_post_build_tags(
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
                "linear0",
                "linear1",
                "full",
                "linear1",
                "linear2",
                "linear2",
                "full",
            ],
        )

    def test_kimi_linear_group_layer_num_fallback_keeps_sparse_linear_contiguous(self):
        tags = self._kimi_post_build_tags(
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
            ]
        )
        self.assertEqual(
            tags,
            [
                "linear0",
                "full",
                "full",
                "linear0",
                "full",
                "full",
                "linear0",
                "full",
                "full",
                "linear0",
            ],
        )

    def test_kimi_linear_keeps_single_linear_tag_without_full_layers(self):
        tags = self._kimi_post_build_tags(
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
