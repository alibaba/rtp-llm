from unittest import TestCase, main

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV3Mtp
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen2_vl import QWen2_VL
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import Qwen3NextMTP
from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.models.qwen_v2 import QwenV2MTP
from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheGroupType,
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)


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

    def test_desc_retains_model_dsl_tag_identity(self):
        desc = KVCacheSpecDesc()
        desc.tag = "semantic_group"

        self.assertEqual(desc.tag, "semantic_group")

    def test_removed_group_memory_policy_keys_are_unknown(self):
        capacity = CacheCapacityPolicyDesc()
        with self.assertRaises(AttributeError):
            setattr(capacity, "charge_to_" + "paged_budget", True)

        desc = KVCacheSpecDesc()
        with self.assertRaises(AttributeError):
            desc.memory = None

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

    def test_sparse_mla_declares_kernel_compressed_indexer_descriptor(self):
        config = ModelConfig()
        config.num_layers = 2
        config.attn_config.use_mla = True
        config.mla_ops_type = "FLASH_MLA"
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = 256
        config.attn_config.tokens_per_block = 512

        BaseModel._post_build_model_config(config)
        self.assertEqual(len(config.kv_cache_spec_descs), 2)
        for layer_descs in config.kv_cache_spec_descs:
            self.assertEqual(
                [desc.tag for desc in layer_descs], ["default", "indexer_kv"]
            )
            self.assertEqual(layer_descs[0].cache_type, KVCacheSpecType.MLA)
            indexer_desc = layer_descs[1]
            self.assertEqual(indexer_desc.cache_type, KVCacheSpecType.OPAQUE_KV)
            self.assertEqual(indexer_desc.entry_dtype, DataType.TYPE_UINT8)
            self.assertEqual(indexer_desc.entry_elems, 264)
            self.assertEqual(
                indexer_desc.entry_count_mode,
                OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
            )
            self.assertEqual(indexer_desc.compression_ratio, 1)
            self.assertNotEqual(
                indexer_desc.explicit_entry_count,
                config.attn_config.tokens_per_block,
            )

    def test_sparse_mla_mha_fallback_still_declares_indexer_descriptor(self):
        config = ModelConfig()
        config.num_layers = 2
        config.attn_config.use_mla = True
        config.mla_ops_type = "MHA"
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = 128
        config.attn_config.tokens_per_block = 256

        BaseModel._post_build_model_config(config)
        for layer_descs in config.kv_cache_spec_descs:
            self.assertEqual(
                [desc.tag for desc in layer_descs], ["default", "indexer_kv"]
            )
            self.assertEqual(layer_descs[0].cache_type, KVCacheSpecType.MHA)
            self.assertEqual(layer_descs[1].cache_type, KVCacheSpecType.OPAQUE_KV)
            self.assertEqual(layer_descs[1].entry_elems, 132)
            self.assertEqual(
                layer_descs[1].entry_count_mode,
                OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED,
            )
            self.assertEqual(layer_descs[1].compression_ratio, 1)
            self.assertNotEqual(
                layer_descs[1].explicit_entry_count,
                config.attn_config.tokens_per_block,
            )

    def test_sparse_mtp_declares_same_indexer_descriptor_as_target(self):
        target = ModelConfig()
        target.num_layers = 1
        target.attn_config.use_mla = True
        target.mla_ops_type = "FLASH_MLA"
        target.attn_config.is_sparse = True
        target.attn_config.indexer_head_dim = 128
        target.attn_config.tokens_per_block = 256
        propose = ModelConfig()
        propose.num_layers = 1
        propose.is_mtp = True
        propose.attn_config.use_mla = True
        propose.mla_ops_type = "FLASH_MLA"
        propose.attn_config.is_sparse = True
        propose.attn_config.indexer_head_dim = 128
        propose.attn_config.tokens_per_block = 256

        BaseModel._post_build_model_config(target)
        DeepSeekV3Mtp._post_build_model_config(propose)
        self.assertEqual(
            [desc.tag for desc in target.kv_cache_spec_descs[0]],
            ["default", "indexer_kv"],
        )
        self.assertEqual(
            [desc.tag for desc in propose.kv_cache_spec_descs[0]],
            ["default", "indexer_kv"],
        )
        self.assertEqual(
            propose.kv_cache_spec_descs[0][1].entry_elems,
            target.kv_cache_spec_descs[0][1].entry_elems,
        )

    def test_sparse_mtp_mha_fallback_aligns_indexer_descriptor_with_target(self):
        target = ModelConfig()
        target.num_layers = 2
        target.attn_config.use_mla = True
        target.mla_ops_type = "MHA"
        target.attn_config.is_sparse = True
        target.attn_config.indexer_head_dim = 256
        target.attn_config.tokens_per_block = 128
        propose = ModelConfig()
        propose.num_layers = 1
        propose.is_mtp = True
        propose.attn_config.use_mla = True
        propose.mla_ops_type = "MHA"
        propose.attn_config.is_sparse = True
        propose.attn_config.indexer_head_dim = 256
        propose.attn_config.tokens_per_block = 128

        BaseModel._post_build_model_config(target)
        DeepSeekV3Mtp._post_build_model_config(propose)

        for config in (target, propose):
            self.assertEqual(
                [desc.tag for desc in config.kv_cache_spec_descs[0]],
                ["default", "indexer_kv"],
            )
            self.assertEqual(
                config.kv_cache_spec_descs[0][0].cache_type, KVCacheSpecType.MHA
            )
            self.assertEqual(
                config.kv_cache_spec_descs[0][1].cache_type,
                KVCacheSpecType.OPAQUE_KV,
            )
        self.assertEqual(
            propose.kv_cache_spec_descs[0][1].entry_elems,
            target.kv_cache_spec_descs[0][1].entry_elems,
        )
        self.assertEqual(
            propose.kv_cache_spec_descs[0][1].explicit_entry_count,
            target.kv_cache_spec_descs[0][1].explicit_entry_count,
        )

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
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].cache_type, KVCacheSpecType.MHA
        )
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].group_type, CacheGroupType.FULL
        )

    def test_qwen3_next_40_layers_uses_one_homogeneous_linear_tag(self):
        layer_types = [
            HybridAttentionType.NONE if (i + 1) % 4 == 0 else HybridAttentionType.LINEAR
            for i in range(40)
        ]
        config = self._build_model_config(layer_types)

        Qwen3Next._post_build_model_config(config)

        tags = [layer_descs[0].tag for layer_descs in config.kv_cache_spec_descs]
        self.assertEqual(tags.count("full"), 10)
        self.assertEqual(tags.count("linear"), 30)
        self.assertEqual(tags[11], "full")
        self.assertEqual(tags[12], "linear")
        self.assertEqual(tags[13], "linear")
        self.assertEqual(
            config.kv_cache_spec_descs[11][0].group_type, CacheGroupType.FULL
        )
        self.assertEqual(
            config.kv_cache_spec_descs[12][0].group_type, CacheGroupType.LINEAR
        )

    def test_qwen35_defaults_missing_mrope_interleaved_to_true(self):
        config = ModelConfig()
        config.attn_config.size_per_head = 256
        rope_parameters = {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
        }

        Qwen35Moe._parse_rope_config({"rope_parameters": rope_parameters}, config)

        self.assertTrue(config.attn_config.rope_config.mrope_interleaved)

    def test_qwen35_rejects_non_interleaved_mrope(self):
        config = ModelConfig()
        config.attn_config.size_per_head = 256
        rope_parameters = {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": False,
        }

        with self.assertRaisesRegex(ValueError, "Qwen3Next requires.*true"):
            Qwen35Moe._parse_rope_config({"rope_parameters": rope_parameters}, config)

    def test_qwen35_rejects_non_three_axis_mrope_section(self):
        config = ModelConfig()
        config.attn_config.size_per_head = 256
        rope_parameters = {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [16, 16],
        }

        with self.assertRaisesRegex(ValueError, "exactly 3 T/H/W sections"):
            Qwen35Moe._parse_rope_config({"rope_parameters": rope_parameters}, config)

    def test_qwen3_vl_defaults_to_interleaved_mrope_sections(self):
        config = ModelConfig()
        QWen3_VL._from_config_json(
            config,
            {
                "vision_start_token_id": 1,
                "vision_end_token_id": 2,
                "text_config": {
                    "intermediate_size": 256,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 128,
                    "hidden_size": 256,
                    "num_hidden_layers": 2,
                    "vocab_size": 1024,
                },
            },
        )

        rope_config = config.attn_config.rope_config
        self.assertTrue(rope_config.mrope_interleaved)
        self.assertEqual(rope_config.index_factor, 3)
        self.assertEqual(
            [
                rope_config.mrope_dim1,
                rope_config.mrope_dim2,
                rope_config.mrope_dim3,
            ],
            [24, 20, 20],
        )
        self.assertEqual(
            rope_config.mrope_dim1 + rope_config.mrope_dim2 + rope_config.mrope_dim3,
            rope_config.dim // 2,
        )

    def test_qwen3_vl_rejects_non_three_axis_mrope_section(self):
        config = ModelConfig()
        with self.assertRaisesRegex(ValueError, "exactly 3 T/H/W sections"):
            QWen3_VL._from_config_json(
                config,
                {
                    "vision_start_token_id": 1,
                    "vision_end_token_id": 2,
                    "text_config": {
                        "intermediate_size": 256,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 128,
                        "hidden_size": 256,
                        "num_hidden_layers": 2,
                        "vocab_size": 1024,
                        "rope_scaling": {"mrope_section": [32, 32]},
                    },
                },
            )

    def test_qwen2_vl_parses_explicit_mrope_layout(self):
        config = ModelConfig()
        QWen2_VL._from_hf(
            config,
            {
                "vocab_size": 1024,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 256,
                "head_dim": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
                "rope_scaling": {
                    "mrope_section": [20, 22, 22],
                    "mrope_interleaved": True,
                },
            },
        )

        rope_config = config.attn_config.rope_config
        self.assertTrue(rope_config.mrope_interleaved)
        self.assertEqual(rope_config.index_factor, 3)
        self.assertEqual(
            [
                rope_config.mrope_dim1,
                rope_config.mrope_dim2,
                rope_config.mrope_dim3,
            ],
            [20, 22, 22],
        )

    def test_qwen2_vl_defaults_when_rope_scaling_is_missing(self):
        config = ModelConfig()
        QWen2_VL._from_hf(
            config,
            {
                "vocab_size": 1024,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 256,
                "head_dim": 128,
                "num_hidden_layers": 2,
                "intermediate_size": 256,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1_000_000,
            },
        )

        rope_config = config.attn_config.rope_config
        self.assertFalse(rope_config.mrope_interleaved)
        self.assertEqual(
            [rope_config.mrope_dim1, rope_config.mrope_dim2, rope_config.mrope_dim3],
            [16, 24, 24],
        )

    def test_qwen2_vl_rejects_non_three_axis_mrope_section(self):
        config = ModelConfig()
        with self.assertRaisesRegex(ValueError, "exactly 3 T/H/W sections"):
            QWen2_VL._from_hf(
                config,
                {
                    "vocab_size": 1024,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "hidden_size": 256,
                    "head_dim": 128,
                    "num_hidden_layers": 2,
                    "intermediate_size": 256,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1_000_000,
                    "rope_scaling": {"mrope_section": [32, 32]},
                },
            )

    def test_kimi_linear_uses_one_homogeneous_tag_across_hybrid_cycles(self):
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
                "linear",
                "linear",
                "linear",
                "full",
                "linear",
                "linear",
                "linear",
                "full",
            ],
        )

    def test_kimi_linear_sparse_pattern_uses_one_homogeneous_tag(self):
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
                "linear",
                "full",
                "full",
                "linear",
                "full",
                "full",
                "linear",
                "full",
                "full",
                "linear",
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
        self.assertEqual(
            config.kv_cache_spec_descs[0][0].group_type, CacheGroupType.FULL
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
