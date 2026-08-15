import pickle
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import rtp_llm.config.model_config as model_config_module
from rtp_llm.config.kv_cache_config import DEFAULT_KV_CACHE_TAG, INDEXER_KV_CACHE_TAG
from rtp_llm.config.model_config import (
    ModelConfig,
    build_model_config,
    resolve_kv_cache_kernel_seq_size_per_block,
)
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV3Mtp
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen2_vl import QWen2_VL
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next, Qwen35Moe
from rtp_llm.models.qwen3_next.qwen3_next_mtp import Qwen3NextMTP
from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.models.qwen_v2 import QwenV2MTP
from rtp_llm.ops import DEFAULT_KV_CACHE_TAG as CPP_DEFAULT_KV_CACHE_TAG
from rtp_llm.ops import INDEXER_KV_CACHE_TAG as CPP_INDEXER_KV_CACHE_TAG
from rtp_llm.ops import (
    CacheCapacityPolicyDesc,
    CacheCpPolicyDesc,
    CpBlockMappingMode,
    CpBlockSliceMode,
    CpPrefillSliceLayout,
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    TaskType,
)


class HybridKVCacheSpecTest(TestCase):
    def test_cache_group_tags_are_exported_from_cpp(self):
        self.assertEqual(DEFAULT_KV_CACHE_TAG, CPP_DEFAULT_KV_CACHE_TAG)
        self.assertEqual(INDEXER_KV_CACHE_TAG, CPP_INDEXER_KV_CACHE_TAG)
        self.assertEqual(
            (DEFAULT_KV_CACHE_TAG, INDEXER_KV_CACHE_TAG), ("default", "indexer_kv")
        )

    def _build_model_config(self, layer_types):
        config = ModelConfig()
        config.num_layers = len(layer_types)
        config.attn_config.tokens_per_block = 64
        config.attn_config.kernel_tokens_per_block = 64
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

    def test_sparse_mla_uses_independent_indexer_descriptor(self):
        config = ModelConfig()
        config.num_layers = 2
        config.attn_config.use_mla = True
        config.mla_ops_type = "FLASH_MLA"
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = 128
        config.attn_config.tokens_per_block = 512
        config.attn_config.kernel_tokens_per_block = 64

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
            self.assertEqual(indexer_desc.entry_elems, 132)
            self.assertEqual(indexer_desc.explicit_entry_count, 512)
            self.assertEqual(indexer_desc.kernel_seq_size_per_block, 64)

    def test_sparse_deepseek_mtp_uses_same_descriptor_helper(self):
        config = ModelConfig()
        config.num_layers = 1
        config.is_mtp = True
        config.attn_config.use_mla = True
        config.mla_ops_type = "FLASH_MLA"
        config.attn_config.is_sparse = True
        config.attn_config.indexer_head_dim = 256
        config.attn_config.tokens_per_block = 256
        config.attn_config.kernel_tokens_per_block = 64

        DeepSeekV3Mtp._post_build_model_config(config)

        self.assertEqual(
            [desc.tag for desc in config.kv_cache_spec_descs[0]],
            ["default", "indexer_kv"],
        )
        self.assertEqual(config.kv_cache_spec_descs[0][1].entry_elems, 264)

    def test_sparse_mla_rejects_invalid_indexer_head_dim(self):
        for indexer_head_dim in (0, 100):
            with self.subTest(indexer_head_dim=indexer_head_dim):
                config = ModelConfig()
                config.num_layers = 1
                config.attn_config.is_sparse = True
                config.attn_config.indexer_head_dim = indexer_head_dim

                with self.assertRaisesRegex(
                    ValueError,
                    "sparse indexer_head_dim must be positive and divisible by 128",
                ):
                    BaseModel._post_build_model_config(config)

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

    def test_qwen3_next_40_layers_uses_one_linear_group(self):
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

    def test_kimi_linear_uses_one_linear_tag_across_hybrid_cycles(self):
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

    def test_kimi_linear_sparse_pattern_uses_one_linear_tag(self):
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

    def test_kimi_linear_does_not_override_existing_descs(self):
        config = self._build_model_config([HybridAttentionType.LINEAR])
        desc = KVCacheSpecDesc()
        desc.tag = "sentinel"
        desc.cache_type = KVCacheSpecType.LINEAR
        config.kv_cache_spec_descs = [[desc]]

        KimiLinear._post_build_model_config(config)

        self.assertEqual(config.kv_cache_spec_descs[0][0].tag, "sentinel")

    def test_descriptor_kernel_resolver_uses_only_effective_non_default_k(self):
        config = ModelConfig()
        config.attn_config.tokens_per_block = 128
        for kernel_tokens_per_block, expected in ((0, None), (128, None), (32, 32)):
            with self.subTest(kernel_tokens_per_block=kernel_tokens_per_block):
                config.attn_config.kernel_tokens_per_block = kernel_tokens_per_block
                self.assertEqual(
                    resolve_kv_cache_kernel_seq_size_per_block(config), expected
                )

    def test_build_model_config_resolves_kernel_block_precedence(self):
        def resolved_kernel(model_kernel: int, runtime_kernel: int) -> int:
            config = SimpleNamespace(
                attn_config=SimpleNamespace(
                    tokens_per_block=0,
                    kernel_tokens_per_block=model_kernel,
                    size_per_head=1,
                    head_num=1,
                ),
                linear_attention_config=SimpleNamespace(),
                max_seq_len=1,
                hidden_size=1,
                data_type=DataType.TYPE_FP16,
                init_precision_config=lambda **_: None,
                apply_override_args=lambda _: None,
            )
            model_args = SimpleNamespace(
                ckpt_path="",
                tokenizer_path="",
                model_type="test",
                phy2log_path="",
                mla_ops_type="",
                max_seq_len=0,
                task_type="",
                act_type="fp16",
                enable_fp32_lm_head=None,
                json_model_override_args="",
            )
            kv_cache_config = SimpleNamespace(
                seq_size_per_block=128,
                kernel_seq_size_per_block=runtime_kernel,
                ssm_state_dtype="fp32",
            )
            profiling_config = SimpleNamespace(hack_layer_num=0)
            with patch.object(
                model_config_module,
                "get_task_type_from_ckpt_path",
                return_value=TaskType.LANGUAGE_MODEL,
            ):
                build_model_config(
                    config, model_args, kv_cache_config, profiling_config
                )
            return config.attn_config.kernel_tokens_per_block

        self.assertEqual(resolved_kernel(model_kernel=32, runtime_kernel=16), 16)
        self.assertEqual(resolved_kernel(model_kernel=32, runtime_kernel=0), 32)
        self.assertEqual(resolved_kernel(model_kernel=0, runtime_kernel=0), 128)

    def test_full_descriptor_producers_own_optional_kernel_geometry(self):
        producers = (
            ("base", BaseModel, [HybridAttentionType.NONE]),
            (
                "qwen_hybrid",
                Qwen3Next,
                [HybridAttentionType.LINEAR, HybridAttentionType.NONE],
            ),
            (
                "kimi_hybrid",
                KimiLinear,
                [HybridAttentionType.LINEAR, HybridAttentionType.NONE],
            ),
            ("qwen_mtp", QwenV2MTP, [HybridAttentionType.NONE]),
            ("deepseek_mtp", DeepSeekV3Mtp, [HybridAttentionType.NONE]),
            ("qwen3_mtp", Qwen3NextMTP, [HybridAttentionType.NONE]),
        )
        for name, model_cls, layer_types in producers:
            for kernel_tokens_per_block, expected in ((64, None), (16, 16)):
                with self.subTest(name=name, kernel=kernel_tokens_per_block):
                    config = self._build_model_config(layer_types)
                    config.attn_config.kernel_tokens_per_block = kernel_tokens_per_block
                    model_cls._post_build_model_config(config)
                    for layer_descs in config.kv_cache_spec_descs:
                        for desc in layer_descs:
                            if desc.cache_type == KVCacheSpecType.LINEAR:
                                self.assertIsNone(desc.kernel_seq_size_per_block)
                            else:
                                self.assertEqual(
                                    desc.kernel_seq_size_per_block, expected
                                )

    def test_cache_cp_policy_pickle_uses_strict_four_field_state(self):
        policy = CacheCpPolicyDesc()
        policy.mapping = CpBlockMappingMode.BLOCK_ROUND_ROBIN
        policy.slice = CpBlockSliceMode.EQUAL_BYTES
        policy.align_payload = True
        policy.prefill_slice_layout = CpPrefillSliceLayout.BLOCK_STRIDE
        restored = pickle.loads(pickle.dumps(policy))
        self.assertEqual(restored.mapping, policy.mapping)
        self.assertEqual(restored.slice, policy.slice)
        self.assertEqual(restored.align_payload, policy.align_payload)
        self.assertEqual(restored.prefill_slice_layout, policy.prefill_slice_layout)
        self.assertFalse(hasattr(restored, "scale_seq_size"))

        four_field_state = pickle.dumps(CacheCpPolicyDesc(), protocol=2)
        self.assertIn(b"(NNNNt", four_field_state)
        legacy_five_field_state = four_field_state.replace(b"(NNNNt", b"(NNNNNt", 1)
        with self.assertRaisesRegex(RuntimeError, "Invalid CacheCpPolicyDesc state"):
            pickle.loads(legacy_five_field_state)

    def test_cache_capacity_policy_pickle_drops_legacy_budget_field(self):
        policy = CacheCapacityPolicyDesc()
        policy.reservable = False
        policy.explicit_block_num = 17

        restored = pickle.loads(pickle.dumps(policy))
        self.assertEqual(restored.reservable, policy.reservable)
        self.assertEqual(restored.explicit_block_num, policy.explicit_block_num)
        self.assertFalse(hasattr(restored, "charge_to_paged_budget"))

        legacy = CacheCapacityPolicyDesc.__new__(CacheCapacityPolicyDesc)
        legacy.__setstate__((True, 23, False))
        self.assertTrue(legacy.reservable)
        self.assertEqual(legacy.explicit_block_num, 23)
        self.assertFalse(hasattr(legacy, "charge_to_paged_budget"))

    def test_kv_cache_spec_desc_pickle_schema_v1_round_trip(self):
        desc = KVCacheSpecDesc()
        desc.tag = "indexer_kv"
        desc.cache_type = KVCacheSpecType.OPAQUE_KV
        desc.entry_dtype = DataType.TYPE_UINT8
        desc.entry_elems = 132
        desc.explicit_entry_count = 64
        desc.kernel_seq_size_per_block = 16

        restored = pickle.loads(pickle.dumps(desc))
        self.assertEqual(restored.tag, desc.tag)
        self.assertEqual(restored.cache_type, desc.cache_type)
        self.assertEqual(restored.entry_dtype, desc.entry_dtype)
        self.assertEqual(restored.entry_elems, desc.entry_elems)
        self.assertEqual(restored.explicit_entry_count, desc.explicit_entry_count)
        self.assertEqual(
            restored.kernel_seq_size_per_block, desc.kernel_seq_size_per_block
        )

        state = desc.__getstate__()
        self.assertEqual((state[0], len(state)), (1, 21))

        invalid = KVCacheSpecDesc.__new__(KVCacheSpecDesc)
        with self.assertRaisesRegex(
            RuntimeError,
            "cross-version pickle is unsupported; expected version=1 fields=21 actual version=0 fields=20",
        ):
            invalid.__setstate__((None,) * 20)

        with self.assertRaisesRegex(
            RuntimeError, "expected version=1 actual version=2"
        ):
            invalid.__setstate__((2,) + state[1:])


if __name__ == "__main__":
    main()
