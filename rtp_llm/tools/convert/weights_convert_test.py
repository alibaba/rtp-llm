import unittest

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.cpp.cache.test.libcache_config_creator_py_test import validate_basic_config
from rtp_llm.models.kimi_linear.kimi_linear import KimiLinear
from rtp_llm.models.qwen3_next.qwen3_next import Qwen3Next
from rtp_llm.ops import DataType, HybridAttentionType, KVCacheSpecDesc, KVCacheSpecType
from rtp_llm.tools.convert.weights_convert_utils import (
    ModelConfigPostBuilder,
    apply_layer_override_and_post_build,
)


class WeightsConvertLayerOverrideTest(unittest.TestCase):
    def _model_config(self) -> ModelConfig:
        model_config = ModelConfig()
        model_config.num_layers = 4
        model_config.data_type = "fp16"
        model_config.attn_config.head_num = 2
        model_config.attn_config.kv_head_num = 2
        model_config.attn_config.size_per_head = 16
        model_config.attn_config.tokens_per_block = 4
        model_config.hybrid_attention_config.enable_hybrid_attention = True
        model_config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        model_config.linear_attention_config.linear_conv_kernel_dim = 2
        model_config.linear_attention_config.linear_key_head_dim = 8
        model_config.linear_attention_config.linear_value_head_dim = 8
        model_config.linear_attention_config.linear_num_key_heads = 2
        model_config.linear_attention_config.linear_num_value_heads = 2
        model_config.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP16
        model_config.linear_attention_config.conv_state_dtype = DataType.TYPE_FP16
        model_config.moe_layer_index = [1, 3]
        stale_desc = KVCacheSpecDesc()
        stale_desc.tag = "stale"
        stale_desc.cache_type = KVCacheSpecType.MHA
        model_config.kv_cache_spec_descs = [[stale_desc] for _ in range(4)]
        return model_config

    def _assert_real_post_build(self, model_cls: type[ModelConfigPostBuilder]) -> None:
        model_config = self._model_config()

        result = apply_layer_override_and_post_build(
            model_config, model_cls, {"HACK_LAYER_NUM": "2"}
        )

        self.assertIs(result, model_config)
        self.assertEqual(result.num_layers, 2)
        self.assertEqual(result.moe_layer_index, [1])
        self.assertEqual(
            result.hybrid_attention_config.hybrid_attention_types,
            [HybridAttentionType.LINEAR, HybridAttentionType.NONE],
        )
        self.assertEqual(len(result.kv_cache_spec_descs), 2)
        self.assertEqual(
            [layer_descs[0].tag for layer_descs in result.kv_cache_spec_descs],
            ["linear0", "full"],
        )
        self.assertEqual(
            [layer_descs[0].cache_type for layer_descs in result.kv_cache_spec_descs],
            [KVCacheSpecType.LINEAR, KVCacheSpecType.MHA],
        )
        validate_basic_config(result)

    def test_qwen3_next_real_post_build_uses_overridden_layer_metadata(self):
        self._assert_real_post_build(Qwen3Next)

    def test_kimi_real_post_build_replaces_stale_layer_metadata(self):
        self._assert_real_post_build(KimiLinear)

    def test_qwen3_next_linear_only_override_is_rejected_by_cpp_creator(self):
        model_config = self._model_config()
        model_config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.LINEAR,
            HybridAttentionType.NONE,
        ]
        result = apply_layer_override_and_post_build(
            model_config, Qwen3Next, {"HACK_LAYER_NUM": "2"}
        )

        with self.assertRaisesRegex(
            RuntimeError, "exactly one FULL MHA/MLA cache group"
        ):
            validate_basic_config(result)

    def test_invalid_layer_override_preserves_metadata_and_can_retry(self):
        model_config = self._model_config()
        with self.assertRaisesRegex(ValueError, "layer override must be"):
            apply_layer_override_and_post_build(
                model_config, Qwen3Next, {"HACK_LAYER_NUM": "5"}
            )

        self.assertEqual(model_config.num_layers, 4)
        self.assertEqual(
            model_config.hybrid_attention_config.hybrid_attention_types,
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
            ],
        )
        self.assertEqual(
            [layer_descs[0].tag for layer_descs in model_config.kv_cache_spec_descs],
            ["stale"] * 4,
        )

        apply_layer_override_and_post_build(
            model_config, Qwen3Next, {"HACK_LAYER_NUM": "2"}
        )
        self.assertEqual(model_config.num_layers, 2)
        self.assertEqual(len(model_config.kv_cache_spec_descs), 2)


if __name__ == "__main__":
    unittest.main()
