import tempfile
import unittest

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3, KimiK3ModelConfig
from rtp_llm.ops import DataType, KvCacheDataType


class KimiK3Fp8PrecisionTest(unittest.TestCase):
    def _config(self, model_type, quantization, cache):
        config = KimiK3ModelConfig()
        config.model_type = model_type
        config.config_dtype = "bfloat16"
        config.quantization = quantization
        config.attn_config.use_mla = True
        config.num_layers = 4
        KimiK3._parse_hybrid_attention_config(
            {"linear_attn_config": {"kda_layers": [1, 2, 3], "full_attn_layers": [4]}},
            config,
        )
        kv_config = KVCacheConfig()
        kv_config.fp8_kv_cache = int(cache)
        with tempfile.TemporaryDirectory() as checkpoint:
            config.ckpt_path = checkpoint
            config.init_precision_config(kv_config, "BF16")
        # ModelFactory invokes this hook after build_model_config(), which
        # invokes init_precision_config(). Preserve that production order.
        KimiK3._post_build_model_config(config)
        return config

    def test_target_fp8_gemm_and_mla_keep_bf16_compute(self):
        config = self._config("kimi_k3", "FP8_PER_BLOCK", "1")
        self.assertIsNotNone(config.attention_projection_quant_config)
        self.assertIsNone(config.quant_config)
        self.assertTrue(config.attn_config.mla_fp8_compute)
        self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.FP8)
        self.assertEqual(config.attn_config.mla_fp8_q_scale, 1.0)
        self.assertEqual(config.attn_config.mla_fp8_kv_scale, 1.0)

    def test_draft_stays_bf16_with_target_fp8_cache_setting(self):
        config = self._config("kimi_k3_mtp", None, "1")
        self.assertIsNone(config.attention_projection_quant_config)
        self.assertFalse(config.attn_config.mla_fp8_compute)
        self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_cache_dtype_selects_matching_mla_compute(self):
        config = self._config("kimi_k3", None, "1")
        self.assertTrue(config.attn_config.mla_fp8_compute)
        self.assertIsNone(config.attention_projection_quant_config)

    def test_projection_quantization_can_keep_bf16_cache(self):
        config = self._config("kimi_k3", "FP8_PER_BLOCK", "0")
        self.assertIsNotNone(config.attention_projection_quant_config)
        self.assertFalse(config.attn_config.mla_fp8_compute)
        self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_reject_other_global_quantization_and_draft_quantization(self):
        with self.assertRaisesRegex(ValueError, "FP8_PER_BLOCK"):
            self._config("kimi_k3", "FP8", "0")
        with self.assertRaisesRegex(ValueError, "draft"):
            self._config("kimi_k3_mtp", "FP8_PER_BLOCK", "0")

    def test_full_group_declares_ordinary_e4m3_only(self):
        config = self._config("kimi_k3", None, "1")
        linear = config.kv_cache_spec_descs[0][0]
        full = config.kv_cache_spec_descs[3][0]
        self.assertFalse(linear.mla_fp8_e4m3)
        self.assertEqual(linear.dtype, DataType.TYPE_BF16)
        self.assertTrue(full.mla_fp8_e4m3)
        self.assertEqual(full.dtype, DataType.TYPE_FP8_E4M3)


if __name__ == "__main__":
    unittest.main()
