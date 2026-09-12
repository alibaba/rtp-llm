import os
import unittest
from unittest import mock

from rtp_llm.models.kimi_k3.kimi_k3 import (
    KimiK3,
    KimiK3ModelConfig,
    _mla_prefill_expanded_kv_budget_bytes,
)


class KimiK3MLAWorkspaceConfigTest(unittest.TestCase):
    _BUDGET_ENV = "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_BYTES"

    @staticmethod
    def _parse_budget_bytes() -> int:
        config = KimiK3ModelConfig()
        KimiK3._parse_attention_config(
            {
                "num_attention_heads": 96,
                "num_key_value_heads": 96,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
                "kv_lora_rank": 512,
                "linear_attn_config": {"num_heads": 96, "head_dim": 128},
            },
            config,
        )
        return config.attn_config.mla_prefill_expanded_kv_budget_bytes

    def test_k3_defaults_to_disabled_expanded_kv_planner(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._BUDGET_ENV, None)
            self.assertEqual(_mla_prefill_expanded_kv_budget_bytes(), 0)
            self.assertEqual(self._parse_budget_bytes(), 0)

    def test_explicit_budget_and_zero_disable_are_forwarded(self) -> None:
        for raw, expected in (("1073741824", 1024**3), ("0", 0)):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    self.assertEqual(self._parse_budget_bytes(), expected)

    def test_invalid_budget_is_rejected(self) -> None:
        for raw in ("-1", "invalid"):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        _mla_prefill_expanded_kv_budget_bytes()


class KimiK3MLAFp8ConfigTest(unittest.TestCase):
    def _config(self, model_type="kimi_k3", **env):
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.ops import KvCacheDataType
        config = KimiK3ModelConfig()
        config.model_type = model_type
        config.data_type = "bf16"
        config.config_dtype = "bfloat16"
        config.quant_config = None
        config.attn_config.use_mla = True
        config.attn_config.kv_cache_dtype = KvCacheDataType.BASE
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch.object(ModelConfig, "init_precision_config", return_value=None):
                config.init_precision_config(None, None)
        return config

    def test_default_and_draft_are_unchanged(self):
        from rtp_llm.ops import KvCacheDataType
        for config in (self._config(), self._config("kimi_k3_mla_swa_eagle3", KIMI_K3_MLA_FP8="1")):
            self.assertFalse(config.attn_config.mla_fp8_compute)
            self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_weight_and_mla_quantization_are_independent(self):
        from rtp_llm.ops import KvCacheDataType
        for method in ("none", "fp8_per_block"):
            config = self._config(KIMI_K3_MLA_FP8="1", KIMI_K3_ATTENTION_QUANTIZATION=method,
                                  KIMI_K3_MLA_FP8_Q_SCALE="0.5", KIMI_K3_MLA_FP8_KV_SCALE="0.25")
            self.assertTrue(config.attn_config.mla_fp8_compute)
            self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.FP8)
            self.assertEqual(config.attn_config.mla_fp8_q_scale, 0.5)
            self.assertEqual(config.attn_config.mla_fp8_kv_scale, 0.25)
            self.assertEqual(config.k3_attention_quant_config is not None, method != "none")
            self.assertIsNone(config.quant_config)

    def test_all_four_switch_combinations_and_draft_isolation(self):
        from rtp_llm.ops import KvCacheDataType
        for weight in ("none", "fp8_per_block"):
            for mla in ("0", "1"):
                for model in ("kimi_k3", "kimi_k3_mtp", "kimi_k3_mla_swa_eagle3"):
                    with self.subTest(weight=weight, mla=mla, model=model):
                        config = self._config(model, KIMI_K3_ATTENTION_QUANTIZATION=weight,
                                              KIMI_K3_MLA_FP8=mla)
                        target = model == "kimi_k3"
                        self.assertEqual(config.k3_attention_quant_config is not None,
                                         target and weight == "fp8_per_block")
                        enabled = target and mla == "1"
                        self.assertEqual(config.attn_config.mla_fp8_compute, enabled)
                        self.assertEqual(config.attn_config.kv_cache_dtype,
                                         KvCacheDataType.FP8 if enabled else KvCacheDataType.BASE)
                        self.assertIsNone(config.quant_config)

    def test_real_common_initialization_isolates_mtp_and_shared_cache(self):
        import itertools
        import tempfile
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.ops import KvCacheDataType
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        import torch

        # Exercise the production base initializer, including checkpoint lookup,
        # target ACT_TYPE and global KV flags. The objects share one KV config.
        models = ("kimi_k3", "kimi_k3_mtp", "kimi_k3_mla_swa_eagle3")
        with tempfile.TemporaryDirectory() as checkpoint:
            for method, mla, cache_flags, order in itertools.product(
                ("none", "fp8_per_block"), ("0", "1"),
                ((False, False), (True, False), (False, True), (True, True)),
                itertools.permutations(models),
            ):
                with self.subTest(method=method, mla=mla, cache=cache_flags, order=order):
                    cache = KVCacheConfig()
                    cache.fp8_kv_cache, cache.int8_kv_cache = cache_flags
                    configs = {}
                    with mock.patch.dict(os.environ, {
                        "KIMI_K3_ATTENTION_QUANTIZATION": method,
                        "KIMI_K3_MLA_FP8": mla,
                    }, clear=True):
                        for model in order:
                            config = KimiK3ModelConfig()
                            config.model_type = model
                            config.config_dtype = "bfloat16"
                            config.ckpt_path = checkpoint
                            config.attn_config.use_mla = True
                            if model == "kimi_k3_mtp":
                                # Simulate stale state from an earlier setup.
                                config.k3_attention_quant_config = Fp8BlockWiseQuantConfig()
                                config.quant_config = Fp8BlockWiseQuantConfig()
                                config.quant_algo.setQuantAlgo("fp8", 8, 128)
                                config.attn_config.mla_fp8_compute = True
                                config.attn_config.kv_cache_dtype = KvCacheDataType.INT8
                            if model == "kimi_k3" and mla == "1" and cache.int8_kv_cache:
                                with self.assertRaisesRegex(ValueError, "incompatible with INT8"):
                                    config.init_precision_config(cache, "BF16")
                                continue
                            for _ in range(2):
                                config.init_precision_config(cache, "FP16" if model == "kimi_k3_mtp" else "BF16")
                                if model == "kimi_k3_mtp":
                                    self.assertEqual(config.compute_dtype, torch.bfloat16)
                                    self.assertIsNone(config.quant_config)
                                    self.assertFalse(config.quant_algo.isQuant())
                                    self.assertIsNone(config.k3_attention_quant_config)
                                    self.assertFalse(config.attn_config.mla_fp8_compute)
                                    self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)
                            configs[model] = config
                        for model, config in configs.items():
                            expected = KvCacheDataType.BASE
                            if model != "kimi_k3_mtp":
                                if cache.int8_kv_cache:
                                    expected = KvCacheDataType.INT8
                                elif cache.fp8_kv_cache or (model == "kimi_k3" and mla == "1"):
                                    expected = KvCacheDataType.FP8
                            self.assertEqual(config.attn_config.kv_cache_dtype, expected)
                            self.assertEqual(config.k3_attention_quant_config is not None,
                                             model == "kimi_k3" and method == "fp8_per_block")
                    self.assertEqual((cache.fp8_kv_cache, cache.int8_kv_cache), cache_flags)

    def test_mtp_rejects_non_native_dtype_and_runtime_quantization(self):
        import tempfile
        from rtp_llm.config.kv_cache_config import KVCacheConfig
        with tempfile.TemporaryDirectory() as checkpoint:
            config = KimiK3ModelConfig()
            config.model_type = "kimi_k3_mtp"
            config.ckpt_path = checkpoint
            for dtype in (None, "float16", "float32", "fp8"):
                config.config_dtype = dtype
                with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, "checkpoint-native BF16"):
                    config.init_precision_config(KVCacheConfig(), "BF16")
            config.config_dtype = "bfloat16"
            config.quantization = "FP8_PER_BLOCK"
            with self.assertRaisesRegex(ValueError, "runtime weight quantization"):
                config.init_precision_config(KVCacheConfig(), None)
            config.quantization = ""
            config.init_precision_config(KVCacheConfig(), None)
            self.assertFalse(config.quant_algo.isQuant())
            self.assertIsNone(config.quant_config)

    def test_invalid_scales_and_switch_fail_at_initialization(self):
        for value in ("0", "-1", "nan", "inf", "1e-100", "1e100"):
            with self.subTest(scale=value), self.assertRaises(ValueError):
                self._config(KIMI_K3_MLA_FP8="1", KIMI_K3_MLA_FP8_KV_SCALE=value)
        with self.assertRaises(ValueError):
            self._config(KIMI_K3_MLA_FP8="true")


if __name__ == "__main__":
    unittest.main()
