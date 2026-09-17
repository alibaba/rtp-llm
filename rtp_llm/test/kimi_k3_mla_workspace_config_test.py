import os
import unittest
from unittest import mock

from rtp_llm.models.kimi_k3.kimi_k3 import (
    KimiK3,
    KimiK3ModelConfig,
    _mla_prefill_expanded_kv_budget_gib,
)


class KimiK3MLAWorkspaceConfigTest(unittest.TestCase):
    _BUDGET_ENV = "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_GIB"

    @staticmethod
    def _parse_budget_gib() -> float:
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
        return config.attn_config.mla_prefill_expanded_kv_budget_gib

    def test_k3_defaults_to_six_gib_expanded_kv_budget(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self._BUDGET_ENV, None)
            self.assertEqual(_mla_prefill_expanded_kv_budget_gib(), 6.0)
            self.assertEqual(self._parse_budget_gib(), 6.0)

    def test_explicit_budget_and_zero_disable_are_forwarded(self) -> None:
        for raw, expected in (("1.5", 1.5), ("0", 0.0)):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    self.assertEqual(self._parse_budget_gib(), expected)

    def test_invalid_budget_is_rejected(self) -> None:
        for raw in ("-1", "invalid", "nan", "inf"):
            with self.subTest(raw=raw):
                with mock.patch.dict(
                    os.environ,
                    {self._BUDGET_ENV: raw},
                    clear=False,
                ):
                    with self.assertRaisesRegex(ValueError, "non-negative"):
                        _mla_prefill_expanded_kv_budget_gib()


class KimiK3MLAFp8ConfigTest(unittest.TestCase):
    def _config(self, model_type="kimi_k3", **env):
        import tempfile

        from rtp_llm.config.kv_cache_config import KVCacheConfig

        config = KimiK3ModelConfig()
        config.model_type = model_type
        config.config_dtype = "bfloat16"
        config.attn_config.use_mla = True
        cache = KVCacheConfig()
        cache.fp8_kv_cache = int(env.get("FP8_KV_CACHE", "0"))
        with tempfile.TemporaryDirectory() as checkpoint:
            config.ckpt_path = checkpoint
            with mock.patch.dict(os.environ, env, clear=True):
                config.init_precision_config(cache, "BF16")
        return config

    def test_default_and_draft_are_unchanged(self):
        from rtp_llm.ops import KvCacheDataType

        for config in (
            self._config(),
            self._config("kimi_k3_mtp", FP8_GEMM="1", FP8_KV_CACHE="1", FP8_MLA="1"),
        ):
            self.assertFalse(config.attn_config.mla_fp8_compute)
            self.assertIsNone(config.k3_attention_quant_config)
            self.assertEqual(config.attn_config.kv_cache_dtype, KvCacheDataType.BASE)

    def test_all_eight_combinations_and_draft_isolation(self):
        import itertools

        from rtp_llm.ops import KvCacheDataType

        for gemm, cache, mla, model in itertools.product(
            ("0", "1"),
            ("0", "1"),
            ("0", "1"),
            ("kimi_k3", "kimi_k3_mtp", "kimi_k3_mla_swa_eagle3"),
        ):
            with self.subTest(gemm=gemm, cache=cache, mla=mla, model=model):
                env = dict(FP8_GEMM=gemm, FP8_KV_CACHE=cache, FP8_MLA=mla)
                if model == "kimi_k3" and cache != mla:
                    with self.assertRaisesRegex(ValueError, "requires matching"):
                        self._config(model, **env)
                    continue
                config = self._config(model, **env)
                target = model == "kimi_k3"
                self.assertEqual(
                    config.k3_attention_quant_config is not None,
                    target and gemm == "1",
                )
                self.assertEqual(
                    config.attn_config.mla_fp8_compute, target and mla == "1"
                )
                expected_cache = (
                    KvCacheDataType.FP8
                    if cache == "1" and model != "kimi_k3_mtp"
                    else KvCacheDataType.BASE
                )
                self.assertEqual(config.attn_config.kv_cache_dtype, expected_cache)
                self.assertEqual(config.attn_config.mla_fp8_q_scale, 1.0)
                self.assertEqual(config.attn_config.mla_fp8_kv_scale, 1.0)
                self.assertIsNone(config.quant_config)

    def test_real_common_initialization_isolates_mtp_and_shared_cache(self):
        import itertools
        import tempfile

        import torch

        from rtp_llm.config.kv_cache_config import KVCacheConfig
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig
        from rtp_llm.ops import KvCacheDataType

        models = ("kimi_k3", "kimi_k3_mtp", "kimi_k3_mla_swa_eagle3")
        with tempfile.TemporaryDirectory() as checkpoint:
            for gemm, mla, cache_flags, order in itertools.product(
                ("0", "1"),
                ("0", "1"),
                ((False, False), (True, False), (False, True), (True, True)),
                itertools.permutations(models),
            ):
                with self.subTest(gemm=gemm, mla=mla, cache=cache_flags, order=order):
                    cache = KVCacheConfig()
                    cache.fp8_kv_cache, cache.int8_kv_cache = cache_flags
                    with mock.patch.dict(
                        os.environ, {"FP8_GEMM": gemm, "FP8_MLA": mla}, clear=True
                    ):
                        for model in order:
                            config = KimiK3ModelConfig()
                            config.model_type = model
                            config.config_dtype = "bfloat16"
                            config.ckpt_path = checkpoint
                            config.attn_config.use_mla = True
                            if model == "kimi_k3_mtp":
                                config.k3_attention_quant_config = (
                                    Fp8BlockWiseQuantConfig()
                                )
                                config.quant_config = Fp8BlockWiseQuantConfig()
                                config.quant_algo.setQuantAlgo("fp8", 8, 128)
                                config.attn_config.mla_fp8_compute = True
                                config.attn_config.kv_cache_dtype = KvCacheDataType.INT8
                                config.attn_config.mla_fp8_q_scale = 0.5
                                config.attn_config.mla_fp8_kv_scale = 0.25
                            if model == "kimi_k3":
                                if mla == "1" and cache.int8_kv_cache:
                                    with self.assertRaisesRegex(
                                        ValueError, "incompatible with INT8"
                                    ):
                                        config.init_precision_config(cache, "BF16")
                                    continue
                                effective_fp8 = (
                                    cache.fp8_kv_cache and not cache.int8_kv_cache
                                )
                                if (mla == "1") != effective_fp8:
                                    with self.assertRaisesRegex(
                                        ValueError, "requires matching"
                                    ):
                                        config.init_precision_config(cache, "BF16")
                                    continue
                            for _ in range(2):
                                config.init_precision_config(
                                    cache, "FP16" if model == "kimi_k3_mtp" else "BF16"
                                )
                                self.assertEqual(config.compute_dtype, torch.bfloat16)
                                self.assertEqual(
                                    config.attn_config.mla_fp8_q_scale, 1.0
                                )
                                self.assertEqual(
                                    config.attn_config.mla_fp8_kv_scale, 1.0
                                )
                                expected = KvCacheDataType.BASE
                                if model != "kimi_k3_mtp":
                                    if cache.int8_kv_cache:
                                        expected = KvCacheDataType.INT8
                                    elif cache.fp8_kv_cache:
                                        expected = KvCacheDataType.FP8
                                self.assertEqual(
                                    config.attn_config.kv_cache_dtype, expected
                                )
                                self.assertEqual(
                                    config.k3_attention_quant_config is not None,
                                    model == "kimi_k3" and gemm == "1",
                                )
                                self.assertEqual(
                                    config.attn_config.mla_fp8_compute,
                                    model == "kimi_k3" and mla == "1",
                                )
                                if model == "kimi_k3_mtp":
                                    self.assertIsNone(config.quant_config)
                                    self.assertFalse(config.quant_algo.isQuant())
                    self.assertEqual(
                        (cache.fp8_kv_cache, cache.int8_kv_cache), cache_flags
                    )

    def test_invalid_switches_fail_at_initialization(self):
        for flag in ("FP8_GEMM", "FP8_MLA"):
            for value in ("true", "2", "-1", ""):
                with self.subTest(flag=flag, value=value):
                    with self.assertRaisesRegex(ValueError, flag + " must be 0 or 1"):
                        self._config(**{flag: value})

    def test_mtp_rejects_non_native_dtype_and_runtime_quantization(self):
        import tempfile

        from rtp_llm.config.kv_cache_config import KVCacheConfig

        with tempfile.TemporaryDirectory() as checkpoint:
            config = KimiK3ModelConfig()
            config.model_type = "kimi_k3_mtp"
            config.ckpt_path = checkpoint
            for dtype in (None, "float16", "float32", "fp8"):
                config.config_dtype = dtype
                with self.subTest(dtype=dtype), self.assertRaisesRegex(
                    ValueError, "checkpoint-native BF16"
                ):
                    config.init_precision_config(KVCacheConfig(), "BF16")
            config.config_dtype = "bfloat16"
            config.quantization = "FP8_PER_BLOCK"
            with self.assertRaisesRegex(ValueError, "runtime weight quantization"):
                config.init_precision_config(KVCacheConfig(), None)
            config.quantization = ""
            config.init_precision_config(KVCacheConfig(), None)
            self.assertFalse(config.quant_algo.isQuant())
            self.assertIsNone(config.quant_config)


if __name__ == "__main__":
    unittest.main()
