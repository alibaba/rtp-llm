import pickle
import unittest
from tempfile import TemporaryDirectory

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory import _resolve_propose_kv_cache_dtype
from rtp_llm.ops import KvCacheDataType, SpeculativeExecutionConfig


class ModelFactorySpeculativeKVCacheTest(unittest.TestCase):
    def setUp(self):
        self.sp_config = SpeculativeExecutionConfig()

    def test_default_has_no_draft_dtype_override(self):
        self.assertEqual(self.sp_config.fp8_kv_cache, -1)
        self.assertIsNone(_resolve_propose_kv_cache_dtype(self.sp_config))

    def test_explicit_bf16_resolves_to_base_dtype(self):
        self.sp_config.fp8_kv_cache = 0
        self.assertEqual(
            _resolve_propose_kv_cache_dtype(self.sp_config),
            KvCacheDataType.BASE,
        )

    def test_explicit_fp8_resolves_to_fp8_dtype(self):
        self.sp_config.fp8_kv_cache = 1
        self.assertEqual(
            _resolve_propose_kv_cache_dtype(self.sp_config),
            KvCacheDataType.FP8,
        )

    def test_invalid_override_fails(self):
        self.sp_config.fp8_kv_cache = 2
        with self.assertRaisesRegex(ValueError, "must be -1, 0 or 1"):
            _resolve_propose_kv_cache_dtype(self.sp_config)

    def test_target_fp8_and_draft_bf16_precision_are_independent(self):
        target_kv_cache_config = KVCacheConfig()
        target_kv_cache_config.fp8_kv_cache = 1
        draft_config = ModelConfig()
        with TemporaryDirectory() as tmpdir:
            draft_config.ckpt_path = tmpdir
            draft_config.init_precision_config(
                kv_cache_config=target_kv_cache_config,
                act_type="BF16",
                kv_cache_dtype_override=KvCacheDataType.BASE,
            )

        self.assertEqual(target_kv_cache_config.fp8_kv_cache, 1)
        self.assertEqual(
            draft_config.attn_config.kv_cache_dtype,
            KvCacheDataType.BASE,
        )

    def test_speculative_kv_cache_config_survives_pickle(self):
        self.sp_config.fp8_kv_cache = 0

        restored = pickle.loads(pickle.dumps(self.sp_config))

        self.assertEqual(restored.fp8_kv_cache, 0)


if __name__ == "__main__":
    unittest.main()
