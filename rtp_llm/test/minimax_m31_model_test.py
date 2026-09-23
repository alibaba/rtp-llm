import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.minimax_m31 import (
    MiniMaxM31Model,
    _MiniMaxM31MSAQueryContext,
)
from rtp_llm.ops import KVCacheConfig, KvCacheDataType, TaskType


class MiniMaxM31PrepareTest(unittest.TestCase):
    def test_all_sparse_model_never_builds_generic_fmha_context(self):
        model = object.__new__(MiniMaxM31Model)

        for is_target_verify in (False, True):
            inputs = SimpleNamespace(
                attention_inputs=SimpleNamespace(
                    is_target_verify=is_target_verify,
                )
            )
            with self.subTest(is_target_verify=is_target_verify), patch.object(
                GenericMoeModel,
                "prepare_fmha_impl",
                side_effect=AssertionError("generic FMHA prepare must not run"),
            ):
                actual = model.prepare_fmha_impl(inputs, is_cuda_graph=True)

            self.assertIsInstance(actual, _MiniMaxM31MSAQueryContext)
            self.assertIsNone(actual.fmha_params)
            self.assertTrue(actual.support_cuda_graph())


class KVCacheEstimateTest(unittest.TestCase):
    @staticmethod
    def _estimate(kv_dtype, indexer_mode, nvfp4=False):
        config = SimpleNamespace(
            task_type=TaskType.LANGUAGE_MODEL,
            num_layers=2,
            max_seq_len=10,
            attn_config=SimpleNamespace(
                kv_cache_dtype=kv_dtype,
                nvfp4_kv_cache=nvfp4,
                kv_head_num=2,
                size_per_head=16,
                indexer_head_dim=16,
                indexer_cache_fp8_mode=indexer_mode,
            ),
        )
        return ModelConfig._eval_kv_cache_mem_size(config)

    def test_estimate_includes_indexer_sidecar(self):
        self.assertEqual(self._estimate(KvCacheDataType.BASE, 0), 3200)
        self.assertEqual(self._estimate(KvCacheDataType.FP8, 1), 1680)
        self.assertEqual(self._estimate(KvCacheDataType.BASE, 3, nvfp4=True), 900)

    def test_nvfp4_kv_cache_config_pickle_round_trip(self):
        config = KVCacheConfig()
        config.nvfp4_kv_cache = 1

        restored = pickle.loads(pickle.dumps(config))

        self.assertEqual(restored.nvfp4_kv_cache, 1)


if __name__ == "__main__":
    unittest.main()
