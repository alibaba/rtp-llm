import pickle
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.minimax_m3 import MiniMaxM3Weight
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeModel
from rtp_llm.models_py.model_desc.minimax_m31 import (
    MiniMaxM31Model,
    _MiniMaxM31MSAQueryContext,
)
from rtp_llm.ops import KVCacheConfig, KvCacheDataType, TaskType
from rtp_llm.utils.model_weight import W


class MiniMaxM3IndexWeightTest(unittest.TestCase):
    def test_raw_mxfp8_idx_follows_checkpoint_scales(self):
        weight = object.__new__(MiniMaxM3Weight)
        weight.prefix = "language_model."
        weight._hidden_size = 512
        weight._size_per_head = 128
        weight._head_num = 4
        weight._head_num_kv = 4
        weight._use_qk_norm = True
        keys = {
            f"language_model.model.layers.{layer}.self_attn.index_{name}_proj.weight"
            for layer in (1, 2)
            for name in ("q", "k")
        }
        keys.update(
            f"language_model.model.layers.1.self_attn.index_{name}_proj.weight_scale_inv"
            for name in ("q", "k")
        )
        weight._process_meta({}, keys)
        self.assertEqual(weight._raw_mxfp8_idx_layers, {1})
        with patch.object(
            MiniMaxM3Weight, "_get_hf_ffn_layer_weight_info", return_value=[]
        ):
            layer_1 = weight._get_hf_layer_weight_info(1)
            layer_2 = weight._get_hf_layer_weight_info(2)

        raw = {
            W.msa_idx_q_raw_w,
            W.msa_idx_q_raw_s,
            W.msa_idx_k_raw_w,
            W.msa_idx_k_raw_s,
        }
        self.assertTrue(raw <= {module.name for module in layer_1})
        self.assertFalse(raw & {module.name for module in layer_2})
        self.assertNotIn(W.msa_idx_q_w, {module.name for module in layer_1})
        self.assertIn(W.msa_idx_q_w, {module.name for module in layer_2})

        keys.add(
            "language_model.model.layers.2.self_attn.index_q_proj.weight_scale_inv"
        )
        with self.assertRaisesRegex(ValueError, "incomplete MXFP8 index scales"):
            weight._process_meta({}, keys)

    def test_index_weights_follow_checkpoint_sparse_layers(self):
        weight = object.__new__(MiniMaxM3Weight)
        weight.prefix = "language_model."
        weight._hidden_size = 512
        weight._size_per_head = 128
        weight._head_num = 4
        weight._head_num_kv = 4
        weight._use_qk_norm = True
        weight._raw_mxfp8_idx_layers = set()
        keys = {
            "language_model.model.layers.1.self_attn.index_q_proj.weight",
            "language_model.model.layers.1.self_attn.index_k_proj.weight",
            "language_model.model.mtp.layers.0.transformer_layer.self_attn.index_q_proj.weight",
        }

        weight._process_meta({}, keys)
        with patch.object(
            MiniMaxM3Weight, "_get_hf_ffn_layer_weight_info", return_value=[]
        ):
            dense = weight._get_hf_layer_weight_info(0)
            sparse = weight._get_hf_layer_weight_info(1)

        self.assertEqual(weight._sparse_layer_set, {1})
        self.assertNotIn(W.msa_idx_q_w, {module.name for module in dense})
        self.assertTrue(
            {W.msa_idx_q_w, W.msa_idx_k_w, W.msa_idx_q_norm, W.msa_idx_k_norm}
            <= {module.name for module in sparse}
        )


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
