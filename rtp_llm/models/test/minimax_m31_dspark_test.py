"""CPU checks for M3.1 DSpARK checkpoint and main model ownership."""

import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.models.minimax_m31 import MiniMaxM31
from rtp_llm.models.minimax_m31_dspark import MiniMaxM31DSpark, MiniMaxM31DSparkWeight
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


def draft_config():
    return {
        "architectures": ["DSparkMiniMaxDraftModel"],
        "quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]},
        "text_config": {
            "layer_types": ["sliding_attention"] * 5,
            "moe_layer_freq": [0] * 5,
            "use_gemma_norm": True,
            "hidden_size": 6144,
            "vocab_size": 200064,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "dense_intermediate_size": 12288,
            "dspark_noise_token_id": 200058,
            "dspark_target_layer_ids": [3, 17, 31, 45, 59],
            "dspark_markov_rank": 256,
            "dspark_block_size": 7,
            "sliding_window": 4096,
        },
    }


class MiniMaxM31DSparkTest(unittest.TestCase):
    def test_dense_draft_uses_main_cache_and_weight_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps(draft_config()))
            config = MiniMaxM31DSpark._create_config(directory)
            kv_config = KVCacheConfig()
            kv_config.seq_size_per_block = 64
            config.init_precision_config(kv_config, "BF16")
            config.attn_config.tokens_per_block = kv_config.seq_size_per_block
            config.attn_config.kernel_tokens_per_block = kv_config.seq_size_per_block
            MiniMaxM31DSpark._apply_kv_cache_config(config, kv_config)
            MiniMaxM31DSpark._post_build_model_config(config)
            self.assertEqual(config.attn_config.sliding_window, 4096)
            self.assertEqual(config.dspark_target_layer_ids, [3, 17, 31, 45, 59])
            self.assertEqual(
                [[desc.tag for desc in layer] for layer in config.kv_cache_spec_descs],
                [["m31_kv"]] * 5,
            )
            self.assertEqual(config.attn_config.tokens_per_block, 128)
            self.assertEqual(config.attn_config.kernel_tokens_per_block, 128)
            weights = MiniMaxM31DSparkWeight(
                config, ParallelismConfig(), HWKernelConfig(), KVCacheConfig()
            )._get_weight_info()
            self.assertEqual(len(weights.layer_weights), 5)
            self.assertEqual(
                {weight.name for weight in weights.weights} & {W.embedding, W.lm_head},
                {W.embedding, W.lm_head},
            )
            first_sources = [
                info.name
                for weight in weights.layer_weights[0]
                for component in weight.get_components()
                for info in getattr(component, "weights", ())
            ]
            self.assertIn(
                "language_model.model.dspark.layers.{i}.decoder_layer.self_attn.q_proj.weight",
                first_sources,
            )
            self.assertFalse(any("index_" in name for name in first_sources))
            target = MiniMaxM31.__new__(MiniMaxM31)
            target.model_config = config
            self.assertEqual(
                MiniMaxM31DSpark.speculative_weight_alias_names(target, config),
                (W.embedding, W.lm_head),
            )

    def test_mxfp8_kernel_imports_on_current_main(self):
        from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
            mxfp8_quant_act_triton,
        )

        self.assertTrue(callable(mxfp8_quant_act_triton))

    def test_rejects_sparse_draft_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            value = draft_config()
            value["text_config"]["moe_layer_freq"][0] = 1
            Path(directory, "config.json").write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "dense Gemma"):
                MiniMaxM31DSpark.create_config(directory)


if __name__ == "__main__":
    unittest.main()
