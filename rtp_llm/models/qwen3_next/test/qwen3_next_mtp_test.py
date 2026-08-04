import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.model_factory_register import _model_factory
from rtp_llm.model_loader.ffn_weight import FfnWeight, MoeWeight
from rtp_llm.models.qwen3_next.qwen3_next_mtp import (
    Qwen35DenseMTP,
    Qwen35DenseMTPWeight,
)
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeMixin,
)
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    ParallelismConfig,
    RopeStyle,
)


class Qwen35DenseMTPTest(unittest.TestCase):
    def test_dense_mtp_config_and_registration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(self._config()))

            config = Qwen35DenseMTP.create_config(temp_dir)

        self.assertIs(_model_factory["qwen35_dense_mtp"], Qwen35DenseMTP)
        self.assertEqual(config.num_layers, 1)
        self.assertTrue(config.is_mtp)
        self.assertEqual(config.moe_style, 0)
        self.assertEqual(list(config.moe_layer_index), [])
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [HybridAttentionType.NONE],
        )
        self.assertEqual(config.attn_config.rope_config.style, RopeStyle.Base)
        self.assertEqual(len(config.kv_cache_spec_descs), 1)

    def test_dense_mtp_uses_dense_ffn_checkpoint_keys(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(self._config()))
            config = Qwen35DenseMTP.create_config(temp_dir)
            weight = Qwen35DenseMTPWeight(
                model_config=config,
                parallelism_config=ParallelismConfig(),
                hw_kernel_config=HWKernelConfig(),
                kv_cache_config=KVCacheConfig(),
            )

        modules = weight._create_ffn_weight()

        self.assertEqual(weight.prefix, "mtp.")
        self.assertEqual(weight.model_prefix, "model.language_model.")
        self.assertEqual(len(modules), 1)
        self.assertIsInstance(modules[0], FfnWeight)
        self.assertNotIsInstance(modules[0], MoeWeight)
        ffn = modules[0]
        checkpoint_keys = {
            info.name
            for atomic_weight in (ffn.origin_w1, ffn.origin_w3, ffn.w2)
            for info in atomic_weight.weights
        }
        self.assertEqual(
            checkpoint_keys,
            {
                "mtp.layers.{i}.mlp.gate_proj.weight",
                "mtp.layers.{i}.mlp.up_proj.weight",
                "mtp.layers.{i}.mlp.down_proj.weight",
            },
        )

    def test_dense_mtp_uses_qwen35_multimodal_mixin(self):
        self.assertIs(
            get_multimodal_mixin_cls("qwen35_dense_mtp"), Qwen3_5MoeMixin
        )

    @staticmethod
    def _config():
        return {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
            "text_config": {
                "num_attention_heads": 24,
                "num_key_value_heads": 4,
                "head_dim": 256,
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "rms_norm_eps": 1e-6,
                "intermediate_size": 17408,
                "full_attention_interval": 4,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_value_head_dim": 128,
                "rope_parameters": {
                    "mrope_interleaved": True,
                    "mrope_section": [11, 11, 10],
                    "rope_theta": 10000000,
                    "partial_rotary_factor": 0.25,
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
