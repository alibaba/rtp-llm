import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.deepseek_v2 import DeepSeekV2, PackedIndexerWeight, pack_indexer_weight
from rtp_llm.ops import DataType, KVCacheConfig, KvCacheDataType
from rtp_llm.utils.model_weight import CkptWeightInfo


class GlmIndexerWeightTest(unittest.TestCase):
    def test_glm5_indexer_layernorm_eps_and_hybrid_precision_config(self):
        cfg_json = {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "attention_bias": False,
            "first_k_dense_replace": 3,
            "hidden_act": "silu",
            "hidden_size": 6144,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "index_topk": 2048,
            "indexer_rope_interleave": True,
            "intermediate_size": 12288,
            "kv_lora_rank": 512,
            "max_position_embeddings": 202752,
            "model_type": "glm_moe_dsa",
            "moe_intermediate_size": 2048,
            "moe_layer_freq": 1,
            "n_group": 1,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_attention_heads": 64,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 78,
            "num_key_value_heads": 64,
            "num_nextn_predict_layers": 0,
            "q_lora_rank": 2048,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {
                "partial_rotary_factor": 1.0,
                "rope_theta": 8000000.0,
                "rope_type": "default",
            },
            "routed_scaling_factor": 2.5,
            "torch_dtype": "bfloat16",
            "v_head_dim": 256,
            "vocab_size": 154880,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(cfg_json, f)
            config = DeepSeekV2._create_config(tmpdir)
            config.model_type = "glm_5"
            self.assertAlmostEqual(config.layernorm_eps, 1e-5)
            self.assertAlmostEqual(config.indexer_layernorm_eps, 1e-6)
            self.assertFalse(config.attn_config.rope_config.is_neox_style)
            self.assertFalse(config.attn_config.rope_config.indexer_is_neox_style)

            for quant_method in (
                "FP8_PER_BLOCK_NO_MOE",
                "MODELOPT_FP4_HYBRID_FP8_PER_BLOCK",
            ):
                with patch.dict(
                    os.environ,
                    {"MOE_STRATEGY": "mega_moe", "RTP_LLM_MOE_WEIGHT_FORMAT": "nvfp4"},
                ):
                    cfg_copy = DeepSeekV2._create_config(tmpdir)
                    cfg_copy.model_type = "glm_5"
                    cfg_copy.quantization = quant_method
                    kv_cfg = KVCacheConfig()
                    kv_cfg.nvfp4_kv_cache = 1
                    kv_cfg.nvfp4_indexer_kv_cache = 0
                    cfg_copy.init_precision_config(kv_cfg, None)
                    self.assertEqual(cfg_copy.data_type, DataType.TYPE_BF16)
                    self.assertEqual(
                        cfg_copy.attn_config.kv_cache_dtype, KvCacheDataType.NVFP4
                    )
                    self.assertFalse(cfg_copy.attn_config.nvfp4_indexer_kv_cache)

    def test_checkpoint_scale_precision_and_block_order(self):
        generator = torch.Generator().manual_seed(37)
        raw = torch.randn(128, 384, generator=generator).to(torch.float8_e4m3fn)
        gate = torch.randn(32, 384, generator=generator).bfloat16()
        for scale in (
            torch.tensor([[0.01567, 0.02453, 0.03591]]),
            torch.tensor([[0.015625, 0.03125, 0.0625]]).to(torch.float8_e8m0fnu),
        ):
            with self.subTest(dtype=scale.dtype):
                tensors = {"wk": raw, "scale": scale, "gate": gate}
                source = SimpleNamespace(
                    load_tensor=lambda name, dtype: [tensors[name].to(dtype)]
                )
                packed = PackedIndexerWeight(
                    "packed",
                    [CkptWeightInfo(n) for n in tensors],
                    pack_indexer_weight,
                    data_type=torch.float32,
                )
                result = packed._load_raw_tensor(
                    source, 0, "cpu", SimpleNamespace(compute_dtype=torch.bfloat16)
                )["packed"]
                expected = torch.cat(
                    [
                        block.double() * s
                        for block, s in zip(raw.split(128, dim=1), scale.double()[0])
                    ],
                    dim=1,
                ).bfloat16()
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(result[:128], expected))
                self.assertTrue(torch.equal(result[128:], gate))


if __name__ == "__main__":
    unittest.main()
