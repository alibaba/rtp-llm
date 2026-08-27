import copy
import unittest

import torch

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory_register import (
    _tokenizer_factory,
    ensure_tokenizer_registered,
)
from rtp_llm.model_factory_register import ModelDict, ensure_model_registered
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.glm5_3_flash import Glm53Flash
from rtp_llm.models.glm5_3_flash_weight import Glm53FlashWeight
from rtp_llm.openai.renderer_factory_register import (
    _renderer_factory,
    ensure_renderer_registered,
)
from rtp_llm.ops import DataType, HybridAttentionType
from rtp_llm.utils.model_weight import W


def released_text_config():
    full_layers = list(range(3, 44, 4))
    kda_layers = [layer for layer in range(45) if layer not in full_layers]
    return {
        "model_type": "glm5_next",
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "hidden_size": 4096,
            "vocab_size": 154880,
            "max_position_embeddings": 1048576,
            "intermediate_size": 12288,
            "rms_norm_eps": 1e-5,
            "dtype": "bfloat16",
            "eos_token_id": [154820, 154827, 154829],
            "pad_token_id": 154820,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "qk_nope_head_dim": 256,
            "qk_rope_head_dim": 0,
            "v_head_dim": 256,
            "q_lora_rank": 1536,
            "kv_lora_rank": 512,
            "mla_use_nope": True,
            "indexer_rope_interleave": True,
            "index_kpool": 4,
            "index_topk": 2048,
            "index_kpool_compress": True,
            "index_kpool_always_select_tail": True,
            "index_share_for_mtp_iteration": True,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "hidden_act": "silu",
            "scoring_func": "sigmoid",
            "n_routed_experts": 288,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 2.5,
            "norm_topk_prob": True,
            "n_shared_experts": 1,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42,
            "layer_types": [
                (
                    "deepseek_sparse_attention"
                    if layer in full_layers
                    else "linear_attention"
                )
                for layer in range(45)
            ],
            "linear_attn_config": {
                "num_heads": 64,
                "gate_lower_bound": -5.0,
                "head_dim": 128,
                "short_conv_kernel_size": 4,
                "kda_layers": kda_layers,
                "full_attn_layers": full_layers,
            },
            "mhc": True,
            "hc_mult": 4,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "swiglu_limit": 10.0,
        },
    }


class Glm53FlashConfigTest(unittest.TestCase):
    def test_supports_cuda_graph(self):
        self.assertTrue(Glm53Flash.support_cuda_graph(None))

    def test_unscaled_indexer_tensors_remain_bf16(self):
        names = {
            W.mla_indexer_qb_w,
            W.mla_indexer_k_w,
            W.mla_indexer_k_norm_w,
            W.mla_indexer_k_norm_b,
            W.mla_indexer_weights_proj_w,
            W.mla_indexer_kpool_gate_w,
            W.mla_indexer_kpool_ape,
        }
        modules = [AtomicWeight(name, [], data_type=torch.float32) for name in names]
        Glm53FlashWeight._mark_checkpoint_bf16(modules)
        for module in modules:
            self.assertTrue(module.skip_quantization)
            self.assertEqual(module.data_type, torch.bfloat16)

    def test_kda_state_cache_is_always_fp32(self):
        config = Glm53Flash._from_config_json(released_text_config())
        kv_cache_config = type("KVCacheConfig", (), {"ssm_state_dtype": "bf16"})()
        config.init_linear_attention_cache_precision(kv_cache_config)
        self.assertEqual(
            config.linear_attention_config.ssm_state_dtype, DataType.TYPE_FP32
        )

    def test_mhc_projection_weights_load_as_fp32(self):
        descriptor = Glm53FlashWeight.__new__(Glm53FlashWeight)
        modules = {module.name: module for module in descriptor._mhc_weights()}
        self.assertEqual(modules[W.v4_hc_attn_fn].data_type, torch.float32)
        self.assertEqual(modules[W.v4_hc_ffn_fn].data_type, torch.float32)

    def test_released_flash_geometry(self):
        config = Glm53Flash._from_config_json(released_text_config())

        self.assertEqual(config.model_type, "glm5_3_flash")
        self.assertEqual(config.num_layers, 45)
        self.assertEqual(config.hidden_size, 4096)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.attn_config.indexer_head_num, 32)
        self.assertEqual(config.attn_config.indexer_head_dim, 128)
        self.assertEqual(config.attn_config.indexer_topk, 512)
        self.assertEqual(config.attn_config.indexer_compress_ratio, 4)
        self.assertEqual(config.attn_config.sparse_attention_topk, 2051)
        self.assertEqual(config.attn_config.indexer_layer_ids, list(range(3, 44, 4)))
        self.assertEqual(config.moe_layer_index, list(range(3, 45)))
        self.assertEqual(config.swiglu_limit, 10.0)

        kinds = config.hybrid_attention_config.hybrid_attention_types
        self.assertEqual(kinds.count(HybridAttentionType.LINEAR), 34)
        self.assertEqual(len(kinds) - kinds.count(HybridAttentionType.LINEAR), 11)

    def test_rejects_disabled_mhc(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["mhc"] = False
        with self.assertRaisesRegex(ValueError, "mhc=true"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_disabled_tail_selection(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["index_kpool_always_select_tail"] = False
        with self.assertRaisesRegex(ValueError, "always_select_tail=true"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_incomplete_attention_schedule(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["linear_attn_config"]["kda_layers"].remove(44)
        with self.assertRaisesRegex(ValueError, "miss layers.*44"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_empty_eos_token_list(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["eos_token_id"] = []
        with self.assertRaisesRegex(ValueError, "at least one token"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_zero_index_topk(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["index_topk"] = 0
        with self.assertRaisesRegex(ValueError, "index_topk must be positive"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_attention_schedule_without_kda(self):
        raw = copy.deepcopy(released_text_config())
        layers = list(range(raw["text_config"]["num_hidden_layers"]))
        linear = raw["text_config"]["linear_attn_config"]
        linear["kda_layers"] = []
        linear["full_attn_layers"] = layers
        raw["text_config"]["layer_types"] = ["deepseek_sparse_attention"] * len(layers)
        with self.assertRaisesRegex(ValueError, "at least one KDA layer"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_attention_schedule_without_sparse_mla(self):
        raw = copy.deepcopy(released_text_config())
        layers = list(range(raw["text_config"]["num_hidden_layers"]))
        linear = raw["text_config"]["linear_attn_config"]
        linear["kda_layers"] = layers
        linear["full_attn_layers"] = []
        raw["text_config"]["layer_types"] = ["linear_attention"] * len(layers)
        with self.assertRaisesRegex(ValueError, "at least one sparse MLA layer"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_invalid_moe_group_geometry(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["n_group"] = 5
        with self.assertRaisesRegex(ValueError, "must be divisible by n_group"):
            Glm53Flash._from_config_json(raw)

    def test_zero_swiglu_limit_explicitly_disables_clamp(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["swiglu_limit"] = 0.0
        config = Glm53Flash._from_config_json(raw)
        self.assertEqual(config.swiglu_limit, 0.0)
        self.assertEqual(config.glm5_3_flash_runtime_config.swiglu_limit, 0.0)

    def test_rejects_negative_swiglu_limit(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["swiglu_limit"] = -1.0
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            Glm53Flash._from_config_json(raw)

    def test_rejects_zero_shared_experts(self):
        raw = copy.deepcopy(released_text_config())
        raw["text_config"]["n_shared_experts"] = 0
        with self.assertRaisesRegex(ValueError, "n_shared_experts must be positive"):
            Glm53Flash._from_config_json(raw)

    def test_glm53_flash_uses_tokenizers_backend_compatibility_loader(self):
        self.assertTrue(ensure_tokenizer_registered("glm5_3_flash"))
        self.assertEqual(
            _tokenizer_factory["glm5_3_flash"].__name__, "ChatGLMV5Tokenizer"
        )

    def test_rtp_registries_expose_glm53_flash_name(self):
        self.assertTrue(ensure_model_registered("glm5_3_flash"))
        self.assertEqual(
            ModelDict.get_ft_model_type_by_config(released_text_config()),
            "glm5_3_flash",
        )
        self.assertTrue(ensure_renderer_registered("glm5_3_flash"))
        self.assertEqual(
            _renderer_factory["glm5_3_flash"].__name__, "ChatGlm47Renderer"
        )


if __name__ == "__main__":
    unittest.main()
