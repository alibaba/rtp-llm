import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.model_factory_register import (
    ModelDict,
    _hf_architecture_2_ft,
    _lazy_model_registry,
    _model_factory,
    ensure_model_registered,
)
from rtp_llm.models.nemotron_h.nemotron_h import NemotronH, NemotronHBase
from rtp_llm.models.nemotron_h.nemotron_h_weight import NemotronHWeight
from rtp_llm.ops import HybridAttentionType, KVCacheSpecType


def _config_json():
    return {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "hidden_size": 2688,
        "vocab_size": 131072,
        "max_position_embeddings": 1048576,
        "num_hidden_layers": 6,
        "layers_block_type": [
            "mamba",
            "moe",
            "mamba",
            "moe",
            "mamba",
            "attention",
        ],
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "rope_theta": 10000,
        "partial_rotary_factor": 1.0,
        "layer_norm_epsilon": 1e-05,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "ssm_state_size": 128,
        "n_groups": 8,
        "conv_kernel": 4,
        "chunk_size": 128,
        "mamba_ssm_cache_dtype": "float32",
        "mamba_hidden_act": "silu",
        "mlp_hidden_act": "relu2",
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 512,
        "moe_shared_expert_intermediate_size": 1024,
        "n_shared_experts": 1,
        "moe_shared_expert_overlap": True,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "num_nextn_predict_layers": 1,
        "mtp_layers_block_type": ["attention", "moe"],
        "tie_word_embeddings": False,
    }


class NemotronHRegistrationTest(unittest.TestCase):
    def test_lazy_registry_entry(self):
        module_path = _lazy_model_registry.get_module_path("nemotron_h")
        self.assertEqual(module_path, "rtp_llm.models.nemotron_h.nemotron_h")

    def test_architecture_resolution(self):
        model_type = ModelDict.get_ft_model_type_by_config(
            {"architectures": ["NemotronHForCausalLM"]}
        )
        self.assertEqual(model_type, "nemotron_h")
        self.assertIn("NemotronHForCausalLM", _hf_architecture_2_ft)

    def test_module_registration(self):
        self.assertTrue(ensure_model_registered("nemotron_h"))
        self.assertIs(_model_factory["nemotron_h"], NemotronH)


class NemotronHConfigTest(unittest.TestCase):
    def _make_config(self, overrides=None):
        config_json = _config_json()
        if overrides:
            config_json.update(overrides)
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "config.json").write_text(json.dumps(config_json))
            config = NemotronHBase.create_config(temp_dir)
        return config

    def test_basic_config(self):
        config = self._make_config()
        self.assertEqual(config.num_layers, 6)
        self.assertEqual(config.hidden_size, 2688)
        self.assertEqual(config.attn_config.head_num, 32)
        self.assertEqual(config.attn_config.kv_head_num, 2)
        self.assertEqual(config.attn_config.size_per_head, 128)
        self.assertEqual(config.vocab_size, 131072)
        self.assertEqual(config.max_seq_len, 1048576)
        self.assertFalse(config.tie_word_embeddings)

    def test_hybrid_attention_types(self):
        config = self._make_config()
        self.assertTrue(config.hybrid_attention_config.enable_hybrid_attention)
        self.assertEqual(
            list(config.hybrid_attention_config.hybrid_attention_types),
            [
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
                HybridAttentionType.LINEAR,
                HybridAttentionType.NONE,
            ],
        )

    def test_linear_attention_geometry(self):
        config = self._make_config()
        linear = config.linear_attention_config
        self.assertEqual(linear.linear_conv_kernel_dim, 4)
        self.assertEqual(linear.linear_key_head_dim, 128)
        self.assertEqual(linear.linear_num_key_heads, 8)
        self.assertEqual(linear.linear_num_value_heads, 64)
        self.assertEqual(linear.linear_value_head_dim, 64)

    def test_moe_config(self):
        config = self._make_config()
        self.assertEqual(config.expert_num, 8)
        self.assertEqual(config.moe_k, 2)
        self.assertEqual(config.moe_inter_size, 512)
        self.assertEqual(config.inter_size, 1024)
        self.assertEqual(config.n_shared_experts, 1)
        self.assertEqual(config.moe_style, 2)
        self.assertTrue(config.has_moe_norm)
        self.assertEqual(config.moe_layer_index, [1, 3])
        self.assertEqual(config.moe_w1_layout, "up_only")
        self.assertAlmostEqual(config.routed_scaling_factor, 2.5)

    def test_kv_cache_spec_descs(self):
        config = self._make_config()
        descs = config.kv_cache_spec_descs
        self.assertEqual(len(descs), 6)
        # mamba layers -> single LINEAR spec
        for layer_idx in (0, 2, 4):
            self.assertEqual(len(descs[layer_idx]), 1)
            self.assertEqual(descs[layer_idx][0].cache_type, KVCacheSpecType.LINEAR)
            self.assertEqual(descs[layer_idx][0].tag, "linear")
        # attention layer -> single MHA spec
        self.assertEqual(len(descs[5]), 1)
        self.assertEqual(descs[5][0].cache_type, KVCacheSpecType.MHA)
        self.assertEqual(descs[5][0].tag, "full")
        # moe layers -> attention-free, no descriptors
        for layer_idx in (1, 3):
            self.assertEqual(len(descs[layer_idx]), 0)

    def test_weight_cls(self):
        self.assertIs(NemotronH.get_weight_cls(), NemotronHWeight)

    def test_cuda_graph_supported(self):
        self.assertTrue(NemotronH.support_cuda_graph(None))


if __name__ == "__main__":
    unittest.main()
