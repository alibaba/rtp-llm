import json
import tempfile
import unittest
from pathlib import Path

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.kimi_linear.language import (
    KimiLinearForCausalLM,
    KimiLinearKDA,
)
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.ops import DataType, HybridAttentionType, RopeStyle


def _raw_config():
    return {
        "model_type": "kimi_linear",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 16,
        "intermediate_size": 12,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": False,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": None,
        "kv_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 2,
        "mla_use_nope": True,
        "rope_scaling": None,
        "linear_attn_config": {
            "kda_layers": [1],
            "full_attn_layers": [2],
            "head_dim": 4,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
        },
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "moe_intermediate_size": 4,
        "num_shared_experts": 1,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 2.0,
        "num_expert_group": 1,
        "topk_group": 1,
    }


def _model_config(checkpoint_path: str):
    config = ModelConfig()
    config.ckpt_path = checkpoint_path
    config.num_layers = 2
    config.max_seq_len = 128
    config.hidden_size = 8
    config.vocab_size = 16
    config.inter_size = 4
    config.layernorm_eps = 1e-5
    config.tie_word_embeddings = False
    config.enable_fp32_lm_head = False
    config.attn_config.use_mla = True
    config.mla_ops_type = "FLASH_INFER"
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 2
    config.attn_config.q_lora_rank = 0
    config.attn_config.kv_lora_rank = 4
    config.attn_config.nope_head_dim = 2
    config.attn_config.rope_head_dim = 2
    config.attn_config.v_head_dim = 2
    config.attn_config.size_per_head = 4
    config.attn_config.rope_config.style = RopeStyle.No
    config.linear_attention_config.linear_key_head_dim = 4
    config.linear_attention_config.linear_value_head_dim = 4
    config.linear_attention_config.linear_num_key_heads = 2
    config.linear_attention_config.linear_num_value_heads = 2
    config.linear_attention_config.linear_conv_kernel_dim = 4
    config.linear_attention_config.ssm_state_dtype = DataType.TYPE_FP32
    config.linear_attention_config.conv_state_dtype = DataType.TYPE_FP32
    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.hybrid_attention_types = [
        HybridAttentionType.LINEAR,
        HybridAttentionType.NONE,
    ]
    config.expert_num = 4
    config.moe_k = 2
    config.moe_inter_size = 4
    config.moe_n_group = 1
    config.moe_topk_group = 1
    config.scoring_func = 1
    config.has_moe_norm = True
    config.routed_scaling_factor = 2.0
    config.moe_style = 2
    config.moe_layer_index = [1]
    return config


def _load_config(**overrides):
    values = dict(
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        compute_dtype=torch.float32,
        device="cpu",
    )
    values.update(overrides)
    return NewLoaderConfig(**values)


def _tensor(shape, value):
    return torch.full(shape, float(value), dtype=torch.float32)


def _checkpoint_weights():
    weights = {
        "model.embed_tokens.weight": _tensor((16, 8), 1),
        "model.norm.weight": _tensor((8,), 2),
        "lm_head.weight": _tensor((16, 8), 3),
    }
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        weights[prefix + "input_layernorm.weight"] = _tensor((8,), 4 + layer)
        weights[prefix + "post_attention_layernorm.weight"] = _tensor((8,), 6 + layer)
    kda = "model.layers.0.self_attn."
    for index, projection in enumerate(("q_proj", "k_proj", "v_proj"), 10):
        weights[kda + projection + ".weight"] = _tensor((8, 8), index)
    weights[kda + "b_proj.weight"] = _tensor((2, 8), 13)
    weights[kda + "f_a_proj.weight"] = _tensor((4, 8), 14)
    weights[kda + "f_b_proj.weight"] = _tensor((8, 4), 15)
    weights[kda + "g_a_proj.weight"] = _tensor((4, 8), 16)
    weights[kda + "g_b_proj.weight"] = _tensor((8, 4), 17)
    for index, projection in enumerate(("q_conv1d", "k_conv1d", "v_conv1d"), 18):
        weights[kda + projection + ".weight"] = _tensor((8, 1, 4), index)
    weights[kda + "dt_bias"] = _tensor((8,), 21)
    weights[kda + "A_log"] = _tensor((1, 1, 2, 1), 22)
    weights[kda + "o_norm.weight"] = _tensor((4,), 23)
    weights[kda + "o_proj.weight"] = _tensor((8, 8), 24)
    dense = "model.layers.0.mlp."
    weights[dense + "gate_proj.weight"] = _tensor((12, 8), 25)
    weights[dense + "up_proj.weight"] = _tensor((12, 8), 26)
    weights[dense + "down_proj.weight"] = _tensor((8, 12), 27)

    mla = "model.layers.1.self_attn."
    weights[mla + "q_proj.weight"] = _tensor((8, 8), 30)
    weights[mla + "kv_a_proj_with_mqa.weight"] = _tensor((6, 8), 31)
    weights[mla + "kv_a_layernorm.weight"] = _tensor((4,), 32)
    weights[mla + "kv_b_proj.weight"] = _tensor((8, 4), 33)
    weights[mla + "o_proj.weight"] = _tensor((8, 4), 34)
    moe = "model.layers.1.block_sparse_moe."
    weights[moe + "gate.weight"] = _tensor((4, 8), 35)
    weights[moe + "gate.e_score_correction_bias"] = _tensor((4,), 36)
    weights[moe + "shared_experts.gate_proj.weight"] = _tensor((4, 8), 37)
    weights[moe + "shared_experts.up_proj.weight"] = _tensor((4, 8), 38)
    weights[moe + "shared_experts.down_proj.weight"] = _tensor((8, 4), 39)
    for expert in range(4):
        expert_prefix = moe + f"experts.{expert}."
        weights[expert_prefix + "w1.weight"] = _tensor((4, 8), 40 + expert)
        weights[expert_prefix + "w3.weight"] = _tensor((4, 8), 50 + expert)
        weights[expert_prefix + "w2.weight"] = _tensor((8, 4), 60 + expert)
    return weights


class KimiLinearNewLoaderTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        Path(self.tempdir.name, "config.json").write_text(
            json.dumps(_raw_config()), encoding="utf-8"
        )

    def test_registry_resolves_clean_model(self):
        self.assertIs(get_model_class("kimi_linear"), KimiLinearForCausalLM)

    def test_expert_mapper_preserves_quantization_suffixes(self):
        cases = {
            "model.layers.3.block_sparse_moe.experts.7.w1.weight": (
                "layers.3.block_sparse_moe.experts.7.gate_proj.weight"
            ),
            "model.layers.3.block_sparse_moe.experts.7.w3.weight_scale_inv": (
                "layers.3.block_sparse_moe.experts.7.up_proj.weight_scale_inv"
            ),
            "model.layers.3.block_sparse_moe.experts.7.w2.weight_scale": (
                "layers.3.block_sparse_moe.experts.7.down_proj.weight_scale"
            ),
            "model.norm.weight": "norm.weight",
        }
        for checkpoint_name, module_name in cases.items():
            with self.subTest(checkpoint_name=checkpoint_name):
                self.assertEqual(
                    KimiLinearForCausalLM.WEIGHTS_MAPPER.map_name(checkpoint_name),
                    module_name,
                )

    def test_complete_checkpoint_dispatches_without_legacy_layout(self):
        model = KimiLinearForCausalLM(_model_config(self.tempdir.name), _load_config())
        model.load_weights(_checkpoint_weights())
        NewModelLoader._validate_loaded_weights(model)
        model.layers[1].self_attn.process_weights_after_loading()
        model._ensure_mla_kernel_layout()
        self.assertEqual(model.layers[0].self_attn.qkv_proj.weight.shape, (24, 8))
        self.assertEqual(model.layers[0].self_attn.conv1d.weight.shape, (24, 4))
        self.assertEqual(len(model._mla_kernel_layout.weights), 2)
        self.assertEqual(model._mla_kernel_layout.weights[0], {})

    def test_unknown_and_missing_tensors_fail_fast(self):
        with self.subTest("unknown"):
            model = KimiLinearForCausalLM(
                _model_config(self.tempdir.name), _load_config()
            )
            weights = _checkpoint_weights()
            weights["model.layers.0.self_attn.typo.weight"] = _tensor((1,), 1)
            with self.assertRaisesRegex(RuntimeError, r"typo\.weight"):
                model.load_weights(weights)
        with self.subTest("missing"):
            model = KimiLinearForCausalLM(
                _model_config(self.tempdir.name), _load_config()
            )
            weights = _checkpoint_weights()
            del weights["model.layers.0.self_attn.dt_bias"]
            model.load_weights(weights)
            with self.assertRaisesRegex(RuntimeError, "dt_bias"):
                NewModelLoader._validate_loaded_weights(model)

    def test_invalid_hybrid_topology_fails_before_module_construction(self):
        cases = (
            ("duplicate", [1, 1], r"must not contain duplicate"),
            ("out_of_range", [1, 3], r"integers in \[1, 2\]"),
        )
        for name, kda_layers, pattern in cases:
            with self.subTest(name=name):
                raw = _raw_config()
                raw["linear_attn_config"]["kda_layers"] = kda_layers
                Path(self.tempdir.name, "config.json").write_text(
                    json.dumps(raw), encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    KimiLinearForCausalLM(
                        _model_config(self.tempdir.name), _load_config()
                    )

    def test_kda_tp2_splits_every_head_owned_tensor(self):
        cfg = {
            "hidden_size": 8,
            "kda_head_dim": 4,
            "kda_num_heads": 4,
            "attn_tp_size": 2,
            "attn_tp_rank": 1,
            "params_dtype": torch.float32,
            "quant_config": None,
            "ssm_state_dtype": torch.float32,
            "conv_state_dtype": torch.float32,
            "conv_kernel": 4,
            "rms_norm_eps": 1e-5,
        }
        module = KimiLinearKDA(cfg, prefix="layers.0.self_attn")
        weights = {}
        for offset, projection in enumerate(("q_proj", "k_proj", "v_proj")):
            base = torch.arange(16 * 8, dtype=torch.float32).reshape(16, 8)
            weights[projection + ".weight"] = base + 1000 * offset
        for offset, projection in enumerate(("q_conv1d", "k_conv1d", "v_conv1d")):
            base = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 1, 4)
            weights[projection + ".weight"] = base + 1000 * offset
        weights["b_proj.weight"] = torch.arange(4 * 8, dtype=torch.float32).reshape(
            4, 8
        )
        for offset, projection in enumerate(("f_b_proj", "g_b_proj"), 1):
            base = torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4)
            weights[projection + ".weight"] = base + 1000 * offset
        weights["dt_bias"] = torch.arange(16, dtype=torch.float32)
        weights["A_log"] = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
        module.load_weights(weights)
        for shard in range(3):
            expected = weights[("q_proj", "k_proj", "v_proj")[shard] + ".weight"][8:]
            torch.testing.assert_close(
                module.qkv_proj.weight[shard * 8 : (shard + 1) * 8], expected
            )
            conv_name = ("q_conv1d", "k_conv1d", "v_conv1d")[shard]
            expected_conv = weights[conv_name + ".weight"][8:].squeeze(1)
            torch.testing.assert_close(
                module.conv1d.weight[shard * 8 : (shard + 1) * 8], expected_conv
            )
        torch.testing.assert_close(module.b_proj.weight, weights["b_proj.weight"][2:])
        torch.testing.assert_close(
            module.f_b_proj.weight, weights["f_b_proj.weight"][8:]
        )
        torch.testing.assert_close(
            module.g_b_proj.weight, weights["g_b_proj.weight"][8:]
        )
        torch.testing.assert_close(
            module.dt_bias, torch.arange(16, dtype=torch.float32)[8:]
        )
        torch.testing.assert_close(module.A_log, torch.tensor([2.0, 3.0]))

    def test_checkpoint_filter_only_skips_valid_truncated_tail_layers(self):
        model_config = _model_config(self.tempdir.name)
        model_config.num_layers = 1
        model_config.moe_layer_index = []
        model = KimiLinearForCausalLM(model_config, _load_config())
        should_load = model.checkpoint_weight_name_filter()
        self.assertTrue(should_load("model.layers.0.self_attn.q_proj.weight"))
        self.assertFalse(should_load("model.layers.1.self_attn.q_proj.weight"))
        self.assertTrue(should_load("model.layers.2.self_attn.q_proj.weight"))

    def test_single_head_a_log_preserves_vector_shape(self):
        cfg = {
            "hidden_size": 8,
            "kda_head_dim": 4,
            "kda_num_heads": 1,
            "attn_tp_size": 1,
            "attn_tp_rank": 0,
            "params_dtype": torch.float32,
            "quant_config": None,
            "ssm_state_dtype": torch.float32,
            "conv_state_dtype": torch.float32,
            "conv_kernel": 4,
            "rms_norm_eps": 1e-5,
        }
        module = KimiLinearKDA(cfg, prefix="layers.0.self_attn")
        module.load_weights({"A_log": torch.tensor([[[[7.0]]]])})
        self.assertEqual(module.A_log.shape, (1,))
        torch.testing.assert_close(module.A_log, torch.tensor([7.0]))

    def test_tied_lm_head_can_be_omitted_from_checkpoint(self):
        raw = _raw_config()
        raw["tie_word_embeddings"] = True
        Path(self.tempdir.name, "config.json").write_text(
            json.dumps(raw), encoding="utf-8"
        )
        model_config = _model_config(self.tempdir.name)
        model_config.tie_word_embeddings = True
        model = KimiLinearForCausalLM(model_config, _load_config())
        weights = _checkpoint_weights()
        del weights["lm_head.weight"]
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)

    def test_cpu_and_rocm_runtime_are_rejected_before_forward(self):
        cfg = {
            "hidden_size": 8,
            "kda_head_dim": 4,
            "kda_num_heads": 2,
            "attn_tp_size": 1,
            "attn_tp_rank": 0,
            "params_dtype": torch.float32,
            "quant_config": None,
            "ssm_state_dtype": torch.float32,
            "conv_state_dtype": torch.float32,
            "conv_kernel": 4,
            "rms_norm_eps": 1e-5,
        }
        module = KimiLinearKDA(cfg, prefix="layers.0.self_attn")
        with self.assertRaisesRegex(RuntimeError, "only on CUDA"):
            module.validate_runtime_device(torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
