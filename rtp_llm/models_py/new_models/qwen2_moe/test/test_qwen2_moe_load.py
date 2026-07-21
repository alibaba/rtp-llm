import types
import unittest

import torch
import torch.nn.functional as F
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen2_moe.language import (
    Qwen2MoeForCausalLM,
    Qwen2SharedExpert,
)
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod
from rtp_llm.models_py.registry import get_model_class


def _parallelism(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=1,
        dp_rank=0,
        world_size=max(tp_size, ep_size),
        local_rank=tp_rank,
        get_attn_tp_size=lambda: tp_size,
        get_attn_tp_rank=lambda: tp_rank,
        get_ffn_tp_size=lambda: tp_size,
        get_ffn_tp_rank=lambda: tp_rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
    )


def _moe_config():
    return types.SimpleNamespace(
        ll_num_max_token=1,
        masked_max_token_num=1,
        moe_strategy=0,
        use_mori_ep=False,
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_all_gather=True,
        fake_balance_expert=False,
    )


def _config(tie_word_embeddings=True):
    config = ModelConfig()
    config.model_type = "qwen_2_moe"
    config.num_layers = 1
    config.vocab_size = 8
    config.hidden_size = 4
    config.inter_size = 4
    config.expert_num = 2
    config.moe_inter_size = 4
    config.moe_k = 1
    config.moe_style = 2
    config.moe_topk_group = 1
    config.attn_config.head_num = 2
    config.attn_config.kv_head_num = 1
    config.attn_config.size_per_head = 2
    config.layernorm_eps = 1e-6
    config.enable_fp32_lm_head = False
    config.tie_word_embeddings = tie_word_embeddings
    config.data_type = "fp32"
    config.quant_config = None
    config.activation_type = "SiGLU"
    config.moe_layer_index = [0]
    return config


def _load_config(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, quant_config=None):
    parallelism = _parallelism(tp_size, tp_rank, ep_size, ep_rank)
    return NewLoaderConfig(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        compute_dtype=torch.float32,
        device="cpu",
        quant_config=quant_config or QuantizationConfig("none"),
        parallelism_config=parallelism,
        moe_config=_moe_config(),
    )


def _weights():
    weights = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.layers.0.input_layernorm.weight": torch.ones(4),
        "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
        "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
        "model.layers.0.self_attn.q_proj.bias": torch.zeros(4),
        "model.layers.0.self_attn.k_proj.bias": torch.zeros(2),
        "model.layers.0.self_attn.v_proj.bias": torch.zeros(2),
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
        "model.layers.0.mlp.gate.weight": torch.ones(2, 4),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.ones(4, 4),
        "model.layers.0.mlp.shared_expert.up_proj.weight": torch.full((4, 4), 2.0),
        "model.layers.0.mlp.shared_expert.down_proj.weight": torch.eye(4),
        "model.layers.0.mlp.shared_expert_gate.weight": torch.ones(1, 4),
        "model.norm.weight": torch.ones(4),
    }
    for expert_id in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert_id}"
        weights[f"{prefix}.gate_proj.weight"] = torch.ones(4, 4)
        weights[f"{prefix}.up_proj.weight"] = torch.full((4, 4), 2.0)
        weights[f"{prefix}.down_proj.weight"] = torch.eye(4)
    return weights


class _CpuSelectTopk(torch.nn.Module):
    def forward(self, router_logits, topk_ids, topk_weights):
        probabilities = torch.softmax(router_logits, dim=-1)
        values, indices = torch.topk(probabilities, topk_ids.shape[-1], dim=-1)
        topk_ids.copy_(indices)
        topk_weights.copy_(values)


class _CpuFusedMoe(torch.nn.Module):
    topk_ids_dtype = torch.int64

    def __init__(self, w13, w2):
        super().__init__()
        self.register_buffer("w13", w13.detach().clone())
        self.register_buffer("w2", w2.detach().clone())

    def forward(self, hidden_states, topk_weights, topk_ids, activation):
        if activation != "SiGLU":
            raise AssertionError(f"unexpected activation {activation!r}")
        intermediate_size = self.w13.shape[1] // 2
        output = torch.zeros_like(hidden_states)
        for token_idx in range(hidden_states.shape[0]):
            token = hidden_states[token_idx : token_idx + 1]
            for slot_idx in range(topk_ids.shape[1]):
                expert_idx = int(topk_ids[token_idx, slot_idx])
                up = self.w13[expert_idx, :intermediate_size]
                gate = self.w13[expert_idx, intermediate_size:]
                down = self.w2[expert_idx]
                expert_output = F.linear(
                    F.silu(F.linear(token, gate)) * F.linear(token, up), down
                )
                output[token_idx] += topk_weights[
                    token_idx, slot_idx
                ] * expert_output.squeeze(0)
        return output


class _IdentityAttention(torch.nn.Module):
    def forward(self, hidden_states, fmha_impl, kv_cache=None):
        return hidden_states


def _rms_norm_reference(hidden_states, weight, eps):
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(hidden_states.dtype)


def _expert_reference(hidden_states, gate, up, down):
    return F.linear(
        F.silu(F.linear(hidden_states, gate)) * F.linear(hidden_states, up), down
    )


class Qwen2MoeLoadTest(unittest.TestCase):
    def test_registry_exposes_qwen2_dense_and_moe(self):
        self.assertEqual(get_model_class("qwen_2").__name__, "Qwen2ForCausalLM")
        self.assertIs(get_model_class("qwen_2_moe"), Qwen2MoeForCausalLM)

    def test_real_tree_loads_shared_and_routed_experts(self):
        model = Qwen2MoeForCausalLM(_config(), _load_config())
        model.load_weights(_weights())
        NewModelLoader._validate_loaded_weights(model)

        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertEqual(model.layers[0].mlp.experts._loaded_count, 6)
        shared_gate_up = model.layers[0].mlp.shared_expert.gate_up_proj.weight
        torch.testing.assert_close(shared_gate_up[:4], torch.ones(4, 4))
        torch.testing.assert_close(shared_gate_up[4:], torch.full((4, 4), 2.0))

    def test_untied_missing_lm_head_fails_integrity(self):
        model = Qwen2MoeForCausalLM(_config(tie_word_embeddings=False), _load_config())
        model.load_weights(_weights())

        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_partial_moe_layer_config_is_rejected(self):
        config = _config()
        config.moe_layer_index = []

        with self.assertRaisesRegex(ValueError, "every decoder layer"):
            Qwen2MoeForCausalLM(config, _load_config())

    def test_non_hybrid_moe_style_is_rejected(self):
        config = _config()
        config.moe_style = 1

        with self.assertRaisesRegex(ValueError, "moe_style=2"):
            Qwen2MoeForCausalLM(config, _load_config())

    def test_raw_hf_dict_is_rejected_before_router_construction(self):
        with self.assertRaisesRegex(TypeError, "requires a typed ModelConfig"):
            Qwen2MoeForCausalLM(
                {
                    "model_type": "qwen_2_moe",
                    "num_hidden_layers": 1,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                },
                _load_config(),
            )

    def test_invalid_shared_expert_size_fails_fast(self):
        config = _config()
        config.inter_size = 0

        with self.assertRaisesRegex(ValueError, "shared_expert_intermediate_size"):
            Qwen2MoeForCausalLM(config, _load_config())

    def test_tp_topology_is_applied_to_attention_experts_and_shared_ffn(self):
        model = Qwen2MoeForCausalLM(_config(), _load_config(tp_size=2, tp_rank=1))
        layer = model.layers[0]

        self.assertEqual(
            (layer.self_attn.qkv_proj.tp_size, layer.self_attn.qkv_proj.tp_rank), (2, 1)
        )
        self.assertEqual(
            (
                layer.mlp.experts.moe_expert_tp_size,
                layer.mlp.experts.moe_expert_tp_rank,
            ),
            (2, 1),
        )
        self.assertEqual(
            (
                layer.mlp.shared_expert.gate_up_proj.tp_size,
                layer.mlp.shared_expert.gate_up_proj.tp_rank,
            ),
            (2, 1),
        )

    def test_router_and_shared_gate_stay_unquantized(self):
        config = _config()
        config.quant_config = types.SimpleNamespace(
            get_runtime_method_key=lambda: "fp8",
            get_method=lambda: "fp8",
        )
        model = Qwen2MoeForCausalLM(
            config, _load_config(quant_config=QuantizationConfig("fp8"))
        )
        block = model.layers[0].mlp

        self.assertIsInstance(block.gate.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(
            block.shared_expert_gate.quant_method, UnquantizedLinearMethod
        )

    def test_shared_expert_matches_dense_reference(self):
        layer = Qwen2SharedExpert(
            hidden_size=4,
            intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            quant_config=QuantizationConfig("none"),
            params_dtype=torch.float32,
            prefix="layers.0.mlp.shared_expert",
        )
        gate = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10
        up = torch.arange(16, 32, dtype=torch.float32).reshape(4, 4) / 10
        down = torch.eye(4)
        layer.load_weights(
            {
                "gate_proj.weight": gate,
                "up_proj.weight": up,
                "down_proj.weight": down,
            }
        )
        hidden = torch.tensor([[1.0, -1.0, 0.5, 2.0]])

        expected = F.linear(F.silu(F.linear(hidden, gate)) * F.linear(hidden, up), down)
        torch.testing.assert_close(layer(hidden), expected)

    def test_moe_decoder_forward_matches_cpu_reference(self):
        weights = _weights()
        weights["model.layers.0.mlp.gate.weight"] = torch.tensor(
            [[1.0, -0.5, 0.25, 0.0], [-0.75, 0.5, 0.0, 0.25]]
        )
        weights["model.layers.0.mlp.shared_expert_gate.weight"] = torch.tensor(
            [[0.5, -0.25, 0.75, 0.1]]
        )
        for expert_idx, scale in enumerate((0.25, 0.75)):
            prefix = f"model.layers.0.mlp.experts.{expert_idx}"
            weights[f"{prefix}.gate_proj.weight"] = torch.eye(4) * (scale + 0.5)
            weights[f"{prefix}.up_proj.weight"] = torch.eye(4) * (scale + 1.0)
            weights[f"{prefix}.down_proj.weight"] = torch.eye(4) * (scale + 1.5)

        model = Qwen2MoeForCausalLM(_config(), _load_config())
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        layer = model.layers[0]
        layer.self_attn = _IdentityAttention()
        layer.mlp.select_topk = _CpuSelectTopk()
        layer.mlp.experts.fused_moe = _CpuFusedMoe(
            layer.mlp.experts.w13, layer.mlp.experts.w2
        )

        hidden_states = torch.tensor(
            [[0.5, -1.0, 0.75, 1.5], [-0.25, 1.25, -0.5, 0.75]]
        )
        residual = torch.tensor([[0.1, 0.2, -0.3, 0.4], [0.5, -0.25, 0.75, -1.0]])
        actual_hidden, actual_residual = layer(hidden_states, residual, fmha_impl=None)

        first_residual = hidden_states + residual
        attention_input = _rms_norm_reference(
            first_residual,
            weights["model.layers.0.input_layernorm.weight"],
            layer.input_layernorm.eps,
        )
        expected_residual = attention_input + first_residual
        mlp_input = _rms_norm_reference(
            expected_residual,
            weights["model.layers.0.post_attention_layernorm.weight"],
            layer.post_attention_layernorm.eps,
        )

        router_logits = F.linear(mlp_input, weights["model.layers.0.mlp.gate.weight"])
        shifted_logits = router_logits - router_logits.max(dim=-1, keepdim=True).values
        router_probabilities = shifted_logits.exp()
        router_probabilities /= router_probabilities.sum(dim=-1, keepdim=True)
        selected_probabilities, selected_experts = router_probabilities.max(dim=-1)
        self.assertEqual(selected_experts.tolist(), [0, 1])
        routed = torch.empty_like(mlp_input)
        for token_idx, expert_idx in enumerate(selected_experts.tolist()):
            prefix = f"model.layers.0.mlp.experts.{expert_idx}"
            routed[token_idx] = selected_probabilities[token_idx] * _expert_reference(
                mlp_input[token_idx : token_idx + 1],
                weights[f"{prefix}.gate_proj.weight"],
                weights[f"{prefix}.up_proj.weight"],
                weights[f"{prefix}.down_proj.weight"],
            ).squeeze(0)

        shared = _expert_reference(
            mlp_input,
            weights["model.layers.0.mlp.shared_expert.gate_proj.weight"],
            weights["model.layers.0.mlp.shared_expert.up_proj.weight"],
            weights["model.layers.0.mlp.shared_expert.down_proj.weight"],
        )
        shared_gate = torch.sigmoid(
            F.linear(
                mlp_input,
                weights["model.layers.0.mlp.shared_expert_gate.weight"],
            )
        )
        expected_hidden = routed + shared * shared_gate

        torch.testing.assert_close(actual_residual, expected_residual)
        torch.testing.assert_close(actual_hidden, expected_hidden)


if __name__ == "__main__":
    unittest.main()
