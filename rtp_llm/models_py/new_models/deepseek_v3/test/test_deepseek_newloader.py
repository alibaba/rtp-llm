import math
import types
import unittest

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.deepseek_v3.attention import (
    DeepSeekV32MlaAttention,
    _linear_weight_bf16,
)
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    _build_rope_cache,
    _extract_config_values,
)
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3_mtp.language import (
    DeepSeekV32MTPForCausalLM,
    _draft_checkpoint_layer,
    _remap_key,
)
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class


def _model_config(*, sparse=False, q_lora_rank=4):
    return types.SimpleNamespace(
        model_type="deepseek3",
        hidden_size=8,
        num_layers=4,
        vocab_size=16,
        max_seq_len=128,
        rms_norm_eps=1e-6,
        expert_num=8,
        moe_k=2,
        scoring_func=1,
        routed_scaling_factor=1.0,
        moe_n_group=2,
        moe_topk_group=1,
        has_moe_norm=True,
        enable_fp32_lm_head=False,
        tie_word_embeddings=False,
        attn_config=types.SimpleNamespace(
            head_num=4,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            is_sparse=sparse,
            indexer_head_dim=4,
            indexer_head_num=2,
            indexer_topk=8,
            kernel_tokens_per_block=64,
        ),
    )


def _raw_config():
    return {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 4,
        "num_nextn_predict_layers": 1,
        "intermediate_size": 16,
        "moe_intermediate_size": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "n_group": 2,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "greedy",
        "rope_interleave": True,
        "indexer_rope_interleave": False,
        "qk_rope_head_dim": 2,
    }


def _load_config():
    return NewLoaderConfig(
        tp_size=2,
        tp_rank=1,
        ep_size=2,
        ep_rank=1,
        attn_tp_size=1,
        attn_tp_rank=0,
        ffn_tp_size=2,
        ffn_tp_rank=1,
        lm_head_tp_size=2,
        lm_head_tp_rank=1,
        compute_dtype=torch.float32,
        device="cpu",
    )


def _uninitialized_model(model_type, *, layer_count=0, checkpoint_prefix=None):
    model = object.__new__(model_type)
    torch.nn.Module.__init__(model)
    if layer_count:
        model.layers = torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(layer_count)]
        )
    if checkpoint_prefix is not None:
        model._checkpoint_prefix = checkpoint_prefix
    return model


class DeepSeekNewloaderTest(unittest.TestCase):
    def test_local_rope_cache_preserves_reference_numerics(self):
        max_seq_len = 32
        rope_dim = 8
        base = 10000.0
        plain = _build_rope_cache(
            {"qk_rope_head_dim": rope_dim, "rope_theta": base},
            max_seq_len,
            torch.device("cpu"),
        )
        inv_freq = 1.0 / (base ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
        freqs = torch.outer(torch.arange(max_seq_len, dtype=inv_freq.dtype), inv_freq)
        self.assertTrue(
            torch.equal(
                plain,
                torch.cat([freqs.cos(), freqs.sin()], dim=-1),
            )
        )

        factor = 8.0
        original_max = 16
        beta_fast = 32.0
        beta_slow = 1.0
        mscale = 1.0
        mscale_all_dim = 1.0
        yarn = _build_rope_cache(
            {
                "qk_rope_head_dim": rope_dim,
                "rope_theta": base,
                "rope_scaling": {
                    "factor": factor,
                    "original_max_position_embeddings": original_max,
                    "beta_fast": beta_fast,
                    "beta_slow": beta_slow,
                    "mscale": mscale,
                    "mscale_all_dim": mscale_all_dim,
                },
            },
            max_seq_len,
            torch.device("cpu"),
        )
        freq_extra = 1.0 / (
            base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        freq_inter = freq_extra / factor

        def correction_dim(num_rotations):
            return (
                rope_dim * math.log(original_max / (num_rotations * 2 * math.pi))
            ) / (2 * math.log(base))

        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), rope_dim - 1)
        if low == high:
            high += 0.001
        ramp = torch.clamp(
            (torch.arange(rope_dim // 2, dtype=torch.float32) - low) / (high - low),
            0,
            1,
        )
        mask = 1.0 - ramp
        yarn_inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        yarn_freqs = torch.outer(
            torch.arange(max_seq_len, dtype=torch.float32),
            yarn_inv_freq,
        )

        def yarn_mscale(value):
            if factor <= 1:
                return 1.0
            return 0.1 * value * math.log(factor) + 1.0

        scale = yarn_mscale(mscale) / yarn_mscale(mscale_all_dim)
        self.assertTrue(
            torch.equal(
                yarn,
                torch.cat(
                    [
                        yarn_freqs.cos() * scale,
                        yarn_freqs.sin() * scale,
                    ],
                    dim=-1,
                ),
            )
        )

    def test_registry_aliases_are_typed(self):
        for model_type in (
            "deepseek2",
            "deepseek3",
            "deepseek_v31",
            "deepseek_v32",
            "glm_5",
        ):
            with self.subTest(model_type=model_type):
                self.assertIs(get_model_class(model_type), DeepSeekV32ForCausalLM)
        self.assertIs(
            get_model_class("deepseek-v3-mtp"),
            DeepSeekV32MTPForCausalLM,
        )

    def test_core_filter_excludes_appended_draft_and_unrelated_tensors(self):
        model = _uninitialized_model(DeepSeekV32ForCausalLM, layer_count=4)
        should_load = model.checkpoint_weight_name_filter()
        accepted = (
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.3.mlp.experts.7.down_proj.weight",
            "model.layers.bad.weight",
            "model.norm.weight",
            "model.unknown.weight",
            "model.visual.weight",
            "lm_head.weight",
        )
        rejected = (
            "model.layers.4.self_attn.q_a_proj.weight",
            "model.layers.40.self_attn.q_a_proj.weight",
            "mtp.layers.0.weight",
        )
        for name in accepted:
            with self.subTest(name=name):
                self.assertTrue(should_load(name))
        for name in rejected:
            with self.subTest(name=name):
                self.assertFalse(should_load(name))

    def test_mtp_prefix_supports_standalone_and_full_checkpoints(self):
        standalone = {
            "architectures": ["DeepseekV3ForCausalLMNextN"],
            "num_hidden_layers": 1,
            "num_nextn_predict_layers": 1,
        }
        self.assertEqual(_draft_checkpoint_layer(standalone), 0)
        with self.assertRaisesRegex(ValueError, "at most one draft"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLMNextN"],
                    "num_hidden_layers": 1,
                    "num_nextn_predict_layers": 2,
                }
            )
        self.assertEqual(_draft_checkpoint_layer(_raw_config()), 4)
        with self.assertRaisesRegex(ValueError, "exactly one appended"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "num_hidden_layers": 4,
                    "num_nextn_predict_layers": 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "exactly one appended"):
            _draft_checkpoint_layer(
                {
                    "architectures": ["DeepseekV3ForCausalLM"],
                    "num_hidden_layers": 4,
                    "num_nextn_predict_layers": 2,
                }
            )

    def test_mtp_filter_is_exact_and_rank_invariant(self):
        model = _uninitialized_model(
            DeepSeekV32MTPForCausalLM,
            checkpoint_prefix="model.layers.4.",
        )
        should_load = model.checkpoint_weight_name_filter()
        self.assertTrue(should_load("model.layers.4.embed_tokens.weight"))
        self.assertTrue(should_load("model.layers.4.mlp.experts.7.gate_proj.weight"))
        self.assertFalse(should_load("model.layers.3.embed_tokens.weight"))
        self.assertFalse(should_load("model.layers.40.embed_tokens.weight"))
        self.assertFalse(should_load("model.layers.4."))
        self.assertFalse(should_load("lm_head.weight"))

    def test_mtp_key_remap_is_explicit(self):
        expected = {
            "embed_tokens.weight": "embed_tokens.weight",
            "shared_head.head.weight": "lm_head.weight",
            "shared_head.norm.weight": "norm.weight",
            "enorm.weight": "mtp_block.e_norm.weight",
            "hnorm.weight": "mtp_block.h_norm.weight",
            "eh_proj.weight": "mtp_block.fc.weight",
            "self_attn.q_a_proj.weight": "layers.0.self_attn.q_a_proj.weight",
            "mlp.gate.weight": "layers.0.mlp.gate.weight",
        }
        for name, remapped in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_remap_key(name), remapped)

    def test_mtp_load_weights_maps_real_modules_and_rejects_extra_tensors(self):
        model = _uninitialized_model(
            DeepSeekV32MTPForCausalLM,
            checkpoint_prefix="model.layers.4.",
        )
        model.embed_tokens = torch.nn.Embedding(3, 2)
        model.lm_head = torch.nn.Linear(2, 3, bias=False)
        model.norm = torch.nn.LayerNorm(2, elementwise_affine=True, bias=False)
        model.mtp_block = MTPBlock(hidden_size=2, params_dtype=torch.float32)

        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_a_proj = torch.nn.Linear(2, 2, bias=False)
        model.layers = torch.nn.ModuleList([layer])

        values = {
            "model.layers.4.embed_tokens.weight": torch.arange(
                6, dtype=torch.float32
            ).reshape(3, 2),
            "model.layers.4.shared_head.head.weight": torch.arange(
                6, 12, dtype=torch.float32
            ).reshape(3, 2),
            "model.layers.4.shared_head.norm.weight": torch.tensor([12.0, 13.0]),
            "model.layers.4.enorm.weight": torch.tensor([14.0, 15.0]),
            "model.layers.4.hnorm.weight": torch.tensor([16.0, 17.0]),
            "model.layers.4.eh_proj.weight": torch.arange(
                18, 26, dtype=torch.float32
            ).reshape(2, 4),
            "model.layers.4.self_attn.q_a_proj.weight": torch.arange(
                26, 30, dtype=torch.float32
            ).reshape(2, 2),
        }
        model.load_weights(values)

        self.assertTrue(
            torch.equal(
                model.embed_tokens.weight,
                values["model.layers.4.embed_tokens.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.lm_head.weight,
                values["model.layers.4.shared_head.head.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.norm.weight,
                values["model.layers.4.shared_head.norm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.e_norm.weight,
                values["model.layers.4.enorm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.h_norm.weight,
                values["model.layers.4.hnorm.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.mtp_block.fc.weight,
                values["model.layers.4.eh_proj.weight"],
            )
        )
        self.assertTrue(
            torch.equal(
                model.layers[0].self_attn.q_a_proj.weight,
                values["model.layers.4.self_attn.q_a_proj.weight"],
            )
        )

        with self.assertRaisesRegex(RuntimeError, "non-draft"):
            model.load_weights(
                {
                    "model.layers.3.embed_tokens.weight": torch.zeros(
                        3, 2, dtype=torch.float32
                    )
                }
            )
        with self.assertRaisesRegex(RuntimeError, "could not dispatch"):
            model.load_weights(
                {"model.layers.4.unknown.weight": torch.zeros(1, dtype=torch.float32)}
            )

    def test_config_keeps_attention_and_ffn_topologies_distinct(self):
        cfg = _extract_config_values(_model_config(), _load_config(), _raw_config())
        self.assertEqual((cfg["attn_tp_size"], cfg["attn_tp_rank"]), (1, 0))
        self.assertEqual((cfg["ffn_tp_size"], cfg["ffn_tp_rank"]), (2, 1))
        self.assertEqual((cfg["lm_head_tp_size"], cfg["lm_head_tp_rank"]), (2, 1))
        self.assertEqual((cfg["ep_size"], cfg["ep_rank"]), (2, 1))
        self.assertEqual(cfg["moe_layer_index"], [3])

    def test_sparse_indexer_rejects_no_q_lora(self):
        with self.assertRaisesRegex(ValueError, "requires q_lora_rank"):
            _extract_config_values(
                _model_config(sparse=True, q_lora_rank=0),
                _load_config(),
                _raw_config(),
            )

    def test_grouped_router_rejects_invalid_expert_partition(self):
        raw = _raw_config()
        raw.update(
            {
                "topk_method": "noaux_tc",
                "topk_group": 1,
            }
        )
        model_config = _model_config()
        model_config.moe_n_group = 3
        with self.assertRaisesRegex(ValueError, "divisible by n_group"):
            _extract_config_values(model_config, _load_config(), raw)

    def test_no_q_lora_has_no_empty_checkpoint_parameters(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=0,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            params_dtype=torch.float32,
        )
        self.assertIsNone(attention.q_a_layernorm)
        self.assertIsNone(attention.q_b_proj)
        self.assertFalse(
            any(
                name.startswith("q_b_proj.") for name, _ in attention.named_parameters()
            )
        )
        attention.process_weights_after_loading()
        derived = {
            "_fused_qkv_a_w",
            "_fused_qkv_b_w",
            "_kv_b_w",
            "_kc_w",
            "_vc_w",
        }
        self.assertTrue(derived.isdisjoint(dict(attention.named_parameters())))
        self.assertTrue(derived.isdisjoint(attention.state_dict()))

    def test_mla_output_projection_defers_tp_reduction_to_attention(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
        )
        self.assertFalse(attention.o_proj.reduce_output)

    def test_mla_fp8_derived_weights_support_tensor_and_channel_scales(self):
        weight = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float8_e4m3fn)
        for scale in (
            torch.tensor([0.5], dtype=torch.float32),
            torch.tensor([[0.5], [2.0]], dtype=torch.float32),
        ):
            with self.subTest(scale_shape=tuple(scale.shape)):
                linear = torch.nn.Module()
                linear.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
                linear.weight_scale = torch.nn.Parameter(
                    scale.clone(), requires_grad=False
                )
                expected_scale = scale.to(torch.bfloat16).float()
                if expected_scale.numel() == 1:
                    expected_scale = expected_scale.reshape(1, 1)
                expected = (weight.to(torch.bfloat16).float() * expected_scale).to(
                    torch.bfloat16
                )
                self.assertTrue(torch.equal(_linear_weight_bf16(linear), expected))

    def test_mla_misaligned_fp8_a_projection_uses_bf16_fallback(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=129,
            kv_lora_rank=128,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.process_weights_after_loading()
        self.assertIsNone(attention._fused_qkv_a_runtime)
        self.assertEqual(
            tuple(attention._fused_qkv_a_w.shape),
            (129 + 128 + 2, 128),
        )

    def test_mla_non_block_fp8_scales_use_bf16_fallback(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.q_a_proj.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        attention.kv_a_proj_with_mqa.weight_scale_inv = torch.nn.Parameter(
            torch.ones(1, dtype=torch.float32), requires_grad=False
        )
        attention.process_weights_after_loading()
        self.assertIsNone(attention._fused_qkv_a_runtime)
        self.assertEqual(
            tuple(attention._fused_qkv_a_w.shape),
            (128 + 128 + 2, 128),
        )

    @unittest.skipIf(
        getattr(torch.version, "hip", None) is not None,
        "online fused FP8 A projection is CUDA-only",
    )
    def test_mla_online_fp8_uses_models_py_quantizer(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            quant_config=QuantizationConfig("fp8_block_online"),
            params_dtype=torch.bfloat16,
        )
        for parameter in attention.parameters():
            parameter.data.zero_()
        attention.process_weights_after_loading()
        self.assertIsNotNone(attention._fused_qkv_a_runtime)
        self.assertEqual(
            tuple(attention._fused_qkv_a_runtime.weight.shape),
            (128 + 128 + 2, 128),
        )

    def test_mlp_tp_reduction_is_owned_by_row_parallel_linear(self):
        dense = DeepSeekV32MLP(
            hidden_size=4,
            intermediate_size=4,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
            reduce_output=True,
        )
        shared = DeepSeekV32MLP(
            hidden_size=4,
            intermediate_size=4,
            tp_size=2,
            tp_rank=0,
            params_dtype=torch.float32,
            reduce_output=False,
        )
        self.assertTrue(dense.down_proj.reduce_output)
        self.assertFalse(shared.down_proj.reduce_output)

    def test_fp8_mlp_pads_weights_and_scales_before_tp_split(self):
        hidden_size = 128
        intermediate_size = 257
        block_size = 128
        mlp = DeepSeekV32MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_size=2,
            tp_rank=1,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        self.assertEqual(mlp.padded_intermediate_size, 512)

        gate = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .remainder(17)
            .sub(8)
            .to(torch.float8_e4m3fn)
        )
        up = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .remainder(13)
            .sub(6)
            .to(torch.float8_e4m3fn)
        )
        down = (
            torch.arange(hidden_size * intermediate_size, dtype=torch.float32)
            .reshape(hidden_size, intermediate_size)
            .remainder(11)
            .sub(5)
            .to(torch.float8_e4m3fn)
        )
        scale_blocks = (intermediate_size + block_size - 1) // block_size
        gate_scale = torch.arange(1, scale_blocks + 1, dtype=torch.float32).reshape(
            scale_blocks, 1
        )
        up_scale = gate_scale + 10
        down_scale = gate_scale.t() + 20

        mlp.load_weights(
            {
                "gate_proj.weight": gate,
                "gate_proj.weight_scale_inv": gate_scale,
                "up_proj.weight": up,
                "up_proj.weight_scale_inv": up_scale,
                "down_proj.weight": down,
                "down_proj.weight_scale_inv": down_scale,
            }
        )

        expected_gate = torch.zeros(256, hidden_size, dtype=torch.float8_e4m3fn)
        expected_gate[0] = gate[-1]
        expected_up = torch.zeros(256, hidden_size, dtype=torch.float8_e4m3fn)
        expected_up[0] = up[-1]
        self.assertTrue(torch.equal(mlp.gate_up_proj.weight[:256], expected_gate))
        self.assertTrue(torch.equal(mlp.gate_up_proj.weight[256:], expected_up))

        expected_down = torch.zeros(hidden_size, 256, dtype=torch.float8_e4m3fn)
        expected_down[:, 0] = down[:, -1]
        self.assertTrue(torch.equal(mlp.down_proj.weight, expected_down))

        expected_gate_scale = torch.tensor([[3.0], [0.0]])
        expected_up_scale = torch.tensor([[13.0], [0.0]])
        self.assertTrue(
            torch.equal(
                mlp.gate_up_proj.weight_scale_inv[:2],
                expected_gate_scale,
            )
        )
        self.assertTrue(
            torch.equal(
                mlp.gate_up_proj.weight_scale_inv[2:],
                expected_up_scale,
            )
        )
        self.assertTrue(
            torch.equal(
                mlp.down_proj.weight_scale_inv,
                torch.tensor([[23.0, 0.0]]),
            )
        )

    def test_runtime_weight_view_exports_only_runtime_globals(self):
        for model_type in (DeepSeekV32ForCausalLM, DeepSeekV32MTPForCausalLM):
            with self.subTest(model_type=model_type.__name__):
                model = _uninitialized_model(model_type)
                model.embed_tokens = torch.nn.Embedding(4, 2)
                model.norm = torch.nn.LayerNorm(2)
                model.lm_head = torch.nn.Linear(2, 4, bias=False)
                view = model.runtime_weight_view()
                self.assertEqual(
                    set(view),
                    {"embedding", "final_layernorm.gamma", "lm_head"},
                )
                self.assertIs(view["embedding"], model.embed_tokens.weight)
                self.assertIs(view["lm_head"], model.lm_head.weight)

    def test_mtp_block_rejects_inconsistent_inputs(self):
        block = MTPBlock(hidden_size=4, params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            block(torch.zeros(2, 4), torch.zeros(1, 4))
        with self.assertRaisesRegex(TypeError, "share a dtype"):
            block(
                torch.zeros(2, 4, dtype=torch.float32),
                torch.zeros(2, 4, dtype=torch.float64),
            )


if __name__ == "__main__":
    unittest.main()
