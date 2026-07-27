import json
import math
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.deepseek_v3.attention import (
    DeepSeekV32MlaAttention,
    _kernel_fp8_weight_and_scale,
    _linear_weight_bf16,
    _prepare_fused_fp8_runtime_weight,
)
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    DeepSeekV32ForCausalLM,
    _build_rope_cache,
    _extract_config_values,
)
from rtp_llm.models_py.new_models.deepseek_v3.mlp import DeepSeekV32MLP
from rtp_llm.models_py.new_models.deepseek_v3.moe import (
    DeepSeekV32MoEBlock,
    _select_deepseek_noaux_topk,
    _select_deepseek_topk,
)
from rtp_llm.models_py.new_models.deepseek_v3.rotary_embedding import (
    DeepseekV3RotaryEmbedding,
)
from rtp_llm.models_py.new_models.deepseek_v3_mtp.language import (
    DeepSeekV32MTPForCausalLM,
    _draft_checkpoint_layer,
    _remap_key,
)
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import get_model_class
from rtp_llm.utils.model_weight import W


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
            "kimi_k2",
        ):
            with self.subTest(model_type=model_type):
                self.assertIs(get_model_class(model_type), DeepSeekV32ForCausalLM)
        self.assertIs(
            get_model_class("deepseek-v3-mtp"),
            DeepSeekV32MTPForCausalLM,
        )

    def test_rotary_cache_is_valid_immediately_after_construction(self):
        rotary = DeepseekV3RotaryEmbedding(
            dim=8,
            max_position_embeddings=16,
            device=torch.device("cpu"),
        )
        cache_ptr = rotary.cos_cached.data_ptr()
        self.assertEqual(rotary.max_seq_len_cached, 16)
        rotary(torch.zeros(1, 8), seq_len=8)
        self.assertEqual(rotary.cos_cached.data_ptr(), cache_ptr)

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

    def test_mtp_real_constructor_uses_appended_layer_from_full_checkpoint(self):
        config_json = _raw_config()
        model_config = _model_config()
        # ModelFactory creates the draft config from the full checkpoint, so
        # this remains the main-model layer count at construction time.
        model_config.num_layers = config_json["num_hidden_layers"]
        with tempfile.TemporaryDirectory() as checkpoint_path:
            model_config.ckpt_path = checkpoint_path
            with open(
                f"{checkpoint_path}/config.json", "w", encoding="utf-8"
            ) as config_file:
                json.dump(config_json, config_file)
            with torch.device("cpu"):
                model = DeepSeekV32MTPForCausalLM(
                    model_config,
                    _load_config(),
                )

        self.assertEqual(model._checkpoint_layer, 4)
        self.assertEqual(model._checkpoint_prefix, "model.layers.4.")
        self.assertEqual(model.layer_num, 1)
        self.assertEqual(len(model.layers), 1)
        self.assertEqual(model.cos_sin_cache.device.type, "cpu")

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
        self.assertEqual(cfg["topk_method"], "greedy")

    def test_raw_router_config_is_canonical(self):
        raw = _raw_config()
        raw.update(
            {
                "scoring_func": "softmax",
                "routed_scaling_factor": 2.5,
                "n_group": 4,
                "topk_group": 2,
                "norm_topk_prob": False,
                "topk_method": "group_limited_greedy",
            }
        )
        cfg = _extract_config_values(_model_config(), _load_config(), raw)
        self.assertEqual(cfg["scoring_func"], 0)
        self.assertEqual(cfg["routed_scaling_factor"], 2.5)
        self.assertEqual(cfg["n_group"], 4)
        self.assertEqual(cfg["topk_group"], 2)
        self.assertFalse(cfg["has_moe_norm"])
        self.assertEqual(cfg["topk_method"], "group_limited_greedy")

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
                "n_group": 3,
                "topk_method": "noaux_tc",
                "topk_group": 1,
            }
        )
        with self.assertRaisesRegex(ValueError, "divisible by n_group"):
            _extract_config_values(_model_config(), _load_config(), raw)

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

    def test_mla_kernel_fp8_layout_preserves_ue8m0_contract(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        fp32_scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        normal_weight, normal_scale = _kernel_fp8_weight_and_scale(weight, fp32_scale)
        self.assertEqual(tuple(normal_weight.shape), (4, 6))
        self.assertEqual(tuple(normal_scale.shape), (2, 3))
        self.assertEqual(normal_weight.data_ptr(), weight.data_ptr())
        self.assertEqual(normal_scale.data_ptr(), fp32_scale.data_ptr())

        ue8m0_scale = torch.ones(6, 1, dtype=torch.int32)
        ue8m0_weight, preserved_scale = _kernel_fp8_weight_and_scale(
            weight, ue8m0_scale
        )
        self.assertIs(ue8m0_weight, weight)
        self.assertIs(preserved_scale, ue8m0_scale)
        self.assertEqual(tuple(ue8m0_weight.shape), (6, 4))
        self.assertEqual(tuple(preserved_scale.shape), (6, 1))

    def test_mla_fused_runtime_requants_before_child_post_load(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        requant_weight = weight + 1
        requant_scale = torch.ones(6, 1, dtype=torch.int32)
        requant = mock.Mock(return_value=(requant_weight, requant_scale))

        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "is_deep_gemm_e8m0_used",
                return_value=True,
            ) as e8m0_used,
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.attention."
                "_resolve_requant_weight_ue8m0",
                return_value=requant,
            ),
        ):
            runtime_weight, runtime_scale = _prepare_fused_fp8_runtime_weight(
                weight, scale
            )

        e8m0_used.assert_called_once_with(weight.device)
        requant.assert_called_once_with(weight, scale)
        self.assertIs(runtime_weight, requant_weight)
        self.assertIs(runtime_scale, requant_scale)
        self.assertEqual(runtime_scale.dtype, torch.int32)

    def test_mla_fused_runtime_keeps_standard_block_scales(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        scale = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        with mock.patch(
            "rtp_llm.models_py.new_models.deepseek_v3.attention."
            "is_deep_gemm_e8m0_used",
            return_value=False,
        ):
            runtime_weight, runtime_scale = _prepare_fused_fp8_runtime_weight(
                weight, scale
            )
        self.assertIs(runtime_weight, weight)
        self.assertIs(runtime_scale, scale)

    def test_mla_bf16_kernel_views_match_legacy_orientation(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=8,
            num_heads=2,
            q_lora_rank=4,
            kv_lora_rank=4,
            nope_head_dim=2,
            rope_head_dim=2,
            v_head_dim=2,
            layer_idx=0,
            params_dtype=torch.float32,
        )
        attention.process_weights_after_loading()
        weights = attention._build_mla_kernel_weights()

        self.assertFalse(hasattr(attention, "_fused_qkv_b_w"))
        self.assertEqual(
            tuple(weights[W.mla_fusedqkrope_w].shape),
            (8, 4 + 4 + 2),
        )
        self.assertEqual(
            tuple(weights[W.mla_q_b_w].shape),
            (4, 2 * (2 + 2)),
        )
        self.assertTrue(
            torch.equal(
                weights[W.mla_fusedqkrope_w],
                attention._fused_qkv_a_w.t(),
            )
        )
        self.assertTrue(
            torch.equal(weights[W.mla_q_b_w], attention.q_b_proj.weight.t())
        )
        self.assertNotIn(W.mla_fusedqkrope_s, weights)
        self.assertNotIn(W.mla_q_b_s, weights)

    def test_mla_fp8_kernel_views_keep_scales_without_bf16_qb_copy(self):
        attention = DeepSeekV32MlaAttention(
            hidden_size=128,
            num_heads=2,
            q_lora_rank=128,
            kv_lora_rank=128,
            nope_head_dim=64,
            rope_head_dim=64,
            v_head_dim=64,
            layer_idx=0,
            quant_config=QuantizationConfig("FP8_PER_BLOCK"),
            params_dtype=torch.bfloat16,
        )
        fused_weight = torch.zeros(
            128 + 128 + 64,
            128,
            dtype=torch.float8_e4m3fn,
        )
        fused_scale = torch.ones(3, 1, dtype=torch.float32)
        attention._fused_qkv_a_w = fused_weight
        attention._fused_qkv_a_s = fused_scale
        del attention.q_b_proj.weight_scale_inv
        attention.q_b_proj.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.ones(2, 1, dtype=torch.float32),
                requires_grad=False,
            ),
        )
        attention._kv_b_w = torch.empty(128, 256)
        attention._kc_w = torch.empty(2, 64, 128)
        attention._vc_w = torch.empty(2, 128, 64)

        weights = attention._build_mla_kernel_weights()

        self.assertEqual(
            tuple(weights[W.mla_fusedqkrope_w].shape),
            (128, 320),
        )
        self.assertEqual(
            tuple(weights[W.mla_fusedqkrope_s].shape),
            (1, 3),
        )
        self.assertEqual(
            tuple(weights[W.mla_q_b_w].shape),
            (128, 256),
        )
        self.assertEqual(
            tuple(weights[W.mla_q_b_s].shape),
            (1, 2),
        )
        self.assertEqual(weights[W.mla_q_b_w].dtype, torch.float8_e4m3fn)

    def test_non_noaux_router_preserves_scoring_grouping_and_scaling(self):
        logits = torch.tensor(
            [
                [1.0, 2.0, -1.0, 0.0],
                [-2.0, -1.0, 3.0, 2.0],
            ],
            dtype=torch.float32,
        )
        weights, ids = _select_deepseek_topk(
            logits,
            top_k=2,
            scoring_func=1,
            n_group=2,
            topk_group=1,
            group_limited=True,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        scores = logits.sigmoid()
        group_scores = scores.view(2, 2, 2).amax(dim=-1)
        selected_groups = group_scores.topk(1, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        masked = scores.masked_fill(
            ~group_mask.unsqueeze(-1).expand(-1, -1, 2).reshape_as(scores),
            float("-inf"),
        )
        expected_weights, expected_ids = masked.topk(2, dim=-1, sorted=False)
        expected_weights = (
            expected_weights
            / expected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            * 2.5
        )
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(weights, expected_weights))

        softmax_weights, softmax_ids = _select_deepseek_topk(
            logits,
            top_k=2,
            scoring_func=0,
            n_group=2,
            topk_group=1,
            group_limited=False,
            renormalize=False,
            routed_scaling_factor=1.5,
        )
        reference_weights, reference_ids = logits.softmax(dim=-1).topk(
            2, dim=-1, sorted=False
        )
        self.assertTrue(torch.equal(softmax_ids, reference_ids))
        self.assertTrue(torch.equal(softmax_weights, reference_weights * 1.5))

    def test_noaux_router_uses_bias_only_for_selection(self):
        logits = torch.tensor(
            [
                [2.0, 1.0, 0.0, -1.0, -2.0, 3.0, 0.5, -0.5],
                [-1.0, 0.0, 1.0, 2.0, 3.0, -2.0, -0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        correction_bias = torch.tensor(
            [-0.3, 0.2, 0.1, -0.2, 0.4, -0.4, 0.3, -0.1],
            dtype=torch.float32,
        )
        weights, ids = _select_deepseek_noaux_topk(
            logits,
            correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )

        scores = logits.sigmoid()
        choice_scores = scores + correction_bias
        grouped = choice_scores.view(2, 4, 2)
        group_scores = grouped.topk(2, dim=-1, sorted=False).values.sum(dim=-1)
        selected_groups = group_scores.topk(2, dim=-1, sorted=False).indices
        group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
        group_mask.scatter_(1, selected_groups, True)
        expert_mask = (
            group_mask.unsqueeze(-1).expand(-1, -1, 2).reshape_as(choice_scores)
        )
        expected_ids = (
            choice_scores.masked_fill(~expert_mask, float("-inf"))
            .topk(2, dim=-1, sorted=False)
            .indices
        )
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = (
            expected_weights
            / expected_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
            * 2.5
        )
        self.assertTrue(torch.equal(ids, expected_ids))
        self.assertTrue(torch.equal(weights, expected_weights))
        self.assertFalse(
            torch.equal(
                weights,
                choice_scores.gather(1, ids)
                / choice_scores.gather(1, ids).sum(dim=-1, keepdim=True)
                * 2.5,
            )
        )

    def test_noaux_router_constructs_without_cuda_group_topk(self):
        parallelism_config = types.SimpleNamespace(
            dp_rank=0,
            dp_size=1,
        )
        moe_config = types.SimpleNamespace(fake_balance_expert=False)
        model_config = types.SimpleNamespace(quant_config=None)
        with torch.device("cpu"):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=4,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=1.0,
                n_group=2,
                topk_group=1,
                topk_method="noaux_tc",
                correction_bias=True,
            )
        expect_fast_group_topk = get_device_type() == DeviceType.Cuda
        self.assertEqual(block._use_fast_group_topk, expect_fast_group_topk)
        self.assertEqual(block.group_topk is not None, expect_fast_group_topk)

    def test_non_noaux_router_avoids_cuda_select_topk_on_other_devices(self):
        parallelism_config = types.SimpleNamespace(dp_rank=0, dp_size=1)
        moe_config = types.SimpleNamespace(fake_balance_expert=False)
        model_config = types.SimpleNamespace(quant_config=None)
        with (
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.get_device_type",
                return_value=DeviceType.ROCm,
            ),
            mock.patch(
                "rtp_llm.models_py.new_models.deepseek_v3.moe.SelectTopk"
            ) as select_topk,
            torch.device("cpu"),
        ):
            block = DeepSeekV32MoEBlock(
                hidden_size=8,
                moe_intermediate_size=4,
                num_experts=4,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=model_config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=0,
                routed_scaling_factor=1.0,
                n_group=1,
                topk_group=1,
                topk_method="greedy",
                correction_bias=False,
            )
        self.assertFalse(block._use_fast_select_topk)
        self.assertIsNone(block.select_topk)
        select_topk.assert_not_called()

    def test_moe_forward_uses_reference_noaux_routing_on_cpu(self):
        parallelism_config = types.SimpleNamespace(
            dp_rank=0,
            dp_size=1,
        )
        with torch.device("cpu"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=4,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=types.SimpleNamespace(quant_config=None),
                parallelism_config=parallelism_config,
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.0,
                n_group=2,
                topk_group=1,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int64)
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states + topk_weights.sum(dim=-1, keepdim=True)

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor([[1.0, 0.5, -1.0, 2.0], [-0.5, 1.5, 0.25, -1.0]])
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.tensor([0.1, -0.2, 0.3, -0.4])
            )

        expected_weights, expected_ids = _select_deepseek_noaux_topk(
            block.gate(hidden_states).float(),
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=2,
            topk_group=1,
            renormalize=True,
            routed_scaling_factor=2.0,
        )
        output = block(hidden_states)

        self.assertTrue(torch.equal(capturing_experts.ids, expected_ids))
        self.assertTrue(torch.equal(capturing_experts.weights, expected_weights))
        self.assertTrue(
            torch.equal(
                output,
                hidden_states + expected_weights.sum(dim=-1, keepdim=True),
            )
        )

    def test_moe_rejects_invalid_fake_balance_and_input_rank(self):
        kwargs = dict(
            hidden_size=4,
            moe_intermediate_size=2,
            num_experts=4,
            top_k=2,
            layer_idx=0,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            model_config=types.SimpleNamespace(quant_config=None),
            parallelism_config=types.SimpleNamespace(dp_rank=0, dp_size=1),
            quant_config=None,
            params_dtype=torch.float32,
            has_shared_expert=False,
            scoring_func=0,
            routed_scaling_factor=2.0,
            n_group=1,
            topk_group=1,
            topk_method="greedy",
            correction_bias=False,
        )
        with self.assertRaisesRegex(TypeError, "fake_balance_expert"):
            DeepSeekV32MoEBlock(
                moe_config=types.SimpleNamespace(fake_balance_expert=1),
                **kwargs,
            )

        block = DeepSeekV32MoEBlock(
            moe_config=types.SimpleNamespace(fake_balance_expert=False),
            **kwargs,
        )
        block.experts.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            block(torch.zeros(1, 2, 4))

    @unittest.skipUnless(
        get_device_type() == DeviceType.Cuda,
        "CUDA GroupTopK is required",
    )
    def test_moe_cuda_noaux_router_matches_reference(self):
        with torch.device("cuda"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=8,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=types.SimpleNamespace(quant_config=None),
                parallelism_config=types.SimpleNamespace(dp_rank=0, dp_size=1),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.5,
                n_group=4,
                topk_group=2,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )
        self.assertTrue(block._use_fast_group_topk)

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int32)
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [-0.5, 1.5, 0.25, -1.0],
                [0.75, -0.25, 1.25, 0.5],
            ],
            device="cuda",
        )
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [0.5, -0.5, 0.25, 0.0],
                        [-0.25, 0.75, 0.0, 0.5],
                        [0.0, 0.25, 0.75, -0.5],
                        [0.5, 0.0, -0.25, 0.75],
                    ],
                    device="cuda",
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.tensor(
                    [0.1, -0.2, 0.3, -0.4, 0.05, 0.15, -0.1, 0.2],
                    device="cuda",
                )
            )

        router_logits = block.gate(hidden_states).float()
        expected_weights, expected_ids = _select_deepseek_noaux_topk(
            router_logits,
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        output = block(hidden_states)

        actual_order = capturing_experts.ids.argsort(dim=-1)
        expected_order = expected_ids.argsort(dim=-1)
        actual_ids = capturing_experts.ids.gather(1, actual_order)
        expected_ids = expected_ids.gather(1, expected_order).to(actual_ids.dtype)
        actual_weights = capturing_experts.weights.gather(1, actual_order)
        expected_weights = expected_weights.gather(1, expected_order)
        self.assertTrue(torch.equal(actual_ids, expected_ids))
        self.assertTrue(torch.allclose(actual_weights, expected_weights, atol=1e-6))
        self.assertTrue(torch.equal(output, hidden_states))

    @unittest.skipUnless(
        get_device_type() == DeviceType.ROCm,
        "ROCm is required",
    )
    def test_moe_rocm_noaux_reference_router_matches_expected(self):
        with torch.device("cuda"):
            block = DeepSeekV32MoEBlock(
                hidden_size=4,
                moe_intermediate_size=2,
                num_experts=8,
                top_k=2,
                layer_idx=0,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                model_config=types.SimpleNamespace(quant_config=None),
                parallelism_config=types.SimpleNamespace(dp_rank=0, dp_size=1),
                moe_config=types.SimpleNamespace(fake_balance_expert=False),
                quant_config=None,
                params_dtype=torch.float32,
                has_shared_expert=False,
                scoring_func=1,
                routed_scaling_factor=2.5,
                n_group=4,
                topk_group=2,
                topk_method="noaux_tc",
                has_moe_norm=True,
                correction_bias=True,
            )
        self.assertFalse(block._use_fast_group_topk)
        self.assertIsNone(block.group_topk)

        class CapturingExperts(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fused_moe = types.SimpleNamespace(topk_ids_dtype=torch.int32)
                self.weights = None
                self.ids = None

            def forward(self, hidden_states, topk_weights, topk_ids):
                self.weights = topk_weights.clone()
                self.ids = topk_ids.clone()
                return hidden_states

        capturing_experts = CapturingExperts()
        block.experts = capturing_experts
        hidden_states = torch.tensor(
            [
                [1.0, 0.5, -1.0, 2.0],
                [-0.5, 1.5, 0.25, -1.0],
            ],
            device="cuda",
        )
        with torch.no_grad():
            block.gate.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0, 1.0],
                    ],
                    device="cuda",
                )
            )
            block.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.2, 0.2, 8, device="cuda")
            )

        expected_weights, expected_ids = _select_deepseek_noaux_topk(
            block.gate(hidden_states).float(),
            block.gate.e_score_correction_bias,
            top_k=2,
            n_group=4,
            topk_group=2,
            renormalize=True,
            routed_scaling_factor=2.5,
        )
        block(hidden_states)
        self.assertTrue(torch.equal(capturing_experts.ids, expected_ids))
        torch.testing.assert_close(capturing_experts.weights, expected_weights)

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
        self.assertEqual(attention._fused_qkv_a_w.dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(attention._fused_qkv_a_s)
        self.assertEqual(
            tuple(attention._fused_qkv_a_runtime.weight.shape),
            (128 + 128 + 2, 128),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
    @unittest.skipIf(
        getattr(torch.version, "hip", None) is not None,
        "DeepGEMM fused FP8 projection is CUDA-only",
    )
    def test_mla_online_fused_fp8_projection_matches_bf16_reference(self):
        device = torch.device("cuda")
        with torch.device(device):
            attention = DeepSeekV32MlaAttention(
                hidden_size=128,
                num_heads=2,
                q_lora_rank=128,
                kv_lora_rank=128,
                nope_head_dim=64,
                rope_head_dim=128,
                v_head_dim=64,
                layer_idx=0,
                quant_config=QuantizationConfig("fp8_block_online"),
                params_dtype=torch.bfloat16,
            )
        generator = torch.Generator(device=device).manual_seed(20260727)
        q_a_weight = torch.randn(
            attention.q_a_proj.weight.shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        kv_a_weight = torch.randn(
            attention.kv_a_proj_with_mqa.weight.shape,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        attention.q_a_proj.weight.data.copy_(q_a_weight)
        attention.kv_a_proj_with_mqa.weight.data.copy_(kv_a_weight)
        reference_weight = torch.cat([q_a_weight, kv_a_weight], dim=0)

        attention.process_weights_after_loading()
        self.assertIsNotNone(attention._fused_qkv_a_runtime)
        x = torch.randn(
            (7, 128),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        actual = attention._fused_qkv_a_runtime(x)
        expected = torch.nn.functional.linear(x, reference_weight)
        difference = actual.float() - expected.float()
        relative_rmse = (
            difference.square().mean().sqrt() / expected.float().square().mean().sqrt()
        )
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(),
            expected.float().flatten(),
            dim=0,
        )
        self.assertLess(float(relative_rmse), 0.05)
        self.assertGreater(float(cosine), 0.998)

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
