import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QuantizationConfig as SourceQuantizationConfig
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops import EplbMode

_MODEL_QUANT_UNSET = object()


def _parallelism(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, dp_size=1):
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=dp_size,
        dp_rank=0,
        world_size=max(tp_size, ep_size, dp_size),
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
    )


def _runtime_model_config(num_experts=2, quant_config=None, hidden_size=8):
    return types.SimpleNamespace(
        data_type="fp32",
        quant_config=quant_config,
        exported_device=None,
        expert_num=num_experts,
        moe_k=1,
        moe_topk_group=1,
        hidden_size=hidden_size,
        activation_type="SiGLU",
        attn_config=types.SimpleNamespace(head_num=2),
    )


def _make_experts(
    num_experts=2,
    top_k=1,
    hidden_size=8,
    moe_intermediate_size=8,
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    dp_size=1,
    quant_config=None,
    model_quant_config=_MODEL_QUANT_UNSET,
):
    quant_config = quant_config or QuantizationConfig("none")
    if model_quant_config is _MODEL_QUANT_UNSET:
        runtime_method = quant_config.quant_type
        model_quant_config = (
            None
            if runtime_method == "none"
            else types.SimpleNamespace(
                get_runtime_method_key=lambda: runtime_method,
                get_method=lambda: runtime_method,
            )
        )
    return BaseMoEExperts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        params_dtype=torch.float32,
        model_config=_runtime_model_config(
            num_experts, model_quant_config, hidden_size
        ),
        parallelism_config=_parallelism(tp_size, tp_rank, ep_size, ep_rank, dp_size),
        moe_config=_moe_config(),
        quant_config=quant_config,
        layer_idx=0,
        prefix="layers.0.mlp.experts",
    )


def _expert_weights(num_experts, hidden_size, intermediate_size):
    result = {}
    for expert_id in range(num_experts):
        offset = float(expert_id * 1000)
        result[f"{expert_id}.gate_proj.weight"] = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .add(offset + 100)
        )
        result[f"{expert_id}.up_proj.weight"] = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .add(offset + 200)
        )
        result[f"{expert_id}.down_proj.weight"] = (
            torch.arange(hidden_size * intermediate_size, dtype=torch.float32)
            .reshape(hidden_size, intermediate_size)
            .add(offset + 300)
        )
    return result


class MoEWeightDispatchTest(unittest.TestCase):
    def test_ep_filter_selects_local_experts_before_tensor_loading(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)

        self.assertFalse(
            rank_one.should_load("model.layers.0.mlp.experts.1.gate_proj.weight")
        )
        self.assertTrue(
            rank_one.should_load("model.layers.0.mlp.experts.2.gate_proj.weight")
        )
        self.assertTrue(rank_one.should_load("model.layers.0.self_attn.q_proj.weight"))
        with self.assertRaisesRegex(ValueError, "per-expert checkpoint keys"):
            rank_one.should_load("model.layers.0.mlp.experts.gate_up_proj.weight")
        with self.assertRaisesRegex(ValueError, "outside"):
            rank_one.should_load("model.layers.0.mlp.experts.4.gate_proj.weight")

    def test_safetensors_ep_filter_runs_before_get_tensor(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)
        handle = mock.MagicMock()
        handle.keys.return_value = [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.2.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        handle.get_tensor.side_effect = lambda name: torch.tensor([len(name)])
        context = mock.MagicMock()
        context.__enter__.return_value = handle

        with mock.patch("safetensors.safe_open", return_value=context):
            loaded = list(
                weight_mapper.get_all_weights(
                    ["model.safetensors"],
                    name_filter=rank_one.should_load,
                )
            )

        self.assertEqual(
            [name for name, _ in loaded],
            [
                "model.layers.0.mlp.experts.2.gate_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
        )
        handle.get_tensor.assert_has_calls(
            [
                mock.call("model.layers.0.mlp.experts.2.gate_proj.weight"),
                mock.call("model.layers.0.self_attn.q_proj.weight"),
            ]
        )
        self.assertEqual(handle.get_tensor.call_count, 2)

    def test_rms_res_norm_preserves_decoder_residual_contract(self):
        layer = RMSResNorm(2, eps=1e-6, params_dtype=torch.float32)
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])

        normalized, residual_out = layer(hidden, residual)

        expected_residual = hidden + residual
        variance = expected_residual.float().pow(2).mean(-1, keepdim=True)
        expected = expected_residual * torch.rsqrt(variance + 1e-6)
        torch.testing.assert_close(residual_out, expected_residual)
        torch.testing.assert_close(normalized, expected)

    def test_rms_res_norm_rejects_mismatched_dtype(self):
        layer = RMSResNorm(2, params_dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "share a dtype"):
            layer(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.float64))

    def test_rms_res_norm_rejects_non_matrix_and_weight_dtype_mismatch(self):
        layer = RMSResNorm(2, params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "expected \\[tokens, 2\\]"):
            layer(torch.ones(1, 1, 2), torch.ones(1, 1, 2))

        layer = RMSResNorm(2, params_dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "weight and inputs"):
            layer(torch.ones(1, 2), torch.ones(1, 2))

    def test_ep_rejects_pytorch_checkpoint_before_deserialization(self):
        with tempfile.TemporaryDirectory() as model_path:
            checkpoint = os.path.join(model_path, "model.pt")
            with open(checkpoint, "wb"):
                pass
            loader = NewModelLoader(
                types.SimpleNamespace(
                    model_type="qwen_3_moe",
                    model_path=model_path,
                    expert_num=2,
                ),
                NewLoaderConfig(
                    ep_size=2,
                    ep_rank=0,
                    device="cpu",
                ),
            )
            with mock.patch.object(
                weight_mapper, "_load_pytorch", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(ValueError, "EP streaming requires"):
                    loader.load()

    def test_per_expert_weights_pack_up_before_gate(self):
        layer = _make_experts(num_experts=2)
        weights = _expert_weights(2, 8, 8)

        for name, tensor in weights.items():
            layer.load_weights({name: tensor})
        layer._check_load_complete()

        for expert_id in range(2):
            torch.testing.assert_close(
                layer.w13[expert_id, :8],
                weights[f"{expert_id}.up_proj.weight"],
            )
            torch.testing.assert_close(
                layer.w13[expert_id, 8:],
                weights[f"{expert_id}.gate_proj.weight"],
            )
            torch.testing.assert_close(
                layer.w2[expert_id],
                weights[f"{expert_id}.down_proj.weight"],
            )

    def test_tp_slices_gate_up_rows_and_down_columns(self):
        weights = _expert_weights(1, 8, 8)
        for rank in range(2):
            layer = _make_experts(num_experts=1, tp_size=2, tp_rank=rank)
            layer.load_weights(weights)
            start = rank * 4
            torch.testing.assert_close(
                layer.w13[0, :4], weights["0.up_proj.weight"][start : start + 4]
            )
            torch.testing.assert_close(
                layer.w13[0, 4:], weights["0.gate_proj.weight"][start : start + 4]
            )
            torch.testing.assert_close(
                layer.w2[0], weights["0.down_proj.weight"][:, start : start + 4]
            )

    def test_ep_loads_only_local_experts(self):
        weights = _expert_weights(4, 8, 8)
        layer = _make_experts(num_experts=4, ep_size=2, ep_rank=1)
        layer.load_weights(weights)
        layer._check_load_complete()

        torch.testing.assert_close(layer.w13[0, :8], weights["2.up_proj.weight"])
        torch.testing.assert_close(layer.w13[1, :8], weights["3.up_proj.weight"])

    def test_invalid_ep_and_expert_count_fail_fast(self):
        with self.assertRaisesRegex(
            ValueError, "num_experts must be a positive integer"
        ):
            _make_experts(num_experts=0)
        with self.assertRaisesRegex(ValueError, "divisible by ep_size"):
            _make_experts(num_experts=3, ep_size=2)
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            _make_experts(num_experts=4, ep_size=2, ep_rank=2)
        for kwargs in (
            {"num_experts": True},
            {"tp_size": True},
            {"tp_size": 2, "tp_rank": True},
            {"ep_size": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "positive integer|rank"
            ):
                _make_experts(**kwargs)

    def test_malformed_unknown_and_out_of_range_names_fail(self):
        layer = _make_experts(num_experts=2)
        cases = {
            "bad": torch.zeros(8, 8),
            "0.unknown.weight": torch.zeros(8, 8),
            "2.gate_proj.weight": torch.zeros(8, 8),
            "0.gate_proj.weight.extra": torch.zeros(8, 8),
        }
        for name, tensor in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                _make_experts(num_experts=2).load_weights({name: tensor})

    def test_duplicate_and_missing_weights_fail(self):
        layer = _make_experts(num_experts=1)
        gate = torch.zeros(8, 8)
        layer.load_weights({"0.gate_proj.weight": gate})
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights({"0.gate_proj.weight": gate})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            layer._check_load_complete()

    def test_fused_and_separate_projection_aliases_cannot_overwrite(self):
        layer = _make_experts(num_experts=1, hidden_size=12, moe_intermediate_size=4)
        gate = torch.ones(4, 12)
        layer.load_weights({"0.gate_proj.weight": gate})
        before = layer.w13[0, 4:].clone()

        with self.assertRaisesRegex(RuntimeError, "Duplicate MoE projection"):
            layer.load_weights({"0.gate_up_proj.weight": torch.zeros(8, 12)})
        torch.testing.assert_close(layer.w13[0, 4:], before)

    def test_stacked_weight_suffix_supports_both_layouts(self):
        for transpose_layout in (False, True):
            layer = _make_experts(
                num_experts=2, hidden_size=12, moe_intermediate_size=4
            )
            gate = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12)
            up = gate + 1000
            down = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4)
            down = down + 2000
            gate_up = torch.cat((gate, up), dim=1)
            stacked_gate_up = (
                gate_up.transpose(1, 2).contiguous() if transpose_layout else gate_up
            )
            stacked_down = (
                down.transpose(1, 2).contiguous() if transpose_layout else down
            )
            layer.load_weights(
                {
                    "gate_up_proj.weight": stacked_gate_up,
                    "down_proj.weight": stacked_down,
                }
            )
            layer._check_load_complete()
            torch.testing.assert_close(layer.w13[:, :4], up)
            torch.testing.assert_close(layer.w13[:, 4:], gate)
            torch.testing.assert_close(layer.w2, down)

    def test_ambiguous_stacked_layout_fails_instead_of_guessing(self):
        layer = _make_experts(num_experts=1, hidden_size=8, moe_intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            layer.load_weights({"gate_up_proj.weight": torch.zeros(1, 8, 8)})

    def test_ambiguous_per_expert_fused_layout_fails(self):
        layer = _make_experts(num_experts=1, hidden_size=8, moe_intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            layer.load_weights({"0.gate_up_proj.weight": torch.zeros(8, 8)})

    def test_forward_requires_completed_post_load_processing(self):
        layer = _make_experts(num_experts=1)
        with self.assertRaisesRegex(RuntimeError, "post-load processing"):
            layer(
                torch.zeros(1, 8),
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
            )

    def test_forward_rejects_non_matrix_hidden_states(self):
        layer = _make_experts(num_experts=1)
        layer.fused_moe = mock.MagicMock()
        with self.assertRaisesRegex(ValueError, "\[tokens, 8\]"):
            layer(
                torch.zeros(1, 1, 8),
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
            )

    def test_ep_rejects_stacked_all_expert_checkpoint(self):
        layer = _make_experts(num_experts=2, ep_size=2, ep_rank=0)
        with self.assertRaisesRegex(ValueError, "per-expert checkpoint keys"):
            layer.load_weights({"gate_up_proj.weight": torch.zeros(2, 16, 8)})


class MoEQuantizedDispatchTest(unittest.TestCase):
    def test_prequantized_fp8_rejects_unquantized_weight_dtype(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        with self.assertRaisesRegex(TypeError, "requires FP8 storage"):
            layer.load_weights({"0.gate_proj.weight": torch.zeros(8, 8)})

    def test_fp8_scale_dtype_and_values_are_validated(self):
        for scale in (
            torch.tensor(1, dtype=torch.int32),
            torch.tensor(0.0),
            torch.tensor(float("inf")),
        ):
            with self.subTest(scale=scale), self.assertRaises((TypeError, ValueError)):
                layer = _make_experts(
                    num_experts=1,
                    quant_config=QuantizationConfig("fp8"),
                )
                layer.load_weights({"0.gate_proj.weight_scale": scale})

    def test_single_channel_fp8_scale_keeps_vector_shape(self):
        layer = _make_experts(
            num_experts=1,
            hidden_size=1,
            moe_intermediate_size=1,
            quant_config=QuantizationConfig("fp8_per_channel"),
        )
        scale = torch.ones(1, 1)

        layer.load_weights(
            {
                "0.gate_proj.weight_scale": scale,
                "0.up_proj.weight_scale": scale,
                "0.down_proj.weight_scale": scale,
            }
        )

        self.assertEqual(tuple(layer._gate_ch_scales.shape), (1, 1))
        self.assertEqual(tuple(layer._up_ch_scales.shape), (1, 1))
        self.assertEqual(tuple(layer._down_ch_scales.shape), (1, 1))

    def test_custom_block_conversion_still_applies_ue8m0(self):
        layer = torch.nn.Module()
        layer._FP8_BLOCK_SIZE = 128
        layer._fp8_moe_weight_block_size = [64, 64]
        layer.w13 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_scale = torch.ones(1, 2, 2)
        layer.w2_scale = torch.ones(1, 2, 2)
        layer.num_local_experts = 1
        converted = []

        def convert(weight, scale):
            converted.append((weight, scale))
            return weight, scale.to(torch.uint8)

        from rtp_llm.models_py.quant_methods.fp8_moe import Fp8MoEMethod

        method = Fp8MoEMethod.__new__(Fp8MoEMethod)
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8_moe._runtime_fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ), mock.patch.multiple(
            "rtp_llm.models_py.quant_methods.fp8",
            per_block_quant_like_legacy=mock.DEFAULT,
            is_deep_gemm_e8m0_used=mock.DEFAULT,
            _resolve_requant_weight_ue8m0=mock.DEFAULT,
        ) as dependencies:
            dependencies["per_block_quant_like_legacy"].side_effect = (
                lambda weight, block: (
                    weight.to(torch.float8_e4m3fn),
                    torch.ones(1, 1),
                )
            )
            dependencies["is_deep_gemm_e8m0_used"].return_value = True
            dependencies["_resolve_requant_weight_ue8m0"].return_value = convert
            method._requant_per_block_if_needed(layer)

        self.assertEqual(len(converted), 2)
        self.assertEqual(layer.w13_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_scale.dtype, torch.uint8)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fnuz"), "FNUZ dtype is unavailable")
    def test_custom_block_conversion_preserves_runtime_fp8_dtype(self):
        layer = torch.nn.Module()
        layer._FP8_BLOCK_SIZE = 128
        layer._fp8_moe_weight_block_size = [64, 64]
        layer.w13 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_scale = torch.ones(1, 2, 2)
        layer.w2_scale = torch.ones(1, 2, 2)
        layer.num_local_experts = 1

        from rtp_llm.models_py.quant_methods.fp8_moe import Fp8MoEMethod

        method = Fp8MoEMethod.__new__(Fp8MoEMethod)
        runtime_dtype = torch.float8_e4m3fnuz
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8_moe._runtime_fp8_dtype",
            return_value=runtime_dtype,
        ), mock.patch.multiple(
            "rtp_llm.models_py.quant_methods.fp8",
            per_block_quant_like_legacy=mock.DEFAULT,
            is_deep_gemm_e8m0_used=mock.DEFAULT,
            _resolve_requant_weight_ue8m0=mock.DEFAULT,
        ) as dependencies:
            dependencies["per_block_quant_like_legacy"].side_effect = (
                lambda weight, block: (
                    weight.to(runtime_dtype),
                    torch.ones(1, 1),
                )
            )
            dependencies["is_deep_gemm_e8m0_used"].return_value = False
            method._requant_per_block_if_needed(layer)

        self.assertEqual(layer.w13.dtype, runtime_dtype)
        self.assertEqual(layer.w2.dtype, runtime_dtype)

    def test_w4a8_rejects_unsupported_cuda_capability_before_import(self):
        from rtp_llm.models_py.quant_methods.w4a8_moe import W4A8Int4MoEMethod

        method = W4A8Int4MoEMethod(QuantizationConfig("W4A8_INT4_PER_CHANNEL"))
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ), mock.patch.object(
            torch.version, "hip", None
        ):
            with self.assertRaisesRegex(RuntimeError, "capability 8.9"):
                method.validate_runtime_device(torch.device("cuda:0"))

    def test_online_and_unquantized_moe_reject_integer_weights(self):
        for quant_type in ("none", "fp8_online", "W4A8_INT4_PER_CHANNEL"):
            with self.subTest(quant_type=quant_type), self.assertRaisesRegex(
                TypeError, "floating"
            ):
                size = 128 if quant_type == "W4A8_INT4_PER_CHANNEL" else 8
                layer = _make_experts(
                    num_experts=1,
                    hidden_size=size,
                    moe_intermediate_size=size,
                    quant_config=QuantizationConfig(quant_type),
                )
                layer.load_weights(
                    {"0.gate_proj.weight": torch.zeros(size, size, dtype=torch.int8)}
                )

    def test_w4a8_repack_rejects_invalid_scales_before_kernel_import(self):
        from rtp_llm.models_py.quant_methods.w4a8_utils import (
            repack_compressed_int4_to_cutlass,
        )

        packed = torch.zeros(1, 64, dtype=torch.int8)
        for scale in (
            torch.tensor([[0.0]]),
            torch.tensor([[float("inf")]]),
            torch.tensor([[1000.0]]),
        ):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "finite and positive|representable"
            ):
                repack_compressed_int4_to_cutlass(packed, scale, 128)

    def test_invalid_fp8_block_size_types_fail_fast(self):
        for block_size in (128, [True, 128], [64.5, 128], [128]):
            with self.subTest(block_size=block_size), self.assertRaises(
                (TypeError, ValueError)
            ):
                quant = QuantizationConfig(
                    "fp8_block",
                    source_config=types.SimpleNamespace(weight_block_size=block_size),
                )
                _make_experts(num_experts=1, quant_config=quant)

    def test_runtime_and_model_quantization_must_match(self):
        source_quant = types.SimpleNamespace(get_runtime_method_key=lambda: "fp8_block")
        with self.assertRaisesRegex(ValueError, "quantization mismatch"):
            _make_experts(
                num_experts=1,
                quant_config=QuantizationConfig("fp8"),
                model_quant_config=source_quant,
            )
        with self.assertRaisesRegex(ValueError, "without model_config.quant_config"):
            _make_experts(
                num_experts=1,
                quant_config=QuantizationConfig("fp8"),
                model_quant_config=None,
            )

    def test_missing_fp8_scale_fails(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        fp8 = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
        layer.load_weights(
            {
                "0.gate_proj.weight": fp8,
                "0.up_proj.weight": fp8,
                "0.down_proj.weight": fp8,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "auxiliary tensors"):
            layer._check_load_complete()

    def test_fused_and_separate_scale_aliases_cannot_overwrite(self):
        layer = _make_experts(
            num_experts=1,
            hidden_size=12,
            moe_intermediate_size=4,
            quant_config=QuantizationConfig("fp8"),
        )
        layer.load_weights({"0.gate_proj.weight_scale": torch.tensor(1.0)})
        with self.assertRaisesRegex(RuntimeError, "Duplicate MoE auxiliary"):
            layer.load_weights(
                {"0.gate_up_proj.weight_scale": torch.tensor([2.0, 3.0])}
            )

    def test_fused_block_scale_uses_global_grid_before_tp_slice(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=16,
            tp_size=2,
            tp_rank=1,
            quant_config=quant,
        )
        fused = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8).add(1)
        layer.load_weights({"gate_up_proj.weight_scale_inv": fused})

        gate_global = fused[0, :, :4].t()
        up_global = fused[0, :, 4:].t()
        torch.testing.assert_close(layer.w13_scale[0, :2], up_global[2:4])
        torch.testing.assert_close(layer.w13_scale[0, 2:], gate_global[2:4])

    def test_single_tp_allows_partial_final_fp8_block(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=10,
            quant_config=quant,
        )
        layer.load_weights(
            {
                "0.gate_proj.weight_scale_inv": torch.ones(3, 2),
                "0.up_proj.weight_scale_inv": torch.ones(3, 2),
                "0.down_proj.weight_scale_inv": torch.ones(2, 3),
            }
        )
        self.assertEqual(len(layer._loaded_aux_keys), 3)

    def test_non_aligned_fp8_block_tp_is_rejected(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        with self.assertRaisesRegex(ValueError, "block-aligned"):
            _make_experts(
                num_experts=1,
                hidden_size=8,
                moe_intermediate_size=10,
                tp_size=2,
                quant_config=quant,
            )

    def test_stacked_w4a8_weight_packed_is_not_treated_as_scale(self):
        quant = QuantizationConfig(
            "W4A8_INT4_PER_CHANNEL_COMPRESSED",
            source_config=types.SimpleNamespace(group_size=lambda: 4),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=8,
            quant_config=quant,
        )
        gate_up = torch.arange(16 * 4, dtype=torch.int8).reshape(1, 16, 4)
        down = torch.arange(8 * 4, dtype=torch.int8).reshape(1, 8, 4)
        gate_up_scale = torch.ones(1, 16, 2, dtype=torch.bfloat16)
        down_scale = torch.ones(1, 8, 2, dtype=torch.bfloat16)
        layer.load_weights(
            {
                "gate_up_proj.weight_packed": gate_up,
                "down_proj.weight_packed": down,
                "gate_up_proj.weight_scale": gate_up_scale,
                "down_proj.weight_scale": down_scale,
            }
        )
        layer._check_load_complete()
        self.assertTrue(layer.requires_staged_device_postprocess())

        torch.testing.assert_close(layer._w4a8_gate_packed[0], gate_up[0, :8])
        torch.testing.assert_close(layer._w4a8_up_packed[0], gate_up[0, 8:])
        torch.testing.assert_close(layer._w4a8_down_packed[0], down[0])

    def test_ignored_moe_layer_uses_unquantized_method_and_runtime_config(self):
        source_quant = types.SimpleNamespace(get_method=lambda: "FP8")
        quant = QuantizationConfig(
            "fp8",
            ignored_layers=["model.layers.{i}.mlp"],
        )
        layer = _make_experts(
            num_experts=1,
            quant_config=quant,
            model_quant_config=source_quant,
        )

        self.assertEqual(layer._quant_family, "none")
        self.assertEqual(layer._required_aux_param_names(), ())
        self.assertIsNone(layer._effective_model_quant_config)
        self.assertEqual(layer.w13.dtype, torch.float32)

    def test_partial_projection_ignore_is_rejected(self):
        quant = QuantizationConfig(
            "fp8",
            ignored_layers=["layers.0.mlp.experts.gate_proj"],
        )
        with self.assertRaisesRegex(ValueError, "partially match fused MoE"):
            _make_experts(num_experts=1, quant_config=quant)

    def test_all_projection_ignores_disable_quantization(self):
        patterns = [
            f"layers.0.mlp.experts.{projection}"
            for projection in BaseMoEExperts.PROJ_NAMES
        ]
        quant = QuantizationConfig("fp8", ignored_layers=patterns)
        layer = _make_experts(num_experts=1, quant_config=quant)
        self.assertEqual(layer._quant_family, "none")

    def test_w4a8_is_moe_only_for_linear_dispatch(self):
        quant = QuantizationConfig("W4A8_INT4_PER_CHANNEL")
        method = quant.get_quant_method(
            types.SimpleNamespace(shard_names=()), "layers.0.self_attn.o_proj"
        )
        self.assertEqual(type(method).__name__, "UnquantizedLinearMethod")

    def test_cuda_graph_flag_is_forwarded_to_moe_adapter(self):
        quant = QuantizationConfig(
            "none",
            hw_kernel_config=types.SimpleNamespace(enable_cuda_graph=True),
        )
        layer = _make_experts(num_experts=1, quant_config=quant)
        self.assertTrue(layer._enable_cuda_graph())


class MoeRuntimeConfigTest(unittest.TestCase):
    def test_adapter_distinguishes_inherited_and_explicit_none_quant_config(self):
        source_quant = types.SimpleNamespace(get_method=lambda: "FP8")
        model_config = _runtime_model_config(2, source_quant)
        parallelism = _parallelism()
        moe_config = _moe_config()

        inherited = MoEConfigAdapter(model_config, parallelism, moe_config)
        ignored = MoEConfigAdapter(
            model_config, parallelism, moe_config, quant_config=None
        )

        self.assertIs(inherited.quant_config, source_quant)
        self.assertTrue(MoeConfigResolver.has_quantization(inherited))
        self.assertIsNone(ignored.quant_config)
        self.assertFalse(MoeConfigResolver.has_quantization(ignored))


class Qwen3MoeModelTest(unittest.TestCase):
    def _config(self, tie_word_embeddings=True):
        config = ModelConfig()
        config.model_type = "qwen_3_moe"
        config.num_layers = 1
        config.vocab_size = 8
        config.hidden_size = 4
        config.inter_size = 4
        config.expert_num = 2
        config.moe_inter_size = 4
        config.moe_k = 1
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
        return config

    def _load_config(self, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
        parallelism = _parallelism(tp_size, tp_rank, ep_size, ep_rank)
        return NewLoaderConfig(
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            compute_dtype=torch.float32,
            device="cpu",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
            moe_config=_moe_config(),
        )

    def _weights(self):
        weights = {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
                8, 4
            ),
            "model.layers.0.input_layernorm.weight": torch.ones(4),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
            "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
            "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
            "model.layers.0.mlp.gate.weight": torch.ones(2, 4),
            "model.norm.weight": torch.ones(4),
        }
        for name, tensor in _expert_weights(2, 4, 4).items():
            weights[f"model.layers.0.mlp.experts.{name}"] = tensor
        return weights

    def test_real_model_tree_loads_and_ties_lm_head(self):
        model = Qwen3MoeForCausalLM(self._config(), self._load_config())
        model.load_weights(self._weights())
        NewModelLoader._validate_loaded_weights(model)

        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertEqual(model.layers[0].mlp.experts._loaded_count, 6)
        self.assertEqual(
            set(model.runtime_weight_view()),
            {"embedding", "final_layernorm.gamma", "lm_head"},
        )

    def test_hf_dict_is_rejected_before_router_construction(self):
        config = {
            "model_type": "qwen_3_moe",
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "norm_topk_prob": True,
            "tie_word_embeddings": True,
        }

        with self.assertRaisesRegex(TypeError, "requires a typed ModelConfig"):
            Qwen3MoeForCausalLM(config, self._load_config())

    def test_real_select_topk_receives_runtime_model_config(self):
        config = self._config()

        model = Qwen3MoeForCausalLM(config, self._load_config())

        self.assertIs(model.layers[0].mlp.select_topk.config, config)

    def test_router_config_rejects_invalid_normalization_flag(self):
        config = self._config()
        with self.assertRaises(TypeError):
            config.has_moe_norm = "true"

    def test_zero_dense_intermediate_size_uses_moe_intermediate_size(self):
        config = self._config()
        config.inter_size = 0

        model = Qwen3MoeForCausalLM(config, self._load_config())

        self.assertEqual(model.layers[0].mlp.experts.moe_inter, 4)

    def test_eplb_attribute_is_rejected_before_model_load(self):
        config = self._config()
        config.eplb_config.eplb_mode = EplbMode.EPLB

        with self.assertRaisesRegex(ValueError, "EPLB is not supported"):
            Qwen3MoeForCausalLM(config, self._load_config())

    def test_invalid_adapter_cuda_graph_flag_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "enable_cuda_graph must be a bool"):
            MoEConfigAdapter(
                self._config(),
                _parallelism(),
                _moe_config(),
                enable_cuda_graph=1,
            )

    def test_w4a8_source_config_rejects_invalid_dimensions(self):
        from rtp_llm.config.quant_config import (
            CompressedW4A8Int4PerChannelQuantConfig,
            W4a8Int4PerChannelQuantConfig,
        )

        for config_cls in (
            W4a8Int4PerChannelQuantConfig,
            CompressedW4A8Int4PerChannelQuantConfig,
        ):
            for kwargs in (
                {"bits": 8, "group_size": 128},
                {"bits": 4, "group_size": 0},
                {"bits": True, "group_size": 128},
            ):
                with self.subTest(config=config_cls.__name__, kwargs=kwargs):
                    with self.assertRaises((TypeError, ValueError)):
                        config_cls(is_quanted=True, **kwargs)

    def test_invalid_boolean_config_is_rejected(self):
        config = self._config()
        config.tie_word_embeddings = "true"
        with self.assertRaisesRegex(TypeError, "tie_word_embeddings"):
            Qwen3MoeForCausalLM(config, self._load_config())

        config = self._config()
        load_config = self._load_config()
        load_config.moe_config.fake_balance_expert = 1
        with self.assertRaisesRegex(TypeError, "fake_balance_expert"):
            Qwen3MoeForCausalLM(config, load_config)

    def test_untied_missing_lm_head_fails_integrity(self):
        model = Qwen3MoeForCausalLM(
            self._config(tie_word_embeddings=False), self._load_config()
        )
        model.load_weights(self._weights())
        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_tp_owners_use_reviewed_topology(self):
        model = Qwen3MoeForCausalLM(
            self._config(), self._load_config(tp_size=2, tp_rank=1)
        )
        self.assertEqual(
            (model.embed_tokens.tp_size, model.embed_tokens.tp_rank), (2, 1)
        )
        self.assertEqual(
            (
                model.layers[0].self_attn.qkv_proj.tp_size,
                model.layers[0].self_attn.qkv_proj.tp_rank,
            ),
            (2, 1),
        )
        experts = model.layers[0].mlp.experts
        self.assertEqual(
            (experts.moe_expert_tp_size, experts.moe_expert_tp_rank), (2, 1)
        )
        self.assertEqual((model.lm_head.tp_size, model.lm_head.tp_rank), (2, 1))

    def test_router_gate_respects_fp8_ignore_rule(self):
        source_quant = types.SimpleNamespace(
            get_runtime_method_key=lambda: "FP8_PER_BLOCK",
            get_method=lambda: "FP8_PER_BLOCK",
            weight_block_size=[128, 128],
            modules_to_not_convert=["model.layers.{i}.mlp.gate"],
        )
        config = self._config()
        config.quant_config = source_quant
        load_config = self._load_config()
        load_config = types.SimpleNamespace(
            **{
                **load_config.__dict__,
                "quant_config": QuantizationConfig(
                    "FP8_PER_BLOCK",
                    source_config=source_quant,
                ),
            }
        )

        model = Qwen3MoeForCausalLM(config, load_config)

        self.assertEqual(
            type(model.layers[0].mlp.gate.quant_method).__name__,
            "UnquantizedLinearMethod",
        )
        self.assertEqual(model.layers[0].mlp.experts._quant_family, "fp8_per_block")

    def test_checkpoint_modules_to_not_convert_reaches_runtime_dispatch(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "config.json"), "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "weight_block_size": [128, 128],
                            "modules_to_not_convert": [
                                "model.layers.0.mlp.gate",
                                "lm_head",
                            ],
                        }
                    },
                    output,
                )

            source_quant = SourceQuantizationConfig.load_from_ckpt(model_path)

        self.assertIsNotNone(source_quant)
        runtime_quant = QuantizationConfig(
            source_quant.get_runtime_method_key(), source_config=source_quant
        )
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.mlp.gate"))
        self.assertTrue(runtime_quant.is_layer_ignored("lm_head"))
        self.assertFalse(
            runtime_quant.is_layer_ignored("layers.0.mlp.experts.gate_proj")
        )

    def test_compressed_w4a8_json_preserves_all_ignore_aliases(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "config.json"), "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "ignore": ["model.layers.{i}.self_attn"],
                            "modules_to_not_convert": ["model.layers.{i}.mlp.gate"],
                            "config_groups": {
                                "group_0": {
                                    "weights": {
                                        "type": "int",
                                        "num_bits": 4,
                                        "strategy": "group",
                                        "group_size": 32,
                                    },
                                    "input_activations": {
                                        "type": "float",
                                        "num_bits": 8,
                                        "dynamic": True,
                                    },
                                }
                            },
                        }
                    },
                    output,
                )

            source_quant = SourceQuantizationConfig.load_from_ckpt(model_path)

        self.assertIsNotNone(source_quant)
        runtime_quant = QuantizationConfig(
            source_quant.get_runtime_method_key(), source_config=source_quant
        )
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.self_attn.q_proj"))
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.mlp.gate"))
        self.assertFalse(
            runtime_quant.is_layer_ignored("layers.0.mlp.experts.gate_proj")
        )


if __name__ == "__main__":
    unittest.main()
