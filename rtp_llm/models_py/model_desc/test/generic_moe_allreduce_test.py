"""GenericMoeLayer TP contracts and CUDA BF16 shared-expert rounding regression."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeLayer,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.utils.model_weight import W


def _make_layer(
    *,
    supports_skip_tp_allreduce=True,
    ffn_tp_size=2,
    attn_tp_size=2,
    ep_size=1,
    moe_style=2,
    with_shared_expert_gate=False,
):
    config = SimpleNamespace(
        hidden_size=8,
        inter_size=16,
        expert_num=4,
        moe_k=2,
        quant_config=None,
        activation_type="SiGLU",
        moe_style=moe_style,
        eplb_config=SimpleNamespace(phy_exp_num=lambda count: count),
    )
    parallelism_config = SimpleNamespace(
        ep_size=ep_size,
        dp_rank=0,
        dp_size=1,
        get_ffn_tp_size=lambda: ffn_tp_size,
        get_attn_tp_size=lambda: attn_tp_size,
    )
    moe_config = SimpleNamespace(fake_balance_expert=False)
    weights = {
        W.moe_w1: torch.empty(4, 2, 8),
        W.moe_w2: torch.empty(4, 8, 2),
    }
    if with_shared_expert_gate:
        weights[W.shared_expert_gate] = torch.empty(8, 1)
    fused_moe = SimpleNamespace(
        includes_shared_expert=False,
        topk_ids_dtype=torch.int32,
        router=SimpleNamespace(
            supports_skip_tp_allreduce=supports_skip_tp_allreduce,
            tp_collective_size=attn_tp_size,
        ),
    )

    with (
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.LinearFactory.create_linear_from_weights",
            return_value=Mock(),
        ),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.SelectTopk",
            return_value=Mock(),
        ),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.DenseMLP",
            return_value=Mock(),
        ),
        patch("rtp_llm.models_py.model_desc.generic_moe.MoEConfigAdapter"),
        patch(
            "rtp_llm.models_py.model_desc.generic_moe.FusedMoeFactory"
        ) as fused_moe_factory,
    ):
        fused_moe_factory.return_value.create_fused_moe.return_value = fused_moe
        return GenericMoeLayer(config, parallelism_config, weights, moe_config)


def _configure_forward(layer, *, gate_enabled=False):
    hidden_states = torch.randn(4, 8)
    routed_output = torch.randn_like(hidden_states)
    shared_output = torch.randn_like(hidden_states)
    gate_output = torch.full((4, 1), 2.0)

    layer.gate = Mock(return_value=torch.zeros(4, 4))
    layer.select_topk = Mock(
        side_effect=lambda logits, topk_ids, topk_weights: (
            topk_ids.zero_(),
            topk_weights.fill_(0.5),
        )
    )
    fused_moe = MagicMock(spec=FusedMoe, return_value=routed_output)
    fused_moe.topk_ids_dtype = torch.int32
    layer.fused_moe = fused_moe
    layer.shared_expert = MagicMock(spec=DenseMLP, return_value=shared_output)
    if gate_enabled:
        layer.shared_expert_gate = Mock(return_value=gate_output)
        layer.sigmoid_gate_scale_add = Mock(
            side_effect=lambda gate, shared, output: output.add_(
                torch.sigmoid(gate) * shared
            )
        )
    else:
        layer.shared_expert_gate = None
        layer.sigmoid_gate_scale_add = None
    layer.correction_bias = None
    return hidden_states, routed_output, shared_output, gate_output, fused_moe


class GenericMoeInitializationTest(TestCase):
    def test_unified_decision_covers_all_predicate_terms(self):
        cases = (
            ("pure_tp_shared_supported", True, 2, 1, 2, True),
            ("ffn_tp_one", True, 1, 1, 2, False),
            ("ep_mode", True, 2, 2, 2, False),
            ("no_shared_expert", True, 2, 1, 1, False),
            ("unsupported_router", False, 2, 1, 2, False),
        )
        for name, supports, ffn_tp, ep_size, moe_style, expected in cases:
            with self.subTest(name=name):
                layer = _make_layer(
                    supports_skip_tp_allreduce=supports,
                    ffn_tp_size=ffn_tp,
                    ep_size=ep_size,
                    moe_style=moe_style,
                )
                self.assertEqual(layer.use_unified_tp_allreduce, expected)

    def test_router_tp_size_is_part_of_the_constructor_contract(self):
        self.assertFalse(
            _make_layer(ffn_tp_size=4, attn_tp_size=2).use_unified_tp_allreduce
        )
        self.assertFalse(
            _make_layer(ffn_tp_size=2, attn_tp_size=4).use_unified_tp_allreduce
        )
        self.assertTrue(
            _make_layer(ffn_tp_size=2, attn_tp_size=2).use_unified_tp_allreduce
        )

    def test_shared_expert_gate_is_assembled_by_init(self):
        layer = _make_layer(with_shared_expert_gate=True)
        self.assertIsNotNone(layer.shared_expert_gate)
        self.assertIsNotNone(layer.sigmoid_gate_scale_add)
        self.assertTrue(layer.use_unified_tp_allreduce)


class GenericMoeUnifiedAllreduceTest(TestCase):
    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_pure_tp_combines_partial_outputs_before_reduce(self, mock_all_reduce):
        layer = _make_layer()
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )
        expected_input = routed_output + shared_output
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        mock_all_reduce.assert_called_once()
        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(mock_all_reduce.call_args.kwargs["group"], Group.TP)
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_pure_tp_gate_is_combined_before_reduce(self, mock_all_reduce):
        layer = _make_layer()
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        expected_input = routed_output + torch.sigmoid(gate_output) * shared_output
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(layer.shared_expert_gate.call_args.args[0], hidden_states)
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_cuda_bf16_gate_rounds_before_single_tp_reduce(self, mock_all_reduce):
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA BF16 kernel regression; CPU contracts run separately")
        # Qwen3.5-397B hidden width and one-token / gamma4 verify widths.
        # sigmoid(1) rounded to BF16 is exactly 0.73046875. With shared=1
        # and routed=-0.73046875 the separately materialized BF16 sum is zero.
        # Keeping sigmoid/mul/add in FP32 until the final cast is nonzero.
        for tokens in (1, 5):
            with self.subTest(tokens=tokens):
                mock_all_reduce.reset_mock()
                layer = _make_layer(ffn_tp_size=4, attn_tp_size=4, ep_size=1)
                hidden = torch.zeros(tokens, 4096, device="cuda", dtype=torch.bfloat16)
                routed = torch.full_like(hidden, -0.73046875)
                shared = torch.ones_like(hidden)
                gate = torch.ones(tokens, 1, device="cuda", dtype=torch.bfloat16)
                layer.gate = Mock(return_value=torch.zeros(tokens, 4, device="cuda"))
                layer.select_topk = Mock(
                    side_effect=lambda logits, ids, weights: (
                        ids.zero_(),
                        weights.fill_(0.5),
                    )
                )
                fused_moe = MagicMock(spec=FusedMoe, return_value=routed)
                fused_moe.topk_ids_dtype = torch.int32
                layer.fused_moe = fused_moe
                layer.shared_expert = MagicMock(spec=DenseMLP, return_value=shared)
                layer.shared_expert_gate = Mock(return_value=gate)
                layer.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
                layer.correction_bias = None
                expected = torch.zeros_like(hidden)
                # Exercise the real old CUDA kernel as a discriminating control,
                # rather than approximating that backend with a mock merge.
                fused_result = layer.sigmoid_gate_scale_add(
                    gate, shared, routed.clone()
                )
                self.assertTrue(torch.isfinite(fused_result).all().item())
                self.assertGreater(torch.count_nonzero(fused_result).item(), 0)
                self.assertTrue(
                    torch.equal(torch.sigmoid(gate), torch.full_like(gate, 0.73046875))
                )
                # Inspect the actual local tensor entering the one collective;
                # a real multi-rank NCCL/model acceptance run is a separate test.
                mock_all_reduce.side_effect = lambda tensor, group: tensor.clone()
                result = layer(hidden)
                mock_all_reduce.assert_called_once()
                self.assertIs(mock_all_reduce.call_args.kwargs["group"], Group.TP)
                self.assertTrue(
                    torch.equal(mock_all_reduce.call_args.args[0], expected)
                )
                self.assertTrue(torch.equal(result, expected))
                self.assertEqual(result.dtype, torch.bfloat16)
                self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])
                self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])
                print(
                    f"shared_expert_bf16 tokens={tokens} hidden=4096 "
                    f"old_fused_nonzero={torch.count_nonzero(fused_result).item()} "
                    f"old_fused_max_abs={fused_result.float().abs().max().item()} "
                    "materialized_bf16_exact_zero=True",
                    flush=True,
                )

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_ep_reduces_shared_output_only(self, mock_all_reduce):
        layer = _make_layer(ep_size=2)
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        torch.testing.assert_close(reduce_input, shared_output)
        torch.testing.assert_close(result, routed_output + shared_output * 2)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_ep_gate_is_applied_before_shared_reduce(self, mock_all_reduce):
        layer = _make_layer(ep_size=2)
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        expected_shared = torch.sigmoid(gate_output) * shared_output
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIsNot(reduce_input, routed_output)
        torch.testing.assert_close(reduce_input, expected_shared)
        torch.testing.assert_close(result, routed_output + expected_shared * 2)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_ffn_tp_one_does_not_add_collective(self, mock_all_reduce):
        layer = _make_layer(ffn_tp_size=1)
        hidden_states, routed_output, shared_output, _, fused_moe = _configure_forward(
            layer
        )

        result = layer(hidden_states)

        mock_all_reduce.assert_not_called()
        torch.testing.assert_close(result, routed_output + shared_output)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertFalse(layer.shared_expert.call_args.kwargs["skip_allreduce"])


if __name__ == "__main__":
    main()
