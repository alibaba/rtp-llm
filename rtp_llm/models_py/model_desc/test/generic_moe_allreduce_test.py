"""CPU contract tests for GenericMoeLayer's unified TP all-reduce path."""

from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    FusedMoe,
    SHARED_EXPERT_ACCUMULATE_ARG,
)
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
    # MagicMock auto-creates truthy values for every spec attribute; pin the
    # new fold capability off so legacy-path tests keep the historic merge.
    fused_moe.supports_shared_expert_accumulate = False
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
    def test_pure_tp_gate_is_merged_in_place_before_reduce(self, mock_all_reduce):
        layer = _make_layer()
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        expected_input = routed_output.clone()
        expected_input.add_(torch.sigmoid(gate_output) * shared_output)
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(reduce_input, routed_output)
        self.assertIs(layer.shared_expert_gate.call_args.args[0], hidden_states)
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])

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


class GenericMoeSharedExpertFoldTest(TestCase):
    """Contract for folding the shared-expert add into the MoE combine kernel.

    When the strategy advertises ``supports_shared_expert_accumulate`` and the
    merge is a plain add of aligned partials, the layer must hand the shared
    output to the MoE (which folds it into its combine kernel, e.g. the
    ep_gather accumulate) instead of issuing the separate bf16 add. Every
    other configuration must keep the historic merge.
    """

    def _fold_mock(self, layer, routed_output, shared_output):
        """Make the mocked FusedMoe simulate an executor-side fold."""

        fused_moe = layer.fused_moe

        def fold_and_combine(**kwargs):
            extra = kwargs.get("extra_expert_args")
            if not extra or SHARED_EXPERT_ACCUMULATE_ARG not in extra:
                return routed_output
            return routed_output + extra[SHARED_EXPERT_ACCUMULATE_ARG]

        fused_moe.side_effect = fold_and_combine

    def test_tp_one_folds_shared_into_moe_combine(self):
        layer = _make_layer(ffn_tp_size=1, attn_tp_size=1)
        hidden_states, routed_output, shared_output, _, fused_moe = (
            _configure_forward(layer)
        )
        fused_moe.supports_shared_expert_accumulate = True
        self._fold_mock(layer, routed_output, shared_output)

        order = Mock()
        order.attach_mock(layer.shared_expert, "shared_expert")
        order.attach_mock(fused_moe, "fused_moe")

        result = layer(hidden_states)

        torch.testing.assert_close(result, routed_output + shared_output)
        extra = fused_moe.call_args.kwargs.get("extra_expert_args")
        self.assertIsNotNone(extra)
        self.assertIs(extra[SHARED_EXPERT_ACCUMULATE_ARG], shared_output)
        self.assertFalse(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertFalse(layer.shared_expert.call_args.kwargs["skip_allreduce"])
        # The fold needs the shared output before the routed combine runs.
        names = [name for name, _, _ in order.mock_calls]
        self.assertLess(names.index("shared_expert"), names.index("fused_moe"))

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_unified_tp_folds_before_single_allreduce(self, mock_all_reduce):
        layer = _make_layer(ffn_tp_size=2, attn_tp_size=2)
        hidden_states, routed_output, shared_output, _, fused_moe = (
            _configure_forward(layer)
        )
        fused_moe.supports_shared_expert_accumulate = True
        self._fold_mock(layer, routed_output, shared_output)
        folded = routed_output + shared_output
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        mock_all_reduce.assert_called_once()
        reduce_input = mock_all_reduce.call_args.args[0]
        torch.testing.assert_close(reduce_input, folded)
        torch.testing.assert_close(result, folded * 2)
        extra = fused_moe.call_args.kwargs.get("extra_expert_args")
        self.assertIs(extra[SHARED_EXPERT_ACCUMULATE_ARG], shared_output)
        self.assertTrue(fused_moe.call_args.kwargs["skip_tp_allreduce"])
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])

    def test_no_fold_without_strategy_support(self):
        layer = _make_layer(ffn_tp_size=1, attn_tp_size=1)
        hidden_states, routed_output, shared_output, _, fused_moe = (
            _configure_forward(layer)
        )
        fused_moe.supports_shared_expert_accumulate = False

        result = layer(hidden_states)

        self.assertIsNone(fused_moe.call_args.kwargs.get("extra_expert_args"))
        torch.testing.assert_close(result, routed_output + shared_output)

    def test_no_fold_with_shared_expert_gate(self):
        layer = _make_layer(ffn_tp_size=1, attn_tp_size=1)
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        fused_moe.supports_shared_expert_accumulate = True
        expected = routed_output.clone()
        expected.add_(torch.sigmoid(gate_output) * shared_output)

        result = layer(hidden_states)

        self.assertIsNone(fused_moe.call_args.kwargs.get("extra_expert_args"))
        torch.testing.assert_close(result, expected)

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_no_fold_in_ep_mode(self, mock_all_reduce):
        layer = _make_layer(ffn_tp_size=2, attn_tp_size=2, ep_size=2)
        hidden_states, routed_output, shared_output, _, fused_moe = (
            _configure_forward(layer)
        )
        fused_moe.supports_shared_expert_accumulate = True
        mock_all_reduce.side_effect = lambda tensor, group: tensor * 2

        result = layer(hidden_states)

        self.assertIsNone(fused_moe.call_args.kwargs.get("extra_expert_args"))
        torch.testing.assert_close(result, routed_output + shared_output * 2)

    def test_no_fold_when_router_tp_size_mismatches_ffn_tp(self):
        # TP > 1 without the unified path: the routed and shared partials are
        # reduced separately before the merge, so the fold must stay off to
        # keep the reduction order (and the numerics) unchanged.
        layer = _make_layer(ffn_tp_size=2, attn_tp_size=4)
        hidden_states, routed_output, shared_output, _, fused_moe = (
            _configure_forward(layer)
        )
        fused_moe.supports_shared_expert_accumulate = True

        result = layer(hidden_states)

        self.assertIsNone(fused_moe.call_args.kwargs.get("extra_expert_args"))
        torch.testing.assert_close(result, routed_output + shared_output)


if __name__ == "__main__":
    main()
