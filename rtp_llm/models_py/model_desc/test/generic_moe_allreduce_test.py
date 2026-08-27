"""Contract tests for GenericMoeLayer's unified TP all-reduce paths.

Mostly CPU tensors; the side-stream shared expert needs a real CUDA graph.
"""

import inspect
from functools import partial
from types import SimpleNamespace
from unittest import TestCase, main, skipUnless
from unittest.mock import MagicMock, Mock, patch

import torch

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.model_desc.generic_moe import (
    _SHARED_EXPERT_STREAMS,
    GenericMoeLayer,
    _ensure_shared_expert_stream,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.utils.model_weight import W

# This file also runs as generic_moe_allreduce_test_rocm. Only the CUDA
# low-latency router advertises supports_row_scatter_finalize, so the side
# stream never runs on ROCm.
_CUDA_GRAPH_READY = torch.cuda.is_available() and torch.version.hip is None


def _make_layer(
    *,
    supports_skip_tp_allreduce=True,
    supports_row_scatter_finalize=False,
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
        topk_ids_dtype=torch.int32,
        router=SimpleNamespace(
            supports_skip_tp_allreduce=supports_skip_tp_allreduce,
            supports_row_scatter_finalize=supports_row_scatter_finalize,
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


class GenericMoeEpUnifiedAllreduceTest(TestCase):
    def test_ep_unified_decision_covers_all_predicate_terms(self):
        cases = (
            ("ep_router_supports_row_scatter", True, 2, 2, 2, 2, True),
            ("router_lacks_support", False, 2, 2, 2, 2, False),
            ("pure_tp_is_not_ep", True, 2, 1, 2, 2, False),
            ("ffn_tp_one", True, 1, 2, 1, 2, False),
            ("no_shared_expert", True, 2, 2, 2, 1, False),
            ("router_tp_size_mismatch", True, 4, 2, 2, 2, False),
        )
        for name, supports, ffn_tp, ep_size, attn_tp, moe_style, expected in cases:
            with self.subTest(name=name):
                layer = _make_layer(
                    supports_row_scatter_finalize=supports,
                    ffn_tp_size=ffn_tp,
                    attn_tp_size=attn_tp,
                    ep_size=ep_size,
                    moe_style=moe_style,
                )
                self.assertEqual(layer.use_ep_unified_allreduce, expected)

    def test_ep_unified_displaces_the_separate_shared_reduce(self):
        fused = _make_layer(ep_size=2, supports_row_scatter_finalize=True)
        self.assertTrue(fused.use_ep_unified_allreduce)
        self.assertFalse(fused.use_ep_shared_allreduce)

        legacy = _make_layer(ep_size=2, supports_row_scatter_finalize=False)
        self.assertFalse(legacy.use_ep_unified_allreduce)
        self.assertTrue(legacy.use_ep_shared_allreduce)

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_row_scatter_reduces_both_branches_once(self, mock_all_reduce):
        layer = _make_layer(ep_size=2, supports_row_scatter_finalize=True)
        hidden_states, routed_output, shared_output, gate_output, fused_moe = (
            _configure_forward(layer, gate_enabled=True)
        )
        # Stand in for the router: scatter-add the routed slice into the buffer
        # the layer supplied, then hand that same buffer back.
        fused_moe.side_effect = lambda **kwargs: kwargs["row_scatter_target"].add_(
            routed_output
        )
        expected_input = torch.sigmoid(gate_output) * shared_output + routed_output
        mock_all_reduce.side_effect = lambda tensor, group, inplace: tensor * 2

        result = layer(hidden_states)

        mock_all_reduce.assert_called_once()
        self.assertIs(mock_all_reduce.call_args.kwargs["group"], Group.TP)
        self.assertTrue(mock_all_reduce.call_args.kwargs["inplace"])
        reduce_input = mock_all_reduce.call_args.args[0]
        self.assertIs(reduce_input, fused_moe.call_args.kwargs["row_scatter_target"])
        torch.testing.assert_close(reduce_input, expected_input)
        torch.testing.assert_close(result, expected_input * 2)
        self.assertTrue(layer.shared_expert.call_args.kwargs["skip_allreduce"])
        self.assertNotIn("skip_tp_allreduce", fused_moe.call_args.kwargs)
        # CPU weights mean no side stream, so the router needs no join.
        self.assertIsNone(fused_moe.call_args.kwargs["row_scatter_ready"])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_a_failed_routed_chain_joins_the_shared_branch(self, _):
        # The router does the join on the success path. When the routed chain
        # raises, forward has to do it, or the side stream stays outstanding.
        layer = _make_layer(ep_size=2, supports_row_scatter_finalize=True)
        hidden_states, _, _, _, fused_moe = _configure_forward(layer)
        ready = object()
        layer._shared_expert_partial = Mock(
            return_value=(torch.zeros_like(hidden_states), ready)
        )
        fused_moe.side_effect = RuntimeError("dispatch failed")
        waited = []
        stream = SimpleNamespace(wait_event=waited.append)

        with patch("torch.cuda.current_stream", return_value=stream):
            with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                layer(hidden_states)

        self.assertEqual(waited, [ready])

    @patch("rtp_llm.models_py.model_desc.generic_moe.all_reduce")
    def test_layer_only_passes_keywords_fused_moe_accepts(self, _):
        # The fused_moe mock accepts any keyword, so a layer-only keyword passes
        # every other test here and only fails once a real FusedMoe is called.
        accepted = set(inspect.signature(FusedMoe.forward).parameters)
        cases = (
            ("ep_unified", dict(ep_size=2, supports_row_scatter_finalize=True)),
            ("ep_shared", dict(ep_size=2, supports_row_scatter_finalize=False)),
            ("pure_tp", dict(ep_size=1)),
            ("ffn_tp_one", dict(ffn_tp_size=1)),
        )
        for name, kwargs in cases:
            with self.subTest(name=name):
                layer = _make_layer(**kwargs)
                hidden_states, _, _, _, fused_moe = _configure_forward(layer)

                layer(hidden_states)

                unexpected = set(fused_moe.call_args.kwargs) - accepted
                self.assertFalse(
                    unexpected, f"not FusedMoe.forward parameters: {unexpected}"
                )


@skipUnless(_CUDA_GRAPH_READY, "needs a CUDA device")
class SharedExpertSideStreamCaptureTest(TestCase):
    """Covers _shared_expert_partial's side-stream branch under real capture.

    CPU weights leave shared_expert_stream unset, so only a CUDA device reaches
    this branch, and an unclosed fork only shows up at capture end.
    """

    SHARED_FACTOR = 3.0

    def setUp(self):
        _SHARED_EXPERT_STREAMS.clear()
        self.device = torch.device("cuda", torch.cuda.current_device())

    def _stub_layer(self):
        """Stand-in for ``self``: this path only needs these four attributes."""
        stream = _ensure_shared_expert_stream(self.device)
        self.assertIsNotNone(stream)
        stub = SimpleNamespace(
            # A row-parallel shared expert leaves a TP-partial sum; scaling
            # stands in for that work without needing real weights.
            shared_expert=lambda x, skip_allreduce: x * self.SHARED_FACTOR,
            shared_expert_gate=None,
            shared_expert_stream=stream,
            shared_expert_ready=torch.cuda.Event(),
        )
        stub._gate_shared_expert_output = partial(
            GenericMoeLayer._gate_shared_expert_output, stub
        )
        return stub

    def _fork_and_join(self, stub, hidden):
        shared_partial, ready = GenericMoeLayer._shared_expert_partial(stub, hidden)
        self.assertIs(ready, stub.shared_expert_ready)
        # The join the router performs in _finalize_row_scatter.
        torch.cuda.current_stream().wait_event(ready)
        return shared_partial

    def test_one_stream_per_device_is_reused(self):
        # Every MoE layer shares one side stream per device.
        self.assertIs(
            _ensure_shared_expert_stream(self.device),
            _ensure_shared_expert_stream(self.device),
        )

    def test_capture_closes_the_fork_and_replay_recomputes(self):
        stub = self._stub_layer()
        hidden = torch.ones(8, 16, device=self.device)
        out = torch.empty_like(hidden)

        # torch.cuda.graph requires the work to have run once outside capture.
        out.copy_(self._fork_and_join(stub, hidden))
        torch.cuda.synchronize()

        # Capture fails here if the side stream is still outstanding.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out.copy_(self._fork_and_join(stub, hidden))

        # A new input value distinguishes a real replay from leftover contents.
        hidden.fill_(2.0)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, torch.full_like(out, 2.0 * self.SHARED_FACTOR))

    def test_graphs_per_batch_size_share_the_stream_and_event(self):
        # Decode captures one graph per batch size into a single memory pool
        # (cuda_graph_runner.cc passes one shared_graph_pool_ to every capture),
        # and every graph reuses the device's side stream and the layer's event.
        stub = self._stub_layer()
        pool = torch.cuda.graph_pool_handle()
        captured = []
        forked_on = set()
        for num_tokens in (4, 8):
            hidden = torch.ones(num_tokens, 16, device=self.device)
            out = torch.empty_like(hidden)
            out.copy_(self._fork_and_join(stub, hidden))
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                out.copy_(self._fork_and_join(stub, hidden))
            forked_on.add((id(stub.shared_expert_stream), id(stub.shared_expert_ready)))
            captured.append((graph, hidden, out))

        self.assertEqual(len(forked_on), 1, "captures used different stream/event")

        for index, (graph, hidden, _) in enumerate(captured):
            hidden.fill_(index + 2.0)
            graph.replay()
        torch.cuda.synchronize()

        for index, (_, _, out) in enumerate(captured):
            torch.testing.assert_close(
                out, torch.full_like(out, (index + 2.0) * self.SHARED_FACTOR)
            )


if __name__ == "__main__":
    main()
