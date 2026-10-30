import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.base import FusedSiluAndMul, SigmoidGateScaleAdd
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import (
    Qwen4MoeLayer,
    _shared_stream_for_device,
)


def make_layer():
    """Real shared FFN/gate/merge; isolate stream ownership from EP routing."""
    torch.manual_seed(600)
    layer = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
    torch.nn.Module.__init__(layer)
    shared = DenseMLP.__new__(DenseMLP)
    torch.nn.Module.__init__(shared)

    def linear(n, k):
        w = (torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.2).to(
            torch.float8_e4m3fn
        )
        return Fp8Linear(w, torch.tensor([0.05], device="cuda"))

    shared.is_gated = True
    shared.up_proj = linear(1280, 2560)
    shared.down_proj = linear(2560, 640)
    shared.act_fn = FusedSiluAndMul()
    shared.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 1)
    layer.shared_expert = shared
    layer.shared_expert_gate = linear(1, 2560)
    layer.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
    layer.use_ep_shared_allreduce = False
    layer.use_unified_tp_allreduce = False
    layer._forward_routed = lambda x: x * 0.7
    layer._shared_stream = torch.cuda.Stream()
    layer._shared_input_ready = torch.cuda.Event()
    layer._shared_output_ready = torch.cuda.Event()
    return layer


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SharedExpertOverlapTest(unittest.TestCase):
    def test_shared_dense_mlp_fuses_activation_and_quant_before_down_gemm(self):
        import flashinfer

        shared = make_layer().shared_expert
        for rows in (4, 8, 12, 16):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)

            def legacy():
                return shared.down_proj(shared.act_fn(shared.up_proj(x)))

            expected = legacy()
            with patch.object(
                flashinfer.activation,
                "silu_and_mul",
                side_effect=AssertionError("separate activation kernel"),
            ):
                actual = shared(x, skip_allreduce=True)
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = shared(x, skip_allreduce=True)
            for gain in (0.01, 0.5, 2.0):
                x.normal_(0, gain)
                expected = legacy()
                graph.replay()
                self.assertTrue(
                    torch.equal(
                        captured.view(torch.uint8), expected.view(torch.uint8)
                    )
                )
            graph.reset()

    def test_dynamic_graph_exact_fused_merge_and_live_weights(self):
        layer = make_layer()
        for rows in (1, 4, 16, 32):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                layer(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = layer(x)
            for gain in (0.01, 0.5, 2.0):
                x.normal_(0, gain)
                layer.shared_expert_gate.compute_weight.mul_(0.9)
                expected = GenericMoeLayer.forward(layer, x)
                graph.replay()
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )
            graph.reset()

    def test_two_graph_keys_do_not_reuse_another_keys_output(self):
        layer = make_layer()
        graphs = []
        for rows in (4, 16):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                layer(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = layer(x)
            graphs.append((graph, x, y))
        for key in (0, 1, 1, 0):
            graph, x, y = graphs[key]
            x.normal_()
            graph.replay()
            self.assertTrue(torch.equal(y, GenericMoeLayer.forward(layer, x)))
        for graph, _, _ in graphs:
            graph.reset()

    def test_eager_temporaries_survive_allocator_reuse(self):
        layer = make_layer()
        for _ in range(12):
            x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
            expected = GenericMoeLayer.forward(layer, x)
            actual = layer(x)
            del x
            with torch.cuda.stream(layer._shared_stream):
                trash = torch.empty((256, 2560), device="cuda", dtype=torch.bfloat16)
                trash.fill_(123)
            self.assertTrue(torch.equal(actual, expected))

    def test_routed_exception_joins_aux_work(self):
        layer = make_layer()
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        with patch.object(layer, "_forward_routed", side_effect=RuntimeError("route")):
            with self.assertRaisesRegex(RuntimeError, "route"):
                layer(x)
        torch.cuda.current_stream().synchronize()
        self.assertTrue(layer._shared_output_ready.query())
        self.assertTrue(torch.equal(layer(x), GenericMoeLayer.forward(layer, x)))

    def test_shared_exception_joins_aux_work(self):
        layer = make_layer()
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        original = layer.shared_expert.forward

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError("shared")

        with patch.object(layer.shared_expert, "forward", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "shared"):
                layer(x)
        torch.cuda.current_stream().synchronize()
        self.assertTrue(layer._shared_output_ready.query())

    def test_reused_stream_multi_layer_two_graph_keys(self):
        layers = [make_layer() for _ in range(3)]
        for layer in layers:
            layer._reused_shared_stream = _shared_stream_for_device(
                torch.cuda.current_device()
            )
        self.assertIs(layers[0]._reused_shared_stream, layers[-1]._reused_shared_stream)
        self.assertIsNot(layers[0]._shared_input_ready, layers[-1]._shared_input_ready)
        graphs = []
        for rows in (1, 4, 16):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)

            def call(reference=False):
                value = x
                for layer in layers:
                    value = (
                        GenericMoeLayer.forward(layer, value)
                        if reference
                        else layer(value)
                    )
                return value

            for _ in range(3):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = call()
            graphs.append((rows, graph, x, actual))
        for generation in range(32):
            rows, graph, x, actual = graphs[generation % 3]
            x.normal_(0, (0.01, 0.2, 2.0)[generation % 3])
            expected = x
            for layer in layers:
                expected = GenericMoeLayer.forward(layer, expected)
            graph.replay()
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            )
        for _, graph, _, _ in graphs:
            graph.reset()

    def test_reused_stream_only_for_measured_rows_and_exception_join(self):
        layer = make_layer()
        layer._reused_shared_stream = _shared_stream_for_device(
            torch.cuda.current_device()
        )
        for rows in (1, 3, 4, 8, 16, 32):
            x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            with patch.object(
                torch.cuda, "stream", wraps=torch.cuda.stream
            ) as use_stream:
                actual = layer(x)
            use_stream.assert_called_once_with(
                layer._reused_shared_stream if rows in (1, 4) else layer._shared_stream
            )
            self.assertTrue(torch.equal(actual, GenericMoeLayer.forward(layer, x)))
        x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
        with patch.object(layer, "_forward_routed", side_effect=RuntimeError("route")):
            with self.assertRaisesRegex(RuntimeError, "route"):
                layer(x)
        torch.cuda.current_stream().synchronize()
        self.assertTrue(layer._shared_output_ready.query())
        self.assertTrue(torch.equal(layer(x), GenericMoeLayer.forward(layer, x)))

    def test_unqualified_inputs_and_disabled_initialization_fall_back(self):
        layer = make_layer()
        sentinel = object()
        inputs = [
            torch.empty(0, 2560, device="cuda", dtype=torch.bfloat16),
            torch.empty(33, 2560, device="cuda", dtype=torch.bfloat16),
            torch.empty(4, 2560, device="cuda", dtype=torch.float32),
            torch.empty(4, 5120, device="cuda", dtype=torch.bfloat16)[:, ::2],
            torch.empty(4, 2560, dtype=torch.bfloat16),
        ]
        with patch.object(
            GenericMoeLayer, "forward", return_value=sentinel
        ) as fallback:
            for x in inputs:
                self.assertIs(layer(x), sentinel)
            self.assertEqual(fallback.call_count, len(inputs))


if __name__ == "__main__":
    unittest.main()
