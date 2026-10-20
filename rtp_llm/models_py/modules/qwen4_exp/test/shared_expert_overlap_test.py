import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.base import FusedSiluAndMul, SigmoidGateScaleAdd
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.qwen4_exp.fp8_dense import Qwen4Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.moe.shared_overlap import Qwen4MoeLayer


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
        return Qwen4Fp8Linear(w, torch.tensor([0.05], device="cuda"))

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
        fresh = Qwen4MoeLayer.__new__(Qwen4MoeLayer)
        torch.nn.Module.__init__(fresh)
        with patch.dict(os.environ, QWEN4_SHARED_EXPERT_OVERLAP="0"):
            fresh.prepare_shared_expert_overlap()
        self.assertIsNone(fresh._shared_stream)


if __name__ == "__main__":
    unittest.main()
