"""GLM shared branch overlap: graph replay, row reorder and stream lifetime."""

import unittest

import torch
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from torch import nn


class _Shared(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, x, x_fp8=None, x_scale=None):
        # Exercise optional upstream resources on the auxiliary stream as well.
        value = x if x_fp8 is None else x_fp8.to(x.dtype) * x_scale
        return torch.mm(value, self.weight)


class _Routed(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward_prepacked(self, x):
        return torch.mm(x, self.weight)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53SharedOverlapTest(unittest.TestCase):
    def build_layer(self):
        torch.manual_seed(9360)
        layer = GenericMoeLayer.__new__(GenericMoeLayer)
        nn.Module.__init__(layer)
        weights = [
            torch.randn(512, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)
        ]
        layer.shared_expert = _Shared(weights[0])
        layer.fused_moe = _Routed(weights[1])
        layer.fake_balance_expert = None
        layer._use_mega_moe_fused_shared = False
        layer._shared_expert_stream = torch.cuda.Stream()
        return layer

    def test_prepacked_graph_reorder_and_fallback(self):
        layer = self.build_layer()
        for batch in (1, 7, 48, 64, 65):
            x = torch.randn(batch, 512, device="cuda", dtype=torch.bfloat16)
            expected = layer.fused_moe.forward_prepacked(x) + layer.shared_expert(x)
            actual = layer.forward_prepacked(x, None, None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if batch == 65:
                self.assertIsNone(layer._start_shared_overlap(x))
        x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                layer.forward_prepacked(x, None, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            actual = layer.forward_prepacked(x, None, None)
        for _ in range(8):
            x.copy_(x.flip(0))
            x[3].zero_()
            graph.replay()
            expected = layer.fused_moe.forward_prepacked(x) + layer.shared_expert(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_optional_inputs_and_allocator_lifetime(self):
        layer = self.build_layer()
        for _ in range(8):
            x = torch.randn(48, 512, device="cuda", dtype=torch.bfloat16)
            quant = x.to(torch.float8_e4m3fn)
            scale = torch.full_like(x, 0.5)
            expected = layer.shared_expert(x, quant, scale)
            pending = layer._start_shared_overlap(x, quant, scale)
            del quant, scale
            # Reuse the creating stream's allocator before joining the aux stream.
            scratch = [torch.empty_like(x).fill_(3) for _ in range(8)]
            actual = layer._finish_shared_overlap(pending)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del scratch


if __name__ == "__main__":
    unittest.main()
