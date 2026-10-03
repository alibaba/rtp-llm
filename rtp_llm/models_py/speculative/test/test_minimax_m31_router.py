import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.minimax_m3 import MiniMaxM3DecoderLayer
from rtp_llm.models_py.model_desc.minimax_m31 import (
    MiniMaxM31DecoderLayer,
    MiniMaxM31MoeLayer,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.minimax_m31_router import (
    minimax_m31_router_logits,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMiniMaxM31Router(unittest.TestCase):
    def test_shapes_positions_and_graph(self):
        torch.manual_seed(391)
        for n, k in ((128, 6144),):
            weight = torch.randn(k, n, device="cuda", dtype=torch.float32) * 0.01
            weight = weight.T  # Actual RTP gate physical layout.
            row = torch.randn(1, k, device="cuda", dtype=torch.float32)
            first = minimax_m31_router_logits(row, weight)
            for m in (1, 5, 16, 17, 80, 448):
                for position in sorted({0, m - 1}):
                    x = torch.randn(m, k, device="cuda", dtype=torch.float32)
                    x[position].copy_(row[0])
                    eager = minimax_m31_router_logits(x, weight)
                    self.assertTrue(
                        torch.equal(eager[position], first[0]),
                        f"n={n} k={k} m={m} position={position} max_diff="
                        f"{float((eager[position]-first[0]).abs().max())}",
                    )
                    oracle = x.double() @ weight.double().T
                    torch.testing.assert_close(
                        eager.double(), oracle, atol=2e-5, rtol=2e-5
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = minimax_m31_router_logits(x, weight)
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(3):
                        graph.replay()
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(eager, output))
                    # Refresh fixed-address Graph input, not a stale row test.
                    x[position].mul_(0.5)
                    expected = minimax_m31_router_logits(x, weight)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(expected, output))

    def test_empty_and_invalid(self):
        weight = torch.empty(128, 6144, dtype=torch.float32, device="cuda")
        empty = torch.empty(0, 6144, dtype=torch.float32, device="cuda")
        self.assertEqual(
            tuple(minimax_m31_router_logits(empty, weight).shape), (0, 128)
        )
        with self.assertRaises(ValueError):
            minimax_m31_router_logits(empty.bfloat16(), weight)
        with self.assertRaises(ValueError):
            minimax_m31_router_logits(torch.empty(1, 1, device="cuda"), weight)
        with self.assertRaisesRegex(ValueError, "validated"):
            minimax_m31_router_logits(
                torch.empty(1, 1025, device="cuda"),
                torch.empty(35, 1025, device="cuda"),
            )

    def test_model_local_phase_selection(self):
        layer = object.__new__(MiniMaxM31DecoderLayer)
        nn.Module.__init__(layer)
        layer.mlp = object.__new__(MiniMaxM31MoeLayer)
        nn.Module.__init__(layer.mlp)
        x = torch.zeros(1, 1)
        with patch.object(
            MiniMaxM3DecoderLayer, "_forward_attention", return_value=(x, None)
        ):
            for prefill, verify, expected in (
                (True, False, False),
                (False, False, True),
                (True, True, True),
                (True, False, False),
            ):
                attn = types.SimpleNamespace(
                    is_prefill=prefill, is_target_verify=verify
                )
                layer._forward_attention(x, None, None, None, False, attn)
                self.assertEqual(layer.mlp._batch_invariant_router, expected)

    def test_actual_gate_dispatch_and_prefill_preservation(self):
        mlp = object.__new__(MiniMaxM31MoeLayer)
        nn.Module.__init__(mlp)
        mlp.gate = CudaF16Linear(torch.randn(6144, 128, device="cuda") * 0.01)
        x = torch.randn(5, 6144, device="cuda", dtype=torch.bfloat16)
        mlp._batch_invariant_router = True
        expected = minimax_m31_router_logits(x.float(), mlp.gate.weight)
        self.assertTrue(torch.equal(mlp._compute_router_logits(x), expected))
        mlp._batch_invariant_router = False
        original = mlp.gate(x.float())
        self.assertTrue(torch.equal(mlp._compute_router_logits(x), original))


if __name__ == "__main__":
    unittest.main()
