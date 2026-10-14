"""Check source integration against the diagnostic FP32 arithmetic contract."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe.gate import Gate
from rtp_llm.models_py.modules.dsv4.moe.strategies.nccl_ep_mxfp8 import (
    NcclEpMxfp8Strategy,
)
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Sm120PrecisionContractTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261014)

    def test_router_matches_fp32_output_hook(self):
        weights = {
            W.v4_router_w: torch.randn(256, 4096, device="cuda", dtype=torch.bfloat16)
            * 0.02,
            W.v4_router_bias: torch.randn(256, device="cuda") * 0.01,
        }
        gate = Gate(3, 4096, 256, 8, layer_weights=weights)
        for rows in (0, 1, 3, 8, 127):
            x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
            for fused in ("0", "1"):
                with patch.dict(
                    os.environ,
                    DSV4_GATE_FP32="0",
                    DSV4_GATE_FUSED=fused,
                    DSV4_GATE_OUTPUT_FP32="0",
                ):
                    with patch(
                        "torch.nn.functional.linear",
                        side_effect=lambda a, b: torch.mm(
                            a, b.t(), out_dtype=torch.float32
                        ),
                    ):
                        expected = gate(x)
                with patch.dict(
                    os.environ,
                    DSV4_GATE_FP32="0",
                    DSV4_GATE_FUSED=fused,
                    DSV4_GATE_OUTPUT_FP32="1",
                ):
                    actual = gate(x)
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_lossless_return_matches_ordered_fp32_sum(self):
        strategy = SimpleNamespace(_return_dtype="fp32")
        for world in (2, 4, 8):
            for rows in (0, 1, 7, 31, 257, 513):
                hidden = 4096
                peers = torch.randn(world, rows, hidden, device="cuda")
                peers *= torch.exp2(torch.arange(world, device="cuda") * 3)[
                    :, None, None
                ]
                flat = peers.reshape(world * rows, hidden)
                packed = NcclEpMxfp8Strategy._pack_return(
                    strategy, flat, hidden, hidden // 32, world * rows
                )
                self.assertEqual(packed.data_ptr(), flat.data_ptr())
                actual = NcclEpMxfp8Strategy._combine_return(
                    strategy, packed, rows, hidden, world
                )
                expected = torch.zeros(rows, hidden, device="cuda")
                for peer in peers:
                    expected.add_(peer)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if rows == 31:
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = NcclEpMxfp8Strategy._combine_return(
                            strategy, packed, rows, hidden, world
                        )
                    peers.neg_()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(captured, -expected, rtol=0, atol=0)

    def test_lossless_return_rejects_lossy_payload(self):
        strategy = SimpleNamespace(_return_dtype="fp32")
        received = torch.zeros(8, 132, dtype=torch.uint8, device="cuda")
        with self.assertRaises(ValueError):
            NcclEpMxfp8Strategy._combine_return(strategy, received, 4, 128, 2)


if __name__ == "__main__":
    unittest.main()
