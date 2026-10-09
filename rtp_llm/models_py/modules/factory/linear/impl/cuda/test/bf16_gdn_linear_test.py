"""Batch-position, graph-replay and fallback coverage for BF16 GDN BA repair."""

import os
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.router_linear import (
    maybe_bf16_gdn_linear,
)


class GdnLinearTest(unittest.TestCase):
    @torch.inference_mode()
    def test_batch_positions_graph_replay_bias_and_strides(self):
        torch.manual_seed(971)
        for strided in (False, True):
            stride = 2 if strided else 1
            weight = torch.randn(
                128, 4096 * stride, device="cuda", dtype=torch.bfloat16
            )[:, ::stride]
            if strided:
                weight = weight.T.contiguous().T
            samples = torch.randn(
                3, 4096 * stride, device="cuda", dtype=torch.bfloat16
            )[:, ::stride]
            for bias in (None, torch.randn(128, device="cuda", dtype=torch.bfloat16)):
                reference = torch.cat(
                    [F.linear(row[None], weight, bias) for row in samples]
                )
                exact = F.linear(
                    samples.double(),
                    weight.double(),
                    None if bias is None else bias.double(),
                )
                torch.testing.assert_close(
                    reference.double(), exact, rtol=0.004, atol=0.002
                )
                for batch in (65, 96, 128, 160, 192, 224, 256):
                    x = torch.randn(
                        batch, 4096 * stride, device="cuda", dtype=torch.bfloat16
                    )[:, ::stride]
                    positions = sorted(set((0, 63, 64, batch - 1)))
                    maybe_bf16_gdn_linear(x, weight, bias)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        y = maybe_bf16_gdn_linear(x, weight, bias)
                    for sample in (0, 1, 2):
                        for position in positions:
                            x[position] = samples[sample]
                        graph.replay()
                        for position in positions:
                            self.assertTrue(
                                torch.equal(y[position], reference[sample]),
                                (strided, bias is not None, batch, sample, position),
                            )

    def test_fallback_and_option_snapshot(self):
        w = torch.randn(128, 4096, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}):
            op = CudaF16Linear(w.T)
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
            for rows in (0, 1, 32, 64, 257):
                x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
                self.assertIsNone(maybe_bf16_gdn_linear(x, w))
                self.assertTrue(torch.equal(op(x), F.linear(x, w)))
            x = torch.randn(96, 4096, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(torch.equal(op(x), maybe_bf16_gdn_linear(x, w)))
            disabled = CudaF16Linear(w.T)
            self.assertTrue(torch.equal(disabled(x), F.linear(x, w)))
        self.assertIsNone(maybe_bf16_gdn_linear(x.cpu(), w.cpu()))
        self.assertIsNone(maybe_bf16_gdn_linear(x.float(), w.float()))
        self.assertIsNone(maybe_bf16_gdn_linear(x, w[:64]))
        self.assertIsNone(maybe_bf16_gdn_linear(x[:, :2048], w[:, :2048]))


if __name__ == "__main__":
    unittest.main()
