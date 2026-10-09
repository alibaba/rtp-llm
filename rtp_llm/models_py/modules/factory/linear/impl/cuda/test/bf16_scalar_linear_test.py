"""Correctness and graph coverage for batch-invariant shared-expert gates."""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
from rtp_llm.models_py.triton_kernels.common.scalar_linear import (
    maybe_bf16_scalar_linear,
)


class ScalarLinearTest(unittest.TestCase):
    def test_batch_invariance_graph_and_strides(self):
        torch.manual_seed(917)
        for width in (127, 4096, 8192):
            weight = torch.randn(1, width * 2, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
            rows = torch.randn(9, width * 2, device="cuda", dtype=torch.bfloat16)[
                :, ::2
            ]
            for bias in (
                None,
                torch.tensor([0.125], device="cuda", dtype=torch.bfloat16),
            ):
                ref = maybe_bf16_scalar_linear(rows, weight, bias)
                exact = F.linear(
                    rows.double(),
                    weight.double(),
                    None if bias is None else bias.double(),
                )
                torch.testing.assert_close(ref.double(), exact, rtol=0.004, atol=2e-5)
                for batch in (1, 2, 8, 32, 128, 512):
                    x = torch.randn(
                        batch, width * 2, device="cuda", dtype=torch.bfloat16
                    )[:, ::2]
                    target = batch // 2
                    x[target] = rows[0]
                    maybe_bf16_scalar_linear(x, weight, bias)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        y = maybe_bf16_scalar_linear(x, weight, bias)
                    for row in (0, 5, 8):
                        x[target] = rows[row]
                        graph.replay()
                        self.assertTrue(torch.equal(y[target], ref[row]))

    def test_large_prefill_keeps_original_projection(self):
        x = torch.randn(257, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(4096, 1, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}):
            op = CudaF16Linear(w)
        self.assertTrue(torch.equal(op(x), F.linear(x, w.T)))

    def test_empty_and_fallback(self):
        w = torch.ones(1, 4096, device="cuda", dtype=torch.bfloat16)
        self.assertEqual(maybe_bf16_scalar_linear(w[:0], w).shape, (0, 1))
        self.assertIsNone(maybe_bf16_scalar_linear(w.cpu(), w.cpu()))
        self.assertIsNone(maybe_bf16_scalar_linear(w.float(), w.float()))
        self.assertIsNone(maybe_bf16_scalar_linear(w, w.repeat(2, 1)))
        large = torch.ones(1, 8193, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(maybe_bf16_scalar_linear(large, large))

    def test_option_is_snapshotted_and_other_projections_unchanged(self):
        x = torch.randn(8, 4096, device="cuda", dtype=torch.bfloat16)
        for outputs in (1, 64):
            w = torch.randn(4096, outputs, device="cuda", dtype=torch.bfloat16)
            with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "1"}):
                op = CudaF16Linear(w)
            with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
                expected = (
                    maybe_bf16_scalar_linear(x, w.T)
                    if outputs == 1
                    else F.linear(x, w.T)
                )
                self.assertTrue(torch.equal(op(x), expected))
        with patch.dict(os.environ, {"RTP_BF16_GATE_KERNEL": "0"}):
            op = CudaF16Linear(w[:, :1])
        self.assertTrue(torch.equal(op(x), F.linear(x, w[:, :1].T)))


if __name__ == "__main__":
    unittest.main()
