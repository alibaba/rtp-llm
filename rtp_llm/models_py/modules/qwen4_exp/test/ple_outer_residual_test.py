import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.ple.ple_conv_triton import try_ple_conv
from rtp_llm.models_py.modules.qwen4_exp.test import ple_gate_test


class PLEOuterResidualTest(unittest.TestCase):
    def setUp(self):
        # fail-loud rather than skip: this target is pinned to RTX_5000_PRO_CU13.
        self.assertTrue(torch.cuda.is_available(), "GPU test must run on SM120")
        self.assertEqual(
            torch.cuda.get_device_capability(),
            (12, 0),
            "target pool is RTX_5000_PRO_CU13 (SM120)",
        )
        torch.manual_seed(20261019)

    def test_conv_preserves_bf16_output_boundary_and_input(self):
        for batch, length in ((1, 1), (3, 4), (1, 127), (1, 129), (1, 4093), (1, 4096)):
            shape = (batch, length, 10240)
            gated, normed, residual = [
                torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            weight = torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1
            for initial in (
                None,
                torch.randn(batch, 9, 10240, device="cuda", dtype=torch.bfloat16),
            ):
                before = residual.clone()
                expected = residual + try_ple_conv(gated, normed, initial, weight, 3)
                actual = try_ple_conv(
                    gated, normed, initial, weight, 3, residual=residual
                )
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                self.assertTrue(torch.equal(residual, before))

    def test_complete_prefill_prefix_and_every_page_snapshot(self):
        layer = ple_gate_test.PleGateTest().layer()
        for length in (1, 7, 128, 129, 257, 4093, 4096, 12288):
            hidden = torch.randn(1, length, 10240, device="cuda", dtype=torch.bfloat16)
            embedding = torch.randn(
                1, length, 2560, device="cuda", dtype=torch.bfloat16
            )
            tokens = torch.zeros(1, length + 2, device="cuda", dtype=torch.int64)
            for initial in (
                None,
                torch.randn(1, 9, 10240, device="cuda", dtype=torch.bfloat16),
            ):
                output, states = layer.prefill_page_snapshots(
                    hidden, tokens, 128, initial, embeddings=embedding
                )
                actual, snapshots = layer.prefill_page_snapshots(
                    hidden,
                    tokens,
                    128,
                    initial,
                    embeddings=embedding,
                    add_residual=True,
                )
                self.assertTrue(torch.equal(actual, hidden + output))
                self.assertEqual(len(states), len(snapshots))
                for a, b in zip(states, snapshots):
                    self.assertTrue(torch.equal(a, b))

    def test_native_conv_fallback_adds_residual_once(self):
        layer = ple_gate_test.PleGateTest().layer()
        hidden = torch.randn(1, 257, 10240, device="cuda", dtype=torch.bfloat16)
        embedding = torch.randn(1, 257, 2560, device="cuda", dtype=torch.bfloat16)
        tokens = torch.zeros(1, 259, device="cuda", dtype=torch.int64)
        with patch(
            "rtp_llm.models_py.modules.qwen4_exp.ple.ple_conv_triton.try_ple_conv",
            return_value=None,
        ):
            output, states = layer.prefill_page_snapshots(
                hidden, tokens, 128, embeddings=embedding
            )
            actual, snapshots = layer.prefill_page_snapshots(
                hidden, tokens, 128, embeddings=embedding, add_residual=True
            )
        self.assertTrue(torch.equal(actual, hidden + output))
        for a, b in zip(states, snapshots):
            self.assertTrue(torch.equal(a, b))

    def test_graph_replay_refreshes_outer_residual(self):
        shape = (1, 129, 10240)
        gated, normed, residual = [
            torch.randn(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        weight = torch.randn(10240, 1, 4, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(3):
            try_ple_conv(gated, normed, None, weight, 3, residual=residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = try_ple_conv(gated, normed, None, weight, 3, residual=residual)
        for scale in (0.01, 1.0, 100.0):
            residual.normal_().mul_(scale)
            graph.replay()
            expected = residual + try_ple_conv(gated, normed, None, weight, 3)
            self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
