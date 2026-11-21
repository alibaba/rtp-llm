import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_ple.conv import try_ple_conv
from rtp_llm.models_py.modules.qwen4_exp.test.support.ple_gate import (
    PLEProjectionFixture,
)


class PLEOuterResidualTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU test must run on SM90/SM120")
        self.assertIn(torch.cuda.get_device_capability(), ((9, 0), (12, 0)))
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
        layer = PLEProjectionFixture().layer()
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
        layer = PLEProjectionFixture().layer()
        hidden = torch.randn(1, 257, 10240, device="cuda", dtype=torch.bfloat16)
        embedding = torch.randn(1, 257, 2560, device="cuda", dtype=torch.bfloat16)
        tokens = torch.zeros(1, 259, device="cuda", dtype=torch.int64)
        with patch(
            "rtp_llm.models_py.kernels.cuda.qwen4_ple.conv.try_ple_conv",
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

    def test_decode_chunk_and_short_prefill_add_residual(self):
        layer = PLEProjectionFixture().layer()
        hidden = torch.randn(3, 4, 10240, device="cuda", dtype=torch.bfloat16)
        embedding = torch.randn(3, 4, 2560, device="cuda", dtype=torch.bfloat16)
        tokens = torch.zeros(3, 6, device="cuda", dtype=torch.int64)
        initial = torch.randn(3, 9, 10240, device="cuda", dtype=torch.bfloat16)
        out_plain, cand_plain = layer.decode_chunk(
            hidden, tokens, initial, embeddings=embedding
        )
        out_res, cand_res = layer.decode_chunk(
            hidden, tokens, initial, embeddings=embedding, add_residual=True
        )
        self.assertTrue(torch.equal(out_res, hidden + out_plain))
        self.assertTrue(torch.equal(cand_res, cand_plain))

        short_h = torch.randn(1, 17, 10240, device="cuda", dtype=torch.bfloat16)
        short_e = torch.randn(1, 17, 2560, device="cuda", dtype=torch.bfloat16)
        short_t = torch.zeros(1, 19, device="cuda", dtype=torch.int64)
        p_plain, s_plain = layer.prefill(short_h, short_t, embeddings=short_e)
        p_res, s_res = layer.prefill(
            short_h, short_t, embeddings=short_e, add_residual=True
        )
        self.assertTrue(torch.equal(p_res, short_h + p_plain))
        self.assertTrue(torch.equal(s_res, s_plain))


if __name__ == "__main__":
    unittest.main()
