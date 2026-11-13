import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.hybrid.test import residual_fixtures as oracle
from rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm import (
    try_inject_grouped_norm,
)

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm"


class ResidualPrefillInjectNormTest(unittest.TestCase):
    def test_complete_consumer_matches_unfused_reference(self):
        for rows in (
            1,
            4,
            257,
            2048,
            4093,
            4096,
            5003,
            6144,
            8192,
            12288,
            16384,
            24576,
            32768,
            49152,
        ):
            model = oracle.ResidualFp8Fixture().module(True)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
            with patch(MODULE + ".try_inject_grouped_norm", return_value=None):
                expected = model.forward_with_injection(x, sub, gates)
            actual = model.forward_with_injection(x, sub, gates)
            oracle.ResidualFp8Fixture().assert_outputs(actual, expected)

    def test_decode_consumer_all_rows_and_graph_refresh(self):
        for rows in range(1, 33):
            model = oracle.ResidualFp8Fixture().module(True)
            for projection in (
                model.down_projection,
                model.up_projection,
                model.inject_projection,
            ):
                projection.role = "DECODE"
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
            with patch(MODULE + ".try_inject_grouped_norm", return_value=None):
                expected = model.forward_with_injection(x, sub, gates)
            actual = model.forward_with_injection(x, sub, gates)
            oracle.ResidualFp8Fixture().assert_outputs(actual, expected)
            # Seven rows cross the small quantizer's 65536-element limit.
            fused = try_inject_grouped_norm(x, sub, gates, model.norm_gamma, 1e-6)
            self.assertEqual(fused[2] is not None, rows >= 7)
            if rows not in (1, 4, 6, 7, 8, 24, 28, 32):
                continue
            model.forward_with_injection(x, sub, gates)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model.forward_with_injection(x, sub, gates)
            for magnitude in (0.0, 0.01, 1.0, 100.0):
                x.normal_().mul_(magnitude)
                sub.normal_().mul_(magnitude)
                gates.normal_()
                model.norm_gamma.normal_(0, 0.07)
                with patch(MODULE + ".try_inject_grouped_norm", return_value=None):
                    expected = model.forward_with_injection(x, sub, gates)
                graph.replay()
                torch.cuda.synchronize()
                oracle.ResidualFp8Fixture().assert_outputs(actual, expected)
            graph.reset()

    def test_graph_reads_input_sublayer_gates_gamma_and_scale(self):
        for rows in (4096, 5003, 12288, 16384):
            model = oracle.ResidualFp8Fixture().module(True)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
            model.forward_with_injection(x, sub, gates)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model.forward_with_injection(x, sub, gates)
            for _ in range(2):
                x.normal_()
                sub.normal_()
                gates.normal_()
                model.norm_gamma.normal_(0, 0.07)
                with patch(MODULE + ".try_inject_grouped_norm", return_value=None):
                    expected = model.forward_with_injection(x, sub, gates)
                graph.replay()
                torch.cuda.synchronize()
                oracle.ResidualFp8Fixture().assert_outputs(actual, expected)


if __name__ == "__main__":
    unittest.main()
