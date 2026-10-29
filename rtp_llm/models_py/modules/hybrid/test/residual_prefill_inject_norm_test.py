import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    grouped_rms_norm,
    inject_into_residual,
)
from rtp_llm.models_py.modules.hybrid.test import residual_fixtures as oracle
from rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_inject_norm_triton import (
    try_inject_grouped_norm,
)

MODULE = (
    "rtp_llm.models_py.triton_kernels.common.gated_residual_prefill_inject_norm_triton"
)


class ResidualPrefillInjectNormTest(unittest.TestCase):
    def test_unsupported_prefill_uses_original(self):
        for rows in (0, 33, 257, 2048, 16385):
            x = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.empty(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.empty(rows, 4, device="cuda", dtype=torch.bfloat16)
            gamma = torch.empty(10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_inject_grouped_norm(x, sub, gates, gamma, 1e-6))

    def test_complete_consumer_exact_and_small_fallback(self):
        for rows in (1, 4, 257, 2048, 4093, 4096, 5003, 8192, 12288, 16384):
            model = oracle.ResidualFp8Fixture().module(True)
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
            with patch(MODULE + ".try_inject_grouped_norm", return_value=None):
                expected = model.forward_with_injection(x, sub, gates)
            actual = model.forward_with_injection(x, sub, gates)
            oracle.ResidualFp8Fixture().assert_outputs(actual, expected)

    def test_bytes_scale_zero_and_nonfinite_values(self):
        for rows in (1, 4, 7, 24, 28, 32, 4096, 5003, 12288, 16384):
            for kind in ("random", "zero", "subnormal", "wide", "nonfinite"):
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
                gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
                if kind == "zero":
                    x.zero_()
                    sub.zero_()
                if kind == "wide":
                    x.mul_(1000)
                    sub.mul_(1000)
                if kind == "subnormal":
                    x.fill_(1.0e-40)
                    sub.zero_()
                if kind == "nonfinite":
                    x[0, 0] = float("inf")
                    x[-1, 2560] = float("nan")
                residual, norm, scale = try_inject_grouped_norm(
                    x, sub, gates, gamma, 1e-6
                )
                expected_residual = inject_into_residual(x, sub, gates)
                expected_norm = grouped_rms_norm(expected_residual, gamma, 2560, 1e-6)
                torch.testing.assert_close(
                    residual, expected_residual, rtol=0, atol=0, equal_nan=True
                )
                torch.testing.assert_close(
                    norm, expected_norm, rtol=0, atol=0, equal_nan=True
                )
                quant, sf = scaled_fp8_per_tensor_quant(norm, scale=scale)
                expected_quant, expected_sf = scaled_fp8_per_tensor_quant(expected_norm)
                self.assertTrue(torch.equal(sf, expected_sf))
                self.assertTrue(
                    torch.equal(
                        quant.view(torch.uint8), expected_quant.view(torch.uint8)
                    )
                )

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
