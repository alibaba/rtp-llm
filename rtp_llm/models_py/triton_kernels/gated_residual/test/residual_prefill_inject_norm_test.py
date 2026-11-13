import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    grouped_rms_norm,
    inject_into_residual,
)
from rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm import (
    try_inject_grouped_norm,
)

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.prefill_inject_norm"


class ResidualPrefillInjectNormTest(unittest.TestCase):
    def test_business_large_rows_inject_norm_matches_reference(self):
        torch.manual_seed(20261105)
        for rows in (16383, 16384, 16385, 24576, 32768, 49152):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                sub = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                gates = torch.randn(rows, 4, device="cuda", dtype=torch.bfloat16)
                gamma = torch.zeros(10240, device="cuda", dtype=torch.bfloat16)
                # A maximum in the final group of the final row catches lost
                # reduction tiles and masked tails without changing norm gain.
                x[-1].zero_()
                sub[-1].zero_()
                x[-1, -1] = 100
                fused = try_inject_grouped_norm(x, sub, gates, gamma, 1e-6)
                self.assertIsNotNone(fused, "business prefill must use fusion")
                residual, norm, scale = fused
                self.assertIsNotNone(scale)
                for start in range(0, rows, 1024):
                    stop = min(start + 1024, rows)
                    expected_residual = inject_into_residual(
                        x[start:stop], sub[start:stop], gates[start:stop]
                    )
                    expected_norm = grouped_rms_norm(
                        expected_residual, gamma, 2560, 1e-6
                    )
                    torch.testing.assert_close(
                        residual[start:stop], expected_residual, atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        norm[start:stop], expected_norm, atol=0, rtol=0
                    )
                reference_quant, reference_scale = scaled_fp8_per_tensor_quant(norm)
                quant, _ = scaled_fp8_per_tensor_quant(norm, scale=scale)
                torch.testing.assert_close(scale, reference_scale, atol=0, rtol=0)
                self.assertTrue(
                    torch.equal(
                        quant.view(torch.uint8), reference_quant.view(torch.uint8)
                    )
                )
                self.assertEqual(
                    norm[-1, -1].float().item(), norm.abs().max().float().item()
                )
                del x, sub, gates, residual, norm, fused, quant, reference_quant

    def test_unsupported_prefill_uses_original(self):
        for rows in (0, 33, 257, 2048, 49153):
            x = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            sub = torch.empty(rows, 2560, device="cuda", dtype=torch.bfloat16)
            gates = torch.empty(rows, 4, device="cuda", dtype=torch.bfloat16)
            gamma = torch.empty(10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_inject_grouped_norm(x, sub, gates, gamma, 1e-6))

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


if __name__ == "__main__":
    unittest.main()
