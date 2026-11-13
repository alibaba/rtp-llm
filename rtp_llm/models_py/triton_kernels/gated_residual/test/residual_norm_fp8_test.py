import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.triton_kernels.gated_residual.norm import (
    grouped_rms_norm_triton,
    try_grouped_rms_norm_with_fp8_scale,
)

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.norm"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualNormFp8Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 fast path")
        torch.manual_seed(703)

    def test_business_large_rows_global_scale_includes_last_group(self):
        for rows in (16383, 16384, 16385, 24576, 32768, 49152):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                gamma = torch.zeros(10240, device="cuda", dtype=torch.bfloat16)
                x[-1].zero_()
                x[-1, -1] = 100
                fused = try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1e-6)
                self.assertIsNotNone(fused)
                actual, scale = fused
                reference, reference_scale = scaled_fp8_per_tensor_quant(actual)
                quantized, _ = scaled_fp8_per_tensor_quant(actual, scale=scale)
                self.assertTrue(torch.equal(scale, reference_scale))
                self.assertTrue(
                    torch.equal(
                        quantized.view(torch.uint8), reference.view(torch.uint8)
                    )
                )
                for start in range(0, rows, 1024):
                    expected = grouped_rms_norm_triton(
                        x[start : start + 1024], gamma, 2560, 1e-6
                    )
                    torch.testing.assert_close(
                        actual[start : start + 1024], expected, atol=0, rtol=0
                    )
                del x, fused, actual, reference, quantized

    def test_norm_scale_and_fp8_bytes(self):
        for rows in (256, 257, 4093, 4096, 12288):
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07
            for kind in ("random", "wide", "zero", "nonfinite"):
                with self.subTest(rows=rows, kind=kind):
                    if kind == "wide":
                        x.mul_(1000.0)
                    if kind == "zero":
                        x.zero_()
                    if kind == "nonfinite":
                        x.normal_()
                        x[0, 0] = float("inf")
                        x[1, 2560] = float("nan")
                    expected = grouped_rms_norm_triton(x, gamma, 2560, 1.0e-6)
                    reference, reference_scale = scaled_fp8_per_tensor_quant(expected)
                    actual, scale = try_grouped_rms_norm_with_fp8_scale(
                        x, gamma, 2560, 1.0e-6
                    )
                    torch.testing.assert_close(
                        actual, expected, rtol=0, atol=0, equal_nan=True
                    )
                    self.assertTrue(torch.equal(scale, reference_scale))
                    quantized, _ = scaled_fp8_per_tensor_quant(actual, scale=scale)
                    self.assertTrue(
                        torch.equal(
                            quantized.view(torch.uint8), reference.view(torch.uint8)
                        )
                    )

    def test_unsupported_shapes_and_training_fall_back(self):
        gamma = torch.zeros(10240, device="cuda", dtype=torch.bfloat16)
        for rows in (0, 1, 32, 255, 49153):
            x = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(
                try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1.0e-6)
            )
        x = torch.empty(257, 10240, device="cuda", dtype=torch.bfloat16)
        for eps in (-1.0, float("inf")):
            self.assertIsNone(try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, eps))
        x.requires_grad_(True)
        self.assertIsNone(try_grouped_rms_norm_with_fp8_scale(x, gamma, 2560, 1.0e-6))


if __name__ == "__main__":
    unittest.main()
