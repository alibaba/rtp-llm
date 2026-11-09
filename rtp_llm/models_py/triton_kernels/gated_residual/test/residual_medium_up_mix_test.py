"""Full HC consumer: medium GEMM epilogue, fresh Graph inputs and gains."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.triton_kernels.gated_residual.medium_up_mix import (
    try_fp8_hc_medium_up_mix,
)
from rtp_llm.models_py.triton_kernels.gated_residual.mix import fused_mix_reduce

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.medium_up_mix"


class MediumHCConsumerTest(unittest.TestCase):
    def test_native_quantization_matches_for_edge_values(self):
        w = (torch.randn(10240, 320, device="cuda") * 0.1).to(torch.float8_e4m3fn)
        sw = torch.tensor([0.119713], device="cuda")
        quant_module = "rtp_llm.models_py.triton_kernels.common.fp8_small_quant"
        for rows in (8, 12, 16, 20, 24, 28, 32):
            x = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
            norm = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for kind in ("random", "zero", "subnormal", "nonfinite"):
                if kind == "zero":
                    x.zero_()
                elif kind == "subnormal":
                    x.fill_(1.0e-40)
                elif kind == "nonfinite":
                    x.normal_()
                    x[0, 0] = float("inf")
                    x[-1, 1] = float("nan")
                with patch(quant_module + ".try_small_fp8_quant", return_value=None):
                    expected = try_fp8_hc_medium_up_mix(x, w, sw, norm, 4)
                actual = try_fp8_hc_medium_up_mix(x, w, sw, norm, 4)
                torch.testing.assert_close(
                    actual, expected, rtol=0, atol=0, equal_nan=True
                )

    def test_up_consumer_non_power_of_two_scales(self):
        torch.manual_seed(28173)
        for rows in (8, 16, 20, 24, 28, 32):
            for magnitude, scale in ((0.01, 0.013475), (1.0, 0.119713), (10.0, 0.2719)):
                with self.subTest(rows=rows, magnitude=magnitude, scale=scale):
                    x = (
                        torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
                        * magnitude
                    )
                    w = (torch.randn(10240, 320, device="cuda") * 0.1).to(
                        torch.float8_e4m3fn
                    )
                    sw = torch.tensor([scale], device="cuda")
                    norm = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                    expected = fused_mix_reduce(
                        Fp8Linear(w, sw, role="DECODE")(x), norm, 4
                    )
                    actual = try_fp8_hc_medium_up_mix(x, w, sw, norm, 4)
                    self.assertIsNotNone(actual)
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int16), expected.view(torch.int16)
                        ),
                        int((actual != expected).sum()),
                    )

    def test_unsupported_small_and_large_shapes_fallback(self):
        w = torch.empty((10240, 320), device="cuda", dtype=torch.float8_e4m3fn)
        s = torch.ones(1, device="cuda")
        for rows in (1, 3, 4, 5, 7, 33, 4096):
            x = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            norm = torch.empty(rows, 10240, device="cuda", dtype=x.dtype)
            self.assertIsNone(try_fp8_hc_medium_up_mix(x, w, s, norm, 4))


if __name__ == "__main__":
    unittest.main()
