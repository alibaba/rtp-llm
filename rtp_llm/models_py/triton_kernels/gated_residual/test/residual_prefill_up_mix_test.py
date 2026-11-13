import unittest

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.triton_kernels.gated_residual.mix import fused_mix_reduce
from rtp_llm.models_py.triton_kernels.gated_residual.prefill_up_mix import (
    try_fp8_prefill_up_mix,
)

MODULE = "rtp_llm.models_py.triton_kernels.gated_residual.prefill_up_mix"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualPrefillUpMixTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 measured prefill consumer")
        torch.manual_seed(111201)

    def test_business_prefill_up_mix_preserves_fp8_boundaries(self):
        for rows in (6144, 8192, 16384, 24576, 32768, 49152):
            with self.subTest(rows=rows):
                x = torch.randn(rows, 320, device="cuda", dtype=torch.bfloat16)
                w = (torch.randn(10240, 320, device="cuda") * 0.1).to(
                    torch.float8_e4m3fn
                )
                scale = torch.tensor([0.119713], device="cuda")
                norm = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                expected = fused_mix_reduce(
                    Fp8Linear(w, scale, role="PREFILL")(x), norm, 4
                )
                actual = try_fp8_prefill_up_mix(x, w, scale, norm, 4)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_native_projection_and_bf16_boundaries_with_independent_scales(self):
        for rows in (4093, 4094, 4095, 4096, 12288):
            for magnitude, scale in ((0.01, 0.013475), (1.0, 0.119713), (10.0, 0.2719)):
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
                    Fp8Linear(w, sw, role="PREFILL")(x), norm, 4
                )
                actual = try_fp8_prefill_up_mix(x, w, sw, norm, 4)
                self.assertIsNotNone(actual)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_unmeasured_shapes_and_gradients_decline_before_writes(self):
        w = torch.empty(10240, 320, device="cuda", dtype=torch.float8_e4m3fn)
        scale = torch.ones(1, device="cuda")
        for rows in (1, 3, 4, 8, 16, 32, 33, 4092, 4097, 12287, 12289, 49153):
            x = torch.empty(rows, 320, device="cuda", dtype=torch.bfloat16)
            norm = torch.empty(rows, 10240, device="cuda", dtype=torch.bfloat16)
            self.assertIsNone(try_fp8_prefill_up_mix(x, w, scale, norm, 4))
        x = torch.empty(
            4096, 320, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        norm = torch.empty(4096, 10240, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(try_fp8_prefill_up_mix(x, w, scale, norm, 4))


if __name__ == "__main__":
    unittest.main()
