import unittest

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.triton_kernels.common.silu_fp8_quant import try_silu_mul_quant

MODULE = "rtp_llm.models_py.triton_kernels.common.silu_fp8_quant"


class PrefillSiluQuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(717)

    def test_exact_codes_scales_padding_and_nonfinite_values(self):
        for width in (320, 640):
            for rows in (4093, 4096, 5003, 8192, 12288, 16384):
                gate_up = torch.randn(
                    rows, width * 2, device="cuda", dtype=torch.bfloat16
                )
                for kind in ("random", "zero", "subnormal", "nonfinite"):
                    if kind == "zero":
                        gate_up.zero_()
                    elif kind == "subnormal":
                        gate_up[:, :width].fill_(1.0e-38)
                        gate_up[:, width:].fill_(0.75)
                    elif kind == "nonfinite":
                        gate_up.normal_()
                        gate_up[0, 0] = float("nan")
                        gate_up[1, 0] = float("inf")
                    activated = flashinfer.activation.silu_and_mul(gate_up)
                    padded = torch.empty(
                        ((rows + 15) // 16 * 16, width),
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    )
                    expected = scaled_fp8_per_tensor_quant(activated, output=padded)
                    actual = try_silu_mul_quant(gate_up)
                    for a, b in zip(actual, expected):
                        self.assertTrue(
                            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                        )

    def test_unsupported_inputs_and_output_aliases_are_untouched(self):
        good = torch.ones(4093, 640, device="cuda", dtype=torch.bfloat16)
        for x in (
            good.cpu(),
            good.float(),
            good.t(),
            good[:, ::2],
            good.clone().requires_grad_(),
            torch.empty(4092, 640, device="cuda", dtype=torch.bfloat16),
            torch.empty(16385, 640, device="cuda", dtype=torch.bfloat16),
            torch.empty(4093, 638, device="cuda", dtype=torch.bfloat16),
        ):
            self.assertIsNone(try_silu_mul_quant(x))
        for output in (
            torch.ones(4093, 320, device="cuda", dtype=torch.bfloat16),
            torch.ones(4093, 321, device="cuda", dtype=torch.float8_e4m3fn),
            good.view(torch.float8_e4m3fn).flatten()[: 4093 * 320].view(4093, 320),
        ):
            before = output.view(torch.uint8).clone()
            self.assertIsNone(try_silu_mul_quant(good, output=output))
            self.assertTrue(torch.equal(output.view(torch.uint8), before))


if __name__ == "__main__":
    unittest.main()
