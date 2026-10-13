"""Real SM90 dense linears must keep raw UE8M0 block scales in FP32 layout."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4 import utils
from rtp_llm.models_py.modules.dsv4.fp8 import attention
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.quantized_linear import (
    create_fp8_linear,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "SM90 required",
)
class Fp8LinearSm90Test(unittest.TestCase):
    @torch.inference_mode()
    def test_attention_and_shared_linears_match_block_dequantized_reference(self):
        torch.manual_seed(1303)
        for n, k in ((256, 512), (4096, 1024)):
            with self.subTest(n=n, k=k):
                w = torch.randint(-3, 4, (n, k), device="cuda").to(torch.float8_e4m3fn)
                exponents = torch.randint(
                    123, 127, (n // 128, k // 128), device="cuda", dtype=torch.uint8
                )
                s = exponents.view(torch.float8_e8m0fnu)
                x = torch.randint(-7, 8, (8, k), device="cuda").to(torch.bfloat16)
                x[:, ::128] = 7
                scale = (
                    torch.pow(2.0, exponents.float() - 127)
                    .repeat_interleave(128, 0)
                    .repeat_interleave(128, 1)
                )
                reference = (x.float() @ (w.float() * scale).T).to(torch.bfloat16)
                for build in (
                    attention._v4_fp8_linear,
                    utils._v4_fp8_linear,
                    create_fp8_linear,
                ):
                    output = build(w, s)(x)
                    torch.testing.assert_close(output, reference, rtol=0, atol=0)
                weights = {"w": w, "s": s}
                torch.testing.assert_close(
                    attention._v4_fp8_linear_from_dict(weights, "w", "s")(x),
                    reference,
                    rtol=0,
                    atol=0,
                )
                self.assertIs(weights["s"], s)


if __name__ == "__main__":
    unittest.main()
