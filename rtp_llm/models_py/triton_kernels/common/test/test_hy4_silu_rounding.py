"""HY4 must preserve the BF16 SiLU boundary before intermediate MXFP8."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed
from rtp_llm.models_py.triton_kernels.common.activation import (
    hy4_silu_and_mul,
    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestHy4SiluRounding(unittest.TestCase):
    def test_activation_and_quantization(self):
        for rows in (1, 32, 256, 1250):
            with self.subTest(rows=rows):
                generator = torch.Generator(device="cuda").manual_seed(927)
                x = torch.randn(
                    rows, 4096, generator=generator, device="cuda"
                ).bfloat16()
                # Exercise zero and small-amplitude MXFP8 groups too.
                x[:, :32] = 0
                x[:, 32:64] *= 1e-6
                gate, up = x.chunk(2, dim=-1)
                expected = F.silu(gate) * up
                actual = hy4_silu_and_mul(x)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_q, expected_s = mxfp8_quant_act_packed(expected)
                actual_q, actual_s = (
                    silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                        x,
                        quant_group_size=32,
                        scale_ue8m0=True,
                        mxfp8_semantics=True,
                        round_silu_bf16=True,
                    )
                )
                torch.testing.assert_close(
                    actual_q.view(torch.uint8),
                    expected_q.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(actual_s, expected_s, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
