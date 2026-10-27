"""Fused W4A16 rotate prologues vs the unfused rotate / silu_and_mul path.

The Hadamard sums run in fp32 in a different order than fast_hadamard_transform, so
rare outputs differ by one bf16 ulp; everything else must be bit-identical.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.w4a16_rotate_fused import (
    fused_rotate,
    fused_silu_mul_rotate,
)
from rtp_llm.models_py.kernels.cuda.w4a16_sm120 import rotate
from rtp_llm.models_py.modules.base.cuda.activation import FusedSiluAndMul


def assert_within_one_ulp(test, actual, expected):
    # Adjacent same-sign bf16 values differ by 1 in their bit patterns.
    bits = (actual.view(torch.int16).int() - expected.view(torch.int16).int()).abs()
    bits = torch.where(actual == expected, 0, bits)
    test.assertLessEqual(bits.max().item(), 1)
    test.assertLess((bits != 0).float().mean().item(), 1e-3)


class W4A16RotateFusedTest(unittest.TestCase):
    def test_matches_unfused_path(self):
        # Qwen3.5-27B, TP2: hidden 5120, intermediate 17408 / 2 = 8704.
        hidden, inter = 5120, 8704
        torch.manual_seed(0)
        signs_up = (torch.randint(0, 2, (hidden,), device="cuda") * 2 - 1).bfloat16()
        signs_down = (torch.randint(0, 2, (inter,), device="cuda") * 2 - 1).bfloat16()
        silu_and_mul = FusedSiluAndMul()
        for tokens in (1, 8):
            with self.subTest(tokens=tokens):
                x = torch.randn(tokens, hidden, device="cuda").bfloat16()
                gate_up = (torch.randn(tokens, 2 * inter, device="cuda") * 4).bfloat16()
                assert_within_one_ulp(
                    self, fused_rotate(x, signs_up), rotate(x, signs_up)
                )
                assert_within_one_ulp(
                    self,
                    fused_silu_mul_rotate(gate_up, signs_down),
                    rotate(silu_and_mul(gate_up), signs_down),
                )


if __name__ == "__main__":
    unittest.main()
