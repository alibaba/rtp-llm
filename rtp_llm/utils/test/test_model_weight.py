"""Unit tests for model_weight utility functions.

Covers:
  - stack_moe_w1: optimized 2x stack_0 + concat_1 path
  - concat_1: fp8 uint8-view workaround
"""

import unittest

import torch

from rtp_llm.utils.model_weight import concat_1, stack_0, stack_moe_w1


class TestStackMoeW1(unittest.TestCase):
    def test_equivalence_with_naive(self):
        """New stack_moe_w1 (2x stack_0 + concat_1) should produce same
        result as the naive approach (per-expert concat_0 + stack_0)."""
        num_experts = 8
        intermediate = 16
        hidden = 32
        gate = [torch.randn(intermediate, hidden) for _ in range(num_experts)]
        up = [torch.randn(intermediate, hidden) for _ in range(num_experts)]

        # Naive reference: per-expert concat + stack
        per_expert = []
        for g, u in zip(gate, up):
            per_expert.append(torch.cat([g, u], dim=0))
        expected = torch.stack(per_expert, dim=0)

        # Optimized path
        result = stack_moe_w1(gate + up)

        self.assertEqual(expected.shape, result.shape)
        torch.testing.assert_close(expected, result)

    def test_single_expert(self):
        """Boundary: num_experts=1."""
        gate = [torch.randn(4, 8)]
        up = [torch.randn(4, 8)]
        result = stack_moe_w1(gate + up)
        self.assertEqual(result.shape, (1, 8, 8))

    def test_shape(self):
        """Output shape should be [num_experts, 2*intermediate, hidden]."""
        num_experts = 4
        intermediate = 10
        hidden = 20
        gate = [torch.randn(intermediate, hidden) for _ in range(num_experts)]
        up = [torch.randn(intermediate, hidden) for _ in range(num_experts)]
        result = stack_moe_w1(gate + up)
        self.assertEqual(result.shape, (num_experts, 2 * intermediate, hidden))


class TestConcat1Fp8(unittest.TestCase):
    @unittest.skipUnless(hasattr(torch, "float8_e4m3fn"), "float8 not supported")
    def test_concat_1_fp8(self):
        """concat_1 with fp8 tensors should produce correct result via uint8 view."""
        a = torch.randn(4, 8).to(torch.float8_e4m3fn)
        b = torch.randn(4, 8).to(torch.float8_e4m3fn)
        result = concat_1([a, b])
        self.assertEqual(result.shape, (4, 16))
        self.assertEqual(result.dtype, torch.float8_e4m3fn)

        # Verify values match manual uint8 concat
        expected_u8 = torch.cat(
            [a.view(torch.uint8), b.view(torch.uint8)], dim=1
        ).contiguous()
        expected = expected_u8.view(torch.float8_e4m3fn)
        torch.testing.assert_close(result.view(torch.uint8), expected.view(torch.uint8))

    def test_concat_1_normal(self):
        """concat_1 with normal dtype should work as standard torch.cat."""
        a = torch.randn(3, 5)
        b = torch.randn(3, 7)
        result = concat_1([a, b])
        expected = torch.cat([a, b], dim=1)
        torch.testing.assert_close(result, expected)

    def test_concat_1_single_tensor(self):
        """Single tensor should be returned as-is."""
        a = torch.randn(3, 5)
        result = concat_1([a])
        self.assertIs(result, a)


if __name__ == "__main__":
    unittest.main()
