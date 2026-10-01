"""Owned shared-expert outputs can be consumed by a BF16 GEMM epilogue."""

import unittest

import torch

from rtp_llm.models_py.modules.kimi_k3.linear import bf16_linear_add_inplace


class MoETailInplaceTest(unittest.TestCase):
    def test_inplace_add_returns_owned_residual_and_matches_addmm(self):
        routed = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4) / 8
        weight = torch.arange(20, dtype=torch.bfloat16).reshape(5, 4) / 16
        shared = torch.arange(30, dtype=torch.bfloat16).reshape(6, 5) / 32
        expected = torch.addmm(shared, routed, weight.t())
        pointer = shared.data_ptr()

        result = bf16_linear_add_inplace(routed, weight, shared)

        self.assertEqual(result.data_ptr(), pointer)
        self.assertTrue(torch.equal(result, expected))

    def test_inplace_add_rejects_mismatched_shared_shape(self):
        routed = torch.ones((6, 4), dtype=torch.bfloat16)
        weight = torch.ones((5, 4), dtype=torch.bfloat16)
        shared = torch.zeros((6, 4), dtype=torch.bfloat16)

        with self.assertRaisesRegex(ValueError, "residual must match"):
            bf16_linear_add_inplace(routed, weight, shared)


if __name__ == "__main__":
    unittest.main()
