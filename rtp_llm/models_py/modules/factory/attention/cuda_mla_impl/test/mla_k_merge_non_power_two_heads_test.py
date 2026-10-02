"""Numerical coverage for MLA K merging with non-power-of-two head counts."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla import (
    concat_and_cast_mha_k_triton,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MlaKMergeNonPowerTwoHeadsTest(unittest.TestCase):
    def test_twelve_heads_from_strided_kv_projection(self):
        torch.manual_seed(27)
        tokens, heads, nope_dim, rope_dim, value_dim = 257, 12, 128, 64, 128
        kv_projection = torch.randn(
            tokens, heads, nope_dim + value_dim, device="cuda", dtype=torch.bfloat16
        )
        k_nope = kv_projection[..., :nope_dim]
        k_rope = torch.randn(
            tokens, 1, rope_dim, device="cuda", dtype=torch.bfloat16
        )
        self.assertFalse(k_nope.is_contiguous())

        actual = torch.empty(
            tokens, heads, nope_dim + rope_dim, device="cuda", dtype=torch.bfloat16
        )
        concat_and_cast_mha_k_triton(actual, k_nope, k_rope)
        torch.cuda.synchronize()

        expected = torch.cat((k_nope, k_rope.expand(-1, heads, -1)), dim=-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
