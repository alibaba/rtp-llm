"""GPU tests for the fused pairwise merge kernel (``merge_triton``).

Checks the fused merge against the torch reference implementation on two- and
three-part merges, including empty rows (``-inf`` LSE in every part), and the
dtype/validation contract.
"""

from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.triton_kernels.lsa_attention.merge_triton import (
    merge_attention_triton,
)
from rtp_llm.models_py.triton_kernels.lsa_attention.swa_dense_triton import (
    merge_attention,
)

CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(CUDA_AVAILABLE, "CUDA is required for the Triton merge kernel")
class TestMergeTriton(unittest.TestCase):
    def _case(self, parts: int, rows_with_all_empty: bool = True) -> None:
        torch.manual_seed(4)
        t_count, heads, dv = 128, 32, 512
        device = "cuda"
        outs = [
            (torch.randn(t_count, heads, dv, device=device) * 0.1).bfloat16()
            for _ in range(parts)
        ]
        lses = [torch.randn(t_count, heads, device=device) * 2 for _ in range(parts)]
        if rows_with_all_empty:
            for lse in lses:
                lse[0] = float("-inf")
                lse[1, :] = float("-inf")

        ref_out, ref_lse = merge_attention(list(zip(outs, lses)))
        got_out, got_lse = merge_attention_triton(list(zip(outs, lses)))
        self.assertTrue(torch.allclose(got_out.float(), ref_out.float(), atol=3e-3, rtol=3e-2))
        self.assertTrue(torch.allclose(got_lse, ref_lse, atol=1e-5, rtol=1e-5))
        # Fully empty rows: zero output, -inf LSE.
        self.assertTrue(torch.all(got_out[0] == 0))
        self.assertTrue(torch.all(got_lse[0] == float("-inf")))
        self.assertTrue(torch.all(got_out[1] == 0))
        self.assertTrue(torch.all(got_lse[1] == float("-inf")))

    def test_two_parts(self) -> None:
        self._case(parts=2)

    def test_three_parts(self) -> None:
        self._case(parts=3)

    def test_single_part_is_identity(self) -> None:
        torch.manual_seed(5)
        out = (torch.randn(16, 8, 512, device="cuda") * 0.1).bfloat16()
        lse = torch.randn(16, 8, device="cuda")
        got_out, got_lse = merge_attention_triton([(out, lse)])
        self.assertTrue(torch.equal(got_out, out))
        self.assertTrue(torch.equal(got_lse, lse))

    def test_dv_power_of_two_required(self) -> None:
        out = torch.zeros(2, 2, 100, device="cuda")
        lse = torch.zeros(2, 2, device="cuda")
        with self.assertRaises(ValueError):
            merge_attention_triton([(out, lse), (out, lse)])


if __name__ == "__main__":
    unittest.main()