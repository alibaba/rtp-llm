"""GPU tests: triton sparse FMHA vs the torch baseline.

Verifies that the triton kernel produces the same output as
``gather_and_attend`` (the torch reference) on random data with
a small production-like geometry.
"""

import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.sparse_fmha import (
    sparse_prefill_attn,
    sparse_prefill_attn_torch_reference,
)

_H_Q, _H_KV, _D = 24, 2, 256
_DEV = "cuda"


class SparseFmhaTest(unittest.TestCase):
    def test_matches_torch_reference(self):
        torch.manual_seed(0)
        B, S, T, K = 1, 8, 64, 16
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, T, _D, device=_DEV).to(torch.bfloat16)
        v = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)

        # causal selection
        sel = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)
        for b in range(B):
            for s in range(S):
                n = min(K - 1, s + 1)
                sel[b, s, :n] = torch.arange(n, device=_DEV, dtype=torch.int32)

        triton_out = sparse_prefill_attn(q, k, v, sel)
        torch_out = sparse_prefill_attn_torch_reference(q, k, v, sel)

        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    def test_all_masked_produces_zero(self):
        B, S, K = 1, 4, 8
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        sel = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)

        triton_out = sparse_prefill_attn(q, k, v, sel)
        self.assertTrue((triton_out == 0).all())

    def test_out_of_range_selected_index_is_rejected(self):
        B, S, K = 1, 4, 8
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        selected = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)
        selected[0, 0, 0] = S

        with self.assertRaisesRegex(ValueError, "outside KV sequence length"):
            sparse_prefill_attn(q, k, v, selected)

    def test_matches_torch_on_larger_shape(self):
        torch.manual_seed(1)
        B, S, T, K = 2, 16, 128, 32
        q = (torch.randn(B, _H_Q, S, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        k = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        v = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        sel = torch.randint(0, T, (B, S, K), dtype=torch.int32, device=_DEV)
        sel[:, :, : K // 2] = -1  # mix in some padding

        triton_out = sparse_prefill_attn(q, k, v, sel)
        torch_out = sparse_prefill_attn_torch_reference(q, k, v, sel)
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
