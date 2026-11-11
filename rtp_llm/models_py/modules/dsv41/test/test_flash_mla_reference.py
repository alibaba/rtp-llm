"""Non-SM100 sparse-prefill reference must match gather+softmax semantics."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.flash_mla import (
    _sparse_prefill_reference,
    flash_mla_sparse_fwd,
)


def _reference(q, kv, indices, sm_scale, sink, lens):
    """Materialize the gathered K tensor; only for this small unit test."""
    s_q, heads, _ = q.shape
    kv2 = kv[:, 0, :].float()
    idx = indices[:, 0, :].long()
    valid = (idx >= 0) & (idx < kv2.shape[0])
    valid &= torch.arange(idx.shape[-1]).unsqueeze(0) < lens.unsqueeze(1)
    sel = kv2[idx.masked_fill(~valid, 0)]
    logits = torch.einsum("shd,skd->shk", q.float(), sel) * sm_scale
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    has = valid.any(-1)
    mx = logits.amax(-1)
    mx_safe = torch.where(has.unsqueeze(-1), mx, torch.zeros_like(mx))
    sums = torch.exp(logits - mx_safe.unsqueeze(-1)).sum(-1)
    lse = mx_safe + torch.log(sums.clamp_min(1e-30))
    lse = torch.where(has.unsqueeze(-1), lse, torch.full_like(lse, float("-inf")))
    p = torch.exp(logits - mx_safe.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(sink.view(1, -1) - mx_safe)
    out = torch.einsum("shk,skd->shd", p, sel) / denom.unsqueeze(-1)
    out = torch.where(has.view(-1, 1, 1), out, torch.zeros_like(out))
    return out.to(q.dtype), lse


class SparsePrefillReferenceTest(unittest.TestCase):
    def test_matches_gather_reference_with_length_and_padding(self):
        torch.manual_seed(7)
        s_q, heads, dim, topk = 19, 5, 32, 13
        q = torch.randn(s_q, heads, dim).to(torch.bfloat16)
        kv = torch.randn(37, 1, dim).to(torch.bfloat16)
        sink = torch.randn(heads)
        lens = torch.randint(1, topk + 1, (s_q,), dtype=torch.int32)
        idx = torch.randint(0, 37, (s_q, 1, topk), dtype=torch.int32)
        idx[0, 0, 0] = -1  # explicit padding sentinel
        idx[1, 0, :] = 99  # out of range must be ignored

        out, maximum, lse = _sparse_prefill_reference(
            q, kv, idx, 0.125, d_v=dim, attn_sink=sink, topk_length=lens
        )
        want, want_lse = _reference(q, kv, idx, 0.125, sink, lens)
        torch.testing.assert_close(out, want, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(lse, want_lse, atol=2e-2, rtol=2e-2)
        self.assertTrue(torch.isfinite(maximum[2:]).all())

    def test_empty_rows_are_finite_zero(self):
        q = torch.randn(3, 2, 8, dtype=torch.bfloat16)
        kv = torch.randn(5, 1, 8, dtype=torch.bfloat16)
        idx = torch.full((3, 1, 4), -1, dtype=torch.int32)
        out, _, lse = _sparse_prefill_reference(q, kv, idx, 0.5, d_v=8)
        self.assertEqual(out.shape, (3, 2, 8))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out == 0).all())
        self.assertTrue(torch.isinf(lse).all())

    def test_batch_major_four_dim_input(self):
        q = torch.randn(2, 4, 3, 16, dtype=torch.bfloat16)
        kv = torch.randn(2, 9, 1, 16, dtype=torch.bfloat16)
        idx = torch.randint(0, 9, (2, 4, 1, 3), dtype=torch.int32)
        out, maximum, lse = flash_mla_sparse_fwd(q, kv, idx, 0.25, d_v=16)
        self.assertEqual(out.shape, (2, 4, 3, 16))
        self.assertEqual(maximum.shape, (2, 4, 3))
        self.assertEqual(lse.shape, (2, 4, 3))

    def test_dispatch_uses_reference_off_cuda(self):
        q = torch.randn(4, 2, 16, dtype=torch.bfloat16)
        kv = torch.randn(9, 1, 16, dtype=torch.bfloat16)
        idx = torch.randint(0, 9, (4, 1, 3), dtype=torch.int32)
        out, _, _ = flash_mla_sparse_fwd(q, kv, idx, 0.25, d_v=16)
        self.assertEqual(out.shape, (4, 2, 16))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
