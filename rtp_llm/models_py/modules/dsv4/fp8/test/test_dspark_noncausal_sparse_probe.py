"""Probe that FlashMLA sparse prefill accepts explicitly future KV slots.

DSpark's block forward is non-causal: every query in the five-token draft
block must attend to the complete block.  FlashMLA consumes explicit indices,
so this test supplies the same prefix-plus-full-block row to every query and
checks the real kernel against a dense reference including the attention sink.
"""

from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DSparkNonCausalSparseProbe(unittest.TestCase):
    def test_future_slots_match_dense_reference(self) -> None:
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        torch.manual_seed(20260731)
        num_query = 5
        prefix = 3
        num_kv = prefix + num_query
        num_heads = 64
        head_dim = 512
        topk = 64
        scale = head_dim**-0.5

        q = torch.randn(
            num_query,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        kv = torch.randn(
            num_kv, 1, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.full(
            (num_query, 1, topk), -1, dtype=torch.int32, device="cuda"
        )
        # Every query sees the whole prefix and all five draft positions.  For
        # the first four queries this explicitly includes future slot ids.
        indices[:, 0, :num_kv] = torch.arange(
            num_kv, dtype=torch.int32, device="cuda"
        )
        topk_length = torch.full(
            (num_query,), num_kv, dtype=torch.int32, device="cuda"
        )
        attn_sink = torch.zeros(num_heads, dtype=torch.float32, device="cuda")

        actual, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=scale,
            attn_sink=attn_sink,
            topk_length=topk_length,
        )

        qf = q.float()
        kf = kv[:, 0].float()
        scores = torch.einsum("qhd,kd->qhk", qf, kf) * scale
        row_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, attn_sink.view(1, -1, 1))
        weights = torch.exp(scores - row_max)
        denom = weights.sum(dim=-1, keepdim=True) + torch.exp(
            attn_sink.view(1, -1, 1) - row_max
        )
        expected = torch.einsum("qhk,kd->qhd", weights / denom, kf)

        torch.testing.assert_close(
            actual.float(), expected, rtol=2e-2, atol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
