"""Probe that FlashMLA sparse prefill accepts explicitly future KV slots.

DSpark's block forward is non-causal: every query in the five-token draft
block must attend to the complete block.  FlashMLA consumes explicit indices,
so this test supplies the same prefix-plus-full-block row to every query and
checks the real kernel against a dense reference including the attention sink.
"""

from __future__ import annotations

import unittest

import torch


class DSparkNonCausalMetadataTest(unittest.TestCase):
    def test_varlen_fresh_and_continuation_indices(self) -> None:
        from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
            build_dspark_noncausal_swa_indices,
        )

        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        query_lens = torch.tensor([4, 4], dtype=torch.int32)
        req = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32)

        fresh_indices, fresh_lens = build_dspark_noncausal_swa_indices(
            window_size=8,
            cu_seqlens=cu,
            input_lengths=query_lens,
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            req_id_per_token=req,
        )
        self.assertEqual(fresh_indices.shape, (8, 8))
        torch.testing.assert_close(
            fresh_indices[:4, :4], torch.arange(4, dtype=torch.int32).expand(4, 4)
        )
        torch.testing.assert_close(
            fresh_indices[4:, :4], torch.arange(4, 8, dtype=torch.int32).expand(4, 4)
        )
        self.assertTrue(bool((fresh_indices[:, 4:] == -1).all()))
        torch.testing.assert_close(fresh_lens, torch.full((8,), 4, dtype=torch.int32))

        # Workspace request 0: cached rows [0, 1, 2], new rows [3..6].
        # Request 1: seven cached rows [11..17], but only the trailing four
        # [14..17] fit beside all four new rows [18..21].
        cont_indices, cont_lens = build_dspark_noncausal_swa_indices(
            window_size=8,
            cu_seqlens=cu,
            input_lengths=query_lens,
            prefix_lengths=torch.tensor([3, 10], dtype=torch.int32),
            req_id_per_token=req,
            workspace_stride=11,
        )
        torch.testing.assert_close(
            cont_indices[:4, :7], torch.arange(7, dtype=torch.int32).expand(4, 7)
        )
        self.assertTrue(bool((cont_indices[:4, 7:] == -1).all()))
        torch.testing.assert_close(
            cont_indices[4:], torch.arange(14, 22, dtype=torch.int32).expand(4, 8)
        )
        torch.testing.assert_close(
            cont_lens,
            torch.tensor([7] * 4 + [8] * 4, dtype=torch.int32),
        )


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

    def test_continuation_workspace_indices_match_dense_reference(self) -> None:
        from flash_mla import flash_mla_sparse_fwd  # type: ignore[import-not-found]

        from rtp_llm.models_py.modules.dsv4.fp8._swa_ops_triton import (
            build_dspark_noncausal_swa_indices,
        )

        torch.manual_seed(20260801)
        query_lens = torch.tensor([4, 4], dtype=torch.int32, device="cuda")
        prefix_lens = torch.tensor([6, 130], dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, 4, 8], dtype=torch.int32, device="cuda")
        req = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32, device="cuda")
        window_size = 128
        # The concat path reserves min(prefix, win - 1) + S rows per request.
        workspace_stride = 131
        # FlashMLA sparse prefill specializes the production DSV4 head count.
        num_heads = 64
        head_dim = 512
        scale = head_dim**-0.5

        indices, lengths = build_dspark_noncausal_swa_indices(
            window_size=window_size,
            cu_seqlens=cu,
            input_lengths=query_lens,
            prefix_lengths=prefix_lens,
            req_id_per_token=req,
            workspace_stride=workspace_stride,
        )
        # Query row zero explicitly sees all new rows, including its three
        # future positions at workspace slots 7, 8 and 9.
        self.assertEqual(indices[0, lengths[0] - 1].item(), 9)

        workspace = torch.randn(
            2,
            workspace_stride,
            1,
            head_dim,
            dtype=torch.bfloat16,
            device="cuda",
        )
        q = torch.randn(
            8, num_heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        sink = torch.randn(num_heads, dtype=torch.float32, device="cuda")
        actual, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=workspace.view(2 * workspace_stride, 1, head_dim),
            indices=indices.unsqueeze(1),
            sm_scale=scale,
            attn_sink=sink,
            topk_length=lengths,
        )

        selected = workspace.view(2 * workspace_stride, head_dim).index_select(
            0, indices.clamp_min(0).reshape(-1).long()
        ).view(8, window_size, head_dim)
        scores = torch.einsum("thd,tkd->thk", q.float(), selected.float()) * scale
        valid = torch.arange(window_size, device="cuda").view(1, 1, -1) < lengths.view(
            -1, 1, 1
        )
        scores = scores.masked_fill(~valid, float("-inf"))
        sink3 = sink.view(1, -1, 1)
        row_max = torch.maximum(scores.max(dim=-1, keepdim=True).values, sink3)
        weights = torch.exp(scores - row_max).masked_fill(~valid, 0.0)
        denom = weights.sum(dim=-1, keepdim=True) + torch.exp(sink3 - row_max)
        expected = torch.einsum("thk,tkd->thd", weights / denom, selected.float())

        torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
