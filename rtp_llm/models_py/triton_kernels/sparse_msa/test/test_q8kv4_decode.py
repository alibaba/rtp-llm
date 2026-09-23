import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    cache_layout,
    gather_index_rows,
    gather_main_rows,
    quantize_main_index_rows,
)
from rtp_llm.models_py.triton_kernels.sparse_msa.decode.q8kv4_decode import (
    q8kv4_paged_sparse_decode,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestQ8KV4Decode(unittest.TestCase):
    def test_short_fake_stream_block_table_smaller_than_topk(self):
        """DP idle fake streams reserve only 1 + propose_step cache blocks."""
        torch.manual_seed(20260924)
        device = torch.device("cuda")
        batch, q_heads, kv_heads, dim = 1, 64, 4, 128
        page, blocks, topk = 128, 4, 16
        groups = dim // 16
        main_bytes = 2 * kv_heads * page * dim // 2
        side_bytes = 2 * kv_heads * page * groups + page * dim // 2 + page * groups
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, kv_heads, page, dim)
        slots = torch.arange(blocks * page, dtype=torch.int64, device=device)
        k = torch.randn(
            blocks * page, kv_heads, dim, dtype=torch.bfloat16, device=device
        )
        v = torch.randn_like(k)
        idx_k = torch.randn(blocks * page, 1, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(k, v, idx_k, slots, layout)

        result = q8kv4_paged_sparse_decode(
            torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device=device),
            torch.randn(batch, kv_heads, dim, dtype=torch.bfloat16, device=device),
            layout,
            torch.arange(blocks, dtype=torch.int32, device=device).view(1, blocks),
            torch.tensor([2], dtype=torch.int32, device=device),
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )
        self.assertTrue(torch.isfinite(result.output).all().item())
        self.assertTrue(
            torch.equal(
                result.topk_indices[:, :, 0],
                torch.zeros_like(result.topk_indices[:, :, 0]),
            )
        )
        self.assertTrue((result.topk_indices[:, :, 1:] == -1).all().item())

    def test_rtp_two_region_layout_runs_without_working_pages(self):
        torch.manual_seed(20260923)
        device = torch.device("cuda")
        # MiniMax-M3.1 decode geometry: 64 Q heads / 4 KV and index heads.
        batch, q_heads, kv_heads, dim = 2, 64, 4, 128
        page, blocks_per_request, topk = 128, 20, 16
        blocks = batch * blocks_per_request
        groups = dim // 16
        main_bytes = 2 * kv_heads * page * dim // 2
        side_bytes = 2 * kv_heads * page * groups + page * dim // 2 + page * groups
        base = torch.zeros(blocks, main_bytes, dtype=torch.uint8, device=device)
        side = torch.zeros(blocks, side_bytes, dtype=torch.uint8, device=device)
        layout = cache_layout(base, side, kv_heads, page, dim)

        rows = blocks * page
        slots = torch.arange(rows, dtype=torch.int64, device=device)
        k = torch.randn(rows, kv_heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        idx_k = torch.randn(rows, 1, dim, dtype=torch.bfloat16, device=device)
        quantize_main_index_rows(k, v, idx_k, slots, layout)

        q = torch.randn(batch, q_heads, dim, dtype=torch.bfloat16, device=device)
        idx_q = torch.randn(batch, kv_heads, dim, dtype=torch.bfloat16, device=device)
        block_table = torch.arange(blocks, dtype=torch.int32, device=device).view(
            batch, blocks_per_request
        )
        seq_lens = torch.tensor(
            [blocks_per_request * page - 17, 17 * page + 31],
            dtype=torch.int32,
            device=device,
        )
        result = q8kv4_paged_sparse_decode(
            q,
            idx_q,
            layout,
            block_table,
            seq_lens,
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        )
        self.assertEqual(result.output.dtype, torch.bfloat16)
        self.assertEqual(tuple(result.output.shape), (batch, q_heads, dim))
        self.assertEqual(tuple(result.topk_indices.shape), (kv_heads, batch, topk))
        self.assertTrue(torch.isfinite(result.output).all().item())

        # Reference the same persistent bytes through the established BF16
        # reader, then apply the demo's scale-1 E4M3 compute contract.  This
        # checks RTP byte offsets independently of the native readers.
        k_pages = torch.empty(
            blocks, kv_heads, page, dim, dtype=torch.bfloat16, device=device
        )
        v_pages = torch.empty_like(k_pages)
        idx_pages = torch.empty(
            blocks * page, 1, dim, dtype=torch.bfloat16, device=device
        )
        gather_main_rows(layout, slots, slots, k_pages, v_pages, out_hnd=True)
        gather_index_rows(layout, dim, slots, slots, idx_pages)
        idx_pages = idx_pages.view(blocks, page, dim).to(torch.float8_e4m3fn)
        idx_q8 = idx_q.to(torch.float8_e4m3fn)
        ref_scores = torch.full_like(result.index_scores, float("-inf"))
        for b in range(batch):
            visible = (int(seq_lens[b]) + page - 1) // page
            local_start = max(0, visible - 2)
            for logical_block in range(visible):
                physical = int(block_table[b, logical_block])
                valid = min(page, int(seq_lens[b]) - logical_block * page)
                token_score = torch.einsum(
                    "hd,td->ht",
                    idx_q8[b].float(),
                    idx_pages[physical, :valid].float(),
                )
                ref_scores[:, b, logical_block] = token_score.max(dim=-1).values * (
                    dim**-0.5
                )
                if logical_block >= local_start:
                    ref_scores[:, b, logical_block] = 1.0e29
                elif logical_block < 1:
                    ref_scores[:, b, logical_block] = 1.0e30
        # Equal 1e30/1e29 sentinel scores intentionally leave tie order to the
        # production bitonic TopK.  Selection is the contract; torch.topk's
        # unrelated tie order is not.
        reference_topk = torch.topk(ref_scores, topk, dim=-1).indices.to(torch.int32)
        torch.testing.assert_close(
            result.topk_indices.sort(dim=-1).values,
            reference_topk.sort(dim=-1).values,
            rtol=0,
            atol=0,
        )

        q8 = q.to(torch.float8_e4m3fn)
        k8 = k_pages.to(torch.float8_e4m3fn)
        v8 = v_pages.to(torch.float8_e4m3fn)
        reference = torch.empty_like(result.output)
        for b in range(batch):
            for h in range(q_heads):
                kv_head = h // (q_heads // kv_heads)
                keys, values = [], []
                for logical_block in result.topk_indices[kv_head, b].tolist():
                    physical = int(block_table[b, logical_block])
                    valid = min(
                        page,
                        max(0, int(seq_lens[b]) - logical_block * page),
                    )
                    if valid:
                        keys.append(k8[physical, kv_head, :valid].float())
                        values.append(v8[physical, kv_head, :valid].float())
                key = torch.cat(keys)
                value = torch.cat(values)
                probability = torch.softmax(
                    q8[b, h].float() @ key.T / (dim**0.5), dim=-1
                )
                reference[b, h] = (probability[:, None] * value).sum(0)
        torch.testing.assert_close(
            result.output.float(), reference.float(), rtol=0.03, atol=0.04
        )

        # All temporary tensors are bucket-owned and reused, so the complete
        # Q cast -> score -> top-k -> sparse attention chain is replayable.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = q8kv4_paged_sparse_decode(
                q,
                idx_q,
                layout,
                block_table,
                seq_lens,
                indexer_dim=dim,
                block_size=page,
                topk=topk,
                init_blocks=1,
                local_blocks=2,
                score_type="max",
            )
        graph.replay()
        torch.cuda.synchronize()
        first = captured.output.clone()
        q.normal_()
        idx_q.normal_()
        seq_lens.copy_(torch.tensor([19 * page + 9, 16 * page + 3], device=device))
        graph.replay()
        torch.cuda.synchronize()
        replayed = captured.output.clone()
        expected = q8kv4_paged_sparse_decode(
            q,
            idx_q,
            layout,
            block_table,
            seq_lens,
            indexer_dim=dim,
            block_size=page,
            topk=topk,
            init_blocks=1,
            local_blocks=2,
            score_type="max",
        ).output.clone()
        self.assertFalse(torch.equal(first, replayed))
        torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
