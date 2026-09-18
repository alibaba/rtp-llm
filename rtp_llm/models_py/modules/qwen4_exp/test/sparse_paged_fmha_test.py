import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import sparse_paged_gqa_attn

_DEVICE = "cuda"


def _reference(q, cache, block_table, kv_lens, selected):
    batch, q_heads, query_len, head_dim = q.shape
    page_size = cache.shape[3]
    kv_heads = cache.shape[2]
    out = torch.zeros_like(q)
    for b in range(batch):
        for h in range(q_heads):
            kv_head = h // (q_heads // kv_heads)
            for s in range(query_len):
                indices = selected[b, s]
                indices = indices[(indices >= 0) & (indices < kv_lens[b, s])]
                if not indices.numel():
                    continue
                keys, values = [], []
                for index in indices.tolist():
                    page = int(block_table[b, index // page_size])
                    offset = index % page_size
                    keys.append(cache[page, 0, kv_head, offset])
                    values.append(cache[page, 1, kv_head, offset])
                keys = torch.stack(keys).float()
                values = torch.stack(values).float()
                score = (q[b, h, s].float() @ keys.T) / head_dim**0.5
                out[b, h, s] = (score.softmax(dim=-1) @ values).to(out.dtype)
    return out


class SparsePagedGqaAttentionTest(unittest.TestCase):
    def test_ragged_multi_token_decode_matches_reference(self):
        torch.manual_seed(13)
        batch, q_heads, kv_heads, query_len, head_dim = 2, 6, 2, 3, 64
        page_size, cache_blocks = 4, 7
        q = torch.randn(
            batch,
            q_heads,
            query_len,
            head_dim,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        cache = torch.randn(
            cache_blocks,
            2,
            kv_heads,
            page_size,
            head_dim,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        block_table = torch.tensor(
            [[1, 4, 5], [2, 3, 6]], dtype=torch.int32, device=_DEVICE
        )
        kv_lens = torch.tensor(
            [[5, 6, 7], [8, 9, 10]], dtype=torch.int32, device=_DEVICE
        )
        selected = torch.full(
            (batch, query_len, 7), -1, dtype=torch.int32, device=_DEVICE
        )
        selected[0, 0, :4] = torch.tensor([0, 1, 3, 4], device=_DEVICE)
        selected[0, 1, :5] = torch.tensor([0, 2, 3, 4, 5], device=_DEVICE)
        selected[0, 2, :5] = torch.tensor([0, 1, 4, 5, 6], device=_DEVICE)
        selected[1, 0, :5] = torch.tensor([0, 3, 4, 6, 7], device=_DEVICE)
        selected[1, 1, :6] = torch.tensor([0, 2, 4, 5, 7, 8], device=_DEVICE)
        selected[1, 2, :6] = torch.tensor([0, 1, 3, 6, 8, 9], device=_DEVICE)

        got = sparse_paged_gqa_attn(
            q,
            cache,
            block_table,
            kv_lens,
            selected,
            page_size=page_size,
            kv_head_num=kv_heads,
        )
        expected = _reference(q, cache, block_table, kv_lens, selected)

        torch.testing.assert_close(got, expected, atol=2e-2, rtol=2e-2)

    def test_all_padding_selection_returns_zero(self):
        q = torch.randn(1, 2, 1, 64, device=_DEVICE, dtype=torch.bfloat16)
        cache = torch.randn(2, 2, 1, 4, 64, device=_DEVICE, dtype=torch.bfloat16)
        block_table = torch.tensor([[1]], dtype=torch.int32, device=_DEVICE)
        kv_lens = torch.tensor([[4]], dtype=torch.int32, device=_DEVICE)
        selected = torch.full((1, 1, 7), -1, dtype=torch.int32, device=_DEVICE)

        got = sparse_paged_gqa_attn(
            q, cache, block_table, kv_lens, selected, page_size=4
        )

        torch.testing.assert_close(got, torch.zeros_like(got))

    def test_packed_hybrid_pool_view_matches_5d_cache(self):
        torch.manual_seed(17)
        batch, q_heads, kv_heads, head_dim = 1, 4, 2, 64
        page_size, cache_blocks = 4, 3
        q = torch.randn(
            batch, q_heads, 1, head_dim, device=_DEVICE, dtype=torch.bfloat16
        )
        cache = torch.randn(
            cache_blocks,
            2,
            kv_heads,
            page_size,
            head_dim,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        block_table = torch.tensor([[1, 2]], dtype=torch.int32, device=_DEVICE)
        kv_lens = torch.tensor([[6]], dtype=torch.int32, device=_DEVICE)
        selected = torch.tensor([[[0, 2, 4, 5]]], dtype=torch.int32, device=_DEVICE)

        expected = sparse_paged_gqa_attn(
            q, cache, block_table, kv_lens, selected, page_size=page_size
        )
        required_width = cache[0].numel()
        packed = torch.zeros(
            cache_blocks,
            required_width + 37,
            device=_DEVICE,
            dtype=torch.bfloat16,
        )
        packed[:, :required_width].copy_(cache.view(cache_blocks, -1))
        got = sparse_paged_gqa_attn(
            q,
            packed,
            block_table,
            kv_lens,
            selected,
            page_size=page_size,
            kv_head_num=kv_heads,
        )

        torch.testing.assert_close(got, expected)

    def test_rejects_invalid_selected_metadata_before_launch(self):
        q = torch.zeros(1, 2, 1, 64, device=_DEVICE, dtype=torch.bfloat16)
        cache = torch.zeros(3, 2, 1, 4, 64, device=_DEVICE, dtype=torch.bfloat16)
        block_table = torch.tensor([[1, 0]], dtype=torch.int32, device=_DEVICE)
        kv_lens = torch.tensor([[5]], dtype=torch.int32, device=_DEVICE)

        with self.assertRaisesRegex(ValueError, "unallocated"):
            sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                torch.tensor([[[4]]], dtype=torch.int32, device=_DEVICE),
                page_size=4,
            )
        with self.assertRaisesRegex(ValueError, "outside.*kv_len"):
            sparse_paged_gqa_attn(
                q,
                cache,
                block_table,
                kv_lens,
                torch.tensor([[[5]]], dtype=torch.int32, device=_DEVICE),
                page_size=4,
            )


if __name__ == "__main__":
    unittest.main()
