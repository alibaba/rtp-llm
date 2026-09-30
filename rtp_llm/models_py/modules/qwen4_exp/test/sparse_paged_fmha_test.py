import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import (
    _validate_sparse_paged_indices,
    sparse_paged_gqa_attn,
)

_DEVICE = "cuda"


class SparsePagedIndexValidationTest(unittest.TestCase):
    def test_valid_indices_use_one_scalar_read(self):
        table = torch.tensor([[1, 2]], dtype=torch.int32)
        lengths = torch.tensor([[5]], dtype=torch.int32)
        selected = torch.tensor([[[0, 4, -1]]], dtype=torch.int32)
        original_item = torch.Tensor.item
        calls = []

        def count_item(tensor, *args, **kwargs):
            calls.append(1)
            return original_item(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "item", count_item):
            _validate_sparse_paged_indices(
                table, lengths, selected, page_size=4, cache_blocks=3
            )
        self.assertEqual(len(calls), 1)

    def test_error_priority_and_safe_lookup_for_invalid_indices(self):
        table = torch.tensor([[1, 0]], dtype=torch.int32)
        lengths = torch.tensor([[5]], dtype=torch.int32)
        cases = (
            (torch.tensor([[9]], dtype=torch.int32), [[[0]]], "capacity"),
            (lengths, [[[-2]]], "only use -1"),
            (lengths, [[[5]]], "outside its row's kv_len"),
            (lengths, [[[1000000]]], "outside its row's kv_len"),
            (lengths, [[[4]]], "unallocated"),
        )
        for kv_lens, indices, message in cases:
            with self.subTest(message=message, indices=indices):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_sparse_paged_indices(
                        table,
                        kv_lens,
                        torch.tensor(indices, dtype=torch.int32),
                        page_size=4,
                        cache_blocks=3,
                    )
        with self.assertRaisesRegex(ValueError, "exceeds cache blocks"):
            _validate_sparse_paged_indices(
                torch.tensor([[1, 3]], dtype=torch.int32),
                lengths,
                torch.tensor([[[0]]], dtype=torch.int32),
                page_size=4,
                cache_blocks=3,
            )


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

    def test_cuda_graph_replay_uses_updated_page_table_lengths_and_selection(self):
        torch.manual_seed(23)
        q = torch.randn(1, 2, 1, 64, dtype=torch.bfloat16, device=_DEVICE)
        cache = torch.randn(3, 2, 1, 4, 64, dtype=torch.bfloat16, device=_DEVICE)
        table = torch.tensor([[1, 2]], dtype=torch.int32, device=_DEVICE)
        lengths = torch.tensor([[5]], dtype=torch.int32, device=_DEVICE)
        selected = torch.tensor([[[0, 1, 4, -1]]], dtype=torch.int32, device=_DEVICE)

        with self.assertRaisesRegex(RuntimeError, "active capture"):
            sparse_paged_gqa_attn(
                q, cache, table, lengths, selected, page_size=4, graph_capture=True
            )
        # Warm the Triton specialization and validate the initial page mapping.
        eager = sparse_paged_gqa_attn(q, cache, table, lengths, selected, page_size=4)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = sparse_paged_gqa_attn(
                q, cache, table, lengths, selected, page_size=4, graph_capture=True
            )

        graph.replay()
        torch.testing.assert_close(output, eager, atol=2e-2, rtol=2e-2)
        table.copy_(torch.tensor([[2, 1]], dtype=torch.int32, device=_DEVICE))
        lengths.copy_(torch.tensor([[7]], dtype=torch.int32, device=_DEVICE))
        selected.copy_(
            torch.tensor([[[0, 4, 6, -1]]], dtype=torch.int32, device=_DEVICE)
        )
        graph.replay()
        torch.testing.assert_close(
            output, _reference(q, cache, table, lengths, selected), atol=2e-2, rtol=2e-2
        )
        self.assertFalse(torch.equal(output, eager))

        # A stale selection must never dereference a freed or reserved page
        # after the scheduler refreshes a graph-owned device block table.
        selected.copy_(
            torch.tensor([[[4, -1, -1, -1]]], dtype=torch.int32, device=_DEVICE)
        )
        for physical in (0, cache.shape[0] + 7):
            table[0, 1] = physical
            graph.replay()
            torch.testing.assert_close(output, torch.zeros_like(output))
        table[0, 1] = 1
        selected[0, 0, 0] = 1000000
        graph.replay()
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_cuda_graph_replay_ragged_batch_keeps_padding_inert(self):
        torch.manual_seed(37)
        q = torch.randn(2, 2, 3, 64, dtype=torch.bfloat16, device=_DEVICE)
        cache = torch.randn(7, 2, 1, 4, 64, dtype=torch.bfloat16, device=_DEVICE)
        table = torch.tensor(
            [[1, 2, 3], [4, 5, 6]], dtype=torch.int32, device=_DEVICE
        )
        lengths = torch.tensor(
            [[4, 5, 6], [8, 9, 0]], dtype=torch.int32, device=_DEVICE
        )
        selected = torch.tensor(
            [
                [[0, 1, 2, 3], [0, 2, 3, 4], [0, 1, 4, 5]],
                [[0, 3, 4, 7], [0, 2, 5, 8], [-1, -1, -1, -1]],
            ],
            dtype=torch.int32,
            device=_DEVICE,
        )
        initial = sparse_paged_gqa_attn(q, cache, table, lengths, selected, page_size=4)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = sparse_paged_gqa_attn(
                q, cache, table, lengths, selected, page_size=4, graph_capture=True
            )

        graph.replay()
        torch.testing.assert_close(output, initial, atol=2e-2, rtol=2e-2)
        table.copy_(
            torch.tensor([[3, 2, 1], [6, 5, 4]], dtype=torch.int32, device=_DEVICE)
        )
        lengths.copy_(
            torch.tensor([[3, 5, 6], [7, 9, 0]], dtype=torch.int32, device=_DEVICE)
        )
        selected.copy_(
            torch.tensor(
                [
                    [[0, 1, 2, -1], [0, 2, 3, 4], [0, 2, 4, 5]],
                    [[0, 2, 4, 6], [0, 3, 5, 8], [0, 1, 2, 3]],
                ],
                dtype=torch.int32,
                device=_DEVICE,
            )
        )
        graph.replay()
        torch.testing.assert_close(
            output,
            _reference(q, cache, table, lengths, selected),
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(output[1, :, 2], torch.zeros_like(output[1, :, 2]))
        self.assertFalse(torch.equal(output, initial))


if __name__ == "__main__":
    unittest.main()
