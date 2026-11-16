import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.qsa.indexer_paged_score import (
    qsa_paged_indexer_score,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_indexer import (
    try_flashinfer_decode_selection,
    try_flashinfer_prefill_selection,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_prefill import (
    try_flashinfer_cold_prefill,
)


class QsaFlashInferTest(unittest.TestCase):
    def test_indexer_selects_segment_local_causal_blocks_and_tail(self):
        tokens = 2055  # 513 complete blocks plus a partial tail.
        q = torch.zeros((1, tokens, 8, 128), device="cuda", dtype=torch.bfloat16)
        q[..., 0], q[..., 1] = 128, 1
        keys = q.new_zeros((1, tokens // 4, 128))
        blocks = torch.arange(tokens // 4, device=q.device)
        keys[0, :, 0], keys[0, :, 1] = blocks // 128, blocks % 128

        with patch.dict("os.environ", {"QWEN4_QSA_INDEXER_CHUNK_SIZE": "1024"}):
            actual = try_flashinfer_prefill_selection(q, keys)[0]
        with patch.dict("os.environ", {"QWEN4_QSA_INDEXER_CHUNK_SIZE": "0"}):
            unchunked = try_flashinfer_prefill_selection(q, keys)[0]
        torch.testing.assert_close(
            actual.sort(dim=-1).values, unchunked.sort(dim=-1).values
        )
        for length in (1, 3, 4, 2049, 2055):
            complete, tail = divmod(length, 4)
            first = 4 * max(0, complete - 512)
            expected = torch.arange(first, length, device=q.device, dtype=torch.int32)
            row = actual[length - 1]
            torch.testing.assert_close(row[row >= 0].sort().values, expected)
            torch.testing.assert_close(
                row[2048 : 2048 + tail], expected[expected.numel() - tail :]
            )

    def test_paged_indexer_replays_with_live_lengths_and_missing_pages(self):
        rows, pages = 4, 257
        q = torch.zeros((rows, 8, 128), device="cuda", dtype=torch.bfloat16)
        q[..., 0], q[..., 1] = 128, 1
        cache = q.new_zeros((pages + 1, 32, 128))
        blocks = torch.arange(pages * 32, device=q.device).view(pages, 32)
        cache[1:, :, 0], cache[1:, :, 1] = blocks // 128, blocks % 128
        cache[0].fill_(10000)  # Reserved page must never win TopK.
        table = torch.arange(1, pages + 1, device=q.device, dtype=torch.int32)
        table = table.repeat(rows, 1)
        lengths = torch.tensor(
            [32768, 32769, 2055, 0], device=q.device, dtype=torch.int32
        )
        weights = torch.full((rows, 8), 128**-0.5, device=q.device)

        def reference():
            scores = qsa_paged_indexer_score(
                q.unsqueeze(1),
                weights,
                cache.flatten(0, 1),
                table,
                lengths[:, None] // 4,
                block_size=32,
                max_ctx_len=pages * 32,
                validate_block_table=False,
            )
            return (
                select_qsa_paged_tokens(
                    scores,
                    lengths,
                    compress_ratio=4,
                    token_budget=2048,
                    validate_lengths=False,
                    scores_are_masked=True,
                )
                .sort(dim=-1)
                .values
            )

        eager = try_flashinfer_decode_selection(q, cache, table, lengths)
        torch.testing.assert_close(eager.sort(dim=-1).values, reference())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = try_flashinfer_decode_selection(q, cache, table, lengths)
        lengths.copy_(torch.tensor([32761, 2055, 131, 3], dtype=torch.int32))
        table[0, 255] = 0
        table[1, 0] = -1
        table[2, 0] = cache.shape[0]
        graph.replay()
        torch.testing.assert_close(captured.sort(dim=-1).values, reference())

    def test_cold_prefill_matches_ragged_grouped_attention(self):
        torch.manual_seed(42)
        lengths = [3, 5]
        heads, kv_heads, dim = 8, 2, 256
        packed = torch.randn(
            (sum(lengths), (heads + 2 * kv_heads) * dim),
            dtype=torch.bfloat16,
            device="cuda",
        )
        q, k, v = packed.split((heads * dim, kv_heads * dim, kv_heads * dim), dim=-1)
        selected = torch.tensor(
            [[0, -1], [1, 0], [0, 2], [0, -1], [0, 1], [2, -1], [1, 3], [0, 4]],
            device="cuda",
            dtype=torch.int32,
        )
        impl = SimpleNamespace(
            _is_mtp_draft=False,
            attn_inputs=SimpleNamespace(is_cuda_graph=False),
            head_dim=dim,
            head_num=heads,
            kv_head_num=kv_heads,
            q_width=heads * dim,
        )
        actual = try_flashinfer_cold_prefill(
            impl, q, k, v, selected, lengths, trusted_selected=True
        )
        self.assertIsNotNone(actual)
        expected = []
        offset = 0
        group = torch.arange(heads, device="cuda") // (heads // kv_heads)
        for length in lengths:
            end = offset + length
            query = q[offset:end].reshape(length, heads, dim).float()
            keys = k[offset:end].reshape(length, kv_heads, dim).float()[:, group]
            values = v[offset:end].reshape(length, kv_heads, dim).float()[:, group]
            ids = selected[offset:end]
            mask = (ids[:, :, None] == torch.arange(length, device="cuda")).any(dim=1)
            logits = torch.einsum("qhd,khd->qhk", query, keys) / dim**0.5
            weights = logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(dim=-1)
            expected.append(torch.einsum("qhk,khd->qhd", weights, values))
            offset = end
        expected = torch.cat(expected).reshape_as(actual).to(actual.dtype)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.01)


if __name__ == "__main__":
    unittest.main()
