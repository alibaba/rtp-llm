import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_indexer import (
    try_flashinfer_prefill_selection,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.flashinfer_prefill import (
    try_flashinfer_cold_prefill,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import Qwen4ExpQSAIndexer


class QsaFlashInferTest(unittest.TestCase):
    def test_indexer_selects_segment_local_causal_blocks_and_tail(self):
        q = torch.ones((2, 8, 8, 128), device="cuda", dtype=torch.bfloat16)
        q[:, :, :4].neg_()
        # Both requests yield distinct positive scores, with reversed order.
        keys = torch.tensor([[1, 2, 3], [3, 2, 1]], device="cuda", dtype=q.dtype)
        k = keys.unsqueeze(-1).expand(-1, -1, 128).contiguous()
        indexer = SimpleNamespace(
            compress_ratio=4,
            token_budget=8,
            block_topk=2,
            max_selected=11,
        )
        indexer._select_causal_scores = (
            Qwen4ExpQSAIndexer._select_causal_scores.__get__(indexer)
        )
        positions = torch.arange(4, 12, device="cuda")
        actual = try_flashinfer_prefill_selection(indexer, q, k, positions, 512)
        self.assertIsNotNone(actual)

        expected = []
        for key_scores in ([1, 2, 3], [3, 2, 1]):
            rows = []
            for position in range(4, 12):
                complete, tail = divmod(position + 1, 4)
                best = sorted(range(complete), key=key_scores.__getitem__)[-2:]
                tokens = [
                    token for block in best for token in range(4 * block, 4 * block + 4)
                ]
                rows.append(
                    tokens
                    + [-1] * (8 - len(tokens))
                    + list(range(4 * complete, position + 1))
                    + [-1] * (3 - tail)
                )
            expected.append(rows)

        expected = torch.tensor(expected, device="cuda", dtype=torch.int32)
        torch.testing.assert_close(
            actual.sort(dim=-1).values, expected.sort(dim=-1).values
        )

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
