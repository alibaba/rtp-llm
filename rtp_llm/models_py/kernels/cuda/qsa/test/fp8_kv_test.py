"""FP8 QSA cache layout, independent attention reference and graph replay."""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qsa.sparse_paged_fmha import sparse_paged_gqa_attn
from rtp_llm.models_py.kernels.cuda.qsa.test.fp8_kv_fixture import Fp8MainKvTestFixture
from rtp_llm.models_py.kernels.cuda.qsa.test.reference.sparse_paged import _reference


class Fp8MainKvTest(Fp8MainKvTestFixture, unittest.TestCase):

    def test_scaled_cross_page_attention_and_packed_layout(self):
        q, quant, scale, table, lengths, selected = self._fixture()
        expected = _reference(
            q, quant.float() * scale.unsqueeze(-1), table, lengths, selected
        )
        packed = torch.empty(7, quant[0].numel() + 97, device="cuda", dtype=quant.dtype)
        packed[:, : quant[0].numel()].copy_(quant.reshape(7, -1))
        packed_scale = torch.empty(7, scale[0].numel() + 13, device="cuda")
        packed_scale[:, : scale[0].numel()].copy_(scale.reshape(7, -1))
        for cache, scales in ((quant, scale), (packed, packed_scale)):
            with self.subTest(packed=cache.ndim == 2):
                actual = sparse_paged_gqa_attn(
                    q,
                    cache,
                    table,
                    lengths,
                    selected,
                    page_size=128,
                    kv_head_num=2,
                    kv_scale=scales,
                )
                torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_graph_reads_updated_scales_lengths_and_pages(self):
        q, quant, scale, table, lengths, selected = self._fixture()
        for _ in range(2):
            sparse_paged_gqa_attn(
                q, quant, table, lengths, selected, page_size=128, kv_scale=scale
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse_paged_gqa_attn(
                q,
                quant,
                table,
                lengths,
                selected,
                page_size=128,
                kv_scale=scale,
                graph_capture=True,
            )
        for factor in (0.5, 2.0):
            q.mul_(0.75)
            scale.mul_(factor)
            table.copy_(table.flip(1))
            graph.replay()
            expected = _reference(
                q, quant.float() * scale.unsqueeze(-1), table, lengths, selected
            )
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        lengths.zero_()
        graph.replay()
        torch.testing.assert_close(actual, torch.zeros_like(actual))


if __name__ == "__main__":
    unittest.main()
