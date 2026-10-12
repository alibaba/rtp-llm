"""FP8 QSA cache layout, independent attention reference and graph replay."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.sparse_gqa_impl import SparseGqaFmhaImpl
from rtp_llm.models_py.modules.qwen4_exp.sparse_paged_fmha import (
    main_kv_scale_view,
    sparse_paged_gqa_attn,
)
from rtp_llm.models_py.modules.qwen4_exp.test.sparse_paged_fmha_test import _reference


class Fp8MainKvTest(unittest.TestCase):
    def _fixture(self):
        torch.manual_seed(211)
        q = torch.randn(2, 6, 4, 256, device="cuda", dtype=torch.bfloat16)
        source = torch.randn(7, 2, 2, 128, 256, device="cuda")
        scale = torch.rand(7, 2, 2, 128, device="cuda") * 0.4 + 0.1
        quant = (source / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
        table = torch.tensor([[3, 1], [4, 6]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor(
            [[127, 128, 129, 130], [129, 130, 131, 132]],
            device="cuda",
            dtype=torch.int32,
        )
        selected = torch.tensor(
            [0, 7, 126, 127, 128, 129, 130, 131, -1], device="cuda", dtype=torch.int32
        ).repeat(2, 4, 1)
        selected = torch.where(selected < lengths.unsqueeze(-1), selected, -1)
        return q, quant, scale, table, lengths, selected

    def test_scaled_cross_page_attention_and_packed_layout(self):
        q, quant, scale, table, lengths, selected = self._fixture()
        expected = _reference(
            q, quant.float() * scale.unsqueeze(-1), table, lengths, selected
        )
        packed = torch.empty(7, quant[0].numel() + 97, device="cuda", dtype=quant.dtype)
        packed[:, : quant[0].numel()].copy_(quant.reshape(7, -1))
        packed_scale = torch.empty(7, scale[0].numel() + 13, device="cuda")
        packed_scale[:, : scale[0].numel()].copy_(scale.reshape(7, -1))
        for online in ("0", "1"):
            for cache, scales in ((quant, scale), (packed, packed_scale)):
                with self.subTest(online=online, packed=cache.ndim == 2), patch.dict(
                    os.environ,
                    {
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": online,
                        "RTP_LLM_QWEN4_PAGED_ATTENTION_SPLITS": "8",
                    },
                ):
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
        with patch.dict(os.environ, {"RTP_LLM_QWEN4_PAGED_ATTENTION_ONLINE": "1"}):
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

    def test_native_prefill_and_decode_writers_with_graph_replay(self):
        from rtp_llm.ops.fused_rope_kvcache_op import _get_fused_rope_kvcache

        op = _get_fused_rope_kvcache()
        q_heads, kv_heads, dim, page_size = 6, 2, 256, 128
        width = (q_heads + 2 * kv_heads) * dim
        cache = torch.zeros(5, 2, kv_heads, page_size, dim, device="cuda").to(
            torch.float8_e4m3fn
        )
        scales = torch.full((5, 2, kv_heads, page_size), float("nan"), device="cuda")
        table = torch.tensor([[2], [4]], device="cuda", dtype=torch.int32)
        offsets = op.convert_offset_to_block_array(table)
        qkv = torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
        cu = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
        padding = torch.zeros(5, device="cuda", dtype=torch.int32)
        op.prefill_fused_rope_kvcache(
            qkv,
            cu,
            2,
            3,
            q_heads,
            kv_heads,
            dim,
            tokens_per_block=page_size,
            store_qkv=True,
            store_cache=True,
            kv_cache=cache,
            kv_cache_scale=scales,
            kv_cache_offset=offsets,
            padding_offset=padding,
            rope_style=0,
            rope_dim=0,
        )
        for batch, start, length, page in ((0, 0, 3, 2), (1, 3, 2, 4)):
            for which in (0, 1):
                expected = (
                    qkv[
                        start : start + length,
                        (q_heads + which * kv_heads)
                        * dim : (q_heads + (which + 1) * kv_heads)
                        * dim,
                    ]
                    .reshape(length, kv_heads, dim)
                    .transpose(0, 1)
                )
                torch.testing.assert_close(
                    cache[page, which, :, :length].float(),
                    expected.to(cache.dtype).float(),
                    atol=0,
                    rtol=0,
                )
                torch.testing.assert_close(
                    scales[page, which, :, :length],
                    torch.ones_like(scales[page, which, :, :length]),
                    atol=0,
                    rtol=0,
                )
        decode_qkv = torch.randn(2, width, device="cuda", dtype=torch.bfloat16)
        lengths = torch.tensor([3, 2], device="cuda", dtype=torch.int32)

        def decode():
            return op.decode_fused_rope_kvcache(
                decode_qkv,
                None,
                lengths,
                2,
                q_heads,
                kv_heads,
                dim,
                cache,
                offsets,
                tokens_per_block=page_size,
                kv_cache_scale=scales,
                rope_style=0,
                rope_dim=0,
            )

        decode()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = decode()
        for _ in range(2):
            decode_qkv.mul_(0.75)
            graph.replay()
            torch.testing.assert_close(
                output.float().reshape(2, -1),
                decode_qkv[:, : q_heads * dim].float(),
                atol=0,
                rtol=0,
            )
            for row, page in ((0, 2), (1, 4)):
                for which in (0, 1):
                    expected = decode_qkv[
                        row,
                        (q_heads + which * kv_heads)
                        * dim : (q_heads + (which + 1) * kv_heads)
                        * dim,
                    ].reshape(kv_heads, dim)
                    torch.testing.assert_close(
                        cache[page, which, :, int(lengths[row])].float(),
                        expected.to(cache.dtype).float(),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        scales[page, which, :, int(lengths[row])],
                        torch.ones(kv_heads, device="cuda"),
                        atol=0,
                        rtol=0,
                    )

    def test_invalid_scale_rejected_before_any_cache_write(self):
        _, quant, scale, _, _, _ = self._fixture()
        impl = SparseGqaFmhaImpl.__new__(SparseGqaFmhaImpl)
        impl.kv_head_num = 2
        impl.attn_configs = SimpleNamespace(kernel_tokens_per_block=128)
        for invalid in (
            None,
            scale.half(),
            scale.cpu(),
            scale[:, :, :, :-1],
            scale.transpose(1, 2),
        ):
            with self.subTest(invalid=None if invalid is None else invalid.shape):
                with self.assertRaisesRegex(ValueError, "scale"):
                    impl._paged_kv_scale_args(
                        SimpleNamespace(kv_cache_base=quant, kv_scale_base=invalid)
                    )
        self.assertIsNone(
            main_kv_scale_view(quant.bfloat16(), None, kv_heads=2, page_size=128)
        )


if __name__ == "__main__":
    unittest.main()
