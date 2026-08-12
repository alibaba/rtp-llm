"""Unit tests for the FP8 sparse decode FlashMLA wrapper."""

from __future__ import annotations

import sys
import types
import unittest
from math import sqrt

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


class TestSparseAttnV4DecodeFp8Op(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm120_flashinfer_dual_pool_512_rows(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("FlashInfer DSV4 sparse kernel is SM120-only")

        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        torch.manual_seed(23)
        device = torch.device("cuda")
        rows, heads, dim = 512, 64, 512
        swa_count = 640
        q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
        swa_cache = torch.empty((10, 64, 584), dtype=torch.uint8, device=device)
        quantize_and_insert_k_cache(
            torch.randn(swa_count, dim, dtype=torch.bfloat16, device=device),
            swa_cache,
            torch.arange(swa_count, dtype=torch.int64, device=device),
        )
        swa_indices = torch.arange(
            swa_count - 128, swa_count, dtype=torch.int32, device=device
        ).view(1, -1).expand(rows, -1).contiguous()
        swa_lens = torch.arange(1, rows + 1, dtype=torch.int32, device=device).clamp_max(128)
        swa_indices.masked_fill_(
            torch.arange(128, device=device).view(1, -1) >= swa_lens.view(-1, 1),
            -1,
        )
        sinks = torch.zeros(heads, dtype=torch.float32, device=device)

        for page_size, extra_width in ((2, 4),):
            with self.subTest(page_size=page_size, extra_width=extra_width):
                extra_count = max(1024, extra_width)
                extra_cache = torch.empty(
                    ((extra_count + page_size - 1) // page_size, page_size, 584),
                    dtype=torch.uint8,
                    device=device,
                )
                quantize_and_insert_k_cache(
                    torch.randn(
                        extra_count, dim, dtype=torch.bfloat16, device=device
                    ),
                    extra_cache,
                    torch.arange(extra_count, dtype=torch.int64, device=device),
                )
                extra_indices = torch.arange(
                    extra_count - extra_width,
                    extra_count,
                    dtype=torch.int32,
                    device=device,
                ).view(1, 1, -1).expand(rows, -1, -1).contiguous()
                extra_lens = torch.arange(
                    rows, dtype=torch.int32, device=device
                ).clamp_max(extra_width)
                extra_indices.masked_fill_(
                    torch.arange(extra_width, device=device).view(1, 1, -1)
                    >= extra_lens.view(-1, 1, 1),
                    -1,
                )
                out = torch.empty_like(q)
                trtllm_batch_decode_sparse_mla_dsv4(
                    query=q,
                    swa_kv_cache=swa_cache.unsqueeze(-2),
                    workspace_buffer=SparseAttnV4DecodeFp8Op._get_sm120_workspace(
                        device
                    ),
                    sparse_indices=swa_indices,
                    compressed_kv_cache=extra_cache.unsqueeze(-2),
                    out=out,
                    bmm1_scale=1.0 / sqrt(dim),
                    sinks=sinks,
                    kv_layout="NHD",
                    swa_topk_lens=swa_lens,
                    extra_sparse_indices=extra_indices,
                    extra_sparse_topk_lens=extra_lens,
                )
                torch.cuda.synchronize()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm120_flashinfer_dual_pool_layout(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("FlashInfer DSV4 sparse kernel is SM120-only")

        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            dequantize_slots_to_bf16,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        torch.manual_seed(19)
        device = torch.device("cuda")
        rows, heads, dim = 128, 64, 512
        swa_count, extra_count = 257, 129
        q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
        swa_logical = torch.randn(swa_count, dim, dtype=torch.bfloat16, device=device)
        extra_logical = torch.randn(
            extra_count, dim, dtype=torch.bfloat16, device=device
        )
        swa_cache = torch.empty(
            ((swa_count + 63) // 64, 64, 584), dtype=torch.uint8, device=device
        )
        extra_cache = torch.empty(
            ((extra_count + 1) // 2, 2, 584), dtype=torch.uint8, device=device
        )
        quantize_and_insert_k_cache(
            swa_logical,
            swa_cache,
            torch.arange(swa_count, dtype=torch.int64, device=device),
        )
        quantize_and_insert_k_cache(
            extra_logical,
            extra_cache,
            torch.arange(extra_count, dtype=torch.int64, device=device),
        )

        swa_len, extra_len = 128, 4
        swa_indices = torch.arange(
            swa_count - swa_len, swa_count, dtype=torch.int32, device=device
        ).view(1, -1).expand(rows, -1).contiguous()
        selected_extra = torch.arange(
            extra_count - extra_len,
            extra_count,
            dtype=torch.int32,
            device=device,
        )
        extra_indices = torch.full(
            (rows, 1, extra_len), -1, dtype=torch.int32, device=device
        )
        extra_indices[:, :, :extra_len] = selected_extra.view(1, 1, -1)
        swa_lens = torch.arange(rows, dtype=torch.int32, device=device).clamp_max(
            swa_len
        )
        extra_lens = (
            torch.arange(rows, dtype=torch.int32, device=device) * 17
        ).clamp_max(extra_len)
        swa_cols = torch.arange(swa_len, device=device).view(1, -1)
        swa_indices.masked_fill_(swa_cols >= swa_lens.view(-1, 1), -1)
        extra_cols = torch.arange(extra_len, device=device).view(1, 1, -1)
        extra_indices.masked_fill_(extra_cols >= extra_lens.view(-1, 1, 1), -1)
        sinks = torch.zeros(heads, dtype=torch.float32, device=device)
        scale = 1.0 / sqrt(dim)
        actual = torch.empty_like(q)
        trtllm_batch_decode_sparse_mla_dsv4(
            query=q,
            swa_kv_cache=swa_cache.unsqueeze(-2),
            workspace_buffer=SparseAttnV4DecodeFp8Op._get_sm120_workspace(device),
            sparse_indices=swa_indices,
            compressed_kv_cache=extra_cache.unsqueeze(-2),
            out=actual,
            bmm1_scale=scale,
            sinks=sinks,
            kv_layout="NHD",
            swa_topk_lens=swa_lens,
            extra_sparse_indices=extra_indices,
            extra_sparse_topk_lens=extra_lens,
        )
        for row in (1, rows - 1):
            effective = torch.cat(
                [
                    dequantize_slots_to_bf16(
                        swa_cache, swa_indices[row, : swa_lens[row]]
                    ),
                    dequantize_slots_to_bf16(
                        extra_cache, selected_extra[: extra_lens[row]]
                    ),
                ],
                dim=0,
            )
            scores = torch.einsum("hd,kd->hk", q[row].float(), effective.float()) * scale
            weights = torch.softmax(
                torch.cat((scores, sinks[:, None]), dim=-1), dim=-1
            )[:, :-1]
            expected = torch.einsum("hk,kd->hd", weights, effective.float())
            torch.testing.assert_close(
                actual[row].float(), expected, atol=5e-2, rtol=5e-2
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_packed_flat_repage_is_bit_exact(self):
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            gather_k_cache_packed,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            insert_packed_k_cache_flat,
            quantize_and_insert_k_cache,
        )

        device = torch.device("cuda")
        torch.manual_seed(23)
        rows = 131
        logical = torch.randn(rows, 512, dtype=torch.bfloat16, device=device)
        source = torch.empty((3, 64, 584), dtype=torch.uint8, device=device)
        slots = torch.arange(rows, dtype=torch.int64, device=device)
        quantize_and_insert_k_cache(logical, source, slots)
        flat = torch.empty((1, rows, 584), dtype=torch.uint8, device=device)
        gather_k_cache_packed(
            out=flat,
            k_cache=source,
            seq_lens=torch.tensor([rows], dtype=torch.int32, device=device),
            gather_lens=None,
            block_table=torch.arange(3, dtype=torch.int32, device=device).view(1, -1),
            block_size=64,
            offset=0,
        )
        repaged = torch.empty((66, 2, 584), dtype=torch.uint8, device=device)
        insert_packed_k_cache_flat(flat[0].contiguous(), repaged)
        roundtrip = torch.empty_like(flat)
        gather_k_cache_packed(
            out=roundtrip,
            k_cache=repaged,
            seq_lens=torch.tensor([rows], dtype=torch.int32, device=device),
            gather_lens=None,
            block_table=torch.arange(66, dtype=torch.int32, device=device).view(1, -1),
            block_size=2,
            offset=0,
        )
        self.assertTrue(torch.equal(flat, roundtrip))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm120_flashinfer_groupwise_fp8_large_m(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("FlashInfer groupwise FP8 fallback is SM120-only")

        from flashinfer.gemm import gemm_fp8_nt_groupwise

        torch.manual_seed(11)
        device = torch.device("cuda")
        k = 4096
        n = 1024
        b = torch.randn(n, k, dtype=torch.float32, device=device).to(
            torch.float8_e4m3fn
        )
        b_scale = torch.rand(n // 128, k // 128, device=device) + 0.5
        b_dequant = (
            b.float().view(n // 128, 128, k // 128, 128)
            * b_scale[:, None, :, None]
        ).view(n, k)

        for m in (12, 692):
            a = torch.randn(m, k, dtype=torch.float32, device=device).to(
                torch.float8_e4m3fn
            )
            a_scale = torch.rand(m, k // 128, device=device) + 0.5
            actual = gemm_fp8_nt_groupwise(
                a,
                b,
                a_scale,
                b_scale,
                scale_granularity_mnk=(1, 128, 128),
                scale_major_mode="K",
                out_dtype=torch.bfloat16,
            )
            a_dequant = (
                a.float().view(m, k // 128, 128) * a_scale[:, :, None]
            ).view(m, k)
            expected = a_dequant @ b_dequant.T
            torch.testing.assert_close(
                actual.float(), expected, atol=6.0, rtol=2e-2
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm120_transient_page64_long_logical_workspace(self):
        if torch.cuda.get_device_capability()[0] != 12:
            self.skipTest("FlashInfer DSV4 sparse kernel is SM120-only")

        from flashinfer.decode import trtllm_batch_decode_sparse_mla_dsv4
        from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
            dequantize_slots_to_bf16,
        )
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )

        torch.manual_seed(7)
        device = torch.device("cuda")
        num_kv = 2765
        num_queries = 692  # one CP4 rank of the smoke's long prompt
        num_heads = 64
        head_dim = 512

        logical_kv = torch.randn(
            num_kv, head_dim, dtype=torch.bfloat16, device=device
        )
        page_size = 64
        packed = torch.empty(
            ((num_kv + page_size - 1) // page_size, page_size, 584),
            dtype=torch.uint8,
            device=device,
        )
        slots = torch.arange(num_kv, dtype=torch.int64, device=device)
        quantize_and_insert_k_cache(logical_kv, packed, slots)

        query = torch.randn(
            num_queries, num_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        sinks = torch.zeros(num_heads, dtype=torch.float32, device=device)
        scale = 1.0 / sqrt(head_dim)
        for valid_topk, kernel_topk in ((128, 128), (640, 1024)):
            selected_slots = torch.arange(
                num_kv - valid_topk,
                num_kv,
                dtype=torch.int32,
                device=device,
            )
            indices = torch.full(
                (num_queries, kernel_topk),
                -1,
                dtype=torch.int32,
                device=device,
            )
            indices[:, :valid_topk] = selected_slots
            lengths = torch.full(
                (num_queries,), valid_topk, dtype=torch.int32, device=device
            )
            actual = torch.empty_like(query)
            trtllm_batch_decode_sparse_mla_dsv4(
                query=query,
                swa_kv_cache=packed.unsqueeze(-2),
                workspace_buffer=SparseAttnV4DecodeFp8Op._get_sm120_workspace(device),
                sparse_indices=indices,
                out=actual,
                bmm1_scale=scale,
                sinks=sinks,
                kv_layout="NHD",
                swa_topk_lens=lengths,
            )

            # Compare boundary rows against the effective post-quantization cache.
            effective_kv = dequantize_slots_to_bf16(packed, selected_slots)
            for row in (0, num_queries - 1):
                scores = torch.einsum(
                    "hd,kd->hk", query[row].float(), effective_kv.float()
                ) * scale
                # The learned sink participates in the softmax denominator but has
                # a zero value vector, matching the DSV4 kernel contract.
                weights = torch.softmax(
                    torch.cat((scores, sinks[:, None]), dim=-1), dim=-1
                )[:, :-1]
                expected = torch.einsum("hk,kd->hd", weights, effective_kv.float())
                torch.testing.assert_close(
                    actual[row].float(), expected, atol=5e-2, rtol=5e-2
                )

            # Real prefill metadata is row-dependent: compressed Top-K entries
            # are not contiguous and the causal SWA suffix advances with each
            # query row. Cover that contract instead of validating only the
            # degenerate case where every row reads the same consecutive KVs.
            row_ids = torch.arange(num_queries, dtype=torch.int64, device=device)
            col_ids = torch.arange(valid_topk, dtype=torch.int64, device=device)
            causal_ends = torch.clamp(row_ids + num_kv - num_queries + 1, min=1)
            row_slots = (
                col_ids.unsqueeze(0) * 131 + row_ids.unsqueeze(1) * 17
            ) % causal_ends.unsqueeze(1)
            indices.fill_(-1)
            indices[:, :valid_topk] = row_slots.to(torch.int32)
            trtllm_batch_decode_sparse_mla_dsv4(
                query=query,
                swa_kv_cache=packed.unsqueeze(-2),
                workspace_buffer=SparseAttnV4DecodeFp8Op._get_sm120_workspace(device),
                sparse_indices=indices,
                out=actual,
                bmm1_scale=scale,
                sinks=sinks,
                kv_layout="NHD",
                swa_topk_lens=lengths,
            )
            for row in (0, num_queries - 1):
                selected = row_slots[row].to(torch.int32)
                effective_kv = dequantize_slots_to_bf16(packed, selected)
                scores = torch.einsum(
                    "hd,kd->hk", query[row].float(), effective_kv.float()
                ) * scale
                weights = torch.softmax(
                    torch.cat((scores, sinks[:, None]), dim=-1), dim=-1
                )[:, :-1]
                expected = torch.einsum("hk,kd->hd", weights, effective_kv.float())
                torch.testing.assert_close(
                    actual[row].float(), expected, atol=5e-2, rtol=5e-2
                )

    def test_sparse_indices_drop_dense_cache_metadata(self):
        calls = []
        fake_flash_mla = types.ModuleType("flash_mla")

        def fake_flash_mla_with_kvcache(**kwargs):
            calls.append(kwargs)
            q = kwargs["q"]
            head_dim_v = kwargs["head_dim_v"]
            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                head_dim_v,
                dtype=q.dtype,
                device=q.device,
            )
            lse = torch.zeros(
                q.shape[0],
                q.shape[2],
                q.shape[1],
                dtype=torch.float32,
                device=q.device,
            )
            return out, lse

        fake_flash_mla.flash_mla_with_kvcache = fake_flash_mla_with_kvcache
        old_flash_mla = sys.modules.get("flash_mla")
        sys.modules["flash_mla"] = fake_flash_mla
        try:
            op = SparseAttnV4DecodeFp8Op(
                n_heads=4,
                head_dim=512,
                softmax_scale=1.0,
            )
            q = torch.zeros(2, 3, 4, 512, dtype=torch.bfloat16)
            kv_cache = torch.zeros(8, 256, 584, dtype=torch.uint8)
            attn_sink = torch.zeros(4, dtype=torch.float32)
            topk = torch.arange(128, dtype=torch.int32).view(1, 1, 128).expand(
                2, 3, 128
            )
            block_table = torch.full((2, 257), -1, dtype=torch.int32)
            cache_seqlens = torch.tensor([65537, 65537], dtype=torch.int32)

            out = op._forward_flash_mla(
                q=q,
                kv_cache=kv_cache,
                attn_sink=attn_sink,
                topk_idxs=topk,
                sched_meta=object(),
                cache_seqlens=cache_seqlens,
                block_table=block_table,
            )

            self.assertEqual(tuple(out.shape), (2, 3, 4, 512))
            self.assertEqual(len(calls), 1)
            self.assertIsNone(calls[0]["block_table"])
            self.assertIsNone(calls[0]["cache_seqlens"])
            self.assertTrue(torch.equal(calls[0]["indices"], topk))
        finally:
            if old_flash_mla is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_flash_mla


if __name__ == "__main__":
    unittest.main()
