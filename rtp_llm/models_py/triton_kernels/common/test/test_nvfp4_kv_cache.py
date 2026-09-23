import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    cache_layout,
    clear_working_tails,
    convert_active_pages,
    gather_index_rows,
    gather_main_rows,
    materialize_q8kv4_rows,
    quantize_index_rows,
    quantize_main_index_rows,
    quantize_main_index_rows_to_planes,
    quantize_main_rows,
    quantize_query_rows_mma,
    reference_dequantize,
    reference_quantize,
    round_to_e4m3_compute_grid_,
    scatter_main_rows_to_hnd,
)


def _reference_groups(values: torch.Tensor) -> torch.Tensor:
    groups = values.float().reshape(*values.shape[:-1], -1, 16)
    packed, scales = reference_quantize(groups)
    return reference_dequantize(packed, scales).reshape(values.shape).to(torch.bfloat16)


class TestNVFP4KVCache(unittest.TestCase):
    def test_layout_rejects_short_or_mismatched_sidecar(self):
        blocks, heads, page, dim, index_dim = 2, 2, 4, 32, 64
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16
        base = torch.zeros((blocks, main_bytes), dtype=torch.uint8)

        with self.assertRaisesRegex(RuntimeError, "block counts differ"):
            cache_layout(
                base,
                torch.zeros((blocks - 1, main_scale_bytes), dtype=torch.uint8),
                heads,
                page,
                dim,
            )

        # Main-scale validation happens when the layout is created; indexer
        # payload validation happens only once the logical indexer view is
        # requested. Both must fail before a reader can touch an incomplete PD
        # sidecar.
        with self.assertRaisesRegex(RuntimeError, "scale block stride is too small"):
            cache_layout(
                base,
                torch.zeros((blocks, main_scale_bytes - 1), dtype=torch.uint8),
                heads,
                page,
                dim,
            )
        layout = cache_layout(
            base,
            torch.zeros(
                (
                    blocks,
                    main_scale_bytes + index_bytes + index_scale_bytes - 1,
                ),
                dtype=torch.uint8,
            ),
            heads,
            page,
            dim,
        )
        with self.assertRaisesRegex(RuntimeError, "indexer side region is too small"):
            layout.indexer(index_dim)

    def test_rne_boundaries_and_nibble_order(self):
        values = torch.tensor(
            [
                [
                    -6.0,
                    -5.0,
                    -3.5,
                    -2.5,
                    -1.75,
                    -1.25,
                    -0.75,
                    -0.25,
                    0.0,
                    0.25,
                    0.75,
                    1.25,
                    1.75,
                    2.5,
                    3.5,
                    5.0,
                ]
            ]
        )
        packed, scale = reference_quantize(values)
        self.assertEqual(scale.float().tolist(), [1.0])
        # Low-dimension code is the low nibble.  Exact ties exercise the RNE
        # strict/inclusive boundaries specified for E2M1.
        self.assertEqual(
            packed.tolist(), [[0xEF, 0xCE, 0xAC, 0x0A, 0x00, 0x22, 0x44, 0x66]]
        )
        restored = reference_dequantize(packed, scale)
        self.assertEqual(
            restored.tolist(),
            [
                [
                    -6.0,
                    -4.0,
                    -4.0,
                    -2.0,
                    -2.0,
                    -1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    4.0,
                ]
            ],
        )

    def test_triton_main_and_index_round_trip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(7)
        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 4, 2, 4, 32, 64
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16
        base = torch.zeros((blocks, main_bytes), dtype=torch.uint8, device=device)
        side = torch.zeros(
            (blocks, main_scale_bytes + index_bytes + index_scale_bytes),
            dtype=torch.uint8,
            device=device,
        )
        layout = cache_layout(base, side, heads, page, dim)

        rows = 5
        physical_slots = torch.tensor([4, 5, 7, 8, 11], device=device)
        destination_slots = torch.arange(rows, device=device)
        k = (torch.randn(rows, heads, dim, device=device) * 2).to(torch.bfloat16)
        v = (torch.randn(rows, heads, dim, device=device) * 3).to(torch.bfloat16)
        idx = (torch.randn(rows, index_dim, device=device) * 4).to(torch.bfloat16)

        quantize_main_rows(k, v, physical_slots, layout)
        quantize_index_rows(idx, physical_slots, layout)
        out_k = torch.empty_like(k)
        out_v = torch.empty_like(v)
        out_idx = torch.empty(rows, 1, index_dim, dtype=torch.bfloat16, device=device)
        gather_main_rows(
            layout,
            physical_slots,
            destination_slots,
            out_k,
            out_v,
        )
        gather_index_rows(
            layout,
            index_dim,
            physical_slots,
            destination_slots,
            out_idx,
        )
        torch.testing.assert_close(out_k, _reference_groups(k), rtol=0, atol=0)
        torch.testing.assert_close(out_v, _reference_groups(v), rtol=0, atol=0)
        torch.testing.assert_close(
            out_idx[:, 0], _reference_groups(idx), rtol=0, atol=0
        )

        # Compile and validate the page-level dense-attention adapter boundary.
        working = torch.zeros(
            blocks,
            2,
            heads,
            page,
            dim,
            dtype=torch.bfloat16,
            device=device,
        )
        block_table = torch.tensor([[1, 2, 0]], dtype=torch.int32, device=device)
        convert_active_pages(layout, block_table, working, quantize=False)
        torch.testing.assert_close(working[1, 0, :, 0], out_k[0], rtol=0, atol=0)
        replacement = torch.randn_like(working[1])
        working[1].copy_(replacement)
        convert_active_pages(layout, block_table, working, quantize=True)
        restored = torch.zeros_like(working)
        convert_active_pages(layout, block_table, restored, quantize=False)
        torch.testing.assert_close(
            restored[1], _reference_groups(replacement), rtol=0, atol=0
        )

        # Compile the CP-prefill BF16 scatter and short-tail clearing kernels.
        hnd_k = torch.full(
            (2, heads, page, dim), 9, dtype=torch.bfloat16, device=device
        )
        hnd_v = hnd_k.clone()
        scatter_slots = torch.arange(rows, device=device)
        scatter_main_rows_to_hnd(k, v, scatter_slots, hnd_k, hnd_v)
        idx_scratch = torch.full(
            (2 * page, 1, index_dim), 9, dtype=torch.bfloat16, device=device
        )
        clear_working_tails(
            hnd_k,
            hnd_v,
            idx_scratch,
            torch.tensor([3], dtype=torch.int32, device=device),
            2 * page,
        )
        self.assertTrue(torch.count_nonzero(hnd_k[0, :, 3]).item() == 0)
        self.assertTrue(torch.count_nonzero(hnd_v[0, :, 3]).item() == 0)
        self.assertTrue(torch.count_nonzero(idx_scratch[3]).item() == 0)

    def test_triton_gather_can_match_q8kv4_e4m3_compute_grid(self):
        """BF16 working pages can carry the native Q8KV4 input values exactly."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(31)
        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 3, 2, 8, 32, 64
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16
        base = torch.zeros((blocks, main_bytes), dtype=torch.uint8, device=device)
        side = torch.zeros(
            (blocks, main_scale_bytes + index_bytes + index_scale_bytes),
            dtype=torch.uint8,
            device=device,
        )
        layout = cache_layout(base, side, heads, page, dim)
        rows = blocks * page
        slots = torch.arange(rows, device=device, dtype=torch.int64)
        exponents = torch.randint(-5, 7, (rows, heads, dim), device=device)
        k = (torch.randn(rows, heads, dim, device=device) * torch.exp2(exponents)).to(
            torch.bfloat16
        )
        v = (k.float() * 0.625).to(torch.bfloat16)
        idx = torch.randn(rows, index_dim, device=device, dtype=torch.bfloat16) * 9
        quantize_main_rows(k, v, slots, layout)
        quantize_index_rows(idx, slots, layout)

        out_k = torch.empty_like(k)
        out_v = torch.empty_like(v)
        out_idx = torch.empty(rows, 1, index_dim, dtype=torch.bfloat16, device=device)
        gather_main_rows(
            layout,
            slots,
            slots,
            out_k,
            out_v,
            e4m3_compute_grid=True,
        )
        gather_index_rows(
            layout,
            index_dim,
            slots,
            slots,
            out_idx,
            e4m3_compute_grid=True,
        )

        def q8kv4_grid(values: torch.Tensor) -> torch.Tensor:
            return _reference_groups(values).to(torch.float8_e4m3fn).to(torch.bfloat16)

        torch.testing.assert_close(out_k, q8kv4_grid(k), rtol=0, atol=0)
        torch.testing.assert_close(out_v, q8kv4_grid(v), rtol=0, atol=0)
        torch.testing.assert_close(out_idx[:, 0], q8kv4_grid(idx), rtol=0, atol=0)

        suffix_k = torch.empty(
            blocks, heads, page, dim, dtype=torch.bfloat16, device=device
        )
        suffix_v = torch.empty_like(suffix_k)
        suffix_idx = torch.empty_like(out_idx)
        materialize_q8kv4_rows(k, slots, suffix_k, page, out_hnd=True)
        materialize_q8kv4_rows(v, slots, suffix_v, page, out_hnd=True)
        materialize_q8kv4_rows(idx, slots, suffix_idx, page)
        suffix_k_flat = suffix_k.permute(0, 2, 1, 3).reshape_as(k)
        suffix_v_flat = suffix_v.permute(0, 2, 1, 3).reshape_as(v)
        torch.testing.assert_close(suffix_k_flat, out_k, rtol=0, atol=0)
        torch.testing.assert_close(suffix_v_flat, out_v, rtol=0, atol=0)
        torch.testing.assert_close(suffix_idx, out_idx, rtol=0, atol=0)

        q = torch.linspace(-500, 500, 4096, device=device, dtype=torch.bfloat16)
        expected_q = q.clamp(-448, 448).to(torch.float8_e4m3fn).to(torch.bfloat16)
        round_to_e4m3_compute_grid_(q)
        torch.testing.assert_close(q, expected_q, rtol=0, atol=0)

    def test_materialize_q8kv4_accepts_interleaved_qkv_views(self):
        """CP all-gather passes K/V slices with a larger token stride."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(37)
        device = torch.device("cuda")
        rows, heads, page, dim = 7, 4, 8, 32
        q_width = heads * dim
        packed_qkv = torch.randn(rows, 3 * q_width, dtype=torch.bfloat16, device=device)
        k = packed_qkv[:, q_width : 2 * q_width].view(rows, heads, dim)
        self.assertFalse(k.is_contiguous())
        self.assertEqual(k.stride(2), 1)
        out = torch.zeros(1, heads, page, dim, dtype=torch.bfloat16, device=device)
        slots = torch.arange(rows, dtype=torch.int64, device=device)

        materialize_q8kv4_rows(k, slots, out, page, out_hnd=True)

        expected = _reference_groups(k).to(torch.float8_e4m3fn).to(torch.bfloat16)
        torch.testing.assert_close(
            out[0, :, :rows].permute(1, 0, 2), expected, rtol=0, atol=0
        )

    def test_cp_suffix_bf16_working_pages_do_not_qdq(self):
        """The gathered current suffix is scattered verbatim for BF16 prefill."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(20260923)
        device = torch.device("cuda")
        rows, heads, page, dim = 7, 4, 8, 32
        packed = torch.randn(rows, 2 * heads * dim, dtype=torch.bfloat16, device=device)
        k = packed[:, : heads * dim].view(rows, heads, dim)
        v = packed[:, heads * dim :].view(rows, heads, dim)
        self.assertFalse(k.is_contiguous())
        slots = torch.tensor([0, 3, 6, 8, 11, 13, 15], device=device)
        out_k = torch.full(
            (2, heads, page, dim), float("nan"), dtype=torch.bfloat16, device=device
        )
        out_v = torch.full_like(out_k, float("nan"))

        scatter_main_rows_to_hnd(k, v, slots, out_k, out_v)

        page_ids = torch.div(slots, page, rounding_mode="floor")
        page_offsets = slots.remainder(page)
        actual_k = out_k[page_ids, :, page_offsets]
        actual_v = out_v[page_ids, :, page_offsets]
        torch.testing.assert_close(actual_k, k, rtol=0, atol=0)
        torch.testing.assert_close(actual_v, v, rtol=0, atol=0)

        qdq_k = torch.empty_like(out_k)
        materialize_q8kv4_rows(k, slots, qdq_k, page, out_hnd=True)
        self.assertFalse(torch.equal(actual_k, qdq_k[page_ids, :, page_offsets]))

    def test_fused_main_index_writer_is_bitwise_identical(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(20260923)
        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 8, 4, 128, 128, 128
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16

        def make_layout():
            base = torch.full(
                (blocks, main_bytes), 0xA5, dtype=torch.uint8, device=device
            )
            side = torch.full(
                (
                    blocks,
                    main_scale_bytes + index_bytes + index_scale_bytes,
                ),
                0x5A,
                dtype=torch.uint8,
                device=device,
            )
            return base, side, cache_layout(base, side, heads, page, dim)

        reference_base, reference_side, reference_layout = make_layout()
        fused_base, fused_side, fused_layout = make_layout()
        slots = torch.tensor(
            [
                0,
                1,
                126,
                127,
                128,
                129,
                255,
                256,
                383,
                384,
                511,
                512,
                767,
                768,
                895,
                1023,
                -1,
                1024,
            ],
            dtype=torch.int64,
            device=device,
        )
        rows = int(slots.numel())
        k = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        index_k = torch.randn(rows, index_dim, dtype=torch.bfloat16, device=device)

        quantize_main_rows(k, v, slots, reference_layout)
        quantize_index_rows(index_k, slots, reference_layout)
        quantize_main_index_rows(k, v, index_k, slots, fused_layout)
        torch.cuda.synchronize()

        torch.testing.assert_close(fused_base, reference_base, rtol=0, atol=0)
        torch.testing.assert_close(fused_side, reference_side, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize_main_index_rows(k, v, index_k, slots, fused_layout)
        k.copy_(torch.randn_like(k))
        v.copy_(torch.randn_like(v))
        index_k.copy_(torch.randn_like(index_k))
        quantize_main_rows(k, v, slots, reference_layout)
        quantize_index_rows(index_k, slots, reference_layout)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(fused_base, reference_base, rtol=0, atol=0)
        torch.testing.assert_close(fused_side, reference_side, rtol=0, atol=0)

    def test_fused_writer_mma_scales_match_compact_logical_values(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(20260924)
        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 3, 4, 128, 128, 128
        groups = dim // 16
        index_groups = index_dim // 16
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * groups
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_groups

        def make_layout():
            base = torch.zeros((blocks, main_bytes), dtype=torch.uint8, device=device)
            side = torch.zeros(
                (blocks, main_scale_bytes + index_bytes + index_scale_bytes),
                dtype=torch.uint8,
                device=device,
            )
            return base, side, cache_layout(base, side, heads, page, dim)

        compact_base, _, compact_layout = make_layout()
        mma_base, _, mma_layout = make_layout()
        rows = blocks * page
        slots = torch.arange(rows, dtype=torch.int64, device=device)
        k = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        index_k = torch.randn(rows, index_dim, dtype=torch.bfloat16, device=device)

        quantize_main_index_rows(k, v, index_k, slots, compact_layout)
        quantize_main_index_rows(
            k, v, index_k, slots, mma_layout, mma_scale_layout=True
        )
        fused_width = 2 * heads * dim + index_dim
        fused_projection = torch.empty(
            rows, fused_width, dtype=torch.bfloat16, device=device
        )
        fused_projection[:, : heads * dim].copy_(k.reshape(rows, -1))
        fused_projection[:, heads * dim : 2 * heads * dim].copy_(v.reshape(rows, -1))
        fused_projection[:, 2 * heads * dim :].copy_(index_k)
        strided_k = fused_projection[:, : heads * dim].view(rows, heads, dim)
        strided_v = fused_projection[:, heads * dim : 2 * heads * dim].view(
            rows, heads, dim
        )
        strided_idx = fused_projection[:, 2 * heads * dim :].view(rows, 1, index_dim)
        self.assertFalse(strided_k.is_contiguous())
        self.assertFalse(strided_v.is_contiguous())
        self.assertFalse(strided_idx.is_contiguous())
        strided_k_packed = torch.empty(
            blocks, heads, page, dim // 2, dtype=torch.uint8, device=device
        )
        strided_v_packed = torch.empty_like(strided_k_packed)
        strided_k_scales = torch.empty(
            blocks,
            heads * page * groups,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        strided_v_scales = torch.empty_like(strided_k_scales)
        strided_idx_packed = torch.empty(
            blocks, 1, page, index_dim // 2, dtype=torch.uint8, device=device
        )
        strided_idx_scales = torch.empty(
            blocks,
            page * index_groups,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        quantize_main_index_rows_to_planes(
            strided_k,
            strided_v,
            strided_idx,
            slots,
            strided_k_packed,
            strided_k_scales,
            strided_v_packed,
            strided_v_scales,
            strided_idx_packed,
            strided_idx_scales,
            page_size=page,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(mma_base, compact_base, rtol=0, atol=0)
        mma_k_packed, mma_k_scales = mma_layout.main_plane(0)
        mma_v_packed, mma_v_scales = mma_layout.main_plane(1)
        mma_idx_packed, mma_idx_scales = mma_layout.indexer(index_dim)
        torch.testing.assert_close(
            strided_k_packed, mma_k_packed.view_as(strided_k_packed), rtol=0, atol=0
        )
        torch.testing.assert_close(
            strided_v_packed, mma_v_packed.view_as(strided_v_packed), rtol=0, atol=0
        )
        torch.testing.assert_close(strided_k_scales, mma_k_scales, rtol=0, atol=0)
        torch.testing.assert_close(strided_v_scales, mma_v_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            strided_idx_packed,
            mma_idx_packed.view_as(strided_idx_packed),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(strided_idx_scales, mma_idx_scales, rtol=0, atol=0)

        row = torch.arange(page, dtype=torch.int64, device=device)[:, None]
        group = torch.arange(groups, dtype=torch.int64, device=device)[None, :]
        offset = (group // 4) * 512 + (row % 32) * 16 + (row // 32) * 4 + group % 4
        for kv_index in (0, 1):
            _, compact_scale = compact_layout.main_plane(kv_index)
            _, mma_scale = mma_layout.main_plane(kv_index)
            compact_u8 = compact_scale.view(torch.uint8).reshape(
                blocks, heads, page, groups
            )
            mma_u8 = mma_scale.view(torch.uint8).reshape(blocks, heads, page * groups)
            restored = mma_u8[:, :, offset]
            torch.testing.assert_close(restored, compact_u8, rtol=0, atol=0)

        _, compact_idx_scale = compact_layout.indexer(index_dim)
        _, mma_idx_scale = mma_layout.indexer(index_dim)
        compact_idx_u8 = compact_idx_scale.view(torch.uint8).reshape(
            blocks, page, index_groups
        )
        mma_idx_u8 = mma_idx_scale.view(torch.uint8).reshape(
            blocks, page * index_groups
        )
        restored_idx = mma_idx_u8[:, offset]
        torch.testing.assert_close(restored_idx, compact_idx_u8, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize_main_index_rows(
                k, v, index_k, slots, mma_layout, mma_scale_layout=True
            )
        expected_base = mma_base.clone()
        expected_side = mma_layout.side_bytes.clone()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(mma_base, expected_base, rtol=0, atol=0)
        torch.testing.assert_close(mma_layout.side_bytes, expected_side, rtol=0, atol=0)

    def test_query_mma_writer_matches_logical_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.manual_seed(20260925)
        device = torch.device("cuda")
        rows, heads, dim, groups = 257, 8, 128, 8
        q = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device=device)
        packed = torch.empty(rows, heads, dim // 2, dtype=torch.uint8, device=device)
        scales = torch.empty(
            heads,
            (rows + 127) // 128,
            groups // 4,
            32,
            4,
            4,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        quantize_query_rows_mma(q, packed, scales)
        torch.cuda.synchronize()

        ref_packed, ref_scales = reference_quantize(
            q.float().reshape(rows, heads, groups, 16)
        )
        ref_packed = ref_packed.reshape(rows, heads, dim // 2)
        torch.testing.assert_close(packed, ref_packed, rtol=0, atol=0)

        row = torch.arange(rows, dtype=torch.int64, device=device)[:, None]
        group = torch.arange(groups, dtype=torch.int64, device=device)[None, :]
        offset = (
            (row // 128) * (128 * groups)
            + (group // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + group % 4
        )
        scale_flat = scales.view(torch.uint8).reshape(
            heads, ((rows + 127) // 128) * 128 * groups
        )
        restored = scale_flat[:, offset].permute(1, 0, 2)
        torch.testing.assert_close(
            restored,
            ref_scales.view(torch.uint8),
            rtol=0,
            atol=0,
        )

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            quantize_query_rows_mma(q, packed, scales)
        expected_packed = packed.clone()
        expected_scales = scales.clone()
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(packed, expected_packed, rtol=0, atol=0)
        torch.testing.assert_close(
            scales.view(torch.uint8),
            expected_scales.view(torch.uint8),
            rtol=0,
            atol=0,
        )

    def test_two_region_cache_exposes_four_logical_native_views(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 2, 2, 4, 32, 64
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16
        base = torch.zeros((blocks, main_bytes), dtype=torch.uint8, device=device)
        side = torch.zeros(
            (blocks, main_scale_bytes + index_bytes + index_scale_bytes),
            dtype=torch.uint8,
            device=device,
        )
        layout = cache_layout(base, side, heads, page, dim)
        views = layout.logical_views(index_dim)

        self.assertEqual(
            tuple(views.main_k_fp4.shape), (blocks, heads * page * dim // 2)
        )
        self.assertEqual(
            tuple(views.main_v_fp4.shape), (blocks, heads * page * dim // 2)
        )
        self.assertEqual(
            tuple(views.main_k_scale.shape), (blocks, heads * page * (dim // 16))
        )
        self.assertEqual(
            tuple(views.main_v_scale.shape), (blocks, heads * page * (dim // 16))
        )
        self.assertEqual(tuple(views.idx_k_fp4.shape), (blocks, index_bytes))
        self.assertEqual(tuple(views.idx_k_scale.shape), (blocks, index_scale_bytes))

        # The logical views must alias the original two byte regions.  This is
        # the first-stage contract used by a native kernel; no copy or BF16
        # working page is involved in constructing the views.
        self.assertEqual(views.main_k_fp4.data_ptr(), base.data_ptr())
        self.assertEqual(views.main_k_scale.data_ptr(), side.data_ptr())
        self.assertEqual(
            views.idx_k_fp4.data_ptr(),
            side.data_ptr() + main_scale_bytes,
        )
        self.assertEqual(
            views.idx_k_scale.data_ptr(),
            side.data_ptr() + main_scale_bytes + index_bytes,
        )

    def test_decode_write_persists_native_packed_planes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention

        device = torch.device("cuda")
        blocks, heads, page, dim, index_dim = 5, 2, 128, 64, 64
        main_bytes = 2 * heads * page * dim // 2
        main_scale_bytes = 2 * heads * page * (dim // 16)
        index_bytes = page * index_dim // 2
        index_scale_bytes = page * index_dim // 16
        cache = SimpleNamespace(
            kv_cache_base=torch.zeros(
                (blocks, main_bytes), dtype=torch.uint8, device=device
            ),
            kv_scale_base=torch.zeros(
                (blocks, main_scale_bytes + index_bytes + index_scale_bytes),
                dtype=torch.uint8,
                device=device,
            ),
        )
        attn = object.__new__(MSAAttention)
        attn.nvfp4_kv_cache = True
        attn.kv_head_num = heads
        attn.physical_page_size = page
        attn.page_size = page
        attn.head_dim = dim
        attn.idx_head_dim = index_dim

        block_table = torch.tensor(
            [[1, 2, 0], [3, 2, 0]], dtype=torch.int32, device=device
        )
        seq_lens = torch.tensor([2, 129], dtype=torch.int32, device=device)
        torch.manual_seed(19)
        k = torch.randn(2, heads, dim, dtype=torch.bfloat16, device=device)
        v = torch.randn_like(k)
        idx = torch.randn(2, 1, index_dim, dtype=torch.bfloat16, device=device)
        # A non-unit stored scale makes 0.5859375 / 0.234375 exactly 2.5.
        # The kernel must keep that strict midpoint on the even code (2.0),
        # rather than letting GPU division round it just above the threshold.
        k[1, 0, :16] = 0
        k[1, 0, 0] = torch.tensor(1.40625, dtype=torch.bfloat16, device=device)
        k[1, 0, 7] = torch.tensor(0.5859375, dtype=torch.bfloat16, device=device)

        result = attn._write_kv_cache_and_idx_k_for_decode(
            cache, k, v, idx, seq_lens, block_table
        )
        self.assertEqual(result, ())
        expected_base = torch.zeros_like(cache.kv_cache_base)
        expected_side = torch.zeros_like(cache.kv_scale_base)
        expected_layout = cache_layout(expected_base, expected_side, heads, page, dim)
        physical_slots = torch.tensor([129, 256], dtype=torch.int64, device=device)
        quantize_main_index_rows(
            k,
            v,
            idx,
            physical_slots,
            expected_layout,
            mma_scale_layout=True,
        )
        torch.testing.assert_close(cache.kv_cache_base, expected_base, rtol=0, atol=0)
        torch.testing.assert_close(cache.kv_scale_base, expected_side, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
