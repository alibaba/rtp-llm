import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
    cache_layout,
    clear_working_tails,
    convert_active_pages,
    gather_index_rows,
    gather_main_rows,
    quantize_index_rows,
    quantize_main_rows,
    reference_dequantize,
    reference_quantize,
    scatter_main_rows_to_hnd,
)


def _reference_groups(values: torch.Tensor) -> torch.Tensor:
    groups = values.float().reshape(*values.shape[:-1], -1, 16)
    packed, scales = reference_quantize(groups)
    return reference_dequantize(packed, scales).reshape(values.shape).to(torch.bfloat16)


class TestNVFP4KVCache(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
