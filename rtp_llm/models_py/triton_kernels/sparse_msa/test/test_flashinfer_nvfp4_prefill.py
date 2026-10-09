"""RTP-to-FlashInfer sparse prefill page and query contract."""

import unittest

import torch
from flashinfer.msa_ops import _nvfp4_prefill_sm100 as fi_reference


class FlashInferNVFP4PrefillTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_rtp_writer_scales_against_fp32_reference(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
            self.skipTest("FlashInfer NVFP4 prefill requires Blackwell")
        from rtp_llm.models_py.triton_kernels.common.nvfp4_kv_cache import (
            _gather_rows,
            quantize_main_index_rows_to_planes,
        )
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            flashinfer_nvfp4_prefill as adapter,
        )

        device = torch.device("cuda")
        torch.manual_seed(311)
        pages, rows = 16, 16 * 128 - 7
        k_rows = torch.randn((rows, 4, 128), device=device, dtype=torch.bfloat16)
        v_rows = torch.randn_like(k_rows)
        k_rows[:32] *= 16
        v_rows[32:64] *= 0.01
        idx_rows = torch.zeros((rows, 1, 128), device=device, dtype=torch.bfloat16)
        k = torch.empty((pages, 4, 128, 64), device=device, dtype=torch.uint8)
        v = torch.empty_like(k)
        ks = torch.empty((pages, 4, 128, 8), device=device, dtype=torch.float8_e4m3fn)
        vs = torch.empty_like(ks)
        idx = torch.empty((pages, 1, 128, 64), device=device, dtype=torch.uint8)
        idx_scale = torch.empty((pages, 1, 128, 8), device=device, dtype=torch.float8_e4m3fn)
        k.fill_(255)
        v.fill_(255)
        ks.view(torch.uint8).fill_(127)
        vs.view(torch.uint8).fill_(127)
        quantize_main_index_rows_to_planes(
            k_rows, v_rows, idx_rows,
            torch.arange(rows, device=device, dtype=torch.int64),
            k, ks, v, vs, idx, idx_scale,
        )
        q = (torch.randn((4, 64, 128), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        selected = torch.arange(15, -1, -1, device=device, dtype=torch.int32)
        topk = selected.view(1, 1, 16).expand(4, 4, 16).contiguous()
        slots = torch.arange(pages * 128, device=device, dtype=torch.int32).view(1, -1)
        cu_q = torch.tensor([0, 4], device=device, dtype=torch.int32)
        seqused = torch.tensor([rows], device=device, dtype=torch.int32)
        positions = torch.arange(rows - 4, rows, device=device, dtype=torch.int32)
        actual = adapter.flashinfer_sparse_prefill_from_topk_fp4(
            q, k, v, ks.view(pages, -1), vs.view(pages, -1), topk,
            slots, cu_q, seqused, positions, seqused, 128**-0.5,
        )
        pool = adapter._PLANAR_PAGES.pool[:pages]
        layout = adapter.page_layout(4)
        self.assertEqual(layout, fi_reference.page_layout(4))
        data_shape = (pages, 4, 128, 64)
        scale_shape = (pages, 4, 128, 8)
        data_stride = (layout["page_bytes"], 8192, 64, 1)
        scale_stride = (layout["page_bytes"], 1024, 8, 1)
        self.assertEqual(int(torch.count_nonzero(torch.as_strided(
            pool, data_shape, data_stride)[-1, :, -7:, :])), 0)
        expected = torch.empty_like(q)
        fi_reference.reference(
            q=q,
            k_data=torch.as_strided(pool, data_shape, data_stride),
            v_data=torch.as_strided(pool, data_shape, data_stride, layout["v_data_byte_offset"]),
            k_scale=torch.as_strided(pool, scale_shape, scale_stride,
                                     layout["k_scale_byte_offset"]),
            v_scale=torch.as_strided(pool, scale_shape, scale_stride,
                                     layout["v_scale_byte_offset"]),
            q2k_indices=torch.arange(pages, device=device, dtype=torch.int32)
            .view(1, 1, 16).expand(4, 4, 16).contiguous(),
            cu_seqlens_q=cu_q, page_table=(slots[:, ::128] // 128).contiguous(),
            seqused_k=seqused, softmax_scale=128**-0.5,
            k_global_scale=1.0, v_global_scale=1.0, out=expected,
        )
        self.assertTrue(bool(torch.isfinite(actual).all()))
        error = (actual.float() - expected.float()).square().mean().sqrt()
        reference = expected.float().square().mean().sqrt()
        self.assertLess(float(error / reference), 0.005)
        # The removed vendor dispatched this exact low-level operation. Public
        # API validation must not change its output, including CP segments.
        legacy = torch.empty_like(q)
        fi_reference.run(
            q=q, k=torch.as_strided(pool, data_shape, data_stride),
            v=torch.as_strided(pool, data_shape, data_stride, layout["v_data_byte_offset"]),
            k_scale=torch.as_strided(pool, scale_shape, scale_stride, layout["k_scale_byte_offset"]),
            v_scale=torch.as_strided(pool, scale_shape, scale_stride, layout["v_scale_byte_offset"]),
            q2k_indices=torch.arange(pages, device=device, dtype=torch.int32)
            .view(1, 1, 16).expand(4, 4, 16).contiguous(), cu_seqlens_q=cu_q,
            page_table=(slots[:, ::128] // 128).contiguous(), seqused_k=seqused,
            out=legacy, softmax_scale=128**-0.5,
            k_global_scale=1.0, v_global_scale=1.0,
        )
        self.assertTrue(torch.equal(actual, legacy))
        self.assertGreater(int(torch.unique(ks.view(torch.uint8)).numel()), 8)

        physical = torch.arange(rows, device=device, dtype=torch.int64)
        dequant_k = torch.empty((rows, 4, 128), device=device, dtype=torch.bfloat16)
        dequant_v = torch.empty_like(dequant_k)
        linear = torch.arange(1024, device=device)
        token, group = linear // 8, linear % 8
        mma = (group // 4) * 512 + (token % 32) * 16 + (token // 32) * 4 + group % 4
        ks_linear = ks.view(torch.uint8).view(pages, 4, 1024)[:, :, mma]
        vs_linear = vs.view(torch.uint8).view(pages, 4, 1024)[:, :, mma]
        _gather_rows(k.view(pages, -1), ks_linear.contiguous().view(pages, -1)
                     .view(torch.float8_e4m3fn), physical,
                     physical, dequant_k, 128)
        _gather_rows(v.view(pages, -1), vs_linear.contiguous().view(pages, -1)
                     .view(torch.float8_e4m3fn), physical,
                     physical, dequant_v, 128)
        scores = torch.einsum(
            "qhgd,lhd->qhgl", q.float().view(4, 4, 16, 128),
            dequant_k.float()) * (128**-0.5)
        causal = torch.arange(rows, device=device).view(1, 1, 1, -1)
        scores.masked_fill_(causal > positions.view(4, 1, 1, 1), float("-inf"))
        probabilities = scores.softmax(dim=-1)
        independent = torch.einsum(
            "qhgl,lhd->qhgd", probabilities, dequant_v.float()
        ).reshape_as(q)
        independent_error = (actual.float() - independent).square().mean().sqrt()
        self.assertLess(float(independent_error / independent.square().mean().sqrt()), 0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_two_segments_against_fp32_reference(self):
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)):
            self.skipTest("FlashInfer NVFP4 prefill requires Blackwell")
        from rtp_llm.models_py.triton_kernels.sparse_msa.prefill import (
            flashinfer_nvfp4_prefill as adapter,
        )

        device = torch.device("cuda")
        torch.manual_seed(103)
        pages = 6
        k = torch.randint(0, 256, (pages, 4, 128, 64), device=device, dtype=torch.uint8)
        v = torch.randint(0, 256, (pages, 4, 128, 64), device=device, dtype=torch.uint8)
        ks = torch.full((pages, 4096), 0x20, device=device, dtype=torch.uint8)
        vs = torch.full_like(ks, 0x20)
        q = (torch.randn((4, 64, 128), device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        slots = torch.arange(384, device=device, dtype=torch.int32).view(1, 384).expand(2, -1)
        cu_q = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
        seqused = torch.tensor([130, 130], device=device, dtype=torch.int32)
        positions = torch.tensor([128, 129, 128, 129], device=device, dtype=torch.int32)
        topk = torch.full((4, 4, 16), -1, device=device, dtype=torch.int32)
        topk[:, :, 0] = 1
        topk[:, :, 1] = 0
        topk[:, :, 2] = 2  # Future page must be removed.
        actual = adapter.flashinfer_sparse_prefill_from_topk_fp4(
            q, k, v, ks, vs, topk, slots, cu_q, seqused, positions,
            seqused[:1], 128**-0.5, segments_per_request=2,
        )
        pool = adapter._PLANAR_PAGES.pool[:pages]
        layout = adapter.page_layout(4)
        data_shape = (pages, 4, 128, 64)
        scale_shape = (pages, 4, 128, 8)
        data_stride = (layout["page_bytes"], 8192, 64, 1)
        scale_stride = (layout["page_bytes"], 1024, 8, 1)
        ordered = torch.full_like(topk, -1)
        ordered[:, :, :2] = torch.tensor([0, 1], device=device, dtype=torch.int32)
        expected = torch.empty_like(q)
        fi_reference.reference(
            q=q,
            k_data=torch.as_strided(pool, data_shape, data_stride),
            v_data=torch.as_strided(pool, data_shape, data_stride, layout["v_data_byte_offset"]),
            k_scale=torch.as_strided(pool, scale_shape, scale_stride,
                                     layout["k_scale_byte_offset"]),
            v_scale=torch.as_strided(pool, scale_shape, scale_stride,
                                     layout["v_scale_byte_offset"]),
            q2k_indices=ordered, cu_seqlens_q=cu_q,
            page_table=(slots[:, ::128] // 128).contiguous(), seqused_k=seqused,
            softmax_scale=128**-0.5, k_global_scale=1.0,
            v_global_scale=1.0, out=expected,
        )
        self.assertTrue(bool(torch.isfinite(actual).all()))
        error = (actual.float() - expected.float()).square().mean().sqrt()
        reference = expected.float().square().mean().sqrt()
        self.assertLess(float(error / reference), 0.005)


if __name__ == "__main__":
    unittest.main()
