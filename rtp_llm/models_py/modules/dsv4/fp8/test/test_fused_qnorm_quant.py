"""Unit tests for fused Q LoRA RMSNorm + UE8M0 FP8 quantization."""

from __future__ import annotations

import unittest

import flashinfer
import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_gemm_nt,
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.dsv4.fp8._fused_qnorm_quant_triton import (
    can_fuse_qnorm_fp8_ue8m0_quant,
    fused_rmsnorm_fp8_ue8m0_quant,
)


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x32 = x.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (x32 * inv * weight.float()).to(x.dtype)


def _raw_u8_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        a.contiguous().view(torch.uint8).to(torch.int16)
        - b.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class FusedQNormQuantTest(unittest.TestCase):
    def test_empty_batch_returns_expected_shapes(self) -> None:
        for width in (1024, 4096):
            x = torch.empty((0, width), device="cuda", dtype=torch.bfloat16)
            w = torch.ones((width,), device="cuda", dtype=torch.bfloat16)
            self.assertTrue(can_fuse_qnorm_fp8_ue8m0_quant(x, w))
            norm_none, (q_only, s_only) = fused_rmsnorm_fp8_ue8m0_quant(
                x, w, eps=1.0e-6, emit_norm=False
            )
            self.assertIsNone(norm_none)
            self.assertEqual(tuple(q_only.shape), (0, width))
            self.assertEqual(q_only.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(s_only.shape), (0, (width // 128 + 3) // 4))
            self.assertEqual(s_only.dtype, torch.int32)

            norm_out, (q_both, s_both) = fused_rmsnorm_fp8_ue8m0_quant(
                x, w, eps=1.0e-6, emit_norm=True
            )
            self.assertIsNotNone(norm_out)
            assert norm_out is not None
            self.assertEqual(tuple(norm_out.shape), (0, width))
            self.assertEqual(tuple(q_both.shape), (0, width))
            self.assertEqual(tuple(s_both.shape), (0, (width // 128 + 3) // 4))

    def test_matches_flashinfer_and_sgl_quant_across_shapes(self) -> None:
        eps = 1.0e-6
        shapes = (
            (1, 1024),
            (16, 1024),
            (63, 1024),
            (64, 1024),
            (127, 1024),
            (257, 1024),
            (1024, 1024),
            (1, 16, 1024),
            (2, 17, 1536),
            (558, 1536),
            (1, 4096),
            (16, 4096),
            (257, 4096),
            (558, 4096),
            (1024, 4096),
            (2048, 4096),
            (558, 7168),
            (1024, 7168),
        )
        for shape in shapes:
            with self.subTest(shape=shape):
                torch.manual_seed(20261127 + sum(shape))
                width = shape[-1]
                x = (
                    torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 0.4
                ).contiguous()
                w = (
                    (torch.randn((width,), device="cuda") * 0.15 + 1.0)
                    .abs()
                    .to(torch.bfloat16)
                    .contiguous()
                )
                flat_x = x.view(-1, width)
                rows = flat_x.shape[0]
                ref_norm = flashinfer.norm.rmsnorm(flat_x, w, eps=eps).view(*shape)
                ref_norm_torch = _torch_rmsnorm(x, w, eps)
                ref_q, ref_s = sgl_per_token_group_quant_fp8(
                    ref_norm.view(-1, width),
                    group_size=128,
                    eps=1.0e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                ref_q = ref_q.view(*shape)

                got_none, (q_only, s_only) = fused_rmsnorm_fp8_ue8m0_quant(
                    x, w, eps=eps, emit_norm=False
                )
                self.assertIsNone(got_none)

                got_norm, (q_dual, s_dual) = fused_rmsnorm_fp8_ue8m0_quant(
                    x, w, eps=eps, emit_norm=True
                )
                self.assertIsNotNone(got_norm)
                assert got_norm is not None

                # emit_norm=False and emit_norm=True must produce identical FP8 + scales.
                self.assertTrue(
                    torch.equal(q_only.view(torch.uint8), q_dual.view(torch.uint8))
                )
                self.assertTrue(torch.equal(s_only, s_dual))

                # Verify TMA-aligned column-major scale geometry.
                aligned_rows = ((rows + 3) // 4) * 4
                expected_packs = (width // 128 + 3) // 4
                self.assertEqual(tuple(s_only.shape), (rows, expected_packs))
                self.assertEqual(tuple(s_only.stride()), (1, aligned_rows))
                self.assertEqual(s_only.data_ptr() % 16, 0)

                norm_abs = (ref_norm.float() - got_norm.float()).abs()
                torch_norm_abs = (ref_norm_torch.float() - got_norm.float()).abs()
                self.assertLessEqual(float(norm_abs.max().item()), 0.03125)
                self.assertLessEqual(float(torch_norm_abs.max().item()), 0.03125)

                q_diff = _raw_u8_diff(ref_q, q_only)
                s_diff = _raw_u8_diff(ref_s, s_only)
                q_exact = float((q_diff == 0).float().mean().item())
                s_exact = float((s_diff == 0).float().mean().item())
                self.assertGreaterEqual(q_exact, 0.9999)
                self.assertLessEqual(int(q_diff.max().item()), 1)
                self.assertGreaterEqual(s_exact, 0.9999)
                self.assertLessEqual(int(s_diff.max().item()), 1)

    def test_zero_clamp_preallocated_and_cuda_graph_replay(self) -> None:
        eps = 1.0e-6
        for width in (1024, 4096):
            with self.subTest(width=width):
                w = (
                    torch.rand(width, device="cuda", dtype=torch.bfloat16) + 0.5
                ).contiguous()
                zeros = torch.zeros((16, width), device="cuda", dtype=torch.bfloat16)
                out_norm = torch.empty_like(zeros)
                got_norm, (q_z, s_z) = fused_rmsnorm_fp8_ue8m0_quant(
                    zeros, w, eps=eps, emit_norm=True, out_norm=out_norm
                )
                assert got_norm is not None
                self.assertEqual(got_norm.data_ptr(), out_norm.data_ptr())
                self.assertEqual(int((q_z.view(torch.uint8) != 0).sum().item()), 0)

                # Compare clamp exponent against sgl_per_token_group_quant_fp8 on zeros.
                _, ref_sz = sgl_per_token_group_quant_fp8(
                    zeros,
                    group_size=128,
                    eps=1.0e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                self.assertTrue(torch.equal(s_z, ref_sz))

                # CUDA graph capture and replay across input scales.
                x = torch.randn((1, 137, width), device="cuda", dtype=torch.bfloat16)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fused_rmsnorm_fp8_ue8m0_quant(x, w, eps=eps, emit_norm=False)
                torch.cuda.current_stream().wait_stream(stream)

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    g_norm, (g_q, g_s) = fused_rmsnorm_fp8_ue8m0_quant(
                        x, w, eps=eps, emit_norm=False
                    )
                self.assertIsNone(g_norm)

                for scale in (0.0, 1.0, 25.0):
                    x.copy_(torch.randn_like(x) * scale)
                    graph.replay()
                    ref_y = _torch_rmsnorm(x.view(-1, width), w, eps)
                    ref_q, ref_s = sgl_per_token_group_quant_fp8(
                        ref_y,
                        group_size=128,
                        eps=1.0e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    self.assertLessEqual(
                        int(_raw_u8_diff(ref_q.view_as(g_q), g_q).max().item()), 1
                    )
                    self.assertLessEqual(int(_raw_u8_diff(ref_s, g_s).max().item()), 1)

    def test_deepgemm_fp8_gemm_nt_end_to_end_equivalence(self) -> None:
        if not has_deep_gemm() or not is_deep_gemm_e8m0_used():
            self.skipTest("DeepGEMM UE8M0 unavailable")
        torch.manual_seed(77)
        rows, k_dim, n_dim = 128, 1024, 512
        x = (
            torch.randn(rows, k_dim, device="cuda", dtype=torch.bfloat16) * 0.4
        ).contiguous()
        norm_w = (
            torch.rand(k_dim, device="cuda", dtype=torch.bfloat16) + 0.5
        ).contiguous()
        w_bf16 = (
            torch.randn(n_dim, k_dim, device="cuda", dtype=torch.bfloat16) * 0.03
        ).contiguous()
        w_fp8 = w_bf16.to(torch.float8_e4m3fn)
        w_scale_f = torch.ones(
            (n_dim // 128, k_dim // 128), device="cuda", dtype=torch.float32
        )
        w_fp8, w_scale_int = requant_weight_ue8m0(w_fp8, w_scale_f)

        ref_norm = flashinfer.norm.rmsnorm(x, norm_w, eps=1.0e-6)
        ref_q, ref_s = sgl_per_token_group_quant_fp8(
            ref_norm,
            group_size=128,
            eps=1.0e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        ref_out = torch.empty((rows, n_dim), device="cuda", dtype=torch.bfloat16)
        fp8_gemm_nt(
            (ref_q, ref_s), (w_fp8, w_scale_int), ref_out, disable_ue8m0_cast=False
        )

        got_norm, (got_q, got_s) = fused_rmsnorm_fp8_ue8m0_quant(
            x, norm_w, eps=1.0e-6, emit_norm=True
        )
        assert got_norm is not None
        quant_of_norm_q, quant_of_norm_s = sgl_per_token_group_quant_fp8(
            got_norm,
            group_size=128,
            eps=1.0e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        self.assertTrue(
            torch.equal(got_q.view(torch.uint8), quant_of_norm_q.view(torch.uint8))
        )
        self.assertTrue(torch.equal(got_s, quant_of_norm_s))

        got_none, (q_only, s_only) = fused_rmsnorm_fp8_ue8m0_quant(
            x, norm_w, eps=1.0e-6, emit_norm=False
        )
        self.assertIsNone(got_none)
        self.assertTrue(torch.equal(q_only.view(torch.uint8), got_q.view(torch.uint8)))
        self.assertTrue(torch.equal(s_only, got_s))

        got_out = torch.empty((rows, n_dim), device="cuda", dtype=torch.bfloat16)
        fp8_gemm_nt(
            (q_only, s_only), (w_fp8, w_scale_int), got_out, disable_ue8m0_cast=False
        )
        from_norm_out = torch.empty((rows, n_dim), device="cuda", dtype=torch.bfloat16)
        fp8_gemm_nt(
            (quant_of_norm_q, quant_of_norm_s),
            (w_fp8, w_scale_int),
            from_norm_out,
            disable_ue8m0_cast=False,
        )
        self.assertTrue(torch.equal(got_out, from_norm_out))
        torch.testing.assert_close(got_out, ref_out, atol=4e-2, rtol=2e-2)

    def test_2d_row_strided_and_alignment_gates(self) -> None:
        eps = 1.0e-6
        torch.manual_seed(11)
        width = 4096
        rows = 19
        stride_aligned = width + 8  # 16B-aligned padded stride (bf16 * 8 = 16B)
        backing = torch.randn(
            (rows, stride_aligned), device="cuda", dtype=torch.bfloat16
        )
        x_strided = backing[:, :width]
        self.assertFalse(x_strided.is_contiguous())
        w = (torch.rand(width, device="cuda", dtype=torch.bfloat16) + 0.5).contiguous()
        self.assertTrue(can_fuse_qnorm_fp8_ue8m0_quant(x_strided, w))

        got_norm, (got_q, got_s) = fused_rmsnorm_fp8_ue8m0_quant(
            x_strided, w, eps=eps, emit_norm=True
        )
        assert got_norm is not None
        self.assertTrue(got_norm.is_contiguous())
        self.assertTrue(got_q.is_contiguous())
        ref_norm, (ref_q, ref_s) = fused_rmsnorm_fp8_ue8m0_quant(
            x_strided.contiguous(), w, eps=eps, emit_norm=True
        )
        assert ref_norm is not None
        self.assertTrue(torch.equal(got_norm, ref_norm))
        self.assertTrue(torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8)))
        self.assertTrue(torch.equal(got_s, ref_s))

        # Unaligned data_ptr (odd bf16 offset) or unaligned row stride fails gate,
        # while direct execution still stays safe on masked Triton loads.
        unaligned_ptr = torch.randn(
            (rows, width + 1), device="cuda", dtype=torch.bfloat16
        )[:, 1:]
        self.assertNotEqual(unaligned_ptr.data_ptr() % 16, 0)
        self.assertFalse(can_fuse_qnorm_fp8_ue8m0_quant(unaligned_ptr, w))

        unaligned_stride = torch.randn(
            (rows, width + 2), device="cuda", dtype=torch.bfloat16
        )[:, :width]
        self.assertFalse(can_fuse_qnorm_fp8_ue8m0_quant(unaligned_stride, w))
        n_u, (q_u, s_u) = fused_rmsnorm_fp8_ue8m0_quant(
            unaligned_stride, w, eps=eps, emit_norm=True
        )
        n_c, (q_c, s_c) = fused_rmsnorm_fp8_ue8m0_quant(
            unaligned_stride.contiguous(), w, eps=eps, emit_norm=True
        )
        assert n_u is not None and n_c is not None
        self.assertTrue(torch.equal(n_u, n_c))
        self.assertTrue(torch.equal(q_u.view(torch.uint8), q_c.view(torch.uint8)))
        self.assertTrue(torch.equal(s_u, s_c))

        w_unaligned = torch.randn(width + 1, device="cuda", dtype=torch.bfloat16)[1:]
        self.assertFalse(
            can_fuse_qnorm_fp8_ue8m0_quant(backing[:, :width], w_unaligned)
        )

        # Empty 3D leading batches (M=0) preserve shapes safely without launching.
        for empty_shape in ((0, 4, width), (2, 0, width)):
            x0 = torch.empty(empty_shape, device="cuda", dtype=torch.bfloat16)
            self.assertTrue(can_fuse_qnorm_fp8_ue8m0_quant(x0, w))
            n0, (q0, s0) = fused_rmsnorm_fp8_ue8m0_quant(x0, w, eps=eps, emit_norm=True)
            assert n0 is not None
            self.assertEqual(tuple(n0.shape), empty_shape)
            self.assertEqual(tuple(q0.shape), empty_shape)
            self.assertEqual(tuple(s0.shape), (0, (width // 128 + 3) // 4))

    def test_inplace_out_norm_and_overlap_guards(self) -> None:
        eps = 1.0e-6
        torch.manual_seed(23)
        for rows, width in ((65, 4096), (129, 7168)):
            with self.subTest(rows=rows, width=width):
                x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
                w = (
                    torch.rand(width, device="cuda", dtype=torch.bfloat16) + 0.5
                ).contiguous()
                ref_norm, (ref_q, ref_s) = fused_rmsnorm_fp8_ue8m0_quant(
                    x, w, eps=eps, emit_norm=True
                )
                assert ref_norm is not None
                x_inplace = x.clone()
                got_norm, (got_q, got_s) = fused_rmsnorm_fp8_ue8m0_quant(
                    x_inplace, w, eps=eps, emit_norm=True, out_norm=x_inplace
                )
                assert got_norm is not None
                self.assertEqual(got_norm.data_ptr(), x_inplace.data_ptr())
                self.assertTrue(torch.equal(got_norm, ref_norm))
                self.assertTrue(
                    torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8))
                )
                self.assertTrue(torch.equal(got_s, ref_s))

        # Shifted overlap and cross-buffer aliasing must be rejected cleanly.
        w = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
        buf = torch.randn((17, 1024), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "in-place alias"):
            fused_rmsnorm_fp8_ue8m0_quant(
                buf[:16], w, eps=eps, emit_norm=True, out_norm=buf[1:]
            )
        fp8_view = buf[:16].view(torch.float8_e4m3fn)[:, :1024]
        with self.assertRaisesRegex(ValueError, "overlap"):
            fused_rmsnorm_fp8_ue8m0_quant(
                buf[:16], w, eps=eps, emit_norm=False, out_q=fp8_view
            )

    def test_non_divisible_block_widths(self) -> None:
        eps = 1.0e-6
        torch.manual_seed(37)
        for rows in (1, 17, 130):
            for width in (640, 2688, 7168):
                with self.subTest(rows=rows, width=width):
                    x = torch.randn((rows, width), device="cuda", dtype=torch.bfloat16)
                    w = (
                        torch.rand(width, device="cuda", dtype=torch.bfloat16) + 0.5
                    ).contiguous()
                    got_norm, (got_q, got_s) = fused_rmsnorm_fp8_ue8m0_quant(
                        x, w, eps=eps, emit_norm=True
                    )
                    assert got_norm is not None
                    ref_q, ref_s = sgl_per_token_group_quant_fp8(
                        got_norm,
                        group_size=128,
                        eps=1.0e-4,
                        column_major_scales=True,
                        scale_tma_aligned=True,
                        scale_ue8m0=True,
                    )
                    self.assertTrue(
                        torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8))
                    )
                    # sgl_per_token_group_quant_fp8 writes only active group bytes via
                    # uint8_t* into an uninitialized torch.empty int32 pack buffer, so
                    # compare active group bytes and verify fused zero-pads tail bytes.
                    active_groups = width // 128
                    got_bytes = torch.stack(
                        [(got_s.to(torch.int64) >> (8 * b)) & 0xFF for b in range(4)],
                        dim=-1,
                    ).reshape(rows, -1)
                    ref_bytes = torch.stack(
                        [(ref_s.to(torch.int64) >> (8 * b)) & 0xFF for b in range(4)],
                        dim=-1,
                    ).reshape(rows, -1)
                    self.assertTrue(
                        torch.equal(
                            got_bytes[:, :active_groups], ref_bytes[:, :active_groups]
                        )
                    )
                    if got_bytes.shape[1] > active_groups:
                        self.assertEqual(
                            int(got_bytes[:, active_groups:].max().item()), 0
                        )
        bad_x = torch.randn((8, 130), device="cuda", dtype=torch.bfloat16)
        bad_w = torch.ones((130,), device="cuda", dtype=torch.bfloat16)
        self.assertFalse(can_fuse_qnorm_fp8_ue8m0_quant(bad_x, bad_w))
        with self.assertRaises(ValueError):
            fused_rmsnorm_fp8_ue8m0_quant(bad_x, bad_w, eps=eps)


if __name__ == "__main__":
    unittest.main()
