from unittest import SkipTest, TestCase, main

import torch

MX_BLOCK = 32


def _unpack_packed_scale(scale_packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    k_groups = K // MX_BLOCK
    k_packed = k_groups // 4
    packed = scale_packed[:, :k_packed].to(torch.int32)
    shifts = torch.tensor([0, 8, 16, 24], device=scale_packed.device, dtype=torch.int32)
    scale_u8 = ((packed[:, :, None] >> shifts) & 0xFF).reshape(M, k_groups)
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = q.shape
    q_fp32 = q.to(torch.float32).view(M, K // MX_BLOCK, MX_BLOCK)
    return (q_fp32 * scale[:, :, None]).view(M, K)


class Mxfp8QuantActPackedTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        try:
            import deep_gemm  # noqa: F401
            import flashinfer  # noqa: F401
        except Exception as e:
            raise SkipTest(f"flashinfer/deep_gemm unavailable: {e}")

    def _run_shape(self, M: int, K: int, dtype: torch.dtype) -> None:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

        torch.manual_seed(20260706 + M + K)
        x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 2.0).to(dtype)
        x = x.contiguous()

        q, scale_packed = mxfp8_quant_act_packed(x)
        torch.cuda.synchronize()

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_packed.dtype, torch.int32)
        self.assertEqual(tuple(q.shape), (M, K))
        self.assertEqual(scale_packed.shape[0], M)
        self.assertEqual(scale_packed.shape[1], K // (MX_BLOCK * 4))

        scale = _unpack_packed_scale(scale_packed, M, K)
        dequant = _dequant(q, scale)
        abs_err = (dequant - x.float()).abs()
        rel_err = abs_err / x.float().abs().clamp(min=1e-3)
        self.assertLess(abs_err.max().item(), 1.0)
        self.assertLess(rel_err.mean().item(), 0.08)

    def test_decode_and_prefill_shapes(self) -> None:
        cases = [
            (1, 128, torch.bfloat16),
            (8, 3072, torch.bfloat16),
            (32, 6144, torch.bfloat16),
            (1024, 128, torch.bfloat16),
            (1024, 6144, torch.bfloat16),
        ]
        for case in cases:
            with self.subTest(case=case):
                self._run_shape(*case)

    def test_packer_layout_equivalence(self) -> None:
        """deep_gemm transform (tiled path) vs pack_flashinfer (unfused path).

        Both must produce the SAME physical int32 tensor (shape + strides +
        values) for the GEMM's TMA to read them identically. The benchmark's
        dequant check normalized with .contiguous(), so it could not catch a
        physical-layout divergence — this test compares the raw tensors.
        """
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import pack_mxfp8_scale
        from rtp_llm.models_py.triton_kernels.moe.mxfp8_kernels import (
            pack_flashinfer_mxfp8_scale_triton,
        )

        for M, K in [(1024, 3072), (2048, 6144), (17, 3072)]:
            with self.subTest(M=M, K=K):
                torch.manual_seed(1234 + M + K)
                n_groups = K // MX_BLOCK
                # Random power-of-two UE8M0 scale, shared by both packers.
                e_u8 = torch.randint(
                    110, 136, (M, n_groups), device="cuda", dtype=torch.uint8
                )
                scale_fp32 = torch.exp2(e_u8.float() - 127.0).contiguous()

                pack_dg = pack_mxfp8_scale(scale_fp32, mn=M, k=K)  # tiled path
                pack_fi = pack_flashinfer_mxfp8_scale_triton(
                    e_u8.contiguous(), M, K
                )  # unfused path
                torch.cuda.synchronize()

                print(f"\n[M={M} K={K}] "
                      f"deep_gemm: shape={tuple(pack_dg.shape)} "
                      f"stride={pack_dg.stride()} dtype={pack_dg.dtype} "
                      f"contig={pack_dg.is_contiguous()}")
                print(f"[M={M} K={K}] "
                      f"flashinfer: shape={tuple(pack_fi.shape)} "
                      f"stride={pack_fi.stride()} dtype={pack_fi.dtype} "
                      f"contig={pack_fi.is_contiguous()}")

                same_shape = tuple(pack_dg.shape) == tuple(pack_fi.shape)
                same_stride = pack_dg.stride() == pack_fi.stride()
                same_vals = same_shape and torch.equal(
                    pack_dg.to(torch.int32), pack_fi.to(torch.int32)
                )
                # Compare raw storage bytes too (physical layout, TMA-visible).
                raw_dg = pack_dg.flatten()  # honors strides -> logical order
                same_logical = same_shape and torch.equal(
                    pack_dg.reshape(-1).to(torch.int32),
                    pack_fi.reshape(-1).to(torch.int32),
                )
                print(f"[M={M} K={K}] same_shape={same_shape} "
                      f"same_stride={same_stride} same_values={same_vals} "
                      f"same_logical={same_logical}")
                del raw_dg
                self.assertTrue(
                    same_shape and same_stride and same_vals,
                    f"packer layout mismatch at M={M} K={K}: "
                    f"shape {tuple(pack_dg.shape)} vs {tuple(pack_fi.shape)}, "
                    f"stride {pack_dg.stride()} vs {pack_fi.stride()}, "
                    f"values_equal={same_vals}",
                )

    def test_tiled_vs_unfused_raw_equivalence(self) -> None:
        """tiled kernel vs unfused (swiglu_oai_torch + flashinfer) on the SAME
        [gate|up] input, comparing RAW fp8 bytes + RAW int32 scale.

        Stresses small-magnitude / boundary groups (the absmax-floor regime the
        random-data benchmark never hit) to see whether the tiled kernel's
        quantization diverges from the production unfused path.
        """
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (
            mxfp8_quant_act_packed,
            pack_mxfp8_scale,
        )
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul_mxfp8_quant_tiled_fwd,
        )
        from rtp_llm.models_py.triton_kernels.common.swiglu_oai import (
            swiglu_oai_torch,
        )

        alpha, limit = 1.702, 7.0
        for T, H, scale_mode in [
            (1024, 3072, "normal"),
            (1024, 3072, "tiny_rows"),
            (1024, 3072, "wide_dynamic"),
            (2048, 6144, "wide_dynamic"),
        ]:
            with self.subTest(T=T, H=H, scale_mode=scale_mode):
                torch.manual_seed(7 + T + H + len(scale_mode))
                x = torch.randn(T, 2 * H, device="cuda", dtype=torch.float32) * 1.5
                if scale_mode == "tiny_rows":
                    # Every 4th row block scaled to ~1e-5 magnitude.
                    x[::4] *= 1e-5
                elif scale_mode == "wide_dynamic":
                    # Per-row random magnitude spanning ~1e-6 .. 1e2.
                    mag = torch.exp2(
                        torch.randint(-20, 7, (T, 1), device="cuda").float()
                    )
                    x = x * mag
                x = x.to(torch.bfloat16).contiguous()

                # Unfused (production) path.
                activated = swiglu_oai_torch(x, alpha, limit, gate_first=True)
                q_u, s_u = mxfp8_quant_act_packed(activated.contiguous())

                # Tiled path + deep_gemm pack (what the dispatch did).
                q_t, s_t_fp32 = silu_and_mul_mxfp8_quant_tiled_fwd(
                    x, quant_group_size=MX_BLOCK,
                    gemm1_alpha=alpha, gemm1_clamp_limit=limit,
                )
                s_t = pack_mxfp8_scale(s_t_fp32, mn=T, k=H)
                torch.cuda.synchronize()

                q_u_b = q_u.view(torch.uint8)
                q_t_b = q_t.view(torch.uint8)
                fp8_mismatch = (q_u_b != q_t_b).sum().item()
                fp8_total = q_u_b.numel()
                scale_mismatch = (
                    s_u.reshape(-1).to(torch.int32)
                    != s_t.reshape(-1).to(torch.int32)
                ).sum().item()
                scale_total = s_u.numel()
                print(f"\n[T={T} H={H} {scale_mode}] "
                      f"fp8 mismatch={fp8_mismatch}/{fp8_total} "
                      f"({100.0*fp8_mismatch/fp8_total:.3f}%)  "
                      f"scale mismatch={scale_mismatch}/{scale_total} "
                      f"({100.0*scale_mismatch/scale_total:.3f}%)")

    def test_dense_mlp_large_token_fallback_returns_packed_scale(self) -> None:
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
        )

        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16).contiguous()
        q, scale = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
            x,
            quant_group_size=MX_BLOCK,
            scale_ue8m0=False,
            round_to_pow2=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.int32)
        self.assertEqual(tuple(q.shape), (1024, 128))
        self.assertEqual(tuple(scale.shape), (1024, 1))


if __name__ == "__main__":
    main()
