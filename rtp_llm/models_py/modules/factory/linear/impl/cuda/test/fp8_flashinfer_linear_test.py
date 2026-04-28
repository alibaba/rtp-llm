import logging
import os
import unittest

import torch

from rtp_llm.config.quant_config import init_quant_config
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_flashinfer_linear import (
    CudaFp8FlashinferLinear,
    use_flashinfer_fp8_gemm,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.test.utils.bench_util import bench
from rtp_llm.test.utils.numeric_util import calc_diff


def _is_sm90() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = get_sm()
        return major == 9
    except Exception:
        return False


@unittest.skipUnless(_is_sm90(), "CudaFp8FlashinferLinear requires SM90 (Hopper)")
class CudaFp8FlashinferLinearTest(unittest.TestCase):
    """UT + microbenchmark for the flashinfer-backed FP8 swapAB Linear."""

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.device = "cuda"
        logging.getLogger(
            "rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_flashinfer_linear"
        ).setLevel(logging.WARNING)

        # Use a representative qwen3-next-style head proj shape.
        self.K = 4096
        self.N = 4096
        self.scale_K = (self.K + 127) // 128
        self.scale_N = (self.N + 127) // 128
        # Cover the swapAB band (M < 32) plus a few larger sizes.
        self.test_batch_sizes = [1, 4, 8, 16, 24, 31, 32, 64, 128]

        # Reference bf16 weight + per-block FP8.
        self.weight_bf16 = torch.randn(
            (self.N, self.K), dtype=torch.bfloat16, device=self.device
        )
        weight_fp8, weight_scales = per_block_cast_to_fp8(
            self.weight_bf16, use_ue8m0=False
        )
        # Stored layout matches what _postprocess yields: (K, N) / (scale_K, scale_N).
        self.weight_fp8_kn = weight_fp8.reshape(self.K, self.N)
        self.weight_scales_kn = weight_scales.reshape(self.scale_K, self.scale_N)
        self.bias = torch.randn(self.N, dtype=torch.bfloat16, device=self.device)

    def _make_linear(self, with_bias: bool = False) -> CudaFp8FlashinferLinear:
        return CudaFp8FlashinferLinear(
            weight=self.weight_fp8_kn.clone(),
            weight_scales=self.weight_scales_kn.clone(),
            input_scales=None,
            bias=self.bias.clone() if with_bias else None,
            quant_config=init_quant_config("FP8_PER_BLOCK"),
        )

    # ---------------- can_handle ----------------
    def test_can_handle_default_on_sm90(self):
        prev = os.environ.pop("RTP_LLM_USE_FLASHINFER_FP8_GEMM", None)
        try:
            self.assertTrue(use_flashinfer_fp8_gemm())
            self.assertTrue(
                CudaFp8FlashinferLinear.can_handle(
                    init_quant_config("FP8_PER_BLOCK"),
                    self.weight_fp8_kn,
                    self.weight_scales_kn,
                )
            )
        finally:
            if prev is not None:
                os.environ["RTP_LLM_USE_FLASHINFER_FP8_GEMM"] = prev

    def test_can_handle_opt_out(self):
        prev = os.environ.get("RTP_LLM_USE_FLASHINFER_FP8_GEMM")
        os.environ["RTP_LLM_USE_FLASHINFER_FP8_GEMM"] = "0"
        try:
            self.assertFalse(use_flashinfer_fp8_gemm())
            self.assertFalse(
                CudaFp8FlashinferLinear.can_handle(
                    init_quant_config("FP8_PER_BLOCK"),
                    self.weight_fp8_kn,
                    self.weight_scales_kn,
                )
            )
        finally:
            if prev is None:
                os.environ.pop("RTP_LLM_USE_FLASHINFER_FP8_GEMM", None)
            else:
                os.environ["RTP_LLM_USE_FLASHINFER_FP8_GEMM"] = prev

    def test_module_creation_shapes(self):
        linear = self._make_linear(with_bias=True)
        self.assertEqual(linear.K, self.K)
        self.assertEqual(linear.N, self.N)
        self.assertEqual(linear.weight.shape, (self.N, self.K))
        self.assertEqual(linear.weight_scales.shape, (self.scale_N, self.scale_K))
        self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(linear.weight_scales.dtype, torch.float32)

    # ---------------- input validation ----------------
    def test_input_validation(self):
        linear = self._make_linear()
        with self.assertRaises(ValueError):
            linear(torch.randn(self.K, dtype=torch.bfloat16, device=self.device))
        with self.assertRaises(ValueError):
            linear(torch.randn(8, self.K, 2, dtype=torch.bfloat16, device=self.device))
        with self.assertRaises(ValueError):
            linear(torch.randn(8, self.K + 1, dtype=torch.bfloat16, device=self.device))
        with self.assertRaises(ValueError):
            linear(torch.randn(8, self.K, dtype=torch.float32, device=self.device))

    # ---------------- output shape/dtype ----------------
    def test_output_shape_and_dtype(self):
        linear = self._make_linear()
        for m in self.test_batch_sizes:
            with self.subTest(batch_size=m):
                x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
                y = linear(x)
                self.assertEqual(y.shape, (m, self.N))
                self.assertEqual(y.dtype, torch.bfloat16)
                self.assertEqual(y.device.type, "cuda")
                self.assertFalse(torch.isnan(y).any())
                self.assertFalse(torch.isinf(y).any())

    # ---------------- correctness vs bf16 ref ----------------
    def test_correctness_vs_reference(self):
        linear = self._make_linear()
        for m in self.test_batch_sizes:
            with self.subTest(batch_size=m):
                x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
                y = linear(x)
                ref = (x.float() @ self.weight_bf16.float().t()).to(torch.bfloat16)
                diff = calc_diff(y, ref)
                # Same tolerance as deepgemm test.
                self.assertLess(diff, 0.0025, f"diff={diff} m={m}")

    def test_bias_handling(self):
        linear = self._make_linear(with_bias=True)
        for m in [1, 16, 32, 64]:
            with self.subTest(batch_size=m):
                x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
                y = linear(x)
                ref = (x.float() @ self.weight_bf16.float().t() + self.bias.float()).to(
                    torch.bfloat16
                )
                diff = calc_diff(y, ref)
                self.assertLess(diff, 0.0025, f"diff={diff} m={m}")

    # ---------------- numeric parity vs deepgemm ----------------
    def test_parity_vs_deepgemm(self):
        flash_linear = self._make_linear()
        deep_linear = CudaFp8DeepGEMMLinear(
            weight=self.weight_fp8_kn.clone(),
            weight_scales=self.weight_scales_kn.clone(),
            input_scales=None,
            bias=None,
            quant_config=init_quant_config("FP8_PER_BLOCK"),
        )
        for m in self.test_batch_sizes:
            with self.subTest(batch_size=m):
                x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
                y_flash = flash_linear(x)
                y_deep = deep_linear(x)
                # Both kernels are FP8 matmuls — they should agree closely.
                diff = calc_diff(y_flash, y_deep)
                self.assertLess(
                    diff, 0.005, f"flashinfer vs deepgemm diff={diff} m={m}"
                )

    # ---------------- microbenchmark (vs deepgemm) ----------------
    @unittest.skipUnless(
        os.environ.get("RTP_LLM_RUN_BENCH") == "1",
        "Set RTP_LLM_RUN_BENCH=1 to run the microbenchmark.",
    )
    def test_bench_vs_deepgemm(self):
        flash_linear = self._make_linear()
        deep_linear = CudaFp8DeepGEMMLinear(
            weight=self.weight_fp8_kn.clone(),
            weight_scales=self.weight_scales_kn.clone(),
            input_scales=None,
            bias=None,
            quant_config=init_quant_config("FP8_PER_BLOCK"),
        )
        bench_batch_sizes = [1, 2, 4, 8, 16, 24, 31, 32, 64, 128, 256]
        print(
            f"\n[bench] FP8 Linear M=*, N={self.N}, K={self.K} on SM90 "
            f"(flashinfer swapAB vs deepgemm)"
        )
        print(f"{'M':>6} | {'flash(ms)':>10} | {'deep(ms)':>10} | {'speedup':>8}")
        for m in bench_batch_sizes:
            x = torch.randn(m, self.K, dtype=torch.bfloat16, device=self.device)
            t_flash, _, _ = bench(lambda: flash_linear(x), num_warmups=20, num_tests=30)
            t_deep, _, _ = bench(lambda: deep_linear(x), num_warmups=20, num_tests=30)
            speedup = t_deep / t_flash if t_flash > 0 else float("inf")
            print(
                f"{m:>6} | {t_flash * 1e3:>10.4f} | {t_deep * 1e3:>10.4f} | "
                f"{speedup:>8.2f}x"
            )

    # ---------------- microbenchmark on smoke-test shapes ----------------
    # qwen3_next_fp8_tp2_mtp_pd_sep decode-path FP8 GEMM shapes (per-tp_size=2):
    #   N=17408 K=5120  (gate_up_proj-like)
    #   N=5120  K=8704  (down_proj-like)
    #   N=8192  K=5120
    #   N=5120  K=3072
    #   N=4096  K=5120
    #   N=3072  K=5120
    #
    # Two views are reported:
    #   (a) "kernel-only" — FP8 GEMM kernel time with input pre-quantized
    #       (matches what the chrome trace shows for the kernel itself).
    #   (b) "forward"     — full Linear.forward including per-token quant
    #       + Python/TVM-FFI dispatch.
    @unittest.skipUnless(
        os.environ.get("RTP_LLM_RUN_BENCH") == "1",
        "Set RTP_LLM_RUN_BENCH=1 to run the smoke-shape microbenchmark.",
    )
    def test_bench_smoke_shapes_vs_deepgemm(self):
        smoke_shapes = [
            (17408, 5120),
            (5120, 8704),
            (8192, 5120),
            (5120, 3072),
            (4096, 5120),
            (3072, 5120),
        ]
        bench_ms = [8, 16, 32, 64]

        # ----- (a) kernel-only header -----
        print(
            "\n[bench] qwen3-next FP8 PD-Sep decode shapes — "
            "flashinfer swapAB vs deepgemm on SM90 (H20)"
        )
        print("\n--- Kernel-only (input pre-quantized, no per-token quant kernel) ---")
        kheader = (
            f"{'N':>6} | {'K':>6} | {'M':>4} | "
            f"{'flash(us)':>10} | {'deep(us)':>10} | {'speedup':>8}"
        )
        print(kheader)
        print("-" * len(kheader))

        agg_flash_k = {m: 0.0 for m in bench_ms}
        agg_deep_k = {m: 0.0 for m in bench_ms}
        agg_flash_f = {m: 0.0 for m in bench_ms}
        agg_deep_f = {m: 0.0 for m in bench_ms}

        # Cache resources per-shape so we can run both views back-to-back.
        per_shape = []
        for N, K in smoke_shapes:
            weight_bf16 = torch.randn((N, K), dtype=torch.bfloat16, device=self.device)
            weight_fp8, weight_scales = per_block_cast_to_fp8(
                weight_bf16, use_ue8m0=False
            )
            weight_fp8_kn = weight_fp8.reshape(K, N)
            weight_scales_kn = weight_scales.reshape((K + 127) // 128, (N + 127) // 128)
            flash_linear = CudaFp8FlashinferLinear(
                weight=weight_fp8_kn.clone(),
                weight_scales=weight_scales_kn.clone(),
                input_scales=None,
                bias=None,
                quant_config=init_quant_config("FP8_PER_BLOCK"),
            )
            deep_linear = CudaFp8DeepGEMMLinear(
                weight=weight_fp8_kn.clone(),
                weight_scales=weight_scales_kn.clone(),
                input_scales=None,
                bias=None,
                quant_config=init_quant_config("FP8_PER_BLOCK"),
            )
            per_shape.append((N, K, flash_linear, deep_linear))

        # ----- (a) kernel-only loop -----
        from flashinfer.gemm import fp8_blockscale_gemm_sm90 as _fi_gemm

        for N, K, flash_linear, deep_linear in per_shape:
            for m in bench_ms:
                x = torch.randn(m, K, dtype=torch.bfloat16, device=self.device)
                # Pre-quantize input once; reuse the same fp8/scale tensors.
                input_fp8, input_scales = sgl_per_token_group_quant_fp8(
                    x,
                    group_size=128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=False,
                )
                output = torch.empty(m, N, dtype=torch.bfloat16, device=self.device)

                def run_flash():
                    _fi_gemm(
                        input_fp8,
                        flash_linear.weight,
                        input_scales,
                        flash_linear.weight_scales,
                        out=output,
                    )

                def run_deep():
                    fp8_gemm_nt(
                        (input_fp8, input_scales),
                        (deep_linear.weight, deep_linear.weight_scales),
                        output,
                        c=None,
                        disable_ue8m0_cast=True,
                    )

                # Warm both (flashinfer JITs the swapAB cubin first time).
                for _ in range(5):
                    run_flash()
                    run_deep()
                t_flash, _, _ = bench(run_flash, num_warmups=30, num_tests=50)
                t_deep, _, _ = bench(run_deep, num_warmups=30, num_tests=50)
                speedup = t_deep / t_flash if t_flash > 0 else float("inf")
                agg_flash_k[m] += t_flash
                agg_deep_k[m] += t_deep
                print(
                    f"{N:>6} | {K:>6} | {m:>4} | "
                    f"{t_flash * 1e6:>10.2f} | {t_deep * 1e6:>10.2f} | "
                    f"{speedup:>7.2f}x"
                )

        # ----- (b) full forward loop -----
        print("\n--- Forward (per-token quant + GEMM + Python/TVM-FFI dispatch) ---")
        print(kheader)
        print("-" * len(kheader))
        for N, K, flash_linear, deep_linear in per_shape:
            for m in bench_ms:
                x = torch.randn(m, K, dtype=torch.bfloat16, device=self.device)
                for _ in range(5):
                    flash_linear(x)
                    deep_linear(x)
                t_flash, _, _ = bench(
                    lambda: flash_linear(x), num_warmups=30, num_tests=50
                )
                t_deep, _, _ = bench(
                    lambda: deep_linear(x), num_warmups=30, num_tests=50
                )
                speedup = t_deep / t_flash if t_flash > 0 else float("inf")
                agg_flash_f[m] += t_flash
                agg_deep_f[m] += t_deep
                print(
                    f"{N:>6} | {K:>6} | {m:>4} | "
                    f"{t_flash * 1e6:>10.2f} | {t_deep * 1e6:>10.2f} | "
                    f"{speedup:>7.2f}x"
                )

        # Per-M aggregate across all shapes.
        for label, af, ad in [
            ("kernel-only", agg_flash_k, agg_deep_k),
            ("forward    ", agg_flash_f, agg_deep_f),
        ]:
            print(f"\n--- Aggregate per-M ({label}) ---")
            agg_header = (
                f"{'M':>4} | {'flash_total(us)':>16} | "
                f"{'deep_total(us)':>15} | {'avg_speedup':>11}"
            )
            print(agg_header)
            print("-" * len(agg_header))
            for m in bench_ms:
                sp = ad[m] / af[m] if af[m] > 0 else float("inf")
                print(
                    f"{m:>4} | {af[m] * 1e6:>16.2f} | "
                    f"{ad[m] * 1e6:>15.2f} | {sp:>10.2f}x"
                )


if __name__ == "__main__":
    unittest.main()
