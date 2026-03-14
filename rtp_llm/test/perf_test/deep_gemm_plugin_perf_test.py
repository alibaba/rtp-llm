"""
Performance and correctness comparison between DeepGemmPlugin (C++ with swap_ab),
deepgemm_wrapper (Python deep_gemm), and flashinfer fp8_blockscale_gemm_sm90.

Test cases:
1. Normal GEMM: m=1-64, k=16384, n=6144
2. Grouped Masked GEMM: expert_num=16, m=16-64, k=6144, n=4096
3. Three-way comparison: Plugin vs Wrapper vs FlashInfer (normal GEMM)

Correctness is verified in two dimensions:
  (a) All paths vs bf16 reference (torch.matmul) — ensures none is silently wrong.
  (b) Cross-path pairwise — ensures implementations agree with each other.
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import logging

from rtp_llm.test.utils.bench_util import bench

logging.basicConfig(level=logging.INFO)

NSYS_MODE = os.environ.get("NSYS_PROFILE", "0") == "1"
NSYS_WARMUPS = int(os.environ.get("NSYS_WARMUPS", "5"))
NSYS_ITERS = int(os.environ.get("NSYS_ITERS", "20"))


def nvtx_bench(name: str, fn, num_warmups: int = 0, num_iters: int = 0):
    """Run *fn* with NVTX ranges so nsys timeline shows clear per-call markers.

    Returns (avg_sec, min_sec, max_sec) measured via CUDA events, same as bench().
    """
    num_warmups = num_warmups if num_warmups > 0 else NSYS_WARMUPS
    num_iters = num_iters if num_iters > 0 else NSYS_ITERS

    for _ in range(num_warmups):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_iters)]

    torch.cuda.nvtx.range_push(f"BENCH_{name}")
    for i in range(num_iters):
        torch.cuda.nvtx.range_push(f"{name}#{i}")
        start_events[i].record()
        fn()
        end_events[i].record()
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) / 1e3 for s, e in zip(start_events, end_events)]
    import numpy as np

    times = np.array(times)
    return float(np.average(times)), float(np.min(times)), float(np.max(times))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pad_to_multiple(x: int, multiple: int) -> int:
    return ((x + multiple - 1) // multiple) * multiple


def plugin_padding_size(m: int) -> int:
    """Replicate DeepGemmPlugin::getPaddingSize logic for swap_ab case."""
    return 16 if m < 16 else 8


def plugin_pad_m(m: int) -> int:
    return pad_to_multiple(m, plugin_padding_size(m))


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().float()
    b_f = b.flatten().float()
    denom = torch.norm(a_f) * torch.norm(b_f)
    if denom.item() == 0:
        return 1.0 if torch.norm(a_f - b_f).item() == 0 else 0.0
    return (torch.dot(a_f, b_f) / denom).item()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _zero_pad_2d(data: torch.Tensor, target_m: int) -> torch.Tensor:
    m, k = data.shape
    if m >= target_m:
        return data[:target_m].contiguous()
    padded = torch.zeros(target_m, k, dtype=data.dtype, device=data.device)
    padded[:m] = data
    return padded


def _zero_pad_3d(data: torch.Tensor, target_m: int) -> torch.Tensor:
    g, m, k = data.shape
    if m >= target_m:
        return data[:, :target_m].contiguous()
    padded = torch.zeros(g, target_m, k, dtype=data.dtype, device=data.device)
    padded[:, :m] = data
    return padded


def _fp8_ref_matmul_2d(
    data_fp8: torch.Tensor,
    scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference bf16 matmul: dequant both sides then torch.matmul.

    data_fp8:      [m, k]     fp8
    scale:         [m, k/128] fp32   (per-token-per-128)
    weight_fp8:    [n, k]     fp8
    weight_scale:  [n/128, k/128] fp32  (per-block 128x128)
    Returns:       [m, n]     bf16
    """
    m, k = data_fp8.shape
    n = weight_fp8.shape[0]

    # dequant input: repeat scale along k
    inp_f = data_fp8.float()  # [m, k]
    inp_s = scale.float().repeat_interleave(128, dim=1)[:, :k]  # [m, k]
    inp_deq = inp_f * inp_s  # [m, k]

    # dequant weight: repeat scale along both n and k
    w_f = weight_fp8.float()  # [n, k]
    w_s = (
        weight_scale.float()
        .repeat_interleave(128, dim=0)[:n]
        .repeat_interleave(128, dim=1)[:, :k]
    )
    w_deq = w_f * w_s  # [n, k]

    return (inp_deq @ w_deq.t()).to(torch.bfloat16)


def _fp8_ref_matmul_3d(
    data_fp8: torch.Tensor,
    scale: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Reference bf16 matmul for grouped [g, m, k] layout."""
    g = data_fp8.shape[0]
    results = []
    for i in range(g):
        results.append(
            _fp8_ref_matmul_2d(data_fp8[i], scale[i], weight_fp8[i], weight_scale[i])
        )
    return torch.stack(results)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class DeepGemmPluginPerfTest(unittest.TestCase):
    """Performance + correctness comparison: DeepGemmPlugin vs deepgemm_wrapper."""

    # Cosine > 0.999 → PASS for cross-path comparison (plugin vs wrapper)
    CROSS_COS_THR = 0.999
    # Cosine > 0.99 → PASS for each path vs bf16 reference
    # (FP8 quantization error makes this naturally lower)
    REF_COS_THR = 0.99

    @classmethod
    def setUpClass(cls):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import init_swapab_once
        from rtp_llm.ops import HWKernelConfig

        config = HWKernelConfig()
        config.deep_gemm_use_swap_ab = True
        init_swapab_once(config)
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:
            raise unittest.SkipTest(f"Requires SM90+, got SM{cap[0]}{cap[1]}")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _get_plugin_ops(self):
        try:
            from librtp_compute_ops import rtp_llm_ops

            return rtp_llm_ops.deep_gemm_fp8, rtp_llm_ops.deep_gemm_grouped_fp8_masked
        except (ImportError, AttributeError) as e:
            self.skipTest(f"DeepGemmPlugin pybind not available: {e}")

    def _get_wrapper_ops(self):
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                fp8_gemm_nt,
                has_deep_gemm,
                m_grouped_fp8_gemm_nt_masked,
            )

            if not has_deep_gemm():
                self.skipTest("deep_gemm package not available")
            return fp8_gemm_nt, m_grouped_fp8_gemm_nt_masked
        except ImportError as e:
            self.skipTest(f"deepgemm_wrapper not available: {e}")

    # ------------------------------------------------------------------
    # Normal GEMM
    # ------------------------------------------------------------------
    def test_normal_gemm_perf(self):
        """Normal FP8 GEMM: perf + correctness for selected m values, k=16384, n=6144.

        Plugin now accepts bf16 input and does quantization + padding internally.
        """
        plugin_fp8, _ = self._get_plugin_ops()
        wrapper_fp8, _ = self._get_wrapper_ops()

        k, n = 16384, 6144
        m_values = [1, 8, 16, 24, 32, 40, 48, 56, 64]

        weight_data = torch.randn(n, k, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(n // 128, k // 128, dtype=torch.float32, device="cuda") * 0.01
            + 0.001
        )

        header = (
            f"{'m':>4} | {'Plugin(ms)':>11} | {'W_pad':>5} | {'Wrap(ms)':>9} "
            f"| {'Speed':>6} | {'P_vs_W':>7} | {'P_vs_R':>7} | {'W_vs_R':>7}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print(f"Normal FP8 GEMM (plugin takes bf16): k={k}, n={n}")
        print(header)
        print("-" * len(header))

        all_pass = True

        for m in m_values:
            # ---- Shared bf16 input ----
            raw_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

            # ---- Reference (bf16 matmul with weight dequant) ----
            w_deq = _dequant_weight(weight_data, weight_scale)
            ref_out = (raw_bf16.float() @ w_deq.t()).to(torch.bfloat16)

            # ---- Plugin: takes bf16 input, quantize+pad+gemm in C++, returns output ----
            output_p = plugin_fp8(raw_bf16, weight_data, weight_scale)
            torch.cuda.synchronize()

            # ---- Wrapper: quantize in Python, gemm via deep_gemm lib ----
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            w_pad = pad_to_multiple(m, 64)
            w_bf16 = _zero_pad_2d(raw_bf16, w_pad)
            w_data_fp8, w_scale_fp32 = sgl_per_token_group_quant_fp8(
                w_bf16,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            output_w = torch.zeros(w_pad, n, dtype=torch.bfloat16, device="cuda")
            wrapper_fp8(
                (w_data_fp8, w_scale_fp32), (weight_data, weight_scale), output_w
            )
            torch.cuda.synchronize()

            # ---- Correctness metrics ----
            p_valid = output_p
            w_valid = output_w[:m]
            cos_pw = cosine_sim(p_valid, w_valid)
            cos_pr = cosine_sim(p_valid, ref_out)
            cos_wr = cosine_sim(w_valid, ref_out)

            ok_pw = cos_pw >= self.CROSS_COS_THR
            ok_pr = cos_pr >= self.REF_COS_THR
            ok_wr = cos_wr >= self.REF_COS_THR
            if not (ok_pw and ok_pr and ok_wr):
                all_pass = False

            # ---- Performance ----
            def run_plugin(_bf16=raw_bf16):
                plugin_fp8(_bf16, weight_data, weight_scale)

            def run_wrapper(_d=w_data_fp8, _s=w_scale_fp32, _o=output_w):
                wrapper_fp8((_d, _s), (weight_data, weight_scale), _o)

            if NSYS_MODE:
                t_p, _, _ = nvtx_bench(f"plugin_normal_m{m}", run_plugin)
                t_w, _, _ = nvtx_bench(f"wrapper_normal_m{m}", run_wrapper)
            else:
                t_p, _, _ = bench(run_plugin, num_warmups=30, num_tests=100)
                t_w, _, _ = bench(run_wrapper, num_warmups=30, num_tests=100)
            spd = t_w / t_p if t_p > 0 else float("inf")

            print(
                f"{m:>4} | {t_p*1000:>11.4f} | {w_pad:>5} | {t_w*1000:>9.4f} "
                f"| {spd:>5.2f}x | {cos_pw:>7.5f} | {cos_pr:>7.5f} | {cos_wr:>7.5f}"
                f"  {'PASS' if (ok_pw and ok_pr and ok_wr) else 'FAIL'}"
            )

        print(sep)
        self.assertTrue(all_pass, "Normal GEMM correctness check failed")

    # ------------------------------------------------------------------
    # Grouped Masked GEMM
    # ------------------------------------------------------------------
    def test_grouped_masked_gemm_perf(self):
        """Grouped Masked FP8 GEMM: perf + correctness for expert_num=16, selected m values, k=7168, n=4096."""
        _, plugin_masked = self._get_plugin_ops()
        _, wrapper_masked = self._get_wrapper_ops()

        num_groups = 16
        k, n = 6144, 4096
        m_values = [16, 24, 32, 40, 48, 56, 64]
        weight_data = torch.randn(
            num_groups, n, k, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn)
        weight_scale = (
            torch.rand(
                num_groups, n // 128, k // 128, dtype=torch.float32, device="cuda"
            )
            * 0.01
            + 0.001
        )

        header = (
            f"{'m':>4} | {'P_pad':>5} | {'Plugin(ms)':>11} | {'W_pad':>5} | {'Wrap(ms)':>9} "
            f"| {'Speed':>6} | {'P_vs_W':>7} | {'P_vs_R':>7} | {'W_vs_R':>7}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print(f"Grouped Masked FP8 GEMM: groups={num_groups}, k={k}, n={n}")
        print(header)
        print("-" * len(header))

        all_pass = True

        for m in m_values:
            # ---- Shared raw input ----
            raw_data = torch.randn(
                num_groups, m, k, dtype=torch.bfloat16, device="cuda"
            ).to(torch.float8_e4m3fn)
            raw_scale = (
                torch.rand(num_groups, m, k // 128, dtype=torch.float32, device="cuda")
                * 0.01
                + 0.001
            )

            # ---- Reference (bf16 matmul) ----
            ref_out = _fp8_ref_matmul_3d(raw_data, raw_scale, weight_data, weight_scale)

            # ---- Plugin: single correctness run ----
            p_pad = plugin_pad_m(m)
            p_data = _zero_pad_3d(raw_data, p_pad)
            p_scale = _zero_pad_3d(raw_scale, p_pad)
            output_p = torch.zeros(
                num_groups, p_pad, n, dtype=torch.bfloat16, device="cuda"
            )
            masked_m_p = torch.full((num_groups,), m, dtype=torch.int32, device="cuda")
            plugin_masked(
                p_data, p_scale, weight_data, weight_scale, output_p, masked_m_p, p_pad
            )
            torch.cuda.synchronize()

            # ---- Wrapper: single correctness run ----
            w_pad = pad_to_multiple(m, 64)
            w_data = _zero_pad_3d(raw_data, w_pad)
            w_scale = _zero_pad_3d(raw_scale, w_pad)
            output_w = torch.zeros(
                num_groups, w_pad, n, dtype=torch.bfloat16, device="cuda"
            )
            masked_m_w = torch.full((num_groups,), m, dtype=torch.int32, device="cuda")
            wrapper_masked(
                (w_data, w_scale),
                (weight_data, weight_scale),
                output_w,
                masked_m_w,
                w_pad,
            )
            torch.cuda.synchronize()

            # ---- Correctness metrics ----
            p_valid = output_p[:, :m, :]
            w_valid = output_w[:, :m, :]
            cos_pw = cosine_sim(p_valid, w_valid)
            cos_pr = cosine_sim(p_valid, ref_out)
            cos_wr = cosine_sim(w_valid, ref_out)

            ok_pw = cos_pw >= self.CROSS_COS_THR
            ok_pr = cos_pr >= self.REF_COS_THR
            ok_wr = cos_wr >= self.REF_COS_THR
            if not (ok_pw and ok_pr and ok_wr):
                all_pass = False

            # ---- Performance ----
            def run_plugin(
                _d=p_data,
                _s=p_scale,
                _o=output_p,
                _mm=masked_m_p,
                _pm=p_pad,
            ):
                plugin_masked(_d, _s, weight_data, weight_scale, _o, _mm, _pm)

            def run_wrapper(
                _d=w_data,
                _s=w_scale,
                _o=output_w,
                _mm=masked_m_w,
                _pm=w_pad,
            ):
                wrapper_masked((_d, _s), (weight_data, weight_scale), _o, _mm, _pm)

            if NSYS_MODE:
                t_p, _, _ = nvtx_bench(f"plugin_masked_m{m}", run_plugin)
                t_w, _, _ = nvtx_bench(f"wrapper_masked_m{m}", run_wrapper)
            else:
                t_p, _, _ = bench(run_plugin, num_warmups=30, num_tests=100)
                t_w, _, _ = bench(run_wrapper, num_warmups=30, num_tests=100)
            spd = t_w / t_p if t_p > 0 else float("inf")

            print(
                f"{m:>4} | {p_pad:>5} | {t_p*1000:>11.4f} | {w_pad:>5} | {t_w*1000:>9.4f} "
                f"| {spd:>5.2f}x | {cos_pw:>7.5f} | {cos_pr:>7.5f} | {cos_wr:>7.5f}"
                f"  {'PASS' if (ok_pw and ok_pr and ok_wr) else 'FAIL'}"
            )

        print(sep)
        self.assertTrue(all_pass, "Grouped Masked GEMM correctness check failed")


class FlashInferComparisonTest(unittest.TestCase):
    """Three-way perf + correctness: Plugin vs Wrapper vs FlashInfer fp8_blockscale_gemm_sm90."""

    CROSS_COS_THR = 0.999
    REF_COS_THR = 0.99

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:
            raise unittest.SkipTest(f"Requires SM90+, got SM{cap[0]}{cap[1]}")
        try:
            from flashinfer.gemm import fp8_blockscale_gemm_sm90 as _

            cls._has_flashinfer = True
        except ImportError:
            cls._has_flashinfer = False
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import init_swapab_once
            from rtp_llm.ops import HWKernelConfig

            config = HWKernelConfig()
            config.deep_gemm_use_swap_ab = True
            init_swapab_once(config)
        except Exception:
            pass
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _get_flashinfer_op(self):
        if not self._has_flashinfer:
            self.skipTest("flashinfer not available")
        from flashinfer.gemm import fp8_blockscale_gemm_sm90

        return fp8_blockscale_gemm_sm90

    def _get_plugin_op(self):
        try:
            from librtp_compute_ops import rtp_llm_ops

            return rtp_llm_ops.deep_gemm_fp8
        except (ImportError, AttributeError) as e:
            self.skipTest(f"DeepGemmPlugin pybind not available: {e}")

    def _get_wrapper_op(self):
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                fp8_gemm_nt,
                has_deep_gemm,
            )

            if not has_deep_gemm():
                self.skipTest("deep_gemm package not available")
            return fp8_gemm_nt
        except ImportError as e:
            self.skipTest(f"deepgemm_wrapper not available: {e}")

    def test_normal_gemm_three_way(self):
        """Three-way comparison: Plugin vs Wrapper vs FlashInfer for normal FP8 GEMM.

        All three accept bf16 input + fp8 weight (per-block 128x128 scale).
        FlashInfer internally quantizes bf16 input, same as Plugin.
        """
        flashinfer_gemm = self._get_flashinfer_op()
        plugin_fp8 = self._get_plugin_op()
        wrapper_fp8 = self._get_wrapper_op()

        k, n = 16384, 6144
        m_values = [1, 4, 8, 16, 24, 32, 48, 64]

        weight_data = torch.randn(n, k, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        weight_scale = (
            torch.rand(n // 128, k // 128, dtype=torch.float32, device="cuda") * 0.01
            + 0.001
        )

        header = (
            f"{'m':>4} | {'Plugin(ms)':>11} | {'Wrap(ms)':>9} | {'FI(ms)':>9} "
            f"| {'P/FI':>5} | {'W/FI':>5} "
            f"| {'P_W':>7} | {'P_FI':>7} | {'W_FI':>7} "
            f"| {'P_R':>7} | {'W_R':>7} | {'FI_R':>7}"
        )
        sep = "=" * len(header)
        print(f"\n{sep}")
        print(f"Three-way FP8 GEMM (bf16 input): k={k}, n={n}")
        print(header)
        print("-" * len(header))

        all_pass = True

        for m in m_values:
            raw_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")

            w_deq = _dequant_weight(weight_data, weight_scale)
            ref_out = (raw_bf16.float() @ w_deq.t()).to(torch.bfloat16)

            # ---- Plugin ----
            output_p = plugin_fp8(raw_bf16, weight_data, weight_scale)
            torch.cuda.synchronize()

            # ---- Wrapper ----
            from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
                sgl_per_token_group_quant_fp8,
            )

            w_pad = pad_to_multiple(m, 64)
            w_bf16 = _zero_pad_2d(raw_bf16, w_pad)
            w_data_fp8, w_scale_fp32 = sgl_per_token_group_quant_fp8(
                w_bf16,
                group_size=128,
                eps=1e-4,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            )
            output_w = torch.zeros(w_pad, n, dtype=torch.bfloat16, device="cuda")
            wrapper_fp8(
                (w_data_fp8, w_scale_fp32), (weight_data, weight_scale), output_w
            )
            torch.cuda.synchronize()

            # ---- FlashInfer ----
            output_fi = flashinfer_gemm(raw_bf16, weight_data, None, weight_scale)
            torch.cuda.synchronize()

            # ---- Correctness ----
            p_valid = output_p
            w_valid = output_w[:m]
            fi_valid = output_fi

            cos_pw = cosine_sim(p_valid, w_valid)
            cos_pfi = cosine_sim(p_valid, fi_valid)
            cos_wfi = cosine_sim(w_valid, fi_valid)
            cos_pr = cosine_sim(p_valid, ref_out)
            cos_wr = cosine_sim(w_valid, ref_out)
            cos_fir = cosine_sim(fi_valid, ref_out)

            ok = all(c >= self.REF_COS_THR for c in [cos_pr, cos_wr, cos_fir]) and all(
                c >= self.CROSS_COS_THR for c in [cos_pw, cos_pfi, cos_wfi]
            )
            if not ok:
                all_pass = False

            # ---- Performance ----
            def run_plugin(_bf16=raw_bf16):
                plugin_fp8(_bf16, weight_data, weight_scale)

            def run_wrapper(_d=w_data_fp8, _s=w_scale_fp32, _o=output_w):
                wrapper_fp8((_d, _s), (weight_data, weight_scale), _o)

            def run_flashinfer(_bf16=raw_bf16):
                flashinfer_gemm(_bf16, weight_data, None, weight_scale)

            if NSYS_MODE:
                t_p, _, _ = nvtx_bench(f"plugin_3way_m{m}", run_plugin)
                t_w, _, _ = nvtx_bench(f"wrapper_3way_m{m}", run_wrapper)
                t_fi, _, _ = nvtx_bench(f"flashinfer_3way_m{m}", run_flashinfer)
            else:
                t_p, _, _ = bench(run_plugin, num_warmups=30, num_tests=100)
                t_w, _, _ = bench(run_wrapper, num_warmups=30, num_tests=100)
                t_fi, _, _ = bench(run_flashinfer, num_warmups=30, num_tests=100)

            spd_p_fi = t_fi / t_p if t_p > 0 else float("inf")
            spd_w_fi = t_fi / t_w if t_w > 0 else float("inf")

            print(
                f"{m:>4} | {t_p*1000:>11.4f} | {t_w*1000:>9.4f} | {t_fi*1000:>9.4f} "
                f"| {spd_p_fi:>4.2f}x | {spd_w_fi:>4.2f}x "
                f"| {cos_pw:>7.5f} | {cos_pfi:>7.5f} | {cos_wfi:>7.5f} "
                f"| {cos_pr:>7.5f} | {cos_wr:>7.5f} | {cos_fir:>7.5f}"
                f"  {'PASS' if ok else 'FAIL'}"
            )

        print(sep)
        self.assertTrue(all_pass, "Three-way GEMM correctness check failed")

    def test_flashinfer_bf16_input_shapes(self):
        """FlashInfer vs Plugin with realistic model shapes (qwen3.5-fp8-tp2).

        Tests bf16 input mode across various (K, N) pairs and M values.
        """
        flashinfer_gemm = self._get_flashinfer_op()
        plugin_fp8 = self._get_plugin_op()

        shapes = [
            (7168, 3584),
            (7168, 7168),
            (3584, 7168),
            (16384, 6144),
        ]
        m_values = [1, 2, 4, 8, 16, 24, 32, 48, 64]

        for k, n in shapes:
            weight_data = torch.randn(n, k, dtype=torch.bfloat16, device="cuda").to(
                torch.float8_e4m3fn
            )
            weight_scale = (
                torch.rand(n // 128, k // 128, dtype=torch.float32, device="cuda")
                * 0.01
                + 0.001
            )
            w_deq = _dequant_weight(weight_data, weight_scale)

            header = (
                f"{'m':>4} | {'Plugin(ms)':>11} | {'FI(ms)':>9} "
                f"| {'FI/P':>5} | {'P_FI':>7} | {'P_R':>7} | {'FI_R':>7}"
            )
            sep = "=" * len(header)
            print(f"\n{sep}")
            print(f"Plugin vs FlashInfer (bf16 input): k={k}, n={n}")
            print(header)
            print("-" * len(header))

            all_pass = True
            for m in m_values:
                raw_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                ref_out = (raw_bf16.float() @ w_deq.t()).to(torch.bfloat16)

                output_p = plugin_fp8(raw_bf16, weight_data, weight_scale)
                torch.cuda.synchronize()

                output_fi = flashinfer_gemm(raw_bf16, weight_data, None, weight_scale)
                torch.cuda.synchronize()

                cos_pfi = cosine_sim(output_p, output_fi)
                cos_pr = cosine_sim(output_p, ref_out)
                cos_fir = cosine_sim(output_fi, ref_out)

                ok = (
                    cos_pr >= self.REF_COS_THR
                    and cos_fir >= self.REF_COS_THR
                    and cos_pfi >= self.CROSS_COS_THR
                )
                if not ok:
                    all_pass = False

                def run_plugin(_bf16=raw_bf16):
                    plugin_fp8(_bf16, weight_data, weight_scale)

                def run_flashinfer(_bf16=raw_bf16):
                    flashinfer_gemm(_bf16, weight_data, None, weight_scale)

                if NSYS_MODE:
                    t_p, _, _ = nvtx_bench(f"plugin_fi_k{k}_n{n}_m{m}", run_plugin)
                    t_fi, _, _ = nvtx_bench(f"fi_k{k}_n{n}_m{m}", run_flashinfer)
                else:
                    t_p, _, _ = bench(run_plugin, num_warmups=30, num_tests=100)
                    t_fi, _, _ = bench(run_flashinfer, num_warmups=30, num_tests=100)

                spd = t_fi / t_p if t_p > 0 else float("inf")

                print(
                    f"{m:>4} | {t_p*1000:>11.4f} | {t_fi*1000:>9.4f} "
                    f"| {spd:>4.2f}x | {cos_pfi:>7.5f} | {cos_pr:>7.5f} | {cos_fir:>7.5f}"
                    f"  {'PASS' if ok else 'FAIL'}"
                )

            print(sep)
            self.assertTrue(
                all_pass,
                f"FlashInfer comparison failed for k={k}, n={n}",
            )


def _dequant_weight(
    weight_fp8: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize FP8 weight → float: w_deq[n, k] = w_fp8[n,k] * scale[n/128, k/128]."""
    n, k = weight_fp8.shape
    w_f = weight_fp8.float()
    w_s = (
        weight_scale.float()
        .repeat_interleave(128, dim=0)[:n]
        .repeat_interleave(128, dim=1)[:, :k]
    )
    return w_f * w_s


class CudaFp8DeepGEMMLinearIntegrationTest(unittest.TestCase):
    """Integration test: CudaFp8DeepGEMMLinear routes to plugin for small M."""

    COS_THR = 0.98

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        cap = torch.cuda.get_device_capability()
        if cap[0] < 9:
            raise unittest.SkipTest(f"Requires SM90+, got SM{cap[0]}{cap[1]}")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def _make_linear(self, K: int, N: int):
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            CudaFp8DeepGEMMLinear,
        )

        class _FakeQuantConfig:
            def get_method(self):
                return "FP8_PER_BLOCK"

        # Non-e8m0 storage: weight [K, N], scale [K/128, N/128]
        # __init__ will reshape to [N, K] and [N/128, K/128]
        weight = torch.randn(K, N, dtype=torch.bfloat16, device="cuda").to(
            torch.float8_e4m3fn
        )
        weight_scales = (
            torch.rand(K // 128, N // 128, dtype=torch.float32, device="cuda") * 0.01
            + 0.001
        )

        linear = CudaFp8DeepGEMMLinear(
            weight=weight,
            weight_scales=weight_scales,
            quant_config=_FakeQuantConfig(),
        )
        # After init, linear.weight is [N, K], linear.weight_scales is [N/128, K/128]
        return linear, linear.weight, linear.weight_scales

    def test_plugin_routing_and_correctness(self):
        """Verify small-M goes through swap_ab path and produces correct output.

        Tests a broad range of M values including primes (3,5,7,11,13,17,19,23,29,31)
        to catch alignment/padding bugs, plus smoke-realistic KN sizes for qwen3.5 tp=2.
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import enable_swapab
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            _SWAP_AB_M_THRESHOLD,
        )

        if not enable_swapab():
            self.skipTest("swap_ab not available")

        # (K, N) pairs: synthetic + qwen3.5-fp8-tp2 realistic sizes
        shapes = [
            (4096, 2048),  # synthetic baseline
            (7168, 3584),  # qwen3.5 tp=2: hidden=7168, ffn/2
            (7168, 7168),  # qwen3.5 attention projection
            (3584, 7168),  # qwen3.5 down projection
        ]

        # Comprehensive M range: powers-of-2, primes, edge cases around thresholds
        m_values = [1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 23, 24, 29, 31]

        all_pass = True
        for K, N in shapes:
            linear, weight, weight_scales = self._make_linear(K, N)
            w_deq = _dequant_weight(weight, weight_scales)

            print(f"\n{'='*72}")
            print(
                f"CudaFp8DeepGEMMLinear integration: K={K}, N={N}, threshold={_SWAP_AB_M_THRESHOLD}"
            )
            print(f"{'m':>4} | {'path':>8} | {'cos_vs_ref':>10} | {'status':>6}")
            print(f"{'-'*72}")

            for m in m_values:
                inp = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
                out = linear(inp)
                torch.cuda.synchronize()

                ref = (inp.float() @ w_deq.t()).to(torch.bfloat16)
                cos = cosine_sim(out, ref)
                path = "swap_ab" if m < _SWAP_AB_M_THRESHOLD else "wrapper"
                ok = cos >= self.COS_THR
                if not ok:
                    all_pass = False
                print(f"{m:>4} | {path:>8} | {cos:>10.5f} | {'PASS' if ok else 'FAIL'}")

            print(f"{'='*72}")

        self.assertTrue(all_pass, "Integration correctness check failed")

    def test_decode_simulation(self):
        """Simulate 100 continuous decode steps (M=1..31) to catch state corruption.

        Reproduces the smoke test scenario: qwen3.5-fp8-tp2 generates tokens
        one by one (M=1) and in small batches (M=2..31).
        """
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import enable_swapab
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
            _SWAP_AB_M_THRESHOLD,
        )

        if not enable_swapab():
            self.skipTest("swap_ab not available")

        # qwen3.5-fp8-tp2: hidden=7168, each GPU sees K=7168
        K, N = 7168, 3584
        linear, weight, weight_scales = self._make_linear(K, N)
        w_deq = _dequant_weight(weight, weight_scales)

        print(f"\n{'='*60}")
        print(f"Decode simulation: K={K}, N={N}, 100 steps each M=1..31")

        all_pass = True
        # Test 100 decode steps for each of these M values
        decode_m_values = [1, 2, 3, 5, 7, 8, 11, 13, 15, 16, 17, 19, 23, 24, 29, 31]
        for m in decode_m_values:
            fail_step = None
            for step in range(100):
                inp = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
                out = linear(inp)
                torch.cuda.synchronize()

                ref = (inp.float() @ w_deq.t()).to(torch.bfloat16)
                cos = cosine_sim(out, ref)
                if cos < self.COS_THR or out.isnan().any() or out.isinf().any():
                    fail_step = step
                    all_pass = False
                    break

            status = (
                f"FAIL at step {fail_step}"
                if fail_step is not None
                else "PASS (100 steps)"
            )
            path = "swap_ab" if m < _SWAP_AB_M_THRESHOLD else "wrapper"
            print(f"  m={m:2d} ({path}): {status}")

        print(f"{'='*60}")
        self.assertTrue(all_pass, "Decode simulation correctness check failed")

    def test_swap_ab_vs_wrapper_consistency(self):
        """Compare _forward_swap_ab (C++ plugin) vs _forward_wrapper (deep_gemm lib)."""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import enable_swapab

        K, N = 4096, 2048
        linear, weight, weight_scales = self._make_linear(K, N)

        if not enable_swapab():
            self.skipTest("swap_ab not available")

        w_deq = _dequant_weight(weight, weight_scales)

        small_m_values = [1, 4, 8, 16, 24, 32, 39]
        print(f"\n{'='*70}")
        print(f"swap_ab vs Wrapper consistency: K={K}, N={N}")
        print(
            f"{'m':>4} | {'cos_swap_wrap':>12} | {'cos_swap_ref':>12} | {'cos_wrap_ref':>12} | {'status':>6}"
        )
        print(f"{'-'*70}")

        all_pass = True
        for m in small_m_values:
            inp = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")

            out_swap = linear._forward_swap_ab(inp, m)
            torch.cuda.synchronize()

            out_wrap = linear._forward_wrapper(inp, m)
            torch.cuda.synchronize()

            ref = (inp.float() @ w_deq.t()).to(torch.bfloat16)

            cos_sw = cosine_sim(out_swap, out_wrap)
            cos_sr = cosine_sim(out_swap, ref)
            cos_wr = cosine_sim(out_wrap, ref)
            ok = cos_sw >= 0.999 and cos_sr >= self.COS_THR and cos_wr >= self.COS_THR
            if not ok:
                all_pass = False
            print(
                f"{m:>4} | {cos_sw:>12.5f} | {cos_sr:>12.5f} | {cos_wr:>12.5f} "
                f"| {'PASS' if ok else 'FAIL'}"
            )

        print(f"{'='*70}")
        self.assertTrue(all_pass, "swap_ab vs wrapper consistency check failed")

    def test_large_m_uses_wrapper(self):
        """Verify M >= 64 still uses wrapper path (no crash)."""
        K, N = 4096, 2048
        linear, _, _ = self._make_linear(K, N)

        for m in [64, 128, 256]:
            inp = torch.randn(m, K, dtype=torch.bfloat16, device="cuda")
            out = linear(inp)
            torch.cuda.synchronize()
            self.assertEqual(out.shape, (m, N))


if __name__ == "__main__":
    unittest.main()
