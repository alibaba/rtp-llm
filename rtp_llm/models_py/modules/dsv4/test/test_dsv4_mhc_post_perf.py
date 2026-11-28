"""Correctness, determinism, and SM103/SM100 benchmark for dsv4_mhc_post_fwd vs tk_mhc_post & mega_mhc."""

from __future__ import annotations

import statistics
import tempfile
import unittest

import torch

from rtp_llm.models_py.modules import RMSNorm
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.dsv4.hc import mega_mhc as _mega_mhc
from rtp_llm.models_py.modules.dsv4.hc._fused_mhc_pre_norm import (
    can_use_fused_mhc_pre_norm,
    fused_mhc_pre_norm,
)
from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import tk_mhc_post
from rtp_llm.models_py.modules.dsv4.test.dsv4_kernel_perf_utils import (
    measure_kernel,
)


def _call_tk_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    M, H, D = residual.shape
    tk_x = x.unsqueeze(1)
    tk_r = residual.unsqueeze(1)
    tk_p = post.reshape(M, 1, H, 1)
    tk_c = comb.unsqueeze(1)
    tk_out = None if out is None else out.unsqueeze(1)
    res = tk_mhc_post(tk_x, tk_r, tk_p, tk_c, hc_mult=H, out=tk_out)
    if res is None:
        raise RuntimeError(f"tk_mhc_post returned None for M={M}, H={H}, D={D}")
    return res.squeeze(1)


def _get_cuda_mhc_post_op():
    from rtp_llm.ops import rtp_llm_ops

    op = getattr(rtp_llm_ops, "dsv4_mhc_post_fwd", None)
    if op is None:
        raise AssertionError("CUDA op dsv4_mhc_post_fwd is missing from rtp_llm_ops")
    return op


def _ref_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    lead = x.shape[:-1]
    dim = x.shape[-1]
    hc = residual.shape[-2]
    xf = x.reshape(-1, dim).float()
    rf = residual.reshape(-1, hc, dim).float()
    pf = post.reshape(-1, hc).float()
    cf = comb.reshape(-1, hc, hc).float()
    out = xf.unsqueeze(1) * pf.unsqueeze(-1)
    for i in range(hc):
        out = out + rf[:, i : i + 1, :] * cf[:, i, :].unsqueeze(-1)
    return out.to(torch.bfloat16).reshape(*lead, hc, dim)


def _make_hc_unit(dim: int, hc_mult: int, device: torch.device, name: str):
    rows = 2 * hc_mult + hc_mult * hc_mult
    fn = torch.randn((rows, hc_mult * dim), dtype=torch.float32, device=device) * 0.02
    base = torch.randn((rows,), dtype=torch.float32, device=device) * 0.05
    scale = torch.rand((3,), dtype=torch.float32, device=device) * 0.5 + 0.5
    return build_hc_unit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=20,
        norm_eps=1.0e-6,
        hc_eps=1.0e-6,
        layer_id=3,
        name=name,
    )


def _measure_us(
    fn,
    *,
    warmup: int = 20,
    iters: int = 60,
    flush_buf: torch.Tensor | None = None,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(iters):
        if flush_buf is not None:
            # Read-only scan to evict L2 without dirtying writeback buffers.
            _ = flush_buf.sum()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times_ms.append(s.elapsed_time(e))
    med_us = statistics.median(times_ms) * 1000.0
    mean_us = statistics.mean(times_ms) * 1000.0
    return med_us, mean_us


class Dsv4MhcPostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is required for Dsv4MhcPostTest")
        cls.device = torch.device("cuda", torch.cuda.current_device())
        cls.op = _get_cuda_mhc_post_op()

    @torch.inference_mode()
    def test_parity_edge_cases_and_determinism(self) -> None:
        torch.manual_seed(2026)
        cases = [
            (0, 4, 4096, torch.float32, False),
            (1, 4, 4096, torch.float32, False),
            (17, 4, 4096, torch.float32, False),
            (512, 4, 4096, torch.float32, False),
            (2048, 4, 7168, torch.float32, False),
            (64, 4, 515, torch.float32, False),
            (33, 2, 123, torch.bfloat16, False),
            (19, 8, 256, torch.float32, False),
            (32, 1, 128, torch.bfloat16, False),
            (4, 12, 64, torch.float32, False),
            (16, 4, 128, torch.float32, True),  # unaligned storage offset
        ]
        for M, H, D, coeff_dtype, unaligned in cases:
            with self.subTest(
                M=M, H=H, D=D, coeff_dtype=coeff_dtype, unaligned=unaligned
            ):
                if unaligned:
                    x_backing = torch.randn(
                        M * D + 3, dtype=torch.bfloat16, device=self.device
                    )
                    r_backing = torch.randn(
                        M * H * D + 3, dtype=torch.bfloat16, device=self.device
                    )
                    x = x_backing[1 : 1 + M * D].view(M, D)
                    residual = r_backing[1 : 1 + M * H * D].view(M, H, D)
                    assert x.data_ptr() % 16 != 0 and residual.data_ptr() % 16 != 0
                else:
                    x = torch.randn(M, D, dtype=torch.bfloat16, device=self.device)
                    residual = torch.randn(
                        M, H, D, dtype=torch.bfloat16, device=self.device
                    )

                post = torch.randn(M, H, 1, dtype=coeff_dtype, device=self.device)
                comb = torch.randn(M, H, H, dtype=coeff_dtype, device=self.device)
                ref = _ref_mhc_post(x, residual, post, comb)

                out_fresh = torch.empty_like(residual)
                self.op(x, residual, post, comb, out_fresh)
                res_inplace = residual.clone()
                self.op(x, res_inplace, post, comb, res_inplace)

                if M > 0:
                    torch.testing.assert_close(out_fresh, ref, atol=2e-2, rtol=2e-2)
                    self.assertTrue(torch.equal(out_fresh, res_inplace))
                else:
                    self.assertEqual(out_fresh.shape, (0, H, D))

                if (
                    M >= 16
                    and H == 4
                    and (D % 8 == 0)
                    and not unaligned
                    and coeff_dtype == torch.float32
                ):
                    tk_ref = _call_tk_mhc_post(x, residual.clone(), post, comb)
                    torch.testing.assert_close(out_fresh, tk_ref, atol=2e-2, rtol=2e-2)

        # 3D [B, S, D] + chunked determinism check
        B, S, H, D = 2, 1024, 4, 4096
        x3 = torch.randn(B, S, D, dtype=torch.bfloat16, device=self.device)
        r3 = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=self.device)
        p3 = torch.randn(B, S, H, dtype=torch.float32, device=self.device)
        c3 = torch.randn(B, S, H, H, dtype=torch.float32, device=self.device)
        o_a = r3.clone()
        o_b = r3.clone()
        self.op(x3, o_a, p3, c3, o_a)
        self.op(x3, o_b, p3, c3, o_b)
        self.assertTrue(torch.equal(o_a, o_b))

        # Split M=2048 into [512, 1536] and confirm bit-identical results
        x_flat = x3.view(B * S, D)
        r_flat = r3.view(B * S, H, D)
        p_flat = p3.view(B * S, H)
        c_flat = c3.view(B * S, H, H)
        c_part1 = r_flat[:512].clone()
        c_part2 = r_flat[512:].clone()
        self.op(x_flat[:512], c_part1, p_flat[:512], c_flat[:512], c_part1)
        self.op(x_flat[512:], c_part2, p_flat[512:], c_flat[512:], c_part2)
        self.assertTrue(
            torch.equal(o_a.view(B * S, H, D), torch.cat([c_part1, c_part2], dim=0))
        )

    def test_overlap_rejected(self) -> None:
        M, H, D = 8, 4, 64
        x = torch.randn(M, D, dtype=torch.bfloat16, device=self.device)
        backing = torch.randn(M * H * D + 16, dtype=torch.bfloat16, device=self.device)
        residual = backing[: M * H * D].view(M, H, D)
        shifted_out = backing[16 : 16 + M * H * D].view(M, H, D)
        post = torch.randn(M, H, dtype=torch.float32, device=self.device)
        comb = torch.randn(M, H, H, dtype=torch.float32, device=self.device)
        with self.assertRaises(RuntimeError):
            self.op(x, residual, post, comb, shifted_out)

    @torch.inference_mode()
    def test_benchmark_mhc_post_and_mega_mhc(self) -> None:
        dev_name = torch.cuda.get_device_name(self.device)
        flush_buf = torch.empty(
            128 * 1024 * 1024 // 4, dtype=torch.float32, device=self.device
        )
        with tempfile.TemporaryDirectory() as trace_dir:
            print(
                f"\n=== [mhc_post standalone benchmark on {dev_name}] ===", flush=True
            )
            print(
                f"{'D':>5} {'M':>6} | {'TK_cold':>8} {'CU_cold':>8} {'Spd':>5} | {'TK_kern':>8} {'CU_kern':>8} {'CU_BW':>8} {'KSpd':>6}",
                flush=True,
            )
            H = 4
            for D in (4096, 7168):
                for M in (
                    1,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    558,
                    1024,
                    2048,
                    4096,
                    8192,
                    12288,
                ):
                    x = (
                        torch.randn(M, D, dtype=torch.bfloat16, device=self.device)
                        * 0.2
                    )
                    residual = torch.randn(
                        M, H, D, dtype=torch.bfloat16, device=self.device
                    )
                    post = (
                        torch.rand(M, H, 1, dtype=torch.float32, device=self.device)
                        * 0.25
                    )
                    comb = torch.softmax(
                        torch.randn(M, H, H, dtype=torch.float32, device=self.device),
                        dim=-1,
                    )

                    def run_tk():
                        _call_tk_mhc_post(x, residual, post, comb, out=residual)

                    def run_cuda():
                        self.op(x, residual, post, comb, residual)

                    out_tk = _call_tk_mhc_post(x, residual.clone(), post, comb)
                    out_cu = residual.clone()
                    self.op(x, out_cu, post, comb, out_cu)
                    torch.testing.assert_close(out_cu, out_tk, atol=2e-2, rtol=2e-2)

                    tk_med, _ = _measure_us(run_tk, flush_buf=flush_buf)
                    cu_med, _ = _measure_us(run_cuda, flush_buf=flush_buf)
                    tk_k = measure_kernel(
                        run_tk,
                        label=f"tk_D{D}_M{M}",
                        trace_dir=trace_dir,
                        iters=30,
                        warmup=10,
                    ).kernel_span_us
                    cu_k = measure_kernel(
                        run_cuda,
                        label=f"cu_D{D}_M{M}",
                        trace_dir=trace_dir,
                        iters=30,
                        warmup=10,
                    ).kernel_span_us
                    bytes_moved = (2 * H + 1) * M * D * 2
                    cu_bw = bytes_moved / (cu_k * 1e-6) / 1e9
                    print(
                        f"{D:5d} {M:6d} | {tk_med:8.2f} {cu_med:8.2f} {tk_med / cu_med:4.2f}x | "
                        f"{tk_k:8.2f} {cu_k:8.2f} {cu_bw:8.1f} {tk_k / cu_k:5.2f}x",
                        flush=True,
                    )

            if _mega_mhc._get_mega_mhc() is None:
                print(
                    "\n=== [mega_mhc unavailable on this device/runtime] ===",
                    flush=True,
                )
                return

            print(
                f"\n=== [intra-block post+pre+norm vs mega_mhc on {dev_name}] ===",
                flush=True,
            )
            print(
                f"{'D':>5} {'M':>6} | {'mega_k(us)':>10} {'CU+fuse_k':>10} {'TK+fuse_k':>10} | {'mega_ev':>8} {'CU+fuse_ev':>10} | {'mega/CU_k':>9}",
                flush=True,
            )
            for D in (4096,):
                hc_prev = _make_hc_unit(D, H, self.device, "attn_hc")
                hc_next = _make_hc_unit(D, H, self.device, "ffn_hc")
                norm = RMSNorm(
                    torch.ones(D, dtype=torch.bfloat16, device=self.device), eps=1e-6
                )

                for M in (16, 128, 512, 1024, 2048, 4096, 8192, 12288):
                    sub_out = (
                        torch.randn(M, D, dtype=torch.bfloat16, device=self.device)
                        * 0.2
                    )
                    res_init = torch.randn(
                        M, H, D, dtype=torch.bfloat16, device=self.device
                    )
                    post = (
                        torch.rand(M, H, 1, dtype=torch.float32, device=self.device)
                        * 0.2
                    )
                    comb = (
                        torch.rand(M, H, H, dtype=torch.float32, device=self.device)
                        * 0.2
                    )
                    res_sep = res_init.clone()

                    mega_res = _mega_mhc.try_fused_post_pre(
                        sub_out,
                        res_init.clone(),
                        post,
                        comb,
                        previous=hc_prev,
                        next_hc=hc_next,
                        norm=norm,
                    )
                    if mega_res is None:
                        print(f"{D:5d} {M:6d} | mega_mhc returned None", flush=True)
                        continue

                    def run_mega():
                        _mega_mhc.try_fused_post_pre(
                            sub_out,
                            res_init,
                            post,
                            comb,
                            previous=hc_prev,
                            next_hc=hc_next,
                            norm=norm,
                        )

                    def _hc_pre_norm_step(res_tensor):
                        if can_use_fused_mhc_pre_norm(res_tensor, hc_next, norm):
                            return fused_mhc_pre_norm(res_tensor, hc_next, norm)
                        x_p, p_n, c_n = hc_next.pre(res_tensor)
                        return norm(x_p), p_n, c_n

                    def run_cuda_sep():
                        self.op(sub_out, res_sep, post, comb, res_sep)
                        _hc_pre_norm_step(res_sep)

                    def run_tk_sep():
                        _call_tk_mhc_post(sub_out, res_sep, post, comb, out=res_sep)
                        _hc_pre_norm_step(res_sep)

                    r_chk = res_init.clone()
                    x_mega, x_pre_mega, _, _ = _mega_mhc.try_fused_post_pre(
                        sub_out,
                        r_chk,
                        post,
                        comb,
                        previous=hc_prev,
                        next_hc=hc_next,
                        norm=norm,
                    )
                    r_sep_chk = res_init.clone()
                    self.op(sub_out, r_sep_chk, post, comb, r_sep_chk)
                    x_pre_sep, _, _ = _hc_pre_norm_step(r_sep_chk)
                    torch.testing.assert_close(x_mega, r_sep_chk, atol=2e-2, rtol=2e-2)
                    torch.testing.assert_close(
                        x_pre_mega, x_pre_sep, atol=5e-2, rtol=5e-2
                    )

                    m_meas = measure_kernel(
                        run_mega,
                        label=f"mega_M{M}",
                        trace_dir=trace_dir,
                        iters=30,
                        warmup=10,
                    )
                    res_sep.copy_(res_init)
                    cs_meas = measure_kernel(
                        run_cuda_sep,
                        label=f"cufuse_M{M}",
                        trace_dir=trace_dir,
                        iters=30,
                        warmup=10,
                    )
                    res_sep.copy_(res_init)
                    ts_meas = measure_kernel(
                        run_tk_sep,
                        label=f"tkfuse_M{M}",
                        trace_dir=trace_dir,
                        iters=30,
                        warmup=10,
                    )
                    print(
                        f"{D:5d} {M:6d} | {m_meas.kernel_span_us:10.2f} {cs_meas.kernel_span_us:10.2f} {ts_meas.kernel_span_us:10.2f} | "
                        f"{m_meas.event_us:8.2f} {cs_meas.event_us:10.2f} | {m_meas.kernel_span_us / cs_meas.kernel_span_us:8.2f}x",
                        flush=True,
                    )


if __name__ == "__main__":
    unittest.main()
