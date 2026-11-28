"""Unit tests and microbenchmark for Triton low-overhead ``fast_launch``."""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fast_launch import (
    clear_fast_launch_cache,
    fast_launch,
    fast_launch_capture_scope,
    is_cuda_stream_capturing,
)


@triton.jit(do_not_specialize=["n_elems", "stride0"])
def _scale_add_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elems,
    stride0,
    scale,
    USE_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems
    val = tl.load(x_ptr + offs * stride0, mask=mask, other=0.0).to(tl.float32)
    if USE_BIAS:
        if bias_ptr is not None:
            b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            val = val + b
    out = (val * scale).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _specialized_ptr_int_kernel(
    in_ptr,
    out_ptr,
    val_mult,
    N: tl.constexpr,
):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, x * val_mult)


@triton.jit
def _sum_tuple_kernel(
    in_ptr,
    out_ptr,
    pair,
    N: tl.constexpr,
):
    offs = tl.arange(0, N)
    x = tl.load(in_ptr + offs)
    tl.store(out_ptr + offs, x + pair[0])


@triton.jit(do_not_specialize=["n_elems", "s0", "s1", "s2", "s3", "s4", "s5", "s6"])
def _wide_args_kernel(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    p8,
    p9,
    n_elems,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    C4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    acc = tl.load(p0 + offs, mask=mask, other=0.0) + tl.load(
        p1 + offs, mask=mask, other=0.0
    )
    acc = (
        acc
        + tl.load(p2 + offs, mask=mask, other=0.0)
        + tl.load(p3 + offs, mask=mask, other=0.0)
    )
    acc = (
        acc
        + tl.load(p4 + offs, mask=mask, other=0.0)
        + tl.load(p5 + offs, mask=mask, other=0.0)
    )
    acc = (
        acc
        + tl.load(p6 + offs, mask=mask, other=0.0)
        + tl.load(p7 + offs, mask=mask, other=0.0)
    )
    acc = acc + tl.load(p8 + offs, mask=mask, other=0.0)
    acc = acc + (s0 + s1 + s2 + s3 + s4 + s5 + s6 + C0 + C1 + C2 + C3 + C4)
    tl.store(p9 + offs, acc, mask=mask)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FastLaunchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda:0")
        clear_fast_launch_cache(_scale_add_kernel)
        clear_fast_launch_cache(_specialized_ptr_int_kernel)
        clear_fast_launch_cache(_sum_tuple_kernel)
        clear_fast_launch_cache(_wide_args_kernel)

    def test_parity_with_various_types_constexpr_none_and_alignment(self) -> None:
        for dtype in (torch.bfloat16, torch.float32):
            for n in (1, 16, 17, 128, 255):
                x = torch.randn(n, dtype=dtype, device=self.device)
                bias = torch.randn(n, dtype=dtype, device=self.device)
                out_fast = torch.empty_like(x)
                out_ref = torch.empty_like(x)

                grid = (triton.cdiv(n, 64),)
                fast_launch(
                    _scale_add_kernel,
                    grid,
                    x,
                    bias,
                    out_fast,
                    n,
                    1,
                    1.5,
                    USE_BIAS=True,
                    BLOCK_SIZE=64,
                    num_warps=4,
                )
                _scale_add_kernel[grid](
                    x,
                    bias,
                    out_ref,
                    n,
                    1,
                    1.5,
                    USE_BIAS=True,
                    BLOCK_SIZE=64,
                    num_warps=4,
                )
                torch.testing.assert_close(out_fast, out_ref)

                # None optional pointer with keyword variation & callable grid
                fast_launch(
                    _scale_add_kernel,
                    lambda meta: (triton.cdiv(meta["n_elems"], meta["BLOCK_SIZE"]),),
                    x,
                    None,
                    out_fast,
                    n_elems=n,
                    stride0=1,
                    scale=0.5,
                    USE_BIAS=True,
                    BLOCK_SIZE=64,
                )
                _scale_add_kernel[grid](
                    x,
                    None,
                    out_ref,
                    n_elems=n,
                    stride0=1,
                    scale=0.5,
                    USE_BIAS=True,
                    BLOCK_SIZE=64,
                )
                torch.testing.assert_close(out_fast, out_ref)

        # Unaligned slice pointer and int==1 vs 16 vs 17 specialization
        base = torch.arange(64, dtype=torch.float32, device=self.device)
        aligned_in = base[:16]
        unaligned_in = base[1:17]  # 4-byte aligned, not 16-byte aligned
        out_fast = torch.empty(16, dtype=torch.float32, device=self.device)
        out_ref = torch.empty(16, dtype=torch.float32, device=self.device)

        for ptr_in in (aligned_in, unaligned_in):
            for mult in (1, 16, 17, 1 << 33):
                fast_launch(
                    _specialized_ptr_int_kernel,
                    (1,),
                    ptr_in,
                    out_fast,
                    mult,
                    N=16,
                )
                _specialized_ptr_int_kernel[(1,)](
                    ptr_in,
                    out_ref,
                    mult,
                    N=16,
                )
                torch.testing.assert_close(out_fast, out_ref)

    def test_no_recompile_and_cache_clear_invalidation(self) -> None:
        _specialized_ptr_int_kernel.device_caches.clear()
        clear_fast_launch_cache(_specialized_ptr_int_kernel)

        x = torch.randn(32, dtype=torch.float32, device=self.device)
        out = torch.empty_like(x)
        fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 3, N=32)
        dev = torch.cuda.current_device()
        cache_before = len(_specialized_ptr_int_kernel.device_caches[dev][0])
        self.assertEqual(cache_before, 1)

        # Repeat calls with different aligned buffer / unaligned==False, different %16!=0 int
        x2 = torch.randn(32, dtype=torch.float32, device=self.device)
        out2 = torch.empty_like(x2)
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 5, N=32)
        self.assertEqual(len(_specialized_ptr_int_kernel.device_caches[dev][0]), 1)

        # Clearing Triton device_caches invalidates fast_launch safely
        _specialized_ptr_int_kernel.device_caches.clear()
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 7, N=32)
        self.assertEqual(len(_specialized_ptr_int_kernel.device_caches[dev][0]), 1)
        torch.testing.assert_close(out2, x2 * 7)

        # Clearing only the per-device triton cache map also invalidates MRU safely
        _specialized_ptr_int_kernel.device_caches[dev][0].clear()
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 9, N=32)
        self.assertEqual(len(_specialized_ptr_int_kernel.device_caches[dev][0]), 1)
        torch.testing.assert_close(out2, x2 * 9)

        # Alternating MRU miss -> hit across 2 int specializations stays cached
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 16, N=32)
        self.assertEqual(len(_specialized_ptr_int_kernel.device_caches[dev][0]), 2)
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 3, N=32)
        torch.testing.assert_close(out2, x2 * 3)
        fast_launch(_specialized_ptr_int_kernel, (1,), x2, out2, 16, N=32)
        torch.testing.assert_close(out2, x2 * 16)
        self.assertEqual(len(_specialized_ptr_int_kernel.device_caches[dev][0]), 2)

    def test_cuda_graph_capture_and_special_fallbacks(self) -> None:
        x = torch.arange(16, dtype=torch.float32, device=self.device)
        out = torch.empty_like(x)
        # Tuple arg fallback
        fast_launch(_sum_tuple_kernel, (1,), x, out, (3, 0), N=16)
        torch.testing.assert_close(out, x + 3)

        # Warm kernel before capture
        fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 2, N=16)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 4, N=16)
        g.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, x * 4)

        # Capture scope hint bypasses capturing query and supports sequential + nested reuse
        scope = fast_launch_capture_scope(False)
        with scope:
            self.assertFalse(is_cuda_stream_capturing())
            fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 5, N=16)
        self.assertFalse(is_cuda_stream_capturing())
        torch.testing.assert_close(out, x * 5)
        with scope:
            with fast_launch_capture_scope(True):
                self.assertTrue(is_cuda_stream_capturing())
                fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 6, N=16)
            self.assertFalse(is_cuda_stream_capturing())
        torch.testing.assert_close(out, x * 6)

        # Monkeypatched torch.cuda.is_current_stream_capturing forces safe fallback on warmed MRU
        fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 7, N=16)
        self.assertIsNotNone(getattr(_specialized_ptr_int_kernel, "_rtp_fast_fn", None))
        with patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=True
        ) as mock_cap:
            self.assertTrue(is_cuda_stream_capturing())
            fast_launch(_specialized_ptr_int_kernel, (1,), x, out, 8, N=16)
            self.assertGreaterEqual(mock_cap.call_count, 1)
        torch.testing.assert_close(out, x * 8)

        # Mocked non-JITFunction fallback
        mock_fn = MagicMock()
        fast_launch(mock_fn, (1,), x, out)
        mock_fn.__getitem__.assert_called_once_with((1,))

    def test_cpu_tensor_raises_error_cleanly(self) -> None:
        x_cpu = torch.zeros(16, dtype=torch.float32)
        out_cpu = torch.empty_like(x_cpu)
        with self.assertRaises(Exception):
            fast_launch(_specialized_ptr_int_kernel, (1,), x_cpu, out_cpu, 2, N=16)

    def test_host_dispatch_speedup(self) -> None:
        n = 512
        x = torch.randn(n, dtype=torch.bfloat16, device=self.device)
        bias = torch.randn(n, dtype=torch.bfloat16, device=self.device)
        out = torch.empty_like(x)
        grid = (triton.cdiv(n, 128),)

        for _ in range(20):
            _scale_add_kernel[grid](
                x, bias, out, n, 1, 1.5, USE_BIAS=True, BLOCK_SIZE=128, num_warps=4
            )
            fast_launch(
                _scale_add_kernel,
                grid,
                x,
                bias,
                out,
                n,
                1,
                1.5,
                USE_BIAS=True,
                BLOCK_SIZE=128,
                num_warps=4,
            )
        torch.cuda.synchronize()
        self.assertIsNotNone(
            getattr(_scale_add_kernel, "_rtp_fast_fn", None),
            "fast_launch MRU closure was not built for _scale_add_kernel",
        )

        iters = 2000
        t0 = time.perf_counter()
        for _ in range(iters):
            _scale_add_kernel[grid](
                x, bias, out, n, 1, 1.5, USE_BIAS=True, BLOCK_SIZE=128, num_warps=4
            )
        t1 = time.perf_counter()
        torch.cuda.synchronize()

        with fast_launch_capture_scope(False):
            t2 = time.perf_counter()
            for _ in range(iters):
                fast_launch(
                    _scale_add_kernel,
                    grid,
                    x,
                    bias,
                    out,
                    n,
                    1,
                    1.5,
                    USE_BIAS=True,
                    BLOCK_SIZE=128,
                    num_warps=4,
                )
            t3 = time.perf_counter()
        torch.cuda.synchronize()

        raw_us = (t1 - t0) * 1e6 / iters
        fast_us = (t3 - t2) * 1e6 / iters

        t4 = time.perf_counter()
        for _ in range(iters):
            fast_launch(
                _scale_add_kernel,
                grid,
                x,
                bias,
                out,
                n,
                1,
                1.5,
                USE_BIAS=True,
                BLOCK_SIZE=128,
                num_warps=4,
            )
        t5 = time.perf_counter()
        torch.cuda.synchronize()
        fast_unscoped_us = (t5 - t4) * 1e6 / iters

        print(
            f"\n[fast_launch_test] raw_us={raw_us:.2f}us "
            f"fast_scoped_us={fast_us:.2f}us fast_unscoped_us={fast_unscoped_us:.2f}us "
            f"speedup_scoped={raw_us / max(fast_us, 1e-6):.2f}x"
        )
        self.assertLess(fast_us, raw_us)

        # Wide 24-arg kernel benchmark (typical of compressor/SWA/indexer kernels).
        bufs = [
            torch.randn(n, dtype=torch.float32, device=self.device) for _ in range(10)
        ]
        wide_grid = (triton.cdiv(n, 128),)
        for _ in range(20):
            _wide_args_kernel[wide_grid](
                *bufs,
                n,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                C0=1,
                C1=2,
                C2=3,
                C3=4,
                C4=5,
                BLOCK=128,
                num_warps=4,
            )
            fast_launch(
                _wide_args_kernel,
                wide_grid,
                *bufs,
                n,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                C0=1,
                C1=2,
                C2=3,
                C3=4,
                C4=5,
                BLOCK=128,
                num_warps=4,
            )
        torch.cuda.synchronize()
        wt0 = time.perf_counter()
        for _ in range(iters):
            _wide_args_kernel[wide_grid](
                *bufs,
                n,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                C0=1,
                C1=2,
                C2=3,
                C3=4,
                C4=5,
                BLOCK=128,
                num_warps=4,
            )
        wt1 = time.perf_counter()
        torch.cuda.synchronize()
        with fast_launch_capture_scope(False):
            wt2 = time.perf_counter()
            for _ in range(iters):
                fast_launch(
                    _wide_args_kernel,
                    wide_grid,
                    *bufs,
                    n,
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    C0=1,
                    C1=2,
                    C2=3,
                    C3=4,
                    C4=5,
                    BLOCK=128,
                    num_warps=4,
                )
            wt3 = time.perf_counter()
        torch.cuda.synchronize()
        w_raw_us = (wt1 - wt0) * 1e6 / iters
        w_fast_us = (wt3 - wt2) * 1e6 / iters
        print(
            f"[fast_launch_test] wide24 raw_us={w_raw_us:.2f}us "
            f"fast_scoped_us={w_fast_us:.2f}us speedup={w_raw_us / max(w_fast_us, 1e-6):.2f}x"
        )
        self.assertLess(w_fast_us, w_raw_us)


if __name__ == "__main__":
    unittest.main()
