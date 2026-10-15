"""Equivalence UT for the fused Xing4 hc-row RMSNorm + hc_fn GEMV kernel
(fusion round-2, K1+GEMM).

Reference (numerical authority, hand-written step by step from
``Xing4HCUnit._pre_impl`` / ``_pre_fused`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py``)::

    x_flat      = x.flatten(-2)
    rsqrt       = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + eps)
    flat_normed = (x_flat.float() * rsqrt).to(x.dtype)   # bf16, feeds F.linear
    mixes       = F.linear(flat_normed, fn)              # [T, (hc+2)*hc] bf16

The kernel reproduces the reference rounding structure exactly per element
(bf16 RTNE norm rounding via the opaque bitcast, exact bf16 products in
fp32, one bf16 round after the full dot); the ONLY differences are the two
fp32 summation orders (rstd block-cyclic, dot ascending-c) versus torch's
``mean`` and cuBLAS's split-K — the 1-ulp tolerance class every fused K lane
gates with rtol = atol = 2**-7. Gates covered:

* T in {1, 2, 8, 512, 4096} at hc=4, H=3584 (cols=14336) bf16, plus the
  production fn domain (0.02-scaled, the dsv4 weights recipe) and a raw
  randn fn (larger accumulated magnitudes);
* small-H generality (cols=64), 4D batched input, row-sliced input;
* the rows gate resolver ``XING4_HC_FN_GEMV_MAX_TOKENS`` (default 32, 0
  disables, unparsable falls back to the default) — CPU-runnable;
* CPU-only input validation (dtype/device/fn contiguity/width) before any
  launch — CPU-runnable;
* launch-count report: the reference chain (torch rmsnorm ops + cuBLAS)
  versus the single fused Triton launch;
* a printed device-only (CUDA-graph) timing report of K1 + F.linear vs the
  fused GEMV — the evidence for the rows gate; printed, not gated, because
  shared-GPU timing variance must not flake the equivalence UT.
"""

import os
import unittest
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.xing4.hc import fn_gemv_triton
from rtp_llm.models_py.modules.xing4.hc.fn_gemv_triton import (
    hc_fn_gemv_max_tokens,
    hc_fn_gemv_triton,
)
from rtp_llm.models_py.modules.xing4.hc.norm_triton import (
    hc_row_rms_norm_triton,
)

HC = 4
HIDDEN = 3584
COLS = HC * HIDDEN  # 14336 = 7 * 2048
N_OUT = (HC + 2) * HC  # 24
RTOL = 2.0**-7
ATOL = 2.0**-7
_TS = (1, 2, 8, 512, 4096)
GATE_ENV = "XING4_HC_FN_GEMV_MAX_TOKENS"


@contextmanager
def _env(name, value):
    old = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _make_x(*shape, seed=0, device="cuda"):
    torch.manual_seed(seed)
    return torch.randn(*shape, device=device, dtype=torch.float32).to(
        torch.bfloat16
    )


def _make_fn(cols, seed=0, scale=0.02, device="cuda"):
    torch.manual_seed(seed)
    return (torch.randn(N_OUT, cols, device=device, dtype=torch.float32) * scale).to(
        torch.bfloat16
    )


def _reference_mixes(x, fn, eps):
    """Hand-written step-by-step reference (xing4_0_model.py chain)."""
    x_flat = x.flatten(-2)
    rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + eps)
    flat_normed = (x_flat.float() * rsqrt).to(x.dtype)
    return torch.nn.functional.linear(flat_normed, fn)


def _report_diff(out, ref, label):
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item() if ref.numel() else 0.0
    n_diff = int((out != ref).sum().item())
    total = max(ref.numel(), 1)
    print(
        f"[fn-gemv] {label}: max|diff|={max_diff:.3e} "
        f"bitwise_diff={n_diff}/{total} ({100.0 * n_diff / total:.4f}%)",
        flush=True,
    )
    return max_diff


def _assert_equiv(out, ref, label, atol=ATOL):
    torch.testing.assert_close(out.float(), ref.float(), rtol=RTOL, atol=atol)
    _report_diff(out, ref, label)


def _eager_compat_atol(x, fn, eps):
    """One bf16 ulp of the accumulated dot magnitude (randn-fn stress gate).

    Same scheme as test_hc_post_triton.py: ``|mixes| <= sum_c |v_c * f_c|``
    (computed exactly as an abs-dot over the reference flat_normed), and one
    bf16 ulp of a value v is at most 2**-7 * v, so atol = 2**-7 * bound
    covers a summation-order tie flip even where cancellation makes the
    FINAL output tiny (the plain atol cannot cover those columns). The
    production fn domain (0.02-scaled) stays on the plain 2**-7 gate — the
    strongest signal in the regime production actually runs.
    """
    x_flat = x.flatten(-2).float()
    rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + eps)
    v = (x_flat * rsqrt).to(torch.bfloat16).float()
    return ATOL * (1.0 + (v.abs() @ fn.float().abs().t()).amax().item())


def _graph_time_ms(fn, n=200):
    """Device-only timing: capture n calls in one CUDA graph and replay.

    The production decode path runs the mHC chain inside a CUDA graph, so
    kernel wall cost is what matters; an event loop around Python dispatch
    measures host overhead instead (the Triton dispatch alone is ~10us and
    hides sub-10us kernel differences).
    """
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(n):
            fn()
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(10):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 10 / n


class TestHcFnGemvGate(unittest.TestCase):
    """CPU-runnable rows-gate resolver contract."""

    def test_env_matrix(self):
        cases = [
            (None, 32),  # unset -> default
            ("", 32),
            ("  ", 32),
            ("0", 0),  # explicit disable
            ("16", 16),
            ("64", 64),
            ("4096", 4096),
            ("bogus", 32),  # unparsable -> default, no raise
            ("-5", 0),  # clamped to >= 0
        ]
        for raw, want in cases:
            with self.subTest(raw=raw), _env(GATE_ENV, raw):
                self.assertEqual(hc_fn_gemv_max_tokens(), want)
        print(
            f"[fn-gemv] gate env matrix OK: default 32, 0 disables, "
            f"unparsable -> default",
            flush=True,
        )


class TestHcFnGemvContract(unittest.TestCase):
    """CPU-only input validation: the checks run before the launch."""

    def _cpu_case(self):
        x = torch.zeros(2, HC, 8, dtype=torch.bfloat16)
        fn = torch.zeros(N_OUT, HC * 8, dtype=torch.bfloat16)
        return x, fn

    def test_rejects_cpu_tensors(self):
        x, fn = self._cpu_case()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            hc_fn_gemv_triton(x, fn, 1e-6)

    def test_rejects_non_bf16(self):
        x, fn = self._cpu_case()
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            hc_fn_gemv_triton(x.float(), fn, 1e-6)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            hc_fn_gemv_triton(x, fn.float(), 1e-6)

    def test_rejects_noncontig_fn(self):
        x, fn = self._cpu_case()
        fn_t = fn.t().contiguous().t()  # [N, cols] with col-major strides
        self.assertFalse(fn_t.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous 2D"):
            hc_fn_gemv_triton(x, fn_t, 1e-6)

    def test_rejects_width_mismatch(self):
        x, fn = self._cpu_case()  # cols = 4*8 = 32
        fn_narrow = torch.zeros(N_OUT, 7 * 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fn columns"):
            hc_fn_gemv_triton(x, fn_narrow, 1e-6)

    def test_rejects_low_rank(self):
        fn = torch.zeros(N_OUT, 32, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "dim >= 2"):
            hc_fn_gemv_triton(torch.zeros(32, dtype=torch.bfloat16), fn, 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHcFnGemvTritonCuda(unittest.TestCase):
    def test_t_sweep_matches_reference(self):
        # Production fn domain (dsv4 weights recipe, 0.02-scaled) plus the
        # raw randn fn (larger accumulated magnitudes stress the fp32
        # accumulation-order equivalence; gated with one bf16 ulp of the
        # accumulated dot magnitude — near-cancellation columns legitimately
        # exceed a plain atol, the same root cause and treatment as
        # test_hc_post_triton.py).
        for scale, tag in ((0.02, "prod-domain"), (1.0, "randn")):
            for t in _TS:
                with self.subTest(scale=scale, T=t):
                    x = _make_x(t, HC, HIDDEN, seed=t)
                    fn = _make_fn(COLS, seed=100 + t, scale=scale)
                    ref = _reference_mixes(x, fn, 1e-6)
                    out = hc_fn_gemv_triton(x, fn, 1e-6)
                    self.assertIs(out.dtype, torch.bfloat16)
                    self.assertEqual(tuple(out.shape), (t, N_OUT))
                    atol = ATOL if scale == 0.02 else _eager_compat_atol(x, fn, 1e-6)
                    _assert_equiv(out, ref, f"gemv T={t} fn={tag}", atol=atol)

    def test_eps_boundaries(self):
        for eps in (0.0, 1e-6, 1e-4):
            with self.subTest(eps=eps):
                x = _make_x(8, HC, HIDDEN, seed=11)
                fn = _make_fn(COLS, seed=12)
                ref = _reference_mixes(x, fn, eps)
                out = hc_fn_gemv_triton(x, fn, eps)
                _assert_equiv(out, ref, f"gemv T=8 eps={eps}")

    def test_small_hidden_generality(self):
        # cols=64 exercises the small-BLOCK_N specialization path.
        x = _make_x(8, HC, 16, seed=3)
        fn = _make_fn(HC * 16, seed=4)
        ref = _reference_mixes(x, fn, 1e-6)
        out = hc_fn_gemv_triton(x, fn, 1e-6)
        self.assertEqual(tuple(out.shape), (8, N_OUT))
        _assert_equiv(out, ref, "gemv H=16 cols=64")

    def test_batched_4d_input(self):
        x = _make_x(2, 5, HC, HIDDEN, seed=4)
        fn = _make_fn(COLS, seed=5)
        ref = _reference_mixes(x, fn, 1e-6)
        out = hc_fn_gemv_triton(x, fn, 1e-6)
        self.assertEqual(tuple(out.shape), (2, 5, N_OUT))
        _assert_equiv(out, ref, "gemv 4D [2,5,hc,H]")

    def test_row_sliced_input(self):
        # Chunked callers pass row slices: row stride == cols, no copy.
        big = _make_x(64, HC, HIDDEN, seed=12)
        x = big[16:32]
        self.assertEqual(x.stride(0), COLS)
        fn = _make_fn(COLS, seed=13)
        ref = _reference_mixes(x, fn, 1e-6)
        out = hc_fn_gemv_triton(x, fn, 1e-6)
        _assert_equiv(out, ref, "gemv row-slice T=16")

    def test_input_not_modified(self):
        x = _make_x(8, HC, HIDDEN, seed=21)
        x_snapshot = x.clone()
        fn = _make_fn(COLS, seed=22)
        hc_fn_gemv_triton(x, fn, 1e-6)
        self.assertTrue(torch.equal(x, x_snapshot))

    def test_matches_k1_plus_linear_path(self):
        # The gate's else-branch composition: kernel output must equal the
        # K1(norm_triton) + F.linear chain the large-T path uses (within the
        # same tolerance class; both are order-only variants of the eager
        # reference).
        x = _make_x(512, HC, HIDDEN, seed=31)
        fn = _make_fn(COLS, seed=32)
        flat = hc_row_rms_norm_triton(x, 1e-6)
        chained = torch.nn.functional.linear(flat, fn)
        out = hc_fn_gemv_triton(x, fn, 1e-6)
        _assert_equiv(out, chained, "gemv vs K1+F.linear T=512")

    def test_launch_count_report(self):
        x = _make_x(8, HC, HIDDEN, seed=9)
        fn = _make_fn(COLS, seed=10)
        eps = 1e-6
        hc_fn_gemv_triton(x, fn, eps)  # JIT-compile outside the measurement

        kernel = fn_gemv_triton._hc_fn_gemv_kernel
        orig_run = kernel.run
        counter = {"launches": 0}

        def counting_run(*args, **kwargs):
            counter["launches"] += 1
            return orig_run(*args, **kwargs)

        kernel.run = counting_run
        try:
            out = hc_fn_gemv_triton(x, fn, eps)
        finally:
            del kernel.run  # restore the class-level JITFunction.run
        self.assertEqual(counter["launches"], 1)
        ref = _reference_mixes(x, fn, eps)
        torch.testing.assert_close(out.float(), ref.float(), rtol=RTOL, atol=ATOL)
        print(
            f"[fn-gemv] fused path: {counter['launches']} Triton kernel launch "
            "per call (reference chain: ~6-8 aten micro kernels + the cuBLAS "
            "split-K pair)",
            flush=True,
        )

    def test_timing_report(self):
        # Evidence for the rows gate (printed, not gated: shared-GPU
        # variance). Device-only CUDA-graph timing — the production decode
        # path replays the mHC chain inside a graph, so kernel wall cost is
        # the quantity that reaches TPOT; an event loop around Python
        # dispatch would measure the ~10-20us Triton/torch host overhead
        # instead and hide the differences.
        for t in (1, 8, 64):
            x = _make_x(t, HC, HIDDEN, seed=40 + t)
            fn = _make_fn(COLS, seed=50 + t)
            eps = 1e-6
            flat = hc_row_rms_norm_triton(x, eps)
            t_k1 = _graph_time_ms(lambda: hc_row_rms_norm_triton(x, eps))
            t_lin = _graph_time_ms(lambda: torch.nn.functional.linear(flat, fn))
            t_gemv = _graph_time_ms(lambda: hc_fn_gemv_triton(x, fn, eps))
            print(
                f"[fn-gemv] T={t} (graph device time): K1 {t_k1 * 1000:.2f}us "
                f"+ F.linear {t_lin * 1000:.2f}us = "
                f"{(t_k1 + t_lin) * 1000:.2f}us vs fused gemv "
                f"{t_gemv * 1000:.2f}us "
                f"(save {(t_k1 + t_lin - t_gemv) * 1000:+.2f}us per call)",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
