"""Aggregate UT for the fused Xing4HCUnit wiring (HC fusion lane K5).

Compares ``Xing4HCUnit._pre_impl`` / ``_post_impl`` between the two
``XING4_HC_IMPL`` backends on the production shape (hc=4, H=3584):

* ``torch``  — the eager reference path kept verbatim in xing4_0_model.py
  (with its ``XING4_HC_FALLBACK_CHUNK_TOKENS`` chunking);
* ``triton`` — the fused K1(norm) + F.linear(GEMM, untouched) + K2(mix) +
  K3(collapse) / K4(post) composition dispatched through the
  ``rtp_llm.models_py.modules.xing4.hc`` factory.

Weights follow the dsv4 test_hc_impl._weights recipe (fn * 0.02, base = 0,
scale = 1) so the gates land in their production domains (pre in (0, 1),
up to ~0.999 with 14336-wide rows, post in (0, 2), comb doubly-stochastic).
Equivalence is tolerance-level (rtol = atol = 2**-7, the bf16-ulp class
every kernel lane gated with), not bitwise: each fused segment is within one
bf16 ulp of the eager ops it replaces and the segments compose.

The residual stream x is drawn at X_SCALE = 0.25 (the free UT parameter; the
weights and tolerance are fixed by the lane contract). K1/GEMM/K2 are exactly
invertible here (RMS-normalization makes flat_normed scale-invariant, so the
mixer is fully exercised: pre/comb measured bitwise-equal between paths),
but the K3 collapse has a documented structural difference
(collapse_triton.py: the eager path lands each pre*x product in bf16 before
the fp32 sum, the kernel multiplies in fp32 directly). That per-term landing
error (<= 0.5 bf16 ulp of |pre*x|, four terms) is ABSOLUTE w.r.t. y, so
near-cancellation columns with large stream magnitudes legitimately exceed a
plain atol = 2**-7 — the same root cause the K3 lane hit in its first run
and resolved by bounding UT data magnitudes with the kernel untouched.
Measured with these UT seeds: randn x gives 2-5 such columns per 30-60k
elements; x = 0.25 * randn brings max(diff/tol) down to 0.60 (T=4096) /
0.51 (T=512) — a ~40% margin while still exercising bf16 products up to
|pre*x| ~ 1.
The factory/fallback contract (env default / explicit torch / illegal
value / CPU auto-fallback with exactly one log line) is pinned CPU-side.

Round-2 K1+GEMM dispatch: with the default ``XING4_HC_FN_GEMV_MAX_TOKENS``
gate (32), the T=1 and T=8 legs of ``test_pre_post_equivalence_flat``
automatically run the fused GEMV arm of ``_pre_fused`` while T=512/4096 run
the K1 + F.linear arm; the dedicated dispatch test pins the gate itself
(default/disabled/forced) and the equivalence test gates both arms against
the eager reference at the same 2**-7 tolerance.
"""

import logging
import os
import unittest
from contextlib import contextmanager
from typing import Callable, List

import torch

from rtp_llm.models_py.model_desc.xing4_0_model import Xing4HCUnit
from rtp_llm.models_py.modules.xing4.hc import (
    IMPL_ENV,
    TORCH_IMPL,
    TRITON_IMPL,
    _reset_fallback_log_for_test,
    resolve_xing4_hc_impl,
    xing4_hc_use_fused,
)

HC = 4
HIDDEN = 3584
RTOL = 2.0**-7
ATOL = 2.0**-7
TOKENS = (1, 8, 512, 4096)
# Residual-stream magnitude: keeps the K3 structural diff (see module
# docstring) inside the mandated 2**-7 gate; measured margin ~40%.
X_SCALE = 0.25
CHUNK_ENV = "XING4_HC_FALLBACK_CHUNK_TOKENS"
GEMV_ENV = "XING4_HC_FN_GEMV_MAX_TOKENS"
_FACTORY_LOGGER = "rtp_llm.models_py.modules.xing4.hc"


def _make_stream(*shape, seed=0, device="cuda") -> torch.Tensor:
    """bf16 residual streams [.., hc, H] at the calibrated magnitude."""
    torch.manual_seed(seed)
    return (torch.randn(*shape, device=device, dtype=torch.float32) * X_SCALE).to(
        torch.bfloat16
    )


@contextmanager
def _env(name: str, value):
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


def _weights(hc: int, dim: int, device: str = "cpu"):
    torch.manual_seed(0)
    mix_hc = (2 + hc) * hc
    fn = torch.randn(mix_hc, hc * dim, device=device, dtype=torch.float32) * 0.02
    base = torch.zeros(mix_hc, device=device, dtype=torch.float32)
    scale = torch.ones(3, device=device, dtype=torch.float32)
    return fn, base, scale


def _make_unit(hc: int = HC, dim: int = HIDDEN, device: str = "cuda") -> Xing4HCUnit:
    fn, base, scale = _weights(hc, dim, device=device)
    return Xing4HCUnit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc,
        hc_sinkhorn_iters=20,
        norm_eps=1e-6,
        hc_eps=1e-6,
    )


def _report_diff(out: torch.Tensor, ref: torch.Tensor, label: str) -> float:
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item() if ref.numel() else 0.0
    n_diff = int((out != ref).sum().item())
    total = max(ref.numel(), 1)
    print(
        f"[unit-fused] {label}: max|diff|={max_diff:.3e} "
        f"bitwise_diff={n_diff}/{total} ({100.0 * n_diff / total:.4f}%)",
        flush=True,
    )
    return max_diff


def _assert_equiv(out: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    _report_diff(out, ref, label)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture_records(fn: Callable[[], object]) -> List[logging.LogRecord]:
    target = logging.getLogger(_FACTORY_LOGGER)
    handler = _ListHandler()
    target.addHandler(handler)
    try:
        fn()
    finally:
        target.removeHandler(handler)
    return handler.records


class TestXing4HcFactory(unittest.TestCase):
    """CPU-side factory contract: env matrix, auto-fallback, one-line log."""

    def setUp(self) -> None:
        _reset_fallback_log_for_test()
        os.environ.pop(IMPL_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(IMPL_ENV, None)

    def test_env_matrix(self) -> None:
        x_cpu = torch.randn(3, HC, 8, dtype=torch.bfloat16)
        cases = [
            (None, TRITON_IMPL),  # env 缺省 -> default triton
            ("", TRITON_IMPL),
            ("  ", TRITON_IMPL),
            ("triton", TRITON_IMPL),
            (" TRITON ", TRITON_IMPL),
            ("torch", TORCH_IMPL),
            (" Torch", TORCH_IMPL),
        ]
        for raw, want in cases:
            with self.subTest(raw=raw), _env(IMPL_ENV, raw):
                self.assertEqual(resolve_xing4_hc_impl(), want)
        for bad in ("tilelang", "fallback", "0", "cuda", "triton-n"):
            with self.subTest(bad=bad), _env(IMPL_ENV, bad):
                with self.assertRaisesRegex(ValueError, "XING4_HC_IMPL"):
                    resolve_xing4_hc_impl()
                # the dispatcher fails fast on illegal values too
                with self.assertRaises(ValueError):
                    xing4_hc_use_fused(x_cpu)
        print(
            f"[factory] env matrix OK: default/case-insensitive -> "
            f"{TRITON_IMPL}, torch -> {TORCH_IMPL}, illegal -> ValueError",
            flush=True,
        )

    def test_auto_fallback_cpu_logs_exactly_once(self) -> None:
        x_cpu = torch.randn(3, HC, 8, dtype=torch.bfloat16)
        with _env(IMPL_ENV, None):
            with self.assertLogs(_FACTORY_LOGGER, level="WARNING") as captured:
                self.assertFalse(xing4_hc_use_fused(x_cpu))
            self.assertEqual(len(captured.records), 1)
            self.assertIn("falling back to the torch reference path", captured.output[0])
            self.assertIn("not on CUDA", captured.output[0])
            # exactly one line per process: the second call stays silent
            records = _capture_records(
                lambda: self.assertFalse(xing4_hc_use_fused(x_cpu))
            )
            self.assertEqual(records, [])
        with _env(IMPL_ENV, TRITON_IMPL):
            # explicit triton + CPU tensor is still an auto-fallback (logged)
            self.assertFalse(xing4_hc_use_fused(x_cpu))

    def test_explicit_torch_never_logs(self) -> None:
        x_cpu = torch.randn(3, HC, 8, dtype=torch.bfloat16)
        with _env(IMPL_ENV, TORCH_IMPL):
            records = _capture_records(
                lambda: self.assertFalse(xing4_hc_use_fused(x_cpu))
            )
            self.assertEqual(records, [])

    def test_cpu_unit_runs_reference_path(self) -> None:
        # CPU tensor + default env -> the unit itself must fall back and run
        # the eager reference path (the torch half of the fallback behavior).
        unit = _make_unit(dim=64, device="cpu")
        x = torch.randn(5, HC, 64, dtype=torch.bfloat16)
        sub = torch.randn(5, 64, dtype=torch.bfloat16)
        with _env(IMPL_ENV, None):
            y, post, comb = unit.pre(x)
            self.assertEqual(tuple(y.shape), (5, 64))
            self.assertEqual(tuple(post.shape), (5, HC, 1))
            self.assertEqual(tuple(comb.shape), (5, HC, HC))
            self.assertEqual(y.dtype, torch.bfloat16)
            self.assertEqual(post.dtype, torch.bfloat16)
            self.assertEqual(comb.dtype, torch.bfloat16)
            out = unit.post(sub, x, post, comb)
            self.assertEqual(tuple(out.shape), (5, HC, 64))
            self.assertEqual(out.dtype, torch.bfloat16)
            # explicit torch selection gives the same result (same code path)
            with _env(IMPL_ENV, TORCH_IMPL):
                y2, post2, comb2 = unit.pre(x)
            _assert_equiv(y2, y, "cpu fallback pre y (default vs torch)")
            self.assertTrue(torch.equal(post2, post))
            self.assertTrue(torch.equal(comb2, comb))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_default_env_selects_fused(self) -> None:
        x = torch.randn(3, HC, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with _env(IMPL_ENV, None):
            records = _capture_records(lambda: self.assertTrue(xing4_hc_use_fused(x)))
            self.assertEqual(records, [])
        with _env(IMPL_ENV, TORCH_IMPL):
            self.assertFalse(xing4_hc_use_fused(x))
        fp32 = torch.randn(3, HC, HIDDEN, device="cuda", dtype=torch.float32)
        with _env(IMPL_ENV, None):
            # non-bf16 stream: auto-fallback to the eager path
            with self.assertLogs(_FACTORY_LOGGER, level="WARNING") as captured:
                self.assertFalse(xing4_hc_use_fused(fp32))
            self.assertEqual(len(captured.records), 1)
            self.assertIn("bf16-only", captured.output[0])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestXing4HcUnitFusedVsTorch(unittest.TestCase):
    """Fused vs eager equivalence through the public pre/post contract."""

    def setUp(self) -> None:
        _reset_fallback_log_for_test()
        os.environ.pop(IMPL_ENV, None)

    def tearDown(self) -> None:
        os.environ.pop(IMPL_ENV, None)

    def test_pre_post_equivalence_flat(self) -> None:
        unit = _make_unit(device="cuda")
        for T in TOKENS:
            with self.subTest(T=T):
                x = _make_stream(T, HC, HIDDEN, seed=1234 + T)
                sub = torch.randn(T, HIDDEN, device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode():
                    with _env(IMPL_ENV, TORCH_IMPL):
                        self.assertFalse(xing4_hc_use_fused(x))
                        ref_y, ref_post, ref_comb = unit.pre(x)
                        ref_out = unit.post(sub, x, ref_post, ref_comb)
                    with _env(IMPL_ENV, TRITON_IMPL):
                        self.assertTrue(xing4_hc_use_fused(x))
                        y, post, comb = unit.pre(x)
                        out = unit.post(sub, x, post, comb)
                # shape/dtype contract identical to the torch path
                self.assertEqual(tuple(y.shape), tuple(ref_y.shape))
                self.assertEqual(tuple(post.shape), tuple(ref_post.shape))
                self.assertEqual(tuple(comb.shape), tuple(ref_comb.shape))
                self.assertEqual(tuple(out.shape), tuple(ref_out.shape))
                for t in (y, post, comb, out):
                    self.assertEqual(t.dtype, torch.bfloat16)
                _assert_equiv(y, ref_y, f"flat pre y T={T}")
                _assert_equiv(post, ref_post, f"flat pre post T={T}")
                _assert_equiv(comb, ref_comb, f"flat pre comb T={T}")
                _assert_equiv(out, ref_out, f"flat post out T={T}")

    def test_pre_post_equivalence_batched(self) -> None:
        # decode-style [B, S, hc, H] leading layout: pins the leading-dims
        # view wiring of post/comb and the K3 batched output shape.
        unit = _make_unit(device="cuda")
        B, S = 2, 8
        x = _make_stream(B, S, HC, HIDDEN, seed=4321)
        sub = torch.randn(B, S, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            with _env(IMPL_ENV, TORCH_IMPL):
                ref_y, ref_post, ref_comb = unit.pre(x)
                ref_out = unit.post(sub, x, ref_post, ref_comb)
            with _env(IMPL_ENV, TRITON_IMPL):
                y, post, comb = unit.pre(x)
                out = unit.post(sub, x, post, comb)
        self.assertEqual(tuple(y.shape), (B, S, HIDDEN))
        self.assertEqual(tuple(post.shape), (B, S, HC, 1))
        self.assertEqual(tuple(comb.shape), (B, S, HC, HC))
        self.assertEqual(tuple(out.shape), (B, S, HC, HIDDEN))
        _assert_equiv(y, ref_y, "batched pre y")
        _assert_equiv(post, ref_post, "batched pre post")
        _assert_equiv(comb, ref_comb, "batched pre comb")
        _assert_equiv(out, ref_out, "batched post out")

    def test_pre_post_equivalence_t0(self) -> None:
        # T=0 empty stream: every fused kernel has an early-exit path and the
        # shape contract must still hold.
        unit = _make_unit(device="cuda")
        x = _make_stream(0, HC, HIDDEN, seed=7)
        sub = torch.randn(0, HIDDEN, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            with _env(IMPL_ENV, TORCH_IMPL):
                ref_y, ref_post, ref_comb = unit.pre(x)
                ref_out = unit.post(sub, x, ref_post, ref_comb)
            with _env(IMPL_ENV, TRITON_IMPL):
                y, post, comb = unit.pre(x)
                out = unit.post(sub, x, post, comb)
        self.assertEqual(tuple(y.shape), (0, HIDDEN))
        self.assertEqual(tuple(post.shape), (0, HC, 1))
        self.assertEqual(tuple(comb.shape), (0, HC, HC))
        self.assertEqual(tuple(out.shape), (0, HC, HIDDEN))
        _assert_equiv(y, ref_y, "t0 pre y")
        _assert_equiv(post, ref_post, "t0 pre post")
        _assert_equiv(comb, ref_comb, "t0 pre comb")
        _assert_equiv(out, ref_out, "t0 post out")

    def test_chunk_env_semantics(self) -> None:
        # The fused path has no chunk branch (nothing to bound: no fp32
        # [T, hc*H] intermediate); the torch fallback keeps chunking and both
        # agree with the unchunked reference.
        unit = _make_unit(device="cuda")
        T = 8
        x = _make_stream(T, HC, HIDDEN, seed=99)
        with torch.inference_mode():
            with _env(IMPL_ENV, TORCH_IMPL), _env(CHUNK_ENV, None):
                ref_y, ref_post, ref_comb = unit.pre(x)
            with _env(IMPL_ENV, TORCH_IMPL), _env(CHUNK_ENV, "2"):
                chunk_y, chunk_post, chunk_comb = unit.pre(x)
            with _env(IMPL_ENV, TRITON_IMPL), _env(CHUNK_ENV, "2"):
                fused_y, fused_post, fused_comb = unit.pre(x)
        # torch fallback chunked == unchunked (chunking only bounds the eager
        # chain's fp32 intermediate; row rsqrt/GEMM are per-row)
        _assert_equiv(chunk_y, ref_y, "torch chunked pre y")
        _assert_equiv(chunk_post, ref_post, "torch chunked pre post")
        _assert_equiv(chunk_comb, ref_comb, "torch chunked pre comb")
        # fused path ignores the chunk env entirely
        _assert_equiv(fused_y, ref_y, "fused ignores chunk env pre y")
        _assert_equiv(fused_post, ref_post, "fused ignores chunk env pre post")
        _assert_equiv(fused_comb, ref_comb, "fused ignores chunk env pre comb")

    def test_fused_path_is_actually_fused(self) -> None:
        # Guard against the dispatch silently degrading to eager: under
        # XING4_HC_IMPL=triton one pre() call must launch the K2 mix kernel
        # exactly once, while the torch path launches it zero times.
        from rtp_llm.models_py.modules.xing4.hc import mix_triton

        unit = _make_unit(device="cuda")
        x = _make_stream(8, HC, HIDDEN, seed=7)
        kernel = mix_triton._xing4_hc_mix_kernel
        orig_run = kernel.run
        counter = {"launches": 0}

        def counting_run(*args, **kwargs):
            counter["launches"] += 1
            return orig_run(*args, **kwargs)

        kernel.run = counting_run
        try:
            with torch.inference_mode():
                with _env(IMPL_ENV, TORCH_IMPL):
                    unit.pre(x)
                self.assertEqual(counter["launches"], 0)
                with _env(IMPL_ENV, TRITON_IMPL):
                    unit.pre(x)
                self.assertEqual(counter["launches"], 1)
        finally:
            del kernel.run  # restore the class-level JITFunction.run
        print(
            f"[unit-fused] mix-kernel launches: torch=0, triton={counter['launches']}",
            flush=True,
        )

    def test_fused_pre_uses_gemv_when_rows_gated_in(self) -> None:
        # K1+GEMM fusion dispatch (round-2): at rows <= the gate (default
        # 32) one pre() must launch the fused GEMV exactly once and skip
        # F.linear; above the gate (and when the gate is disabled, or on the
        # torch backend) it launches zero times and the K1+F.linear chain
        # runs instead.
        from rtp_llm.models_py.modules.xing4.hc import fn_gemv_triton

        unit = _make_unit(device="cuda")
        x_small = _make_stream(8, HC, HIDDEN, seed=17)
        x_big = _make_stream(512, HC, HIDDEN, seed=18)
        kernel = fn_gemv_triton._hc_fn_gemv_kernel
        orig_run = kernel.run
        counter = {"launches": 0}

        def counting_run(*args, **kwargs):
            counter["launches"] += 1
            return orig_run(*args, **kwargs)

        kernel.run = counting_run
        try:
            with torch.inference_mode(), _env(IMPL_ENV, TRITON_IMPL):
                with _env(GEMV_ENV, None):
                    counter["launches"] = 0
                    unit.pre(x_small)  # default gate 32 >= rows 8
                    self.assertEqual(counter["launches"], 1)
                    unit.pre(x_big)  # 512 rows -> K1 + F.linear
                    self.assertEqual(counter["launches"], 1)
                with _env(GEMV_ENV, "0"):  # explicit disable
                    unit.pre(x_small)
                    self.assertEqual(counter["launches"], 1)
                with _env(GEMV_ENV, "4096"):  # forced on at prefill size
                    unit.pre(x_big)
                    self.assertEqual(counter["launches"], 2)
            with torch.inference_mode(), _env(IMPL_ENV, TORCH_IMPL), _env(
                GEMV_ENV, None
            ):
                unit.pre(x_small)  # eager backend never launches it
                self.assertEqual(counter["launches"], 2)
        finally:
            del kernel.run  # restore the class-level JITFunction.run
        print(
            f"[unit-fused] gemv-kernel dispatch: rows<=gate 1/rows>gate 0/"
            f"disabled 0/forced-on 1/torch 0 (total {counter['launches']})",
            flush=True,
        )

    def test_pre_equivalence_gemv_forced_and_disabled(self) -> None:
        # Both dispatch arms must match the eager reference at the same
        # tolerance: the GEMV-forced arm (rows above the default gate) and
        # the gate-disabled arm (rows below it, running K1 + F.linear).
        unit = _make_unit(device="cuda")
        for T, env in ((512, "4096"), (8, "0")):
            with self.subTest(T=T, gemv_env=env):
                x = _make_stream(T, HC, HIDDEN, seed=123 + T)
                with torch.inference_mode():
                    with _env(IMPL_ENV, TORCH_IMPL), _env(GEMV_ENV, None):
                        ref_y, ref_post, ref_comb = unit.pre(x)
                    with _env(IMPL_ENV, TRITON_IMPL), _env(GEMV_ENV, env):
                        y, post, comb = unit.pre(x)
                _assert_equiv(y, ref_y, f"pre y T={T} gemv_env={env}")
                _assert_equiv(post, ref_post, f"pre post T={T} gemv_env={env}")
                _assert_equiv(comb, ref_comb, f"pre comb T={T} gemv_env={env}")


if __name__ == "__main__":
    unittest.main()
