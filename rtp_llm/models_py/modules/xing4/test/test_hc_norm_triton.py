"""K1 equivalence UT for the Xing4 hc-row RMSNorm Triton fusion.

The reference is hand-written step by step (lane contract) from the
mathematical authority ``Xing4HCUnit._pre_impl`` in
``rtp_llm/models_py/model_desc/xing4_0_model.py``:

    x_flat      = x.flatten(-2)
    rsqrt       = torch.rsqrt(
        x_flat.float().square().mean(-1, keepdim=True) + eps)
    flat_normed = (x_flat.float() * rsqrt).to(x.dtype)

Gates covered:

* T in {1, 2, 8, 512, 4096, 16384} at hc=4, H=3584 (cols=14336) bf16 with
  torch.testing.assert_close(rtol=atol=2**-7) and printed max|diff|;
* eps boundaries 0 / 1e-6 / 1e-4;
* small-H generality (H=16 -> cols=64), 4D batched input, row-sliced input,
  and preallocated ``out`` reuse;
* row cap T=65536 and the int64 index contract (T=150000 rows put
  row * 14336 past INT32_MAX);
* launch-count report: eager reference chain 6-8 CUDA kernels vs the fused
  path's single Triton launch (JITFunction.run interception + aten op trace);
* the R1-reuse thin wrapper (XING4_HC_NORM_IMPL=torch, F.rms_norm).
"""

import os
import unittest
from contextlib import contextmanager

import torch

from rtp_llm.models_py.modules.xing4.hc import norm_triton
from rtp_llm.models_py.modules.xing4.hc.norm_triton import (
    hc_row_rms_norm,
    hc_row_rms_norm_torch,
    hc_row_rms_norm_triton,
)

HC = 4
HIDDEN = 3584
COLS = HC * HIDDEN  # 14336 = 7 * 2048
RTOL = 2.0**-7
ATOL = 2.0**-7

# Leaf aten ops that launch a CUDA kernel for the eager reference chain.
# Composite wrappers (aten::to / aten::_to_copy) are excluded on purpose:
# each cast shows up here exactly once as aten::copy_.
_COMPUTE_LEAF_OPS = (
    "aten::copy_",
    "aten::mul",
    "aten::mean",
    "aten::sum",
    "aten::div",
    "aten::add",
    "aten::sub",
    "aten::rsqrt",
    "aten::sqrt",
)


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


def _make_x(*shape, seed=0, scale=3.0, device="cuda"):
    torch.manual_seed(seed)
    x = torch.randn(*shape, device=device, dtype=torch.bfloat16)
    if scale != 1.0:
        x = x.mul_(scale)
    return x


def _reference_flat_normed(x, eps):
    """Hand-written step-by-step reference (xing4_0_model.py chain)."""
    x_flat = x.flatten(-2)
    rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + eps)
    return (x_flat.float() * rsqrt).to(x.dtype)


def _report_diff(out, ref, label):
    diff = (out.float() - ref.float()).abs()
    max_diff = diff.max().item()
    n_diff = int((out != ref).sum().item())
    total = max(ref.numel(), 1)
    print(
        f"[equiv] {label}: max|diff|={max_diff:.3e} "
        f"bitwise_diff={n_diff}/{total} ({100.0 * n_diff / total:.4f}%)",
        flush=True,
    )
    return max_diff


def _assert_equiv(out, ref, label):
    torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)
    _report_diff(out, ref, label)


def _count_triton_launches(fn):
    """Count actual Triton kernel launches by intercepting JITFunction.run."""
    kernel = norm_triton._hc_row_rmsnorm_kernel
    orig_run = kernel.run
    counter = {"launches": 0}

    def counting_run(*args, **kwargs):
        counter["launches"] += 1
        return orig_run(*args, **kwargs)

    kernel.run = counting_run
    try:
        fn()
    finally:
        del kernel.run  # restore the class-level JITFunction.run
    return counter["launches"]


def _aten_ops(fn):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.name.startswith("aten::")]


def _cuda_time_ms(fn, warmup=5, iters=20):
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


class TestHCNormDispatch(unittest.TestCase):
    """CPU-runnable env-switch contract (mirrors dsv4 test_hc_impl.py)."""

    def test_env_selects_torch_backend(self):
        x = _make_x(3, HC, 16, device="cpu")
        ref = _reference_flat_normed(x, 1e-6)
        with _env("XING4_HC_NORM_IMPL", "torch"):
            out = hc_row_rms_norm(x, 1e-6)
        self.assertEqual(tuple(out.shape), tuple(ref.shape))
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    def test_default_and_triton_env_require_cuda(self):
        x = _make_x(3, HC, 16, device="cpu")
        for value in (None, "triton"):
            with self.subTest(env=value), _env("XING4_HC_NORM_IMPL", value):
                with self.assertRaises(ValueError):
                    hc_row_rms_norm(x, 1e-6)

    def test_invalid_impl_value_raises(self):
        x = _make_x(3, HC, 16, device="cpu")
        with _env("XING4_HC_NORM_IMPL", "bogus"):
            with self.assertRaises(ValueError):
                hc_row_rms_norm(x, 1e-6)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestHCNormTritonCuda(unittest.TestCase):
    def test_t_sweep_matches_reference(self):
        for t in (1, 2, 8, 512, 4096, 16384):
            with self.subTest(T=t):
                x = _make_x(t, HC, HIDDEN, seed=t)
                x_snapshot = x.clone()
                ref = _reference_flat_normed(x, 1e-6)
                out = hc_row_rms_norm_triton(x, 1e-6)
                self.assertIs(out.dtype, torch.bfloat16)
                self.assertEqual(tuple(out.shape), (t, COLS))
                _assert_equiv(out, ref, f"triton T={t} eps=1e-6")
                self.assertTrue(torch.equal(x, x_snapshot))

    def test_eps_boundaries(self):
        for eps in (0.0, 1e-6, 1e-4):
            with self.subTest(eps=eps):
                x = _make_x(512, HC, HIDDEN, seed=11)
                ref = _reference_flat_normed(x, eps)
                out = hc_row_rms_norm_triton(x, eps)
                _assert_equiv(out, ref, f"triton T=512 eps={eps}")

    def test_small_hidden_generality(self):
        # cols=64 exercises the small-BLOCK_N specialization path.
        x = _make_x(8, HC, 16, seed=3)
        ref = _reference_flat_normed(x, 1e-6)
        out = hc_row_rms_norm_triton(x, 1e-6)
        self.assertEqual(tuple(out.shape), (8, HC * 16))
        _assert_equiv(out, ref, "triton H=16 cols=64")

    def test_batched_4d_input(self):
        x = _make_x(2, 5, HC, HIDDEN, seed=4)
        ref = _reference_flat_normed(x, 1e-6)
        out = hc_row_rms_norm_triton(x, 1e-6)
        self.assertEqual(tuple(out.shape), (2, 5, COLS))
        _assert_equiv(out, ref, "triton 4D [2,5,hc,H]")

    def test_row_sliced_input(self):
        # Chunked callers pass row slices: row stride == cols, no copy.
        big = _make_x(600, HC, HIDDEN, seed=12)
        x = big[100:228]
        self.assertEqual(x.stride(0), COLS)
        ref = _reference_flat_normed(x, 1e-6)
        out = hc_row_rms_norm_triton(x, 1e-6)
        _assert_equiv(out, ref, "triton row-slice T=128")

    def test_out_buffer_reuse(self):
        x = _make_x(64, HC, HIDDEN, seed=10)
        ref = _reference_flat_normed(x, 1e-6)
        out = torch.empty((64, COLS), dtype=torch.bfloat16, device="cuda")
        ret = hc_row_rms_norm_triton(x, 1e-6, out=out)
        self.assertEqual(ret.data_ptr(), out.data_ptr())
        _assert_equiv(ret, ref, "triton out= T=64")

    def test_row_cap_and_int64_offsets(self):
        # T=65536 is the required cap; T=150000 pushes row * 14336 past
        # INT32_MAX (2^31 / 14336 = 149725.6 rows), the failure mode the
        # int64 pid cast prevents. The reference is compared chunk-wise
        # (16384 rows, mirroring _pre_impl's chunk loop) to bound fp32
        # intermediates exactly like the eager path does.
        free_bytes, _ = torch.cuda.mem_get_info()
        elem = HC * HIDDEN
        for t, note in ((65536, "row cap"), (150000, "int32-overflow")):
            with self.subTest(T=t):
                needed = t * elem * 2 * 2 + (3 << 30)
                if free_bytes < needed:
                    print(
                        f"[skip] T={t} ({note}): need ~{needed / 2**30:.1f}GiB, "
                        f"free {free_bytes / 2**30:.1f}GiB",
                        flush=True,
                    )
                    continue
                x = _make_x(t, HC, HIDDEN, seed=1)
                out = hc_row_rms_norm_triton(x, 1e-6)
                self.assertEqual(tuple(out.shape), (t, COLS))
                worst = 0.0
                for s in range(0, t, 16384):
                    e = min(s + 16384, t)
                    ref = _reference_flat_normed(x[s:e], 1e-6)
                    torch.testing.assert_close(
                        out[s:e], ref, rtol=RTOL, atol=ATOL
                    )
                    worst = max(
                        worst,
                        (out[s:e].float() - ref.float()).abs().max().item(),
                    )
                print(
                    f"[equiv] triton T={t} ({note}): max|diff|={worst:.3e} "
                    "(chunked reference)",
                    flush=True,
                )
                del x, out
                torch.cuda.empty_cache()

    def test_launch_count_report(self):
        x = _make_x(512, HC, HIDDEN, seed=9)
        eps = 1e-6
        hc_row_rms_norm_triton(x, eps)  # JIT-compile outside the measurement

        ref_ops = _aten_ops(lambda: _reference_flat_normed(x, eps))
        ref_compute = [n for n in ref_ops if n in _COMPUTE_LEAF_OPS]
        # Eager chain: copy_(x.float()), mul(square), mean, add, rsqrt,
        # copy_(x.float()), mul, copy_(->bf16) ~= 8 kernel-launching ops.
        self.assertGreaterEqual(len(ref_compute), 6)
        self.assertLessEqual(len(ref_compute), 12)
        print(
            f"[launches] reference eager aten trace: {ref_ops}",
            flush=True,
        )
        print(
            f"[launches] reference eager kernel-launching aten ops: "
            f"{len(ref_compute)} ({ref_compute})",
            flush=True,
        )

        state = {}

        def fused_once():
            state["launches"] = _count_triton_launches(
                lambda: hc_row_rms_norm_triton(x, eps)
            )

        fused_ops = _aten_ops(fused_once)
        fused_compute = [n for n in fused_ops if n in _COMPUTE_LEAF_OPS]
        self.assertEqual(fused_compute, [])
        self.assertEqual(state["launches"], 1)
        print(f"[launches] fused aten trace: {fused_ops}", flush=True)
        print(
            f"[launches] fused path: {state['launches']} Triton kernel "
            "launch per call, 0 torch compute kernels",
            flush=True,
        )
        print(
            f"[launches] kernel-launch reduction: reference ~{len(ref_compute)} "
            f"micro kernels -> fused {state['launches']}",
            flush=True,
        )

        ref = _reference_flat_normed(x, eps)
        out = hc_row_rms_norm_triton(x, eps)
        torch.testing.assert_close(out, ref, rtol=RTOL, atol=ATOL)

    def test_timing_report(self):
        # Informational CUDA-event timings; not a gate.
        for t in (512, 4096, 16384):
            x = _make_x(t, HC, HIDDEN, seed=2)
            t_ref = _cuda_time_ms(lambda: _reference_flat_normed(x, 1e-6))
            t_fused = _cuda_time_ms(lambda: hc_row_rms_norm_triton(x, 1e-6))
            print(
                f"[timing] T={t}: reference={t_ref:.3f}ms "
                f"fused={t_fused:.3f}ms "
                f"speedup={t_ref / max(t_fused, 1e-9):.2f}x",
                flush=True,
            )

    def test_torch_backend_matches_reference(self):
        # R1 reuse verdict: thin F.rms_norm wrapper, same equivalence class.
        for t in (1, 512, 4096):
            with self.subTest(T=t):
                x = _make_x(t, HC, HIDDEN, seed=6)
                ref = _reference_flat_normed(x, 1e-6)
                out = hc_row_rms_norm_torch(x, 1e-6)
                self.assertEqual(tuple(out.shape), (t, COLS))
                _assert_equiv(out, ref, f"torch-backend T={t}")

    def test_env_dispatch_backends_cuda(self):
        x = _make_x(64, HC, HIDDEN, seed=8)
        ref = _reference_flat_normed(x, 1e-6)
        with _env("XING4_HC_NORM_IMPL", "torch"):
            out = hc_row_rms_norm(x, 1e-6)
        _assert_equiv(out, ref, "env=torch dispatch T=64")
        with _env("XING4_HC_NORM_IMPL", "triton"):
            out = hc_row_rms_norm(x, 1e-6)
        _assert_equiv(out, ref, "env=triton dispatch T=64")
        with _env("XING4_HC_NORM_IMPL", None):
            out = hc_row_rms_norm(x, 1e-6)
        _assert_equiv(out, ref, "env=default(triton) dispatch T=64")


if __name__ == "__main__":
    unittest.main()
