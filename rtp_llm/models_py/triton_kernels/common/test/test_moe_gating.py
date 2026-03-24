"""Unit tests for moe_gating.py — SigmoidGateScaleAdd kernel.

Validates numerical correctness and benchmarks performance against
a reference PyTorch implementation across a Cartesian product of
(T, H, dtype) parameter combinations.

Run with unittest:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_moe_gating.py -v -s
or directly:
    python rtp_llm/models_py/triton_kernels/common/test/test_moe_gating.py
"""

import sys
import unittest

import torch
from torch.profiler import ProfilerActivity, profile

# nn.Module wrapper lives in the cuda modules layer
from rtp_llm.models_py.modules.base.cuda.moe_gating import SigmoidGateScaleAdd

# Kernel internals (heuristics + constants live in the triton_kernels layer)
from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    _MAX_BLOCK_H,
    _MIN_BLOCK_H,
    _select_block_h,
)

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

T_VALUES = [1, 8, 32, 256, 1024, 4096, 8192, 16384]
H_VALUES = [2048, 4096, 7168]
DTYPES = [torch.bfloat16, torch.float16]

WARMUP = 10
REPEAT = 50

# ---------------------------------------------------------------------------
# Reference implementation (fp32 arithmetic — used for correctness checks)
# ---------------------------------------------------------------------------


def _ref_SigmoidGateScaleAdd(
    gate: torch.Tensor,  # [T, 1]
    shared: torch.Tensor,  # [T, H]
    experts: torch.Tensor,  # [T, H]
) -> torch.Tensor:
    """Pure-PyTorch reference using fp32 arithmetic."""
    result = torch.sigmoid(gate.float()) * shared.float() + experts.float()
    return result.to(shared.dtype)


# ---------------------------------------------------------------------------
# Benchmark helper using torch.profiler
# ---------------------------------------------------------------------------


def _benchmark_us(fn, warmup: int = WARMUP, repeat: int = REPEAT) -> float:
    """Measure average CUDA execution time (µs) using torch.profiler.

    All CUDA kernels launched inside *fn* are captured; their total
    device_time_total (across all `repeat` calls) is divided by `repeat`
    to obtain the per-call average.

    Note: uses `device_time_total` (the portable API); older PyTorch used
    `cuda_time_total` which was renamed in recent releases.
    """
    # Warmup — not profiled
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(repeat):
            fn()
    torch.cuda.synchronize()

    total_us = sum(e.device_time_total for e in prof.key_averages())
    return total_us / repeat


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCorrectness(unittest.TestCase):
    """Correctness: Triton kernel vs fp32 PyTorch reference over T × H × dtype."""

    def _check(self, T: int, H: int, dtype: torch.dtype) -> float:
        torch.manual_seed(42 + T + H)
        gate = torch.randn(T, 1, dtype=dtype, device="cuda")
        shared = torch.randn(T, H, dtype=dtype, device="cuda")
        experts = torch.randn(T, H, dtype=dtype, device="cuda")

        op = SigmoidGateScaleAdd()
        ref = _ref_SigmoidGateScaleAdd(gate.clone(), shared.clone(), experts.clone())
        actual = op(gate.clone(), shared.clone(), experts.clone())

        atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
        rtol = 1e-2
        max_diff = (actual.float() - ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(actual.float(), ref.float(), atol=atol, rtol=rtol),
            f"T={T}, H={H}, dtype={dtype}: max_diff={max_diff:.4e} > atol={atol}",
        )
        return max_diff

    def test_correctness_cartesian(self):
        """Correctness over full T × H × dtype Cartesian grid."""
        for T in T_VALUES:
            for H in H_VALUES:
                for dtype in DTYPES:
                    with self.subTest(T=T, H=H, dtype=dtype):
                        max_diff = self._check(T, H, dtype)
                        print(
                            f"  correctness  T={T:5d} H={H:5d} "
                            f"dtype={str(dtype).replace('torch.',''):>8}  "
                            f"max_diff={max_diff:.3e}  OK"
                        )

    def test_nonpow2_H(self):
        """Non-power-of-2 H values to verify boundary mask correctness."""
        for T, H, dtype in [
            (16, 3000, torch.bfloat16),
            (32, 5000, torch.float16),
            (128, 6500, torch.bfloat16),
        ]:
            with self.subTest(T=T, H=H, dtype=dtype):
                self._check(T, H, dtype)

    def test_inplace_same_object(self):
        """SigmoidGateScaleAdd must return the same experts tensor (in-place)."""
        gate = torch.randn(8, 1, dtype=torch.bfloat16, device="cuda")
        shared = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        experts = torch.randn(8, 4096, dtype=torch.bfloat16, device="cuda")
        ret = SigmoidGateScaleAdd()(gate, shared, experts)
        self.assertEqual(
            ret.data_ptr(),
            experts.data_ptr(),
            "Return value must be the same tensor as experts",
        )

    def test_empty_T(self):
        """T=0 edge case must not crash and return correct shape."""
        gate = torch.empty(0, 1, dtype=torch.bfloat16, device="cuda")
        shared = torch.empty(0, 4096, dtype=torch.bfloat16, device="cuda")
        experts = torch.empty(0, 4096, dtype=torch.bfloat16, device="cuda")
        ret = SigmoidGateScaleAdd()(gate, shared, experts)
        self.assertEqual(list(ret.shape), [0, 4096])


# ---------------------------------------------------------------------------
# _select_block_h heuristic tests
# ---------------------------------------------------------------------------


class TestSelectBlockH(unittest.TestCase):
    """Unit tests for the BLOCK_H selection heuristic."""

    def test_power_of_two(self):
        """BLOCK_H must always be a power of 2."""
        for T in T_VALUES:
            for H in H_VALUES:
                with self.subTest(T=T, H=H):
                    bh = _select_block_h(T, H)
                    self.assertEqual(
                        bh & (bh - 1),
                        0,
                        f"T={T}, H={H}: BLOCK_H={bh} is not a power of 2",
                    )

    def test_within_hard_bounds(self):
        """BLOCK_H must be within [MIN_BLOCK_H, MAX_BLOCK_H]."""
        for T in T_VALUES:
            for H in H_VALUES:
                with self.subTest(T=T, H=H):
                    bh = _select_block_h(T, H)
                    self.assertGreaterEqual(bh, _MIN_BLOCK_H)
                    self.assertLessEqual(bh, _MAX_BLOCK_H)

    def test_small_T_smaller_block_h(self):
        """Small T should yield smaller BLOCK_H than large T (more SM parallelism)."""
        for H in H_VALUES:
            bh_decode = _select_block_h(1, H)
            bh_prefill = _select_block_h(1024, H)
            self.assertLessEqual(
                bh_decode,
                bh_prefill,
                f"H={H}: BLOCK_H for T=1 ({bh_decode}) should be ≤ T=1024 ({bh_prefill})",
            )

    def test_small_H_respects_min_block_h(self):
        """For H=2048/4096, BLOCK_H must not drop below MIN_BLOCK_H."""
        for T in T_VALUES:
            for H in [2048, 4096]:
                with self.subTest(T=T, H=H):
                    bh = _select_block_h(T, H)
                    self.assertGreaterEqual(
                        bh,
                        _MIN_BLOCK_H,
                        f"T={T}, H={H}: BLOCK_H={bh} < MIN_BLOCK_H={_MIN_BLOCK_H}",
                    )


# ---------------------------------------------------------------------------
# Performance tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPerformance(unittest.TestCase):
    """Performance: Triton kernel vs native PyTorch ops, measured with torch.profiler.

    Benchmarks the actual production ops:
      - Torch baseline : torch.sigmoid(gate) * shared + experts  (3 separate kernels)
      - Triton kernel  : SigmoidGateScaleAdd(gate, shared, experts)  (1 fused kernel)

    Results are printed as a formatted table at the end (via tearDownClass).

    The test asserts only that the Triton kernel is not more than 50% slower
    (generous tolerance for measurement noise in CI environments).
    """

    @classmethod
    def setUpClass(cls):
        # List of (T, H, dtype_str, torch_us, triton_us) tuples
        cls._results: list = []

    @classmethod
    def tearDownClass(cls):
        _print_perf_table(cls._results)

    def _bench_pair(self, T: int, H: int, dtype: torch.dtype):
        """Return (torch_us, triton_us) average per-call times."""
        torch.manual_seed(0)
        gate = torch.randn(T, 1, dtype=dtype, device="cuda")
        shared = torch.randn(T, H, dtype=dtype, device="cuda")
        experts = torch.randn(T, H, dtype=dtype, device="cuda")

        # Torch baseline: the 3-op sequence that production currently runs.
        def torch_fn():
            return torch.sigmoid(gate) * shared + experts

        # Triton kernel: reads gate, shared, experts; writes experts in-place.
        op = SigmoidGateScaleAdd()

        def triton_fn():
            op(gate, shared, experts)

        torch_us = _benchmark_us(torch_fn)
        triton_us = _benchmark_us(triton_fn)
        return torch_us, triton_us

    def test_performance_cartesian(self):
        """Benchmark over full T × H × dtype Cartesian grid."""
        for dtype in DTYPES:
            for T in T_VALUES:
                for H in H_VALUES:
                    with self.subTest(T=T, H=H, dtype=dtype):
                        torch_us, triton_us = self._bench_pair(T, H, dtype)
                        dtype_str = str(dtype).replace("torch.", "")
                        self.__class__._results.append(
                            (T, H, dtype_str, torch_us, triton_us)
                        )
                        # Soft assertion: not catastrophically slower
                        self.assertLessEqual(
                            triton_us,
                            torch_us * 1.5,
                            f"T={T}, H={H}, dtype={dtype_str}: "
                            f"Triton {triton_us:.1f}µs is >50% slower than "
                            f"Torch {torch_us:.1f}µs",
                        )


# ---------------------------------------------------------------------------
# Pretty-print performance table
# ---------------------------------------------------------------------------


def _print_perf_table(results: list):
    if not results:
        return

    col_w = dict(T=6, H=6, dtype=9, torch_us=12, triton_us=12, speedup=10)
    hdr = (
        f"{'T':>{col_w['T']}} {'H':>{col_w['H']}} {'dtype':>{col_w['dtype']}} "
        f"{'torch(µs)':>{col_w['torch_us']}} {'triton(µs)':>{col_w['triton_us']}} "
        f"{'speedup':>{col_w['speedup']}}"
    )
    sep = "─" * len(hdr)

    lines = [
        "",
        "=" * len(hdr),
        "  Performance: SigmoidGateScaleAdd — Torch vs Triton",
        "=" * len(hdr),
        hdr,
        sep,
    ]

    prev_dtype = None
    for T, H, dtype_str, torch_us, triton_us in sorted(
        results, key=lambda r: (r[2], r[0], r[1])
    ):
        if prev_dtype is not None and dtype_str != prev_dtype:
            lines.append(sep)
        prev_dtype = dtype_str

        speedup_pct = (torch_us - triton_us) / torch_us * 100
        sign = "+" if speedup_pct >= 0 else ""
        lines.append(
            f"{T:>{col_w['T']}} {H:>{col_w['H']}} {dtype_str:>{col_w['dtype']}} "
            f"{torch_us:>{col_w['torch_us']}.2f} "
            f"{triton_us:>{col_w['triton_us']}.2f} "
            f"{sign}{speedup_pct:>{col_w['speedup'] - 1}.1f}%"
        )

    lines += [
        sep,
        "  (+) Triton faster   (−) Triton slower",
        "=" * len(hdr),
        "",
    ]
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
