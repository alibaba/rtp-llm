"""V4.1 norm+group32-quant fusion tests against independent references.

The norm is checked against an fp64 mathematical RMSNorm (modeling the two
documented CUDA13 native-kernel semantics: fp32 FTZ on subnormal products and
weight_bias=0 signed-zero normalization) within 1 bf16 ULP — the bound caused
by the native kernel's fp32 accumulation order versus exact math.

The quant half is checked byte-exactly against the existing standalone
production kernel ``sgl_per_token_group_quant_fp8`` applied to the candidate's
own materialized bf16 norm: the fused kernel must see exactly that bf16 value.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

# Load this standalone candidate without importing the attention/model package.
_spec = importlib.util.spec_from_file_location(
    "v41_norm_quant_candidate",
    Path(__file__).resolve().parents[1] / "_v41_norm_quant.py",
)
candidate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(candidate)


def reference_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent fp64 RMSNorm with the native kernel's documented FTZ semantics."""
    # Weight path: the native kernel loads w as fp32 and applies
    # ``add.rn.ftz.f32 w, +0.0`` (weight_bias=0). FTZ flushes fp32-subnormal
    # weights (bf16 subnormals included) to signed zero; RN addition then
    # normalizes -0.0 to +0.0.
    wf = weight.float()
    wf = torch.where(wf.abs() < 2.0**-126, wf * 0.0, wf)
    wf = wf + 0.0
    wd = wf.double()
    xd = x.double()
    inv = torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)
    y = xd * inv * wd
    # Output path: fp32-subnormal products flush to signed zero before the
    # bf16 store (mul.ftz.f32 in the native kernel).
    y = torch.where(y.abs() < 2.0**-126, y * 0.0, y)
    return y.to(torch.bfloat16)


def reference_quant(norm: torch.Tensor, mode: str):
    """Standalone production quant kernel over an already-materialized norm."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

    with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=mode):
        return sgl_per_token_group_quant_fp8(
            norm,
            group_size=32,
            eps=torch.finfo(torch.float32).tiny,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )


def _bf16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    ai = actual.view(torch.int16).to(torch.int32)
    bi = expected.view(torch.int16).to(torch.int32)
    return (ai - bi).abs()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0),
    "requires SM100 CUDA",
)
class V41NormQuantTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260922)

    def assertBytesEqual(self, actual, expected, label):
        self.assertIsNotNone(actual)
        for a, b in zip(actual, expected):
            self.assertEqual(a.shape, b.shape, label)
            self.assertEqual(a.dtype, b.dtype, label)
            # contiguous() permits byte comparisons of column-major int32.
            aa, bb = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            self.assertTrue(
                torch.equal(aa, bb),
                f"{label} ({a.dtype}): {torch.count_nonzero(aa != bb).item()} differing bytes",
            )

    def check(self, x, w, eps=1e-6, mode="auto", inplace=False):
        # Snapshot for the reference first: the inplace path overwrites x.
        ref = reference_norm(x, w, eps)
        norm, quant, scales = candidate.rmsnorm_group32_quant(
            x, w, eps, quant_kernel=mode, out_norm=x if inplace else None
        )
        self.assertEqual(norm.dtype, torch.bfloat16)

        # Independent math oracle: <= 1 bf16 ULP (fp32 accumulation order
        # versus exact fp64 rounding at the bf16 store).
        dist = _bf16_ulp_distance(norm, ref)
        max_ulp = int(dist.max().item()) if dist.numel() else 0
        self.assertLessEqual(
            max_ulp, 1, f"norm differs from fp64 reference by {max_ulp} bf16 ULP"
        )

        # Standalone production kernel over the candidate's own bf16 norm must
        # reproduce the fused quant and packed scales byte-exactly.
        q_ref, s_ref = reference_quant(norm, mode)
        self.assertBytesEqual((quant, scales), (q_ref, s_ref), "fused quant")

        if inplace:
            self.assertEqual(norm.data_ptr(), x.data_ptr())
        self.assertEqual(scales.shape, (x.shape[0], 40))
        self.assertEqual(scales.stride(), (1, max(1, (x.shape[0] + 3) // 4 * 4)))
        return norm, quant, scales

    def test_shapes_modes_and_padding(self):
        for m in [0, 1, 32, 33, 819, 820, 1021, 1664, 4096, 8192, 32768]:
            x = torch.randn(m, 5120, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
            for mode in ["legacy", "v2", "auto"]:
                with self.subTest(m=m, mode=mode):
                    self.check(x, w, mode=mode)

    def test_zero_tiny_signed_zero_and_scale_boundaries(self):
        # Mixed groups expose both quant floors without relying on RNG to
        # generate tiny values. +/-0 and fp32-subnormal products must flush
        # consistently with the documented FTZ semantics.
        values = [
            0.0,
            -0.0,
            2.0**-133,
            -(2.0**-126),
            2.0**-110,
            2.0**-40,
            2.0**-25,
            1.0,
            -1.0,
            448.0,
            450.0,
        ]
        x = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
        x = x.repeat((5120 + len(values) - 1) // len(values))[:5120].repeat(33, 1)
        for scale in [0.0, -0.0, 2.0**-126, 2.0**-100, 1e-8, 1.0]:
            w = torch.full((5120,), scale, device="cuda", dtype=torch.bfloat16)
            for mode in ["legacy", "v2", "auto"]:
                with self.subTest(scale=scale, mode=mode):
                    self.check(x, w, mode=mode)
        # The discontinuity in auto must be observable on the zero matrix.
        w = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        for m, selected in [(819, "legacy"), (820, "v2")]:
            x = torch.zeros(m, 5120, device="cuda", dtype=torch.bfloat16)
            with self.subTest(m=m):
                norm, quant, scales = self.check(x, w)
                # Cross-check: the other mode's standalone kernel differs on
                # the zero matrix (auto picks per-shape).
                other = "v2" if selected == "legacy" else "legacy"
                q_other, s_other = reference_quant(norm, other)
                self.assertFalse(
                    torch.equal(
                        scales.contiguous().view(torch.uint8),
                        s_other.contiguous().view(torch.uint8),
                    ),
                    f"auto mode at m={m} did not observe the {selected}/{other} split",
                )

    def test_epsilon_dynamic_range_and_inplace(self):
        for seed, amplitude in [(7, 1e-20), (31, 1e-3), (997, 1.0), (8191, 1e10)]:
            torch.manual_seed(seed)
            x = (torch.randn(1021, 5120, device="cuda") * amplitude).bfloat16()
            w = (torch.randn(5120, device="cuda") * 0.1 + 1).bfloat16()
            for eps in [1e-12, 1e-6, 1e-5]:
                for mode in ["legacy", "v2"]:
                    with self.subTest(seed=seed, eps=eps, mode=mode):
                        self.check(x.clone(), w, eps, mode, inplace=True)

    def test_ue8m0_exponent_and_fp8_midpoint_boundaries(self):
        x = torch.ones(37, 5120, device="cuda", dtype=torch.bfloat16)
        # With constant x, BF16 norm materialization retains these weights.
        # Groups straddle powers of two in amax/448 and E4M3 half-way values.
        groups = []
        for exponent in [-130, -126, -120, -34, -33, -32, -10, -1, 0, 1, 10, 60]:
            for maximum in [446.0, 448.0, 450.0]:
                values = [
                    maximum,
                    -maximum,
                    0.0,
                    -0.0,
                    1.0625,
                    1.1875,
                    0.0009765625,
                    -1.0625,
                ]
                groups.extend([v * 2.0**exponent for v in values] * 4)
        w = torch.tensor(groups, device="cuda", dtype=torch.bfloat16).repeat(5)[:5120]
        for mode in ["legacy", "v2", "auto"]:
            with self.subTest(mode=mode):
                self.check(x, w, mode=mode)

    def test_existing_env_selection(self):
        x = torch.zeros(820, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        for mode in ["auto", "legacy", "v2"]:
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=" " + mode.upper() + " "):
                with self.subTest(mode=mode):
                    self.check(x, w, mode="auto")

    def test_graph_replay_and_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            x = torch.randn(820, 5120, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                candidate.rmsnorm_group32_quant(x, w)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                actual = candidate.rmsnorm_group32_quant(x, w)
            for i in range(3):
                x.fill_(i * 1e-10)
                graph.replay()
                # Replay must be deterministic against a fresh host-side run.
                replayed = candidate.rmsnorm_group32_quant(x, w)
                self.assertBytesEqual(actual, replayed, "graph replay")
                # ... and correct against the independent oracles.
                self.check(x, w)
        torch.cuda.current_stream().wait_stream(stream)

    def test_unsupported_inputs_return_none(self):
        x = torch.empty(33, 5120, device="cuda", dtype=torch.bfloat16)
        w = torch.ones(5120, device="cuda", dtype=torch.bfloat16)
        for bad, weight in [
            (x.cpu(), w.cpu()),
            (x.float(), w),
            (x[:, ::2], w),
            (x, w[::2]),
            (x.view(1, 33, 5120), w),
            (x, w.clone().requires_grad_()),
        ]:
            self.assertFalse(candidate.is_supported(bad, weight))
            self.assertIsNone(candidate.rmsnorm_group32_quant(bad, weight))


if __name__ == "__main__":
    unittest.main(verbosity=2)
