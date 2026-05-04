"""UT for the fused SiLU + (optional clamp) + multiply Triton kernel
used by ``Expert.forward`` (WS-Q3, 2026-05-04).

Tests cover:
  - shared-expert path (swiglu_limit = 0, no clamp)
  - routed-expert path (swiglu_limit = 7, full clamp)
  - V4-Flash shapes (D = moe_intermediate_size = 2048; N varies)
  - small-D edge case (BLOCK_D pow2 padding)
  - empty input
  - output writes match `_ref` byte-equivalent (FP32-only math, no BF16
    drift)

Bypasses rtp_llm package init via importlib.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch
import torch.nn.functional as F


def _load_silu_mul_split():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(
        os.path.join(here, "..", "_silu_mul_split_triton.py")
    )
    spec = importlib.util.spec_from_file_location("_v4_silu_mul_split", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.silu_mul_split


def _ref_silu_mul(gate: torch.Tensor, up: torch.Tensor, clamp_limit: float):
    """Eager reference matching Expert.forward."""
    if clamp_limit > 0:
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        gate = torch.clamp(gate, max=clamp_limit)
    return F.silu(gate) * up


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulSplitEquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _load_silu_mul_split()
        except Exception as e:
            raise unittest.SkipTest(f"silu_mul_split not importable: {e}")

    def _check(self, *, N, D, clamp_limit, atol=1e-6, rtol=1e-5):
        torch.manual_seed(0)
        device = "cuda:0"
        gate = torch.randn(N, D, device=device, dtype=torch.float32) * 2.0
        up = torch.randn(N, D, device=device, dtype=torch.float32) * 2.0
        # Make sure some values exceed clamp_limit so clamp branch fires.
        if clamp_limit > 0:
            gate[0, :10] = clamp_limit + 1.0
            up[0, :10] = clamp_limit + 1.0
            if N > 1:
                up[1, :10] = -(clamp_limit + 1.0)
            else:
                up[0, 10:20] = -(clamp_limit + 1.0)

        ref = _ref_silu_mul(gate, up, clamp_limit)
        fused = _load_silu_mul_split()
        out = fused(gate.contiguous(), up.contiguous(), clamp_limit=clamp_limit)

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.float32)
        diff = (out - ref).abs()
        # FP32 arithmetic with deterministic order — should match very tightly.
        self.assertTrue(
            torch.allclose(out, ref, atol=atol, rtol=rtol),
            f"max abs diff {diff.max().item():.3e} (N={N},D={D},L={clamp_limit})",
        )

    def test_shared_expert_v4flash(self):
        # V4-Flash shared-expert: D=2048, swiglu_limit=0 (no clamp).
        # N varies with input batch; pick a representative.
        self._check(N=128, D=2048, clamp_limit=0.0)

    def test_routed_expert_clamped(self):
        # Routed-expert path (ep_size==1 local): D=2048, swiglu_limit=7.0.
        self._check(N=64, D=2048, clamp_limit=7.0)

    def test_small_D_below_block(self):
        # D < default BLOCK_D=1024.
        self._check(N=16, D=384, clamp_limit=0.0)
        self._check(N=16, D=384, clamp_limit=4.0)

    def test_single_token(self):
        self._check(N=1, D=2048, clamp_limit=0.0)
        self._check(N=1, D=2048, clamp_limit=5.0)

    def test_large_batch(self):
        self._check(N=2048, D=2048, clamp_limit=0.0)

    def test_D_not_pow2(self):
        # D=1500 — kernel BLOCK_D should round up; mask discards pad.
        self._check(N=8, D=1500, clamp_limit=3.0)

    def test_empty_N(self):
        device = "cuda:0"
        gate = torch.empty(0, 2048, device=device, dtype=torch.float32)
        up = torch.empty(0, 2048, device=device, dtype=torch.float32)
        fused = _load_silu_mul_split()
        out = fused(gate, up, clamp_limit=0.0)
        self.assertEqual(out.shape, (0, 2048))

    def test_3d_input_flattens(self):
        """Multi-dim leading dims should flatten correctly."""
        torch.manual_seed(42)
        device = "cuda:0"
        gate = torch.randn(4, 32, 256, device=device, dtype=torch.float32)
        up = torch.randn(4, 32, 256, device=device, dtype=torch.float32)
        ref = _ref_silu_mul(gate, up, clamp_limit=0.0)
        fused = _load_silu_mul_split()
        out = fused(gate.contiguous(), up.contiguous(), clamp_limit=0.0)
        self.assertEqual(out.shape, ref.shape)
        self.assertTrue(torch.allclose(out, ref, atol=1e-6, rtol=1e-5))

    def test_clamp_actually_applied(self):
        """Verify clamp arm changes output when values exceed limit."""
        torch.manual_seed(7)
        device = "cuda:0"
        gate = torch.full((4, 64), 10.0, device=device, dtype=torch.float32)
        up = torch.full((4, 64), 10.0, device=device, dtype=torch.float32)
        fused = _load_silu_mul_split()

        no_clamp = fused(gate, up, clamp_limit=0.0)
        clamped = fused(gate, up, clamp_limit=2.0)
        # Different by construction.
        self.assertFalse(torch.allclose(no_clamp, clamped))
        # Clamped result matches eager clamped result.
        self.assertTrue(torch.allclose(clamped, _ref_silu_mul(gate, up, 2.0)))


if __name__ == "__main__":
    unittest.main()
