"""UT for the fused BF16 SiLU + (optional clamp) + multiply Triton kernel
used by the V4.1 MXFP8 ``W13SharedExpert`` fused mid-chain.

The kernel must be bit-identical to the reference eager chain it replaces::

    gate_up = w13(x).float()               # bf16 -> fp32 (exact)
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu_mul_split(               # fp32 math
        gate.contiguous(), up.contiguous(), clamp_limit=L)
    hidden.to(torch.bfloat16)              # single fp32 -> bf16 rounding

Tests cover:
  - V4.1 shared-expert shape (D = 2304, clamp via swiglu_limit = 10)
  - no-clamp path (swiglu_limit = 0)
  - small-D / non-pow2 D edge cases (BLOCK_N padding)
  - single-row and large-batch shapes
  - empty input
  - clamp arm actually firing
  - bit-exact equality (torch.equal) against the reference chain

Bypasses rtp_llm package init via importlib.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch


def _load_kernel():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "moe", "_silu_mul_bf16_triton.py"))
    spec = importlib.util.spec_from_file_location("_v4_silu_mul_bf16", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.silu_mul_split_bf16


def _load_split_kernel():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_silu_mul_split_triton.py"))
    spec = importlib.util.spec_from_file_location("_v4_silu_mul_split", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.silu_mul_split


def _ref_chain(gate_up_bf16: torch.Tensor, clamp_limit: float):
    """Reference chain matching the pre-fusion W13SharedExpert.forward."""
    silu_mul_split = _load_split_kernel()
    gate_up = gate_up_bf16.float()
    gate, up = gate_up.chunk(2, dim=-1)
    hidden = silu_mul_split(gate.contiguous(), up.contiguous(), clamp_limit=clamp_limit)
    return hidden.to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SiluMulSplitBf16EquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _load_kernel()
        except Exception as e:
            raise unittest.SkipTest(f"silu_mul_split_bf16 not importable: {e}")

    def _check(self, *, M, D, clamp_limit):
        torch.manual_seed(0)
        device = "cuda:0"
        gate_up = (torch.randn(M, 2 * D, device=device, dtype=torch.float32) * 3.0).to(
            torch.bfloat16
        )
        # Make sure some values exceed clamp_limit so the clamp branch fires.
        if clamp_limit > 0 and M > 0:
            gate_up[0, :10] = clamp_limit + 1.0
            gate_up[0, D : D + 10] = clamp_limit + 1.0
            if M > 1:
                gate_up[1, D : D + 10] = -(clamp_limit + 1.0)

        ref = _ref_chain(gate_up, clamp_limit)
        out = _load_kernel()(gate_up, clamp_limit=clamp_limit)

        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(out, ref),
            f"bit mismatch (M={M},D={D},L={clamp_limit}): "
            f"max abs diff {(out.float() - ref.float()).abs().max().item():.3e}, "
            f"mismatches {(out != ref).sum().item()}",
        )

    def test_shared_expert_v41_shape(self):
        # V4.1: moe_intermediate_size=2304, swiglu_limit=10.0.
        self._check(M=128, D=2304, clamp_limit=10.0)

    def test_no_clamp(self):
        self._check(M=64, D=2304, clamp_limit=0.0)

    def test_small_D(self):
        # D < default BLOCK_N=1024.
        self._check(M=16, D=384, clamp_limit=0.0)
        self._check(M=16, D=384, clamp_limit=4.0)

    def test_D_not_pow2(self):
        # D=1500 — BLOCK_N rounds up; mask discards padding.
        self._check(M=8, D=1500, clamp_limit=3.0)

    def test_single_row(self):
        self._check(M=1, D=2304, clamp_limit=10.0)
        self._check(M=1, D=2304, clamp_limit=0.0)

    def test_prefill_batch(self):
        # 16K CP4: 4096 rank-local tokens.
        self._check(M=4096, D=2304, clamp_limit=10.0)

    def test_empty_M(self):
        device = "cuda:0"
        gate_up = torch.empty(0, 4608, device=device, dtype=torch.bfloat16)
        out = _load_kernel()(gate_up, clamp_limit=10.0)
        self.assertEqual(out.shape, (0, 2304))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_out_buffer(self):
        torch.manual_seed(1)
        device = "cuda:0"
        gate_up = (torch.randn(32, 512, device=device, dtype=torch.float32)).to(
            torch.bfloat16
        )
        ref = _ref_chain(gate_up, 0.0)
        out = torch.empty(32, 256, device=device, dtype=torch.bfloat16)
        ret = _load_kernel()(gate_up, clamp_limit=0.0, out=out)
        self.assertIs(ret, out)
        self.assertTrue(torch.equal(out, ref))

    def test_clamp_actually_applied(self):
        torch.manual_seed(7)
        device = "cuda:0"
        gate_up = torch.full((4, 128), 10.0, device=device, dtype=torch.bfloat16)
        kernel = _load_kernel()
        no_clamp = kernel(gate_up, clamp_limit=0.0)
        clamped = kernel(gate_up, clamp_limit=2.0)
        self.assertFalse(torch.equal(no_clamp, clamped))
        self.assertTrue(torch.equal(clamped, _ref_chain(gate_up, 2.0)))


if __name__ == "__main__":
    unittest.main()
