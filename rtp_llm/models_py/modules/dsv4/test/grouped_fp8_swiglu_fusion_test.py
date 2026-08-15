"""Equivalence test for the fused SwiGLU + per-token-group FP8 quant kernel.

``_silu_mul_quant_fp32scale`` is on by default (``DSV4_MOE_FUSED_SWIGLU=1``) and
replaces this explicit sequence, spelled out in ``grouped_fp8.py``'s unfused
branch::

    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    sgl_per_token_group_quant_fp8(
        hidden.contiguous(), group_size=128, eps=1e-4,
        column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False)

The kernel is not bit-exact against it and is not supposed to be, so the assertions
state what *is* true, measured rather than assumed:

* the fused scale is exactly ``max(absmax_fp32, eps) / 448``, i.e. the definition;
* the reference's scale differs from that by at most one fp32 ULP, on roughly half
  the groups -- it is the reference that deviates from the fp32 definition here,
  presumably in how it reduces absmax, and this is *not* the "scales are
  bit-identical" relationship an earlier reading of a narrower sample suggested;
* the resulting fp8 bytes differ by at most one e4m3 ULP, on 0.03-0.15% of
  elements over the distributions below.

Nothing else in the suite pins this; the masked-vs-contiguous test deliberately
switches this kernel off so that the layout is its only variable.

The four axes that change the kernel's control flow are all covered: ``HAS_CLAMP``
(``swiglu_limit`` zero or not), ``E == 1`` versus ``E > 1`` (which selects the
``_ROWS_PER_E_FLAT`` constant and so the group-index arithmetic), and an ``M`` that
is not a multiple of ``BLOCK_M``, which exercises the row-tail mask.

Needs a CUDA GPU, Triton and the SGL quant kernel.  Runs on the H20 lane; a box
with CUDA where the imports fail is a build problem and fails rather than skips.
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_CUDA = torch.cuda.is_available()
_IMPORT_ERROR = None
try:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
    from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import (
        _ROWS_PER_E_FLAT,
        _silu_mul_quant_fp32scale,
    )
    from rtp_llm.models_py.modules.dsv4.quant_layouts import FP8_BLOCK
except Exception as exc:  # noqa: BLE001 - re-raised as a failure below
    _IMPORT_ERROR = exc


_FP8_MAX = 448.0


def _hidden_bf16(gate_up: torch.Tensor, inter: int, limit: float) -> torch.Tensor:
    """The bf16 SwiGLU output both paths quantise."""
    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    return (F.silu(gate.float()) * up.float()).to(torch.bfloat16)


def _reference(gate_up: torch.Tensor, inter: int, limit: float):
    """The unfused sequence, verbatim from ``grouped_fp8.py``'s else branch."""
    gate, up = gate_up[:, :inter], gate_up[:, inter:]
    if limit > 0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    hidden = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    return sgl_per_token_group_quant_fp8(
        hidden.contiguous(),
        group_size=FP8_BLOCK,
        eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=False,
    )


class GroupedFP8FusedSwigluEquivalenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("needs a CUDA GPU")
        if _IMPORT_ERROR is not None:
            raise AssertionError(
                "CUDA is present but the kernels under test could not be "
                f"imported, which is a build problem rather than a missing "
                f"capability: {_IMPORT_ERROR!r}"
            )
        cls.device = "cuda:0"

    def _gate_up(self, E: int, T: int, inter: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        # Wide enough to produce a range of per-group absmax values, and to put
        # values on both sides of the clamp when one is set.
        x = torch.randn(E, T, 2 * inter, generator=g) * 3.0
        return x.to(self.device).to(torch.bfloat16)

    def _check(self, *, E: int, T: int, inter: int, limit: float, seed: int):
        M = E * T
        gate_up3 = self._gate_up(E, T, inter, seed)
        q_fused, s_fused = _silu_mul_quant_fp32scale(gate_up3, inter, limit)

        flat = gate_up3.reshape(M, 2 * inter)
        q_ref, s_ref = _reference(flat, inter, limit)

        self.assertEqual(tuple(q_fused.shape), tuple(q_ref.shape))
        self.assertEqual(q_fused.dtype, q_ref.dtype)
        # scale_tma_aligned=True pads the reference's M up to a multiple of 4, so
        # only the first M rows are the comparable region. DeepGEMM reads M rows
        # either way, which is why the unpadded fused scale is a valid operand.
        self.assertGreaterEqual(s_ref.shape[0], M)
        s_ref = s_ref[:M]
        self.assertEqual(tuple(s_fused.shape), tuple(s_ref.shape))
        # Both scales are M-major: stride 1 down the token axis, which is the
        # column-major/TMA-aligned layout DeepGEMM wants for the LHS. The *group*
        # stride differs (the reference's is its padded M), so only the token
        # stride is the shared contract.
        self.assertEqual(s_fused.stride(0), 1)
        self.assertEqual(s_ref.stride(0), 1)

        # The fused kernel is exactly the fp32 definition.
        hidden = _hidden_bf16(flat, inter, limit)
        absmax = (
            hidden.float().abs().reshape(M, inter // FP8_BLOCK, FP8_BLOCK).amax(-1)
        )
        want = absmax.clamp(min=1e-4) / _FP8_MAX
        self.assertTrue(
            torch.equal(s_fused, want),
            "fused scale is not max(absmax, eps)/448: max |d| = "
            f"{(s_fused - want).abs().max().item()}",
        )

        # The reference deviates from that definition by at most one fp32 ULP.
        # Compared as integers because that is what "one ULP" means for floats of
        # the same sign, and both are positive by construction.
        ulp = (
            (s_fused.contiguous().view(torch.int32) - s_ref.contiguous().view(torch.int32))
            .abs()
            .max()
            .item()
        )
        self.assertLessEqual(
            ulp,
            1,
            f"scale differs from the reference by {ulp} fp32 ULP, not <= 1",
        )

        # Quantised bytes: at most one fp8 e4m3 ULP apart, on a small fraction.
        a = q_fused.view(torch.uint8).to(torch.int16)
        b = q_ref.view(torch.uint8).to(torch.int16)
        # e4m3 is monotone in its bit pattern within a sign, so for equal signs an
        # adjacent code point differs by exactly 1 in the raw byte.
        same_sign = (a >= 0x80) == (b >= 0x80)
        self.assertTrue(bool(same_sign.all()), "a byte changed sign")
        delta = (a - b).abs()
        self.assertLessEqual(
            int(delta.max()),
            1,
            f"byte delta {int(delta.max())} exceeds one ULP",
        )
        frac = float((delta > 0).float().mean())
        self.assertLess(frac, 1e-2, f"{frac:.5%} of bytes differ by one ULP")
        return frac

    def test_with_clamp(self):
        """``swiglu_limit > 0`` -> HAS_CLAMP=True, the production DSV4 path."""
        for E, T, inter in ((1, 96, 256), (4, 24, 256)):
            with self.subTest(E=E, T=T, inter=inter):
                self._check(E=E, T=T, inter=inter, limit=7.0, seed=20260816 + T)

    def test_without_clamp(self):
        """``HAS_CLAMP=False`` is a different compiled kernel."""
        for E, T, inter in ((1, 96, 256), (4, 24, 256)):
            with self.subTest(E=E, T=T, inter=inter):
                self._check(E=E, T=T, inter=inter, limit=0.0, seed=771 + T)

    def test_row_tail_mask(self):
        """``M`` not a multiple of BLOCK_M (64): the last block is partial."""
        for E, T in ((1, 65), (1, 127), (3, 7), (5, 13)):
            with self.subTest(E=E, T=T):
                self.assertNotEqual((E * T) % 64, 0, "test setup: M must be partial")
                self._check(E=E, T=T, inter=256, limit=7.0, seed=4242 + E * 100 + T)

    def test_flat_and_grouped_agree_on_the_same_rows(self):
        """``E == 1`` pins ROWS_PER_E to a constant; ``E > 1`` passes T.

        Both must produce the same result for the same rows, which is what makes
        the constant safe -- when E is 1 the group index is identically 0, so any
        constant >= M yields the same arithmetic.
        """
        E, T, inter = 4, 24, 256
        gate_up3 = self._gate_up(E, T, inter, seed=99)
        q_grouped, s_grouped = _silu_mul_quant_fp32scale(gate_up3, inter, 7.0)
        q_flat, s_flat = _silu_mul_quant_fp32scale(
            gate_up3.reshape(1, E * T, 2 * inter), inter, 7.0
        )
        self.assertTrue(torch.equal(q_grouped, q_flat))
        self.assertTrue(torch.equal(s_grouped, s_flat))
        self.assertGreater(_ROWS_PER_E_FLAT, E * T, "the flat constant must exceed M")

    def test_eps_floor_applies_to_an_all_zero_group(self):
        """``max(absmax, eps)`` is the only thing keeping the scale off zero."""
        E, T, inter = 1, 64, 256
        gate_up3 = torch.zeros(
            E, T, 2 * inter, dtype=torch.bfloat16, device=self.device
        )
        _, s_fused = _silu_mul_quant_fp32scale(gate_up3, inter, 0.0)
        want = torch.full_like(s_fused, 1e-4 / _FP8_MAX)
        self.assertTrue(torch.equal(s_fused, want), s_fused.flatten()[:4])


if __name__ == "__main__":
    unittest.main()
