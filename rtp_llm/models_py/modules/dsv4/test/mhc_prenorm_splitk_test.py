"""Numerical test for the split-K mHC pre-norm GEMM against the single-block one.

``_mhc_pre_norm_fn_fwd_mul`` computes, for ``x [T, H]`` bf16 and ``fn [mult3, H]``
fp32 rounded to tf32, both ``out[t, j] = sum_h x[t,h] * fn[j,h]`` and
``sqrsum[t] = sum_h x[t,h]^2``. Its grid is ``(ceildiv(T, 32), n_rms_group)`` and
every call site passes ``n_rms_group=1``, so at decode shapes it is one CUDA block
walking ``H / 256`` K-blocks serially -- 64 of them for DSV4-Flash, where
``H = hc_mult * dim`` is 16384.

``_mhc_pre_norm_fn_fwd_mul_splitk`` spreads those K-blocks over the grid and writes
per-split partials that ``_mhc_pre_big_fuse`` (and ``_mhc_pre_norm_fn_fwd_norm``)
already sum. That reorders the summation, so this is deliberately *not* a
bit-equality test. What it asserts instead:

  * summing the split partials reproduces the single-block result to tf32 rounding;
  * accuracy does not degrade -- against an fp64 reference the split-K error is no
    worse than the single-block one, and in practice better, because each
    accumulation chain is ``n_splits`` times shorter;
  * ``n_splits = 1`` is exactly equivalent, which is the case prefill takes
    (``resolve_n_splits`` returns 1 once the token grid already fills the GPU);
  * the split count must divide the K-block count, which is what
    ``largest_divisor_le`` guarantees;
  * the store mask follows ``mhc_mult3`` rather than the DSV4-Flash value 24, which
    only a shape with a different ``hc_mult`` can show.

Every case seeds its own generator from its parameters, so a single failing
subTest reproduces on its own rather than depending on how many cases ran first.

Needs a CUDA GPU and the vendored tilelang. Absent CUDA it skips; with CUDA present
but the kernels unimportable it fails, because that is a broken build rather than a
missing capability.
"""

import importlib
import os
import sys
import unittest

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# DSV4-Flash: hc_mult=4, dim=4096 -> mhc_mult3 = 4*2 + 4*4 = 24, H = 16384.
_MULT3 = 24
_HC_MULT = 4
_DIM = 4096
_H = _HC_MULT * _DIM  # 16384, so 64 K-blocks of 256
_K_BLOCKS = _H // 256

# A second shape whose mhc_mult3 is not 24, which is what pins the store mask to
# the parameter instead of the literal it used to be.
_ALT_HC_MULT = 2
_ALT_MULT3 = _ALT_HC_MULT * 2 + _ALT_HC_MULT * _ALT_HC_MULT  # 8
_ALT_H = _ALT_HC_MULT * _DIM  # 8192, 32 K-blocks


def _kernels():
    """Import the vendored kernels, after the module that preps tilelang's env."""
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401

    nfk = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.mhc.norm_fn_kernel"
    )
    ops = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
    )
    return nfk, ops


_CUDA = torch.cuda.is_available()
_IMPORT_ERROR = None
if _CUDA:
    try:
        _kernels()
    except Exception as exc:  # noqa: BLE001 - re-raised as a failure below
        _IMPORT_ERROR = exc


class MhcPrenormSplitKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _CUDA:
            raise unittest.SkipTest("needs a CUDA GPU")
        if _IMPORT_ERROR is not None:
            # Deliberately not a skip: tilelang missing on a GPU box, or either
            # kernel module failing to import, is the failure this suite exists to
            # catch. Folding it into a skip is how a broken split-K kernel would
            # come back green.
            raise AssertionError(
                "CUDA is present but the mHC kernels could not be imported, "
                f"which is a build problem: {_IMPORT_ERROR!r}"
            )
        cls.nfk, cls.ops = _kernels()
        cls.device = "cuda:0"

    def _fn(self, mult3: int, h: int, seed: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        fn = (torch.randn(mult3, h, generator=g) * 0.05).to(self.device)
        return self.nfk.round_to_tf32(fn.contiguous())

    def _inputs(self, tokens: int, h: int, seed: int) -> torch.Tensor:
        """Explicitly seeded so each subTest reproduces standalone."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        x = (torch.randn(tokens, h, generator=g) * 0.5).to(self.device)
        return x.to(torch.bfloat16)

    def _single(self, x, fn, mult3: int, h: int):
        t = x.shape[0]
        out = torch.empty(t, 1, mult3, dtype=torch.float32, device=self.device)
        sq = torch.empty(t, 1, dtype=torch.float32, device=self.device)
        self.nfk._mhc_pre_norm_fn_fwd_mul(mult3, 1, h)(x, fn, out, sq)
        return out[:, 0], sq[:, 0]

    def _splitk(self, x, fn, mult3: int, h: int, n_splits: int):
        t = x.shape[0]
        out = torch.empty(
            n_splits, t, 1, mult3, dtype=torch.float32, device=self.device
        )
        sq = torch.empty(n_splits, t, 1, dtype=torch.float32, device=self.device)
        self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(mult3, 1, h, n_splits)(
            x, fn, out, sq
        )
        # This is the reduction _mhc_pre_big_fuse performs over the split axis.
        return out[:, :, 0].sum(0), sq[:, :, 0].sum(0)

    @staticmethod
    def _reference(x, fn):
        xd = x.double()
        return xd @ fn.double().T, (xd * xd).sum(-1)

    def test_matches_single_block_and_is_no_less_accurate(self):
        fn = self._fn(_MULT3, _H, seed=20260815)
        for tokens in (2, 8, 16):
            for n_splits in (4, 16, 64):
                with self.subTest(tokens=tokens, n_splits=n_splits):
                    x = self._inputs(
                        tokens, _H, seed=1000 + tokens * 31 + n_splits
                    )
                    o_s, q_s = self._single(x, fn, _MULT3, _H)
                    o_k, q_k = self._splitk(x, fn, _MULT3, _H, n_splits)
                    o_ref, q_ref = self._reference(x, fn)

                    # Agreement between the two implementations, tf32 scale.
                    rel = ((o_k - o_s).norm() / o_s.norm().clamp(min=1e-9)).item()
                    self.assertLess(rel, 5e-3, f"rel_fro vs single-block = {rel}")
                    self.assertTrue(torch.allclose(q_k, q_s, rtol=1e-5, atol=1e-5))

                    # Accuracy against fp64: split-K must not be worse.
                    err_s = ((o_s.double() - o_ref).norm() / o_ref.norm()).item()
                    err_k = ((o_k.double() - o_ref).norm() / o_ref.norm()).item()
                    self.assertLessEqual(
                        err_k,
                        err_s * 1.05,
                        f"split-K error {err_k} exceeds single-block {err_s}",
                    )

    def test_one_split_is_the_degenerate_case(self):
        """What prefill takes: the grid already fills the GPU, so n_splits == 1."""
        fn = self._fn(_MULT3, _H, seed=4242)
        x = self._inputs(8, _H, seed=4243)
        o_s, q_s = self._single(x, fn, _MULT3, _H)
        o_k, q_k = self._splitk(x, fn, _MULT3, _H, 1)
        self.assertTrue(torch.allclose(o_k, o_s, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(q_k, q_s, rtol=1e-6, atol=1e-6))

    def test_store_mask_follows_mhc_mult3(self):
        """A shape whose mhc_mult3 is not DSV4-Flash's 24.

        Both forward stores used to mask with the literal 24, so this shape wrote
        24 of its 8 columns -- past the end of the output row. Only a non-24
        mhc_mult3 can distinguish the literal from the parameter.
        """
        self.assertNotEqual(_ALT_MULT3, _MULT3, "test setup: shapes must differ")
        fn = self._fn(_ALT_MULT3, _ALT_H, seed=777)
        x = self._inputs(8, _ALT_H, seed=778)
        o_s, q_s = self._single(x, fn, _ALT_MULT3, _ALT_H)
        o_k, q_k = self._splitk(x, fn, _ALT_MULT3, _ALT_H, 8)
        o_ref, _ = self._reference(x, fn)

        self.assertEqual(o_s.shape, (8, _ALT_MULT3))
        rel_s = ((o_s.double() - o_ref).norm() / o_ref.norm()).item()
        rel_k = ((o_k.double() - o_ref).norm() / o_ref.norm()).item()
        # A store that masked at 24 would leave columns unwritten (or write past
        # the row), so this is really a "did every column get a real value" check.
        self.assertLess(rel_s, 1e-2, f"single-block rel_fro = {rel_s}")
        self.assertLess(rel_k, 1e-2, f"split-K rel_fro = {rel_k}")
        self.assertTrue(torch.allclose(q_k, q_s, rtol=1e-5, atol=1e-5))

    def test_split_count_must_divide_the_k_blocks(self):
        """``largest_divisor_le`` is what keeps the caller from violating this."""
        with self.assertRaisesRegex(AssertionError, r"must divide the 64 K blocks"):
            self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(_MULT3, 1, _H, 9)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 9), 8)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 78), _K_BLOCKS)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 1), 1)

    def test_mhc_mult3_bound_is_checked(self):
        """The 32-wide store fragment is the limit, and 0 is not a shape."""
        with self.assertRaisesRegex(AssertionError, "mhc_mult3"):
            self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(33, 1, _H, 1)
        with self.assertRaisesRegex(AssertionError, "mhc_mult3"):
            self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(0, 1, _H, 1)


if __name__ == "__main__":
    unittest.main()
