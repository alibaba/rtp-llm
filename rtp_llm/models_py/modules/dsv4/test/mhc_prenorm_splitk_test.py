"""Numerical test for the split-K mHC pre-norm GEMM against the single-block one.

`_mhc_pre_norm_fn_fwd_mul` computes, for `x [T, H]` bf16 and `fn [mult3, H]` fp32
rounded to tf32, both `out[t, j] = sum_h x[t,h] * fn[j,h]` and
`sqrsum[t] = sum_h x[t,h]^2`. Its grid is `(ceildiv(T, 32), n_rms_group)` and every
call site passes `n_rms_group=1`, so at decode shapes it is one CUDA block walking
`H / 256` K-blocks serially -- 64 of them for DSV4-Flash, where `H = hc_mult * dim`
is 16384.

`_mhc_pre_norm_fn_fwd_mul_splitk` spreads those K-blocks over the grid and writes
per-split partials that `_mhc_pre_big_fuse` (and `_mhc_pre_norm_fn_fwd_norm`) already
sum. That reorders the summation, so this is deliberately *not* a bit-equality test.
What it asserts instead:

  * summing the split partials reproduces the single-block result to tf32 rounding;
  * accuracy does not degrade -- against an fp64 reference the split-K error is no
    worse than the single-block one, and in practice better, because each
    accumulation chain is `n_splits` times shorter;
  * `n_splits = 1` is exactly equivalent, which is the case prefill takes
    (`_compute_num_split` returns 1 once the token grid already fills the GPU);
  * the split count must divide the K-block count, which is what
    `_largest_divisor_le` in `ops/pre_big_fuse.py` guarantees.

Needs a CUDA GPU and the vendored tilelang; skips otherwise. Run with the conda env
the engine uses, e.g.
``CUDA_VISIBLE_DEVICES=7 /opt/conda310/bin/python3 -m unittest <this module>``.
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

_MULT3 = 24
_HC_MULT = 4
_DIM = 4096
_H = _HC_MULT * _DIM  # 16384, so 64 K-blocks of 256
_K_BLOCKS = _H // 256


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


def _cuda_ok() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        _kernels()
        return True
    except Exception:
        return False


@unittest.skipUnless(_cuda_ok(), "needs CUDA and the vendored tilelang kernels")
class MhcPrenormSplitKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nfk, cls.ops = _kernels()
        cls.device = "cuda:0"
        g = torch.Generator(device="cpu").manual_seed(20260815)
        fn = (torch.randn(_MULT3, _H, generator=g) * 0.05).to(cls.device)
        cls.fn = cls.nfk.round_to_tf32(fn.contiguous())
        cls.g = g

    def _inputs(self, tokens: int):
        x = (torch.randn(tokens, _H, generator=self.g) * 0.5).to(self.device)
        return x.to(torch.bfloat16)

    def _single(self, x):
        t = x.shape[0]
        out = torch.empty(t, 1, _MULT3, dtype=torch.float32, device=self.device)
        sq = torch.empty(t, 1, dtype=torch.float32, device=self.device)
        self.nfk._mhc_pre_norm_fn_fwd_mul(_MULT3, 1, _H)(x, self.fn, out, sq)
        return out[:, 0], sq[:, 0]

    def _splitk(self, x, n_splits: int):
        t = x.shape[0]
        out = torch.empty(
            n_splits, t, 1, _MULT3, dtype=torch.float32, device=self.device
        )
        sq = torch.empty(n_splits, t, 1, dtype=torch.float32, device=self.device)
        self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(_MULT3, 1, _H, n_splits)(
            x, self.fn, out, sq
        )
        # This is the reduction _mhc_pre_big_fuse performs over the split axis.
        return out[:, :, 0].sum(0), sq[:, :, 0].sum(0)

    def _reference(self, x):
        xd = x.double()
        return xd @ self.fn.double().T, (xd * xd).sum(-1)

    def test_matches_single_block_and_is_no_less_accurate(self):
        for tokens in (2, 8, 16):
            for n_splits in (4, 16, 64):
                with self.subTest(tokens=tokens, n_splits=n_splits):
                    x = self._inputs(tokens)
                    o_s, q_s = self._single(x)
                    o_k, q_k = self._splitk(x, n_splits)
                    o_ref, q_ref = self._reference(x)

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
        x = self._inputs(8)
        o_s, q_s = self._single(x)
        o_k, q_k = self._splitk(x, 1)
        self.assertTrue(torch.allclose(o_k, o_s, rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.allclose(q_k, q_s, rtol=1e-6, atol=1e-6))

    def test_split_count_must_divide_the_k_blocks(self):
        """`_largest_divisor_le` is what keeps the caller from violating this."""
        with self.assertRaises(AssertionError):
            self.nfk._mhc_pre_norm_fn_fwd_mul_splitk(_MULT3, 1, _H, 9)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 9), 8)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 78), _K_BLOCKS)
        self.assertEqual(self.ops._largest_divisor_le(_K_BLOCKS, 1), 1)


if __name__ == "__main__":
    unittest.main()
