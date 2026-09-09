import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import LayerNorm, LayerNormTorch

# Tolerance is derived from the output dtype, not picked by hand.
#
# Both sides of this comparison reduce in fp32: LayerNormTorch upcasts to
# float32, and aiter's ck_tile layernorm2d sets ComputeDataType = float for
# fp16 and bf16 alike (see, in the aiter wheel,
# aiter_meta/3rdparty/composable_kernel/example/ck_tile/02_layernorm2d/
# layernorm2d_fwd.hpp).  Over the widths exercised here (n <= 8199) an fp32
# reduction contributes at most ~sqrt(n) * 2**-24 ~= 1e-5 relative, so it is
# *not* what sets the bound -- and the bound must therefore not be scaled by
# the reduction width.  Measured on MI308X the worst deviation is exactly one
# output-dtype ulp and is flat in n: max(diff/tol) is 0.06 at n=768 and 0.08 at
# n=8192 for fp16.
#
# What does set the bound is the single rounding of the fp32 result into the
# output dtype, i.e. half an ulp, so the ulp of the output dtype is the natural
# unit.  Measured worst diff/tol over the full 60-case matrix on MI308X:
#
#     bound              fp16           bf16
#     flat 1e-2 (old)    0.084 (12x)    0.709 (1.4x)
#     8 ulp     (new)    0.216 (4.6x)   0.227 (4.4x)
#
# The flat value was simultaneously 12x looser than needed on fp16 and nearly
# exhausted on bf16, i.e. it was never a bound on anything -- it just happened
# to sit where it sat.  Eight output-dtype ulps is 2.6x tighter than 1e-2 for
# fp16 and 3.1x looser for bf16, and leaves both dtypes the same ~4.5x margin.
#
# Detection power, measured by injecting known-wrong results:
#   - mean dropped on one row:            new rejects by >11x, old by 4.3x
#   - 1e-3 relative error in inv-stddev:  new rejects at n=5120 (1.01) but not
#                                         at n=8192 (0.87); old rejects neither
#                                         (0.34-0.40), so it could not have
#                                         caught that defect at all
# A 4-ulp budget rejects strictly more (inv-stddev 1e-3 everywhere, 1.5-2.0x,
# and a reduction count off by one 8-element vector, 1.2-1.6x) but halves the
# margin to ~2.2x.  8 was chosen over 4 because this test's historical failures
# are still unexplained and margin is worth more here than the extra 2x of
# sensitivity; revisit if a real kernel regression ever slips through.
_OUTPUT_ULP = {
    torch.float16: 2.0**-11,
    torch.bfloat16: 2.0**-8,
}
_ULP_BUDGET = 8


def _tolerance(dtype: _dtype) -> tuple:
    """(atol, rtol) worth _ULP_BUDGET roundings of the output dtype."""
    ulp = _OUTPUT_ULP[dtype]
    tol = _ULP_BUDGET * ulp
    # rtol covers elements whose magnitude is O(1) or larger; atol covers those
    # near zero, where the surviving error is the fp32 rounding of a cancelling
    # sum rather than a relative one.  Both are the same multiple of the ulp.
    return tol, tol


class _MismatchReport:
    """Lazily-rendered diagnostics for a failed allclose.

    unittest only calls str() on the assertion message when the assertion has
    already failed, so nothing here runs on the passing path.
    """

    def __init__(self, expected, actual, layernorm, x, num_tokens, hidden_size, dtype):
        self._expected = expected
        self._actual = actual
        self._layernorm = layernorm
        self._x = x
        self._shape = (num_tokens, hidden_size)
        self._dtype = dtype

    def __str__(self) -> str:
        try:
            atol, rtol = _tolerance(self._dtype)
            exp = self._expected.to(torch.float32)
            act = self._actual.to(torch.float32)
            diff = (exp - act).abs()
            tol = atol + rtol * exp.abs()
            bad = diff > tol
            n_bad = int(bad.sum().item())
            worst = int(diff.argmax().item())
            row, col = divmod(worst, self._shape[1])
            rel = (diff / exp.abs().clamp_min(1e-6)).max().item()
            # How far past the bound the worst element is.  Just over 1.0 means
            # a marginal case; orders of magnitude means real corruption.
            ratio = float((diff / tol).max().item())

            # Re-run the kernel on the identical input: if these differ, the
            # kernel itself is non-deterministic rather than merely inaccurate.
            repeats = []
            for _ in range(2):
                again = self._layernorm(self._x)
                repeats.append(bool(torch.equal(again, self._actual)))

            return (
                f"layernorm mismatch for num_tokens={self._shape[0]}, "
                f"hidden_size={self._shape[1]}, dtype={self._dtype}: "
                f"atol=rtol={atol:.6g} ({_ULP_BUDGET} output-dtype ulp), "
                f"max_abs_diff={diff.max().item():.6g}, max_rel_diff={rel:.6g}, "
                f"worst_diff_over_tol={ratio:.4g}, "
                f"violating_elems={n_bad}/{exp.numel()} "
                f"({100.0 * n_bad / exp.numel():.4f}%), "
                f"worst@[{row},{col}] expected={exp.reshape(-1)[worst].item():.6g} "
                f"actual={act.reshape(-1)[worst].item():.6g}; "
                f"kernel_rerun_bitwise_identical={repeats}"
            )
        except Exception as diag_err:  # never mask the real assertion failure
            return f"<mismatch report unavailable: {diag_err!r}>"


class LayerNormTest(TestCase):
    DTYPES = [torch.half, torch.bfloat16]
    NUM_TOKENS = [7, 83, 4096]
    HIDDEN_SIZES = [768, 769, 770, 771, 5120, 5124, 5125, 5126, 8192, 8199]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def _run_layernorm_test(self, num_tokens: int, hidden_size: int, dtype: _dtype):
        torch.manual_seed(0)
        w = torch.randn(hidden_size, dtype=dtype)
        beta = torch.randn(hidden_size, dtype=dtype)
        layernorm = LayerNorm(w, beta)
        layernorm_torch = LayerNormTorch(w, beta)
        x = torch.randn(num_tokens, hidden_size, dtype=dtype)
        expected = layernorm_torch(x)
        actual = layernorm(x)
        atol, rtol = _tolerance(dtype)
        # NOTE: allclose stays the sole pass/fail predicate; the report below is
        # only stringified by unittest when the assertion already failed.
        self.assertTrue(
            torch.allclose(expected, actual, atol=atol, rtol=rtol),
            _MismatchReport(
                expected, actual, layernorm, x, num_tokens, hidden_size, dtype
            ),
        )

    def test_layernorm(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.DTYPES,
        ):
            with self.subTest(
                num_tokens=params[0], hidden_size=params[1], dtype=params[2]
            ):
                self._run_layernorm_test(*params)


if __name__ == "__main__":
    main()
