import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import LayerNorm, LayerNormTorch

_ATOL = 1e-2
_RTOL = 1e-2


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
            exp = self._expected.to(torch.float32)
            act = self._actual.to(torch.float32)
            diff = (exp - act).abs()
            tol = _ATOL + _RTOL * exp.abs()
            bad = diff > tol
            n_bad = int(bad.sum().item())
            worst = int(diff.argmax().item())
            row, col = divmod(worst, self._shape[1])
            rel = (diff / exp.abs().clamp_min(1e-6)).max().item()

            # Re-run the kernel on the identical input: if these differ, the
            # kernel itself is non-deterministic rather than merely inaccurate.
            repeats = []
            for _ in range(2):
                again = self._layernorm(self._x)
                repeats.append(bool(torch.equal(again, self._actual)))

            return (
                f"layernorm mismatch for num_tokens={self._shape[0]}, "
                f"hidden_size={self._shape[1]}, dtype={self._dtype}: "
                f"max_abs_diff={diff.max().item():.6g}, max_rel_diff={rel:.6g}, "
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
        # NOTE: allclose stays the sole pass/fail predicate; the report below is
        # only stringified by unittest when the assertion already failed.
        self.assertTrue(
            torch.allclose(expected, actual, atol=_ATOL, rtol=_RTOL),
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
