"""Precision of flashinfer ops whose kernels differ across flashinfer versions.

Covers rmsnorm, fused_add_rmsnorm, and silu_and_mul. Outputs are compared
with an fp32 torch reference. A result may sit on either side of a bf16/fp16
midpoint, so the allowed error is one unit in the last place of the stored dtype.
"""

import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules import FusedSiluAndMul, RMSNorm, RMSResNorm

_EPS = 1e-6


def _fp32_rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    values = hidden_states.float()
    variance = (values * values).mean(dim=-1, keepdim=True)
    return values * torch.rsqrt(variance + _EPS) * weight.float()


def _fp32_silu_and_mul(gate_up: torch.Tensor) -> torch.Tensor:
    half = gate_up.shape[-1] // 2
    gate = gate_up[..., :half].float()
    up = gate_up[..., half:].float()
    return torch.nn.functional.silu(gate) * up


def _assert_within_one_ulp(
    test: TestCase, actual: torch.Tensor, reference: torch.Tensor
) -> None:
    stored = reference.to(actual.dtype)
    upward = torch.full_like(stored, float("inf"))
    downward = torch.full_like(stored, float("-inf"))
    ulp = torch.maximum(
        (torch.nextafter(stored, upward) - stored).abs(),
        (stored - torch.nextafter(stored, downward)).abs(),
    ).float()
    error = (actual.float() - reference.float()).abs()
    mismatch = error > ulp
    if bool(mismatch.any()):
        test.fail(
            "{0} values exceed 1 ULP, max abs {1:.3e}".format(
                int(mismatch.sum().item()),
                float(error.max().item()),
            )
        )


class FlashInferNormSiluTest(TestCase):
    DTYPES = [torch.float16, torch.bfloat16]
    NUM_TOKENS = [1, 16, 128]
    HIDDEN_SIZES = [128, 896]
    SILU_HIDDEN_SIZES = [256, 4864]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def test_rmsnorm(self) -> None:
        for num_tokens, hidden_size, dtype in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype
            ):
                self._check_rmsnorm(num_tokens, hidden_size, dtype)

    def test_rms_res_norm(self) -> None:
        for num_tokens, hidden_size, dtype in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype
            ):
                self._check_rms_res_norm(num_tokens, hidden_size, dtype)

    def test_silu_and_mul(self) -> None:
        for num_tokens, hidden_size, dtype in itertools.product(
            self.NUM_TOKENS, self.SILU_HIDDEN_SIZES, self.DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype
            ):
                self._check_silu_and_mul(num_tokens, hidden_size, dtype)

    def _check_rmsnorm(self, num_tokens: int, hidden_size: int, dtype: _dtype) -> None:
        torch.manual_seed(0)
        weight = torch.randn(hidden_size, dtype=dtype)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
        actual = RMSNorm(weight, eps=_EPS)(hidden_states)
        _assert_within_one_ulp(self, actual, _fp32_rmsnorm(hidden_states, weight))

    def _check_rms_res_norm(
        self, num_tokens: int, hidden_size: int, dtype: _dtype
    ) -> None:
        torch.manual_seed(0)
        weight = torch.randn(hidden_size, dtype=dtype)
        hidden_states = torch.randn(num_tokens, hidden_size, dtype=dtype)
        residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
        actual, actual_residual = RMSResNorm(weight, eps=_EPS)(
            hidden_states.clone(), residual.clone()
        )
        reference_residual = hidden_states.float() + residual.float()
        _assert_within_one_ulp(self, actual_residual, reference_residual)
        _assert_within_one_ulp(
            self, actual, _fp32_rmsnorm(reference_residual, weight)
        )

    def _check_silu_and_mul(
        self, num_tokens: int, hidden_size: int, dtype: _dtype
    ) -> None:
        torch.manual_seed(0)
        gate_up = torch.randn(num_tokens, hidden_size * 2, dtype=dtype)
        actual = FusedSiluAndMul()(gate_up)
        _assert_within_one_ulp(self, actual, _fp32_silu_and_mul(gate_up))


if __name__ == "__main__":
    main()
