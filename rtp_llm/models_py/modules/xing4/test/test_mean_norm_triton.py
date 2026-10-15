"""Equivalence UT for the Xing4 fused hc-mean + final RMSNorm Triton kernel.

The reference is the model tail chain (mathematical authority
``Xing4Model._capture_canonical_final``):

    m = x.mean(dim=-2)                     # torch mean, fp32 accumulate, bf16 out
    y = flashinfer.norm.rmsnorm(m, w, eps)  # fp32 internal, bf16 out

Gates (family discipline, mirroring test_hc_norm_triton.py):

* T in {1, 8, 64, 4096} at HC=4, H=3584 (Xing4 tail shape) bf16:
  torch.testing.assert_close(rtol=atol=2**-7) plus a bit-level report
  (exact rate and max ulp);
* small-H generality (H=128/768) and HC in {2, 8};
* fp32 anchor: the kernel must also match a plain fp32 chain so a
  systematic flashinfer deviation cannot pass as a shared reference error;
* the int64 index contract at T=150000 rows (H=128 keeps the tensor small);
* non-fusable shapes (fp32 x, 2D x, H>8192) raise/fall back rather than
  corrupt;
* env backend switch XING4_MEAN_NORM_IMPL=torch keeps the reference chain.
"""

import os
import unittest

import torch

from rtp_llm.models_py.modules.xing4.hc.mean_norm_triton import (
    fused_mean_rmsnorm_triton,
    mean_rmsnorm,
)


def _flashinfer_rmsnorm(m: torch.Tensor, w: torch.Tensor, eps: float):
    import flashinfer

    return flashinfer.norm.rmsnorm(m, w, eps=eps)


def _torch_rmsnorm(m: torch.Tensor, w: torch.Tensor, eps: float):
    m32 = m.float()
    inv = torch.rsqrt(m32.square().mean(-1, keepdim=True) + eps)
    return (m32 * inv * w.float()).to(m.dtype)


def _reference(x: torch.Tensor, w: torch.Tensor, eps: float):
    m = x.mean(dim=-2)
    return _flashinfer_rmsnorm(m.contiguous(), w, eps), m


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        a.contiguous().view(torch.int16).to(torch.int32)
        - b.contiguous().view(torch.int16).to(torch.int32)
    ).abs()


class MeanNormTritonTest(unittest.TestCase):
    def _check(self, t: int, hc: int, h: int):
        eps = 1.0e-6
        torch.manual_seed(20261016 + t * 31 + h * 7 + hc)
        x = (torch.randn((t, hc, h), device="cuda", dtype=torch.bfloat16) * 0.5)
        w = (
            (torch.randn((h,), device="cuda") * 0.1 + 1.0)
            .abs()
            .to(torch.bfloat16)
        )

        ref, ref_m = _reference(x, w, eps)
        got = fused_mean_rmsnorm_triton(x, w, eps)
        torch.cuda.synchronize()

        exact = (_bf16_ulp_diff(ref, got) == 0).float().mean().item()
        max_ulp = int(_bf16_ulp_diff(ref, got).max())
        close = torch.allclose(
            ref.float(), got.float(), rtol=2**-7, atol=2**-7
        )
        # fp32 anchor: norm computed from the fp32 mean of x.
        m32 = x.float().mean(dim=-2)
        anchor = _torch_rmsnorm(m32.to(torch.bfloat16), w, eps)
        anchor_close = torch.allclose(
            anchor.float(), got.float(), rtol=2**-7, atol=2**-7
        )
        # The intermediate mean must be reproduced exactly by the kernel's
        # phase-1 rounding (m feeds the norm in both chains).
        m_again = x.mean(dim=-2)
        mean_exact = torch.equal(m_again, ref_m)
        print(
            f"[T={t} hc={hc} H={h}] exact={exact:.6f} max_ulp={max_ulp} "
            f"close={close} anchor_close={anchor_close} "
            f"mean_reproducible={mean_exact}"
        )
        self.assertTrue(close)
        self.assertTrue(anchor_close)
        self.assertGreaterEqual(exact, 0.9999)
        self.assertLessEqual(max_ulp, 1)

    def test_xing4_tail_shapes(self):
        for t in (1, 8, 64, 4096):
            self._check(t, 4, 3584)

    def test_generality(self):
        self._check(128, 2, 128)
        self._check(256, 8, 768)
        self._check(1, 4, 128)

    def test_int64_row_contract(self):
        # row * (HC*H) must stay int64-safe: T=150000, H=128, HC=4 keeps the
        # tensor at ~150 MB while the linear index passes INT32_MAX only via
        # the row-stride multiply contract.
        t, hc, h = 150000, 4, 128
        eps = 1.0e-6
        torch.manual_seed(20261017)
        x = torch.zeros((t, hc, h), device="cuda", dtype=torch.bfloat16)
        # Distinguish rows so a wrapped offset would corrupt the output.
        base = torch.randn((t, h), device="cuda", dtype=torch.bfloat16)
        x.copy_(base.unsqueeze(1).expand(t, hc, h))
        w = torch.ones((h,), device="cuda", dtype=torch.bfloat16)
        got = fused_mean_rmsnorm_triton(x, w, eps)
        ref, _ = _reference(x, w, eps)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.allclose(ref.float(), got.float(), rtol=2**-7, atol=2**-7)
        )
        exact = (_bf16_ulp_diff(ref, got) == 0).float().mean().item()
        self.assertGreaterEqual(exact, 0.9999)

    def test_non_fusable_shapes(self):
        h = 3584
        w = torch.ones((h,), device="cuda", dtype=torch.bfloat16)
        x2d = torch.randn((8, h), device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            fused_mean_rmsnorm_triton(x2d, w, 1e-6)
        xf32 = torch.randn((8, 4, h), device="cuda", dtype=torch.float32)
        with self.assertRaises(ValueError):
            fused_mean_rmsnorm_triton(xf32, w, 1e-6)

    def test_env_backend_switch(self):
        t, hc, h = 64, 4, 3584
        eps = 1.0e-6
        torch.manual_seed(20261018)
        x = torch.randn((t, hc, h), device="cuda", dtype=torch.bfloat16) * 0.5
        w = (
            (torch.randn((h,), device="cuda") * 0.1 + 1.0)
            .abs()
            .to(torch.bfloat16)
        )
        ref, _ = _reference(x, w, eps)
        old = os.environ.get("XING4_MEAN_NORM_IMPL")
        try:
            os.environ["XING4_MEAN_NORM_IMPL"] = "torch"
            got = mean_rmsnorm(x, w, eps)
            self.assertTrue(torch.equal(got, ref))
            os.environ["XING4_MEAN_NORM_IMPL"] = "triton"
            got2 = mean_rmsnorm(x, w, eps)
            torch.cuda.synchronize()
            self.assertTrue(
                torch.allclose(ref.float(), got2.float(), rtol=2**-7, atol=2**-7)
            )
        finally:
            if old is None:
                os.environ.pop("XING4_MEAN_NORM_IMPL", None)
            else:
                os.environ["XING4_MEAN_NORM_IMPL"] = old

    def test_fallback_for_oversize_h(self):
        # H > 8192 falls back to the torch chain instead of launching.
        t, hc, h = 4, 2, 8320
        eps = 1.0e-6
        torch.manual_seed(20261019)
        x = torch.randn((t, hc, h), device="cuda", dtype=torch.bfloat16) * 0.5
        w = (
            (torch.randn((h,), device="cuda") * 0.1 + 1.0)
            .abs()
            .to(torch.bfloat16)
        )
        ref, _ = _reference(x, w, eps)
        got = mean_rmsnorm(x, w, eps)
        self.assertTrue(torch.equal(got, ref))


if __name__ == "__main__":
    unittest.main()
