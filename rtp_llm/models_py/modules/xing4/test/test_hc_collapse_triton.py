"""Equivalence UT for the Xing4 HC collapse Triton kernel (lane K3).

Math authority: ``Xing4HCUnit._pre_impl``'s stream collapse in
``rtp_llm/models_py/model_desc/xing4_0_model.py``::

    y = torch.sum(pre.to(dtype).unsqueeze(-1) * x, dim=-2)   # [T, hc, H] -> [T, H]

The reference below is that expression written out in torch. Structural
difference under test (documented in ``collapse_triton.py``): the eager
path lands each bf16*bf16 product in bf16 before the fp32 ``torch.sum``,
while the Triton kernel multiplies and accumulates in fp32 directly, so
each summed term differs by at most 1 bf16 ulp (err <= 2**-8 * |p|). The
mandated tolerance ``rtol=atol=2**-7`` covers it only when the summed
arbitrary-mantissa products stay bounded: at a fully-cancelled column the
per-term rounding is measured against a near-zero output, where no
relative tolerance can absorb it. The data below therefore keeps
sum(|p|) over arbitrary-mantissa terms <= 1.5 (typical pre <= 0.5,
negative pre <= 0.25, x clamped to +-2), which bounds the worst-case
structural error by 1.5 * 2**-8 < atol even at total cancellation, while
``pre`` rows still mix zeros, negatives, and large power-of-two
magnitudes (power-of-two pre makes those products bit-exact in bf16, so
large values probe magnitude handling without adding structural
rounding). Per-case max diff is printed for the report.
"""

import unittest

import torch

from rtp_llm.models_py.modules.xing4.hc.collapse_triton import hc_collapse

HC = 4
H = 3584
RTOL = 2**-7
ATOL = 2**-7
_HAS_CUDA = torch.cuda.is_available()
_DEVICE = "cuda" if _HAS_CUDA else "cpu"


def _reference_collapse(pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Hand-written torch reference of the Xing4HCUnit collapse."""
    pre_dt = pre.to(x.dtype)
    return torch.sum(pre_dt.unsqueeze(-1) * x, dim=-2)


def _randn_bf16(shape, seed: int) -> torch.Tensor:
    # |x| <= 2 keeps sum(|p|) over arbitrary-mantissa terms <= 1.5 (see the
    # module docstring): the per-term structural rounding (<= 2**-8 * |p|)
    # then stays under atol even when the streams fully cancel.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return (
        torch.randn(shape, generator=gen)
        .clamp(-2.0, 2.0)
        .to(torch.bfloat16)
        .to(_DEVICE)
    )


def _mixed_pre(T: int, seed: int) -> torch.Tensor:
    """[T, HC] bf16 mix of typical sigmoid-range, 0, negative, large values."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    pre = torch.rand((T, HC), generator=gen) * 0.45 + 0.05
    if T >= 1:
        pre[:, 0] = 0.0  # exact zeros
    if T >= 2:
        pre[:, 1] = -0.5 * pre[:, 1]  # negatives in (-0.25, -0.025)
    if T >= 4:
        # Powers of two: bf16 product is exact, probing large magnitudes
        # without adding structural rounding. One large entry per row.
        pre[0::4, 2] = 4.0
        pre[1::4, 3] = 32.0
        pre[2::4, 2] = -4.0
        pre[3::4, 3] = -32.0
    return pre.to(torch.bfloat16).to(_DEVICE)


def _report_max_diff(y: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    diff = (y.float() - ref.float()).abs()
    if diff.numel() == 0:
        print(f"[hc_collapse] {label}: empty output")
        return
    max_abs = diff.max().item()
    rel = diff / ref.float().abs().clamp_min(1e-6)
    max_rel = rel.max().item()
    print(
        f"[hc_collapse] {label}: max_abs_diff={max_abs:.3e} "
        f"max_rel_diff={max_rel:.3e}"
    )


def _assert_equivalent(y: torch.Tensor, ref: torch.Tensor, label: str) -> None:
    _report_max_diff(y, ref, label)
    torch.testing.assert_close(y, ref, rtol=RTOL, atol=ATOL)


@unittest.skipUnless(_HAS_CUDA, "requires CUDA")
class TestHcCollapseTriton(unittest.TestCase):
    def test_matches_reference_token_sweep(self) -> None:
        for T in (1, 8, 512, 4096):
            with self.subTest(T=T):
                pre = _mixed_pre(T, seed=1000 + T)
                x = _randn_bf16((T, HC, H), seed=2000 + T)
                y = hc_collapse(pre, x)
                self.assertEqual(tuple(y.shape), (T, H))
                self.assertEqual(y.dtype, torch.bfloat16)
                ref = _reference_collapse(pre, x)
                _assert_equivalent(y, ref, f"T={T}, H={H}")

    def test_noncontiguous_x_stride_variant(self) -> None:
        T = 8
        storage = _randn_bf16((HC, T, H), seed=3001)
        x = storage.transpose(0, 1)  # [T, HC, H], strides (H, T*H, 1)
        self.assertFalse(x.is_contiguous())
        pre = _mixed_pre(T, seed=3002)
        y = hc_collapse(pre, x)
        ref = _reference_collapse(pre, x)
        _assert_equivalent(y, ref, f"noncontiguous T={T}, H={H}")

    def test_fp32_pre_is_cast_before_product(self) -> None:
        T = 8
        pre_fp32 = _mixed_pre(T, seed=4001).float()
        x = _randn_bf16((T, HC, H), seed=4002)
        y = hc_collapse(pre_fp32, x)
        ref = _reference_collapse(pre_fp32, x)
        _assert_equivalent(y, ref, f"fp32 pre, T={T}, H={H}")

    def test_leading_batch_layout_is_flattened(self) -> None:
        B, S = 2, 16
        pre = _mixed_pre(B * S, seed=5001).view(B, S, HC)
        x = _randn_bf16((B, S, HC, H), seed=5002)
        y = hc_collapse(pre, x)
        self.assertEqual(tuple(y.shape), (B, S, H))
        ref = _reference_collapse(pre, x)
        _assert_equivalent(y, ref, f"4D x B={B}, S={S}, H={H}")

    def test_out_buffer_is_written_in_place(self) -> None:
        T = 8
        pre = _mixed_pre(T, seed=6001)
        x = _randn_bf16((T, HC, H), seed=6002)
        out = torch.empty((T, H), dtype=torch.bfloat16, device=_DEVICE)
        y = hc_collapse(pre, x, out=out)
        self.assertIs(y, out)
        ref = _reference_collapse(pre, x)
        _assert_equivalent(y, ref, f"out= reuse, T={T}, H={H}")

    def test_masked_tail_and_small_h(self) -> None:
        T = 8
        for h in (1, 97, 1000):
            with self.subTest(H=h):
                pre = _mixed_pre(T, seed=7000 + h)
                x = _randn_bf16((T, HC, h), seed=7100 + h)
                y = hc_collapse(pre, x)
                self.assertEqual(tuple(y.shape), (T, h))
                ref = _reference_collapse(pre, x)
                _assert_equivalent(y, ref, f"T={T}, H={h}")

    def test_zero_tokens(self) -> None:
        pre = _mixed_pre(0, seed=8001)
        x = _randn_bf16((0, HC, H), seed=8002)
        y = hc_collapse(pre, x)
        ref = _reference_collapse(pre, x)
        self.assertEqual(tuple(y.shape), (0, H))
        _assert_equivalent(y, ref, "T=0")


if __name__ == "__main__":
    unittest.main()
