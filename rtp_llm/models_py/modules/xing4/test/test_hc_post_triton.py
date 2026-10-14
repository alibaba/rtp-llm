"""Equivalence UT for the Xing4 HC post-combination Triton kernel (K4).

Reference (numerical authority, handwritten copy of
``rtp_llm/models_py/model_desc/xing4_0_model.py`` ``Xing4HCUnit._post_impl``,
L160-170)::

    post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
    first  = post_b.to(dt).unsqueeze(-1) * x.unsqueeze(-2)   # [T, hc, H]
    second = torch.matmul(comb.to(dt), residual.to(dt))      # comb untransposed
    out    = (first + second).to(dt)

Rounding-structure contract (post_triton.py): the kernel rounds the post*x
fp32 product to bf16, accumulates the comb@residual sum in fp32 (ascending-j
order) and rounds it to bf16 once per element, then adds the two bf16 values
in fp32 opmath and rounds to bf16 — the exact eager op sequence. Products of
two bf16 values are exact in fp32, so the only differences the eager
reference can show are:

* zero/identity comb, tie pin: NONE — the 4-term sums are exact in every
  accumulation order, so those cases are gated BIT-EXACT and prove the whole
  rounding chain (including the Triton extf/truncf fold workaround).
* sinkhorn comb: the eager cuBLAS bf16 bmm rounds its own fp32 accumulation
  in a different order than the kernel's ascending-j chain; on ~1e-5 of
  elements the two land on opposite sides of a bf16 tie, a one-ulp-of-second
  difference that cancellation in the final add can amplify RELATIVE to the
  (smaller) output (measured: worst 0.03125 absolute at second magnitude
  [4,8), 484/58.7M elements at T=4096). Those cases are gated with the
  lane-pinned rtol=2**-7 plus an atol of one bf16 ulp of the accumulated
  stream magnitude — atol = 2**-7 * (1 + max_t,i sum_j |comb| * max|residual|)
  — which bounds one ulp of |second| = |sum_j comb*residual| exactly. Every
  case also reports the measured max diff and bitwise-diff element count.

Fixture domains follow the lane contract: T in {1, 8, 512, 4096}, H=3584,
hc=4; comb in the real _pre_impl sinkhorn output domain (non-negative, row
sums ~1) plus adversarial zero/identity combs; post in the reference (0, 2)
domain (2 * sigmoid(N(0,1))).
"""

import unittest

import torch

from rtp_llm.models_py.modules.xing4.hc.post_triton import (
    hc_post_triton,
    xing4_hc_post,
)

HC = 4
H = 3584
RTOL = 2**-7
ATOL = 2**-7
_TS = (1, 8, 512, 4096)
_SINKHORN_ITERS = 20
_SINKHORN_EPS = 1e-6


def _reference_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Handwritten copy of Xing4HCUnit._post_impl (xing4_0_model.py L160-170)."""
    dt = x.dtype
    post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
    first = post_b.to(dt).unsqueeze(-1) * x.unsqueeze(-2)
    second = torch.matmul(comb.to(dt), residual.to(dt))
    return (first + second).to(dt)


def _cpu_randn(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Deterministic fp32 fixture (CPU generation keeps runs reproducible)."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, dtype=torch.float32, generator=gen)


def _sinkhorn_comb(T: int, hc: int, seed: int) -> torch.Tensor:
    """comb in the real _pre_impl output domain (xing4_0_model.py L137-144):
    exp(clamp(logits, +-30) - rowmax) then 20 (row+eps, col+eps) sinkhorn
    rounds -> non-negative with row sums ~= 1."""
    logits = (_cpu_randn((T, hc, hc), seed) * 4.0).clamp(min=-30.0, max=30.0)
    comb = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    for _ in range(_SINKHORN_ITERS):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _SINKHORN_EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _SINKHORN_EPS)
    return comb


def _post_domain(T: int, hc: int, seed: int) -> torch.Tensor:
    """post in the reference (0, 2) domain: 2 * sigmoid(N(0,1))."""
    return 2.0 * torch.sigmoid(_cpu_randn((T, hc), seed))


def _cuda_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(device="cuda", dtype=torch.bfloat16)


def _eager_compat_atol(residual: torch.Tensor, comb: torch.Tensor) -> float:
    """One bf16 ulp of the accumulated stream magnitude (see module docstring).

    |second| = |sum_j comb[t,i,j] * residual[t,j,h]| is bounded by
    (max row abs-sum of comb) * max|residual|, and one bf16 ulp of a value v
    is at most 2**-7 * v, so atol = 2**-7 * row_abs_sum * res_max bounds the
    one-ulp-of-second cuBLAS tie flips; the extra 2**-7 keeps zero-output
    elements inside the gate.
    """
    row_abs_sum = comb.float().abs().sum(dim=-1).amax().item()
    res_max = residual.float().abs().amax().item()
    return ATOL * (1.0 + row_abs_sum * res_max)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestXing4HcPostTriton(unittest.TestCase):
    def _check(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
        label: str,
        bitexact: bool = False,
    ) -> torch.Tensor:
        fused = hc_post_triton(post, comb, x, residual)
        ref = _reference_post(x, residual, post, comb)
        if bitexact:
            self.assertTrue(torch.equal(fused, ref), f"{label}: not bit-exact")
        else:
            torch.testing.assert_close(
                fused.float(), ref.float(), rtol=RTOL, atol=_eager_compat_atol(residual, comb)
            )
        diff = (fused.float() - ref.float()).abs()
        max_diff = diff.max().item()
        bit_diff = int((diff > 0).sum().item())
        print(
            f"[hc_post_triton] {label}: T={x.shape[0]} H={H} hc={HC} "
            f"max|diff|={max_diff:.3e} bitdiff={bit_diff}/{diff.numel()}"
        )
        return fused

    def test_equivalence_sinkhorn_comb_domain(self) -> None:
        for T in _TS:
            with self.subTest(T=T):
                comb = _cuda_bf16(_sinkhorn_comb(T, HC, seed=100 + T))
                post = _cuda_bf16(_post_domain(T, HC, seed=200 + T))
                x = _cuda_bf16(_cpu_randn((T, H), seed=300 + T))
                residual = _cuda_bf16(_cpu_randn((T, HC, H), seed=400 + T))
                self._check(x, residual, post, comb, "comb=sinkhorn")

    def test_equivalence_adversarial_comb_bitexact(self) -> None:
        # zero and identity comb make the 4-term sums exact in ANY
        # accumulation order, so the kernel must match the eager reference
        # bitwise: this pins the whole rounding chain (first product round,
        # second round, final add round).
        for T in _TS:
            for mode in ("zero", "identity"):
                with self.subTest(T=T, comb=mode):
                    if mode == "zero":
                        comb = torch.zeros(
                            T, HC, HC, dtype=torch.bfloat16, device="cuda"
                        )
                    else:
                        comb = _cuda_bf16(
                            torch.eye(HC, dtype=torch.float32).repeat(T, 1, 1)
                        )
                    post = _cuda_bf16(_post_domain(T, HC, seed=500 + T))
                    x = _cuda_bf16(_cpu_randn((T, H), seed=600 + T))
                    residual = _cuda_bf16(_cpu_randn((T, HC, H), seed=700 + T))
                    self._check(x, residual, post, comb, f"comb={mode}", bitexact=True)

    def test_round_ties_to_even_final_add(self) -> None:
        # Deterministic RTNE pin for the final add: first=1.0 and
        # second=0.49609375 (both bf16) sum to 1.49609375, an exact bf16 tie
        # midway between 1.4921875 (odd mantissa) and 1.5 (even mantissa).
        # Round-to-nearest-even must pick 1.5; the Triton extf/truncf fold
        # (see post_triton.py) used to bypass the roundings and produced the
        # wrong side of the tie on ~28% of random elements.
        T = 1
        post = torch.ones(T, HC, device="cuda", dtype=torch.bfloat16)
        comb = _cuda_bf16(torch.eye(HC, dtype=torch.float32).repeat(T, 1, 1))
        x = torch.ones(T, H, device="cuda", dtype=torch.bfloat16)
        residual = (
            torch.tensor(0.49609375, dtype=torch.float32)
            .to(device="cuda", dtype=torch.bfloat16)
            .view(1, 1, 1)
            .repeat(T, HC, H)
        )
        fused = hc_post_triton(post, comb, x, residual)
        ref = _reference_post(x, residual, post, comb)
        torch.testing.assert_close(fused.float(), ref.float(), rtol=0.0, atol=0.0)
        self.assertEqual(fused[0, 0, 0].item(), 1.5)

    def test_dropin_matches_reference(self) -> None:
        # Production shapes straight out of _pre_impl: post [T, hc, 1] and
        # comb [T, hc, hc], both already bf16.
        T = 8
        x = _cuda_bf16(_cpu_randn((T, H), seed=11))
        residual = _cuda_bf16(_cpu_randn((T, HC, H), seed=12))
        post_bf = _cuda_bf16(_post_domain(T, HC, seed=13))
        comb_bf = _cuda_bf16(_sinkhorn_comb(T, HC, seed=14))
        out = xing4_hc_post(x, residual, post_bf.unsqueeze(-1), comb_bf)
        self.assertEqual(tuple(out.shape), (T, HC, H))
        self.assertEqual(out.dtype, torch.bfloat16)
        ref = _reference_post(x, residual, post_bf.unsqueeze(-1), comb_bf)
        torch.testing.assert_close(
            out.float(), ref.float(), rtol=RTOL, atol=_eager_compat_atol(residual, comb_bf)
        )

        # Un-cast fp32 post/comb (same device, as production would produce
        # them): the drop-in casts exactly like the reference (the cast happens
        # before the post*x bf16 rounding in both paths).
        post_f32 = _post_domain(T, HC, seed=15).to(device="cuda")
        comb_f32 = _sinkhorn_comb(T, HC, seed=16).to(device="cuda")
        out2 = xing4_hc_post(x, residual, post_f32.unsqueeze(-1), comb_f32)
        ref2 = _reference_post(x, residual, post_f32.unsqueeze(-1), comb_f32)
        torch.testing.assert_close(
            out2.float(), ref2.float(), rtol=RTOL, atol=_eager_compat_atol(residual, comb_f32)
        )

    def test_dropin_batched_leading_dims(self) -> None:
        # x [B, S, H], residual [B, S, hc, H]: the drop-in flattens to the
        # grid=(T,) launch layout while the reference handles the batch dims
        # through broadcasting + batched matmul.
        B, S = 2, 4
        T = B * S
        x = _cuda_bf16(_cpu_randn((B, S, H), seed=21))
        residual = _cuda_bf16(_cpu_randn((B, S, HC, H), seed=22))
        post = _cuda_bf16(_post_domain(T, HC, seed=23)).view(B, S, HC, 1)
        comb = _cuda_bf16(_sinkhorn_comb(T, HC, seed=24)).view(B, S, HC, HC)
        out = xing4_hc_post(x, residual, post, comb)
        self.assertEqual(tuple(out.shape), (B, S, HC, H))
        ref = _reference_post(x, residual, post, comb)
        torch.testing.assert_close(
            out.float(), ref.float(), rtol=RTOL, atol=_eager_compat_atol(residual, comb)
        )

    def test_out_alias_residual_bit_identical(self) -> None:
        # Pins the memory-saving wiring: out may alias residual because each
        # program reads its whole residual[t, :, h_block] slice before storing
        # out[t, :, h_block]; a kernel change that breaks that ordering fails
        # here loudly (in-place must be bit-identical to a fresh buffer).
        T = 512
        post = _cuda_bf16(_post_domain(T, HC, seed=31))
        comb = _cuda_bf16(_sinkhorn_comb(T, HC, seed=32))
        x = _cuda_bf16(_cpu_randn((T, H), seed=33))
        residual = _cuda_bf16(_cpu_randn((T, HC, H), seed=34))

        out_fresh = hc_post_triton(post, comb, x, residual)
        residual_alias = residual.clone()
        out_alias = hc_post_triton(post, comb, x, residual_alias, out=residual_alias)
        self.assertEqual(out_alias.data_ptr(), residual_alias.data_ptr())
        self.assertNotEqual(out_fresh.data_ptr(), residual.data_ptr())
        torch.testing.assert_close(
            out_alias.float(), out_fresh.float(), rtol=0.0, atol=0.0
        )


class TestHcPostTritonContract(unittest.TestCase):
    """CPU-only input validation: the contract checks run before the launch."""

    def _cpu_case(self) -> tuple[torch.Tensor, ...]:
        x = torch.zeros(2, H, dtype=torch.bfloat16)
        residual = torch.zeros(2, HC, H, dtype=torch.bfloat16)
        post = torch.zeros(2, HC, dtype=torch.bfloat16)
        comb = torch.zeros(2, HC, HC, dtype=torch.bfloat16)
        return post, comb, x, residual

    def test_rejects_non_bf16(self) -> None:
        post, comb, x, residual = self._cpu_case()
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            hc_post_triton(post, comb, torch.zeros(2, H, dtype=torch.float16), residual)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            hc_post_triton(
                post, comb, x, torch.zeros(2, HC, H, dtype=torch.float32)
            )
        # post/comb must arrive already cast: rounding-structure contract.
        with self.assertRaisesRegex(ValueError, "already be cast"):
            hc_post_triton(post.float(), comb, x, residual)
        with self.assertRaisesRegex(ValueError, "already be cast"):
            hc_post_triton(post, comb.float(), x, residual)

    def test_rejects_shape_mismatch(self) -> None:
        post, comb, x, residual = self._cpu_case()
        with self.assertRaises(ValueError):
            hc_post_triton(
                post, comb, torch.zeros(2, H + 1, dtype=torch.bfloat16), residual
            )
        with self.assertRaises(ValueError):
            hc_post_triton(torch.zeros(3, HC, dtype=torch.bfloat16), comb, x, residual)
        with self.assertRaises(ValueError):
            hc_post_triton(
                post, torch.zeros(2, HC, HC + 1, dtype=torch.bfloat16), x, residual
            )

    def test_rejects_cpu_tensors(self) -> None:
        post, comb, x, residual = self._cpu_case()
        with self.assertRaisesRegex(ValueError, "CUDA"):
            hc_post_triton(post, comb, x, residual)


if __name__ == "__main__":
    unittest.main()
