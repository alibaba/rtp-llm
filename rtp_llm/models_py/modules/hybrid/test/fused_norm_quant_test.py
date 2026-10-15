"""Correctness UT for the shared fused RMSNorm + FP8 UE8M0 quant wrapper.

Reference chain (the exact two-kernel sequence the fusion replaces):
    flashinfer.norm.rmsnorm(x, w, eps)
    -> sgl_per_token_group_quant_fp8(normed, group_size=128, eps=1e-4,
       column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True)

The wrapper must reproduce that chain at the Xing4 decode/prefill shapes
(input/post layernorm H=3584, q_a_layernorm N=768) including:

* row-strided x (the [T, 768] split view of the 1344-wide fused QKV
  projection, stride(0)=1344) — fused kernel reads rows through
  x_stride_m, matching flashinfer's strided-input contract;
* in-place ``out_norm=x`` (input/post layernorm sites norm the dead
  pre-norm stream in place);
* M in {1, 8, 64, 4096} covering decode bs1, the graph capture buckets,
  and prefill 4K;
* a torch fp32 anchor so a systematic FlashInfer rmsnorm deviation cannot
  pass as a shared reference error (dsv4 discipline).

Bit-level thresholds follow the dsv4 fused kernel contract
(test_fused_rmsnorm_fp8_quant.py): norm_exact >= 0.9999 with max|diff| in
bf16 ulp terms, fp8/scale byte-exact rates reported and asserted.
"""

import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.hybrid.fused_norm_quant import (
    fused_rmsnorm_fp8_quant_ue8m0,
    linear_supports_quantized_input,
)


def _flashinfer_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    import flashinfer

    return flashinfer.norm.rmsnorm(x, weight, eps=eps)


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float):
    x32 = x.float()
    inv = torch.rsqrt(x32.square().mean(-1, keepdim=True) + eps)
    return (x32 * inv * weight.float()).to(x.dtype)


def _raw_fp8_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (
        a.contiguous().view(torch.uint8).to(torch.int16)
        - b.contiguous().view(torch.uint8).to(torch.int16)
    ).abs()


def _reference_chain(x: torch.Tensor, weight: torch.Tensor, eps: float):
    ref_norm = _flashinfer_rmsnorm(x.contiguous(), weight, eps)
    ref_q, ref_s = sgl_per_token_group_quant_fp8(
        ref_norm,
        group_size=128,
        eps=1.0e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    return ref_norm, ref_q, ref_s


class _QuantizedInputStub(torch.nn.Module):
    """Duck-typed ue8m0 DeepGEMM linear for capability-check coverage."""

    def __init__(self, k: int, scale_ue8m0: bool):
        super().__init__()
        self.K = k
        self.scale_ue8m0 = scale_ue8m0

    def quantize_input(self, x):
        raise NotImplementedError

    def forward_quantized(self, fp8, scales):
        raise NotImplementedError


class _WrapperStub(torch.nn.Module):
    """Duck-typed CudaFp8GEMMLinear wrapper around a DeepGEMM inner."""

    FLASHINFER_M_THRESHOLD = 32

    def __init__(self, k: int, with_flashinfer: bool):
        super().__init__()
        self._deepgemm_linear = _QuantizedInputStub(k, True)
        self._flashinfer_linear = (
            object() if with_flashinfer else None
        )


class FusedNormQuantTest(unittest.TestCase):
    def _check_case(self, m: int, n: int, strided: bool, inplace: bool):
        eps = 1.0e-6
        torch.manual_seed(20261016 + m * 17 + n)
        x = (torch.randn((m, n), device="cuda", dtype=torch.bfloat16) * 0.3).contiguous()
        weight = (
            (torch.randn((n,), device="cuda") * 0.1 + 1.0)
            .abs()
            .to(torch.bfloat16)
            .contiguous()
        )

        if strided:
            # Fuse the [T, n] rows into a wider (n + 576)-wide projection
            # output and take the leading split view, mirroring the
            # q_lora=768 view of the 1344-wide fused QKV projection.
            wide = torch.randn(
                (m, n + 576), device="cuda", dtype=torch.bfloat16
            ).contiguous()
            wide[:, :n] = x
            x_view = wide[:, :n]
            self.assertEqual(x_view.stride(0), n + 576)
        else:
            x_view = x

        ref_norm, ref_q, ref_s = _reference_chain(x, weight, eps)
        ref_norm_torch = _torch_rmsnorm(x, weight, eps)

        if inplace:
            self.assertTrue(x_view.is_contiguous())
            got_norm, got_q, got_s = fused_rmsnorm_fp8_quant_ue8m0(
                x_view, weight, eps=eps, out_norm=x_view
            )
            self.assertIs(got_norm, x_view)
        else:
            x_copy = x_view.clone()
            got_norm, got_q, got_s = fused_rmsnorm_fp8_quant_ue8m0(
                x_copy, weight, eps=eps
            )
            # Non-inplace keeps the input untouched.
            self.assertTrue(torch.equal(x_copy, x_view))
        torch.cuda.synchronize()

        norm_bit_diff = (
            (
                ref_norm.contiguous().view(torch.int16)
                != got_norm.contiguous().view(torch.int16)
            )
            .sum()
            .item()
        )
        norm_abs = (ref_norm.float() - got_norm.float()).abs()
        torch_norm_abs = (ref_norm_torch.float() - got_norm.float()).abs()
        q_diff = _raw_fp8_diff(ref_q, got_q)
        # Compare only the real per-group scale nibbles. K % 512 != 0 (e.g.
        # N=768 -> 6 groups) leaves dead nibbles in the last packed int32
        # word: the reference leaves them as torch.empty garbage (its byte
        # values differ run to run), the fused kernel writes a deterministic
        # exponent, and DeepGEMM only reads the K/128 real nibbles.
        n_groups = n // 128
        ref_s_bytes = ref_s.contiguous().view(torch.uint8).view(m, -1)
        got_s_bytes = got_s.contiguous().view(torch.uint8).view(m, -1)
        s_diff = (
            ref_s_bytes[:, :n_groups].to(torch.int16)
            - got_s_bytes[:, :n_groups].to(torch.int16)
        ).abs()

        total = ref_norm.numel()
        norm_exact = 1.0 - norm_bit_diff / total
        q_exact = (q_diff == 0).float().mean().item()
        s_exact = (s_diff == 0).float().mean().item()
        print(
            f"[m={m} n={n} strided={strided} inplace={inplace}] "
            f"norm_max={float(norm_abs.max()):.8f} norm_exact={norm_exact:.8f} "
            f"torch_max={float(torch_norm_abs.max()):.8f} "
            f"q_max_ulp={int(q_diff.max())} q_exact={q_exact:.8f} "
            f"s_max_byte={int(s_diff.max())} s_exact={s_exact:.8f}"
        )

        self.assertLessEqual(float(norm_abs.max()), 0.03125)
        self.assertGreaterEqual(norm_exact, 0.9999)
        self.assertLessEqual(float(torch_norm_abs.max()), 0.03125)
        # The quantized payload the DeepGEMM consumers read must be
        # byte-stable: fp8 exact on real activations, packed scales
        # byte-identical (degenerate all-zero groups can legitimately differ
        # in the packed exponent, which the random input never produces).
        self.assertGreaterEqual(q_exact, 0.9999)
        self.assertLessEqual(int(q_diff.max()), 1)
        # Real nibbles only (padding nibbles are dead; see comment above).
        self.assertEqual(s_exact, 1.0)

    def test_xing4_input_norm_shapes(self):
        # input_layernorm / post_attention_layernorm, H = 3584.
        for m in (1, 8, 64, 4096):
            self._check_case(m, 3584, strided=False, inplace=True)

    def test_xing4_qa_norm_strided(self):
        # q_a_layernorm: 768-wide strided split view of the fused QKV proj.
        for m in (1, 8, 64, 4096):
            self._check_case(m, 768, strided=True, inplace=False)

    def test_qa_norm_contiguous_inplace(self):
        self._check_case(64, 768, strided=False, inplace=True)
        self._check_case(1, 768, strided=False, inplace=False)

    def test_validation_rejects(self):
        m, n = 8, 3584
        x = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)
        w = torch.ones((n,), device="cuda", dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            fused_rmsnorm_fp8_quant_ue8m0(x.float(), w, eps=1e-6)
        with self.assertRaises(ValueError):
            fused_rmsnorm_fp8_quant_ue8m0(
                x.reshape(m * n).unsqueeze(0), w, eps=1e-6
            )
        with self.assertRaises(ValueError):
            fused_rmsnorm_fp8_quant_ue8m0(x, w[: n - 128], eps=1e-6)
        # stride(0) < N is not a valid row layout.
        transposed = torch.randn((n, m), device="cuda", dtype=torch.bfloat16).t()
        with self.assertRaises(ValueError):
            fused_rmsnorm_fp8_quant_ue8m0(transposed, w, eps=1e-6)

    def test_linear_capability_check(self):
        ok = _QuantizedInputStub(3584, True)
        ue8m0_off = _QuantizedInputStub(3584, False)
        k_mismatch = _QuantizedInputStub(768, True)
        plain = torch.nn.Linear(3584, 16).cuda()
        self.assertTrue(linear_supports_quantized_input(ok, 3584))
        self.assertFalse(linear_supports_quantized_input(ok, 768))
        self.assertFalse(linear_supports_quantized_input(ue8m0_off, 3584))
        self.assertFalse(linear_supports_quantized_input(k_mismatch, 3584))
        self.assertFalse(linear_supports_quantized_input(plain, 3584))

    def test_wrapper_capability_check(self):
        # Wrapper without a FlashInfer backend: DeepGEMM runs for every m.
        wrapper_pure = _WrapperStub(3584, with_flashinfer=False)
        self.assertTrue(linear_supports_quantized_input(wrapper_pure, 3584))
        self.assertTrue(
            linear_supports_quantized_input(wrapper_pure, 3584, m_rows=1)
        )
        self.assertFalse(linear_supports_quantized_input(wrapper_pure, 768))
        # Wrapper with the small-M FlashInfer backend: the quantized pair is
        # only usable once DeepGEMM is the active backend (m >= 32).
        wrapper_fi = _WrapperStub(3584, with_flashinfer=True)
        self.assertFalse(linear_supports_quantized_input(wrapper_fi, 3584))
        self.assertFalse(
            linear_supports_quantized_input(wrapper_fi, 3584, m_rows=8)
        )
        self.assertTrue(
            linear_supports_quantized_input(wrapper_fi, 3584, m_rows=32)
        )
        self.assertTrue(
            linear_supports_quantized_input(wrapper_fi, 3584, m_rows=4096)
        )


if __name__ == "__main__":
    unittest.main()
