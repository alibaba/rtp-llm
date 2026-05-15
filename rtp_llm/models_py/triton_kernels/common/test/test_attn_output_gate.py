"""Unit tests for attn_output_gate.py — SigmoidMulInplace + FP8 quant kernel.

Validates numerical correctness against an eager PyTorch reference across
a Cartesian product of (T, H, dtype) combinations matching the shapes seen
in Qwen3.5 full-attention layers (T=tokens, H=head_num*head_dim).

Run with pytest:
    python -m pytest rtp_llm/models_py/triton_kernels/common/test/test_attn_output_gate.py -v -s
"""

import sys
import unittest

import torch

from rtp_llm.models_py.modules.base.cuda.attn_output_gate import SigmoidMulInplace
from rtp_llm.models_py.triton_kernels.common.attn_output_gate import (
    sigmoid_mul_fp8_quant_fwd,
)

T_VALUES = [1, 2, 4, 8, 16, 32, 256, 1024, 4096]
H_VALUES = [2048, 4096, 7168]
DTYPES = [torch.bfloat16, torch.float16]


def _ref_sigmoid_mul(attn_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """fp32 reference: out = attn_output * sigmoid(gate), cast back to dtype."""
    return (attn_output.float() * torch.sigmoid(gate.float())).to(attn_output.dtype)


class TestSigmoidMulInplace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(0)
        cls.op = SigmoidMulInplace().cuda()

    def _run_case(self, T, H, dtype):
        attn_output = (torch.randn(T, H, device="cuda") * 2.0).to(dtype)
        gate = (torch.randn(T, H, device="cuda") * 4.0).to(dtype)

        ref = _ref_sigmoid_mul(attn_output, gate)
        out = self.op(attn_output.clone(), gate)

        # Triton's tl.sigmoid uses fast-math approximation; torch.sigmoid is
        # bit-exact. Allow ~1 ULP slack at the dtype's epsilon magnitude.
        atol = 5e-3 if dtype == torch.bfloat16 else 1e-3
        rtol = 5e-3 if dtype == torch.bfloat16 else 2e-3
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_correctness_grid(self):
        for T in T_VALUES:
            for H in H_VALUES:
                for dtype in DTYPES:
                    with self.subTest(T=T, H=H, dtype=dtype):
                        self._run_case(T, H, dtype)

    def test_inplace_semantics(self):
        T, H, dtype = 8, 4096, torch.bfloat16
        attn_output = (torch.randn(T, H, device="cuda") * 2.0).to(dtype)
        gate = (torch.randn(T, H, device="cuda") * 4.0).to(dtype)
        ret = self.op(attn_output, gate)
        # The op should return the same object as the in-place modified tensor
        self.assertTrue(ret.data_ptr() == attn_output.data_ptr())

    def test_zero_size(self):
        attn_output = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
        gate = torch.empty(0, 4096, device="cuda", dtype=torch.bfloat16)
        out = self.op(attn_output, gate)
        self.assertEqual(out.shape, (0, 4096))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSigmoidMulFp8Quant(unittest.TestCase):
    """Correctness: dequantized fused output vs exact float reference."""

    def _check(self, T: int, H: int, dtype: torch.dtype) -> float:
        torch.manual_seed(42 + T + H)
        attn_output = (torch.randn(T, H, device="cuda") * 2.0).to(dtype)
        gate = (torch.randn(T, H, device="cuda") * 4.0).to(dtype)

        # Compute reference BEFORE calling fused (which may modify attn_output in-place)
        ref = attn_output.float() * torch.sigmoid(gate.float())

        fp8_out, scale_out = sigmoid_mul_fp8_quant_fwd(
            attn_output, gate, quant_group_size=128, scale_ue8m0=False
        )

        # Dequantize kernel output
        num_groups = H // 128
        actual_deq = (
            fp8_out.float().view(T, num_groups, 128) * scale_out.float().unsqueeze(-1)
        ).view(T, H)

        # FP8 e4m3fn has ~1/16 relative precision; per-group quant adds
        # another factor. Relative error is bounded by the group quantization
        # step size (~absmax/448 per group).
        max_diff = (actual_deq - ref).abs().max().item()
        rel_err = ((actual_deq - ref).abs() / (ref.abs() + 1e-6)).mean().item()
        self.assertLess(
            rel_err,
            0.05,
            f"T={T}, H={H}, dtype={dtype}: mean_rel_err={rel_err:.4e}, max_diff={max_diff:.4e}",
        )
        return max_diff

    def test_correctness_grid(self):
        for T in [1, 4, 16, 128, 1024]:
            for H in [2048, 4096]:
                for dtype in [torch.bfloat16, torch.float16]:
                    with self.subTest(T=T, H=H, dtype=dtype):
                        max_diff = self._check(T, H, dtype)
                        print(
                            f"  fp8_quant  T={T:5d} H={H:5d} "
                            f"dtype={str(dtype).replace('torch.',''):>8}  "
                            f"max_diff={max_diff:.3e}  OK"
                        )

    def test_output_dtype(self):
        T, H = 16, 2048
        attn = (torch.randn(T, H, device="cuda") * 2.0).to(torch.bfloat16)
        gate = (torch.randn(T, H, device="cuda") * 4.0).to(torch.bfloat16)
        fp8_out, scale_out = sigmoid_mul_fp8_quant_fwd(attn, gate)
        self.assertEqual(fp8_out.dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8_out.shape, (T, H))

    def test_empty_T(self):
        attn = torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
        gate = torch.empty(0, 2048, device="cuda", dtype=torch.bfloat16)
        fp8_out, scale_out = sigmoid_mul_fp8_quant_fwd(attn, gate)
        self.assertEqual(fp8_out.shape, (0, 2048))


if __name__ == "__main__":
    unittest.main(verbosity=2)
