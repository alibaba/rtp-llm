"""Correctness tests for MOSS-VL cross-attn Triton fused kernels.

Each test compares the fused Triton kernel output against the pure-torch
reference path (the code it replaces in mossvl.py).
"""

import unittest

import torch

from rtp_llm.models_py.model_desc.mossvl import (
    MossVLQKNorm,
    apply_mrope_rope,
    _rotate_half,
)
from rtp_llm.models_py.model_desc.mossvl_triton import (
    fused_gated_residual_add,
    fused_mrope_rope,
    fused_qk_rmsnorm,
)


class GatedResidualAddTest(unittest.TestCase):
    def test_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(42)
        for shape in [(1, 4096), (4, 4096), (16, 128, 4096)]:
            residual = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            gate = torch.tensor([0.73], dtype=torch.float32, device="cuda")
            ref = residual + gate.to(torch.bfloat16) * x
            got = fused_gated_residual_add(residual, x, gate)
            self.assertIsNotNone(got, f"fused returned None for {shape}")
            torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    def test_cpu_fallback(self):
        x = torch.randn(4, 8, dtype=torch.bfloat16)
        gate = torch.tensor([0.5], dtype=torch.float32)
        self.assertIsNone(fused_gated_residual_add(x, x, gate))


class MropeRopeTest(unittest.TestCase):
    def test_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(42)
        head_dim = 128
        for n_rows, n_heads in [(1, 32), (4, 32), (16, 8)]:
            x = torch.randn(n_rows, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            cos = torch.randn(n_rows, head_dim, dtype=torch.bfloat16, device="cuda").cos()
            sin = torch.randn(n_rows, head_dim, dtype=torch.bfloat16, device="cuda").sin()
            # broadcast cos/sin over heads
            cos_b = cos.unsqueeze(1)  # [rows, 1, dim]
            sin_b = sin.unsqueeze(1)
            # reference: apply_mrope_rope expects x [rows, heads, dim], cos [rows, dim]
            # it unsqueezes cos to [rows, 1, dim] internally
            ref = apply_mrope_rope(x, cos, sin)
            got = fused_mrope_rope(x, cos_b.expand_as(x), sin_b.expand_as(x))
            self.assertIsNotNone(got, f"fused returned None for {n_rows}x{n_heads}")
            # allow bf16 tolerance
            diff = (ref.float() - got.float()).abs().max().item()
            self.assertLess(diff, 0.1, f"max diff {diff} for {n_rows}x{n_heads}")

    def test_cpu_fallback(self):
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        cos = torch.randn(2, 128, dtype=torch.bfloat16).cos()
        self.assertIsNone(fused_mrope_rope(x, cos, cos))


class QKRMSNormTest(unittest.TestCase):
    def test_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        torch.manual_seed(42)
        head_dim = 128
        weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda")
        norm = MossVLQKNorm(weight, eps=1e-6)
        for n_rows, n_heads in [(1, 32), (4, 8), (16, 16)]:
            x = torch.randn(n_rows, n_heads, head_dim, dtype=torch.bfloat16, device="cuda")
            ref = norm(x)
            got = fused_qk_rmsnorm(x, weight, 1e-6)
            self.assertIsNotNone(got, f"fused returned None for {n_rows}x{n_heads}")
            torch.testing.assert_close(got, ref, rtol=1e-2, atol=1e-2)

    def test_cpu_fallback(self):
        x = torch.randn(2, 4, 128, dtype=torch.bfloat16)
        w = torch.randn(128, dtype=torch.bfloat16)
        self.assertIsNone(fused_qk_rmsnorm(x, w, 1e-6))


if __name__ == "__main__":
    unittest.main()
