"""Precision tests for fused_logits_head_gate (F3)."""

import unittest

import torch
from torch import nn

from rtp_llm.models_py.triton_kernels.common.fused_logits_head_gate import (
    _baseline_logits_head_gate,
    fused_logits_head_gate,
)


class TestFusedLogitsHeadGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(42)

    def _run(self, T: int, K: int, N: int, weight_dtype=torch.float32):
        device = "cuda"
        x = torch.randn(T, K, dtype=torch.bfloat16, device=device)
        # DeepSeek-V3.2 default: weights_proj.weight is fp32 (see
        # models/deepseek_v2.py:237 data_type=torch.float32).
        weight = (torch.randn(N, K, device=device) * 0.02).to(weight_dtype)
        q_scale = torch.randn(T, N, 1, dtype=torch.float32, device=device).abs() + 0.1
        scale_const = K**-0.5 * N**-0.5
        linear = nn.Linear(K, N, bias=False, device=device)
        linear.weight.data = weight.float()

        ref = _baseline_logits_head_gate(x, q_scale, linear, scale_const)
        out = fused_logits_head_gate(
            x, q_scale, weight, scale_const, fallback_proj=linear
        )

        self.assertEqual(out.shape, ref.shape)
        diff = (out.float() - ref.float()).abs()
        rel = diff / (ref.float().abs() + 1e-6)
        max_abs = diff.max().item()
        # bf16 tensor cores → fp32 accum: typical ~1e-7 abs error
        self.assertLess(max_abs, 1e-3, f"T={T} K={K} N={N}: max_abs={max_abs}")
        # Bulk relative error should be small (small abs / small ref → big rel
        # is OK for outliers; check p99 for stability).
        sorted_rel = rel.flatten().sort().values
        p99 = sorted_rel[int(rel.numel() * 0.99)].item()
        self.assertLess(p99, 5e-2, f"T={T} K={K} N={N}: p99 rel={p99}")

    def test_dsv32_typical_fp32_weight(self):
        """DSV3.2 typical: hidden=7168, indexer_n_heads=64, weight=fp32 (production)."""
        for T in (1, 8, 16, 32, 128, 256, 1024, 2048):
            with self.subTest(T=T):
                self._run(T, K=7168, N=64, weight_dtype=torch.float32)

    def test_dsv32_typical_bf16_weight(self):
        """Smoke test bf16 weight path (kernel skips the in-register cast)."""
        for T in (32, 1024):
            with self.subTest(T=T):
                self._run(T, K=7168, N=64, weight_dtype=torch.bfloat16)

    def test_glm5_like(self):
        """GLM5-like shape: hidden=6144 (non power-of-2)."""
        for T in (1, 32, 256):
            with self.subTest(T=T):
                self._run(T, K=6144, N=64)

    def test_smaller_n(self):
        for T, K, N in [(32, 7168, 16), (32, 4096, 32), (32, 2048, 128)]:
            with self.subTest(T=T, K=K, N=N):
                self._run(T, K, N)

    def test_transposed_weight_production_layout(self):
        """Production layout: weight stored as [K, N] contiguous, accessed as [N, K] transposed."""
        device = "cuda"
        for T in (1, 8, 16, 32, 128, 1024):
            for K, N in [(6144, 32), (7168, 64)]:
                with self.subTest(T=T, K=K, N=N):
                    x = torch.randn(T, K, dtype=torch.bfloat16, device=device)
                    w_storage = (
                        torch.randn(K, N, dtype=torch.float32, device=device) * 0.02
                    )
                    weight = w_storage.t()  # [N, K] with stride=(1, N)
                    q_scale = (
                        torch.randn(T, N, 1, dtype=torch.float32, device=device).abs()
                        + 0.1
                    )
                    scale_const = K**-0.5 * N**-0.5
                    linear = nn.Linear(K, N, bias=False, device=device)
                    linear.weight.data = weight.contiguous()

                    ref = _baseline_logits_head_gate(x, q_scale, linear, scale_const)
                    out = fused_logits_head_gate(
                        x, q_scale, weight, scale_const, fallback_proj=linear
                    )

                    self.assertEqual(out.shape, ref.shape)
                    diff = (out.float() - ref.float()).abs()
                    rel = diff / (ref.float().abs() + 1e-6)
                    max_abs = diff.max().item()
                    self.assertLess(
                        max_abs, 1e-3, f"T={T} K={K} N={N}: max_abs={max_abs}"
                    )
                    sorted_rel = rel.flatten().sort().values
                    p99 = sorted_rel[int(rel.numel() * 0.99)].item()
                    self.assertLess(p99, 5e-2, f"T={T} K={K} N={N}: p99 rel={p99}")

    def test_fallback_path(self):
        """Out-of-range K triggers fallback."""
        device = "cuda"
        T, K, N = 8, 12288, 64  # K > MAX_K=8192
        x = torch.randn(T, K, dtype=torch.bfloat16, device=device)
        weight = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02
        q_scale = torch.randn(T, N, 1, dtype=torch.float32, device=device).abs() + 0.1
        scale_const = K**-0.5 * N**-0.5
        linear = nn.Linear(K, N, bias=False, device=device)
        linear.weight.data = weight.float()
        out = fused_logits_head_gate(
            x, q_scale, weight, scale_const, fallback_proj=linear
        )
        ref = _baseline_logits_head_gate(x, q_scale, linear, scale_const)
        self.assertTrue(torch.allclose(out, ref, atol=1e-4, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
