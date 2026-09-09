"""Precision tests for fused_logits_head_gate (F3)."""

import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
    _baseline_logits_head_gate,
    fp32_linear_logits_head_gate,
    fused_logits_head_gate,
    project_fp32_logits_head_gate,
    scale_fp32_logits_head_gate,
)


class TestSplitFP32HeadGateCpu(unittest.TestCase):
    """No CUDA required; the production helper module needs torch + triton."""

    def test_early_and_late_formula_2d_3d_scale_one_projection(self):
        torch.manual_seed(123)
        x = torch.randn(8, 17, dtype=torch.bfloat16)
        linear = nn.Linear(17, 3, bias=False, dtype=torch.float32)
        scales = torch.rand(8, 6)[:, ::2]  # Exercise non-contiguous fallback.
        scale_const = 0.03125
        expected = linear(x.float()).unsqueeze(-1) * scales.unsqueeze(-1) * scale_const
        for q_scale in (scales, scales.unsqueeze(-1)):
            with self.subTest(dim=q_scale.dim()):
                early_proj = Mock(wraps=linear)
                raw = project_fp32_logits_head_gate(x, early_proj)
                raw_before = raw.clone()
                early = scale_fp32_logits_head_gate(raw, q_scale, scale_const)
                early_proj.assert_called_once()
                self.assertEqual(raw.shape, (8, 3))
                self.assertEqual(raw.dtype, torch.float32)
                self.assertTrue(torch.equal(raw, raw_before))
                self.assertTrue(torch.equal(early, expected))
                late_proj = Mock(wraps=linear)
                late = fp32_linear_logits_head_gate(x, q_scale, late_proj, scale_const)
                late_proj.assert_called_once()
                self.assertEqual(late.shape, (8, 3, 1))
                self.assertTrue(torch.equal(early, late))

    def test_projection_does_not_depend_on_later_query_scale(self):
        x = torch.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=torch.bfloat16)
        linear = nn.Linear(2, 3, bias=False, dtype=torch.float32)
        projection = Mock(wraps=linear)
        raw = project_fp32_logits_head_gate(x, projection)
        for value in (0.0, -1.0, 2.0):
            scales = torch.full_like(raw, value)
            out = scale_fp32_logits_head_gate(raw, scales, 0.25)
            self.assertTrue(torch.equal(out, (raw * scales).unsqueeze(-1) * 0.25))
        projection.assert_called_once()

    def test_invalid_precomputed_fp32_falls_back_to_original_input(self):
        x = torch.randn(4, 17, dtype=torch.bfloat16)
        projection = Mock(side_effect=lambda value: value[:, :3])
        # CPU / wrong-shaped producer views must not bypass the input cast.
        supplied = torch.full((1, 2), 99.0, dtype=torch.float32)
        with patch(
            "rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate._hy4_cast_fp32",
            side_effect=AssertionError("CPU must not invoke Triton cast"),
        ):
            out = project_fp32_logits_head_gate(x, projection, x_fp32=supplied)
        projection.assert_called_once()
        self.assertTrue(torch.equal(projection.call_args.args[0], x.float()))
        self.assertTrue(torch.equal(out, x.float()[:, :3]))


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSplitFP32HeadGateCuda(unittest.TestCase):
    def test_early_late_byte_exact_with_producer_fp32(self):
        torch.manual_seed(321)
        for rows in (1, 8, 32, 256):
            x = torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            linear = nn.Linear(6144, 32, bias=False, device="cuda", dtype=torch.float32)
            scales = torch.rand(rows, 32, device="cuda", dtype=torch.float32)
            for q_scale in (scales, scales.unsqueeze(-1)):
                for x_fp32 in (None, x.float()):
                    with self.subTest(rows=rows, dim=q_scale.dim(), producer=x_fp32 is not None):
                        projection = Mock(wraps=linear)
                        raw = project_fp32_logits_head_gate(x, projection, x_fp32=x_fp32)
                        early = scale_fp32_logits_head_gate(raw, q_scale, 0.03125)
                        projection.assert_called_once()
                        if x_fp32 is not None:
                            self.assertIs(projection.call_args.args[0], x_fp32)
                        late = fp32_linear_logits_head_gate(x, q_scale, linear, 0.03125, x_fp32=x_fp32)
                        reference = _baseline_logits_head_gate(x, q_scale, linear, 0.03125)
                        self.assertTrue(torch.equal(early, late))
                        self.assertTrue(torch.equal(early, reference))

    def test_multistream_cuda_graph_replays_fresh_inputs(self):
        torch.manual_seed(99)
        x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
        x_fp32 = x.float()
        scales = torch.rand(8, 32, 1, device="cuda", dtype=torch.float32)
        linear = nn.Linear(6144, 32, bias=False, device="cuda", dtype=torch.float32)
        side = torch.cuda.Stream()
        ready = torch.cuda.Event()

        def run_split():
            caller = torch.cuda.current_stream()
            side.wait_stream(caller)
            with torch.cuda.stream(side):
                raw = project_fp32_logits_head_gate(x, linear, x_fp32=x_fp32)
                ready.record()
            # Query scales become available independently on the caller.
            query_scales = scales * 1.25
            caller.wait_event(ready)
            return scale_fp32_logits_head_gate(raw, query_scales, 0.03125)

        for _ in range(3):
            run_split()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run_split()
        for _ in range(3):
            x.copy_(torch.randn_like(x))
            x_fp32.copy_(x.float())
            scales.copy_(torch.rand_like(scales))
            graph.replay()
            reference = fp32_linear_logits_head_gate(x, scales * 1.25, linear, 0.03125, x_fp32=x_fp32)
            self.assertTrue(torch.equal(out, reference))


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

    def test_hy4_high_precision(self):
        """HY4 uses TF32x3 because the result directly gates discrete top-k."""
        device = "cuda"
        for T in (128, 1024):
            with self.subTest(T=T):
                K, N = 6144, 32
                x = torch.randn(T, K, dtype=torch.bfloat16, device=device)
                weight = (
                    torch.randn(N, K, dtype=torch.float32, device=device) * 0.02
                )
                q_scale = (
                    torch.randn(T, N, 1, dtype=torch.float32, device=device).abs()
                    + 0.1
                )
                scale_const = K**-0.5 * N**-0.5
                linear = nn.Linear(K, N, bias=False, device=device)
                linear.weight.data.copy_(weight)

                ref = _baseline_logits_head_gate(x, q_scale, linear, scale_const)
                out = fused_logits_head_gate(
                    x,
                    q_scale,
                    weight,
                    scale_const,
                    fallback_proj=linear,
                    high_precision=True,
                )

                self.assertTrue(torch.allclose(out, ref, atol=1e-5, rtol=1e-5))
                topk = min(8, N)
                self.assertTrue(
                    torch.equal(
                        out.squeeze(-1).topk(topk, dim=-1).indices,
                        ref.squeeze(-1).topk(topk, dim=-1).indices,
                    )
                )

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
