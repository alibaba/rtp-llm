"""UT for the V4 fused router-gate epilogue (DSV4_GATE_FUSED).

Replaces the per-token chain
    scores = F.softplus(scores).sqrt()       # 2 elementwise launches
    scores_b = scores + bias                  # 1 elementwise
    indices = scores_b.topk(topk)[1]          # mbtopk: ~3 launches
    weights = scores.gather(1, indices)       # 1 vectorized_gather
    weights = weights / (weights.sum(-1) + eps) * route_scale  # 2 launches
with one Triton kernel (~7-10 launches → 1 per layer × 43 layers).

Default flipped to ON (DSV4_GATE_FUSED=1) on 2026-05-04; UT verifies
indices match exactly and weights are within tight tolerance vs the
eager epilogue.

Bypasses rtp_llm package init via importlib.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch
import torch.nn.functional as F


def _load_fused_gate():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(
        os.path.join(here, "..", "_gate_fused_triton.py")
    )
    spec = importlib.util.spec_from_file_location("_v4_gate_fused", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_sqrtsoftplus_gate


def _eager_sqrtsoftplus_gate(
    scores: torch.Tensor,    # [N, E] fp32
    bias: torch.Tensor,      # [E] fp32
    topk: int,
    route_scale: float,
    norm_eps: float = 1e-12,
):
    """Eager epilogue mirroring moe.py:Gate.forward when score_func='sqrtsoftplus'."""
    s = F.softplus(scores).sqrt()
    s_biased = s + bias
    indices = s_biased.topk(topk, dim=-1)[1]
    weights = s.gather(1, indices)
    weights = weights / (weights.sum(-1, keepdim=True) + norm_eps) * route_scale
    return weights, indices


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GateFusedEquivTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            _load_fused_gate()
        except Exception as e:
            raise unittest.SkipTest(f"fused_sqrtsoftplus_gate not importable: {e}")

    def _check(self, *, N, E, K, route_scale=2.5):
        torch.manual_seed(0)
        device = "cuda:0"
        scores = torch.randn(N, E, device=device, dtype=torch.float32)
        bias = torch.randn(E, device=device, dtype=torch.float32) * 0.1

        w_ref, i_ref = _eager_sqrtsoftplus_gate(
            scores, bias, K, route_scale=route_scale,
        )

        fused = _load_fused_gate()
        w_fused, i_fused = fused(
            scores.contiguous(), bias.contiguous(),
            topk=K, route_scale=route_scale, norm_eps=1e-12,
        )

        # Indices must match exactly (downstream expert selection MUST agree).
        self.assertEqual(
            i_ref.tolist(), i_fused.tolist(),
            f"indices differ at N={N},E={E},K={K}",
        )
        # Weights agree to ~ULP (FP32 reduction-order may drift slightly).
        diff = (w_ref - w_fused).abs()
        ref_mag = w_ref.abs().mean().item() + 1e-9
        rel_max = diff.max().item() / ref_mag
        self.assertLess(
            rel_max, 1e-4,
            f"weights rel max {rel_max:.3e} exceeds 1e-4 (N={N},E={E},K={K})",
        )

    def test_v4_flash_default_shape(self):
        # V4-Flash: E=256 experts, K=topk=6
        self._check(N=128, E=256, K=6, route_scale=2.5)

    def test_single_token(self):
        self._check(N=1, E=256, K=6)

    def test_small_batch(self):
        self._check(N=16, E=256, K=6)

    def test_large_batch(self):
        self._check(N=1024, E=256, K=6)

    def test_smaller_E(self):
        # E=128 power of 2
        self._check(N=64, E=128, K=4)

    def test_topk_8(self):
        # K=8 is the next power of 2 above the 6-default — kernel BLOCK_K
        # should pick this up.
        self._check(N=64, E=256, K=8)

    def test_indices_unique_per_row(self):
        """topk indices must be unique within each row (sanity check)."""
        torch.manual_seed(1)
        device = "cuda:0"
        N, E, K = 32, 256, 6
        scores = torch.randn(N, E, device=device, dtype=torch.float32)
        bias = torch.randn(E, device=device, dtype=torch.float32) * 0.1
        fused = _load_fused_gate()
        _, i_fused = fused(scores.contiguous(), bias.contiguous(),
                           topk=K, route_scale=1.0, norm_eps=1e-12)
        for row in range(N):
            unique = set(i_fused[row].tolist())
            self.assertEqual(
                len(unique), K,
                f"row {row} has duplicate indices: {i_fused[row].tolist()}",
            )

    def test_weights_normalized(self):
        """sum(weights) per row should equal route_scale (within tol)."""
        torch.manual_seed(2)
        device = "cuda:0"
        scores = torch.randn(32, 256, device=device, dtype=torch.float32)
        bias = torch.randn(256, device=device, dtype=torch.float32) * 0.1
        route_scale = 2.5
        fused = _load_fused_gate()
        w, _ = fused(scores.contiguous(), bias.contiguous(),
                     topk=6, route_scale=route_scale, norm_eps=1e-12)
        sums = w.sum(dim=-1)
        # Each row should sum to ~route_scale (post-normalization × route_scale).
        rel = (sums - route_scale).abs() / route_scale
        self.assertLess(
            rel.max().item(), 1e-5,
            f"row sums deviate from route_scale={route_scale}; max rel={rel.max().item():.3e}",
        )


if __name__ == "__main__":
    unittest.main()
