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
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.nn.functional as F


def _load_fused_gate():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_gate_fused_triton.py"))
    spec = importlib.util.spec_from_file_location("_v4_gate_fused", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_sqrtsoftplus_gate


def _load_eager_route_selector():
    from rtp_llm.models_py.modules.dsv4.moe.gate import (
        _select_routes_with_nonfinite_fallback,
    )

    return _select_routes_with_nonfinite_fallback


def _load_projection_gate():
    path = os.path.join(os.path.dirname(__file__), "..", "moe", "gate.py")
    spec = importlib.util.spec_from_file_location("_v41_projection_gate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.Gate.__new__(module.Gate)
    torch.nn.Module.__init__(gate)
    return gate


class GateProjectionTest(unittest.TestCase):
    def setUp(self):
        self.gate = _load_projection_gate()
        self.gate._prefill_gate_chunk_rows = 32768

    def test_partition_rows_mm_out_and_no_retained_scores(self):
        # CPU arithmetic with CUDA metadata mocked exercises the actual wrapper.
        weight = torch.arange(30).reshape(3, 10).to(torch.bfloat16)[:, ::2]
        for rows in (32769, 32770, 32856, 33280, 65537):
            with self.subTest(rows=rows), torch.no_grad():
                x = torch.zeros((rows * 2, 10), dtype=torch.bfloat16)[::2, ::2]
                x[:, 0] = 1
                expected = torch.cat(
                    [F.linear(x[s : s + 32768], weight) for s in range(0, rows, 32768)]
                )
                state = dict(vars(self.gate))
                with patch.object(
                    torch.Tensor,
                    "is_cuda",
                    new_callable=PropertyMock,
                    return_value=True,
                ), patch.object(torch, "mm", wraps=torch.mm) as mm, patch.object(
                    torch, "cat", side_effect=AssertionError("no cat")
                ):
                    result = self.gate._project_scores(x, weight)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                self.assertEqual(
                    [c.args[0].shape[0] for c in mm.call_args_list],
                    [min(32768, rows - s) for s in range(0, rows, 32768)],
                )
                self.assertTrue(
                    all(
                        c.kwargs["out"].untyped_storage().data_ptr()
                        == result.untyped_storage().data_ptr()
                        for c in mm.call_args_list
                    )
                )
                self.assertEqual(result.untyped_storage().nbytes(), rows * 3 * 2)
                self.assertEqual(vars(self.gate), state)

    def test_original_branches(self):
        x = torch.zeros((32856, 5), dtype=torch.bfloat16)
        w = torch.zeros((3, 5), dtype=torch.bfloat16)
        cases = [
            ("cpu", x, w, 32768, False),
            ("default", x, w, None, True),
            ("disabled", x, w, 0, True),
            ("fp32", x.float(), w.float(), 32768, True),
            ("weight_fp32", x, w.float(), 32768, True),
            ("3d", x.unsqueeze(0), w, 32768, True),
        ]
        cases += [("small_%d" % n, x[:n], w, 32768, True) for n in (0, 1, 32768)]
        for name, a, b, rows, cuda in cases:
            with self.subTest(case=name), torch.no_grad():
                if rows is None:
                    del self.gate._prefill_gate_chunk_rows
                else:
                    self.gate._prefill_gate_chunk_rows = rows
                marker = object()
                with patch.object(
                    torch.Tensor,
                    "is_cuda",
                    new_callable=PropertyMock,
                    return_value=cuda,
                ), patch.object(
                    F, "linear", return_value=marker
                ) as linear, patch.object(
                    torch, "mm", side_effect=AssertionError("no mm-out")
                ):
                    self.assertIs(self.gate._project_scores(a, b), marker)
                    linear.assert_called_once_with(a, b)

    def test_grad_enabled_falls_back_and_backward_works(self):
        x = torch.ones((32769, 2), dtype=torch.bfloat16, requires_grad=True)
        w = torch.ones((3, 2), dtype=torch.bfloat16, requires_grad=True)
        with torch.enable_grad(), patch.object(
            torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
        ), patch.object(F, "linear", wraps=F.linear) as linear, patch.object(
            torch, "mm", side_effect=AssertionError("no out with autograd")
        ):
            result = self.gate._project_scores(x, w)
            linear.assert_called_once_with(x, w)
            result.float().sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    def test_ced_precedes_partition_and_dtype_grad_gates(self):
        projection = Mock(return_value=object())
        self.gate._ced_row_projection = projection
        x = torch.empty((32856, 5))
        w = torch.empty((3, 5))
        with torch.enable_grad(), patch.object(
            F, "linear", side_effect=AssertionError("CED first")
        ):
            self.assertIs(self.gate._project_scores(x, w), projection.return_value)
        projection.assert_called_once_with(x, w)


def _eager_sqrtsoftplus_gate(
    scores: torch.Tensor,  # [N, E] fp32
    bias: torch.Tensor,  # [E] fp32
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


class GateVisionRoutingTest(unittest.TestCase):
    def test_hash_routes_across_text_and_image_batches(self):
        from rtp_llm.models_py.modules.dsv4.moe.gate import Gate
        from rtp_llm.utils.model_weight import W

        gate = Gate(
            layer_id=0,
            dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            n_hash_layers=1,
            vocab_size=8,
            layer_weights={
                W.v4_router_w: torch.zeros(4, 4, dtype=torch.bfloat16),
                W.v4_router_tid2eid: torch.tensor([[0, 1]] * 8),
                W.v4_router_bias_vl: torch.tensor([0.0, 0.0, 10.0, 20.0]),
            },
        )
        x = torch.zeros(3, 4, dtype=torch.bfloat16)
        for has_visual_tokens in (False, True, False):
            with self.subTest(has_visual_tokens=has_visual_tokens):
                gate._has_visual_tokens = has_visual_tokens
                if has_visual_tokens:
                    weights, indices = gate(x, torch.tensor([1, 8, 2]))
                    expected_indices = torch.tensor([[0, 1], [3, 2], [0, 1]])
                else:
                    with patch.object(
                        torch.Tensor,
                        "topk",
                        side_effect=AssertionError(
                            "text hash routing must not run topk"
                        ),
                    ):
                        weights, indices = gate(x, torch.tensor([1, 2, 3]))
                    expected_indices = torch.tensor([[0, 1]] * 3)
                torch.testing.assert_close(indices, expected_indices)
                torch.testing.assert_close(weights, torch.full((3, 2), 0.5))


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
            scores,
            bias,
            K,
            route_scale=route_scale,
        )

        fused = _load_fused_gate()
        w_fused, i_fused = fused(
            scores.contiguous(),
            bias.contiguous(),
            topk=K,
            route_scale=route_scale,
            norm_eps=1e-12,
        )

        # Indices must match exactly (downstream expert selection MUST agree).
        self.assertEqual(
            i_ref.tolist(),
            i_fused.tolist(),
            f"indices differ at N={N},E={E},K={K}",
        )
        # Weights agree to ~ULP (FP32 reduction-order may drift slightly).
        diff = (w_ref - w_fused).abs()
        ref_mag = w_ref.abs().mean().item() + 1e-9
        rel_max = diff.max().item() / ref_mag
        self.assertLess(
            rel_max,
            1e-4,
            f"weights rel max {rel_max:.3e} exceeds 1e-4 (N={N},E={E},K={K})",
        )

    def test_v4_flash_default_shape(self):
        # V4-Flash: E=256 experts, K=topk=6
        self._check(N=128, E=256, K=6, route_scale=2.5)

    def test_text_tokens_keep_fused_gate_with_vision_bias_loaded(self):
        from rtp_llm.models_py.modules.dsv4.moe.gate import Gate
        from rtp_llm.utils.model_weight import W

        layer_weights = {
            W.v4_router_w: torch.randn(4, 8, device="cuda", dtype=torch.bfloat16),
            W.v4_router_bias: torch.randn(4, device="cuda", dtype=torch.float32),
            W.v4_router_bias_vl: torch.randn(4, device="cuda", dtype=torch.float32),
        }
        gate = Gate(
            layer_id=1,
            dim=8,
            n_routed_experts=4,
            n_activated_experts=2,
            vocab_size=32,
            layer_weights=layer_weights,
        )
        gate._has_visual_tokens = False
        expected = (
            torch.ones(3, 2, device="cuda"),
            torch.zeros(3, 2, device="cuda", dtype=torch.long),
        )
        with patch(
            "rtp_llm.models_py.modules.dsv4.moe.gate.fused_sqrtsoftplus_gate",
            return_value=expected,
        ) as fused, patch(
            "rtp_llm.models_py.modules.dsv4.moe.gate._use_fused_gate",
            return_value=True,
        ):
            result = gate(
                torch.randn(3, 8, device="cuda", dtype=torch.bfloat16),
                torch.tensor([1, 2, 3], device="cuda"),
            )

        self.assertIs(result, expected)
        fused.assert_called_once()
        self.assertIs(fused.call_args.args[1], gate.bias)

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
        _, i_fused = fused(
            scores.contiguous(),
            bias.contiguous(),
            topk=K,
            route_scale=1.0,
            norm_eps=1e-12,
        )
        for row in range(N):
            unique = set(i_fused[row].tolist())
            self.assertEqual(
                len(unique),
                K,
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
        w, _ = fused(
            scores.contiguous(),
            bias.contiguous(),
            topk=6,
            route_scale=route_scale,
            norm_eps=1e-12,
        )
        sums = w.sum(dim=-1)
        # Each row should sum to ~route_scale (post-normalization × route_scale).
        rel = (sums - route_scale).abs() / route_scale
        self.assertLess(
            rel.max().item(),
            1e-5,
            f"row sums deviate from route_scale={route_scale}; max rel={rel.max().item():.3e}",
        )

    def test_nonfinite_rows_use_safe_fallback(self):
        device = "cuda:0"
        N, E, K = 4, 256, 6
        route_scale = 2.5
        scores = torch.randn(N, E, device=device, dtype=torch.float32)
        bias = torch.randn(E, device=device, dtype=torch.float32) * 0.1
        scores[0, 3] = float("nan")
        scores[1, 7] = float("inf")
        scores[2, 11] = -float("inf")

        fused = _load_fused_gate()
        weights, indices = fused(
            scores.contiguous(),
            bias.contiguous(),
            topk=K,
            route_scale=route_scale,
            norm_eps=1e-12,
        )
        torch.cuda.synchronize()

        expected_indices = torch.arange(K, device=device, dtype=torch.int64)
        expected_weights = torch.full(
            (K,), route_scale / K, device=device, dtype=torch.float32
        )
        for row in range(3):
            self.assertTrue(torch.equal(indices[row], expected_indices))
            self.assertTrue(torch.allclose(weights[row], expected_weights))
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue(((indices >= 0) & (indices < E)).all().item())


class GateEagerNonfiniteTest(unittest.TestCase):
    def test_nonfinite_rows_use_safe_fallback(self):
        N, E, K = 4, 16, 6
        route_scale = 2.5
        original_scores = torch.rand(N, E, dtype=torch.float32)
        ranking_scores = original_scores + torch.randn(E) * 0.1
        original_scores[0, 3] = float("nan")
        ranking_scores[0, 3] = float("nan")
        original_scores[1, 7] = float("inf")
        ranking_scores[1, 7] = float("inf")
        original_scores[2, 11] = -float("inf")
        ranking_scores[2, 11] = -float("inf")

        select_routes = _load_eager_route_selector()
        weights, indices = select_routes(
            original_scores,
            ranking_scores,
            K,
            route_scale,
            True,
        )

        expected_weights = torch.full((K,), route_scale / K, dtype=torch.float32)
        for row in range(3):
            self.assertTrue(torch.allclose(weights[row], expected_weights))
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue(((indices >= 0) & (indices < E)).all().item())


if __name__ == "__main__":
    unittest.main()
