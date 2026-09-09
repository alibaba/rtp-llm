"""Platform-neutral numerical contract tests for the DeepSeek-V4 router."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe"),
)

from rtp_llm.models_py.modules.dsv4.moe import gate as gate_module
from rtp_llm.models_py.modules.dsv4 import platform_provider as provider_module
from rtp_llm.models_py.modules.dsv4.moe.gate import (
    Gate,
    _select_routes_with_nonfinite_fallback,
)


def _canonical_finite_oracle(
    original_scores: torch.Tensor,
    ranking_scores: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent Python oracle: score descending, then eid ascending."""
    oracle_indices: list[list[int]] = []
    oracle_weights: list[list[float]] = []
    for original_row, ranking_row in zip(
        original_scores.tolist(), ranking_scores.tolist()
    ):
        selected = sorted(
            range(len(ranking_row)), key=lambda eid: (-ranking_row[eid], eid)
        )[:topk]
        selected_scores = [original_row[eid] for eid in selected]
        denominator = sum(selected_scores)
        oracle_indices.append(selected)
        oracle_weights.append([score / denominator for score in selected_scores])
    return (
        torch.tensor(oracle_weights, dtype=torch.float32),
        torch.tensor(oracle_indices, dtype=torch.long),
    )


def _gate_from_logits(
    n_experts: int,
    topk: int,
    route_scale: float,
    bias: torch.Tensor,
    *,
    device: torch.device | str = "cpu",
) -> Gate:
    """Construct Gate without framework weight descriptors for contract UTs."""
    gate = Gate.__new__(Gate)
    nn.Module.__init__(gate)
    gate._fp32_gemm = False
    gate._fused_gate = True
    gate._bf16_fp32_linear = provider_module.build_dsv4_bf16_fp32_linear(
        lambda x, w: torch.nn.functional.linear(x, w).float(),
        platform_provider=provider_module.DefaultDsv4PlatformProvider(),
    )
    gate.dim = n_experts
    gate.topk = topk
    gate.score_func = "sqrtsoftplus"
    gate.route_scale = route_scale
    gate.hash = False
    gate.bias = bias.to(device=device, dtype=torch.float32).contiguous()
    gate.weight = torch.eye(n_experts, device=device, dtype=torch.float32)
    gate._dbg_prefix = None
    return gate


class RouterEagerContractTest(unittest.TestCase):
    def test_gate_preserves_provider_fp32_logits_before_selection(self):
        gate = _gate_from_logits(8, 2, 1.0, torch.zeros(8))
        x = torch.zeros((1, 8), dtype=torch.bfloat16)
        # All logits collapse to 1 in BF16; retaining FP32 selects experts7/6.
        precise = 1.0 + torch.arange(8, dtype=torch.float32).view(1, 8) * 0.0001
        self.assertTrue(torch.equal(precise.bfloat16(), torch.ones_like(x)))
        with mock.patch.dict(os.environ, {"DSV4_GATE_FP32": "0"}), mock.patch.object(
            gate_module, "_use_fused_gate", return_value=False
        ), mock.patch.object(
            gate,
            "_bf16_fp32_linear",
            return_value=precise,
        ) as dispatch:
            _, indices = gate(x)
        self.assertEqual(indices.tolist(), [[7, 6]])
        dispatch.assert_called_once()
        self.assertIs(dispatch.call_args.args[0], x)
        self.assertEqual(dispatch.call_args.args[1].dtype, torch.bfloat16)

    def test_gate_legacy_provider_fallback_remains_unchanged(self):
        gate = _gate_from_logits(8, 2, 1.0, torch.zeros(8))
        x = torch.arange(8, dtype=torch.bfloat16).view(1, 8)
        with mock.patch.dict(os.environ, {"DSV4_GATE_FP32": "0"}), mock.patch.object(
            gate_module, "_use_fused_gate", return_value=False
        ), mock.patch.object(
            gate,
            "_bf16_fp32_linear",
            side_effect=lambda x, w: torch.nn.functional.linear(x, w).float(),
        ):
            actual_w, actual_i = gate(x)
        gate._fp32_gemm = True
        with mock.patch.dict(os.environ, {"DSV4_GATE_FP32": "1"}), mock.patch.object(
            gate_module, "_use_fused_gate", return_value=False
        ):
            expected_w, expected_i = gate(x)
        self.assertTrue(torch.equal(actual_i, expected_i))
        self.assertTrue(torch.equal(actual_w, expected_w))

    def test_gate_fp32_override_and_empty_batch_bypass_provider(self):
        gate = _gate_from_logits(8, 2, 1.0, torch.zeros(8))
        gate._fp32_gemm = True
        with mock.patch.dict(os.environ, {"DSV4_GATE_FP32": "1"}), mock.patch.object(
            gate_module, "_use_fused_gate", return_value=False
        ), mock.patch.object(gate, "_bf16_fp32_linear") as dispatch:
            gate(torch.arange(8, dtype=torch.float32).view(1, 8))
            weights, indices = gate(torch.empty((0, 8), dtype=torch.bfloat16))
        dispatch.assert_not_called()
        self.assertEqual(weights.shape, (0, 2))
        self.assertEqual(indices.shape, (0, 2))

    def test_tiny_e8_top2_bias_only_ranks_and_unbiased_scores_weight(self):
        original = torch.tensor(
            [
                [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
                [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
            ],
            dtype=torch.float32,
        )
        bias = torch.tensor(
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0], dtype=torch.float32
        )
        ranking = original + bias
        expected_w, expected_i = _canonical_finite_oracle(original, ranking, 2)

        weights, indices = _select_routes_with_nonfinite_fallback(
            original, ranking, 2, 1.0, True
        )

        self.assertTrue(torch.equal(indices, expected_i))
        self.assertTrue(torch.allclose(weights, expected_w, atol=0.0, rtol=1e-6))
        # Row 0 selects eid=3 because of bias, but its weight is the unbiased
        # 0.60 rather than the ranking score 1.10.
        self.assertEqual(indices[0].tolist(), [3, 0])
        self.assertTrue(torch.allclose(weights[0], torch.tensor([0.4, 0.6])))

    def test_full_e256_top6_ties_are_lowest_eids_and_deterministic(self):
        original = torch.ones((3, 256), dtype=torch.float32)
        ranking = torch.ones_like(original)
        expected_w, expected_i = _canonical_finite_oracle(original, ranking, 6)

        first = _select_routes_with_nonfinite_fallback(
            original, ranking, 6, 1.0, True
        )
        second = _select_routes_with_nonfinite_fallback(
            original, ranking, 6, 1.0, True
        )

        self.assertTrue(torch.equal(first[1], expected_i))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.allclose(first[0], expected_w))
        self.assertEqual(first[1][0].tolist(), [0, 1, 2, 3, 4, 5])

    def test_raw_activated_or_ranking_nonfinite_invalidates_whole_row(self):
        n_experts, topk = 8, 2
        original = torch.arange(1, 1 + 4 * n_experts, dtype=torch.float32).view(
            4, n_experts
        )
        ranking = original.clone()
        router_logits = torch.zeros_like(original)
        router_logits[0, 7] = float("nan")  # raw-only failure
        original[1, 6] = float("inf")  # activated-score failure
        ranking[2, 5] = -float("inf")  # ranking-score failure

        weights, indices = _select_routes_with_nonfinite_fallback(
            original,
            ranking,
            topk,
            1.0,
            True,
            router_logits=router_logits,
        )

        expected_indices = torch.arange(topk, dtype=torch.long)
        expected_weights = torch.full((topk,), 1.0 / topk)
        for row in range(3):
            self.assertTrue(torch.equal(indices[row], expected_indices))
            self.assertTrue(torch.equal(weights[row], expected_weights))
        self.assertEqual(indices[3].tolist(), [7, 6])
        self.assertTrue(torch.isfinite(weights).all())

    def test_hash_supplied_indices_are_overridden_for_bad_row(self):
        original = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]] * 2,
            dtype=torch.float32,
        )
        ranking = original.clone()
        router_logits = torch.zeros_like(original)
        router_logits[0, 0] = float("nan")
        # Arbitrary/OOB ids on a bad row are replaced before valid-row checks.
        supplied = torch.tensor([[99, -1], [5, 4]], dtype=torch.long)

        weights, indices = _select_routes_with_nonfinite_fallback(
            original,
            ranking,
            2,
            1.0,
            True,
            supplied,
            router_logits=router_logits,
        )

        self.assertEqual(indices[0].tolist(), [0, 1])
        self.assertEqual(weights[0].tolist(), [0.5, 0.5])
        self.assertEqual(indices[1].tolist(), [5, 4])

    def test_actual_eager_gate_scale_switch_preserves_default_api(self):
        gate = _gate_from_logits(
            8, 2, 2.5, torch.zeros(8, dtype=torch.float32)
        )
        logits = torch.zeros((2, 8), dtype=torch.float32)
        input_ids = torch.arange(2, dtype=torch.long)
        with mock.patch.dict(
            os.environ, {"DSV4_GATE_FP32": "1", "DSV4_GATE_FUSED": "0"}
        ), mock.patch.object(gate_module, "_use_fused_gate", return_value=False):
            scaled_w, scaled_i = gate(logits, input_ids)
            unit_w, unit_i = gate(logits, input_ids, include_route_scale=False)

        self.assertTrue(torch.equal(scaled_i, unit_i))
        self.assertTrue(torch.allclose(scaled_w, unit_w * 2.5))
        self.assertTrue(torch.allclose(unit_w.sum(dim=-1), torch.ones(2)))
        self.assertTrue(torch.allclose(scaled_w.sum(dim=-1), torch.full((2,), 2.5)))

    def test_selector_error_matrix(self):
        scores = torch.ones((2, 8), dtype=torch.float32)
        rank3_scores = scores.view(2, 2, 4)
        cases = (
            (ValueError, "rank", lambda: _select_routes_with_nonfinite_fallback(
                rank3_scores, rank3_scores, 2, 1.0, True
            )),
            (ValueError, "shape", lambda: _select_routes_with_nonfinite_fallback(
                scores, scores[:, :7], 2, 1.0, True
            )),
            (ValueError, "topk_zero", lambda: _select_routes_with_nonfinite_fallback(
                scores, scores, 0, 1.0, True
            )),
            (ValueError, "topk_large", lambda: _select_routes_with_nonfinite_fallback(
                scores, scores, 9, 1.0, True
            )),
            (ValueError, "logit_shape", lambda: _select_routes_with_nonfinite_fallback(
                scores,
                scores,
                2,
                1.0,
                True,
                router_logits=scores[:, :7],
            )),
            (ValueError, "index_shape", lambda: _select_routes_with_nonfinite_fallback(
                scores,
                scores,
                2,
                1.0,
                True,
                indices=torch.zeros((2, 3), dtype=torch.long),
            )),
            (TypeError, "index_dtype", lambda: _select_routes_with_nonfinite_fallback(
                scores,
                scores,
                2,
                1.0,
                True,
                indices=torch.zeros((2, 2), dtype=torch.int32),
            )),
            (ValueError, "index_device", lambda: _select_routes_with_nonfinite_fallback(
                scores,
                scores,
                2,
                1.0,
                True,
                indices=torch.zeros((2, 2), dtype=torch.long, device="meta"),
            )),
            (RuntimeError, "index_oob", lambda: _select_routes_with_nonfinite_fallback(
                scores,
                scores,
                2,
                1.0,
                True,
                indices=torch.tensor([[0, 8], [1, 2]], dtype=torch.long),
            )),
        )
        for error, name, call in cases:
            with self.subTest(name=name), self.assertRaises(error):
                call()


_ACTUAL_FUSED_AVAILABLE = bool(
    torch.cuda.is_available() and gate_module._GATE_FUSED_OK
)


@unittest.skipUnless(
    _ACTUAL_FUSED_AVAILABLE,
    "UNVERIFIED: actual CUDA Triton fused Gate is unavailable on this device",
)
class RouterActualFusedContractTest(unittest.TestCase):
    def _run_actual_fused(
        self, logits: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, mock.Mock]:
        gate = _gate_from_logits(
            logits.size(1), 6, 1.5, bias, device=logits.device
        )
        # Construction freezes this choice; preserve FP32 extreme inputs before
        # exercising the actual fused router's finite/nonfinite contract.
        gate._fp32_gemm = True
        actual_fused = gate_module.fused_sqrtsoftplus_gate
        assert actual_fused is not None
        with mock.patch.dict(
            os.environ, {"DSV4_GATE_FP32": "1", "DSV4_GATE_FUSED": "1"}
        ), mock.patch.object(
            gate_module, "fused_sqrtsoftplus_gate", wraps=actual_fused
        ) as fused_spy:
            result = gate(
                logits,
                torch.arange(logits.size(0), device=logits.device),
                include_route_scale=False,
            )
        return result[0], result[1], fused_spy

    def test_actual_fused_tie_uses_lowest_expert_ids(self):
        logits = torch.zeros((3, 256), device="cuda", dtype=torch.float32)
        weights, indices, fused_spy = self._run_actual_fused(
            logits, torch.zeros(256, device="cuda", dtype=torch.float32)
        )

        fused_spy.assert_called_once()
        expected_indices = torch.arange(6, device="cuda", dtype=torch.long)
        self.assertTrue(torch.equal(indices, expected_indices.expand(3, 6)))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 6.0)))

    def test_actual_fused_raw_nonfinite_falls_back_whole_row(self):
        logits = torch.zeros((4, 256), device="cuda", dtype=torch.float32)
        logits[0, 3] = float("nan")
        logits[1, 7] = float("inf")
        logits[2, 11] = -float("inf")
        weights, indices, fused_spy = self._run_actual_fused(
            logits, torch.zeros(256, device="cuda", dtype=torch.float32)
        )

        fused_spy.assert_called_once()
        expected_indices = torch.arange(6, device="cuda", dtype=torch.long)
        expected_weights = torch.full((6,), 1.0 / 6.0, device="cuda")
        for row in range(3):
            self.assertTrue(torch.equal(indices[row], expected_indices))
            self.assertTrue(torch.allclose(weights[row], expected_weights))
        self.assertTrue(torch.isfinite(weights).all())

    def test_actual_fused_finite_extremes_are_finite_and_not_fallback(self):
        f32 = torch.finfo(torch.float32)
        logits = torch.stack(
            (
                torch.linspace(-40.0, 40.0, 256, device="cuda"),
                torch.linspace(-3.0, 3.0, 256, device="cuda"),
            )
        )
        logits[0, 0] = f32.min
        logits[0, 255] = f32.max
        bias = torch.zeros(256, device="cuda", dtype=torch.float32)
        bias[0] = f32.min
        bias[253] = f32.max
        weights, indices, fused_spy = self._run_actual_fused(logits, bias)

        fused_spy.assert_called_once()
        fallback = torch.arange(6, device="cuda", dtype=torch.long)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(
            torch.allclose(weights.sum(dim=-1), torch.ones(2, device="cuda"))
        )
        # Both rows are deliberately non-uniform and must remain real fused
        # selections rather than being replaced by the canonical bad-row ids.
        self.assertFalse(torch.equal(indices[0], fallback))
        self.assertFalse(torch.equal(indices[1], fallback))

    def test_actual_fused_ranking_nonfinite_falls_back_whole_row(self):
        logits = torch.zeros((2, 256), device="cuda", dtype=torch.float32)
        bias = torch.zeros(256, device="cuda", dtype=torch.float32)
        bias[17] = float("nan")
        weights, indices, fused_spy = self._run_actual_fused(logits, bias)

        fused_spy.assert_called_once()
        expected_indices = torch.arange(6, device="cuda", dtype=torch.long)
        self.assertTrue(torch.equal(indices, expected_indices.expand(2, 6)))
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 1.0 / 6.0)))


if __name__ == "__main__":
    unittest.main()
