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
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", ".."))


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

from rtp_llm.platforms.ppu.models.dsv4 import (
    gate as gate_module,
)
from rtp_llm.platforms.ppu.models.dsv4.impl import platform_provider as provider_module
from rtp_llm.platforms.ppu.models.dsv4.gate import (
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
    gate._stable_topk = False
    gate._bf16_fp32_linear = provider_module.build_dsv4_bf16_fp32_linear(
        lambda x, w: torch.nn.functional.linear(x, w).float(),
        platform_provider=provider_module.DefaultDsv4PlatformProvider(),
    )
    gate.dim = n_experts
    gate.topk = topk
    gate.score_func = "sqrtsoftplus"
    gate.route_scale = route_scale
    gate.hash = False
    gate.fuse_hash_gate = False
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




    def test_actual_eager_gate_scale_switch_preserves_default_api(self):
        gate = _gate_from_logits(8, 2, 2.5, torch.zeros(8, dtype=torch.float32))
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



_ACTUAL_FUSED_AVAILABLE = bool(torch.cuda.is_available() and gate_module._GATE_FUSED_OK)




if __name__ == "__main__":
    unittest.main()
