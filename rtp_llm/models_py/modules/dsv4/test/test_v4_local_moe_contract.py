"""CPU oracles for V4 routed Expert and LocalLoop numerical ordering."""

from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "rtp_llm.models_py.kernels",
    os.path.join(_REPO, "rtp_llm", "models_py", "kernels"),
)
_stub_package(
    "rtp_llm.models_py.kernels.cuda",
    os.path.join(_REPO, "rtp_llm", "models_py", "kernels", "cuda"),
)

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    expert as expert_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.expert import Expert
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import (
    local_loop as local_loop_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.local_loop import (
    LocalLoopExecutor as LocalLoopStrategy,
)


def _reference_silu_mul(
    gate: torch.Tensor, up: torch.Tensor, *, clamp_limit: float
) -> torch.Tensor:
    if clamp_limit > 0:
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return F.silu(gate) * up


class _QuantizingW2(nn.Module):
    """Coarse activation quantizer used to expose pre/post-W2 weighting."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    @staticmethod
    def quantize(x: torch.Tensor) -> torch.Tensor:
        return torch.round(x.float() * 4.0) / 4.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.inputs.append(x.detach().clone())
        return self.quantize(x)


def _manual_expert(w2: nn.Module, swiglu_limit: float = 10.0) -> Expert:
    expert = Expert.__new__(Expert)
    nn.Module.__init__(expert)
    expert._uses_fp8_linear = False
    expert.w1 = nn.Identity()
    expert.w2 = w2
    expert.w3 = nn.Identity()
    expert.swiglu_limit = swiglu_limit
    return expert


class ExpertPostW2WeightContractTest(unittest.TestCase):
    def test_quantized_w2_input_is_independent_of_router_weight(self):
        spy = _QuantizingW2()
        expert = _manual_expert(spy)
        x = torch.tensor([[0.8, 0.9]], dtype=torch.float32)
        weight_a = torch.tensor([[0.25]], dtype=torch.float32)
        weight_b = torch.tensor([[0.75]], dtype=torch.float32)

        with mock.patch.object(
            expert_module, "require_silu_mul_split", return_value=_reference_silu_mul
        ):
            unweighted = expert(x)
            routed_a = expert(x, weight_a)
            routed_b = expert(x, weight_b)

        self.assertEqual(len(spy.inputs), 3)
        self.assertTrue(torch.equal(spy.inputs[0], spy.inputs[1]))
        self.assertTrue(torch.equal(spy.inputs[0], spy.inputs[2]))
        self.assertTrue(torch.equal(routed_a, unweighted.float() * weight_a))
        self.assertTrue(torch.equal(routed_b, unweighted.float() * weight_b))

        activation = _reference_silu_mul(x, x, clamp_limit=10.0)
        legacy_pre_w2 = spy.quantize(activation * weight_a)
        self.assertFalse(torch.equal(routed_a, legacy_pre_w2))

    def test_v4_swiglu_clamp10_matches_independent_formula(self):
        expert = _manual_expert(nn.Identity(), swiglu_limit=10.0)
        x = torch.tensor([[12.0, -12.0]], dtype=torch.float32)
        seen_limits: list[float] = []

        def fused_spy(gate, up, *, clamp_limit):
            seen_limits.append(float(clamp_limit))
            return _reference_silu_mul(gate, up, clamp_limit=clamp_limit)

        with mock.patch.object(
            expert_module, "require_silu_mul_split", return_value=fused_spy
        ):
            actual = expert(x)

        expected_gate = torch.clamp(x, max=10.0)
        expected_up = torch.clamp(x, min=-10.0, max=10.0)
        expected = F.silu(expected_gate) * expected_up
        self.assertEqual(seen_limits, [10.0])
        self.assertTrue(torch.equal(actual, expected))


class _OracleExpert(nn.Module):
    def __init__(self, expert_id: int, dim: int) -> None:
        super().__init__()
        self.expert_id = expert_id
        self.dim = dim

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        base = torch.full(
            (x.size(0), self.dim),
            float(self.expert_id + 1),
            dtype=torch.float32,
            device=x.device,
        )
        return base * weights


def _local_loop_oracle(
    indices: torch.Tensor, weights: torch.Tensor, dim: int
) -> torch.Tensor:
    result = torch.zeros((indices.size(0), dim), dtype=torch.float32)
    for token in range(indices.size(0)):
        for slot in range(indices.size(1)):
            result[token] += weights[token, slot] * float(indices[token, slot] + 1)
    return result


class LocalLoopE8Top2OccupancyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dim = 3
        self.strategy = LocalLoopStrategy.__new__(LocalLoopStrategy)
        nn.Module.__init__(self.strategy)
        self.strategy.experts = nn.ModuleList(
            [_OracleExpert(expert_id, self.dim) for expert_id in range(8)]
        )

    def _check(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        x = torch.zeros((indices.size(0), self.dim), dtype=torch.float32)
        eager_actual = torch.zeros_like(x)
        graph_actual = torch.zeros_like(x)
        self.strategy._forward_eager(x, weights, indices, eager_actual, 0, 8)
        self.strategy._forward_graph_safe(x, weights, indices, graph_actual, 0, 8)
        expected = _local_loop_oracle(indices, weights, self.dim)
        self.assertTrue(torch.allclose(eager_actual, expected, atol=0.0, rtol=0.0))
        self.assertTrue(torch.allclose(graph_actual, expected, atol=0.0, rtol=0.0))

    def test_zero_one_hot_and_all_expert_occupancy(self):
        cases = {
            "zero_tokens": (
                torch.empty((0, 2), dtype=torch.long),
                torch.empty((0, 2), dtype=torch.float32),
            ),
            "one_token": (
                torch.tensor([[3, 4]], dtype=torch.long),
                torch.tensor([[0.25, 0.75]], dtype=torch.float32),
            ),
            "duplicate_eid_accumulates": (
                torch.tensor([[3, 3]], dtype=torch.long),
                torch.tensor([[0.25, 0.75]], dtype=torch.float32),
            ),
            "hot_expert": (
                torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=torch.long),
                torch.tensor(
                    [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]],
                    dtype=torch.float32,
                ),
            ),
            "all_experts": (
                torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], dtype=torch.long),
                torch.tensor([[0.5, 0.5]] * 4, dtype=torch.float32),
            ),
        }
        for name, (indices, weights) in cases.items():
            with self.subTest(name=name):
                self._check(indices, weights)


class LocalLoopRoutedStorageTest(unittest.TestCase):
    def test_canonical_fp4_weights_are_consumed_without_payload_copies(self):
        from rtp_llm.utils.model_weight import W

        cfg = SimpleNamespace(
            dim=128,
            moe_inter_dim=64,
            moe_w1_layout="gate_up",
            n_local_experts=2,
            n_routed_experts=2,
            local_expert_start=0,
            local_expert_end=2,
            swiglu_limit=10,
            tp_size=1,
        )
        w13 = torch.zeros(2, 128, 64, dtype=torch.int8)
        weights = {
            W.moe_w1: w13,
            W.moe_s1: torch.ones(2, 128, 4),
            W.moe_w2: torch.zeros(2, 128, 32, dtype=torch.int8),
            W.moe_s2: torch.ones(2, 128, 2),
        }
        seen = []

        class FakeExpert(nn.Module):
            def __init__(self, *_args, expert_weights, **_kwargs):
                super().__init__()
                seen.append(expert_weights)

        strategy = LocalLoopStrategy.__new__(LocalLoopStrategy)
        nn.Module.__init__(strategy)
        strategy.cfg = cfg
        with mock.patch.object(
            local_loop_module, "Expert", FakeExpert
        ), mock.patch.object(
            local_loop_module,
            "prepare_fp4_weight_scale_for_deepgemm",
            side_effect=lambda s, *args: s.to(torch.int32),
        ):
            strategy.setup_weights(weights)
        self.assertEqual(strategy._routed_storage, "fp4")
        self.assertEqual(strategy.routed_tp_size, 1)
        self.assertEqual(weights, {})
        self.assertEqual(len(seen), 2)
        self.assertEqual(
            seen[0]["w1_w"].untyped_storage().data_ptr(),
            w13.untyped_storage().data_ptr(),
        )


class LocalLoopFastPathQuantizationSpyTest(unittest.TestCase):
    def _strategy(self) -> LocalLoopStrategy:
        strategy = LocalLoopStrategy.__new__(LocalLoopStrategy)
        nn.Module.__init__(strategy)
        strategy.cfg = SimpleNamespace(
            dim=2, moe_inter_dim=3, swiglu_limit=10.0, n_routed_experts=1
        )
        strategy._W1_w = torch.ones((1, 1), dtype=torch.int8)
        strategy._W2_w = torch.full((1, 1), 2, dtype=torch.int8)
        strategy._W3_w = torch.full((1, 1), 3, dtype=torch.int8)
        scale = torch.ones((1, 1, 1), dtype=torch.int32)
        strategy._W1_s_gemm_t = scale
        strategy._W2_s_gemm_t = scale
        strategy._W3_s_gemm_t = scale
        return strategy

    def _fake_kernel_modules(
        self, quant_inputs: list[torch.Tensor]
    ) -> dict[str, types.ModuleType]:
        deepgemm = types.ModuleType("rtp_llm.models_py.kernels.cuda.deepgemm_wrapper")
        fp8_kernel = types.ModuleType("rtp_llm.models_py.kernels.cuda.fp8_kernel")

        def fake_quant(x, **_kwargs):
            quant_inputs.append(x.detach().clone())
            quantized = (torch.round(x.float() * 4.0) / 4.0).to(x.dtype)
            return quantized, torch.ones((1, 1), dtype=torch.float32)

        def fake_gemm(a_pair, b_pair, out, **_kwargs):
            role = int(b_pair[0].reshape(-1)[0].item())
            if role == 1:
                out.fill_(0.8)
            elif role == 3:
                out.fill_(0.9)
            else:
                source = a_pair[0].float()
                out.copy_((source[:, : out.size(1)] * 2.0).to(out.dtype))

        deepgemm.fp8_fp4_gemm_nt = fake_gemm
        fp8_kernel.sgl_per_token_group_quant_fp8 = fake_quant
        return {
            deepgemm.__name__: deepgemm,
            fp8_kernel.__name__: fp8_kernel,
        }

    def _run(
        self, method_name: str, n_tokens: int, route_weight: float
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        strategy = self._strategy()
        quant_inputs: list[torch.Tensor] = []
        x = torch.full((n_tokens, 2), 0.5, dtype=torch.bfloat16)
        weights = torch.full((n_tokens, 1), route_weight, dtype=torch.float32)
        indices = torch.zeros((n_tokens, 1), dtype=torch.long)
        y = torch.zeros((n_tokens, 2), dtype=torch.float32)
        modules = self._fake_kernel_modules(quant_inputs)
        with mock.patch.dict(sys.modules, modules), mock.patch.object(
            local_loop_module,
            "require_silu_mul_split",
            return_value=_reference_silu_mul,
        ):
            getattr(strategy, method_name)(x, weights, indices, y)
        return y, quant_inputs

    def test_bs1_and_bsn_w2_quant_inputs_ignore_router_weight(self):
        for method_name, n_tokens in (
            ("_forward_topk_bs1", 1),
            ("_forward_topk_bsN", 2),
        ):
            with self.subTest(method=method_name):
                y_a, inputs_a = self._run(method_name, n_tokens, 0.25)
                y_b, inputs_b = self._run(method_name, n_tokens, 0.75)
                # Quant calls alternate input-x then W2 activation for each
                # token. Every W2 activation is independent of router weight.
                w2_inputs_a = inputs_a[1::2]
                w2_inputs_b = inputs_b[1::2]
                self.assertEqual(len(w2_inputs_a), n_tokens)
                for actual_a, actual_b in zip(w2_inputs_a, w2_inputs_b):
                    self.assertTrue(torch.equal(actual_a, actual_b))
                # The only route-weight effect is on W2's output.
                self.assertTrue(torch.equal(y_b, y_a * 3.0))


if __name__ == "__main__":
    unittest.main()
