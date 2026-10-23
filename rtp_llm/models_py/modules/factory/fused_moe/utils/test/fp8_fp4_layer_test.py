import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertGatePayload,
    FusedMoe,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.fp8_fp4_router import (
    Fp8Fp4Router,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    gate as gate_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    layer as layer_module,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.gate import Gate
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.utils.model_weight import W


class _FakeGate:
    def __init__(self, events, gate_pack_enabled):
        self.events = events
        self.gate_pack_enabled = gate_pack_enabled

    def can_prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.can_prepare")
        return self.gate_pack_enabled

    def prepare_gate_payload(self, x, input_ids):
        self.events.append("gate.prepare")
        return object()

    def __call__(self, x, input_ids):
        self.events.append("gate.forward")
        tokens = x.size(0)
        return (
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class _FakeFusedMoe:
    supports_gate_pack = True
    fused_experts = SimpleNamespace(preserve_output_dtype=True)

    def __init__(self, events):
        self.events = events

    def forward_gate_pack(
        self, *, hidden_states, gate_payload, activation, extra_expert_args
    ):
        assert extra_expert_args == {"preserve_output_dtype": True}
        self.events.append("fused_moe.forward_gate_pack")
        return torch.ones_like(hidden_states)

    def __call__(
        self, *, hidden_states, topk_weights, topk_ids, activation, extra_expert_args
    ):
        assert extra_expert_args == {"preserve_output_dtype": True}
        self.events.append("fused_moe.forward")
        return torch.ones_like(hidden_states)


class _FakeSharedExecutor:
    def __init__(self, events):
        self.events = events
        self.x = None

    def start(self, shared_experts, x):
        self.events.append("shared.start")
        self.x = x

    def finish(self):
        self.events.append("shared.finish")
        return torch.full_like(self.x, getattr(self, "value", 1.0))


class _Fp32Executor(Fp8Fp4ExecutorBase):
    def setup_weights(self, weights):
        pass

    def forward(self, x, topk_weights, topk_ids):
        return torch.full_like(x, getattr(self, "value", 1.0), dtype=torch.float32)

    def forward_gate_pack(self, x, gate_payload):
        tokens = x.size(0)
        return (
            torch.full_like(x, getattr(self, "value", 1.0), dtype=torch.float32),
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class Fp8Fp4MoeLayerSchedulingTest(unittest.TestCase):

    def test_shared_expert_sum_rounds_only_after_fp32_routed_output(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(executor)
        executor.value = 1.003
        executor.preserve_output_dtype = True
        layer.fused_moe = FusedMoe(Fp8Fp4Router.__new__(Fp8Fp4Router), executor, 1)
        layer._shared_executor.value = 2**-8
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        with patch.object(
            layer_module, "combine_routed_and_shared", side_effect=self._combine(events)
        ):
            output = layer(x, None)
        expected = (
            torch.full_like(x, executor.value, dtype=torch.float32) + 2**-8
        ).to(x.dtype)
        early_rounded = (torch.full_like(x, executor.value).float() + 2**-8).to(x.dtype)
        self.assertFalse(torch.equal(expected, early_rounded))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_other_executors_keep_routed_output_rounding(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(executor)
        executor.value = 1.003
        self.assertFalse(executor.preserve_output_dtype)
        layer.fused_moe = FusedMoe(Fp8Fp4Router.__new__(Fp8Fp4Router), executor, 1)
        layer._shared_executor.value = 2**-8
        x = torch.ones((2, 4), dtype=torch.bfloat16)
        with patch.object(
            layer_module, "combine_routed_and_shared", side_effect=self._combine(events)
        ):
            output = layer(x, None)
        expected = (torch.full_like(x, executor.value).float() + 2**-8).to(x.dtype)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def _layer(self, events, gate_pack_enabled, with_shared=True):
        layer = Fp8Fp4MoeLayer.__new__(Fp8Fp4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.dim = 4
        layer.gate = _FakeGate(events, gate_pack_enabled)
        layer.fused_moe = _FakeFusedMoe(events)
        layer.shared_experts = object() if with_shared else None
        layer._shared_executor = _FakeSharedExecutor(events) if with_shared else None
        return layer

    @staticmethod
    def _combine(events):
        def combine(routed, shared, out_dtype, out=None):
            events.append("combine")
            result = (routed.float() + shared.float()).to(out_dtype)
            if out is None:
                return result
            out.copy_(result)
            return out

        return combine

    def test_gate_pack_starts_shared_expert_before_gate(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            output = layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "shared.start",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
                "shared.finish",
                "combine",
            ],
        )
        torch.testing.assert_close(output, torch.full_like(x, 2))

    def test_ordinary_gate_stays_before_shared_expert(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        with patch.object(
            layer_module,
            "combine_routed_and_shared",
            side_effect=self._combine(events),
        ):
            layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "gate.forward",
                "shared.start",
                "fused_moe.forward",
                "shared.finish",
                "combine",
            ],
        )

    def test_fused_shared_expert_keeps_gate_first(self):
        events = []
        layer = self._layer(events, gate_pack_enabled=True, with_shared=False)
        x = torch.ones((2, 4), dtype=torch.bfloat16)

        layer(x, None)

        self.assertEqual(
            events,
            [
                "gate.can_prepare",
                "gate.prepare",
                "fused_moe.forward_gate_pack",
            ],
        )


class Fp8Fp4ExecutorOutputDtypeTest(unittest.TestCase):
    def setUp(self):
        self.executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(self.executor)
        self.x = torch.ones((2, 4), dtype=torch.bfloat16)

    def _execute(self, payload, extra_expert_args=None):
        return self.executor.execute(
            payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=extra_expert_args,
        ).fused_expert_output

    def test_ordinary_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                expert_topk_weights=torch.ones((2, 1), dtype=torch.float32),
                expert_topk_ids=torch.zeros((2, 1), dtype=torch.long),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual((output + self.x).dtype, torch.bfloat16)

    def test_gate_pack_output_restores_router_input_dtype(self):
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                gate_payload=ExpertGatePayload(
                    scores=torch.ones((2, 1), dtype=torch.float32),
                    topk=1,
                    score_func="sigmoid",
                    route_scale=1.0,
                ),
            )
        )

        self.assertEqual(output.dtype, torch.bfloat16)

    def test_gate_pack_can_preserve_precision_for_layer_combine(self):
        self.executor.value = 1.003
        output = self._execute(
            ExpertForwardPayload(
                expert_x=self.x,
                expert_x_origin_dtype=self.x.dtype,
                gate_payload=ExpertGatePayload(
                    scores=torch.ones((2, 1), dtype=torch.float32),
                    topk=1,
                    score_func="sigmoid",
                    route_scale=1.0,
                ),
            ),
            extra_expert_args={"preserve_output_dtype": True},
        )
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output, torch.full_like(output, 1.003), rtol=0, atol=0
        )


@contextmanager
def _gate_env(dsv4_gate_output_fp32=None):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("MOE_GATE_FP32", "DSV4_GATE_OUTPUT_FP32")
    }
    env["MOE_GATE_FP32"] = "0"
    if dsv4_gate_output_fp32 is not None:
        env["DSV4_GATE_OUTPUT_FP32"] = dsv4_gate_output_fp32
    with patch.dict(os.environ, env, clear=True):
        yield


class Dsv4GateFp32OutputEnvTest(unittest.TestCase):
    def _gate(self, *, hash_layer=False):
        if hash_layer:
            layer_weights = {
                W.moe_gate: torch.zeros((4, 8), dtype=torch.float32),
                W.moe_gate_tid2eid: torch.tensor(
                    [[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long
                ),
            }
        else:
            layer_weights = {
                W.moe_gate: torch.zeros((4, 8), dtype=torch.float32),
                W.moe_gate_bias: torch.zeros((4,), dtype=torch.float32),
            }
        return Gate(
            layer_id=0,
            dim=8,
            n_routed_experts=4,
            n_activated_experts=2,
            score_func="sqrtsoftplus",
            n_hash_layers=1 if hash_layer else 0,
            vocab_size=4 if hash_layer else 0,
            layer_weights=layer_weights,
        )

    def test_gate_pack_skipped_when_fp32_output_requested(self):
        gate = self._gate()
        x = torch.ones((2, 8), dtype=torch.bfloat16)
        with patch.object(gate_module, "_use_fused_gate", return_value=True):
            with _gate_env("1"):
                self.assertFalse(gate.can_prepare_gate_payload(x))
            with _gate_env():
                self.assertTrue(gate.can_prepare_gate_payload(x))

    def test_fused_hash_gate_skipped_when_fp32_output_requested(self):
        gate = self._gate(hash_layer=True)
        gate.fuse_hash_gate = True
        x = torch.ones((2, 8), dtype=torch.bfloat16)
        input_ids = torch.tensor([0, 1], dtype=torch.long)
        fused = MagicMock(
            return_value=(torch.ones((2, 2)), torch.zeros((2, 2), dtype=torch.long))
        )
        with (
            patch.object(gate_module, "_use_fused_gate", return_value=True),
            patch.object(gate_module, "fused_sqrtsoftplus_hash_gate", fused),
            patch.object(gate, "_bf16_scores", return_value=torch.zeros((2, 4))),
        ):
            with _gate_env("1"):
                weights, indices = gate(x, input_ids)
            fused.assert_not_called()
            self.assertEqual(weights.dtype, torch.float32)
            self.assertTrue(
                torch.equal(indices, torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
            )
            with _gate_env():
                gate(x, input_ids)
            fused.assert_called_once()


if __name__ == "__main__":
    unittest.main()
