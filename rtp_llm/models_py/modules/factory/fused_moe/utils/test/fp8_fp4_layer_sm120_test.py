"""SM120 cases split out of fp8_fp4_layer_test.py.
These cases live in this sibling file, leaving the original test file
untouched."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

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


class Dsv4GateSm120PolicyTest(unittest.TestCase):
    def _gate(self, *, hash_layer=False, model_type="deepseek_v4"):
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
            model_type=model_type,
        )

    def test_policy_is_model_and_exact_arch_scoped(self):
        dsv4_gate = self._gate()
        generic_gate = self._gate(model_type="generic_fp8_fp4_moe")
        x_cuda = SimpleNamespace(is_cuda=True, device=torch.device("cuda", 0))
        x_cpu = torch.ones((2, 8), dtype=torch.bfloat16)
        with patch.object(gate_module, "is_sm120", return_value=True):
            self.assertTrue(dsv4_gate._sm120_dsv4_policy(x_cuda))
            self.assertFalse(generic_gate._sm120_dsv4_policy(x_cuda))
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cpu))
            dsv4_gate._ced_row_projection = lambda x, w: x
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cuda))
            del dsv4_gate._ced_row_projection
        with patch.object(gate_module, "is_sm120", return_value=False):
            self.assertFalse(dsv4_gate._sm120_dsv4_policy(x_cuda))

    def test_ced_projection_precedes_padding(self):
        gate = self._gate()
        calls = []

        def projection(x, weight):
            calls.append(x.shape[0])
            return torch.zeros((x.shape[0], 4), dtype=x.dtype)

        gate._ced_row_projection = projection
        x = torch.ones((2, 8), dtype=torch.bfloat16)
        input_ids = torch.tensor([0, 1], dtype=torch.long)
        with (
            patch.object(gate_module, "_use_fused_gate", return_value=False),
            patch.dict(os.environ, {"MOE_GATE_FP32": "0"}),
        ):
            gate(x, input_ids)
        # CED received the original two rows, never the small-M padded batch,
        # and the SM120 policy stays disabled while it is installed.
        self.assertEqual(calls, [2])
        self.assertFalse(gate._sm120_dsv4_policy(x))


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
    """Placeholder installed by ``_layer``; each test replaces it with a real
    ``FusedMoe`` before forward, so it needs no call expectations."""

    def __init__(self, events):
        self.events = events


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


class Fp8Fp4MoeLayerSchedulingTestSm120(unittest.TestCase):
    """Sibling of ``Fp8Fp4MoeLayerSchedulingTest`` carrying the SM120 cases."""

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


class Fp8Fp4ExecutorOutputDtypeTestSm120(unittest.TestCase):
    """Sibling of ``Fp8Fp4ExecutorOutputDtypeTest`` carrying the SM120 cases."""

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

    def test_gate_pack_can_preserve_precision_for_layer_combine(self):
        self.executor.value = 1.003
        self.executor.preserve_output_dtype = True
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
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(
            output, torch.full_like(output, 1.003), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
