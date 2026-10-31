import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertGatePayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.fp8_fp4_base import (
    Fp8Fp4ExecutorBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4 import (
    layer as layer_module,
)
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

    def __init__(self, events):
        self.events = events

    def forward_gate_pack(self, *, hidden_states, gate_payload, activation):
        self.events.append("fused_moe.forward_gate_pack")
        return torch.ones_like(hidden_states)

    def __call__(self, *, hidden_states, topk_weights, topk_ids, activation):
        self.events.append("fused_moe.forward")
        return torch.ones_like(hidden_states)


class _FakeNormTopP(torch.nn.Module):
    enabled = False

    def forward(self, x, topk_weights, topk_ids):
        return topk_weights, topk_ids


class _FakeSharedExecutor:
    def __init__(self, events):
        self.events = events
        self.x = None

    def start(self, shared_experts, x):
        self.events.append("shared.start")
        self.x = x

    def finish(self):
        self.events.append("shared.finish")
        return torch.ones_like(self.x)


class _Fp32Executor(Fp8Fp4ExecutorBase):
    def setup_weights(self, weights):
        pass

    def forward(self, x, topk_weights, topk_ids):
        return torch.ones_like(x, dtype=torch.float32)

    def forward_gate_pack(self, x, gate_payload):
        tokens = x.size(0)
        return (
            torch.ones_like(x, dtype=torch.float32),
            torch.ones((tokens, 1), dtype=torch.float32),
            torch.zeros((tokens, 1), dtype=torch.long),
        )


class Fp8Fp4MoeLayerSchedulingTest(unittest.TestCase):
    def _layer(self, events, gate_pack_enabled, with_shared=True):
        layer = Fp8Fp4MoeLayer.__new__(Fp8Fp4MoeLayer)
        torch.nn.Module.__init__(layer)
        layer.dim = 4
        layer.gate = _FakeGate(events, gate_pack_enabled)
        layer.norm_top_p = _FakeNormTopP()
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

    def test_top_p_accepts_fp8_per_block_local_routes(self):
        for strategy in (
            "fp8_per_block_no_dp",
            "fp8_per_block_no_dp_masked",
            "fp8_per_block_pure_cp",
        ):
            with self.subTest(strategy=strategy):
                with (
                    patch.object(layer_module, "Gate"),
                    patch.object(layer_module, "Dsv4NormTopP") as norm_top_p,
                    patch.object(layer_module, "FusedMoeFactory") as factory,
                ):
                    norm_top_p.return_value.enabled = True
                    factory.return_value.create_fused_moe.return_value.strategy_name = (
                        strategy
                    )
                    layer = Fp8Fp4MoeLayer(
                        layer_id=3,
                        dim=4,
                        moe_inter_dim=8,
                        n_routed_experts=4,
                        n_activated_experts=2,
                        n_shared_experts=0,
                        score_func="sigmoid",
                        route_scale=1.0,
                        swiglu_limit=1.0,
                        n_hash_layers=3,
                        vocab_size=8,
                        layer_weights={
                            W.moe_w1: torch.empty(
                                (4, 16, 4), dtype=torch.float8_e4m3fn
                            )
                        },
                        model_type="deepseek_v4",
                    )
                    runtime_config = (
                        factory.return_value.create_fused_moe.call_args.args[0]
                    )
                    self.assertTrue(runtime_config.norm_top_p_enabled)
                self.assertEqual(layer.strategy_name, strategy)


class Fp8Fp4ExecutorOutputDtypeTest(unittest.TestCase):
    def setUp(self):
        self.executor = _Fp32Executor.__new__(_Fp32Executor)
        torch.nn.Module.__init__(self.executor)
        self.x = torch.ones((2, 4), dtype=torch.bfloat16)

    def _execute(self, payload):
        return self.executor.execute(
            payload,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
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


if __name__ == "__main__":
    unittest.main()
