import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
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
    Fp8Fp4MoeRuntimeConfig,
)


class DeepSeekBatchInvariantMXFP4Test(unittest.TestCase):
    def _config(self, **overrides):
        options = dict(
            layer_id=0,
            hidden_size=128,
            moe_inter_dim=128,
            expert_num=4,
            moe_k=2,
            n_shared_experts=1,
            swiglu_limit=5.0,
            ep_size=2,
            ep_rank=1,
            tp_size=2,
            tp_rank=1,
            world_size=2,
            world_rank=1,
            max_tokens_per_rank=32,
            moe_strategy="auto",
            batch_invariant=True,
            model_type="deepseek_v41",
            moe_w1_layout="gate_up",
        )
        return Fp8Fp4MoeRuntimeConfig(**(options | overrides))

    def test_registry_selects_replicated_rows_and_tp_reduction(self):
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers import (
            pure_tp_router,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
            CudaBatchInvariantMXFP4Strategy,
            CudaMegaMoeStrategy,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.strategy_registry import (
            StrategyRegistry,
        )

        registry = StrategyRegistry()
        registry.register(CudaMegaMoeStrategy())
        invariant = CudaBatchInvariantMXFP4Strategy()
        registry.register(invariant)
        config = self._config()
        self.assertEqual(config.moe_strategy, invariant.strategy_name)
        self.assertIs(registry.get_strategy(config), invariant)
        self.assertEqual((config.local_expert_start, config.local_expert_end), (2, 4))
        self.assertEqual((config.dp_size, config.dp_rank), (1, 0))
        router = invariant.create_router(config)
        local = torch.ones(1, 128, dtype=torch.bfloat16)
        merged = torch.full_like(local, 3)
        with patch.object(pure_tp_router, "all_reduce", return_value=merged) as reduce:
            result = router.finalize(
                CombineForwardPayload(local), None, None, False, None
            )
        self.assertIs(result, merged)
        reduce.assert_called_once_with(local, group=pure_tp_router.Group.TP)

        ordinary = self._config(batch_invariant=False)
        self.assertEqual(
            (ordinary.moe_strategy, ordinary.tp_size, ordinary.dp_size), ("auto", 1, 2)
        )
        self.assertFalse(invariant.can_handle(ordinary))

    def test_rejects_conflicting_strategy_and_topology(self):
        for options in (
            {"tp_rank": 0},
            {"tp_size": 1},
            {"world_size": 4},
            {"physical_expert_num": 6},
            {"moe_strategy": "mega_moe"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self._config(**options)

    def test_checkpoint_scales_and_e2m1_nibble_order(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires a Blackwell SM100-family GPU")
        from rtp_llm.models_py.determinism.quantized_linear import (
            BlockFp8Linear,
            mxfp4_linear,
        )
        from rtp_llm.models_py.modules.dsv41.utils import V41MXFP8Linear

        device = torch.device("cuda")
        x = torch.tensor([1, 2], dtype=torch.bfloat16, device=device).repeat(1, 64)
        code = torch.arange(16, dtype=torch.uint8, device=device)
        # Each byte holds the low nibble at even K and the high nibble at odd K.
        high = (code + 5) % 16
        packed = (
            (code | (high << 4))[:, None].expand(16, 64).contiguous().view(torch.int8)
        )
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=device,
        )
        scales = torch.tensor(
            [125, 127, 128, 129], device=device, dtype=torch.uint8
        ).repeat(16, 1)
        expected = 16 * (0.25 + 1 + 2 + 4) * (table + 2 * table[high.long()])
        for group in (32, 128):
            output = mxfp4_linear(
                x, packed, scales, group_size=group, out_dtype=torch.float32
            )
            self.assertTrue(torch.equal(output[0], expected))

        weight = torch.ones(65, 128, device=device).to(torch.float8_e4m3fn)
        exponents = scales[:3].clone()
        exponents[1] += 1
        mxfp8 = V41MXFP8Linear(
            weight, exponents.view(torch.float8_e8m0fnu), batch_invariant=True
        )
        # Each K group contributes 16*(1+2). The middle N group doubles its scales.
        expected = torch.full(
            (1, 65), 48 * (0.25 + 1 + 2 + 4), dtype=x.dtype, device=device
        )
        expected[:, 32:64] *= 2
        self.assertTrue(torch.equal(mxfp8(x), expected))
        fp8 = BlockFp8Linear(weight, torch.ones(1, 1, device=device))
        self.assertTrue(torch.equal(fp8(x), torch.full_like(expected, 192)))

    def test_native_experts_match_sparse_reference_across_permuted_rows(self):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            self.skipTest("requires a Blackwell SM100-family GPU")
        from rtp_llm.models_py.determinism.quantized_linear import mxfp4_linear
        from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
        from rtp_llm.utils.model_weight import W

        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(274)
        experts, width, rows = 2, 128, 19
        hidden = torch.randn(
            rows, width, generator=generator, device=device, dtype=torch.bfloat16
        )
        hidden[:, :64] *= 0.075

        def packed(n):
            return torch.randint(
                -128,
                128,
                (experts, n, width // 2),
                generator=generator,
                dtype=torch.int8,
                device=device,
            )

        def scales(n):
            return torch.randint(
                120,
                125,
                (experts, n, width // 32),
                generator=generator,
                dtype=torch.uint8,
                device=device,
            )

        weights = {
            W.moe_w1: packed(width * 2),
            W.moe_w2: packed(width),
            W.moe_s1: scales(width * 2),
            W.moe_s2: scales(width),
        }
        ids = torch.tensor([1, 0], device=device, dtype=torch.int32).repeat(rows, 1)
        routing = torch.tensor([0.375, 0.625], device=device).repeat(rows, 1)
        for model_type, group_size in (("deepseek_v4", 128), ("deepseek_v41", 32)):
            with self.subTest(model_type=model_type):
                config = self._config(
                    ep_size=1,
                    ep_rank=0,
                    tp_size=1,
                    tp_rank=0,
                    world_size=1,
                    world_rank=0,
                    expert_num=experts,
                    model_type=model_type,
                )
                moe = FusedMoeFactory().create_fused_moe(config, weights)
                result = moe(hidden, routing, ids, activation="silu")
                index = torch.tensor([18, 2, 9], device=device)
                subset = moe(
                    hidden[index], routing[index], ids[index], activation="silu"
                )
                self.assertTrue(torch.equal(subset, result[index]))
                # Apply each expert separately to one token. Include checkpoint gate/up
                # order, clamping, per-slot weighting and BF16 rounding of each route.
                routes = []
                for slot, expert in enumerate((1, 0)):
                    gate_up = mxfp4_linear(
                        hidden[:1],
                        weights[W.moe_w1][expert],
                        weights[W.moe_s1][expert],
                        group_size=group_size,
                    ).float()
                    gate, up = gate_up.chunk(2, dim=-1)
                    activated = torch.nn.functional.silu(gate.clamp(max=5)) * up.clamp(
                        -5, 5
                    )
                    down = mxfp4_linear(
                        activated.bfloat16(),
                        weights[W.moe_w2][expert],
                        weights[W.moe_s2][expert],
                        group_size=group_size,
                        out_dtype=torch.float32,
                    )
                    routes.append((down * routing[0, slot]).bfloat16().float())
                expected = (routes[0] + routes[1]).bfloat16()
                torch.testing.assert_close(result[:1], expected, atol=2e-3, rtol=0.015)


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
