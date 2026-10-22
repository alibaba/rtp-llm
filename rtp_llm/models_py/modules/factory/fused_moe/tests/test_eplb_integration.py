"""Exercise the factory, routing boundary, and backend weight ownership on CPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeDataRouter,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.factory import FusedMoeFactory
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.utils.model_weight import W


class _Router(FusedMoeDataRouter):
    supports_gate_pack = True

    def prepare(self, a1, a1_scale, a2_scale, topk_weights, topk_ids):
        self.dispatched_ids = topk_ids
        return ExpertForwardPayload(
            expert_x=a1,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

    def finalize(self, payload, *args):
        return payload.fused_expert_output


class _Experts(FusedMoeExpertExecutor):
    supports_gate_pack = True
    execute_empty_inputs = True

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        self.expert_values = weights.pop(W.moe_w1)
        # Production FP4 backends consume their loader dictionaries as well.
        weights.clear()

    def get_eplb_weights(self):
        return (self.expert_values,)

    def execute(self, payload, **kwargs):
        local_experts = self.expert_values.size(0)
        ids = payload.expert_topk_ids - self.config.ep_rank * local_experts
        valid = (ids >= 0) & (ids < local_experts)
        values = self.expert_values[ids.clamp(min=0, max=local_experts - 1).long(), 0]
        values = values * valid * payload.expert_topk_weights
        return CombineForwardPayload(values.sum(-1, keepdim=True))


class _UnsupportedExperts(_Experts):
    get_eplb_weights = FusedMoeExpertExecutor.get_eplb_weights


def _make_layer(weights, executor_type=_Experts, *, ep_rank=0, ep_size=1, tp_size=1):
    config = SimpleNamespace(
        expert_num=2,
        physical_expert_num=weights[W.moe_w1].size(0) * ep_size,
        ep_rank=ep_rank,
        ep_size=ep_size,
        layer_id=3,
        tp_size=tp_size,
    )
    strategy = SimpleNamespace(
        strategy_name="cpu_reference",
        create_router=lambda cfg: _Router(cfg, None),
        create_executor=lambda cfg, tensors: executor_type(cfg, None, tensors),
    )
    registry = Mock()
    registry.get_strategy.return_value = strategy
    with patch.object(FusedMoeFactory, "_registry", registry):
        return FusedMoeFactory().create_fused_moe(config, weights)


def _weights():
    return {
        W.moe_w1: torch.tensor([[11.0], [37.0], [11.0], [37.0]]),
        W.log2phy: torch.tensor([[0, 2, -1], [1, 3, -1]], dtype=torch.int32),
        W.logic_expert_cnt: torch.tensor([2, 2], dtype=torch.int32),
    }


class EplbIntegrationTest(unittest.TestCase):
    def test_fp4_capacity_uses_physical_experts(self):
        config = Fp8Fp4MoeRuntimeConfig(
            layer_id=0,
            hidden_size=128,
            moe_inter_dim=128,
            expert_num=3,
            physical_expert_num=4,
            moe_k=2,
            n_shared_experts=0,
            swiglu_limit=0,
            ep_size=2,
            ep_rank=1,
            max_tokens_per_rank=16,
            moe_strategy="mega_moe",
        )
        self.assertEqual(config.n_routed_experts, 3)
        self.assertEqual(config.n_local_experts, 2)
        self.assertEqual(config.local_expert_start, 2)

    def test_factory_preserves_loader_mapping_and_actual_backend_weights(self):
        weights = _weights()
        mapping = weights[W.log2phy]
        counts = weights[W.logic_expert_cnt]
        actual_weights = weights[W.moe_w1]
        layer = _make_layer(weights)

        self.assertEqual(weights, {})
        self.assertIs(layer.eplb_layer.logical_to_physical, mapping)
        self.assertIs(layer.eplb_layer.logical_count, counts)
        self.assertIs(layer.eplb_layer.weights[0], actual_weights)
        self.assertEqual(layer.eplb_layer.layer_id, 3)
        self.assertFalse(layer.supports_gate_pack)

    def test_static_redundancy_routes_before_dispatch_and_preserves_output(self):
        layer = _make_layer(_weights())
        logical_ids = torch.tensor([[0, 1], [1, 0], [0, -1]], dtype=torch.int32)
        route_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4], [1.0, 0.0]])
        x = torch.zeros(3, 1)
        expected = torch.tensor([[30.5], [26.6], [11.0]])
        result = layer(x, route_weights, logical_ids, expert_load=None)
        torch.testing.assert_close(result, expected)
        self.assertEqual(layer.router.dispatched_ids.dtype, logical_ids.dtype)
        self.assertEqual(layer.router.dispatched_ids[-1, -1].item(), -1)
        self.assertFalse(torch.equal(layer.router.dispatched_ids, logical_ids))
        torch.testing.assert_close(
            logical_ids, torch.tensor([[0, 1], [1, 0], [0, -1]], dtype=torch.int32)
        )

    def test_next_forward_observes_in_place_layout_change(self):
        layer = _make_layer(_weights())
        ids = torch.tensor([[0, 1], [1, 0]])
        route_weights = torch.tensor([[0.25, 0.75], [0.6, 0.4]])
        x = torch.zeros(2, 1)
        expected = layer(x, route_weights, ids, expert_load=None).clone()
        state = layer.eplb_layer
        pointers = tuple(
            t.data_ptr() for t in (*state.weights, state.logical_to_physical)
        )
        # A completed migration publishes new physical contents and their map
        # in place. The FusedMoe boundary must read these live tensors again.
        state.weights[0].copy_(torch.tensor([[37.0], [11.0], [37.0], [11.0]]))
        state.logical_to_physical.copy_(torch.tensor([[1, 3, -1], [0, 2, -1]]))
        result = layer(x, route_weights, ids, expert_load=None)
        torch.testing.assert_close(result, expected)
        self.assertEqual(
            pointers,
            tuple(t.data_ptr() for t in (*state.weights, state.logical_to_physical)),
        )

    def test_tp_replicated_inputs_reduce_once_per_logical_route(self):
        layers = []
        for rank in range(2):
            weights = _weights()
            weights[W.moe_w1] = weights[W.moe_w1][rank * 2 : (rank + 1) * 2].clone()
            layers.append(_make_layer(weights, ep_rank=rank, ep_size=2, tp_size=2))

        # Static redundant layouts are active even without a balancing runtime.
        ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32)
        x = torch.zeros(4, 1)
        route_weights = torch.ones(4, 1)
        expected = torch.tensor([[11.0], [11.0], [37.0], [37.0]])
        for migrated in (False, True):
            with self.subTest(migrated=migrated):
                if migrated:
                    for layer in layers:
                        state = layer.eplb_layer
                        state.weights[0].copy_(torch.tensor([[37.0], [11.0]]))
                        state.logical_to_physical.copy_(
                            torch.tensor([[1, 3, -1], [0, 2, -1]])
                        )
                partials = [
                    layer(x, route_weights, ids, expert_load=None) for layer in layers
                ]
                torch.testing.assert_close(sum(partials), expected)
                torch.testing.assert_close(
                    layers[0].router.dispatched_ids, layers[1].router.dispatched_ids
                )

    def test_cp_dp_independent_inputs_can_rotate_replicas_by_rank(self):
        mapped = []
        for rank in range(2):
            weights = _weights()
            weights[W.moe_w1] = weights[W.moe_w1][rank * 2 : (rank + 1) * 2].clone()
            # MoEConfigAdapter exposes tp_size=1 for both CP and pure DP.
            layer = _make_layer(weights, ep_rank=rank, ep_size=2, tp_size=1)
            self.assertEqual(layer.eplb_layer.ep_rank, rank)
            mapped.append(layer.eplb_layer.map_ids(torch.tensor([[0]])))
        self.assertEqual([ids.item() for ids in mapped], [0, 2])

    def test_disabled_eplb_preserves_ids_and_gate_pack(self):
        layer = _make_layer({W.moe_w1: torch.tensor([[11.0], [37.0]])})
        ids = torch.tensor([[0, 1]])
        result = layer(
            torch.zeros(1, 1), torch.tensor([[0.25, 0.75]]), ids, expert_load=None
        )
        torch.testing.assert_close(result, torch.tensor([[30.5]]))
        self.assertIs(layer.router.dispatched_ids, ids)
        self.assertIsNone(layer.eplb_layer)
        self.assertTrue(layer.supports_gate_pack)

    def test_empty_input_uses_same_dispatch_boundary(self):
        layer = _make_layer(_weights())
        output = layer(
            torch.empty(0, 1),
            torch.empty(0, 2),
            torch.empty(0, 2, dtype=torch.int64),
            expert_load=None,
        )
        self.assertEqual(output.shape, (0, 1))
        self.assertEqual(layer.router.dispatched_ids.shape, (0, 2))

    def test_incomplete_mapping_fails_at_factory_boundary(self):
        weights = _weights()
        del weights[W.logic_expert_cnt]
        with self.assertRaisesRegex(ValueError, "both log2phy and logic_expert_cnt"):
            _make_layer(weights)

    def test_unsupported_backend_fails_before_serving(self):
        with self.assertRaisesRegex(NotImplementedError, "does not support EPLB"):
            _make_layer(_weights(), _UnsupportedExperts)


if __name__ == "__main__":
    unittest.main()
