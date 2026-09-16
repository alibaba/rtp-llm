import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.weight_adapter import (
    adapt_split_moe_weights,
)
from rtp_llm.utils.model_weight import W


class SplitMoeWeightAdapterTest(unittest.TestCase):
    def test_maps_split_weights_to_canonical_keys(self):
        for shared in (0, 1, 3):
            with self.subTest(n_shared_experts=shared):
                experts, inter, hidden = 2, 4, 8
                gate = torch.full((experts, inter, hidden // 2), 1, dtype=torch.int8)
                up = torch.full_like(gate, 3)
                gate_scale = torch.full((experts, inter, 2), 11, dtype=torch.uint8)
                up_scale = torch.full_like(gate_scale, 13)
                weights = {
                    "router": torch.zeros(experts, hidden),
                    "router_bias": torch.zeros(experts),
                    "router_tid2eid": torch.zeros(16, 1, dtype=torch.int64),
                    "routed_gate": gate,
                    "routed_gate_scale": gate_scale,
                    "routed_up": up,
                    "routed_up_scale": up_scale,
                    "routed_down": torch.zeros(
                        experts, hidden, inter // 2, dtype=torch.int8
                    ),
                    "routed_down_scale": torch.zeros(
                        experts, hidden, 1, dtype=torch.uint8
                    ),
                }
                if shared:
                    weights.update(
                        {
                            "shared_gate_up": torch.zeros(2 * inter * shared, hidden),
                            "shared_gate_up_scale": torch.ones(1, 1),
                            "shared_down": torch.zeros(hidden, inter * shared),
                            "shared_down_scale": torch.ones(1, 1),
                        }
                    )
                names = {name: f"checkpoint.{name}" for name in weights}
                source = {names[name]: value for name, value in weights.items()}
                adapted = adapt_split_moe_weights(source, inter, shared, names)
                self.assertIs(adapted, source)
                torch.testing.assert_close(adapted[W.moe_w1][:, :inter], gate)
                torch.testing.assert_close(adapted[W.moe_w1][:, inter:], up)
                torch.testing.assert_close(adapted[W.moe_s1][:, :inter], gate_scale)
                torch.testing.assert_close(adapted[W.moe_s1][:, inter:], up_scale)
                for target, key in (
                    (W.moe_w2, "routed_down"),
                    (W.moe_s2, "routed_down_scale"),
                    (W.moe_gate, "router"),
                    (W.moe_gate_bias, "router_bias"),
                    (W.moe_gate_tid2eid, "router_tid2eid"),
                ):
                    self.assertIs(adapted[target], weights[key])
                self.assertFalse(set(names.values()) & adapted.keys())
                self.assertEqual(W.ffn_w13 in adapted, shared > 0)
                if shared:
                    self.assertIs(adapted[W.ffn_w13], weights["shared_gate_up"])
                    self.assertIs(adapted[W.ffn_w2], weights["shared_down"])

    def test_canonical_weights_are_not_repacked(self):
        tensor = torch.ones(2, 8, 4)
        weights = {W.moe_w1: tensor}
        self.assertIs(adapt_split_moe_weights(weights, 4, 0, {}), weights)
        self.assertIs(weights[W.moe_w1], tensor)


if __name__ == "__main__":
    unittest.main()
