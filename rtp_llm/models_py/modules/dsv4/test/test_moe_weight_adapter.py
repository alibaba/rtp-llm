import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe_weight_adapter import adapt_dsv4_moe_weights
from rtp_llm.utils.model_weight import W


class Dsv4MoeWeightAdapterTest(unittest.TestCase):
    def test_maps_model_weights_to_canonical_keys(self):
        experts, inter, hidden = 2, 4, 8
        w1 = torch.full((experts, inter, hidden // 2), 1, dtype=torch.int8)
        w3 = torch.full((experts, inter, hidden // 2), 3, dtype=torch.int8)
        s1 = torch.full((experts, inter, hidden // 4), 11, dtype=torch.uint8)
        s3 = torch.full((experts, inter, hidden // 4), 13, dtype=torch.uint8)
        weights = {
            W.v4_router_w: torch.zeros(experts, hidden),
            W.v4_router_bias: torch.zeros(experts),
            W.v4_routed_w1_w: w1,
            W.v4_routed_w1_s: s1,
            W.v4_routed_w2_w: torch.zeros(
                experts, hidden, inter // 2, dtype=torch.int8
            ),
            W.v4_routed_w2_s: torch.zeros(experts, hidden, 1, dtype=torch.uint8),
            W.v4_routed_w3_w: w3,
            W.v4_routed_w3_s: s3,
            W.v4_shared_w13_w: torch.zeros(2 * inter, hidden),
            W.v4_shared_w13_s: torch.zeros(1, 1),
            W.v4_shared_w2_w: torch.zeros(hidden, inter),
            W.v4_shared_w2_s: torch.zeros(1, 1),
        }

        adapted = adapt_dsv4_moe_weights(weights, inter, n_shared_experts=1)

        self.assertTrue(torch.equal(adapted[W.moe_w1][:, :inter], w1))
        self.assertTrue(torch.equal(adapted[W.moe_w1][:, inter:], w3))
        self.assertTrue(torch.equal(adapted[W.moe_s1][:, :inter], s1))
        self.assertTrue(torch.equal(adapted[W.moe_s1][:, inter:], s3))
        self.assertIn(W.moe_gate, adapted)
        self.assertIn(W.moe_gate_bias, adapted)
        self.assertIn(W.ffn_w13, adapted)
        self.assertIn(W.ffn_w2, adapted)
        self.assertNotIn(W.v4_routed_w1_w, adapted)
        self.assertNotIn(W.v4_router_w, adapted)


if __name__ == "__main__":
    unittest.main()
