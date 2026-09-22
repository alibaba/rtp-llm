import unittest

import torch

from rtp_llm.models.kimi_k3.kimi_k3_weight import KimiK3WeightNames as K3W
from rtp_llm.models_py.modules.dsv4.moe.input_pack_options import mask_pack_routes
from rtp_llm.models_py.modules.kimi_k3.moe import KimiK3LatentMoE


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MoeResidualGpuTest(unittest.TestCase):
    def test_forward_keeps_route_validity_and_residual(self):
        module = KimiK3LatentMoE.__new__(KimiK3LatentMoE)
        torch.nn.Module.__init__(module)
        module.routed_norm = None
        module.weights = {
            K3W.MOE_ROUTED_DOWN: torch.eye(2, dtype=torch.bfloat16, device="cuda"),
            K3W.MOE_ROUTED_UP: torch.eye(2, dtype=torch.bfloat16, device="cuda"),
        }
        ids = torch.tensor([[2], [3], [4]], device="cuda")
        weights = torch.tensor([[0.25], [0.5], [0.75]], device="cuda")
        module._route = lambda _: (ids, weights)
        captured = {}

        def routed_expert(x, routed_ids, routed_weights, **pack_options):
            routed_ids, routed_weights = mask_pack_routes(
                routed_ids, routed_weights, **pack_options
            )
            captured.update(ids=routed_ids, weights=routed_weights)
            return x

        module._mega_expert_sum = routed_expert
        module._shared_expert_forward = torch.ones_like
        hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.bfloat16, device="cuda"
        )
        residual = torch.full_like(hidden, 4)
        residual_before = residual.clone()

        output = module(
            hidden,
            valid_token_count=2,
            valid_token_mask=torch.tensor([1, 0, 1], device="cuda"),
            residual=residual,
        )

        self.assertEqual(captured["ids"].tolist(), [[2], [0], [0]])
        self.assertEqual(captured["weights"].tolist(), [[0.25], [0.0], [0.0]])
        torch.testing.assert_close(output, (hidden + 1) + residual, rtol=0, atol=0)
        torch.testing.assert_close(residual, residual_before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
