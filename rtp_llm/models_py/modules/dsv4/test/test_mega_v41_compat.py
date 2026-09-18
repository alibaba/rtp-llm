"""Real MegaMoE coverage for V4.1's native and legacy padded kernel layouts."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import MegaMoEStrategy
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "requires SM100 CUDA")
class V41MegaCompatibilityTest(unittest.TestCase):
    # Set for a package-specific regression run; normal CI supports either API.
    expected_kernel_intermediate = None

    @staticmethod
    def _weights(experts=8, hidden=5120, intermediate=2304):
        weights = {}
        for weight_key, scale_key, n, k in (
            (W.v4_routed_w1_w, W.v4_routed_w1_s, intermediate, hidden),
            (W.v4_routed_w3_w, W.v4_routed_w3_s, intermediate, hidden),
            (W.v4_routed_w2_w, W.v4_routed_w2_s, hidden, intermediate),
        ):
            weights[weight_key] = torch.zeros(
                (experts, n, k // 2), dtype=torch.int8, device="cuda"
            )
            weights[scale_key] = torch.full(
                (experts, n, k // 32), 127, dtype=torch.uint8, device="cuda"
            ).view(torch.float8_e8m0fnu)
        # Two independent SwiGLU neurons, including the last checkpoint row.
        # FP4 nibble 2 encodes +1. The rest of each matrix is exactly zero.
        weights[W.v4_routed_w1_w][:, 0, 0] = 2
        weights[W.v4_routed_w3_w][:, 0, 0] = 2 << 4
        weights[W.v4_routed_w1_w][:, -1, 1] = 2
        weights[W.v4_routed_w3_w][:, -1, 1] = 2 << 4
        weights[W.v4_routed_w2_w][:, 0, 0] = 2
        weights[W.v4_routed_w2_w][:, 1, -1] = 2 << 4
        return weights

    @torch.inference_mode()
    def test_original_topk_and_verify_width_survive_layout_and_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dist.init_process_group(
                "gloo", init_method="file://" + tmpdir + "/init", rank=0, world_size=1
            )
            try:
                config = MoeCfg(
                    layer_id=0,
                    dim=5120,
                    moe_inter_dim=2304,
                    n_routed_experts=8,
                    n_activated_experts=6,
                    swiglu_limit=10.0,
                    ep_size=1,
                    ep_rank=0,
                    n_local_experts=8,
                    local_expert_start=0,
                    local_expert_end=8,
                    max_tokens_per_rank=32,
                    shared_fp8_block_size=32,
                )
                for topk in (6, 3):
                    with self.subTest(topk=topk), patch.dict(
                        os.environ, {"WARM_UP": "0"}
                    ):
                        strategy = MegaMoEStrategy(
                            replace(config, n_activated_experts=topk)
                        )
                        strategy.setup_weights(self._weights())
                        strategy.setup_runtime()
                        if self.expected_kernel_intermediate is None:
                            self.assertIn(
                                strategy._mega_buf.intermediate_hidden, (2304, 2560)
                            )
                        else:
                            self.assertEqual(
                                strategy._mega_buf.intermediate_hidden,
                                self.expected_kernel_intermediate,
                            )
                        self.assertEqual(strategy._mega_buf.num_topk, topk)
                        x = torch.zeros((6, 5120), dtype=torch.bfloat16, device="cuda")
                        x[:, :4] = torch.tensor([1, 2, -1, 4], device="cuda")
                        weights = torch.full((6, topk), 1 / topk, device="cuda")
                        indices = (
                            torch.arange(topk, device="cuda").expand(6, -1).contiguous()
                        )
                        for _ in range(3):
                            strategy(x, weights, indices)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            result = strategy(x, weights, indices)
                        for scale in (1.0, 0.5, -1.0):
                            x[:, :4] = (
                                torch.tensor([1, 2, -1, 4], device="cuda") * scale
                            )
                            graph.replay()
                            torch.cuda.synchronize()
                            reference = torch.zeros_like(x).float()
                            reference[:, 0] = F.silu(x[:, 0].float()) * x[:, 1].float()
                            reference[:, 1] = F.silu(x[:, 2].float()) * x[:, 3].float()
                            torch.testing.assert_close(
                                result.float(), reference, rtol=0.05, atol=0.01
                            )
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
