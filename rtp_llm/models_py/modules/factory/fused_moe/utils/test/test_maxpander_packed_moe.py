import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import ExpertGatePayload
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe import (
    MegaMoeExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.maxpander import PackedMoeExecutorMixin


class MaxPanderPackedMoeTest(unittest.TestCase):
    def test_both_pack_paths_remap_once_before_launch(self):
        for executor_type in (MegaMoeExecutor, MegaMoeSEExecutor):
            for enabled in (False, True):
                for tokens in (0, 3):
                    with self.subTest(
                        executor=executor_type, enabled=enabled, tokens=tokens
                    ):
                        self.check_paths(executor_type, enabled, tokens)

    def check_paths(self, executor_type, enabled, tokens):
        executor = executor_type.__new__(executor_type)
        torch.nn.Module.__init__(executor)
        executor.cfg = SimpleNamespace(n_activated_experts=2, max_tokens_per_rank=4)
        executor._mega_y = torch.zeros((4, 128), dtype=torch.bfloat16)
        executor._mega_buf = SimpleNamespace(
            num_max_tokens_per_rank=4,
            x=torch.empty((4, 128)),
            x_sf=torch.empty((4, 1)),
            topk_idx=torch.empty((4, 4), dtype=torch.int64)[:, :2],
            topk_weights=torch.empty((4, 2)),
            shared_l1_acts_sf=torch.empty(1),
        )
        logical = torch.tensor([[0, 3]]).expand(tokens, 2).contiguous()
        weights = torch.full((tokens, 2), 0.25)
        inputs = torch.ones((tokens, 128), dtype=torch.bfloat16)
        events = []

        def pack(*args):
            events.append("pack")
            executor._mega_buf.topk_idx[:tokens].copy_(logical)
            executor._mega_buf.topk_weights[:tokens].copy_(weights)

        def remap(packed):
            events.append("map")
            torch.testing.assert_close(packed, logical)
            packed.add_(10)

        def launch(output, actual_tokens, device):
            events.append("launch")
            self.assertEqual(actual_tokens, tokens)
            torch.testing.assert_close(
                executor._mega_buf.topk_idx[:tokens], logical + (10 if enabled else 0)
            )
            torch.testing.assert_close(
                executor._mega_buf.topk_weights[:tokens], weights
            )

        executor._input_packer = SimpleNamespace(name="fused", pack=pack)
        executor._block_m = lambda token_count: 64
        executor._launch = launch
        executor.physical_router = (
            SimpleNamespace(remap_inplace=remap) if enabled else None
        )
        payload = ExpertGatePayload(
            scores=torch.zeros((tokens, 4)),
            topk=2,
            score_func="sqrtsoftplus",
            route_scale=1.0,
        )
        with patch.dict(os.environ, {"MEGA_MOE_INPUT_PACKER_IMPL": "optimized"}), patch(
            "rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack.fused_pack_mega_moe_gate_inputs",
            side_effect=lambda *args, **kwargs: pack(),
        ), patch(
            "rtp_llm.models_py.triton_kernels.moe.mega_moe_se_input_pack.stage_mega_moe_se_shared_l1_scales"
        ):
            self.assertTrue(executor.supports_gate_pack)
            for _ in range(2):
                events.clear()
                executor.forward(inputs, weights, logical)
                self.assertEqual(
                    events, ["pack", "map", "launch"] if enabled else ["pack", "launch"]
                )
                events.clear()
                executor.forward_gate_pack(inputs, payload)
                self.assertEqual(
                    events, ["pack", "map", "launch"] if enabled else ["pack", "launch"]
                )

    def test_v41_strategies_share_the_same_post_pack_tail(self):
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega import MegaMoEStrategy
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_fused import (
            MegaMoEFusedStrategy,
        )
        from rtp_llm.models_py.modules.dsv41.moe.strategies.mega_se import (
            MegaMoEStrategySE,
        )

        for strategy in (
            MegaMoeExecutor,
            MegaMoeSEExecutor,
            MegaMoEStrategy,
            MegaMoEStrategySE,
            MegaMoEFusedStrategy,
        ):
            self.assertIs(
                strategy._execute_packed_inputs,
                PackedMoeExecutorMixin._execute_packed_inputs,
            )


if __name__ == "__main__":
    unittest.main()
