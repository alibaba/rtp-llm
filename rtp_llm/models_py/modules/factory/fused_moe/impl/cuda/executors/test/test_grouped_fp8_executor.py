"""The common MoE API must preserve Hopper's operands and accumulation precision."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors import grouped_fp8
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeRuntimeConfig,
)
from rtp_llm.utils.model_weight import W


class _Backend(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def setup_weights(self, weights):
        self.weights = weights

    def forward(self, x, topk_weights, topk_ids):
        return x.float() + 0.001


class GroupedFp8ExecutorContractTest(unittest.TestCase):
    def test_factory_operands_preserve_gate_up_order_and_block_scales(self):
        for layout in ("gate_up", "up_gate"):
            with self.subTest(layout=layout):
                cfg = Fp8Fp4MoeRuntimeConfig(
                    layer_id=0,
                    hidden_size=128,
                    moe_inter_dim=256,
                    expert_num=2,
                    moe_k=1,
                    n_shared_experts=0,
                    swiglu_limit=7.0,
                    ep_size=1,
                    ep_rank=0,
                    max_tokens_per_rank=8,
                    moe_strategy="grouped_fp8",
                    expert_dtype="fp8",
                    moe_w1_layout=layout,
                )
                gate = torch.ones(2, 256, 128).to(torch.float8_e4m3fn)
                up = (torch.ones(2, 256, 128) * 3).to(torch.float8_e4m3fn)
                gs = torch.full((2, 2, 1), 127, dtype=torch.uint8)
                us = torch.full_like(gs, 130)
                first, second = (gate, up) if layout == "gate_up" else (up, gate)
                fs, ss = (gs, us) if layout == "gate_up" else (us, gs)
                weights = {
                    W.moe_w1: torch.cat((first.float(), second.float()), dim=1).to(
                        gate.dtype
                    ),
                    W.moe_s1: torch.cat((fs, ss), dim=1),
                    W.moe_w2: gate.transpose(1, 2),
                    W.moe_s2: gs.transpose(1, 2),
                }
                with patch.object(grouped_fp8, "GroupedFP8Strategy", _Backend):
                    executor = grouped_fp8.GroupedFp8Executor(
                        cfg, FusedMoEQuantConfig(), weights
                    )
                operands = executor.grouped.weights
                torch.testing.assert_close(
                    operands[W.v4_routed_w1_w].float(), gate.float()
                )
                torch.testing.assert_close(
                    operands[W.v4_routed_w3_w].float(), up.float()
                )
                torch.testing.assert_close(operands[W.v4_routed_w1_s], gs)
                torch.testing.assert_close(operands[W.v4_routed_w3_s], us)
                x = torch.ones(1, 128, dtype=torch.bfloat16)
                payload = ExpertForwardPayload(
                    expert_x=x,
                    expert_x_origin_dtype=x.dtype,
                    expert_topk_weights=torch.ones(1, 1),
                    expert_topk_ids=torch.zeros(1, 1, dtype=torch.int64),
                )
                output = executor.execute(
                    payload, "silu", None, None, False, None
                ).fused_expert_output
                self.assertEqual(output.dtype, torch.float32)
                torch.testing.assert_close(output, x.float() + 0.001)


if __name__ == "__main__":
    unittest.main()
