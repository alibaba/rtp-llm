"""Hopper FP8 routed experts under the common fused-MoE contract."""

from typing import Any, Dict

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp8 import GroupedFP8Strategy
from rtp_llm.utils.model_weight import W

from .fp8_fp4_base import Fp8Fp4ExecutorBase, split_moe_w13_gate_up


class GroupedFp8Executor(Fp8Fp4ExecutorBase):
    @classmethod
    def check_conditions(cls, checker: Any, config) -> None:
        super().check_conditions(checker, config)
        checker.check(getattr(config, "expert_dtype", "fp4") == "fp8")
        checker.check(GroupedFP8Strategy.can_handle(config))

    def setup_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        cfg = self.cfg
        w13 = weights.pop(W.moe_w1)
        s13 = weights.pop(W.moe_s1)
        gate, up = split_moe_w13_gate_up(w13, cfg.moe_inter_dim, cfg.moe_w1_layout)
        gate_scale, up_scale = split_moe_w13_gate_up(
            s13, cfg.moe_inter_dim // 128, cfg.moe_w1_layout
        )
        self.grouped = GroupedFP8Strategy(cfg)
        self.grouped.setup_weights(
            {
                W.v4_routed_w1_w: gate,
                W.v4_routed_w3_w: up,
                W.v4_routed_w1_s: gate_scale,
                W.v4_routed_w3_s: up_scale,
                W.v4_routed_w2_w: weights.pop(W.moe_w2),
                W.v4_routed_w2_s: weights.pop(W.moe_s2),
            }
        )

    @staticmethod
    def _restore_output_dtype(payload, output: torch.Tensor) -> torch.Tensor:
        # Keep the routed sum in FP32 until the shared expert is added, matching
        # the original Hopper path's single BF16 rounding at the layer boundary.
        return output

    def forward(self, x, topk_weights, topk_ids) -> torch.Tensor:
        return self.grouped(x, topk_weights, topk_ids)
