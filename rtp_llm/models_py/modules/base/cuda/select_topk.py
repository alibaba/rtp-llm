import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig, fake_balance_expert: bool, dp_rank: int):
        super().__init__()
        self.config = config
        self.select_topk_op = compute_ops.SelectTopkOp(
            self.config, fake_balance_expert, dp_rank
        )

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        # EPLB parameters (log2phy, logic_expert_cnt, phy_exp_num, ep_rank) are no longer needed here
        # as log2phy conversion is now handled in Python after select_topk call
        self.select_topk_op.forward(
            router_logits_fp32,
            topk_ids,
            topk_weights,
        )


class GroupTopK(nn.Module):
    def __init__(self):
        super().__init__()
        self.group_topk_op = compute_ops.GroupTopKOp()

    def forward(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
    ):
        scores = scores.sigmoid()
        scores_with_bias = scores + correction_bias.unsqueeze(0)
        self.group_topk_op.forward(
            topk_weights,
            topk_ids,
            scores,
            scores_with_bias,
            n_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
        )
