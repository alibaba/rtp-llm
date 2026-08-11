import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.select_topk_op = compute_ops.SelectTopkOp(self.config)

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)


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

    @staticmethod
    def fused_sigmoid_supported(
        scores: torch.Tensor,
        correction_bias: torch.Tensor,
        n_group: int,
        topk_group: int,
        topk: int,
    ) -> bool:
        return (
            scores.is_cuda
            and scores.is_contiguous()
            and scores.dtype == torch.float32
            and scores.ndim == 2
            and scores.shape[0] > 0
            and scores.shape[1] == 896
            and correction_bias.is_cuda
            and correction_bias.device == scores.device
            and correction_bias.is_contiguous()
            and correction_bias.dtype == torch.float32
            and correction_bias.numel() == 896
            and n_group == 1
            and topk_group == 1
            and topk == 16
        )

    def forward_fused_sigmoid(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        topk: int,
        renormalize: bool,
        routed_scaling_factor: float,
    ) -> None:
        self.group_topk_op.forward_fused_sigmoid(
            topk_weights,
            topk_ids,
            router_logits,
            correction_bias,
            topk,
            renormalize,
            routed_scaling_factor,
        )


class FakeBalanceExpert(nn.Module):
    def __init__(
        self,
        expert_num: int,
        moe_k: int,
        dp_rank: int,
        dp_size: int,
        ep_size: int,
    ):
        super().__init__()
        self.fake_balance_expert_op = compute_ops.FakeBalanceExpertOp(
            expert_num, moe_k, dp_rank, dp_size, ep_size
        )

    def forward(
        self,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        self.fake_balance_expert_op.forward(topk_ids, topk_weights)
