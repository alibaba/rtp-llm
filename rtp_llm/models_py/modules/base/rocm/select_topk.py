import aiter
import torch
from torch import nn

import rtp_llm.ops.compute_ops as compute_ops
from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        fake_balance_expert: bool,
        dp_rank: int,
        dp_size: int,
        ep_size: int,
    ):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k
        self.fake_balance_expert = fake_balance_expert
        self.dp_rank = dp_rank
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.expert_num = config.expert_num

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        token_expert_indicies = torch.empty(
            topk_ids.shape[0], self.top_k, dtype=torch.int32, device=topk_ids.device
        )
        # Ensure topk_ids is int32 for aiter.topk_softmax
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            router_logits_fp32,  # TODO(woosuk): Optimize this.
            True,
        )

        if self.fake_balance_expert:
            compute_ops.fake_balance_expert(
                topk_ids,
                topk_weights,
                self.dp_rank,
                self.dp_size,
                self.ep_size,
                self.expert_num,
            )
