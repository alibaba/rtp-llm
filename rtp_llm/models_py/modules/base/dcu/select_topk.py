import torch
from torch import nn
from lightop import op

from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.top_k = config.moe_k

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        topk_weights, topk_ids = torch.topk(
            router_logits_fp32, 
            k=self.top_k, 
            dim=-1, 
            largest=True, 
            sorted=True
        )

        topk_ids = topk_ids.to(torch.int32)
        return topk_weights, topk_ids
