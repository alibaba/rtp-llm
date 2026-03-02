import torch
from torch import nn
import torch.nn.functional as F

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
        weights, ids = torch.topk(
            router_logits_fp32, 
            k=self.top_k, 
            dim=-1, 
            largest=True, 
            sorted=True
        )
        weights = F.softmax(weights, dim=-1)
        topk_weights.copy_(weights)
        topk_ids.copy_(ids.to(topk_ids.dtype))
