import torch
import torch.nn as nn

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
        topk_weights_fp32, topk_ids_fp32 = torch.topk(
            router_logits_fp32, self.top_k, dim=-1
        )
        topk_weights_fp32 = torch.softmax(topk_weights_fp32, dim=-1)
        topk_weights.copy_(topk_weights_fp32.to(topk_weights.dtype))
        topk_ids.copy_(topk_ids_fp32.to(topk_ids.dtype))
