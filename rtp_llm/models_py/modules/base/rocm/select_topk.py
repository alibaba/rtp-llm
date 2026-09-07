import aiter
import torch
from torch import nn

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
        # aiter requires int32 and writes topk_ids in place.
        if topk_ids.dtype != torch.int32:
            raise TypeError(f"expect int32 topk_ids, got {topk_ids.dtype}")
        token_expert_indicies = torch.empty(
            topk_ids.shape[0], self.top_k, dtype=torch.int32, device=topk_ids.device
        )
        aiter.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indicies,
            router_logits_fp32,  # TODO(woosuk): Optimize this.
            True,
        )
