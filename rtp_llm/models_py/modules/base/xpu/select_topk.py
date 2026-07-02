"""XPU MoE Top-K selection - PyTorch fallback.

Mirrors the CUDA ``SelectTopkOp`` semantics: softmax over all experts, take the
top-k experts, and (when the model uses MoE normalization, ``has_moe_norm``)
renormalize the selected weights to sum to 1.  Results are written in place into
the caller's ``topk_ids`` / ``topk_weights`` tensors, matching the CUDA op.
"""
import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig


class SelectTopk(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.moe_k
        self.renormalize = config.has_moe_norm

    def forward(
        self,
        router_logits_fp32: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        probs = torch.softmax(router_logits_fp32.float(), dim=-1)
        weights, ids = torch.topk(probs, self.top_k, dim=-1)
        if self.renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        topk_weights.copy_(weights.to(topk_weights.dtype))
        topk_ids.copy_(ids.to(topk_ids.dtype))
