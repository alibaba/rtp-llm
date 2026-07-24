"""Attention-residual mixing used at K3 block boundaries."""

from __future__ import annotations

import torch
from torch import nn

from rtp_llm.models_py.modules.kimi_k3.reference.common import KimiRMSNorm


class KimiAttentionResidualMixer(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.norm = KimiRMSNorm(hidden_size, eps=eps)
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, prefix_sum: torch.Tensor, block_residual: torch.Tensor
    ) -> torch.Tensor:
        if prefix_sum.ndim != 2 or prefix_sum.shape[-1] != self.hidden_size:
            raise ValueError("prefix_sum must have shape [tokens,hidden_size]")
        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != prefix_sum.shape[0]
            or block_residual.shape[-1] != self.hidden_size
        ):
            raise ValueError(
                "block_residual must have shape [tokens,blocks,hidden_size]"
            )
        candidates = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
        candidates_float = candidates.float()
        normalized = candidates_float * torch.rsqrt(
            candidates_float.square().mean(dim=-1, keepdim=True) + self.norm.eps
        )
        score_weight = self.norm.weight.float() * self.proj.weight.squeeze(0).float()
        probabilities = torch.softmax(
            (normalized * score_weight).sum(dim=-1), dim=-1
        )
        mixed = torch.einsum("tb,tbd->td", probabilities, candidates_float)
        return mixed.to(dtype=prefix_sum.dtype)
