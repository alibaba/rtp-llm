"""Kimi K3 attention-residual selection."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from rtp_llm.models_py.triton_kernels.kimi_kda.attn_res import (
    is_kimi_k3_attn_res_supported,
    kimi_k3_attn_res,
)


class KimiK3AttentionResidual(nn.Module):
    """Select over K3 block anchors and the running prefix residual."""

    def __init__(
        self,
        norm_weight: torch.Tensor,
        projection_weight: torch.Tensor,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm_weight = norm_weight
        self.projection_weight = projection_weight
        self.eps = float(eps)

    def forward(
        self,
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        *,
        output_norm_weight: Optional[torch.Tensor] = None,
        output_norm_eps: Optional[float] = None,
        delta: Optional[torch.Tensor] = None,
        num_blocks: Optional[int] = None,
        block_write_idx: int = -1,
    ) -> torch.Tensor:
        active_blocks = (
            block_residual.shape[1] if num_blocks is None else int(num_blocks)
        )
        if is_kimi_k3_attn_res_supported(
            prefix_sum,
            block_residual,
            self.norm_weight,
            self.projection_weight,
            output_norm_weight,
            delta,
            active_blocks,
            block_write_idx,
        ):
            return kimi_k3_attn_res(
                prefix_sum,
                block_residual,
                self.norm_weight,
                self.projection_weight,
                self.eps,
                output_norm_weight,
                output_norm_eps,
                delta,
                active_blocks,
                block_write_idx,
            )

        if delta is not None:
            prefix_sum.add_(delta)
        if block_write_idx >= 0:
            block_residual[:, block_write_idx].copy_(prefix_sum)
        if active_blocks == 0:
            output = prefix_sum
        else:
            candidates = torch.cat(
                (block_residual[:, :active_blocks], prefix_sum.unsqueeze(1)), dim=1
            )
            candidates_float = candidates.float()
            normalized = candidates_float * torch.rsqrt(
                candidates_float.square().mean(dim=-1, keepdim=True) + self.eps
            )
            score_weight = (
                self.norm_weight.float()
                * self.projection_weight.reshape(-1).float()
            )
            probabilities = torch.softmax(
                (normalized * score_weight).sum(dim=-1), dim=-1
            )
            output = torch.einsum(
                "tb,tbd->td", probabilities, candidates_float
            ).to(dtype=prefix_sum.dtype)
        if output_norm_weight is None:
            return output
        output_float = output.float()
        normalized = output_float * torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + output_norm_eps
        )
        return output_norm_weight * normalized.to(dtype=output.dtype)


__all__ = ["KimiK3AttentionResidual"]
