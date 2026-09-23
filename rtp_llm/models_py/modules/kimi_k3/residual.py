"""Kimi K3 attention-residual selection."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


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
        if prefix_sum.ndim != 2:
            raise ValueError("AttnRes prefix_sum must have shape [tokens, hidden]")
        if (
            block_residual.ndim != 3
            or block_residual.shape[0] != prefix_sum.shape[0]
            or block_residual.shape[2] != prefix_sum.shape[1]
        ):
            raise ValueError(
                "AttnRes block_residual must have shape [tokens, blocks, hidden]: "
                f"prefix_sum={tuple(prefix_sum.shape)}, "
                f"block_residual={tuple(block_residual.shape)}"
            )
        active_blocks = (
            block_residual.shape[1] if num_blocks is None else int(num_blocks)
        )
        if active_blocks < 0 or active_blocks > block_residual.shape[1]:
            raise ValueError("AttnRes valid block count is outside the residual bank")
        if block_write_idx < -1 or block_write_idx >= block_residual.shape[1]:
            raise ValueError("AttnRes block write index is outside the residual bank")
        if output_norm_weight is not None and output_norm_eps is None:
            raise ValueError(
                "output_norm_eps is required when output RMSNorm is requested"
            )
        if (
            prefix_sum.is_cuda
            and prefix_sum.dtype == torch.bfloat16
            and prefix_sum.shape[1] == 7168
            and torch.cuda.get_device_capability(prefix_sum.device)[0] == 10
        ):
            from rtp_llm.ops.compute_ops import rtp_llm_ops

            if not hasattr(rtp_llm_ops, "kimi_k3_attn_res"):
                raise RuntimeError("Native K3 AttnRes requires the CUDA 13 RTP build")
            output = torch.empty_like(prefix_sum)
            rtp_llm_ops.kimi_k3_attn_res(
                prefix_sum, delta, block_residual, self.norm_weight,
                self.projection_weight, output_norm_weight, output,
                active_blocks, block_write_idx, self.eps,
                0.0 if output_norm_eps is None else float(output_norm_eps),
            )
            return output
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
                self.norm_weight.float() * self.projection_weight.reshape(-1).float()
            )
            probabilities = torch.softmax(
                (normalized * score_weight).sum(dim=-1), dim=-1
            )
            # Keep the mixture in FP32 through a fused output RMSNorm. Rounding
            # it to BF16 first changes the K3 decoder's normalization input.
            output = torch.einsum("tb,tbd->td", probabilities, candidates_float)
        if output_norm_weight is None:
            return output.to(dtype=prefix_sum.dtype)
        output_float = output.float()
        normalized = output_float * torch.rsqrt(
            output_float.square().mean(dim=-1, keepdim=True) + output_norm_eps
        )
        return (output_norm_weight.float() * normalized).to(dtype=prefix_sum.dtype)


__all__ = ["KimiK3AttentionResidual"]
