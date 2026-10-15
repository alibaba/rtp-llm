"""Triton staging entry point for FP8xFP4 MegaMoE fused shared experts."""

from __future__ import annotations

import torch

from .input_packer_triton import fused_pack_mega_moe_inputs


def fused_pack_mega_moe_se_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_shared_l1_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    block_m: int,
) -> None:
    """Pack routed inputs and shared-L1 UTCCP scales in one launch."""
    fused_pack_mega_moe_inputs(
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
        out_shared_sf=out_shared_l1_sf,
        shared_block_m=block_m,
    )


__all__ = ["fused_pack_mega_moe_se_inputs"]
