"""Triton staging entry point for FP8xFP4 MegaMoE fused shared experts."""

from __future__ import annotations

import torch

from .input_packer_triton import fused_pack_mega_moe_inputs
from .shared_fp8_scale import stage_shared_fp8_input_scales


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
    """Pack routed inputs, then stage the shared-L1 UTCCP scale layout.

    Both launches are enqueued on the current stream.  Keeping this entry
    point separate prevents the ordinary MegaMoE packer from touching shared
    buffer views.
    """
    fused_pack_mega_moe_inputs(
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
    )
    stage_shared_fp8_input_scales(
        out_sf,
        out_shared_l1_sf,
        int(x.size(0)),
        block_m,
    )


__all__ = ["fused_pack_mega_moe_se_inputs"]
