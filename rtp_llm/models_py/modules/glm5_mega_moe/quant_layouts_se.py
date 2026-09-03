"""Quantization-layout helpers for FP8xFP4 MegaMoE with shared experts.

The routed experts keep the FP4/K32 contract.  Shared experts use FP8 E4M3
weights with either legacy 128x128 or ModelOpt MXFP8 1x32 scales. This module
is the SE-specific import surface so the ordinary ``mega_moe`` path does not
acquire shared-expert branches.
"""

from __future__ import annotations

import torch

from .quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    MXFP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_fp8_weight_scale_for_deepgemm,
)

SHARED_WEIGHT_RECIPE = (1, FP8_BLOCK, FP8_BLOCK)
MXFP8_SHARED_WEIGHT_RECIPE = (1, 1, MXFP8_BLOCK)


def prepare_routed_fp4_scale_for_mega_moe_se(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: int,
) -> torch.Tensor:
    """Prepare routed FP4 scales through the SE-specific entry point."""
    return prepare_fp4_weight_scale_for_deepgemm(scale, mn, k, num_groups)


def prepare_shared_fp8_scale_for_mega_moe_se(
    scale: torch.Tensor,
    mn: int,
    k: int,
    recipe: tuple[int, int] = (FP8_BLOCK, FP8_BLOCK),
) -> torch.Tensor:
    """Prepare a dense shared-expert scale for the fused MegaMoE API."""
    if recipe == (1, MXFP8_BLOCK):
        # Match vLLM's grouped-singleton transform. It requests the MN-major,
        # TMA-aligned scale layout and then removes the synthetic expert axis.
        return prepare_fp8_weight_scale_for_deepgemm(
            scale.unsqueeze(0),
            mn,
            k,
            num_groups=1,
            recipe=recipe,
        ).squeeze(0)
    return prepare_fp8_weight_scale_for_deepgemm(
        scale,
        mn,
        k,
        recipe=recipe,
    )


__all__ = [
    "FP4_BLOCK",
    "FP8_BLOCK",
    "MXFP8_SHARED_WEIGHT_RECIPE",
    "SHARED_WEIGHT_RECIPE",
    "prepare_routed_fp4_scale_for_mega_moe_se",
    "prepare_shared_fp8_scale_for_mega_moe_se",
]
