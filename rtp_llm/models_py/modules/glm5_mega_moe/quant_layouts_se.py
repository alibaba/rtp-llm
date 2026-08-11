"""Quantization-layout helpers for FP8xFP4 MegaMoE with shared experts.

The routed experts keep the FP4/K32 contract.  Shared experts use FP8 E4M3
weights with 128x128 UE8M0 scales.  This module is the SE-specific import
surface so the ordinary ``mega_moe`` path does not acquire shared-expert
branches.
"""

from __future__ import annotations

import torch

from .quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
    prepare_fp8_weight_scale_for_deepgemm,
)

SHARED_WEIGHT_RECIPE = (1, FP8_BLOCK, FP8_BLOCK)


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
) -> torch.Tensor:
    """Prepare a dense shared-expert 128x128 UE8M0 scale tensor."""
    return prepare_fp8_weight_scale_for_deepgemm(scale, mn, k)


__all__ = [
    "FP4_BLOCK",
    "FP8_BLOCK",
    "SHARED_WEIGHT_RECIPE",
    "prepare_routed_fp4_scale_for_mega_moe_se",
    "prepare_shared_fp8_scale_for_mega_moe_se",
]
