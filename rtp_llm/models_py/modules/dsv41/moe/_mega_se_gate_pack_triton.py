"""Mega MoE SE router gate-pack wrappers.

The routed gate/quantization kernel is shared with ordinary Mega MoE.  These
SE-specific entrypoints immediately stage its packed activation scales into
the shared-L1 UTCCP view before the DeepGEMM collective is launched.
"""

from __future__ import annotations

import torch

try:
    import triton
except Exception:  # pragma: no cover - CPU-only import
    triton = None

from ._mega_gate_pack_triton import (
    fused_mega_moe_gate_pack_hash,
    fused_mega_moe_gate_pack_nonhash,
    fused_mega_moe_gate_pack_supported,
)
from ._mega_se_input_pack_triton import stage_mega_moe_se_shared_l1_scales


def fused_mega_moe_se_gate_pack_nonhash(
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    bias: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_shared_l1_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    block_m: int,
    route_scale: float,
    norm_eps: float = 1.0e-12,
) -> None:
    fused_mega_moe_gate_pack_nonhash(
        x,
        scores_bf16,
        bias,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
        route_scale=route_scale,
        norm_eps=norm_eps,
    )
    stage_mega_moe_se_shared_l1_scales(out_sf, out_shared_l1_sf, x.size(0), block_m)


def fused_mega_moe_se_gate_pack_hash(
    x: torch.Tensor,
    scores_bf16: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_shared_l1_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    block_m: int,
    route_scale: float,
    norm_eps: float = 1.0e-12,
) -> None:
    fused_mega_moe_gate_pack_hash(
        x,
        scores_bf16,
        input_ids,
        tid2eid,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
        route_scale=route_scale,
        norm_eps=norm_eps,
    )
    stage_mega_moe_se_shared_l1_scales(out_sf, out_shared_l1_sf, x.size(0), block_m)


def fused_mega_moe_se_gate_pack_supported(**kwargs) -> bool:
    return fused_mega_moe_gate_pack_supported(**kwargs)


__all__ = [
    "fused_mega_moe_se_gate_pack_hash",
    "fused_mega_moe_se_gate_pack_nonhash",
    "fused_mega_moe_se_gate_pack_supported",
    "triton",
]
