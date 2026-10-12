"""Quant constants and helpers for GLM-5 MegaMoE.

Ported from dsv4/quant_layouts.py. Handles FP4/FP8 block sizes and
the activation cast for UE8M0 packed scale factors.
"""

import os
import tempfile
from typing import Optional

import torch
from rtp_llm.models_py.kernels.cuda.quant_layouts import FP4_BLOCK as FP4_BLOCK
from rtp_llm.models_py.kernels.cuda.quant_layouts import FP8_BLOCK as FP8_BLOCK
from rtp_llm.models_py.kernels.cuda.quant_layouts import (
    per_token_cast_to_fp8_packed_ue8m0 as per_token_cast_to_fp8_packed_ue8m0,
)


def prepare_fp4_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert FP4 UE8M0 weight scale to DeepGEMM's SM100 layout.

    Converts raw float8_e8m0fnu or float32 scale tensors into the
    TMA-aligned packed int32 layout that DeepGEMM's FP8xFP4 kernels expect.
    """
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(f"expected FP4 UE8M0 or float32 scale, got {scale.dtype}")

    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
    )
    os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

    import deep_gemm

    scale_fp32 = scale.float()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (1, FP4_BLOCK)
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32, mn, k, (1, FP4_BLOCK), num_groups
    )


def prepare_fp8_weight_scale_for_deepgemm(
    scale: torch.Tensor,
    mn: int,
    k: int,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Convert FP8 128x128 per-block scales to DeepGEMM's packed layout."""
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype not in (torch.float8_e8m0fnu, torch.float32):
        raise TypeError(
            f"expected FP8 per-block scale or packed int32, got {scale.dtype}"
        )

    os.environ.setdefault(
        "DG_JIT_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), f"deep_gemm_jit_{os.getuid()}"),
    )
    os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

    import deep_gemm

    if scale.dim() < 2:
        raise ValueError(
            f"FP8 per-block scale must have at least 2 dims, got shape={tuple(scale.shape)}"
        )
    expected_m_blocks = (mn + FP8_BLOCK - 1) // FP8_BLOCK
    expected_k_blocks = (k + FP8_BLOCK - 1) // FP8_BLOCK
    if tuple(scale.shape[-2:]) != (expected_m_blocks, expected_k_blocks):
        raise ValueError(
            "FP8 mega_moe weight scale must be 128x128 per-block or already "
            f"packed int32. Got shape={tuple(scale.shape)}, expected trailing "
            f"dims=({expected_m_blocks}, {expected_k_blocks}) for mn={mn}, k={k}."
        )

    scale_fp32 = scale.float()
    if num_groups is None:
        return deep_gemm.transform_sf_into_required_layout(
            scale_fp32, mn, k, (FP8_BLOCK, FP8_BLOCK)
        )
    return deep_gemm.transform_sf_into_required_layout(
        scale_fp32, mn, k, (FP8_BLOCK, FP8_BLOCK), num_groups
    )


__all__ = [
    "FP4_BLOCK",
    "FP8_BLOCK",
    "per_token_cast_to_fp8_packed_ue8m0",
    "prepare_fp4_weight_scale_for_deepgemm",
    "prepare_fp8_weight_scale_for_deepgemm",
]
