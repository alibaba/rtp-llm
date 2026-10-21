"""Triton MegaMoE input packer for GLM-5.

Fuses: BF16 activation -> FP8 E4M3, packed UE8M0 group-32 scales, and
router tensor copies into DeepGEMM's symmetric-memory dispatch buffer.
Uses the common MegaMoE kernel without changing GLM5 non-finite handling.
"""

from __future__ import annotations

import torch
import triton

from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
    _pack_mega_moe_inputs_optimized_kernel,
)


def fused_pack_mega_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
) -> None:
    """Fused packing of MegaMoE inputs into the symm-mem buffer.

    Performs BF16 → FP8 E4M3 quantization with packed UE8M0 (group=32) scales,
    and copies router weights/indices, all in a single Triton launch.
    """
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused MegaMoE input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")

    T, D = x.shape
    topk = weights.shape[1]

    if D % 128 != 0:
        raise ValueError(f"fused MegaMoE packer requires D % 128 == 0, got D={D}")
    if T == 0:
        return

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_k = triton.next_power_of_2(topk)
    block_m = 8 if T >= 2048 else 2

    grid = (triton.cdiv(T, block_m), triton.cdiv(D, 128))
    _pack_mega_moe_inputs_optimized_kernel[grid](
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        T,
        D,
        topk,
        x.stride(0),
        weights.stride(0),
        indices.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        1.0e-4,
        fp8_max,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        SANITIZE_NON_FINITE=False,
        num_warps=4,
    )
