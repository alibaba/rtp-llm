"""Triton helpers for packing DeepGEMM MegaMoE-SE inputs.

Routed input packing is mathematically identical to ordinary Mega MoE.  The SE
path additionally stages packed activation scales into the token-UTCCP layout
used by the in-kernel shared L1 GEMM.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["tokens", "block_m", "aligned_block_m"])
    def _stage_shared_l1_scales_kernel(
        source_ptr,
        destination_ptr,
        tokens,
        block_m,
        aligned_block_m,
        source_stride_m,
        source_stride_k,
        destination_stride_m,
        destination_stride_k,
        BLOCK_ROWS: tl.constexpr,
    ):
        rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS).to(
            tl.int64
        )
        packed_k = tl.program_id(1).to(tl.int64)
        mask = rows < tokens
        within = rows % block_m
        transformed = (
            rows // block_m * aligned_block_m
            + within // 128 * 128
            + (within % 32) * 4
            + (within % 128) // 32
        )
        values = tl.load(
            source_ptr + rows * source_stride_m + packed_k * source_stride_k,
            mask=mask,
            other=0,
        )
        tl.store(
            destination_ptr
            + transformed * destination_stride_m
            + packed_k * destination_stride_k,
            values,
            mask=mask,
        )


def stage_mega_moe_se_shared_l1_scales(
    source: torch.Tensor,
    destination: torch.Tensor,
    tokens: int,
    block_m: int,
) -> None:
    """Stage row-major packed K32 scales into DeepGEMM's shared-L1 layout.

    Active padding is zeroed first so reuse after a different token count or
    ``BLOCK_M`` cannot expose stale scale bytes to TMA loads.  Both operations
    run on the current PyTorch stream and are CUDA-graph safe.
    """

    if triton is None:
        raise RuntimeError("triton is unavailable")
    tokens = int(tokens)
    block_m = int(block_m)
    if source.dtype != torch.int32 or destination.dtype != torch.int32:
        raise TypeError(
            "MegaMoESE activation scales must be int32; "
            f"got source={source.dtype}, destination={destination.dtype}"
        )
    if not source.is_cuda or not destination.is_cuda:
        raise RuntimeError("MegaMoESE scale staging requires CUDA tensors")
    if source.dim() != 2 or destination.dim() != 2:
        raise ValueError("MegaMoESE scale tensors must both be rank 2")
    if tokens < 0 or tokens > source.size(0):
        raise ValueError(f"invalid tokens={tokens} for source rows={source.size(0)}")
    if source.size(1) != destination.size(1):
        raise ValueError(
            "MegaMoESE packed scale width mismatch: "
            f"source={source.size(1)}, destination={destination.size(1)}"
        )
    if block_m <= 0:
        raise ValueError(f"MegaMoESE BLOCK_M must be positive, got {block_m}")
    if tokens == 0:
        return

    aligned_block_m = ((block_m + 127) // 128) * 128
    active_rows = ((tokens + block_m - 1) // block_m) * aligned_block_m
    if active_rows > destination.size(0):
        raise RuntimeError(
            "MegaMoESE shared scale buffer is too small: "
            f"need rows={active_rows}, have rows={destination.size(0)}, "
            f"tokens={tokens}, block_m={block_m}"
        )
    destination[:active_rows].zero_()
    block_rows = 128
    grid = (triton.cdiv(tokens, block_rows), source.size(1))
    _stage_shared_l1_scales_kernel[grid](
        source,
        destination,
        tokens,
        block_m,
        aligned_block_m,
        source.stride(0),
        source.stride(1),
        destination.stride(0),
        destination.stride(1),
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )


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
    """Pack routed fields, then stage the additional shared-L1 scale view."""

    from .mega_moe_input_pack import fused_pack_mega_moe_inputs

    fused_pack_mega_moe_inputs(
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_indices,
        out_weights,
    )
    stage_mega_moe_se_shared_l1_scales(
        out_sf,
        out_shared_l1_sf,
        x.size(0),
        block_m,
    )


__all__ = [
    "fused_pack_mega_moe_se_inputs",
    "stage_mega_moe_se_shared_l1_scales",
]
