"""Triton helpers for packing DeepGEMM Mega MoE fused-SE inputs.

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
        # One program per BLOCK_ROWS DESTINATION rows: each padded UTCCP row is
        # mapped back to its source row, so the kernel writes every active row
        # exactly once — real scales where a source row exists, zeros in the
        # alignment padding — replacing the historical separate zero-fill
        # launch (which also double-wrote the data rows). The inverse of
        #   j = within // 128 * 128 + (within % 32) * 4 + (within % 128) // 32
        # over [0, aligned_block_m) is
        #   within = j // 128 * 128 + 32 * ((j % 128) % 4) + (j % 128) // 4
        # because j % 128 = 4 * (within % 32) + (within % 128) // 32 is a
        # bijection of [0, 128) (quotient selects the /32 bucket, remainder the
        # %32 residue). Destination rows whose preimage falls outside
        # [0, block_m) or beyond `tokens` are padding and store zero. The grid
        # covers `active_rows` exactly (a multiple of aligned_block_m, hence of
        # BLOCK_ROWS=128), so the store needs no bounds mask.
        d_rows = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(
            0, BLOCK_ROWS
        ).to(tl.int64)
        packed_k = tl.program_id(1).to(tl.int64)
        block = d_rows // aligned_block_m
        j = d_rows % aligned_block_m
        within = j // 128 * 128 + 32 * ((j % 128) % 4) + (j % 128) // 4
        rows = block * block_m + within
        valid = (within < block_m) & (rows < tokens)
        values = tl.load(
            source_ptr + rows * source_stride_m + packed_k * source_stride_k,
            mask=valid,
            other=0,
        )
        tl.store(
            destination_ptr
            + d_rows * destination_stride_m
            + packed_k * destination_stride_k,
            values,
        )


def stage_mega_moe_se_shared_l1_scales(
    source: torch.Tensor,
    destination: torch.Tensor,
    tokens: int,
    block_m: int,
) -> None:
    """Stage row-major packed K32 scales into DeepGEMM's shared-L1 layout.

    A single fused kernel writes real scales into the transformed rows and
    zeros into the alignment padding of the active region, so reuse after a
    different token count or ``BLOCK_M`` cannot expose stale scale bytes to
    TMA loads. Runs on the current PyTorch stream and is CUDA-graph safe.
    """

    tokens = int(tokens)
    block_m = int(block_m)
    if tokens == 0:
        return

    aligned_block_m = ((block_m + 127) // 128) * 128
    active_rows = ((tokens + block_m - 1) // block_m) * aligned_block_m
    block_rows = 128
    grid = (active_rows // block_rows, source.size(1))
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

    from ._mega_input_pack_triton import fused_pack_mega_moe_inputs

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
