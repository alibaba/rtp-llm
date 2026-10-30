"""Triton GC MoE input packer for GLM-5.

Fuses: BF16 activation -> FP8 E4M3, packed UE8M0 group-32 scales, and
router tensor copies into DeepGEMM's symmetric-memory dispatch buffer.
The four group-32 reductions run in parallel. Routed and shared scale layouts,
router copies, and active shared padding are completed in the same launch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


if triton is not None:

    @triton.jit(do_not_specialize=["M", "SHARED_ACTIVE_ROWS"])
    def _pack_gc_vector_groups_kernel(
        x_ptr,
        weights_ptr,
        indices_ptr,
        out_fp8_ptr,
        out_sf_ptr,
        out_weights_ptr,
        out_indices_ptr,
        out_shared_sf_ptr,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        x_stride_m: tl.constexpr,
        weights_stride_m: tl.constexpr,
        indices_stride_m: tl.constexpr,
        out_stride_m: tl.constexpr,
        sf_stride_m: tl.constexpr,
        out_weights_stride_m: tl.constexpr,
        out_indices_stride_m: tl.constexpr,
        shared_sf_stride_m: tl.constexpr,
        shared_sf_stride_k: tl.constexpr,
        eps: tl.constexpr,
        fp8_max: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_SHARED_SF: tl.constexpr,
        SHARED_BLOCK_M: tl.constexpr,
        SHARED_ALIGNED_BLOCK_M: tl.constexpr,
        SHARED_ACTIVE_ROWS,
        SHARED_ZERO_ROWS: tl.constexpr,
    ):
        # Large prefill rows can make row * hidden_stride approach int32 range.
        # Use int64 row offsets for all pointer arithmetic.
        pid_m_blk = tl.program_id(0).to(tl.int64)
        pid_blk = tl.program_id(1)

        offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_n = pid_blk * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = offs_m < M
        mask = row_mask[:, None] & (offs_n[None, :] < N)
        values = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        groups = tl.reshape(values, (BLOCK_M, BLOCK_N // 32, 32))
        block_absmax = tl.maximum(tl.max(tl.abs(groups), axis=2), eps)
        scale_raw = block_absmax / fp8_max
        bits = scale_raw.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        quantized = tl.clamp(groups / scale[:, :, None], -fp8_max, fp8_max).to(
            tl.float8e4nv
        )
        tl.store(
            out_fp8_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :],
            tl.reshape(quantized, (BLOCK_M, BLOCK_N)),
            mask=mask,
        )
        shift = tl.arange(0, 4) * 8
        packed = tl.sum(
            tl.reshape(exp, (BLOCK_M, BLOCK_N // 128, 4)) << shift[None, None, :],
            axis=2,
        )
        sf_cols = pid_blk * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)
        sf_mask = row_mask[:, None] & (sf_cols[None, :] < N // 128)
        tl.store(
            out_sf_ptr + offs_m[:, None] * sf_stride_m + sf_cols[None, :],
            packed,
            mask=sf_mask,
        )

        if HAS_SHARED_SF:
            # Store the just-computed packed scales directly in DeepGEMM's
            # UTCCP layout. A second set of stores clears only padding rows;
            # it never overlaps with the valid-token stores above.
            within_block = offs_m % SHARED_BLOCK_M
            shared_row = (
                offs_m // SHARED_BLOCK_M * SHARED_ALIGNED_BLOCK_M
                + within_block // 128 * 128
                + (within_block % 32) * 4
                + (within_block % 128) // 32
            )
            tl.store(
                out_shared_sf_ptr
                + shared_row[:, None] * shared_sf_stride_m
                + sf_cols[None, :] * shared_sf_stride_k,
                packed,
                mask=sf_mask,
            )

            destination_row = pid_m_blk * SHARED_ZERO_ROWS + tl.arange(
                0, SHARED_ZERO_ROWS
            ).to(tl.int64)
            local_destination_row = destination_row % SHARED_ALIGNED_BLOCK_M
            row_in_128_tile = local_destination_row % 128
            row_in_block = (
                local_destination_row // 128 * 128
                + (row_in_128_tile % 4) * 32
                + row_in_128_tile // 4
            )
            source_row = (
                destination_row // SHARED_ALIGNED_BLOCK_M * SHARED_BLOCK_M
                + row_in_block
            )
            padding_mask = (destination_row < SHARED_ACTIVE_ROWS) & (
                (row_in_block >= SHARED_BLOCK_M) | (source_row >= M)
            )
            tl.store(
                out_shared_sf_ptr
                + destination_row[:, None] * shared_sf_stride_m
                + sf_cols[None, :] * shared_sf_stride_k,
                0,
                mask=padding_mask[:, None] & (sf_cols[None, :] < N // 128),
            )

        if pid_blk == 0:
            router_offs = tl.arange(0, BLOCK_K)
            router_mask = row_mask[:, None] & (router_offs[None, :] < K)
            w = tl.load(
                weights_ptr + offs_m[:, None] * weights_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0.0,
            ).to(tl.float32)
            idx = tl.load(
                indices_ptr + offs_m[:, None] * indices_stride_m + router_offs[None, :],
                mask=router_mask,
                other=0,
            ).to(tl.int64)
            tl.store(
                out_weights_ptr
                + offs_m[:, None] * out_weights_stride_m
                + router_offs[None, :],
                w,
                mask=router_mask,
            )
            tl.store(
                out_indices_ptr
                + offs_m[:, None] * out_indices_stride_m
                + router_offs[None, :],
                idx,
                mask=router_mask,
            )


def fused_pack_gc_moe_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    out_shared_sf: torch.Tensor | None = None,
    shared_block_m: int = 0,
) -> None:
    """Fused packing of GC MoE inputs into the symm-mem buffer.

    Performs BF16 → FP8 E4M3 quantization with packed UE8M0 (group=32) scales,
    and copies router weights/indices, all in a single Triton launch.
    """
    if triton is None:
        raise RuntimeError("triton is unavailable")
    if not x.is_cuda:
        raise RuntimeError("fused GC MoE input packer requires CUDA tensors")
    if x.dim() != 2:
        raise ValueError(f"x must be [T,D], got {tuple(x.shape)}")

    T, D = x.shape
    topk = weights.shape[1]

    if D % 128 != 0:
        raise ValueError(f"fused GC MoE packer requires D % 128 == 0, got D={D}")
    if out_shared_sf is not None:
        if shared_block_m <= 0:
            raise ValueError(f"shared_block_m must be positive, got {shared_block_m}")
        if out_sf.dtype != torch.int32 or out_shared_sf.dtype != torch.int32:
            raise TypeError("shared FP8 input scales must be packed int32 tensors")
        if out_shared_sf.dim() != 2 or out_shared_sf.size(1) != D // 128:
            raise ValueError(
                "shared FP8 input scale width mismatch: "
                f"expected {D // 128}, got {tuple(out_shared_sf.shape)}"
            )
        aligned_shared_block_m = triton.cdiv(shared_block_m, 128) * 128
        active_shared_rows = triton.cdiv(T, shared_block_m) * aligned_shared_block_m
        if active_shared_rows > out_shared_sf.size(0):
            raise ValueError(
                "shared FP8 input scale buffer is too small: "
                f"required={active_shared_rows}, available={out_shared_sf.size(0)}"
            )
    else:
        aligned_shared_block_m = 0
        active_shared_rows = 0
    if T == 0:
        if out_shared_sf is not None:
            out_shared_sf.zero_()
        return

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    block_k = triton.next_power_of_2(topk)
    # Pack tile sizes are independent of the MoE algorithm's shared_block_m.
    # Small batches favor 128-channel tiles; larger batches amortize the CTA
    # count without changing quantization or the shared scale layout.
    if T <= 32:
        pack_block_m, pack_tile_n, pack_num_warps = 4, 128, 4
    elif T <= 128:
        pack_block_m, pack_tile_n, pack_num_warps = 4, 512, 4
    else:
        pack_block_m, pack_tile_n, pack_num_warps = 8, 128, 1

    pack_programs = triton.cdiv(T, pack_block_m)
    shared_zero_rows = (
        triton.next_power_of_2(triton.cdiv(active_shared_rows, pack_programs))
        if out_shared_sf is not None
        else 1
    )
    grid = (pack_programs, triton.cdiv(D, pack_tile_n))
    _pack_gc_vector_groups_kernel[grid](
        x,
        weights,
        indices,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        out_shared_sf if out_shared_sf is not None else out_sf,
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
        out_shared_sf.stride(0) if out_shared_sf is not None else 0,
        out_shared_sf.stride(1) if out_shared_sf is not None else 0,
        1.0e-4,
        fp8_max,
        BLOCK_M=pack_block_m,
        BLOCK_N=pack_tile_n,
        BLOCK_K=block_k,
        HAS_SHARED_SF=out_shared_sf is not None,
        SHARED_BLOCK_M=shared_block_m,
        SHARED_ALIGNED_BLOCK_M=aligned_shared_block_m,
        SHARED_ACTIVE_ROWS=active_shared_rows,
        SHARED_ZERO_ROWS=shared_zero_rows,
        num_warps=pack_num_warps,
    )
