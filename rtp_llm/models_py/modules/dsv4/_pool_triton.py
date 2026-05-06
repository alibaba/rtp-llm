"""Small fused helpers for DSV4 BlockPool copy/gather hot paths."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _masked_copy_to_pool_kernel(
        src_ptr,
        slot_ptr,
        dst_ptr,
        src_stride0,
        src_stride1,
        dst_stride0,
        dst_stride1,
        D: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        col_mask = offs < D
        slot = tl.load(slot_ptr + row).to(tl.int64)
        valid = slot >= 0
        vals = tl.load(
            src_ptr + row * src_stride0 + offs * src_stride1,
            mask=col_mask,
            other=0.0,
        )
        safe_slot = tl.where(valid, slot, 0).to(tl.int64)
        tl.store(
            dst_ptr + safe_slot * dst_stride0 + offs * dst_stride1,
            vals,
            mask=col_mask & valid,
        )

    @triton.jit
    def _masked_gather_from_pool_kernel(
        pool_ptr,
        slot_ptr,
        valid_ptr,
        out_ptr,
        pool_stride0,
        pool_stride1,
        out_stride0,
        out_stride1,
        D: tl.constexpr,
        HAS_VALID: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        col_mask = offs < D
        slot = tl.load(slot_ptr + row).to(tl.int64)
        valid = slot >= 0
        if HAS_VALID:
            valid = valid & tl.load(valid_ptr + row)
        safe_slot = tl.where(valid, slot, 0).to(tl.int64)
        vals = tl.load(
            pool_ptr + safe_slot * pool_stride0 + offs * pool_stride1,
            mask=col_mask,
            other=0.0,
        )
        vals = tl.where(valid, vals, 0.0)
        tl.store(
            out_ptr + row * out_stride0 + offs * out_stride1,
            vals,
            mask=col_mask,
        )


def _supported_cuda_2d(t: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and t.is_cuda
        and t.dim() == 2
        and t.shape[-1] > 0
        and t.shape[-1] <= 4096
    )


def _flat_index(t: torch.Tensor) -> torch.Tensor:
    flat = t.reshape(-1)
    return flat if flat.is_contiguous() else flat.contiguous()


def masked_copy_to_pool(
    source: torch.Tensor,
    slot_mapping: torch.Tensor,
    pool_view: torch.Tensor,
) -> bool:
    """Fused ``pool_view[slot] = source`` for non-negative slots.

    Returns True when the Triton path ran. Callers keep the torch fallback
    so CPU and unusual shapes remain covered.
    """
    if (
        not _supported_cuda_2d(source)
        or not pool_view.is_cuda
        or source.shape[-1] != pool_view.shape[-1]
        or source.shape[0] != slot_mapping.numel()
    ):
        return False
    slot_flat = _flat_index(slot_mapping)
    D = int(source.shape[-1])
    BLOCK_D = triton.next_power_of_2(D)
    _masked_copy_to_pool_kernel[(int(source.shape[0]),)](
        source,
        slot_flat,
        pool_view,
        source.stride(0),
        source.stride(1),
        pool_view.stride(0),
        pool_view.stride(1),
        D=D,
        BLOCK_D=BLOCK_D,
        num_warps=4 if BLOCK_D <= 1024 else 8,
    )
    return True


def masked_gather_from_pool(
    pool_view: torch.Tensor,
    slot_mapping: torch.Tensor,
    valid: Optional[torch.Tensor],
    *,
    out_shape: tuple[int, ...],
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Gather rows from ``pool_view`` and zero-fill invalid slots.

    ``valid=None`` means every slot is already safe and skips the masked
    path, leaving only a single ``index_select`` fallback.
    """
    out_dtype = dtype if dtype is not None else pool_view.dtype
    slot_flat = _flat_index(slot_mapping)
    rows = int(slot_flat.numel())
    D = int(pool_view.shape[-1])
    if rows == 0:
        return torch.empty(out_shape, dtype=out_dtype, device=pool_view.device)

    if valid is None:
        gathered = pool_view.index_select(0, slot_flat.to(torch.long))
        if gathered.dtype != out_dtype:
            gathered = gathered.to(out_dtype)
        return gathered.view(*out_shape)

    valid_flat = _flat_index(valid.to(torch.bool))
    if (
        _supported_cuda_2d(pool_view)
        and slot_flat.is_cuda
        and valid_flat.is_cuda
        and D <= 4096
    ):
        out = torch.empty((rows, D), dtype=out_dtype, device=pool_view.device)
        BLOCK_D = triton.next_power_of_2(D)
        _masked_gather_from_pool_kernel[(rows,)](
            pool_view,
            slot_flat,
            valid_flat,
            out,
            pool_view.stride(0),
            pool_view.stride(1),
            out.stride(0),
            out.stride(1),
            D=D,
            HAS_VALID=True,
            BLOCK_D=BLOCK_D,
            num_warps=4 if BLOCK_D <= 1024 else 8,
        )
        return out.view(*out_shape)

    safe_slot = torch.where(valid_flat, slot_flat, torch.zeros_like(slot_flat))
    gathered = pool_view.index_select(0, safe_slot.to(torch.long))
    if gathered.dtype != out_dtype:
        gathered = gathered.to(out_dtype)
    zero = torch.zeros((), dtype=out_dtype, device=pool_view.device)
    out = torch.where(valid_flat.unsqueeze(-1), gathered, zero)
    return out.view(*out_shape)
