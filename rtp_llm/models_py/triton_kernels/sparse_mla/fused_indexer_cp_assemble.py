"""Fused Triton kernels for CP-sharded indexer K assemble.

Replaces the element-wise kernel sequences in:
  1. copy_actual_indexer_k_to_padded (scatter actual rows → padded layout)
  2. assemble_indexer_k post-allgather restore (gather by restore_indices)
"""

from typing import Optional

import torch
import torch.distributed
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group, _get_group


@triton.jit
def _fused_zero_scatter_kernel(
    actual_k_ptr,
    actual_s_ptr,
    padded_k_ptr,
    padded_s_ptr,
    src_for_padded_ptr,
    k_stride: tl.constexpr,
    s_stride: tl.constexpr,
    n_padded_rows,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Fused zero-fill + scatter for padded buffer.

    Grid: (n_padded_rows,)
    For each padded row: if src_for_padded[row] >= 0, copy from actual;
    otherwise write zeros.
    """
    row_id = tl.program_id(0)
    if row_id >= n_padded_rows:
        return
    src_row = tl.load(src_for_padded_ptr + row_id)

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < k_stride
    if src_row >= 0:
        k_data = tl.load(actual_k_ptr + src_row * k_stride + k_cols, mask=k_mask)
    else:
        k_data = tl.zeros([BLOCK_K], dtype=tl.uint8)
    tl.store(padded_k_ptr + row_id * k_stride + k_cols, k_data, mask=k_mask)

    s_cols = tl.arange(0, BLOCK_S)
    s_mask = s_cols < s_stride
    if src_row >= 0:
        s_data = tl.load(actual_s_ptr + src_row * s_stride + s_cols, mask=s_mask)
    else:
        s_data = tl.zeros([BLOCK_S], dtype=tl.uint8)
    tl.store(padded_s_ptr + row_id * s_stride + s_cols, s_data, mask=s_mask)


@triton.jit
def _gather_restore_fused_kernel(
    gathered_k_ptr,
    gathered_s_ptr,
    out_k_ptr,
    out_s_ptr,
    indices_ptr,
    k_stride: tl.constexpr,
    s_stride: tl.constexpr,
    n_rows,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Gather rows from allgathered buffers (k+s fused) by restore_indices.

    Grid: (n_rows,)
    Each program copies one row of k_quant and k_scale from gathered → out.
    """
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    src_row = tl.load(indices_ptr + row_id)

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < k_stride
    k_data = tl.load(gathered_k_ptr + src_row * k_stride + k_cols, mask=k_mask)
    tl.store(out_k_ptr + row_id * k_stride + k_cols, k_data, mask=k_mask)

    s_cols = tl.arange(0, BLOCK_S)
    s_mask = s_cols < s_stride
    s_data = tl.load(gathered_s_ptr + src_row * s_stride + s_cols, mask=s_mask)
    tl.store(out_s_ptr + row_id * s_stride + s_cols, s_data, mask=s_mask)


@triton.jit
def _fused_zero_scatter_single_kernel(
    actual_ptr,
    padded_ptr,
    src_for_padded_ptr,
    row_stride: tl.constexpr,
    n_padded_rows,
    BLOCK_D: tl.constexpr,
):
    """Fused zero-fill + scatter for a single tensor.

    Grid: (n_padded_rows,)
    For each padded row: if src_for_padded[row] >= 0, copy from actual;
    otherwise write zeros.
    """
    row_id = tl.program_id(0)
    if row_id >= n_padded_rows:
        return
    src_row = tl.load(src_for_padded_ptr + row_id)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < row_stride
    if src_row >= 0:
        data = tl.load(actual_ptr + src_row * row_stride + cols, mask=mask)
    else:
        data = tl.zeros([BLOCK_D], dtype=tl.uint8)
    tl.store(padded_ptr + row_id * row_stride + cols, data, mask=mask)


@triton.jit
def _gather_restore_single_kernel(
    gathered_ptr,
    out_ptr,
    indices_ptr,
    row_stride: tl.constexpr,
    n_rows,
    BLOCK_D: tl.constexpr,
):
    """Gather rows from allgathered buffer by restore_indices (single tensor).

    Grid: (n_rows,)
    """
    row_id = tl.program_id(0)
    if row_id >= n_rows:
        return
    src_row = tl.load(indices_ptr + row_id)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < row_stride
    data = tl.load(gathered_ptr + src_row * row_stride + cols, mask=mask)
    tl.store(out_ptr + row_id * row_stride + cols, data, mask=mask)


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def fused_copy_and_assemble_indexer_k(
    *,
    plan,
    actual_k_quant: torch.Tensor,
    actual_k_scale: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
    copy_dst_idx: Optional[torch.Tensor] = None,
    src_for_padded: Optional[torch.Tensor] = None,
) -> None:
    """Fused copy_actual_to_padded + assemble_indexer_k.

    Args:
        plan: IndexerCPChunkPlan.
        actual_k_quant: [total_actual_local_T, head_dim] fp8.
        actual_k_scale: [total_actual_local_T, scale_dim] uint8.
        out_k_quant: Output [chunk_T, head_dim] fp8.
        out_k_scale: Output [chunk_T, scale_dim] uint8.
        copy_dst_idx: Precomputed scatter index [total_actual_local_T] int64 (legacy, unused if src_for_padded provided).
        src_for_padded: Precomputed inverse map [total_local_T] int64.
            src_for_padded[padded_row] = actual_row if padded_row is a destination, else -1.
            Precomputed in plan(). Enables fused zero+scatter in one kernel.
    """
    device = out_k_quant.device
    head_dim = out_k_quant.shape[1]
    scale_dim = out_k_scale.shape[1]
    chunk_T = out_k_quant.shape[0]
    total_local_T = plan.total_local_T
    total_actual_T = plan.total_actual_local_T

    if chunk_T == 0:
        return

    BLOCK_K = _next_power_of_2(head_dim)
    BLOCK_S = _next_power_of_2(scale_dim)

    # Step 1: build padded local buffer
    no_padding = total_actual_T == total_local_T
    if no_padding:
        local_k = actual_k_quant
        local_s = actual_k_scale
    elif src_for_padded is not None and total_actual_T > 0:
        local_k = torch.empty(
            (total_local_T, head_dim), dtype=actual_k_quant.dtype, device=device
        )
        local_s = torch.empty(
            (total_local_T, scale_dim), dtype=actual_k_scale.dtype, device=device
        )
        _fused_zero_scatter_kernel[(total_local_T,)](
            actual_k_quant.view(torch.uint8),
            actual_k_scale,
            local_k.view(torch.uint8),
            local_s,
            src_for_padded,
            head_dim,
            scale_dim,
            total_local_T,
            BLOCK_K,
            BLOCK_S,
        )
    else:
        local_k = torch.zeros(
            (total_local_T, head_dim), dtype=actual_k_quant.dtype, device=device
        )
        local_s = torch.zeros(
            (total_local_T, scale_dim), dtype=actual_k_scale.dtype, device=device
        )
        if total_actual_T > 0:
            B = int(plan.per_req_actual_local_kv_lens.numel())
            if B == 1:
                local_k[:total_actual_T].copy_(actual_k_quant)
                local_s[:total_actual_T].copy_(actual_k_scale)

    # Step 2: all_gather with torch.empty
    pg = _get_group(Group.TP)
    world_size = torch.distributed.get_world_size(pg)
    gathered_k = torch.empty(
        (world_size * total_local_T, head_dim),
        device=device,
        dtype=local_k.dtype,
    )
    gathered_s = torch.empty(
        (world_size * total_local_T, scale_dim),
        device=device,
        dtype=local_s.dtype,
    )
    torch.distributed.all_gather_into_tensor(gathered_k, local_k, group=pg)
    torch.distributed.all_gather_into_tensor(gathered_s, local_s, group=pg)

    # Step 3: fused restore gather
    restore_indices = plan.restore_indices
    _gather_restore_fused_kernel[(chunk_T,)](
        gathered_k.view(torch.uint8),
        gathered_s,
        out_k_quant.view(torch.uint8),
        out_k_scale,
        restore_indices,
        head_dim,
        scale_dim,
        chunk_T,
        BLOCK_K,
        BLOCK_S,
    )


def fused_scatter_allgather_restore_single(
    *,
    actual: torch.Tensor,
    out: torch.Tensor,
    total_local_T: int,
    src_for_padded: torch.Tensor,
    restore_indices: torch.Tensor,
    fused_dim: int,
) -> None:
    """Fused scatter + allgather + restore for a single tensor (MLA dense KV path).

    Replaces _scatter_actual_to_padded + all_gather + gathered[restore_indices].

    Args:
        actual: Compact actual rows [total_actual_T, fused_dim].
        out: Output buffer [chunk_T, fused_dim].
        total_local_T: Padded local token count (per rank).
        src_for_padded: Inverse map [total_local_T] int64. -1 for zero rows.
        restore_indices: [chunk_T] int64.
        fused_dim: Number of elements per row (NOT bytes).
    """
    device = actual.device
    chunk_T = out.shape[0]
    if chunk_T == 0:
        return

    row_bytes = fused_dim * actual.element_size()
    BLOCK_D = _next_power_of_2(row_bytes)

    # Step 1: fused zero + scatter into padded local buffer
    local_buf = torch.empty(
        (total_local_T, row_bytes), dtype=torch.uint8, device=device
    )
    _fused_zero_scatter_single_kernel[(total_local_T,)](
        actual.view(torch.uint8),
        local_buf,
        src_for_padded,
        row_bytes,
        total_local_T,
        BLOCK_D,
    )

    # Step 2: all_gather with torch.empty
    pg = _get_group(Group.TP)
    world_size = torch.distributed.get_world_size(pg)
    gathered = torch.empty(
        (world_size * total_local_T, row_bytes),
        device=device,
        dtype=torch.uint8,
    )
    torch.distributed.all_gather_into_tensor(gathered, local_buf, group=pg)

    # Step 3: restore gather
    _gather_restore_single_kernel[(chunk_T,)](
        gathered,
        out.view(torch.uint8),
        restore_indices,
        row_bytes,
        chunk_T,
        BLOCK_D,
    )


@triton.jit
def _topk_to_global_indices_kernel(
    topk_ptr,
    req_ids_ptr,
    ws_starts_ptr,
    out_ptr,
    n_rows,
    topk: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    """Fuse topk index → global index conversion.

    Replaces: offsets = ws_starts[req_ids]; raw = topk + offsets; masked_fill(-1).

    Grid: (n_rows,)
    """
    row = tl.program_id(0)
    if row >= n_rows:
        return
    req_id = tl.load(req_ids_ptr + row)
    offset = tl.load(ws_starts_ptr + req_id).to(tl.int32)
    cols = tl.arange(0, BLOCK_TOPK)
    mask = cols < topk
    idx = tl.load(topk_ptr + row * topk + cols, mask=mask, other=-1)
    result = tl.where(idx >= 0, idx + offset, idx)
    tl.store(out_ptr + row * topk + cols, result, mask=mask)


def fused_topk_to_global_indices(
    topk_2d: torch.Tensor,
    precomputed_req_ids: torch.Tensor,
    workspace_starts: torch.Tensor,
) -> torch.Tensor:
    """Convert per-request topk indices to global fused_kv indices.

    Replaces:
        offsets = ws.workspace_starts[precomputed_req_ids]
        padding_mask = topk_2d < 0
        raw_global = topk_2d + offsets.unsqueeze(1)
        global_indices = raw_global.masked_fill(padding_mask, -1).unsqueeze(1)

    Args:
        topk_2d: [T, topk] int32, per-request topk indices (-1 = padding).
        precomputed_req_ids: [T] int64, request id for each token.
        workspace_starts: [B] int32, per-request offset into fused_kv.

    Returns:
        [T, 1, topk] int32, global indices for flash_mla_sparse_fwd.
    """
    T, topk_val = topk_2d.shape
    out = torch.empty_like(topk_2d)
    if T > 0:
        BLOCK_TOPK = _next_power_of_2(topk_val)
        _topk_to_global_indices_kernel[(T,)](
            topk_2d,
            precomputed_req_ids,
            workspace_starts,
            out,
            T,
            topk_val,
            BLOCK_TOPK,
        )
    return out.unsqueeze(1)
