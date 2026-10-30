"""Fused Triton kernels for CP-sharded indexer K assemble.

Replaces the element-wise kernel sequences in:
  1. copy_actual_indexer_k_to_padded (scatter actual rows → padded layout)
  2. assemble_indexer_k post-allgather restore (gather by restore_indices)
"""

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import all_gather
from rtp_llm.models_py.distributed.pynccl_cp import enabled as cp_opt_enabled


@triton.jit
def _pack_padded_k_scale_kernel(
    K,
    S,
    Map,
    Out,
    KCOL: tl.constexpr,
    SCOL: tl.constexpr,
    ACTUAL: tl.constexpr,
    USE_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    src = tl.load(Map + row).to(tl.int64) if USE_MAP else row
    col = tl.arange(0, BLOCK)
    valid = (src >= 0) & (src < ACTUAL)
    kval = tl.load(K + src * KCOL + col, mask=valid & (col < KCOL), other=0)
    sval = tl.load(
        S + src * SCOL + col - KCOL,
        mask=valid & (col >= KCOL) & (col < KCOL + SCOL),
        other=0,
    )
    tl.store(
        Out + row * (KCOL + SCOL) + col,
        tl.where(col < KCOL, kval, sval),
        mask=col < KCOL + SCOL,
    )


def _pack_padded_k_scale(k, scale, rows, src_for_padded=None):
    """Byte-exact padding and K/scale packing in one launch, without temporaries."""
    if (
        k.ndim != 2
        or scale.ndim != 2
        or k.size(0) != scale.size(0)
        or k.device != scale.device
        or k.element_size() != 1
        or scale.element_size() != 1
        or not k.is_contiguous()
        or not scale.is_contiguous()
        or rows < 0
    ):
        raise ValueError("invalid contiguous one-byte K/scale payload")
    if src_for_padded is not None and (
        src_for_padded.ndim != 1
        or src_for_padded.numel() != rows
        or src_for_padded.device != k.device
        or src_for_padded.dtype != torch.int64
        or not src_for_padded.is_contiguous()
    ):
        raise ValueError("invalid padded K row map")
    if src_for_padded is None and k.size(0) not in (0, rows):
        raise ValueError("nonempty padded payload requires a row map")
    width = k.size(1) + scale.size(1)
    out = torch.empty((rows, width), dtype=torch.uint8, device=k.device)
    if rows:
        _pack_padded_k_scale_kernel[(rows,)](
            k.view(torch.uint8),
            scale.view(torch.uint8),
            src_for_padded,
            out,
            k.size(1),
            scale.size(1),
            k.size(0),
            src_for_padded is not None,
            triton.next_power_of_2(width),
        )
    return out


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
    # Widen before multiplying by row strides, including destination offsets.
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= n_padded_rows:
        return
    src_row = tl.load(src_for_padded_ptr + row_id).to(tl.int64)

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
    gathered_packed_ptr,
    out_k_ptr,
    out_s_ptr,
    indices_ptr,
    packed_stride: tl.constexpr,
    k_stride: tl.constexpr,
    s_stride: tl.constexpr,
    n_rows,
    BLOCK_K: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    """Restore K and scale from a row-packed AllGather payload.

    Grid: (n_rows,)
    Each program copies one row from ``[K bytes | scale bytes]`` to outputs.
    """
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= n_rows:
        return
    src_row = tl.load(indices_ptr + row_id).to(tl.int64)

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < k_stride
    k_data = tl.load(
        gathered_packed_ptr + src_row * packed_stride + k_cols, mask=k_mask
    )
    tl.store(out_k_ptr + row_id * k_stride + k_cols, k_data, mask=k_mask)

    s_cols = tl.arange(0, BLOCK_S)
    s_mask = s_cols < s_stride
    s_data = tl.load(
        gathered_packed_ptr + src_row * packed_stride + k_stride + s_cols,
        mask=s_mask,
    )
    tl.store(out_s_ptr + row_id * s_stride + s_cols, s_data, mask=s_mask)


@triton.jit
def _gather_restore_tiled_kernel(
    P,
    K,
    S,
    I,
    ROWS,
    PACK: tl.constexpr,
    KD: tl.constexpr,
    SD: tl.constexpr,
    BK: tl.constexpr,
    BS: tl.constexpr,
    BR: tl.constexpr,
):
    rows = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    valid = rows < ROWS
    src = tl.load(I + rows, valid, other=0).to(tl.int64)
    col = tl.arange(0, BK)
    val = tl.load(
        P + src[:, None] * PACK + col[None, :],
        valid[:, None] & (col[None, :] < KD),
        other=0,
    )
    tl.store(
        K + rows[:, None] * KD + col[None, :], val, valid[:, None] & (col[None, :] < KD)
    )
    sc = tl.arange(0, BS)
    val = tl.load(
        P + src[:, None] * PACK + KD + sc[None, :],
        valid[:, None] & (sc[None, :] < SD),
        other=0,
    )
    tl.store(
        S + rows[:, None] * SD + sc[None, :], val, valid[:, None] & (sc[None, :] < SD)
    )


def _restore_packed_indexer_k(packed, out_k, out_s, indices):
    """Byte-preserving restore; tile rows without specializing on sequence length."""
    rows, k_cols = out_k.shape
    s_cols = out_s.shape[1]
    if rows == 0:
        return
    block_k = triton.next_power_of_2(k_cols)
    block_s = triton.next_power_of_2(s_cols)
    if (
        os.environ.get("RTP_LLM_CP_TILED_INDEXER_RESTORE", "0") == "1"
        and rows >= 4096
        and k_cols == 128
        and s_cols in (1, 4)
    ):
        _gather_restore_tiled_kernel[(triton.cdiv(rows, 16),)](
            packed,
            out_k,
            out_s,
            indices,
            rows,
            k_cols + s_cols,
            k_cols,
            s_cols,
            block_k,
            block_s,
            16,
            num_warps=4,
        )
    else:
        _gather_restore_fused_kernel[(rows,)](
            packed,
            out_k,
            out_s,
            indices,
            k_cols + s_cols,
            k_cols,
            s_cols,
            rows,
            block_k,
            block_s,
        )


@triton.jit
def _gather_restore_separate_kernel(
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
    """Restore K and scale from two separately gathered tensors."""
    row_id = tl.program_id(0).to(tl.int64)
    if row_id >= n_rows:
        return
    src_row = tl.load(indices_ptr + row_id).to(tl.int64)

    k_cols = tl.arange(0, BLOCK_K)
    k_mask = k_cols < k_stride
    k_data = tl.load(gathered_k_ptr + src_row * k_stride + k_cols, mask=k_mask)
    tl.store(out_k_ptr + row_id * k_stride + k_cols, k_data, mask=k_mask)

    s_cols = tl.arange(0, BLOCK_S)
    s_mask = s_cols < s_stride
    s_data = tl.load(gathered_s_ptr + src_row * s_stride + s_cols, mask=s_mask)
    tl.store(out_s_ptr + row_id * s_stride + s_cols, s_data, mask=s_mask)


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


def _pack_indexer_k_payload(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
):
    """Pack one-byte FP8 K and scale rows for a single NCCL AllGather."""
    if local_k_quant.dim() != 2 or local_k_scale.dim() != 2:
        raise ValueError(
            "indexer K pack expects 2D quant/scale buffers, got "
            f"{tuple(local_k_quant.shape)} and {tuple(local_k_scale.shape)}"
        )
    if local_k_quant.device != local_k_scale.device:
        raise ValueError("indexer K pack requires quant/scale on the same device")
    if local_k_quant.shape[0] != local_k_scale.shape[0]:
        raise ValueError(
            "indexer K pack row mismatch: "
            f"quant={local_k_quant.shape[0]}, scale={local_k_scale.shape[0]}"
        )
    if local_k_quant.element_size() != 1 or local_k_scale.element_size() != 1:
        raise ValueError(
            "indexer K pack requires one-byte quant/scale elements, got "
            f"{local_k_quant.dtype} and {local_k_scale.dtype}"
        )

    local_q_bytes = local_k_quant.contiguous().view(torch.uint8)
    local_s_bytes = local_k_scale.contiguous().view(torch.uint8)
    rows = int(local_q_bytes.shape[0])
    q_cols = int(local_q_bytes.shape[1])
    s_cols = int(local_s_bytes.shape[1])
    packed = torch.empty(
        (rows, q_cols + s_cols),
        dtype=torch.uint8,
        device=local_k_quant.device,
    )
    packed[:, :q_cols].copy_(local_q_bytes)
    packed[:, q_cols:].copy_(local_s_bytes)
    return packed, q_cols, s_cols


def _all_gather_indexer_k_payload(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
    group,
):
    """Pack K+scale and issue exactly one rank-major AllGather."""
    local_packed, q_cols, s_cols = _pack_indexer_k_payload(local_k_quant, local_k_scale)
    # This role is deliberately absent from the SYMM whitelist. Long-context
    # packed payloads can exceed a persistent window, while ordinary pynccl
    # still removes the second collective launch.
    gathered_packed = all_gather(local_packed, group=group, role="indexer_k_packed")
    return gathered_packed, q_cols, s_cols


def _all_gather_indexer_k(
    local_k_quant: torch.Tensor,
    local_k_scale: torch.Tensor,
    group,
):
    """Gather Indexer K using the master switch's packed or baseline path."""
    q_cols = int(local_k_quant.shape[1])
    s_cols = int(local_k_scale.shape[1])
    if cp_opt_enabled():
        gathered, q_cols, s_cols = _all_gather_indexer_k_payload(
            local_k_quant, local_k_scale, group
        )
        return gathered, None, q_cols, s_cols

    gathered_k = all_gather(local_k_quant, group=group, role="indexer_k_quant")
    gathered_s = all_gather(local_k_scale, group=group, role="indexer_k_scale")
    return gathered_k, gathered_s, q_cols, s_cols


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

    # Write the final wire layout directly. This replaces padding/scatter and
    # two strided copy_ launches; the communicated bytes and restore stay exact.
    packed_direct = cp_opt_enabled() and (
        total_actual_T == total_local_T
        or total_actual_T == 0
        or src_for_padded is not None
    )
    if packed_direct:
        payload = _pack_padded_k_scale(
            actual_k_quant,
            actual_k_scale,
            total_local_T,
            src_for_padded if total_actual_T != total_local_T else None,
        )
        gathered_k = all_gather(payload, group=Group.TP, role="indexer_k_packed")
        gathered_s = None
        q_cols, s_cols = head_dim, scale_dim
    else:
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

        # Step 2: GLM5_CP_OPT packs K and scale into one collective. With the
        # master switch disabled, preserve the original two-gather behavior.
        gathered_k, gathered_s, q_cols, s_cols = _all_gather_indexer_k(
            local_k, local_s, Group.TP
        )
    if q_cols != head_dim or s_cols != scale_dim:
        raise RuntimeError(
            "packed indexer K layout mismatch: "
            f"q_cols={q_cols}/{head_dim}, scale_cols={s_cols}/{scale_dim}"
        )

    # Step 3: keep the existing fused local restore in both communication modes.
    restore_indices = plan.restore_indices
    if gathered_s is None:
        _restore_packed_indexer_k(
            gathered_k, out_k_quant.view(torch.uint8), out_k_scale, restore_indices
        )
    else:
        _gather_restore_separate_kernel[(chunk_T,)](
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
