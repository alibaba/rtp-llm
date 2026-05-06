"""Triton helpers for DSV4 CP/pool metadata hot paths."""

from __future__ import annotations

from typing import Optional, Union

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
    def _cp_window_topk_kernel(
        global_pos_ptr,
        out_ptr,
        S: tl.constexpr,
        WINDOW: tl.constexpr,
        SEQ_LEN_TOTAL: tl.constexpr,
        W_ACTIVE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        s = tl.program_id(1)
        tile = tl.program_id(2)
        cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        g = tl.load(global_pos_ptr + s).to(tl.int64)
        window_start = tl.maximum(g - W_ACTIVE + 1, 0)
        vals = window_start + cols
        invalid = (
            (cols >= WINDOW)
            | (cols >= W_ACTIVE)
            | (vals > g)
            | (vals >= SEQ_LEN_TOTAL)
        )
        vals = tl.where(invalid, -1, vals)
        tl.store(
            out_ptr + (b * S + s) * WINDOW + cols,
            vals,
            mask=cols < WINDOW,
        )

    @triton.jit
    def _cp_compress_topk_kernel(
        global_pos_ptr,
        out_ptr,
        S: tl.constexpr,
        T_COMP: tl.constexpr,
        RATIO: tl.constexpr,
        OFFSET: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        s = tl.program_id(1)
        tile = tl.program_id(2)
        cols = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        g = tl.load(global_pos_ptr + s).to(tl.int64)
        max_allowed = (g + 1) // RATIO
        valid = (cols < T_COMP) & (cols < max_allowed)
        vals = tl.where(valid, cols.to(tl.int64) + OFFSET, -1)
        tl.store(out_ptr + (b * S + s) * T_COMP + cols, vals, mask=cols < T_COMP)

    @triton.jit
    def _pool_slots_kernel(
        block_table_ptr,
        valid_ptr,
        safe_slot_ptr,
        B: tl.constexpr,
        T: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        EB: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        pos = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        in_capacity = pos < (MAX_BLOCKS * EB)
        safe_pos = tl.where(in_capacity, pos, 0)
        block_in_seq = safe_pos // EB
        in_block = safe_pos - block_in_seq * EB
        block_id = tl.load(
            block_table_ptr + b * MAX_BLOCKS + block_in_seq,
            mask=pos < T,
            other=0,
        ).to(tl.int64)
        valid = (pos < T) & in_capacity & (block_id > 0)
        slot = tl.where(valid, block_id * EB + in_block, 0)
        off = b * T + pos
        tl.store(valid_ptr + off, valid, mask=pos < T)
        tl.store(safe_slot_ptr + off, slot, mask=pos < T)

    @triton.jit
    def _swa_pool_slot_mapping_kernel(
        block_table_ptr,
        sp_ptr,
        seq_ptr,
        out_ptr,
        B: tl.constexpr,
        T: tl.constexpr,
        MAX_BLOCKS: tl.constexpr,
        EB: tl.constexpr,
        HAS_SP_TENSOR: tl.constexpr,
        HAS_SEQ_TENSOR: tl.constexpr,
        SP_VALUE: tl.constexpr,
        SEQ_VALUE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        b = tl.program_id(0)
        tile = tl.program_id(1)
        j = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        if HAS_SP_TENSOR:
            sp = tl.load(sp_ptr + b).to(tl.int64)
        else:
            sp = SP_VALUE
        if HAS_SEQ_TENSOR:
            seq = tl.load(seq_ptr + b).to(tl.int64)
        else:
            seq = SEQ_VALUE
        global_pos = sp + j
        block_in_seq = global_pos // EB
        in_block = global_pos - block_in_seq * EB
        in_capacity = block_in_seq < MAX_BLOCKS
        safe_block = tl.where(in_capacity, block_in_seq, 0)
        block_id = tl.load(
            block_table_ptr + b * MAX_BLOCKS + safe_block,
            mask=j < T,
            other=0,
        ).to(tl.int64)
        valid = (j < T) & (j < seq) & in_capacity & (block_id > 0)
        slot = tl.where(valid, block_id * EB + in_block, -1)
        tl.store(out_ptr + b * T + j, slot, mask=j < T)


def _require_cuda_triton(op: str, ref: torch.Tensor) -> None:
    if not _TRITON_AVAILABLE or not ref.is_cuda:
        raise RuntimeError(f"{op} requires CUDA Triton metadata helper")


def build_cp_window_topk_idxs(
    global_positions: torch.Tensor,
    *,
    bsz: int,
    seq_len_total: int,
    window_size: int,
) -> torch.Tensor:
    _require_cuda_triton("build_cp_window_topk_idxs", global_positions)
    if window_size < 0:
        raise ValueError(f"window_size must be non-negative, got {window_size}")
    S = int(global_positions.numel())
    out = torch.empty(
        (int(bsz), S, int(window_size)),
        dtype=torch.long,
        device=global_positions.device,
    )
    if S == 0 or bsz == 0 or window_size == 0:
        return out
    BLOCK_N = 256
    w_active = min(int(window_size), max(int(seq_len_total), 1))
    grid = (int(bsz), S, triton.cdiv(int(window_size), BLOCK_N))
    _cp_window_topk_kernel[grid](
        global_positions.contiguous(),
        out,
        S=S,
        WINDOW=int(window_size),
        SEQ_LEN_TOTAL=int(seq_len_total),
        W_ACTIVE=w_active,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out


def build_cp_compress_topk_idxs(
    global_positions: torch.Tensor,
    *,
    bsz: int,
    seq_len_total: int,
    ratio: int,
    offset: int,
) -> torch.Tensor:
    _require_cuda_triton("build_cp_compress_topk_idxs", global_positions)
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    S = int(global_positions.numel())
    T_comp = max(int(seq_len_total) // int(ratio), 0)
    out = torch.empty(
        (int(bsz), S, T_comp),
        dtype=torch.long,
        device=global_positions.device,
    )
    if S == 0 or bsz == 0 or T_comp == 0:
        return out
    BLOCK_N = 256
    grid = (int(bsz), S, triton.cdiv(T_comp, BLOCK_N))
    _cp_compress_topk_kernel[grid](
        global_positions.contiguous(),
        out,
        S=S,
        T_COMP=T_comp,
        RATIO=int(ratio),
        OFFSET=int(offset),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out


def build_pool_slots(
    block_table: torch.Tensor,
    *,
    bsz: int,
    T: int,
    eb: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    if not _TRITON_AVAILABLE or not block_table.is_cuda:
        return None
    if bsz <= 0 or T <= 0:
        shape = (int(bsz), int(T))
        return (
            torch.empty(shape, dtype=torch.bool, device=block_table.device),
            torch.empty(shape, dtype=torch.long, device=block_table.device),
        )
    max_blocks = int(block_table.shape[1])
    valid = torch.empty(
        (int(bsz), int(T)),
        dtype=torch.bool,
        device=block_table.device,
    )
    safe_slot = torch.empty(
        (int(bsz), int(T)),
        dtype=torch.long,
        device=block_table.device,
    )
    BLOCK_N = 256
    grid = (int(bsz), triton.cdiv(int(T), BLOCK_N))
    _pool_slots_kernel[grid](
        block_table.contiguous(),
        valid,
        safe_slot,
        B=int(bsz),
        T=int(T),
        MAX_BLOCKS=max_blocks,
        EB=int(eb),
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return valid, safe_slot


def build_swa_pool_slot_mapping(
    block_table: torch.Tensor,
    *,
    bsz: int,
    T: int,
    eb: int,
    sp: Union[int, torch.Tensor],
    row_seqlens: Union[int, torch.Tensor, None],
) -> Optional[torch.Tensor]:
    if not _TRITON_AVAILABLE or not block_table.is_cuda:
        return None
    if bsz <= 0 or T <= 0:
        return torch.empty(
            (int(bsz) * int(T),),
            dtype=torch.long,
            device=block_table.device,
        )
    has_sp_tensor = isinstance(sp, torch.Tensor)
    has_seq_tensor = isinstance(row_seqlens, torch.Tensor)
    if has_sp_tensor:
        sp_ptr = sp.to(device=block_table.device, dtype=torch.long).reshape(-1)
        if sp_ptr.numel() == 1 and bsz > 1:
            sp_ptr = sp_ptr.expand(int(bsz)).contiguous()
        else:
            sp_ptr = sp_ptr.contiguous()
    else:
        sp_ptr = block_table
    if has_seq_tensor:
        seq_ptr = row_seqlens.to(device=block_table.device, dtype=torch.long).reshape(-1)
        if seq_ptr.numel() == 1 and bsz > 1:
            seq_ptr = seq_ptr.expand(int(bsz)).contiguous()
        else:
            seq_ptr = seq_ptr.contiguous()
    else:
        seq_ptr = block_table
    sp_value = 0 if has_sp_tensor else int(sp)
    seq_value = (
        int(T)
        if row_seqlens is None
        else (0 if has_seq_tensor else int(row_seqlens))
    )
    out = torch.empty((int(bsz), int(T)), dtype=torch.long, device=block_table.device)
    BLOCK_N = 256
    grid = (int(bsz), triton.cdiv(int(T), BLOCK_N))
    _swa_pool_slot_mapping_kernel[grid](
        block_table.contiguous(),
        sp_ptr,
        seq_ptr,
        out,
        B=int(bsz),
        T=int(T),
        MAX_BLOCKS=int(block_table.shape[1]),
        EB=int(eb),
        HAS_SP_TENSOR=has_sp_tensor,
        HAS_SEQ_TENSOR=has_seq_tensor,
        SP_VALUE=sp_value,
        SEQ_VALUE=seq_value,
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out.reshape(-1)
