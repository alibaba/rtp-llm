"""Read only immutable prefix K before current K exists; patch new K byte-exactly."""

import torch
import triton
import triton.language as tl


def make_history_plan(prefix, lengths, cp_size, owner_size, device):
    # Host snapshots from this prepare only. No device readback in forward.
    if prefix.device.type != "cpu" or lengths.device.type != "cpu":
        return None
    prefixes, totals = prefix.tolist(), lengths.tolist()
    if len(prefixes) != len(totals) or any(
        p < 0 or t < p for p, t in zip(prefixes, totals)
    ):
        raise ValueError("invalid indexer prefix/total lengths")
    if not totals or sum(prefixes) == 0 or sum(prefixes) < sum(totals) // 2:
        return None
    rows, new_rows, full_rows = 0, 0, 0
    meta = []
    for pre, total in zip(prefixes, totals):
        padded = triton.cdiv(total, cp_size * owner_size) * owner_size
        meta.append(
            [rows, rows + padded, pre, new_rows, new_rows + total - pre, full_rows]
        )
        rows += padded
        new_rows += total - pre
        full_rows += total
    return (
        torch.tensor(meta, dtype=torch.int64, device=device),
        torch.arange(new_rows, dtype=torch.int64).to(device=device),
    )


@triton.jit
def _prefix_pack(
    C,
    BT,
    M,
    O,
    ROWS: tl.constexpr,
    B: tl.constexpr,
    COLS: tl.constexpr,
    BS: tl.constexpr,
    OWNER: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    req = tl.full((BR,), 0, tl.int64)
    local = tl.full((BR,), 0, tl.int64)
    history = tl.full((BR,), False, tl.int1)
    for b in range(B):
        start, end, pre = (
            tl.load(M + b * 6),
            tl.load(M + b * 6 + 1),
            tl.load(M + b * 6 + 2),
        )
        pos = row - start
        logical = (pos // OWNER * CP + RANK) * OWNER + pos % OWNER
        hit = (row >= start) & (row < end) & (row < ROWS)
        req = tl.where(hit, b, req)
        local = tl.where(hit, pos, local)
        history = history | (hit & (logical < pre))
    # Never even resolve a physical page for a new token or padding row.
    block = tl.load(BT + req * COLS + local // BS, history, other=0).to(tl.int64)
    col = tl.arange(0, 256)
    off = tl.where(
        col < 128,
        (local % BS)[:, None] * 128 + col[None, :],
        BS * 128 + (local % BS)[:, None] * 4 + col[None, :] - 128,
    )
    data = tl.load(
        C + block[:, None] * BS * 132 + off,
        history[:, None] & (col[None, :] < 132),
        other=0,
    )
    tl.store(
        O + row[:, None] * 132 + col[None, :],
        data,
        (row[:, None] < ROWS) & (col[None, :] < 132),
    )


def pack_prefix(cache, block_table, plan, metadata):
    if (
        cache.dtype != torch.uint8
        or cache.shape[-1] != 132
        or not cache.is_contiguous()
    ):
        raise ValueError("early indexer history requires contiguous 128+4-byte cache")
    if block_table.ndim != 2 or not block_table.is_contiguous():
        raise ValueError("early indexer history requires a contiguous block table")
    rows = plan.total_local_T
    payload = torch.empty((rows, 132), dtype=torch.uint8, device=cache.device)
    if rows:
        _prefix_pack[(triton.cdiv(rows, 16),)](
            cache,
            block_table,
            metadata,
            payload,
            rows,
            metadata.shape[0],
            block_table.shape[1],
            cache.shape[1],
            plan.owner_block_size,
            plan.cp_ctx.cp_size,
            plan.cp_ctx.cp_rank,
            16,
            num_warps=4,
        )
    return payload


@triton.jit
def _patch_new(
    T,
    C,
    Slots,
    M,
    K,
    S,
    N: tl.constexpr,
    BS: tl.constexpr,
    B: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    dst = tl.full((BR,), 0, tl.int64)
    for b in range(B):
        start, end = tl.load(M + b * 6 + 3), tl.load(M + b * 6 + 4)
        hit = (row >= start) & (row < end)
        dst = tl.where(
            hit, tl.load(M + b * 6 + 5) + tl.load(M + b * 6 + 2) + row - start, dst
        )
    col = tl.arange(0, 256)
    data = tl.load(
        T + row[:, None] * 132 + col[None, :],
        (row[:, None] < N) & (col[None, :] < 132),
        other=0,
    )
    tl.store(
        K + dst[:, None] * 128 + col[None, :],
        data,
        (row[:, None] < N) & (col[None, :] < 128),
    )
    tl.store(
        S + dst[:, None] * 4 + col[None, :] - 128,
        data,
        (row[:, None] < N) & (col[None, :] >= 128) & (col[None, :] < 132),
    )
    slot = tl.load(Slots + row, row < N, other=-1).to(tl.int64)
    offset = slot[:, None] // BS * BS * 132 + tl.where(
        col[None, :] < 128,
        slot[:, None] % BS * 128 + col[None, :],
        BS * 128 + slot[:, None] % BS * 4 + col[None, :] - 128,
    )
    tl.store(
        C + offset,
        data,
        (row[:, None] < N) & (slot[:, None] >= 0) & (col[None, :] < 132),
    )


def patch_new(quantized, cache, slots, metadata, out_k, out_s):
    rows = quantized.shape[0]
    if slots.numel() != rows or slots.dtype != torch.int64 or not slots.is_contiguous():
        raise ValueError("invalid new indexer slot mapping")
    if rows:
        _patch_new[(triton.cdiv(rows, 16),)](
            quantized,
            cache,
            slots,
            metadata,
            out_k.view(torch.uint8),
            out_s,
            rows,
            cache.shape[1],
            metadata.shape[0],
            16,
            num_warps=4,
        )
