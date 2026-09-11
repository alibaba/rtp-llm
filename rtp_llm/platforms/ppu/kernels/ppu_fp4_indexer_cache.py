"""Translate RTP native block resources into SG FP4 Indexer contracts."""

import torch
import triton
import triton.language as tl


@triton.jit
def _plans(
    POS,
    REQ,
    START,
    STATE_SLOTS,
    KV_SLOTS,
    BT,
    C,
    W,
    OUT,
    N: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    VARLEN: tl.constexpr,
    SEQ_START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    pos = tl.load(POS + i, live, 0).to(tl.int32)
    req = tl.load(REQ + i, live, 0).to(tl.int32)
    if VARLEN:
        start = tl.load(START + req, live, 0).to(tl.int32)
    else:
        start = tl.full((BLOCK,), SEQ_START, tl.int32)
    seq_len = pos + 1
    valid = live & (seq_len % 4 == 0)
    slots = tl.load(KV_SLOTS + i, live, -1).to(tl.int32)
    valid = valid & (slots >= 0)
    buffer_len = tl.minimum(tl.maximum(start - (seq_len - 8), 0), 8)
    for part in tl.static_range(2):
        window = seq_len - 8 + part * 4
        logical = window // TOKENS_PER_BLOCK
        need = valid & (window >= 0) & (buffer_len > part * 4)
        physical = tl.load(
            BT + req * BT_STRIDE + logical, need & (logical < MAX_BLOCKS), 0
        ).to(tl.int32)
        page = physical * 2 + ((window % 8) // 4)
        tl.store(C + i * 4 + 2 + part, tl.where(need, page, 0), live)
    tl.store(C + i * 4, tl.where(valid, seq_len, -1), live)
    tl.store(C + i * 4 + 1, i | (buffer_len << 16), live)
    state_slot = tl.load(STATE_SLOTS + i, live, -1).to(tl.int32)
    tl.store(W + i * 2, tl.where(state_slot >= 0, i, -1), live)
    tl.store(W + i * 2 + 1, state_slot, live)
    tl.store(OUT + i, slots, live)


def build_plans(meta, state_bt, state_entries, state_tokens, seq_start):
    n = meta.positions.numel()
    if n > 65536 or state_entries != 8 or state_tokens % 8:
        raise ValueError("SG C4 plans require <=65536 rows and an 8-row native ring")
    varlen = meta.is_batched and meta.seq_start_per_req is not None
    if seq_start is None and not varlen:
        raise ValueError("FP4 Indexer decode is not qualified")
    c = torch.empty((n, 4), dtype=torch.int32, device=meta.positions.device)
    w = torch.empty((n, 2), dtype=torch.int32, device=meta.positions.device)
    slots = torch.empty((n,), dtype=torch.int32, device=meta.positions.device)
    if n:
        _plans[(triton.cdiv(n, 128),)](
            meta.positions,
            meta.b_idx,
            meta.seq_start_per_req if varlen else meta.positions,
            meta.state_slots,
            meta.kv_slots,
            state_bt,
            c,
            w,
            slots,
            n,
            state_bt.stride(0),
            state_bt.shape[1],
            state_tokens,
            varlen,
            int(seq_start or 0),
            128,
        )
    return c.view(torch.uint8), w.view(torch.uint8), slots


@triton.jit
def _decode_plans(
    POS,
    REQ,
    STATE_SLOTS,
    KV_SLOTS,
    BT,
    PLAN,
    OUT,
    N: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = i < N
    pos = tl.load(POS + i, live, 0).to(tl.int32)
    req = tl.load(REQ + i, live, 0).to(tl.int32)
    write = tl.load(STATE_SLOTS + i, live, -1).to(tl.int32)
    slots = tl.load(KV_SLOTS + i, live, -1).to(tl.int32)
    seq = pos + 1
    active = live & (write >= 0) & (pos >= 0)
    tl.store(PLAN + i * 4, tl.where(active, seq, 1), live)
    tl.store(PLAN + i * 4 + 1, tl.where(active, write, -1), live)
    for part in tl.static_range(2):
        window = seq - 8 + part * 4
        block = window // TOKENS_PER_BLOCK
        need = active & (seq % 4 == 0) & (window >= 0) & (block < MAX_BLOCKS)
        physical = tl.load(BT + req * BT_STRIDE + block, need, 0).to(tl.int32)
        page = physical * 2 + (window % 8) // 4
        tl.store(PLAN + i * 4 + 2 + part, tl.where(need, page, 0), live)
    tl.store(OUT + i, tl.where(active, slots, -1), live)


def build_decode_plan(meta, state_bt, state_entries, state_tokens):
    """Map one token per request onto the native eight-row C4 ring."""
    n = meta.positions.numel()
    if state_entries != 8 or state_tokens % 8:
        raise ValueError("FP4 Decode requires an eight-row native state ring")
    if (
        meta.b_idx.numel() != n
        or meta.state_slots.numel() != n
        or meta.kv_slots.numel() != n
    ):
        raise ValueError("FP4 Decode metadata row counts differ")
    plan = torch.empty((n, 4), dtype=torch.int32, device=meta.positions.device)
    slots = torch.empty(n, dtype=torch.int32, device=meta.positions.device)
    if n:
        _decode_plans[(triton.cdiv(n, 128),)](
            meta.positions,
            meta.b_idx,
            meta.state_slots,
            meta.kv_slots,
            state_bt,
            plan,
            slots,
            n,
            state_bt.stride(0),
            state_bt.shape[1],
            state_tokens,
            128,
        )
    return plan.view(torch.uint8), slots


@triton.jit
def _gather(
    CACHE,
    BT,
    CU,
    K,
    SF,
    T: tl.constexpr,
    B: tl.constexpr,
    BT_STRIDE: tl.constexpr,
    EB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = t < T
    req = tl.full((BLOCK,), 0, tl.int32)
    local = t
    for b in range(B):
        begin = tl.load(CU + b)
        end = tl.load(CU + b + 1)
        here = (t >= begin) & (t < end)
        req = tl.where(here, b, req)
        local = tl.where(here, t - begin, local)
    physical = tl.load(BT + req * BT_STRIDE + local // EB, live, 0)
    slot = local % EB
    d = tl.arange(0, 64)
    payload = tl.load(
        CACHE + physical[:, None] * (EB * 68) + slot[:, None] * 64 + d[None, :],
        live[:, None],
        0,
    )
    tl.store(K + t[:, None] * 64 + d[None, :], payload, live[:, None])
    sfbase = CACHE + physical * (EB * 68) + EB * 64 + slot * 4
    scale = tl.full((BLOCK,), 0, tl.uint32)
    for j in tl.static_range(4):
        byte = tl.load(sfbase + j, live, 0).to(tl.uint32)
        scale = scale | (byte << (8 * j))
    tl.store(SF + t, scale, live)


def gather_k(cache, block_table, cu_kv_seqlens, total, entries):
    if cache.dtype != torch.uint8 or not cache.is_contiguous() or cache.shape[-1] != 68:
        raise ValueError("FP4 gather requires native contiguous 68B cache resources")
    k = torch.empty((total, 64), device=cache.device, dtype=torch.int8)
    sf = torch.empty((total, 1), device=cache.device, dtype=torch.int32)
    if total:
        _gather[(triton.cdiv(total, 16),)](
            cache,
            block_table,
            cu_kv_seqlens,
            k,
            sf,
            total,
            cu_kv_seqlens.numel() - 1,
            block_table.stride(0),
            entries,
            16,
        )
    return k, sf
