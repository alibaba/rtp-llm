"""Transactional, zero-prefix QSA prefill for the released BF16 layout."""

import itertools

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)

from .abi import PREFILL_COMPRESS_RATIO, PREFILL_HEAD_DIM, PREFILL_STATE_RING_ROWS

STATE_RING_ROWS = tl.constexpr(PREFILL_STATE_RING_ROWS)
INDEXER_HEAD_DIM = tl.constexpr(PREFILL_HEAD_DIM)
COMPRESS_RATIO = tl.constexpr(PREFILL_COMPRESS_RATIO)

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


@triton.jit
def _state_snapshot(
    cu,
    table,
    pool,
    slots,
    original,
    # WIDTH/PAGES track the per-request table geometry; constexpr keys here
    # recompiled the kernel for every new request length.
    WIDTH,
    PAGES,
    PAGE: tl.constexpr,
):
    request, page = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * PAGE, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    valid = position >= page * PAGE
    block = tl.load(table + request * WIDTH + page, mask=page * PAGE < length, other=0)
    slot = block * 8 + r
    row = (request * PAGES + page) * 8 + r
    value = tl.load(
        pool + slot[:, None] * 128 + d[None, :], mask=valid[:, None], other=0
    )
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row[:, None] * 128 + d[None, :], value)


@triton.jit
def _kv_snapshot(
    cu,
    table,
    pool,
    slots,
    original,
    WIDTH,
    GROUPS,
    PAGE: tl.constexpr,
):
    request, group = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    valid = group < length // 4
    block = tl.load(table + request * WIDTH + group * 4 // PAGE, mask=valid, other=0)
    slot = block * (PAGE // 4) + group % (PAGE // 4)
    d = tl.arange(0, INDEXER_HEAD_DIM)
    value = tl.load(pool + slot * 128 + d, mask=valid, other=0)
    row = request * GROUPS + group
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row * 128 + d, value)


@triton.jit
def _state_write(
    raw,
    cu,
    slots,
    pool,
    PAGES,
    PAGE: tl.constexpr,
):
    request, page = tl.program_id(0), tl.program_id(1)
    start = tl.load(cu + request)
    length = tl.load(cu + request + 1) - start
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * PAGE, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    row = (request * PAGES + page) * 8 + r
    slot = tl.load(slots + row)
    value = tl.load(
        raw + (start + position[:, None]) * 128 + d[None, :],
        mask=(slot >= 0)[:, None],
        other=0,
    ).to(tl.float32)
    tl.store(pool + slot[:, None] * 128 + d[None, :], value, mask=(slot >= 0)[:, None])


@triton.jit
def _kv_write(
    raw,
    cu,
    cos,
    sin,
    gamma,
    slots,
    pool,
    dense_keys,
    GROUPS,
    MAX_LENGTH,
    EPS: tl.constexpr,
    EMIT_DENSE: tl.constexpr,
):
    request, group = tl.program_id(0), tl.program_id(1)
    slot = tl.load(slots + request * GROUPS + group)
    if slot >= 0:
        start = tl.load(cu + request)
        r, d = tl.arange(0, 4), tl.arange(0, INDEXER_HEAD_DIM)
        value = tl.load(raw + (start + group * 4 + r[:, None]) * 128 + d[None, :]).to(
            tl.float32
        )
        pooled = (tl.sum(value, 0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, 0) / 128 + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)
        partner = tl.where(d < 64, tl.where(d < 32, d + 32, d - 32), d)
        rotated = tl.gather(normalized, partner, 0)
        rope_row = (request * MAX_LENGTH + group * 4) * 64
        c = tl.load(cos + rope_row + d, mask=d < 64, other=1).to(tl.float32)
        s = tl.load(sin + rope_row + d, mask=d < 64, other=0).to(tl.float32)
        left = (normalized.to(tl.float32) * c).to(tl.bfloat16)
        right = (rotated.to(tl.float32) * s * tl.where(d < 32, -1.0, 1.0)).to(
            tl.bfloat16
        )
        output = tl.where(
            d < 64,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )
        tl.store(pool + slot * 128 + d, output)
        if EMIT_DENSE:
            # Selection's standalone CUDA mean uses 32 lanes, each reducing
            # four adjacent squared FP32 values. Keep cache bytes above
            # unchanged and reproduce that consumer's reduction separately.
            lane = tl.arange(0, 32)
            a = tl.gather(pooled, lane * 4, 0)
            b = tl.gather(pooled, lane * 4 + 1, 0)
            c0 = tl.gather(pooled, lane * 4 + 2, 0)
            d0 = tl.gather(pooled, lane * 4 + 3, 0)
            variance = tl.sum(((a * a + b * b) + c0 * c0) + d0 * d0, 0) / 128
            dense_norm = (pooled * tl.rsqrt(variance + EPS) * gain).to(tl.bfloat16)
            partner_value = tl.gather(dense_norm, partner, 0)
            left = (dense_norm.to(tl.float32) * c).to(tl.bfloat16)
            right = (partner_value.to(tl.float32) * s * tl.where(d < 32, -1.0, 1.0)).to(
                tl.bfloat16
            )
            dense = tl.where(
                d < 64,
                (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
                dense_norm,
            )
            tl.store(dense_keys + (request * GROUPS + group) * 128 + d, dense)


@triton.jit
def _metadata(
    cu,
    state_table,
    kv_table,
    state_slots,
    kv_slots,
    completed,
    STATE_WIDTH,
    KV_WIDTH,
    PAGE: tl.constexpr,
):
    request = tl.program_id(0)
    position = tl.program_id(1) * 128 + tl.arange(0, INDEXER_HEAD_DIM)
    start, end = tl.load(cu + request), tl.load(cu + request + 1)
    live = position < end - start
    state_block = tl.load(
        state_table + request * STATE_WIDTH + position // PAGE, mask=live, other=0
    )
    done = (position + 1) % 4 == 0
    kv_block = tl.load(
        kv_table + request * KV_WIDTH + position // PAGE, mask=live & done, other=0
    )
    tl.store(state_slots + start + position, state_block * 8 + position % 8, mask=live)
    tl.store(
        kv_slots + start + position,
        tl.where(done, kv_block * (PAGE // 4) + position // 4 % (PAGE // 4), -1),
        mask=live,
    )
    tl.store(completed + start + position, done, mask=live)


@triton.jit
def _restore_slots(pool, slots, original):
    row = tl.program_id(0)
    slot = tl.load(slots + row)
    d = tl.arange(0, INDEXER_HEAD_DIM)
    value = tl.load(original + row * 128 + d)
    tl.store(pool + slot * 128 + d, value, mask=slot >= 0)


def restore_padded_indexer_cache(undo: IndexerCacheUndo) -> None:
    for pool, slots, original in (
        (undo.state_pool, undo.state_slots, undo.original_state),
        (undo.kv_pool, undo.kv_slots, undo.original_kv),
    ):
        if slots.numel():
            _restore_slots[(slots.numel(),)](pool, slots, original, num_warps=4)


def _check_host_mirrors(lengths, cu, tables, device_tables, pool_sizes, page_size):
    """Validate C++'s CPU sources of the device metadata without a GPU readback.

    None means that a device-only path must use the GPU preflight below. C++
    constructs each device tensor with tensorHoldHostAndToCuda from this CPU
    tensor in the ordinary eager prefill path.
    """
    batch = len(lengths)
    mirrors = (cu, *tables)
    if not all(
        isinstance(t, torch.Tensor)
        and t.device.type == "cpu"
        and t.dtype == torch.int32
        for t in mirrors
    ):
        return None
    if cu.shape != (batch + 1,) or any(
        host.shape != device.shape for host, device in zip(tables, device_tables)
    ):
        return None
    if cu.tolist() != [0, *itertools.accumulate(lengths)]:
        raise ValueError("QSA prefill has invalid cu_seqlens or physical block ids")
    aliased = False
    for table, pool_size, ratio in zip(tables, pool_sizes, (1, 4)):
        seen = set()
        for row, length in zip(table.tolist(), lengths):
            required = triton.cdiv(length // ratio * ratio, page_size)
            for block in row[:required]:
                if block <= 0 or block >= pool_size:
                    raise ValueError(
                        "QSA prefill has invalid cu_seqlens or physical block ids"
                    )
                if block in seen:
                    aliased = True
                seen.add(block)
    return aliased


def write_zero_prefix_prefill(
    raw_keys,
    cu_seqlens,
    lengths,
    rope_cos,
    rope_sin,
    gamma,
    kv_pool,
    kv_table,
    state_pool,
    state_table,
    *,
    page_size,
    norm_eps,
    emit_dense_keys=False,
    host_cu_seqlens=None,
    host_kv_table=None,
    host_state_table=None,
) -> dict | None:
    """Return None for unsupported geometry or aliased pages, before any write.

    The caller validates zero prefixes and position/RoPE semantics. This writer
    validates the host partition against device cu_seqlens and every required
    physical page. Snapshots of BOTH pools finish before the first mutation.
    """
    batch, maximum = len(lengths), max(lengths, default=0)
    device = raw_keys.device
    if not (
        batch > 0
        and maximum > 0
        and raw_keys.is_cuda
        and torch.version.hip is None
        and page_size in (64, 128)
        and raw_keys.dtype == torch.bfloat16
        and raw_keys.shape == (sum(lengths), 128)
        and all(isinstance(n, int) and n > 0 for n in lengths)
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.shape == (batch + 1,)
        and rope_cos.shape == rope_sin.shape == (batch, maximum, 64)
        and rope_cos.dtype == rope_sin.dtype == torch.bfloat16
        and gamma.shape == (128,)
        and gamma.dtype in (torch.bfloat16, torch.float32)
        and kv_pool.dim() == state_pool.dim() == 3
        and kv_pool.shape[1:] == (page_size // 4, 128)
        and kv_pool.dtype == torch.bfloat16
        and state_pool.shape[1:] == (8, 128)
        and state_pool.dtype == torch.float32
        and kv_table.dim() == state_table.dim() == 2
        and kv_table.shape[0] == state_table.shape[0] == batch
        and kv_table.dtype == state_table.dtype == torch.int32
        and all(
            t.device == device and t.is_contiguous()
            for t in (
                cu_seqlens,
                rope_cos,
                rope_sin,
                gamma,
                kv_pool,
                kv_table,
                state_pool,
                state_table,
            )
        )
    ):
        return None
    pages, groups = triton.cdiv(maximum, page_size), maximum // 4
    if state_table.shape[1] < pages or kv_table.shape[1] < triton.cdiv(
        groups * 4, page_size
    ):
        raise ValueError("QSA prefill block table does not cover required pages")
    host_alias = _check_host_mirrors(
        lengths,
        host_cu_seqlens,
        (host_state_table, host_kv_table),
        (state_table, kv_table),
        (state_pool.shape[0], kv_pool.shape[0]),
        page_size,
    )
    if host_alias is None:
        # Device-only metadata (including MTP and CUDA graph paths) retains the
        # strict GPU preflight before any side-cache mutation.
        counts = (
            [triton.cdiv(n, page_size) for n in lengths],
            [triton.cdiv(n // 4 * 4, page_size) for n in lengths],
        )
        metadata = torch.tensor(
            [0, *itertools.accumulate(lengths), *counts[0], *counts[1]],
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).to(device=device, non_blocking=True)
        invalid = torch.any(cu_seqlens != metadata[: batch + 1])
        aliases = torch.zeros((), dtype=torch.bool, device=device)
        for index, (table, pool) in enumerate(
            ((state_table, state_pool), (kv_table, kv_pool))
        ):
            offset = batch + 1 + index * batch
            required = (
                torch.arange(table.shape[1], device=device)[None, :]
                < metadata[offset : offset + batch, None]
            )
            invalid = invalid | torch.any(
                required & ((table <= 0) | (table >= pool.shape[0]))
            )
            ids = (
                torch.where(required, table, torch.iinfo(torch.int32).max)
                .flatten()
                .sort()
                .values
            )
            aliases = aliases | torch.any(
                (ids[1:] == ids[:-1]) & (ids[1:] != torch.iinfo(torch.int32).max)
            )
        if qsa_strict_validation_enabled():
            verdict = int(
                (invalid.to(torch.int32) + 2 * aliases.to(torch.int32)).item()
            )
            if verdict & 1:
                raise ValueError(
                    "QSA prefill has invalid cu_seqlens or physical block ids"
                )
            if verdict & 2:
                return None
        else:
            # Engine/allocator invariants trap the device fail-stop at the next
            # host synchronization instead of raising here; both asserts are
            # enqueued before the first pool mutation below. Production eager
            # callers reach the host-mirror branch above, which still declines
            # aliased pages (vectorized fallback) without any readback.
            qsa_assert_valid(
                ~invalid,
                "QSA prefill has invalid cu_seqlens or physical block ids",
            )
            qsa_assert_valid(
                ~aliases,
                "QSA prefill has aliased physical block ids",
            )
    elif host_alias:
        return None

    # index_qk_proj exposes K as a strided slice of the joint Q/K output.
    raw_keys = raw_keys.contiguous()
    dense_keys = (
        torch.empty((batch, groups, 128), device=device, dtype=torch.bfloat16)
        if emit_dense_keys
        else kv_pool
    )
    state_slots = torch.empty(batch * pages * 8, dtype=torch.int64, device=device)
    kv_slots = torch.empty(batch * groups, dtype=torch.int64, device=device)
    original_state = torch.empty(
        (state_slots.numel(), 128), dtype=torch.float32, device=device
    )
    original_kv = torch.empty(
        (kv_slots.numel(), 128), dtype=torch.bfloat16, device=device
    )
    undo = IndexerCacheUndo(
        state_pool,
        state_slots,
        original_state,
        kv_pool,
        kv_slots,
        original_kv,
        padded_slots=True,
    )
    _state_snapshot[(batch, pages)](
        cu_seqlens,
        state_table,
        state_pool,
        state_slots,
        original_state,
        WIDTH=state_table.shape[1],
        PAGES=pages,
        PAGE=page_size,
        num_warps=4,
    )
    if groups:
        _kv_snapshot[(batch, groups)](
            cu_seqlens,
            kv_table,
            kv_pool,
            kv_slots,
            original_kv,
            WIDTH=kv_table.shape[1],
            GROUPS=groups,
            PAGE=page_size,
            num_warps=4,
        )
    token_state_slots = torch.empty(raw_keys.shape[0], dtype=torch.int64, device=device)
    token_kv_slots = torch.empty_like(token_state_slots)
    completed = torch.empty(raw_keys.shape[0], dtype=torch.bool, device=device)
    # All allocations and snapshot launches precede any mutation.
    try:
        _state_write[(batch, pages)](
            raw_keys,
            cu_seqlens,
            state_slots,
            state_pool,
            PAGES=pages,
            PAGE=page_size,
            num_warps=4,
        )
        if groups:
            _kv_write[(batch, groups)](
                raw_keys,
                cu_seqlens,
                rope_cos,
                rope_sin,
                gamma,
                kv_slots,
                kv_pool,
                dense_keys,
                GROUPS=groups,
                MAX_LENGTH=maximum,
                EPS=norm_eps,
                EMIT_DENSE=emit_dense_keys,
                num_warps=4,
                enable_fp_fusion=False,
            )
        _metadata[(batch, triton.cdiv(maximum, 128))](
            cu_seqlens,
            state_table,
            kv_table,
            token_state_slots,
            token_kv_slots,
            completed,
            STATE_WIDTH=state_table.shape[1],
            KV_WIDTH=kv_table.shape[1],
            PAGE=page_size,
            num_warps=4,
        )
    except BaseException:
        restore_padded_indexer_cache(undo)
        torch.cuda.current_stream(device).synchronize()
        raise
    return dict(
        state_slots=token_state_slots,
        kv_slots=token_kv_slots,
        completed=completed,
        num_state_writes=sum(lengths),
        num_kv_writes=sum(n // 4 for n in lengths),
        undo=undo,
        **({"block_keys": dense_keys} if emit_dense_keys else {}),
    )


# ---------------------------------------------------------------------------
# Prefix-reuse preparation (P3). Two fused writers replace the staged
# vectorized writer plus the per-token query norm/RoPE/padding chain for a
# page-aligned prefix-reuse target prefill. The per-token kernel owns the
# query gemma norm, the partial RoPE, the padded scorer-query layout and the
# fp32 raw-key seal into the state ring; the per-group kernel owns the
# gather -> fp32 mean -> k-norm -> block-start RoPE -> bf16 kv write. Both
# consume the frozen Qwen4ExpPrefillStepPlan tables (P2), so the launches
# never rebuild metadata. Snapshots of BOTH pools still finish before the
# first mutation: the same snapshot -> write transaction the zero-prefix
# writer uses, restorable through restore_padded_indexer_cache.
# ---------------------------------------------------------------------------

_PREFIX_REUSE_FUSED_STATS = {"attempts": 0, "hits": 0}


def prefix_reuse_fused_stats() -> dict:
    """Fused writer counters for benches and diagnostics (attempts/hits)."""
    return dict(_PREFIX_REUSE_FUSED_STATS)


@triton.jit
def _prefix_reuse_state_snapshot(
    cu,
    prefixes,
    table,
    pool,
    slots,
    original,
    WIDTH,
    PAGES,
):
    request, page = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    prefix = tl.load(prefixes + request)
    r, d = tl.arange(0, STATE_RING_ROWS), tl.arange(0, INDEXER_HEAD_DIM)
    end = tl.minimum((page + 1) * 128, length)
    position = end - 1 - (end - 1 - r + 8) % 8
    valid = position >= page * 128
    # Prefixes are page aligned, so a local page index addresses the same
    # ring rows as the absolute one; only the table column shifts.
    column = prefix // 128 + page
    block = tl.load(
        table + request * WIDTH + column,
        mask=(page * 128 < length) & (column < WIDTH),
        other=0,
    )
    slot = block * 8 + r
    row = (request * PAGES + page) * 8 + r
    value = tl.load(
        pool + slot[:, None].to(tl.int64) * 128 + d[None, :],
        mask=valid[:, None],
        other=0,
    )
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row[:, None] * 128 + d[None, :], value)


@triton.jit
def _prefix_reuse_kv_snapshot(
    cu,
    prefixes,
    table,
    pool,
    slots,
    original,
    WIDTH,
    GROUPS,
):
    request, group = tl.program_id(0), tl.program_id(1)
    length = tl.load(cu + request + 1) - tl.load(cu + request)
    prefix = tl.load(prefixes + request)
    valid = group < length // 4
    start = prefix + group * 4
    column = start // 128
    block = tl.load(
        table + request * WIDTH + column, mask=valid & (column < WIDTH), other=0
    )
    slot = block * 32 + (start // 4) % 32
    d = tl.arange(0, INDEXER_HEAD_DIM)
    row = request * GROUPS + group
    value = tl.load(pool + slot.to(tl.int64) * 128 + d, mask=valid, other=0)
    tl.store(slots + row, tl.where(valid, slot, -1))
    tl.store(original + row * 128 + d, value)


@triton.jit
def _prefix_reuse_q_prep(
    q,
    raw,
    cu,
    prefixes,
    cos,
    sin,
    gamma,
    state_table,
    state_pool,
    padded,
    STATE_WIDTH,
    STATE_BLOCKS,
    MAX_ROWS,
    HEADS,
    Q_STRIDE,
    RAW_STRIDE,
    EPS: tl.constexpr,
):
    request, row, head = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start = tl.load(cu + request)
    length = tl.load(cu + request + 1) - start
    prefix = tl.load(prefixes + request)
    # Selection padding repeats the last valid row of short requests, the
    # same row map the staged writer's padded scorer query consumes.
    token = start + tl.minimum(row, length - 1)
    token64 = token.to(tl.int64)
    position = prefix + tl.minimum(row, length - 1)
    if head == 0 and row < length:
        # Seal the raw fp32 key into its state ring slot. Tokens mapping to
        # the same (page, ring offset) arrive as later rows, so only the last
        # one writes: the staged writer's sequential last-write-wins order
        # without its argsort.
        page = position // 128
        last = position + 8 >= tl.minimum((page + 1) * 128, prefix + length)
        state_block = tl.load(
            state_table + request * STATE_WIDTH + page,
            mask=last & (page < STATE_WIDTH),
            other=0,
        )
        slot = state_block.to(tl.int64) * 8 + position % 8
        lane0 = tl.arange(0, 32)
        column0 = lane0 * 4
        for j in tl.static_range(4):
            key_value = tl.load(raw + token64 * RAW_STRIDE + column0 + j).to(tl.float32)
            tl.store(
                state_pool + slot * 128 + column0 + j,
                key_value,
                mask=(state_block > 0) & (state_block < STATE_BLOCKS),
            )
    # Query gemma norm + partial RoPE in the bit-exact _norm_rope order:
    # the eager 32-lane x 4-adjacent variance and the eager BF16 product
    # rounding boundaries.
    lane = tl.arange(0, 32)
    column = lane * 4
    # Cross-column qkv views: token rows stride the projection width, the
    # 128-wide head segment itself stays contiguous.
    offset = token64 * Q_STRIDE + head * 128
    a = tl.load(q + offset + column).to(tl.float32)
    b = tl.load(q + offset + column + 1).to(tl.float32)
    c = tl.load(q + offset + column + 2).to(tl.float32)
    d = tl.load(q + offset + column + 3).to(tl.float32)
    inv = tl.rsqrt(tl.sum(((a * a + b * b) + c * c) + d * d, 0) / 128.0 + EPS)
    out_base = ((request * MAX_ROWS + row).to(tl.int64) * HEADS + head) * 128
    for j in tl.static_range(4):
        i = column + j
        gain = 1.0 + tl.load(gamma + i).to(tl.float32)
        x = tl.load(q + offset + i).to(tl.float32)
        normalized = ((x * inv) * gain).to(tl.bfloat16).to(tl.float32)
        partner = tl.where(i < 32, i + 32, i - 32)
        partner = tl.where(i < 64, partner, i)
        other = tl.load(q + offset + partner).to(tl.float32)
        other_gain = 1.0 + tl.load(gamma + partner).to(tl.float32)
        rotated = ((other * inv) * other_gain).to(tl.bfloat16).to(tl.float32)
        cosine = tl.load(cos + token64 * 64 + i, i < 64, other=1.0).to(tl.float32)
        sine = tl.load(sin + token64 * 64 + i, i < 64, other=0.0).to(tl.float32)
        left = (normalized * cosine).to(tl.bfloat16).to(tl.float32)
        right = (
            (rotated * tl.where(i < 32, -1.0, 1.0) * sine)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        value = tl.where(
            i < 64, (left + right).to(tl.bfloat16), normalized.to(tl.bfloat16)
        )
        tl.store(padded + out_base + i, value)


@triton.jit
def _prefix_reuse_k_compress(
    raw,
    cu,
    prefixes,
    cos,
    sin,
    gamma,
    kv_table,
    kv_pool,
    KV_WIDTH,
    KV_BLOCKS,
    RAW_STRIDE,
    EPS: tl.constexpr,
):
    request, group = tl.program_id(0), tl.program_id(1)
    start = tl.load(cu + request)
    length = tl.load(cu + request + 1) - start
    prefix = tl.load(prefixes + request)
    valid = group < length // 4
    if valid:
        d = tl.arange(0, INDEXER_HEAD_DIM)
        block = start + group * 4
        base = block.to(tl.int64) * RAW_STRIDE
        # fp32 group mean in the staged writer's member accumulation order,
        # then the staged writer's BF16 round trip before the k-norm.
        v0 = tl.load(raw + base + d).to(tl.float32)
        v1 = tl.load(raw + base + RAW_STRIDE + d).to(tl.float32)
        v2 = tl.load(raw + base + 2 * RAW_STRIDE + d).to(tl.float32)
        v3 = tl.load(raw + base + 3 * RAW_STRIDE + d).to(tl.float32)
        pooled = (((v0 + v1) + v2) + v3) * 0.25
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
        # k-norm variance in the eager mean's 32-lane x 4-adjacent order.
        lane = tl.arange(0, 32)
        a = tl.gather(pooled, lane * 4, 0)
        b = tl.gather(pooled, lane * 4 + 1, 0)
        c = tl.gather(pooled, lane * 4 + 2, 0)
        e = tl.gather(pooled, lane * 4 + 3, 0)
        variance = tl.sum(((a * a + b * b) + c * c) + e * e, 0) / 128.0
        rms = tl.rsqrt(variance + EPS)
        gain = 1.0 + tl.load(gamma + d).to(tl.float32)
        normalized = (pooled * rms * gain).to(tl.bfloat16)
        partner = tl.where(d < 64, tl.where(d < 32, d + 32, d - 32), d)
        rotated = tl.gather(normalized, partner, 0)
        rope_row = (block + 3).to(tl.int64) * 64
        cosine = tl.load(cos + rope_row + d, mask=d < 64, other=1).to(tl.float32)
        sine = tl.load(sin + rope_row + d, mask=d < 64, other=0).to(tl.float32)
        left = (normalized.to(tl.float32) * cosine).to(tl.bfloat16)
        right = (rotated.to(tl.float32) * sine * tl.where(d < 32, -1.0, 1.0)).to(
            tl.bfloat16
        )
        value = tl.where(
            d < 64,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )
        # Absolute block-start addressing. One program owns each (request,
        # group) slot, so the store is already last-wins without an argsort.
        block_start = prefix + group * 4
        column = block_start // 128
        kv_block = tl.load(
            kv_table + request * KV_WIDTH + column,
            mask=column < KV_WIDTH,
            other=0,
        )
        slot = kv_block * 32 + (block_start // 4) % 32
        tl.store(
            kv_pool + slot.to(tl.int64) * 128 + d,
            value,
            mask=(kv_block > 0) & (kv_block < KV_BLOCKS),
        )


def write_prefix_reuse_prefill(
    q,
    raw_keys,
    cu_seqlens,
    prefixes,
    lengths,
    prefix_values,
    current_cos,
    current_sin,
    rope_cos,
    rope_sin,
    q_norm_gamma,
    k_norm_gamma,
    kv_pool,
    kv_table,
    state_pool,
    state_table,
    *,
    kv_tokens_per_block,
    state_tokens_per_block,
    norm_eps,
) -> dict | None:
    """Fused prefix-reuse preparation; ``None`` for unsupported geometry.

    ``q`` is the RAW (un-normed) packed projection ``[tokens, heads, 128]``
    and ``raw_keys`` its key head ``[tokens, 128]``, both bf16. Both may be
    zero-copy cross-column views of the packed qkv projection (the
    ``split_projected`` layout): ``q`` with strides ``(S, 128, 1)`` where
    ``S >= heads * 128``, and ``raw_keys`` with strides ``(S2, 1)`` where
    ``S2 >= 128``; any other layout declines. The caller owns a frozen
    ``Qwen4ExpPrefillStepPlan`` hit and guarantees the geometry this writer
    specializes on: ratio 4, 128-token KV/state pages with equal
    page-aligned prefixes and 128-wide heads, so every completed group's
    four source rows lie inside this invocation's packed rows. ``current_cos`` /
    ``current_sin`` carry one per-token rotation row and ``rope_cos`` /
    ``rope_sin`` one block-start row per packed token (the token-aligned
    tables the staged writer consumes).

    The two kernels reproduce the staged vectorized writer's numerics bit for
    bit: the fp32 group mean keeps the member accumulation order, both norms
    keep the eager mean's 32-lane x 4-adjacent reduction, and the partial
    RoPE keeps the eager BF16 product rounding boundaries. Snapshots of BOTH
    pools finish before the first mutation; the returned ``undo`` restores
    them through :func:`restore_padded_indexer_cache`, and ``padded_q`` is
    the [batch, max_rows, heads, 128] scorer query the selection consumes.

    The caller's required-blocks verdicts (enqueued before this call) cover
    every table column both kernels touch; aliased physical blocks remain
    undetected here exactly as in the staged vectorized writer, with the
    same consequence of ill-defined slot contents under an allocator bug.
    """
    _PREFIX_REUSE_FUSED_STATS["attempts"] += 1
    batch = len(lengths)
    maximum = max(lengths, default=0)
    tokens = sum(lengths)
    head_num = int(q.shape[1]) if q.dim() == 3 else -1
    device = raw_keys.device
    if not (
        batch > 0
        and kv_tokens_per_block == 128
        and state_tokens_per_block == 128
        and 0 < maximum <= 65535
        and 0 < head_num <= 65535
        and all(isinstance(n, int) and n > 0 for n in lengths)
        and all(value % 128 == 0 for value in prefix_values)
        and q.is_cuda
        and torch.version.hip is None
        and q.dtype == torch.bfloat16
        and raw_keys.dtype == torch.bfloat16
        and tuple(q.shape) == (tokens, head_num, 128)
        and tuple(raw_keys.shape) == (tokens, 128)
        and q.stride(2) == 1
        and q.stride(1) == 128
        and q.stride(0) >= head_num * 128
        and raw_keys.stride(1) == 1
        and raw_keys.stride(0) >= 128
        and cu_seqlens.dtype == torch.int32
        and tuple(cu_seqlens.shape) == (batch + 1,)
        and cu_seqlens.is_contiguous()
        and prefixes.dtype == torch.int32
        and tuple(prefixes.shape) == (batch,)
        and prefixes.is_contiguous()
        and current_cos.dtype == current_sin.dtype == torch.bfloat16
        and rope_cos.dtype == rope_sin.dtype == torch.bfloat16
        and tuple(current_cos.shape) == tuple(current_sin.shape) == (tokens, 64)
        and tuple(rope_cos.shape) == tuple(rope_sin.shape) == (tokens, 64)
        and current_cos.is_contiguous()
        and current_sin.is_contiguous()
        and rope_cos.is_contiguous()
        and rope_sin.is_contiguous()
        and q_norm_gamma.shape == k_norm_gamma.shape == (128,)
        and q_norm_gamma.dtype in (torch.bfloat16, torch.float32)
        and k_norm_gamma.dtype in (torch.bfloat16, torch.float32)
        and q_norm_gamma.is_contiguous()
        and k_norm_gamma.is_contiguous()
        and kv_pool.dim() == 3
        and tuple(kv_pool.shape[1:]) == (32, 128)
        and kv_pool.dtype == torch.bfloat16
        and kv_pool.is_contiguous()
        and state_pool.dim() == 3
        and tuple(state_pool.shape[1:]) == (8, 128)
        and state_pool.dtype == torch.float32
        and state_pool.is_contiguous()
        and kv_table.dim() == 2
        and state_table.dim() == 2
        and int(kv_table.shape[0]) == batch
        and int(state_table.shape[0]) == batch
        and kv_table.dtype == state_table.dtype == torch.int32
        and kv_table.is_contiguous()
        and state_table.is_contiguous()
        and all(
            tensor.device == device
            for tensor in (
                q,
                raw_keys,
                cu_seqlens,
                prefixes,
                current_cos,
                current_sin,
                rope_cos,
                rope_sin,
                q_norm_gamma,
                k_norm_gamma,
                kv_pool,
                kv_table,
                state_pool,
                state_table,
            )
        )
    ):
        return None
    _PREFIX_REUSE_FUSED_STATS["hits"] += 1
    pages = triton.cdiv(maximum, 128)
    groups = maximum // 4
    padded_q = torch.empty(
        (batch, maximum, head_num, 128), dtype=torch.bfloat16, device=device
    )
    state_slots = torch.empty(batch * pages * 8, dtype=torch.int64, device=device)
    kv_slots = torch.empty(batch * groups, dtype=torch.int64, device=device)
    original_state = torch.empty(
        (state_slots.numel(), 128), dtype=torch.float32, device=device
    )
    original_kv = torch.empty(
        (kv_slots.numel(), 128), dtype=torch.bfloat16, device=device
    )
    undo = IndexerCacheUndo(
        state_pool,
        state_slots,
        original_state,
        kv_pool,
        kv_slots,
        original_kv,
        padded_slots=True,
    )
    _prefix_reuse_state_snapshot[(batch, pages)](
        cu_seqlens,
        prefixes,
        state_table,
        state_pool,
        state_slots,
        original_state,
        WIDTH=state_table.shape[1],
        PAGES=pages,
        num_warps=4,
    )
    if groups:
        _prefix_reuse_kv_snapshot[(batch, groups)](
            cu_seqlens,
            prefixes,
            kv_table,
            kv_pool,
            kv_slots,
            original_kv,
            WIDTH=kv_table.shape[1],
            GROUPS=groups,
            num_warps=4,
        )
    # All allocations and snapshot launches precede any mutation.
    try:
        _prefix_reuse_q_prep[(batch, maximum, head_num)](
            q,
            raw_keys,
            cu_seqlens,
            prefixes,
            current_cos,
            current_sin,
            q_norm_gamma,
            state_table,
            state_pool,
            padded_q,
            STATE_WIDTH=state_table.shape[1],
            STATE_BLOCKS=state_pool.shape[0],
            MAX_ROWS=maximum,
            HEADS=head_num,
            Q_STRIDE=q.stride(0),
            RAW_STRIDE=raw_keys.stride(0),
            EPS=norm_eps,
            num_warps=1,
            enable_fp_fusion=False,
        )
        if groups:
            _prefix_reuse_k_compress[(batch, groups)](
                raw_keys,
                cu_seqlens,
                prefixes,
                rope_cos,
                rope_sin,
                k_norm_gamma,
                kv_table,
                kv_pool,
                KV_WIDTH=kv_table.shape[1],
                KV_BLOCKS=kv_pool.shape[0],
                RAW_STRIDE=raw_keys.stride(0),
                EPS=norm_eps,
                num_warps=4,
                enable_fp_fusion=False,
            )
    except BaseException:
        restore_padded_indexer_cache(undo)
        torch.cuda.current_stream(device).synchronize()
        raise
    return {"undo": undo, "padded_q": padded_q}
