"""Transactional, zero-prefix QSA prefill for the released BF16 layout."""

import itertools

import torch
import triton
import triton.language as tl

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
    WIDTH: tl.constexpr,
    PAGES: tl.constexpr,
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
    WIDTH: tl.constexpr,
    GROUPS: tl.constexpr,
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
    PAGES: tl.constexpr,
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
    GROUPS: tl.constexpr,
    MAX_LENGTH: tl.constexpr,
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
    STATE_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
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


def _check_host_mirrors(lengths, cu, tables, device_tables, pool_sizes):
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
            required = triton.cdiv(length // ratio * ratio, 128)
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
        and page_size == 128
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
        and kv_pool.shape[1:] == (32, 128)
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
        verdict = int((invalid.to(torch.int32) + 2 * aliases.to(torch.int32)).item())
        if verdict & 1:
            raise ValueError("QSA prefill has invalid cu_seqlens or physical block ids")
        if verdict & 2:
            return None
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
