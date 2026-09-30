"""Single-token QSA side-cache writer for the released BF16 indexer layout.

One program owns one request's state-ring slot and optional completed-group
entry. The completion branch depends on a device sequence length, so the
launch geometry is stable across CUDA Graph replays.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_exp.indexer_compressor import IndexerCacheUndo


@triton.jit
def _write_decode_key(
    raw_ptr,
    positions_ptr,
    valid_ptr,
    cos_ptr,
    sin_ptr,
    gamma_ptr,
    kv_pool_ptr,
    kv_table_ptr,
    state_pool_ptr,
    state_table_ptr,
    KV_TABLE_WIDTH: tl.constexpr,
    STATE_TABLE_WIDTH: tl.constexpr,
    KV_POOL_BLOCKS: tl.constexpr,
    STATE_POOL_BLOCKS: tl.constexpr,
    KV_TOKENS_PER_BLOCK: tl.constexpr,
    STATE_TOKENS_PER_BLOCK: tl.constexpr,
    KV_ENTRIES_PER_BLOCK: tl.constexpr,
    D: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    request = tl.program_id(0)
    d = tl.arange(0, D)
    active = tl.load(valid_ptr + request) != 0 if HAS_VALID else True
    position = tl.load(positions_ptr + request)
    state_column = position // STATE_TOKENS_PER_BLOCK
    state_id = tl.load(
        state_table_ptr + request * STATE_TABLE_WIDTH + state_column,
        mask=active & (position >= 0) & (state_column < STATE_TABLE_WIDTH),
        other=0,
    )
    state_valid = (state_id > 0) & (state_id < STATE_POOL_BLOCKS)
    raw = tl.load(raw_ptr + request * D + d, mask=active, other=0).to(tl.float32)
    tl.store(
        state_pool_ptr + (state_id * 8 + position % 8) * D + d,
        raw,
        mask=active & state_valid,
    )

    if active and (position + 1) % 4 == 0:
        group = tl.arange(0, 4)
        source_positions = position - 3 + group
        source_columns = source_positions // STATE_TOKENS_PER_BLOCK
        source_ids = tl.load(
            state_table_ptr + request * STATE_TABLE_WIDTH + source_columns,
            mask=(source_positions >= 0)
            & (source_columns < STATE_TABLE_WIDTH),
            other=0,
        )
        source_valid = (source_ids > 0) & (source_ids < STATE_POOL_BLOCKS)
        source_slots = source_ids * 8 + source_positions % 8
        history = tl.load(
            state_pool_ptr + source_slots[:, None] * D + d[None, :],
            mask=source_valid[:, None],
            other=0.0,
        )
        # The current key has just been written by this program. Read the
        # input directly so no global-memory fence is needed for that row.
        history = tl.where(group[:, None] == 3, raw[None, :], history)
        pooled = (tl.sum(history, axis=0) * 0.25).to(tl.bfloat16).to(tl.float32)
        gain = 1.0 + tl.load(gamma_ptr + d).to(tl.float32)
        rms = tl.rsqrt(tl.sum(pooled * pooled, axis=0) / D + EPS)
        normalized = (pooled * rms * gain).to(tl.bfloat16)

        # Upstream applies RoPE to BF16 values: each multiplication and the
        # final addition rounds to BF16. Preserve that boundary here.
        partner = tl.where(
            d < ROTARY,
            tl.where(d < ROTARY // 2, d + ROTARY // 2, d - ROTARY // 2),
            d,
        )
        rotated_half = tl.gather(normalized, partner, axis=0)
        cosine = tl.load(cos_ptr + request * ROTARY + d, mask=d < ROTARY, other=1)
        sine = tl.load(sin_ptr + request * ROTARY + d, mask=d < ROTARY, other=0)
        left = (normalized.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
        right = (
            rotated_half.to(tl.float32)
            * sine.to(tl.float32)
            * tl.where(d < ROTARY // 2, -1.0, 1.0)
        ).to(tl.bfloat16)
        value = tl.where(
            d < ROTARY,
            (left.to(tl.float32) + right.to(tl.float32)).to(tl.bfloat16),
            normalized,
        )

        kv_column = position // KV_TOKENS_PER_BLOCK
        kv_id = tl.load(
            kv_table_ptr + request * KV_TABLE_WIDTH + kv_column,
            mask=(position >= 0) & (kv_column < KV_TABLE_WIDTH),
            other=0,
        )
        kv_valid = (kv_id > 0) & (kv_id < KV_POOL_BLOCKS)
        entry = (position // 4) % KV_ENTRIES_PER_BLOCK
        tl.store(
            kv_pool_ptr + (kv_id * KV_ENTRIES_PER_BLOCK + entry) * D + d,
            value,
            mask=kv_valid,
        )


def is_supported(
    raw_keys: torch.Tensor,
    positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    ratio: int,
) -> bool:
    if not raw_keys.is_cuda or torch.version.hip is not None:
        return False
    batch = int(raw_keys.shape[0]) if raw_keys.dim() == 2 else -1
    device = raw_keys.device
    tensors = (positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
               state_pool, state_table)
    return (
        batch > 0
        and raw_keys.shape[1] == 128
        and raw_keys.dtype == torch.bfloat16
        and raw_keys.is_contiguous()
        and all(t.device == device and t.is_contiguous() for t in tensors)
        and positions.shape == (batch,)
        and positions.dtype == torch.int32
        and rope_cos.shape == rope_sin.shape == (batch, 64)
        and rope_cos.dtype == rope_sin.dtype == torch.bfloat16
        and gamma.shape == (128,)
        and gamma.dtype == torch.bfloat16
        and ratio == 4
        and kv_tokens_per_block == state_tokens_per_block == 128
        and kv_pool.dim() == 3
        and kv_pool.shape[1:] == (32, 128)
        and kv_pool.dtype == torch.bfloat16
        and state_pool.dim() == 3
        and state_pool.shape[1:] == (8, 128)
        and state_pool.dtype == torch.float32
        and kv_table.dim() == state_table.dim() == 2
        and kv_table.shape[0] == state_table.shape[0] == batch
        and kv_table.dtype == state_table.dtype == torch.int32
    )


def write_decode_key_(
    raw_keys: torch.Tensor,
    positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    norm_eps: float,
    valid: torch.Tensor | None = None,
) -> None:
    """Write one raw key/request and any completed compressed group."""
    if not is_supported(
        raw_keys, positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
        state_pool, state_table, kv_tokens_per_block=128,
        state_tokens_per_block=128, ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA decode writer layout")
    if valid is not None and (
        valid.shape != positions.shape
        or valid.dtype != torch.bool
        or valid.device != raw_keys.device
        or not valid.is_contiguous()
    ):
        raise ValueError("Qwen4 QSA decode writer has invalid row mask")
    _write_decode_key[(raw_keys.shape[0],)](
        raw_keys, positions, raw_keys if valid is None else valid,
        rope_cos, rope_sin, gamma, kv_pool, kv_table,
        state_pool, state_table,
        KV_TABLE_WIDTH=kv_table.shape[1],
        STATE_TABLE_WIDTH=state_table.shape[1],
        KV_POOL_BLOCKS=kv_pool.shape[0],
        STATE_POOL_BLOCKS=state_pool.shape[0],
        KV_TOKENS_PER_BLOCK=128,
        STATE_TOKENS_PER_BLOCK=128,
        KV_ENTRIES_PER_BLOCK=32,
        D=128,
        ROTARY=64,
        EPS=norm_eps,
        HAS_VALID=valid is not None,
        num_warps=4,
    )


def write_decode_key_with_undo_(
    raw_keys: torch.Tensor,
    positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    norm_eps: float,
) -> IndexerCacheUndo:
    """Capture fixed-size undo rows, then write the device-dependent slots.

    The inactive compressed slot uses reserved physical block zero. All
    requests capture the same original sentinel row, so rollback stays safe
    even when no request completes a compression group.
    """
    if not is_supported(
        raw_keys, positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
        state_pool, state_table, kv_tokens_per_block=128,
        state_tokens_per_block=128, ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA decode writer layout")
    pos = positions.to(torch.long)
    batch_rows = torch.arange(raw_keys.shape[0], device=raw_keys.device)
    state_columns = (pos // 128).clamp(0, state_table.shape[1] - 1)
    kv_columns = (pos // 128).clamp(0, kv_table.shape[1] - 1)
    state_ids = state_table[batch_rows, state_columns].to(torch.long)
    kv_ids = kv_table[batch_rows, kv_columns].to(torch.long)
    state_slots = (state_ids * 8 + pos.remainder(8)).clamp(
        0, state_pool.shape[0] * 8 - 1
    )
    completed = (pos + 1).remainder(4) == 0
    kv_slots = torch.where(
        completed, kv_ids * 32 + (pos // 4).remainder(32), 0
    ).clamp(0, kv_pool.shape[0] * 32 - 1)
    original_state = state_pool.flatten(0, 1).index_select(0, state_slots).clone()
    original_kv = kv_pool.flatten(0, 1).index_select(0, kv_slots).clone()
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=state_slots,
        original_state=original_state,
        kv_pool=kv_pool,
        kv_slots=kv_slots,
        original_kv=original_kv,
    )
    write_decode_key_(
        raw_keys, positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
        state_pool, state_table, norm_eps=norm_eps,
    )
    return undo


def write_target_window_with_undo_(
    raw_keys: torch.Tensor,
    prefixes: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    norm_eps: float,
) -> IndexerCacheUndo:
    """Write a uniform one-to-four-row target window with one undo snapshot.

    Ordered launches reuse the graph-safe single-row kernel. At most one
    row completes a compression group, regardless of the prefix residue.
    Capturing all destinations before the first launch preserves the original
    side state if a later main-cache operation fails.
    """
    batch = int(prefixes.numel())
    if (
        raw_keys.dim() != 2
        or batch == 0
        or raw_keys.shape[0] % batch
        or not 1 <= raw_keys.shape[0] // batch <= 4
    ):
        raise ValueError(
            "QSA target writer requires one to four packed rows per request"
        )
    query_len = int(raw_keys.shape[0]) // batch
    if (
        prefixes.dtype != torch.int32
        or prefixes.device != raw_keys.device
        or not prefixes.is_contiguous()
    ):
        raise ValueError("QSA target writer has invalid position or RoPE geometry")
    if rope_cos.shape != (batch * query_len, 64) or rope_sin.shape != rope_cos.shape:
        raise ValueError("QSA target writer requires one RoPE row per packed token")
    if not is_supported(
        raw_keys[:batch].contiguous(),
        prefixes,
        rope_cos[:batch].contiguous(),
        rope_sin[:batch].contiguous(),
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        kv_tokens_per_block=128,
        state_tokens_per_block=128,
        ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA target writer layout")

    offsets = torch.arange(query_len, dtype=torch.int32, device=prefixes.device)
    positions = prefixes[:, None] + offsets[None, :]
    state_columns = (positions // 128).clamp(0, state_table.shape[1] - 1)
    state_ids = state_table.gather(1, state_columns).to(torch.long)
    state_slots = (
        (state_ids * 8 + positions.remainder(8))
        .reshape(-1)
        .clamp(0, state_pool.shape[0] * 8 - 1)
    )
    completion = prefixes + 3 - prefixes.remainder(4)
    kv_columns = (completion // 128).clamp(0, kv_table.shape[1] - 1)
    kv_ids = kv_table.gather(1, kv_columns.unsqueeze(1)).reshape(-1).to(torch.long)
    kv_slots = torch.where(
        completion < prefixes + query_len,
        kv_ids * 32 + (completion // 4).remainder(32),
        0,
    ).clamp(0, kv_pool.shape[0] * 32 - 1)
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=state_slots,
        original_state=state_pool.flatten(0, 1).index_select(0, state_slots).clone(),
        kv_pool=kv_pool,
        kv_slots=kv_slots,
        original_kv=kv_pool.flatten(0, 1).index_select(0, kv_slots).clone(),
    )
    keys = raw_keys.reshape(batch, query_len, 128).transpose(0, 1).contiguous()
    cos = rope_cos.reshape(batch, query_len, 64).transpose(0, 1).contiguous()
    sin = rope_sin.reshape(batch, query_len, 64).transpose(0, 1).contiguous()
    step_positions = positions.transpose(0, 1).contiguous()
    for step in range(query_len):
        write_decode_key_(
            keys[step],
            step_positions[step],
            cos[step],
            sin[step],
            gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            norm_eps=norm_eps,
        )
    return undo


def write_draft_window_with_undo_(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefixes: torch.Tensor,
    lengths: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_table: torch.Tensor,
    *,
    norm_eps: float,
) -> IndexerCacheUndo:
    """Write one to four packed draft rows/request with fixed launch geometry.

    Device lengths mask the inactive rows. The packed source and every cache
    destination are gathered by device metadata, so a CUDA Graph can replay
    different draft lengths and prefix/page assignments at the same capacity.
    Callers still validate their metadata and physical pages before mutation.
    """
    batch = int(prefixes.numel())
    if (
        batch <= 0
        or raw_keys.dim() != 2
        or raw_keys.shape[1] != 128
        or raw_keys.shape[0] < batch
        or raw_keys.shape[0] > batch * 4
        or cu_seqlens.shape != (batch + 1,)
        or lengths.shape != (batch,)
        or prefixes.shape != (batch,)
        or any(t.dtype != torch.int32 for t in (cu_seqlens, prefixes, lengths))
        or any(t.device != raw_keys.device for t in (cu_seqlens, prefixes, lengths))
        or any(not t.is_contiguous() for t in (cu_seqlens, prefixes, lengths))
        or rope_cos.shape != (raw_keys.shape[0], 64)
        or rope_sin.shape != rope_cos.shape
    ):
        raise ValueError("QSA draft writer requires packed one-to-four-row geometry")
    if not is_supported(
        raw_keys[:batch].contiguous(),
        prefixes,
        rope_cos[:batch].contiguous(),
        rope_sin[:batch].contiguous(),
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        kv_tokens_per_block=128,
        state_tokens_per_block=128,
        ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA draft writer layout")

    offsets = torch.arange(4, dtype=torch.int32, device=raw_keys.device)
    active = offsets[None, :] < lengths[:, None]
    source = cu_seqlens[:-1, None] + offsets[None, :]
    source = torch.where(active, source, 0).reshape(-1).to(torch.long)
    positions = prefixes[:, None] + offsets[None, :]
    state_columns = (positions // 128).clamp(0, state_table.shape[1] - 1)
    state_ids = state_table.gather(1, state_columns).to(torch.long)
    state_slots = torch.where(
        active,
        state_ids * 8 + positions.remainder(8),
        0,
    ).reshape(-1).clamp(0, state_pool.shape[0] * 8 - 1)
    completion_offset = 3 - prefixes.remainder(4)
    completion = prefixes + completion_offset
    kv_columns = (completion // 128).clamp(0, kv_table.shape[1] - 1)
    kv_ids = kv_table.gather(1, kv_columns.unsqueeze(1)).reshape(-1).to(torch.long)
    kv_slots = torch.where(
        lengths > completion_offset,
        kv_ids * 32 + (completion // 4).remainder(32),
        0,
    ).clamp(0, kv_pool.shape[0] * 32 - 1)
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=state_slots,
        original_state=state_pool.flatten(0, 1).index_select(0, state_slots).clone(),
        kv_pool=kv_pool,
        kv_slots=kv_slots,
        original_kv=kv_pool.flatten(0, 1).index_select(0, kv_slots).clone(),
    )

    keys = raw_keys.index_select(0, source).view(batch, 4, 128).transpose(0, 1).contiguous()
    cos = rope_cos.index_select(0, source).view(batch, 4, 64).transpose(0, 1).contiguous()
    sin = rope_sin.index_select(0, source).view(batch, 4, 64).transpose(0, 1).contiguous()
    step_positions = positions.transpose(0, 1).contiguous()
    step_active = active.transpose(0, 1).contiguous()
    for step in range(4):
        write_decode_key_(
            keys[step],
            step_positions[step],
            cos[step],
            sin[step],
            gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            norm_eps=norm_eps,
            valid=step_active[step],
        )
    return undo
