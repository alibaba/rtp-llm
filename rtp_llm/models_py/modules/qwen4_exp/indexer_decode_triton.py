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
):
    request = tl.program_id(0)
    d = tl.arange(0, D)
    position = tl.load(positions_ptr + request)
    state_column = position // STATE_TOKENS_PER_BLOCK
    state_id = tl.load(
        state_table_ptr + request * STATE_TABLE_WIDTH + state_column,
        mask=(position >= 0) & (state_column < STATE_TABLE_WIDTH),
        other=0,
    )
    state_valid = (state_id > 0) & (state_id < STATE_POOL_BLOCKS)
    raw = tl.load(raw_ptr + request * D + d).to(tl.float32)
    tl.store(
        state_pool_ptr + (state_id * 8 + position % 8) * D + d,
        raw,
        mask=state_valid,
    )

    if (position + 1) % 4 == 0:
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
) -> None:
    """Write one raw key/request and any completed compressed group."""
    if not is_supported(
        raw_keys, positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
        state_pool, state_table, kv_tokens_per_block=128,
        state_tokens_per_block=128, ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA decode writer layout")
    _write_decode_key[(raw_keys.shape[0],)](
        raw_keys, positions, rope_cos, rope_sin, gamma, kv_pool, kv_table,
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
    """Write the fixed four-row MTP target window with one rollback snapshot.

    Four ordered launches reuse the graph-safe single-row kernel. Exactly one
    row completes a compression group, regardless of the prefix residue.
    Capturing all destinations before the first launch preserves the original
    side state if a later main-cache operation fails.
    """
    if raw_keys.dim() != 2 or raw_keys.shape[0] != prefixes.numel() * 4:
        raise ValueError("QSA target writer requires four packed rows per request")
    batch = int(prefixes.numel())
    if (
        prefixes.dtype != torch.int32
        or prefixes.device != raw_keys.device
        or not prefixes.is_contiguous()
    ):
        raise ValueError("QSA target writer has invalid position or RoPE geometry")
    if rope_cos.shape != (batch * 4, 64) or rope_sin.shape != rope_cos.shape:
        raise ValueError("QSA target writer requires four RoPE rows per request")
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

    offsets = torch.arange(4, dtype=torch.int32, device=prefixes.device)
    positions = prefixes[:, None] + offsets[None, :]
    state_columns = (positions // 128).clamp(0, state_table.shape[1] - 1)
    state_ids = state_table.gather(1, state_columns).to(torch.long)
    state_slots = (state_ids * 8 + positions.remainder(8)).reshape(-1).clamp(
        0, state_pool.shape[0] * 8 - 1
    )
    completion = prefixes + 3 - prefixes.remainder(4)
    kv_columns = (completion // 128).clamp(0, kv_table.shape[1] - 1)
    kv_ids = kv_table.gather(1, kv_columns.unsqueeze(1)).reshape(-1).to(torch.long)
    kv_slots = (kv_ids * 32 + (completion // 4).remainder(32)).clamp(
        0, kv_pool.shape[0] * 32 - 1
    )
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=state_slots,
        original_state=state_pool.flatten(0, 1).index_select(0, state_slots).clone(),
        kv_pool=kv_pool,
        kv_slots=kv_slots,
        original_kv=kv_pool.flatten(0, 1).index_select(0, kv_slots).clone(),
    )
    keys = raw_keys.reshape(batch, 4, 128).transpose(0, 1).contiguous()
    cos = rope_cos.reshape(batch, 4, 64).transpose(0, 1).contiguous()
    sin = rope_sin.reshape(batch, 4, 64).transpose(0, 1).contiguous()
    step_positions = positions.transpose(0, 1).contiguous()
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
        )
    return undo
