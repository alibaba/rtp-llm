"""Single-token QSA side-cache writer for the released BF16 indexer layout.

One program owns one request's state-ring slot and optional completed-group
entry. The completion branch depends on a device sequence length, so the
launch geometry is stable across CUDA Graph replays.
"""

import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qsa.indexer_decode import (
    _prepare_fused_decode,
    _write_decode_key,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import IndexerCacheUndo


def try_prepare_fused_decode(
    q,
    raw_keys,
    lengths,
    positions,
    q_gamma,
    k_gamma,
    inv_freq,
    state_pool,
    state_table,
    kv_pool,
    kv_table,
    position_pool,
    position_table,
    *,
    norm_eps,
    rope_config,
    padded,
):
    """Prepare ordinary SM103 decode from the replay's inclusive token lengths."""
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import is_qsa_rope_style

    if (
        q.ndim != 3
        or not q.is_cuda
        or torch.version.hip is not None
        or torch.cuda.get_device_capability(q.device) != (10, 3)
        or not is_qsa_rope_style(rope_config, "Mrope")
        or int(rope_config.dim) != 64
        or int(rope_config.index_factor) != 3
        or not rope_config.mrope_interleaved
        or not getattr(rope_config, "indexer_is_neox_style", True)
        or inv_freq is None
    ):
        return None
    sections = (
        int(rope_config.mrope_dim1),
        int(rope_config.mrope_dim2),
        int(rope_config.mrope_dim3),
    )
    scale = float(getattr(rope_config, "scale", 1.0))
    if (
        min(sections) < 0
        or sum(sections) != 32
        or sections[1] > 11
        or sections[2] > 10
        or not math.isfinite(scale)
        or scale <= 0
    ):
        return None
    batch = q.shape[0]
    packed = (
        lengths,
        positions,
        q_gamma,
        k_gamma,
        inv_freq,
        state_pool,
        state_table,
        kv_pool,
        kv_table,
        position_pool,
        position_table,
    )
    tables = (state_table, kv_table, position_table)
    if not (
        batch > 0
        and all(t.device == q.device and t.is_contiguous() for t in packed)
        and q.shape == (batch, 8, 128)
        and q.stride(1) == 128
        and q.stride(2) == 1
        and raw_keys.shape == (batch, 128)
        and raw_keys.stride(1) == 1
        and raw_keys.device == q.device
        and q.dtype == raw_keys.dtype == torch.bfloat16
        and lengths.shape == (batch,)
        and positions.shape == (batch, 3)
        and positions.dtype == lengths.dtype == torch.int32
        and q_gamma.shape == k_gamma.shape == (128,)
        and all(t.dtype in (torch.bfloat16, torch.float32) for t in (q_gamma, k_gamma))
        and inv_freq.dtype == torch.float32
        and inv_freq.shape == (32,)
        and state_pool.shape[1:] == (8, 128)
        and state_pool.dtype == torch.float32
        and kv_pool.shape[1:] == (32, 128)
        and kv_pool.dtype == torch.bfloat16
        and position_pool.shape[1:] == (8, 3)
        and position_pool.dtype == torch.int32
        and all(
            t.ndim == 2
            and t.shape[0] == batch
            and t.shape[1] > 0
            and t.dtype == torch.int32
            for t in tables
        )
    ):
        return None
    rotated = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    visible = torch.empty_like(lengths)
    _prepare_fused_decode[(batch, 9)](
        q,
        raw_keys,
        lengths,
        positions,
        q_gamma,
        k_gamma,
        inv_freq,
        state_pool,
        state_table,
        kv_pool,
        kv_table,
        position_pool,
        position_table,
        rotated,
        visible,
        Q_STRIDE=q.stride(0),
        RAW_STRIDE=raw_keys.stride(0),
        STATE_WIDTH=state_table.shape[1],
        KV_WIDTH=kv_table.shape[1],
        POSITION_WIDTH=position_table.shape[1],
        STATE_BLOCKS=state_pool.shape[0],
        KV_BLOCKS=kv_pool.shape[0],
        POSITION_BLOCKS=position_pool.shape[0],
        H_SECTION=sections[1],
        W_SECTION=sections[2],
        INV_SCALE=1.0 / scale,
        EPS=norm_eps,
        PADDED=padded,
        num_warps=1,
    )
    return rotated, visible


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
    tensors = (
        positions,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
    )
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
        raw_keys,
        positions,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        kv_tokens_per_block=128,
        state_tokens_per_block=128,
        ratio=4,
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
        raw_keys,
        positions,
        raw_keys if valid is None else valid,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
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
        raw_keys,
        positions,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        kv_tokens_per_block=128,
        state_tokens_per_block=128,
        ratio=4,
    ):
        raise ValueError("unsupported Qwen4 QSA decode writer layout")
    from rtp_llm.models_py.modules.qwen4_exp.qsa.decode_snapshot import (
        try_snapshot_decode,
    )

    undo = try_snapshot_decode(positions, kv_pool, kv_table, state_pool, state_table)
    if undo is not None:
        write_decode_key_(
            raw_keys,
            positions,
            rope_cos,
            rope_sin,
            gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            norm_eps=norm_eps,
        )
        return undo
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
    kv_slots = torch.where(completed, kv_ids * 32 + (pos // 4).remainder(32), 0).clamp(
        0, kv_pool.shape[0] * 32 - 1
    )
    original_state = state_pool.flatten(0, 1).index_select(0, state_slots).clone()
    original_kv = kv_pool.flatten(0, 1).index_select(0, kv_slots).clone()
    state_live = (
        (pos >= 0)
        & (pos // 128 < state_table.shape[1])
        & (state_ids > 0)
        & (state_ids < state_pool.shape[0])
    )
    kv_live = (
        completed
        & (pos >= 0)
        & (pos // 128 < kv_table.shape[1])
        & (kv_ids > 0)
        & (kv_ids < kv_pool.shape[0])
    )
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=torch.where(state_live, state_slots, -1),
        original_state=original_state,
        kv_pool=kv_pool,
        kv_slots=torch.where(kv_live, kv_slots, -1),
        original_kv=original_kv,
        padded_slots=True,
    )
    write_decode_key_(
        raw_keys,
        positions,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        norm_eps=norm_eps,
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

    from rtp_llm.models_py.modules.qwen4_exp.qsa.target_writer import (
        try_write_target_window,
    )

    fused = try_write_target_window(
        raw_keys,
        prefixes,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        norm_eps=norm_eps,
    )
    if fused is not None:
        return fused

    offsets = torch.arange(query_len, dtype=torch.int32, device=prefixes.device)
    positions = prefixes[:, None] + offsets[None, :]
    state_columns = (positions // 128).clamp(0, state_table.shape[1] - 1)
    state_ids = state_table.gather(1, state_columns.to(torch.long)).to(torch.long)
    state_slots = (
        (state_ids * 8 + positions.remainder(8))
        .reshape(-1)
        .clamp(0, state_pool.shape[0] * 8 - 1)
    )
    completion = prefixes + 3 - prefixes.remainder(4)
    kv_columns = (completion // 128).clamp(0, kv_table.shape[1] - 1)
    kv_ids = (
        kv_table.gather(1, kv_columns.to(torch.long).unsqueeze(1))
        .reshape(-1)
        .to(torch.long)
    )
    kv_slots = torch.where(
        completion < prefixes + query_len,
        kv_ids * 32 + (completion // 4).remainder(32),
        0,
    ).clamp(0, kv_pool.shape[0] * 32 - 1)
    state_live = (
        (positions >= 0)
        & (positions // 128 < state_table.shape[1])
        & (state_ids > 0)
        & (state_ids < state_pool.shape[0])
    )
    kv_live = (
        (completion < prefixes + query_len)
        & (completion >= 0)
        & (completion // 128 < kv_table.shape[1])
        & (kv_ids > 0)
        & (kv_ids < kv_pool.shape[0])
    )
    undo = IndexerCacheUndo(
        state_pool=state_pool,
        state_slots=torch.where(state_live.reshape(-1), state_slots, -1),
        original_state=state_pool.flatten(0, 1).index_select(0, state_slots).clone(),
        kv_pool=kv_pool,
        kv_slots=torch.where(kv_live, kv_slots, -1),
        original_kv=kv_pool.flatten(0, 1).index_select(0, kv_slots).clone(),
        padded_slots=True,
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

    from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_writer import (
        try_write_draft_window,
    )

    fused = try_write_draft_window(
        raw_keys,
        cu_seqlens,
        prefixes,
        lengths,
        rope_cos,
        rope_sin,
        gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        norm_eps=norm_eps,
    )
    if fused is not None:
        return fused

    offsets = torch.arange(4, dtype=torch.int32, device=raw_keys.device)
    active = offsets[None, :] < lengths[:, None]
    source = cu_seqlens[:-1, None] + offsets[None, :]
    source = torch.where(active, source, 0).reshape(-1).to(torch.long)
    positions = prefixes[:, None] + offsets[None, :]
    state_columns = (positions // 128).clamp(0, state_table.shape[1] - 1)
    state_ids = state_table.gather(1, state_columns.to(torch.long)).to(torch.long)
    state_slots = (
        torch.where(
            active,
            state_ids * 8 + positions.remainder(8),
            0,
        )
        .reshape(-1)
        .clamp(0, state_pool.shape[0] * 8 - 1)
    )
    completion_offset = 3 - prefixes.remainder(4)
    completion = prefixes + completion_offset
    kv_columns = (completion // 128).clamp(0, kv_table.shape[1] - 1)
    kv_ids = (
        kv_table.gather(1, kv_columns.to(torch.long).unsqueeze(1))
        .reshape(-1)
        .to(torch.long)
    )
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

    keys = (
        raw_keys.index_select(0, source)
        .view(batch, 4, 128)
        .transpose(0, 1)
        .contiguous()
    )
    cos = (
        rope_cos.index_select(0, source).view(batch, 4, 64).transpose(0, 1).contiguous()
    )
    sin = (
        rope_sin.index_select(0, source).view(batch, 4, 64).transpose(0, 1).contiguous()
    )
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
