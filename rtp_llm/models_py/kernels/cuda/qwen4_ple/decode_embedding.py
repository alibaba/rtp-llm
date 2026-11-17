# SPDX-License-Identifier: Apache-2.0
"""Fused SM10x PLE decode cache reads and pinned n-gram lookup."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.kernels.cuda.qwen4_ple.commit import _graph_page_block


@triton.jit
def _ple_decode_embedding(
    Sequence,
    Ids,
    StateTable,
    CtxTable,
    StatePool,
    CtxPool,
    Vocab,
    Offsets,
    Multipliers,
    Shards,
    Scale,
    StateOut,
    History,
    StateDest,
    CtxDest,
    Embeddings,
    S_COLS: tl.constexpr,
    C_COLS: tl.constexpr,
    S_ROW: tl.constexpr,
    S_COL: tl.constexpr,
    C_ROW: tl.constexpr,
    C_COL: tl.constexpr,
    S_POOL: tl.constexpr,
    C_POOL: tl.constexpr,
    S_STRIDE: tl.constexpr,
    C_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    EOS: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    STATE_TILE: tl.constexpr,
    SHARD_ROWS: tl.constexpr,
    SHARD_COUNT: tl.constexpr,
    EMBED_WIDTH: tl.constexpr,
    FP8: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    prefix = tl.load(Sequence + row).to(tl.int64) - 1
    live = prefix > 0
    read_page = tl.where(live, (prefix - 1) // PAGE_SIZE, -1)
    # Gather tiles read the old context directly. The independent state tiles
    # snapshot the convolution input before any CTA in a later launch writes it.
    if tile < 16:
        cb = _graph_page_block(CtxTable, row, read_page, C_COLS, C_ROW, C_COL, C_POOL)
        previous = tl.load(CtxPool + cb * C_STRIDE + 1, live, other=EOS)
        before_previous = tl.load(CtxPool + cb * C_STRIDE, live, other=EOS)
        current = tl.load(Ids + row).to(tl.int64)
        if tile == 0:
            write_page = tl.where(live, prefix // PAGE_SIZE, -1)
            sw = _graph_page_block(
                StateTable, row, write_page, S_COLS, S_ROW, S_COL, S_POOL
            )
            cw = _graph_page_block(
                CtxTable, row, write_page, C_COLS, C_ROW, C_COL, C_POOL
            )
            tl.store(StateDest + row, sw)
            tl.store(CtxDest + row, cw)
            tl.store(History + row * 3, before_previous)
            tl.store(History + row * 3 + 1, previous)
            tl.store(History + row * 3 + 2, current)
        before_previous = tl.where(previous == EOS, EOS, before_previous)
        two = (current * tl.load(Multipliers)) ^ (previous * tl.load(Multipliers + 1))
        three = two ^ (before_previous * tl.load(Multipliers + 2))
        mixed = tl.where(tile < 8, two, three)
        vocab = tl.load(Vocab + tile)
        remainder = mixed % vocab
        token = tl.where(remainder < 0, remainder + vocab, remainder) + tl.load(
            Offsets + tile
        )
        shard = token // SHARD_ROWS
        in_range = (token >= 0) & (shard < SHARD_COUNT)
        addr = tl.load(Shards + shard, in_range, other=0)
        col = tl.arange(0, EMBED_WIDTH)
        if FP8:
            base = addr.to(tl.pointer_type(tl.uint8))
        else:
            base = addr.to(tl.pointer_type(tl.bfloat16))
        value = tl.load(
            base + (token % SHARD_ROWS) * EMBED_WIDTH + col,
            in_range & (addr != 0),
            other=0,
        )
        if FP8:
            value = value.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16) * tl.load(
                Scale
            ).to(tl.bfloat16)
        tl.store(Embeddings + (row * 16 + tile) * EMBED_WIDTH + col, value)
    else:
        sb = _graph_page_block(StateTable, row, read_page, S_COLS, S_ROW, S_COL, S_POOL)
        state_col = (tile - 16) * STATE_TILE + tl.arange(0, STATE_TILE)
        state_value = tl.load(
            StatePool + sb * S_STRIDE + state_col,
            (state_col < STATE_WIDTH) & live,
            other=0,
        )
        tl.store(
            StateOut + row * STATE_WIDTH + state_col,
            state_value,
            state_col < STATE_WIDTH,
        )


def try_prepare_decode_embedding(
    sequence_plus_one,
    ids,
    state_table,
    ctx_table,
    state_pool,
    ctx_pool,
    page_size,
    eos,
    embedding,
):
    """Snapshot the decode state and produce local PLE embeddings in one launch."""
    if not (
        getattr(embedding, "supports_prefetch", False)
        and not embedding.distributed_reduce
        and embedding.eos_token_id == eos
        and embedding.context_len == 2
        and embedding.heads_per_ngram == 8
        and 0 < sequence_plus_one.numel() <= 2048
        and sequence_plus_one.shape == ids.shape
        and ids.ndim == 1
        and sequence_plus_one.dtype == torch.int32
        and ids.dtype in (torch.int32, torch.int64)
        and sequence_plus_one.is_contiguous()
        and ids.is_contiguous()
        and state_table.ndim == ctx_table.ndim == 2
        and state_table.shape[0] == ctx_table.shape[0] == ids.numel()
        and state_table.dtype == ctx_table.dtype == torch.int32
        and state_pool.ndim == 3
        and state_pool.shape[1:] == (9, 32768)
        and state_pool.stride()[1:] == (32768, 1)
        and state_pool.dtype == torch.bfloat16
        and ctx_pool.ndim == 2
        and ctx_pool.shape[1] == 2
        and ctx_pool.stride(1) == 1
        and ctx_pool.dtype == torch.int64
        and min(
            state_table.shape[1],
            ctx_table.shape[1],
            state_pool.shape[0],
            ctx_pool.shape[0],
        )
        > 0
        and page_size == 128
    ):
        return None
    device = ids.device
    shards = embedding._cpu_uva_pointers
    vocab = embedding.vocab_sizes
    offsets = embedding.offsets
    multipliers = embedding.multipliers
    if not (
        all(
            x is not None and x.is_cuda and x.device == device
            for x in (
                sequence_plus_one,
                state_table,
                ctx_table,
                state_pool,
                ctx_pool,
                shards,
                vocab,
                offsets,
                multipliers,
            )
        )
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
        and embedding.shards
        and embedding.shards[0].shape[1] == 512
        and embedding.shards[0].dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and vocab.shape == offsets.shape == (16,)
        and multipliers.shape == (3,)
        and shards.shape == (embedding.total_shards,)
        and all(
            x.dtype == torch.int64 and x.is_contiguous()
            for x in (shards, vocab, offsets, multipliers)
        )
    ):
        return None
    fp8 = embedding.shards[0].dtype == torch.float8_e4m3fn
    scale = embedding.weight_scale
    if fp8 and not (
        scale is not None
        and scale.device == device
        and scale.numel() == 1
        and scale.dtype in (torch.bfloat16, torch.float32)
    ):
        return None
    batch = ids.numel()
    state = torch.empty((batch, 9, 32768), device=device, dtype=torch.bfloat16)
    history = torch.empty((batch, 3), device=device, dtype=torch.int64)
    state_dest = torch.empty((batch,), device=device, dtype=torch.int64)
    ctx_dest = torch.empty_like(state_dest)
    embeddings = torch.empty((batch, 1, 8192), device=device, dtype=torch.bfloat16)
    state_width = 9 * 32768
    state_tile, warps = (1024, 4) if batch <= 32 else (2048, 2)
    _ple_decode_embedding[(batch, 16 + triton.cdiv(state_width, state_tile))](
        sequence_plus_one,
        ids,
        state_table,
        ctx_table,
        state_pool,
        ctx_pool,
        vocab,
        offsets,
        multipliers,
        shards,
        scale if fp8 else shards,
        state,
        history,
        state_dest,
        ctx_dest,
        embeddings,
        S_COLS=state_table.shape[1],
        C_COLS=ctx_table.shape[1],
        S_ROW=state_table.stride(0),
        S_COL=state_table.stride(1),
        C_ROW=ctx_table.stride(0),
        C_COL=ctx_table.stride(1),
        S_POOL=state_pool.shape[0],
        C_POOL=ctx_pool.shape[0],
        S_STRIDE=state_pool.stride(0),
        C_STRIDE=ctx_pool.stride(0),
        PAGE_SIZE=page_size,
        EOS=eos,
        STATE_WIDTH=state_width,
        STATE_TILE=state_tile,
        SHARD_ROWS=embedding.shard_rows,
        SHARD_COUNT=embedding.total_shards,
        EMBED_WIDTH=512,
        FP8=fp8,
        num_warps=warps,
    )
    return state, history, state_dest, ctx_dest, embeddings
