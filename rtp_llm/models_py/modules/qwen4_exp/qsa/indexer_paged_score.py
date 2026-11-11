"""Paged (decode) 4-head KSA indexer scoring kernel.

Contract mirrors ``fp8_paged_indexer_score``: scores a decode-time query
against the paged indexer KV pool using a ``block_table`` for logical →
physical indirection.  One program per ``(batch, token)`` row tiles over
``max_ctx_len`` output columns in chunks so large pool sizes do not blow
up register usage.

Pool layout is bf16 per entry (each entry is ``[D]`` bf16). q is also
received as bf16 -- the caller is responsible for dequantising before
calling.  :mod:`indexer_score` exercises the same score math, but its legacy
FP8 + scale input is not ABI-compatible with this production side pool.

Per row::

    logits[row, t] = sum_h weight[row, h] * relu(dot(q[row, h, :],
        pool[slot(row, t), :]))

    batch(row) = row // next_n
    slot(row, t) = block_table[batch(row), t // block_size] * block_size + t % block_size
"""

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_N = 64


@triton.jit
def _qsa_paged_score_tiled_kernel(
    q_ptr,
    w_ptr,
    pool_ptr,
    bt_ptr,
    ctx_ptr,
    out_ptr,
    stride_q_row,
    next_n,
    max_ctx_len,
    max_blocks,
    pool_blocks,
    H: tl.constexpr,
    D: tl.constexpr,
    block_size: tl.constexpr,
    TILE_R: tl.constexpr,
    LANES: tl.constexpr,
    BN: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
    SKIP_EMPTY: tl.constexpr = False,
):
    """Tensor-core tiled paged indexer scoring (vLLM qsa_indexer design).

    One program owns TILE_R query rows of one batch and walks K_TILES column
    tiles, so each gathered pool tile is loaded once for all rows and heads of
    the group instead of once per (row, head, column-block). Scores run on
    tensor cores via ``tl.dot(keys, query)``; the per-(row, head) weight is
    applied to ``relu(dot)`` before the head sum, which reproduces the scalar
    kernel's ``sum_h weight * relu(dot_h)`` contract exactly for any weight
    sign.
    """
    H_PAD: tl.constexpr = LANES // TILE_R
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(D))
    batch = tl.program_id(0)
    row_tile = tl.program_id(1)
    k_group = tl.program_id(2)

    row_ids = tl.arange(0, TILE_R)
    tile_rows = row_tile * TILE_R + row_ids
    row_live = tile_rows < next_n
    abs_rows = batch * next_n + tile_rows
    context = tl.load(ctx_ptr + abs_rows, mask=row_live, other=0)

    lanes = tl.arange(0, LANES)
    lane_row = lanes // H_PAD
    lane_head = lanes % H_PAD
    lane_rows = row_tile * TILE_R + lane_row
    lane_abs_rows = batch * next_n + lane_rows
    lane_valid = (lane_rows < next_n) & (lane_head < H)

    if SKIP_EMPTY:
        tl.static_assert(K_TILES == 1)
    if SKIP_EMPTY and k_group * K_TILES * BN >= tl.max(context, 0):
        # Decode reserves the graph's maximum width. Empty tiles still need
        # -inf output, but do not need query/cache loads or a matrix multiply.
        cols = k_group * BN + tl.arange(0, BN)
        tl.store(
            out_ptr + abs_rows[None, :] * max_ctx_len + cols[:, None],
            float("-inf"),
            mask=(cols[:, None] < max_ctx_len) & row_live[None, :],
        )
        return

    dims = tl.arange(0, BLOCK_D)
    query = tl.load(
        q_ptr
        + lane_abs_rows[None, :] * stride_q_row
        + lane_head[None, :] * D
        + dims[:, None],
        mask=lane_valid[None, :] & (dims[:, None] < D),
        other=0.0,
    )
    weights = tl.load(
        w_ptr + lane_abs_rows[None, :] * H + lane_head[None, :],
        mask=lane_valid[None, :],
        other=0.0,
    )

    columns = tl.arange(0, BN)
    k_start = k_group * K_TILES
    k_end = tl.minimum(k_start + K_TILES, tl.cdiv(max_ctx_len, BN))
    for tile in tl.range(k_start, k_end, num_stages=STAGES):
        cols = tile * BN + columns
        col_live = cols < max_ctx_len
        bi = cols // block_size
        ei = cols % block_size
        in_table = bi < max_blocks
        physical = tl.load(
            bt_ptr + batch * max_blocks + bi,
            mask=col_live & in_table,
            other=0,
        )
        has_block = col_live & in_table & (physical > 0) & (physical < pool_blocks)
        safe_phys = tl.maximum(physical, 0).to(tl.int64)
        slot = safe_phys * block_size + ei
        keys = tl.load(
            pool_ptr + slot[:, None] * D + dims[None, :],
            mask=has_block[:, None] & (dims[None, :] < D),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query)
        scores = tl.reshape(scores, (BN, TILE_R, H_PAD))
        score = tl.sum(
            tl.maximum(scores, 0.0) * tl.reshape(weights, (1, TILE_R, H_PAD)), 2
        )
        valid = has_block[:, None] & (cols[:, None] < context[None, :])
        tl.store(
            out_ptr + abs_rows[None, :] * max_ctx_len + cols[:, None],
            tl.where(valid, score, float("-inf")),
            mask=col_live[:, None] & row_live[None, :],
        )


def _score_tiled_config(next_n: int, heads: int, batch: int, columns: int) -> tuple:
    """(TILE_R, BN, K_TILES, stages, warps) for the tiled score kernel.

    Prefill batches share vLLM's tuned tile (64 rows x 16 K-tiles); decode
    shapes keep one K-tile per program so small batches still spread across
    the whole device.
    """
    h_pad = triton.next_power_of_2(heads)
    # Keep the dot N dimension (TILE_R * heads) at least 16.
    tile_r = max(1, 16 // h_pad)
    if next_n >= 64:
        return 64, 64, 16, 2, 4
    while tile_r < next_n and tile_r * h_pad < 64:
        tile_r *= 2
    programs = batch * triton.cdiv(columns, 64)
    if programs < 16384:
        k_tiles = 1
    elif programs < 32768:
        k_tiles = 2
    elif programs < 131072:
        k_tiles = 4
    else:
        k_tiles = 8
    return tile_r, 64, k_tiles, 2, 4


def _score_tiled_supported(q_bf16, kv_pool, next_n, heads) -> bool:
    if torch.version.hip is not None or not q_bf16.is_cuda:
        return False
    if kv_pool.dtype != torch.bfloat16:
        return False
    d = int(q_bf16.shape[3])
    if d <= 0 or d & (d - 1) or d > 256:
        return False
    if heads > 32:
        return False
    tile_r = 64 if next_n >= 64 else max(1, 16 // triton.next_power_of_2(heads))
    if tile_r * heads > 256:
        return False
    # Other single-row geometries retain the scalar fallback. The qualified
    # SM90 ordinary-decode geometry is selected explicitly by the caller.
    return 2 <= next_n


@triton.jit
def _qsa_paged_score_kernel(
    q_ptr,
    w_ptr,
    pool_ptr,
    bt_ptr,
    ctx_ptr,
    out_ptr,
    stride_q_row,
    max_ctx_len,
    max_blocks,
    pool_blocks,
    next_n,
    H: tl.constexpr,
    D: tl.constexpr,
    block_size: tl.constexpr,
    BN: tl.constexpr,
    SKIP_EMPTY: tl.constexpr = False,
):
    rd = tl.arange(0, D)
    row = tl.program_id(0)
    blk = tl.program_id(1)
    batch_idx = row // next_n

    rn = blk * BN + tl.arange(0, BN)
    n_mask = rn < max_ctx_len

    T = tl.load(ctx_ptr + row)
    if not SKIP_EMPTY or blk * BN < T:
        in_context = n_mask & (rn < T)

        bi = rn // block_size
        ei = rn % block_size
        in_block_table = bi < max_blocks
        physical_bt = tl.load(
            bt_ptr + batch_idx * max_blocks + bi,
            mask=in_context & in_block_table,
            other=0,
        )
        # Block zero (and negative IDs) are allocation sentinels.  The upper bound
        # is checked again in the kernel even though the wrapper validates it, so a
        # bad table can never turn into an out-of-bounds pool read.
        has_block = (
            in_context
            & in_block_table
            & (physical_bt > 0)
            & (physical_bt < pool_blocks)
        )
        slot = physical_bt * block_size + ei

        acc = tl.zeros((BN,), dtype=tl.float32)
        for h in tl.static_range(H):
            qh = (
                tl.load(q_ptr + row * stride_q_row + h * D + rd)
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            w_h = tl.load(w_ptr + row * H + h)

            k = (
                tl.load(
                    pool_ptr + slot[:, None] * D + rd[None, :],
                    mask=has_block[:, None],
                    other=0.0,
                )
                .to(tl.bfloat16)
                .to(tl.float32)
            )
            dot = tl.sum(qh[None, :] * k, axis=1)
            # The model contract is ``weight * relu(dot)``.  Applying ReLU after
            # the multiply happens to agree for the released positive dequant
            # weights, but changes the public helper's semantics for negative
            # weights and can turn a negative contribution into zero.
            acc += tl.where(has_block, tl.maximum(dot, 0.0) * w_h, 0.0)

        tl.store(
            out_ptr + row * max_ctx_len + rn,
            tl.where(has_block, acc, float("-inf")),
            mask=n_mask,
        )
    else:
        tl.store(out_ptr + row * max_ctx_len + rn, float("-inf"), mask=n_mask)


def qsa_paged_indexer_score(
    q_bf16: torch.Tensor,
    weight: torch.Tensor,
    kv_pool: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    block_size: int,
    max_ctx_len: Optional[int] = None,
    block_n: int = _BLOCK_N,
    validate_block_table: bool = True,
) -> torch.Tensor:
    """Score one decode step against the paged indexer KV pool.

    Args:
        q_bf16: ``[B, next_n, H, D]`` bf16.
        weight: ``[B * next_n, H]`` fp32 -- per-(row, head) weight.
            Qwen4 folds the q dequant scale into it.
        kv_pool: ``[total_slots, D]`` bf16 -- flat indexer KV pool view.
        block_table: ``[B, max_blocks]`` int32 -- logical -> physical block map.
        context_lens: ``[B, next_n]`` int32 -- valid compressed entries / row.
        block_size: compressed entries per physical pool block
            (``kernel_seq_size_per_block / ratio``).  For the qwen4 production
            shape with ratio=4 and kernel_block=256, block_size=64.
        max_ctx_len: output column count.  Defaults to ``context_lens.max()``
            (requires synchronisation).
        validate_block_table: Check every physical block ID on the host.  The
            QSA runtime already validates its required blocks before scoring
            and can disable this redundant device-to-host synchronization.
            The kernel still masks out-of-range IDs to prevent invalid reads.

    Returns:
        ``[B * next_n, max_ctx_len]`` fp32 logits; -inf past the per-row
        ``context_lens`` boundary or for an unallocated / missing logical block.
    """
    if not q_bf16.is_cuda:
        raise ValueError("qsa_paged_indexer_score requires CUDA tensors")
    if q_bf16.dtype != torch.bfloat16 or q_bf16.dim() != 4:
        raise ValueError("q_bf16 must be a rank-4 bf16 CUDA tensor")
    B, next_n, H, D = q_bf16.shape
    M = B * next_n

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if block_n <= 0 or block_n & (block_n - 1):
        raise ValueError(f"block_n must be a positive power of two, got {block_n}")

    if weight.shape != (M, H) or weight.dtype != torch.float32:
        raise ValueError(f"weight must be fp32 [{M}, {H}]")
    if kv_pool.dim() != 2 or kv_pool.shape[1] != D or kv_pool.dtype != torch.bfloat16:
        raise ValueError(f"kv_pool must be bf16 [total_slots, {D}]")
    if (
        block_table.dim() != 2
        or block_table.shape[0] != B
        or block_table.dtype != torch.int32
    ):
        raise ValueError(f"block_table must be int32 [{B}, max_blocks]")
    if context_lens.shape != (B, next_n) or context_lens.dtype != torch.int32:
        raise ValueError(f"context_lens must be int32 [{B}, {next_n}]")
    tensors = (weight, kv_pool, block_table, context_lens)
    if any(not tensor.is_cuda or tensor.device != q_bf16.device for tensor in tensors):
        raise ValueError(
            "all qsa_paged_indexer_score tensors must share one CUDA device"
        )
    if kv_pool.shape[0] % block_size:
        raise ValueError(
            f"kv_pool slots ({kv_pool.shape[0]}) must be divisible by block_size "
            f"({block_size})"
        )

    pool_blocks = kv_pool.shape[0] // block_size
    if validate_block_table:
        max_physical_block = int(block_table.max().item()) if block_table.numel() else 0
        if max_physical_block >= pool_blocks:
            raise ValueError(
                f"block_table physical block id {max_physical_block} is outside "
                f"kv_pool capacity [0, {pool_blocks})"
            )

    q_bf16 = q_bf16.contiguous()
    weight = weight.contiguous()
    kv_pool = kv_pool.contiguous()
    block_table = block_table.contiguous()
    context_lens = context_lens.contiguous()

    max_blocks = block_table.shape[1]
    if max_ctx_len is None:
        max_ctx_len = int(context_lens.max())
    logits = torch.empty((M, max_ctx_len), device=q_bf16.device, dtype=torch.float32)
    if M == 0 or max_ctx_len == 0:
        return logits

    sm90_decode = (
        next_n == 1
        and H == 4
        and D == 128
        and 1 <= B <= 64
        and torch.version.hip is None
        and torch.cuda.get_device_capability(q_bf16.device) == (9, 0)
    )
    # At B16 and below, the scalar tile is cheaper than padded tensor-core work.
    sm90_tiled = sm90_decode and B >= 32
    if sm90_decode and not sm90_tiled:
        block_n = 16
    if sm90_tiled or _score_tiled_supported(q_bf16, kv_pool, next_n, H):
        if sm90_tiled:
            tile_r, bn, k_tiles, stages, warps = 1, 64, 1, 1, 2
        else:
            tile_r, bn, k_tiles, stages, warps = _score_tiled_config(
                next_n, H, B, max_ctx_len
            )
        lanes = tile_r * triton.next_power_of_2(H)
        tiled_grid = (
            B,
            triton.cdiv(next_n, tile_r),
            triton.cdiv(max_ctx_len, bn * k_tiles),
        )
        _qsa_paged_score_tiled_kernel[tiled_grid](
            q_bf16,
            weight,
            kv_pool,
            block_table,
            context_lens.view(-1),
            logits,
            stride_q_row=H * D,
            next_n=next_n,
            max_ctx_len=max_ctx_len,
            max_blocks=max_blocks,
            pool_blocks=pool_blocks,
            H=H,
            D=D,
            block_size=block_size,
            TILE_R=tile_r,
            LANES=lanes,
            BN=bn,
            K_TILES=k_tiles,
            STAGES=stages,
            SKIP_EMPTY=sm90_tiled,
            num_warps=warps,
        )
        return logits
    # The flattened (batch, token) rows are contiguous, each with H * D values.
    grid = (M, triton.cdiv(max_ctx_len, block_n))
    _qsa_paged_score_kernel[grid](
        q_bf16,
        weight,
        kv_pool,
        block_table,
        context_lens.view(-1),
        logits,
        stride_q_row=H * D,
        max_ctx_len=max_ctx_len,
        max_blocks=max_blocks,
        pool_blocks=pool_blocks,
        next_n=next_n,
        H=H,
        D=D,
        block_size=block_size,
        BN=block_n,
        SKIP_EMPTY=sm90_decode,
        num_warps=2 if sm90_decode else 4,
    )
    return logits
