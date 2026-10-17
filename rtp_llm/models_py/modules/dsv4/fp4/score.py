"""SM120 indexer scoring against native 68-byte FP4 pages."""

import torch
import triton
import triton.language as tl
from rtp_llm.models_py.modules.dsv4.fp4.quantization import dequantize_rows, e2m1_value


@triton.jit
def _gather_cache(
    CACHE,
    TABLE,
    CU,
    Q,
    S,
    T: tl.constexpr,
    B: tl.constexpr,
    EB: tl.constexpr,
    STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    TILE: tl.constexpr,
):
    pos = tl.program_id(0) * TILE + tl.arange(0, TILE)
    req = tl.full((TILE,), 0, tl.int32)
    local = pos
    for b in range(B):
        begin, end = tl.load(CU + b), tl.load(CU + b + 1)
        here = (pos >= begin) & (pos < end)
        req, local = tl.where(here, b, req), tl.where(here, pos - begin, local)
    block = tl.load(
        TABLE + req * STRIDE + (local // EB) * COL_STRIDE,
        (pos < T) & (local // EB < TABLE_WIDTH),
        0,
    )
    valid = (pos < T) & (block > 0) & (block < NUM_BLOCKS)
    base = block.to(tl.int64) * EB * 68
    payload = tl.load(
        CACHE + base[:, None] + (local % EB)[:, None] * 64 + tl.arange(0, 64)[None, :],
        valid[:, None],
        0,
    )
    scales = tl.load(
        CACHE
        + base[:, None]
        + EB * 64
        + (local % EB)[:, None] * 4
        + tl.arange(0, 4)[None, :],
        valid[:, None],
        0,
    )
    tl.store(
        Q + pos[:, None] * 64 + tl.arange(0, 64)[None, :], payload, (pos < T)[:, None]
    )
    tl.store(
        S + pos[:, None] * 4 + tl.arange(0, 4)[None, :], scales, (pos < T)[:, None]
    )


def gather_cache(cache, payload, scales, block_table, cu_lengths):
    if cache.ndim != 3 or cache.shape[-1] != 68 or cache.dtype != torch.uint8:
        raise ValueError("FP4 gather requires [blocks, entries, 68] uint8 cache")
    if payload.shape != (payload.shape[0], 64) or scales.shape != (payload.shape[0], 4):
        raise ValueError("FP4 gather requires 64-byte payloads and four scale bytes")
    if not all(t.is_contiguous() for t in (cache, payload, scales, cu_lengths)):
        raise ValueError("FP4 gather requires contiguous cache, outputs and lengths")
    if payload.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("FP4 gather outputs must use uint8")
    if block_table.ndim != 2 or cu_lengths.numel() != block_table.shape[0] + 1:
        raise ValueError("FP4 gather requires cumulative lengths for each request")
    if payload.shape[0]:
        _gather_cache[(triton.cdiv(payload.shape[0], 16),)](
            cache,
            block_table,
            cu_lengths,
            payload,
            scales,
            payload.shape[0],
            cu_lengths.numel() - 1,
            cache.shape[1],
            block_table.stride(0),
            block_table.stride(1),
            cache.shape[0],
            block_table.shape[1],
            16,
        )


def prefill_score(q, weights, packed_k, k_scales, starts, ends, *, clean_logits=False):
    from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score

    keys = dequantize_rows(packed_k, k_scales)
    scores = v4_indexer_score(
        q.unsqueeze(0).contiguous(),
        keys.unsqueeze(0),
        weights.unsqueeze(0).contiguous(),
    ).squeeze(0)
    if clean_logits:
        columns = torch.arange(keys.shape[0], device=q.device)[None, :]
        scores.masked_fill_(
            (columns < starts[:, None]) | (columns >= ends[:, None]), float("-inf")
        )
    return scores


@triton.jit
def _paged_score(
    Q,
    W,
    CACHE,
    TABLE,
    LENGTHS,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    EB: tl.constexpr,
    WIDTH: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    TABLE_COL: tl.constexpr,
    TABLE_PER_ROW: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    keys = tl.program_id(1) * KEY_TILE + tl.arange(0, KEY_TILE)
    heads, dims = tl.arange(0, HEAD_TILE), tl.arange(0, 128)
    length = tl.load(LENGTHS + row)
    req = row if TABLE_PER_ROW else row // N
    block = tl.load(
        TABLE + req * TABLE_STRIDE + (keys // EB) * TABLE_COL, keys < WIDTH, 0
    )
    valid = (keys < WIDTH) & (keys < length) & (block > 0) & (block < NUM_BLOCKS)
    base = block.to(tl.int64) * EB * 68
    payload = tl.load(
        CACHE + base[:, None] + (keys % EB)[:, None] * 64 + dims[None, :] // 2,
        valid[:, None],
        0,
    )
    codes = (payload >> ((dims[None, :] & 1) * 4)) & 15
    scales = tl.load(
        CACHE
        + base[:, None]
        + EB * 64
        + (keys % EB)[:, None] * 4
        + dims[None, :] // 32,
        valid[:, None],
        0,
    ).to(tl.int32)
    kval = (e2m1_value(codes) * (scales << 23).to(tl.float32, bitcast=True)).to(
        tl.bfloat16
    )
    qval = tl.load(
        Q + (row * H + heads[:, None]) * 128 + dims[None, :], heads[:, None] < H, 0
    )
    per_head = tl.maximum(tl.dot(qval, tl.trans(kval)), 0.0)
    weight = tl.load(W + row * H + heads, heads < H, 0.0)
    score = tl.sum(per_head * weight[:, None], axis=0)
    tl.store(
        OUT + row * WIDTH + keys, tl.where(valid, score, float("-inf")), keys < WIDTH
    )


def paged_score(
    q, weights, pool, block_table, context_lens, *, block_size, max_ctx_len
):
    if q.ndim != 4 or q.dtype != torch.bfloat16 or q.shape[-1] != 128:
        raise ValueError("FP4 score requires reconstructed BF16 Q [B,N,H,128]")
    b, n, h, _ = q.shape
    rows = b * n
    if pool.dtype != torch.uint8 or pool.shape[-1] != 68 or not pool.is_contiguous():
        raise ValueError("FP4 score requires a contiguous native 68-byte cache")
    if weights.shape != (rows, h) or context_lens.numel() != rows:
        raise ValueError("FP4 score weights/lengths must match query rows")
    if (
        block_table.ndim != 2
        or block_table.shape[0] not in (b, rows)
        or max_ctx_len > block_table.shape[1] * block_size
    ):
        raise ValueError("FP4 score block table does not cover the query")
    if block_size <= 0 or pool.numel() % (block_size * 68):
        raise ValueError("FP4 score cache has invalid page geometry")
    out = torch.empty((rows, max_ctx_len), dtype=torch.float32, device=q.device)
    if rows and max_ctx_len:
        _paged_score[(rows, triton.cdiv(max_ctx_len, 32))](
            q.contiguous(),
            weights.contiguous(),
            pool,
            block_table,
            context_lens.contiguous(),
            out,
            n,
            h,
            block_size,
            max_ctx_len,
            block_table.stride(0),
            block_table.stride(1),
            block_table.shape[0] == rows,
            pool.numel() // (block_size * 68),
            max(16, triton.next_power_of_2(h)),
            32,
            num_warps=4,
        )
    return out
