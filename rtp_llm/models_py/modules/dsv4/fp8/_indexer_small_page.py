"""FP8 paged indexer scores for 16-entry GLM53 cache pages."""

import torch
import triton
import triton.language as tl


@triton.jit
def _score(
    Q,
    W,
    Cache,
    Table,
    Lengths,
    Out,
    NEXT: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_TILE: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr,
):
    batch = tl.program_id(0)
    queries = tl.program_id(1) * QUERY_TILE + tl.arange(0, QUERY_TILE)
    lengths = tl.load(Lengths + batch * NEXT + queries, queries < NEXT, other=0)
    cols = tl.program_id(2) * KEY_TILE + tl.arange(0, KEY_TILE)
    valid_keys = (
        (cols < WIDTH) & (cols < tl.max(lengths, 0)) & (cols < TABLE_STRIDE * PAGE)
    )
    page = tl.load(Table + batch * TABLE_STRIDE + cols // PAGE, valid_keys, other=0).to(
        tl.int64
    )
    offsets = cols % PAGE
    rows = tl.arange(0, QUERY_TILE * HEAD_TILE)
    q_rows = tl.program_id(1) * QUERY_TILE + rows // HEAD_TILE
    heads = rows % HEAD_TILE
    dim = tl.arange(0, 128)
    q = tl.load(
        Q.to(tl.pointer_type(tl.uint8))
        + ((batch * NEXT + q_rows[:, None]) * HEADS + heads[:, None]) * 128
        + dim[None, :],
        (q_rows[:, None] < NEXT) & (heads[:, None] < HEADS),
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    k = tl.load(
        Cache + page[:, None] * (PAGE * 132) + offsets[:, None] * 128 + dim[None, :],
        valid_keys[:, None],
        other=0,
    ).to(tl.float8e4nv, bitcast=True)
    scales = tl.load(
        (Cache + page * (PAGE * 132) + PAGE * 128 + offsets * 4).to(
            tl.pointer_type(tl.float32)
        ),
        valid_keys,
        other=0,
    )
    weights = tl.load(
        W + (batch * NEXT + q_rows) * HEADS + heads,
        (q_rows < NEXT) & (heads < HEADS),
        other=0,
    )
    # Put heads on the contiguous output axis to keep their reduction local.
    dots = tl.dot(k, tl.trans(q), out_dtype=tl.float32)
    weighted = tl.maximum(dots, 0) * weights[None, :]
    scores = (
        tl.trans(tl.sum(weighted.reshape(KEY_TILE, QUERY_TILE, HEAD_TILE), 2))
        * scales[None, :]
    )
    valid = valid_keys[None, :] & (cols[None, :] < lengths[:, None])
    tl.store(
        Out + (batch * NEXT + queries[:, None]) * WIDTH + cols[None, :],
        tl.where(valid, scores, -float("inf")),
        (queries[:, None] < NEXT) & (cols[None, :] < WIDTH),
    )


def is_supported(q, block_size):
    return (
        block_size == 16
        and q.is_cuda
        and q.dtype == torch.float8_e4m3fn
        and q.ndim == 4
        and q.shape[-1] == 128
        and 1 <= q.shape[-2] <= 128
    )


def small_page_indexer_score(q, weights, cache, table, lengths, block_size, width):
    if not is_supported(q, block_size):
        raise ValueError(
            "small-page indexer requires CUDA FP8 Q [B,N,H,128] and 16-entry pages"
        )
    batch, next_n, heads, _ = q.shape
    assert weights.shape == (batch * next_n, heads) and weights.dtype == torch.float32
    assert lengths.shape == (batch, next_n) and lengths.dtype == torch.int32
    assert table.shape[0] == batch and table.dtype == torch.int32
    assert cache.dtype == torch.uint8 and cache.is_contiguous()
    out = torch.empty((batch * next_n, width), dtype=torch.float32, device=q.device)
    if out.numel():
        head_tile = max(16, triton.next_power_of_2(heads))
        query_tile = min(4, triton.next_power_of_2(next_n), 256 // head_tile)
        if batch < 32:
            query_tile = min(query_tile, 2)
        key_tile = 256 if query_tile >= 4 else (64 if width <= 64 else 128)
        _score[(batch, triton.cdiv(next_n, query_tile), triton.cdiv(width, key_tile))](
            q.contiguous(),
            weights.contiguous(),
            cache,
            table.contiguous(),
            lengths.contiguous(),
            out,
            next_n,
            heads,
            head_tile,
            block_size,
            table.shape[1],
            width,
            query_tile,
            key_tile,
            num_warps=8 if key_tile == 256 else 4,
        )
    return out
