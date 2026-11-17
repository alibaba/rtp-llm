"""Causal paged attention with one online softmax per query and KV head.

The fixed tiles and unsplit softmax follow vLLM's batch-invariant 2D
attention. Each query uses its own causal length, including within prefill.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attention(
    query,
    key,
    value,
    output,
    block_table,
    request_ids,
    kv_lengths,
    TABLE_STRIDE: tl.constexpr,
    Q_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    QUERIES_PER_KV: tl.constexpr,
    SCALE: tl.constexpr,
    TILE_KV: tl.constexpr,
    TILE_Q: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    kv_head = tl.program_id(1)
    request = tl.load(request_ids + token)
    length = tl.load(kv_lengths + token)

    group_heads = tl.arange(0, TILE_Q)
    heads = kv_head * QUERIES_PER_KV + group_heads
    dims = tl.arange(0, HEAD_SIZE)
    q = tl.load(
        query + token * Q_STRIDE + heads[:, None] * HEAD_SIZE + dims[None, :],
        group_heads[:, None] < QUERIES_PER_KV,
        other=0,
    )
    maximum = tl.full((TILE_Q,), float("-inf"), tl.float32)
    denominator = tl.full((TILE_Q,), 0, tl.float32)
    numerator = tl.full((TILE_Q, HEAD_SIZE), 0, tl.float32)

    for start in range(0, length, TILE_KV):
        positions = start + tl.arange(0, TILE_KV)
        live = positions < length
        pages = tl.load(
            block_table + request * TABLE_STRIDE + positions // PAGE_SIZE,
            live,
            other=0,
        ).to(tl.int64)
        offsets = (
            pages * PAGE_STRIDE
            + kv_head * HEAD_STRIDE
            + (positions % PAGE_SIZE) * TOKEN_STRIDE
        )
        k = tl.load(key + offsets[None, :] + dims[:, None], live[None, :], other=0)
        v = tl.load(value + offsets[:, None] + dims[None, :], live[:, None], other=0)
        scores = tl.dot(q, k) * SCALE
        scores = tl.where(live[None, :], scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        weights = tl.exp2(scores - new_maximum[:, None])
        rescale = tl.exp2(maximum - new_maximum)
        numerator = tl.dot(weights.to(tl.bfloat16), v, numerator * rescale[:, None])
        denominator = denominator * rescale + tl.sum(weights, axis=1)
        maximum = new_maximum

    result = numerator / tl.where(denominator > 0, denominator, 1)[:, None]
    tl.store(
        output
        + token * (QUERIES_PER_KV * tl.num_programs(1) * HEAD_SIZE)
        + heads[:, None] * HEAD_SIZE
        + dims[None, :],
        result,
        group_heads[:, None] < QUERIES_PER_KV,
    )


def batch_invariant_paged_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    kv_lengths: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run BF16 GQA with a cache laid out as [pages, 2, kv_heads, page_size, dim]."""
    tokens, query_heads, head_size = query.shape
    _, _, kv_heads, page_size, _ = cache.shape
    if query.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise ValueError("batch-invariant attention requires BF16 queries and KV cache")
    if query.stride(-1) != 1 or query.stride(-2) != head_size:
        query = query.contiguous()
    query_per_kv = query_heads // kv_heads
    output = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    if tokens == 0:
        return output

    _paged_attention[(tokens, kv_heads)](
        query,
        cache[:, 0],
        cache[:, 1],
        output,
        block_table,
        request_ids,
        kv_lengths,
        block_table.stride(0),
        query.stride(0),
        cache.stride(0),
        cache.stride(2),
        cache.stride(3),
        page_size,
        head_size,
        query_per_kv,
        scale * 1.4426950408889634,
        TILE_KV=32,
        TILE_Q=max(16, triton.next_power_of_2(query_per_kv)),
        num_warps=4,
        num_stages=2,
    )
    return output
