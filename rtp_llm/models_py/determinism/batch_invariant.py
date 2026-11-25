"""Batch-invariant BF16 operators following vLLM's fixed reduction order.

Linear traverses K in fixed tiles. Paged attention uses one unsplit online
softmax per query and KV head. TP all-reduce sums in ascending group-rank order.
MoE uses fixed expert tiles and sums weighted outputs in routing order.
"""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

_ALL_REDUCE_WORKSPACE_BYTES = 16 * 1024 * 1024


@triton.jit(do_not_specialize=["count"])
def _fixed_rank_sum(
    gathered, output, count, WORLD_SIZE: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = offsets < count
    acc = tl.full((BLOCK,), 0, tl.float32)
    for rank in tl.static_range(WORLD_SIZE):
        values = tl.load(gathered + rank * count + offsets, live, other=0)
        acc += values.to(tl.float32)
    tl.store(output + offsets, acc, live)


def _all_reduce_chunks(source: torch.Tensor, dest: torch.Tensor, limit: int):
    """Slice matching logical rows without flattening a strided tensor into a full copy."""
    if source.numel() <= limit:
        yield source, dest
        return
    row_size = source.numel() // source.size(0)
    if row_size > limit:
        for row in range(source.size(0)):
            yield from _all_reduce_chunks(source[row], dest[row], limit)
    else:
        rows = limit // row_size
        for start in range(0, source.size(0), rows):
            yield source[start : start + rows], dest[start : start + rows]


def all_reduce_batch_invariant(
    tensor: torch.Tensor, group: dist.ProcessGroup, *, inplace: bool = True
) -> torch.Tensor:
    """Sum like-shaped BF16 tensors in TP group-rank order and round once per element."""
    if tensor.dtype != torch.bfloat16 or tensor.layout != torch.strided:
        raise ValueError("batch-invariant all-reduce requires a strided BF16 tensor")
    if group is None:
        raise ValueError("batch-invariant all-reduce requires a TP process group")
    world_size = dist.get_world_size(group)
    if world_size < 1:
        raise ValueError(
            "batch-invariant all-reduce requires membership in the TP group"
        )
    if world_size == 1 or tensor.numel() == 0:
        return tensor if inplace else tensor.clone()

    if (
        inplace
        and not tensor.is_contiguous()
        and torch._debug_has_internal_overlap(tensor) == 1
    ):
        raise ValueError(
            "in-place batch-invariant all-reduce requires non-overlapping elements"
        )
    output = tensor if inplace else torch.empty_like(tensor)
    # Reserve space for one rank-major gather and at most one local staging
    # buffer. Chunk boundaries are independent of strides on individual ranks.
    capacity = max(
        1, _ALL_REDUCE_WORKSPACE_BYTES // ((world_size + 1) * tensor.element_size())
    )
    capacity = min(capacity, tensor.numel())
    gathered = tensor.new_empty(world_size * capacity)
    staging = None
    if not (tensor.is_contiguous() and output.is_contiguous()):
        staging = tensor.new_empty(capacity)

    with torch.cuda.device_of(tensor):
        for source, dest in _all_reduce_chunks(tensor, output, capacity):
            count = source.numel()
            if source.is_contiguous():
                send = source.view(-1)
            else:
                send = staging[:count]
                send.view_as(source).copy_(source)
            # Synchronous c10d semantics order NCCL and the caller's current
            # CUDA stream in both directions, including workspace reuse.
            dist.all_gather_into_tensor(
                gathered[: world_size * count], send, group=group
            )
            result = dest.view(-1) if dest.is_contiguous() else staging[:count]
            _fixed_rank_sum[(triton.cdiv(count, 256),)](
                gathered, result, count, WORLD_SIZE=world_size, BLOCK=256
            )
            if not dest.is_contiguous():
                dest.copy_(result.view_as(dest))
    return output


@triton.jit(do_not_specialize=["M"])
def _linear(
    x,
    weight,
    bias,
    output,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    STRIDE_WN: tl.constexpr,
    STRIDE_WK: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)

    # Each output element is owned by one CTA. M only changes grid size and masks.
    for start in range(0, K, BK):
        inner = start + ks
        a = tl.load(
            x + rows[:, None] * K + inner[None, :],
            (rows[:, None] < M) & (inner[None, :] < K),
            other=0,
        )
        b = tl.load(
            weight + inner[:, None] * STRIDE_WK + cols[None, :] * STRIDE_WN,
            (inner[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(bias + cols, cols < N, other=0).to(tl.float32)[None, :]
    tl.store(
        output + rows[:, None] * N + cols[None, :],
        acc,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def batch_invariant_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    output_fp32: bool = False,
) -> torch.Tensor:
    """Multiply BF16 activations by a BF16 weight, with an optional scalar or per-feature bias."""
    n, k = weight.shape
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.size(-1) != k
        or x.device != weight.device
    ):
        raise ValueError(
            "batch-invariant linear requires compatible BF16 input and weight"
        )
    if bias is not None:
        if bias.device != x.device:
            raise ValueError("batch-invariant linear requires bias on the input device")
        # The kernel reads one bias row shared by every input token.
        row_shape = (1,) * (x.ndim - 1) + (n,)
        bias = bias.expand(row_shape).reshape(n).contiguous()
    output = torch.empty(
        (*x.shape[:-1], n),
        dtype=torch.float32 if output_fp32 else torch.bfloat16,
        device=x.device,
    )
    if output.numel() == 0:
        return output

    rows = x.reshape(-1, k).contiguous()
    m = rows.size(0)
    bm, bn, bk = 16, 16 if n <= 16 else 64, 64
    _linear[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        rows,
        weight,
        bias,
        output,
        m,
        n,
        k,
        *weight.stride(),
        bm,
        bn,
        bk,
        bias is not None,
        num_warps=4,
        num_stages=3,
    )
    return output


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


# Each route owns one output row. Keep the same unsplit K traversal in both
# projections regardless of other tokens selecting this expert.
_MOE_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "num_warps": 8,
    "num_stages": 4,
}
_EXPERT_WORKSPACE_BYTES = 64 * 1024 * 1024


def batch_invariant_moe_plan(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> tuple[dict[str, int], int]:
    """Return the fixed GEMM tile and the maximum token count per projection chunk."""
    rows, hidden = hidden_states.shape
    experts, _, intermediate = w2.shape
    if (
        hidden_states.dtype != torch.bfloat16
        or w1.dtype != torch.bfloat16
        or w2.dtype != torch.bfloat16
        or min(hidden, intermediate) <= 0
        or w2.size(1) != hidden
        or w1.shape != (experts, 2 * intermediate, hidden)
    ):
        raise ValueError("batch-invariant MoE requires compatible BF16 expert weights")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.size(0) != rows
        or not 0 < topk_ids.size(1) <= experts
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or topk_weights.dtype != torch.float32
        or any(
            t.device != hidden_states.device for t in (w1, w2, topk_ids, topk_weights)
        )
    ):
        raise ValueError(
            "batch-invariant MoE requires matching expert IDs and FP32 routing weights"
        )

    top_k = topk_ids.size(1)
    # At peak the activation is live alongside either the up/gate or the down
    # projection. Bound these route buffers even for a long prefill.
    bytes_per_row = (
        top_k
        * (intermediate + max(2 * intermediate, hidden))
        * hidden_states.element_size()
    )
    return _MOE_GEMM_CONFIG, max(1, _EXPERT_WORKSPACE_BYTES // bytes_per_row)
