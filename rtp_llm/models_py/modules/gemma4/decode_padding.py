"""Shared elementwise query/position padding for paged decode wrappers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_decode_kernel(
    Q,
    POS,
    PAD_Q,
    PAD_POS,
    COUNT: tl.constexpr,
    POS_COUNT: tl.constexpr,
    QUERIES: tl.constexpr,
    PADDED: tl.constexpr,
    DIM: tl.constexpr,
    HEADS: tl.constexpr,
    Q_BATCH_STRIDE: tl.constexpr,
    Q_HEAD_STRIDE: tl.constexpr,
    Q_QUERY_STRIDE: tl.constexpr,
    POS_ROW_STRIDE: tl.constexpr,
    POS_COL_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    query = (index // DIM) % PADDED
    head = (index // (PADDED * DIM)) % HEADS
    batch = index // (HEADS * PADDED * DIM)
    source = (
        batch * Q_BATCH_STRIDE
        + head * Q_HEAD_STRIDE
        + query * Q_QUERY_STRIDE
        + index % DIM
    )
    value = tl.load(Q + source, mask=(index < COUNT) & (query < QUERIES), other=0)
    tl.store(PAD_Q + index, value, mask=index < COUNT)
    position_query = index % PADDED
    position_source = (
        index // PADDED
    ) * POS_ROW_STRIDE + position_query * POS_COL_STRIDE
    position = tl.load(
        POS + position_source,
        mask=(index < POS_COUNT) & (position_query < QUERIES),
        other=0,
    ).to(tl.int32)
    tl.store(PAD_POS + index, position, mask=index < POS_COUNT)


def pad_decode_queries(q: torch.Tensor, positions: torch.Tensor, padded: int):
    """Copy BF16 Q and integer positions into zero-padded contiguous buffers.

    Inputs follow the decode layout gate: contiguous Q[B,H,T,D] or its canonical
    view of contiguous [B,T,H,D] storage, and positions [B,T] on the same CUDA
    device. Positions may be strided int32 or int64. Query padding reads either
    layout directly; zero-filled padded rows remain mandatory for the partial.
    """
    batch, heads, queries, dim = q.shape
    if (
        queries == padded
        and positions.dtype == torch.int32
        and positions.is_contiguous()
    ):
        return q, positions
    output = torch.empty((batch, heads, padded, dim), dtype=q.dtype, device=q.device)
    padded_positions = torch.empty((batch, padded), dtype=torch.int32, device=q.device)
    _pad_decode_kernel[(triton.cdiv(output.numel(), 1024),)](
        q,
        positions,
        output,
        padded_positions,
        output.numel(),
        padded_positions.numel(),
        queries,
        padded,
        dim,
        heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        positions.stride(0),
        positions.stride(1),
        1024,
    )
    return output, padded_positions
