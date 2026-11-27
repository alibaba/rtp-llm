"""Copy MiniMax-M3.1 CP prefix pages between tagged cache and working pages."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_prefix_pages(
    POOL,
    TABLE,
    SEND,
    REQUESTS,
    BLOCKS,
    OFFSETS,
    WIDTH: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    TABLE_ROW_STRIDE: tl.constexpr,
    TABLE_COL_STRIDE: tl.constexpr,
    SEND_STRIDE: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    SEND_ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    page = tl.program_id(0)
    byte = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    request = tl.load(REQUESTS + page)
    block = tl.load(BLOCKS + page)
    table_valid = (
        (request >= 0) & (request < TABLE_ROWS) & (block >= 0) & (block < TABLE_COLS)
    )
    tl.device_assert(table_valid, "M3.1 CP prefix table index out of range")
    source = tl.load(
        TABLE + request * TABLE_ROW_STRIDE + block * TABLE_COL_STRIDE,
        mask=table_valid,
        other=-1,
    ).to(tl.int64)
    destination = tl.load(OFFSETS + page)
    tl.device_assert(
        (source >= 0)
        & (source < POOL_ROWS)
        & (destination >= 0)
        & (destination < SEND_ROWS),
        "M3.1 CP prefix physical page out of range",
    )
    valid = (
        (source >= 0)
        & (source < POOL_ROWS)
        & (destination >= 0)
        & (destination < SEND_ROWS)
    )
    value = tl.load(
        POOL + source * POOL_STRIDE + byte, mask=(byte < WIDTH) & valid, other=0
    )
    tl.store(
        SEND + destination * SEND_STRIDE + byte, value, mask=(byte < WIDTH) & valid
    )


@triton.jit
def _restore_prefix_pages(
    RECEIVED,
    WORKING,
    DESTINATIONS,
    OFFSETS,
    WIDTH: tl.constexpr,
    RECV_STRIDE: tl.constexpr,
    WORK_STRIDE: tl.constexpr,
    RECV_ROWS: tl.constexpr,
    WORK_ROWS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    page = tl.program_id(0)
    byte = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    source = tl.load(OFFSETS + page)
    destination = tl.load(DESTINATIONS + page)
    tl.device_assert(
        (source >= 0)
        & (source < RECV_ROWS)
        & (destination >= 0)
        & (destination < WORK_ROWS),
        "M3.1 CP restore page out of range",
    )
    valid = (
        (source >= 0)
        & (source < RECV_ROWS)
        & (destination >= 0)
        & (destination < WORK_ROWS)
    )
    value = tl.load(
        RECEIVED + source * RECV_STRIDE + byte, mask=(byte < WIDTH) & valid, other=0
    )
    tl.store(
        WORKING + destination * WORK_STRIDE + byte, value, mask=(byte < WIDTH) & valid
    )


def pack_prefix_pages(
    pool: torch.Tensor,
    table: torch.Tensor,
    send: torch.Tensor,
    requests: torch.Tensor,
    blocks: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    if requests.numel():
        width = pool.shape[1]
        _pack_prefix_pages[(requests.numel(), triton.cdiv(width, 8192))](
            pool,
            table,
            send,
            requests,
            blocks,
            offsets,
            width,
            pool.stride(0),
            table.stride(0),
            table.stride(1),
            send.stride(0),
            pool.shape[0],
            table.shape[0],
            table.shape[1],
            send.shape[0],
            8192,
            num_warps=4,
            debug=True,
        )


def restore_prefix_pages(
    received: torch.Tensor,
    working: torch.Tensor,
    destinations: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    if destinations.numel():
        width = working.shape[1]
        _restore_prefix_pages[(destinations.numel(), triton.cdiv(width, 8192))](
            received,
            working,
            destinations,
            offsets,
            width,
            received.stride(0),
            working.stride(0),
            received.shape[0],
            working.shape[0],
            8192,
            num_warps=4,
            debug=True,
        )
