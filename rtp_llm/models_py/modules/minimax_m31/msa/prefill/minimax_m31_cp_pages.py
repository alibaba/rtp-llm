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
    CHUNK: tl.constexpr,
):
    page = tl.program_id(0)
    byte = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    request = tl.load(REQUESTS + page)
    block = tl.load(BLOCKS + page)
    source = tl.load(TABLE + request * TABLE_ROW_STRIDE + block * TABLE_COL_STRIDE).to(
        tl.int64
    )
    destination = tl.load(OFFSETS + page)
    value = tl.load(POOL + source * POOL_STRIDE + byte, mask=byte < WIDTH, other=0)
    tl.store(SEND + destination * SEND_STRIDE + byte, value, mask=byte < WIDTH)


@triton.jit
def _restore_prefix_pages(
    RECEIVED,
    WORKING,
    DESTINATIONS,
    OFFSETS,
    WIDTH: tl.constexpr,
    RECV_STRIDE: tl.constexpr,
    WORK_STRIDE: tl.constexpr,
    CHUNK: tl.constexpr,
):
    page = tl.program_id(0)
    byte = tl.program_id(1) * CHUNK + tl.arange(0, CHUNK)
    source = tl.load(OFFSETS + page)
    destination = tl.load(DESTINATIONS + page)
    value = tl.load(RECEIVED + source * RECV_STRIDE + byte, mask=byte < WIDTH, other=0)
    tl.store(WORKING + destination * WORK_STRIDE + byte, value, mask=byte < WIDTH)


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
            8192,
            num_warps=4,
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
            8192,
            num_warps=4,
        )
