"""Deterministic helpers for route-local MoE stage-2 output on ROCm."""

import torch
import triton
import triton.language as tl


@triton.jit
def _make_route_local_ids_kernel(
    sorted_ids_ptr,
    route_ids_ptr,
    count,
    token_num,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < count
    fused_ids = tl.load(sorted_ids_ptr + offsets, mask=mask, other=0)

    # AITER encodes the source token in the low 24 bits and the TopK slot in
    # the high 8 bits.  Flatten the pair into a unique stage-2 output row.
    token_ids = fused_ids & 0xFFFFFF
    slot_ids = fused_ids >> 24
    route_num = token_num * TOPK
    route_ids = tl.where(token_ids < token_num, token_ids * TOPK + slot_ids, route_num)
    tl.store(route_ids_ptr + offsets, route_ids, mask=mask)


@triton.jit
def _fixed_order_fp32_route_reduce_kernel(
    route_output_ptr,
    output_ptr,
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_id = tl.program_id(0)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < hidden_size
    base_offset = token_id * TOPK * hidden_size + hidden_offsets

    # A single program owns each output element.  The static loop gives every
    # run the same slot order and keeps all intermediate sums in FP32.
    accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for slot in tl.static_range(0, TOPK):
        route_value = tl.load(
            route_output_ptr + base_offset + slot * hidden_size,
            mask=hidden_mask,
            other=0.0,
        )
        accumulator += route_value.to(tl.float32)

    tl.store(
        output_ptr + token_id * hidden_size + hidden_offsets,
        accumulator,
        mask=hidden_mask,
    )


def make_route_local_ids(
    sorted_token_ids: torch.Tensor, token_num: int, topk: int
) -> torch.Tensor:
    """Map AITER's packed ``(token, slot)`` IDs to unique flat-route rows."""

    assert sorted_token_ids.dtype == torch.int32
    assert sorted_token_ids.is_contiguous()
    assert token_num > 0
    assert 0 < topk <= 255

    route_ids = torch.empty_like(sorted_token_ids)
    block_size = 256
    _make_route_local_ids_kernel[(triton.cdiv(sorted_token_ids.numel(), block_size),)](
        sorted_token_ids,
        route_ids,
        sorted_token_ids.numel(),
        token_num,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return route_ids


def fixed_order_fp32_route_reduce(
    route_output: torch.Tensor, output: torch.Tensor, topk: int
) -> None:
    """Reduce ``[token, slot, hidden]`` BF16 routes into BF16 token output."""

    assert route_output.dtype == output.dtype
    assert route_output.is_contiguous()
    assert output.is_contiguous()
    assert output.dim() == 2
    assert route_output.shape == (output.shape[0] * topk, output.shape[1])

    token_num, hidden_size = output.shape
    block_size = 256
    _fixed_order_fp32_route_reduce_kernel[
        (token_num, triton.cdiv(hidden_size, block_size))
    ](
        route_output,
        output,
        hidden_size,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
