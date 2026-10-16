"""Deterministic helpers for route-local MoE stage-2 output on ROCm."""

from typing import Optional

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
    # int32 >> is an arithmetic shift, so slot >= 128 would sign-extend into a
    # negative route row.  Bitcast to uint32 for a logical shift.
    slot_ids = (fused_ids.to(tl.uint32, bitcast=True) >> 24).to(tl.int32)
    route_num = token_num * TOPK
    # Invalid/padded entries map to the one-past sentinel row.  This mirrors
    # AITER's own MOE_SORTING_MOCK_ID(tokens, topk) convention: the CK stage2
    # epilogue suppresses writes past the route_output element space
    # (is_dst_valid), so the out-of-range sentinel can never write.  Do NOT
    # "fix" this to an in-bounds row: padding entries would then atomically
    # add garbage into a real route row.
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

    if sorted_token_ids.dtype != torch.int32 or not sorted_token_ids.is_contiguous():
        raise ValueError(
            "sorted_token_ids must be a contiguous int32 tensor, got "
            f"dtype={sorted_token_ids.dtype} contiguous={sorted_token_ids.is_contiguous()}"
        )
    if token_num <= 0:
        raise ValueError(f"token_num must be positive, got {token_num}")
    if not 0 < topk <= 255:
        raise ValueError(f"topk must be in (0, 255], got {topk}")

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
    route_output: torch.Tensor,
    output: torch.Tensor,
    topk: int,
    route_rows: Optional[int] = None,
) -> None:
    """Reduce ``[token, slot, hidden]`` BF16 routes into BF16 token output.

    ``route_rows`` consumes exactly that many leading route rows and ignores
    any extra trailing rows; the deterministic MoE caller uses it to pass the
    buffer that still carries its trailing sentinel scratch row.
    """
    if route_rows is not None:
        if route_rows != output.shape[0] * topk:
            raise ValueError(
                f"route_rows {route_rows} does not match token_num*topk "
                f"{output.shape[0] * topk}"
            )
        fixed_order_fp32_route_reduce(route_output[:route_rows], output, topk)
        return

    if route_output.dtype != output.dtype:
        raise ValueError(
            f"route_output dtype {route_output.dtype} does not match output dtype {output.dtype}"
        )
    if not route_output.is_contiguous() or not output.is_contiguous():
        raise ValueError("route_output and output must be contiguous")
    if output.dim() != 2:
        raise ValueError(f"output must be 2-D [token, hidden], got dim={output.dim()}")
    if route_output.shape != (output.shape[0] * topk, output.shape[1]):
        # The reduce kernel indexes route_output rows without a row-count mask;
        # a shape mismatch would read past the allocation.
        raise ValueError(
            f"route_output shape {tuple(route_output.shape)} does not match "
            f"[token*topk, hidden] = ({output.shape[0] * topk}, {output.shape[1]})"
        )

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
