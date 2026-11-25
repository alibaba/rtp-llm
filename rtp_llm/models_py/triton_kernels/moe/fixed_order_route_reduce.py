"""Reduce expert routes in a fixed FP32 order for batch-invariant inference."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fixed_order_fp32_route_reduce_kernel(
    route_output_ptr,
    output_ptr,
    hidden_size,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Element offsets reach token_id*TOPK*hidden_size, which overflows int32
    # once token_num*topk*hidden exceeds 2^31 (token_num >= 131072 at the
    # production topk=8/hidden=2048 shape). Carry the row offsets in int64 so
    # every legal shape addresses rows instead of wrapping into negative
    # (out-of-bounds) reads.
    token_id = tl.program_id(0).to(tl.int64)
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



def fixed_order_fp32_route_reduce(
    route_output: torch.Tensor, output: torch.Tensor, topk: int
) -> None:
    """Sum route rows in top-k order and round once into the token output."""
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
