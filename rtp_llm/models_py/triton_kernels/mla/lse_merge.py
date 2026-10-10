"""Shared natural-log attention merging and empty-shard identity kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_states(O, L, P, PL, HEADS: tl.constexpr, DIM: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    head = col // DIM
    mask = head < HEADS
    old_lse = tl.load(L + row * HEADS + head, mask, other=-float("inf"))
    new_lse = tl.load(PL + row * HEADS + head, mask, other=-float("inf"))
    largest = tl.maximum(old_lse, new_lse)
    valid = largest != -float("inf")
    old_weight = tl.where(valid, tl.exp(old_lse - largest), 0.0)
    new_weight = tl.where(valid, tl.exp(new_lse - largest), 0.0)
    denominator = old_weight + new_weight
    old = tl.load(O + row * HEADS * DIM + col, mask, other=0).to(tl.float32)
    new = tl.load(P + row * HEADS * DIM + col, mask, other=0).to(tl.float32)
    merged = tl.where(
        denominator > 0,
        (old * old_weight + new * new_weight) / denominator,
        0.0,
    )
    tl.store(O + row * HEADS * DIM + col, merged, mask)
    tl.store(
        L + row * HEADS + head,
        tl.where(valid, largest + tl.log(denominator), -float("inf")),
        mask & (col % DIM == 0),
    )


def merge_attention_states_in_place(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    partial: torch.Tensor,
    partial_lse: torch.Tensor,
) -> None:
    """Merge a BF16 partial into a BF16 or FP32 attention state in place.

    Outputs have shape [tokens, heads, dim]; LSE tensors are FP32
    [tokens, heads] in natural-log units. All tensors must be contiguous
    on one CUDA device. Empty states use -inf LSE and contribute zero weight.
    """
    if (output.ndim != 3 or partial.shape != output.shape
            or output.dtype not in (torch.bfloat16, torch.float32)
            or partial.dtype != torch.bfloat16
            or output_lse.shape != output.shape[:2]
            or partial_lse.shape != output_lse.shape
            or output_lse.dtype != torch.float32
            or partial_lse.dtype != torch.float32):
        raise ValueError("Attention state shape or dtype mismatch")
    tensors = (output, output_lse, partial, partial_lse)
    if not output.is_cuda or any(t.device != output.device or not t.is_contiguous()
                                  for t in tensors):
        raise ValueError("Attention states must be contiguous on one CUDA device")
    if output.numel():
        _merge_states[(output.shape[0],)](
            output, output_lse, partial, partial_lse,
            output.shape[1], output.shape[2],
            triton.next_power_of_2(output.shape[1] * output.shape[2]),
        )


@triton.jit
def _set_empty_partial_identity(
    output,
    lse,
    lengths,
    elements: tl.constexpr,
    heads: tl.constexpr,
    block_size: tl.constexpr,
    lse_block_size: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(lengths + row) == 0:
        for start in tl.range(0, elements, block_size):
            offsets = start + tl.arange(0, block_size)
            tl.store(output + row * elements + offsets, 0.0, offsets < elements)
        offsets = tl.arange(0, lse_block_size)
        tl.store(lse + row * heads + offsets, -float("inf"), offsets < heads)
