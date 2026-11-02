"""Shared natural-log attention merging and empty-shard identity kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_states(
    O,
    L,
    P,
    PL,
    P_ROW: tl.constexpr,
    P_HEAD: tl.constexpr,
    P_COL: tl.constexpr,
    PL_ROW: tl.constexpr,
    PL_HEAD: tl.constexpr,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
    RAW_PARTIAL_LSE: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    mask = col < DIM
    old_lse = tl.load(L + row * HEADS + head)
    new_lse = tl.load(PL + row * PL_ROW + head * PL_HEAD)
    if RAW_PARTIAL_LSE:
        # FlashMLA's documented empty-row sentinel. A NaN is not an empty
        # row: preserve it in the canonical state for numerical checks.
        new_lse = tl.where(new_lse == float("inf"), -float("inf"), new_lse)
    largest = tl.maximum(old_lse, new_lse, propagate_nan=tl.PropagateNan.ALL)
    valid = largest != -float("inf")
    old_weight = tl.where(valid, tl.exp(old_lse - largest), 0.0)
    new_weight = tl.where(valid, tl.exp(new_lse - largest), 0.0)
    denominator = old_weight + new_weight
    offset = (row * HEADS + head) * DIM + col
    old = tl.load(O + offset, mask, other=0).to(tl.float32)
    new = tl.load(P + row * P_ROW + head * P_HEAD + col * P_COL, mask, other=0).to(
        tl.float32
    )
    old = tl.where(old_lse == -float("inf"), 0.0, old)
    new = tl.where(new_lse == -float("inf"), 0.0, new)
    merged = tl.where(
        denominator > 0,
        (old * old_weight + new * new_weight) / denominator,
        0.0,
    )
    merged = tl.where(largest != largest, largest, merged)
    tl.store(O + offset, merged, mask)
    tl.store(
        L + row * HEADS + head,
        tl.where(valid, largest + tl.log(denominator), -float("inf")),
    )


def merge_attention_states_in_place(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    partial: torch.Tensor,
    partial_lse: torch.Tensor,
    *,
    raw_partial_lse: bool = False,
) -> None:
    """Merge a strided BF16 partial into a contiguous canonical state.

    FlashMLA pads the head dimension to 64/128. A slice of its real heads
    retains that row stride; reading it as a contiguous tensor mixes rows.
    LSE values use natural logarithms, including the empty-row -inf sentinel.
    """
    if (
        output.ndim != 3
        or partial.shape != output.shape
        or output.dtype not in (torch.bfloat16, torch.float32)
        or partial.dtype != torch.bfloat16
        or output_lse.shape != output.shape[:2]
        or partial_lse.shape != output_lse.shape
        or output_lse.dtype != torch.float32
        or partial_lse.dtype != torch.float32
    ):
        raise ValueError("MLA attention state shape or dtype mismatch")
    tensors = (output, output_lse, partial, partial_lse)
    if (
        not output.is_cuda
        or any(t.device != output.device for t in tensors)
        or not output.is_contiguous()
        or not output_lse.is_contiguous()
    ):
        raise ValueError("MLA canonical states must be contiguous on one CUDA device")
    if output.numel():
        _merge_states[(output.shape[0], output.shape[1])](
            output,
            output_lse,
            partial,
            partial_lse,
            *partial.stride(),
            *partial_lse.stride(),
            output.shape[1],
            output.shape[2],
            triton.next_power_of_2(output.shape[2]),
            raw_partial_lse,
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
