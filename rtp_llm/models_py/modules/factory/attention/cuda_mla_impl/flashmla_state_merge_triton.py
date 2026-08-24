# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vLLM's vllm/v1/attention/ops/triton_merge_attn_states.py.

"""In-place merge for natural-log FlashMLA attention states.

The accumulator may be FP32 while each FlashMLA partial output remains BF16 or
FP16.  Keeping a long merge chain in FP32 prevents every intermediate state
from being rounded back to BF16; callers cast only the final state.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _merge_attention_states_in_place_kernel(
    output_ptr,
    output_lse_ptr,
    other_output_ptr,
    other_lse_ptr,
    output_stride_token,
    output_stride_head,
    other_output_stride_token,
    other_output_stride_head,
    output_lse_stride_token,
    output_lse_stride_head,
    other_lse_stride_token,
    other_lse_stride_head,
    num_heads,
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    head_offsets = tl.program_id(1) * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
    head_offsets_i64 = head_offsets.to(tl.int64)
    head_mask = head_offsets < num_heads
    value_offsets = tl.arange(0, PADDED_HEAD_SIZE)
    value_mask = value_offsets < HEAD_SIZE

    output_lse_offsets = (
        token_idx * output_lse_stride_token + head_offsets_i64 * output_lse_stride_head
    )
    other_lse_offsets = (
        token_idx * other_lse_stride_token + head_offsets_i64 * other_lse_stride_head
    )
    output_lse = tl.load(
        output_lse_ptr + output_lse_offsets,
        mask=head_mask,
        other=float("-inf"),
    ).to(tl.float32)
    other_lse = tl.load(
        other_lse_ptr + other_lse_offsets,
        mask=head_mask,
        other=float("-inf"),
    ).to(tl.float32)

    # ``output_lse`` is both an input and an in-place output.  The 2-D output
    # tile and the 1-D LSE tile can be assigned to different warps.  Without a
    # CTA barrier, the warp responsible for the LSE store may overwrite the old
    # value before another warp has loaded it for ``output_scale``.  This is a
    # real read-after-write race for 8-warp layouts; it only appeared safe for
    # some 4-warp layouts because their layout conversion happened to emit a
    # barrier.  Synchronize the reads explicitly instead of relying on a
    # compiler-selected layout.
    tl.debug_barrier()

    max_lse = tl.maximum(output_lse, other_lse)
    has_values = max_lse != float("-inf")
    output_exp = tl.exp(output_lse - max_lse)
    other_exp = tl.exp(other_lse - max_lse)
    denominator = output_exp + other_exp
    output_scale = tl.where(has_values, output_exp / denominator, 0.0)
    other_scale = tl.where(has_values, other_exp / denominator, 0.0)

    output_offsets = (
        token_idx * output_stride_token
        + head_offsets_i64[:, None] * output_stride_head
        + value_offsets[None, :]
    )
    other_output_offsets = (
        token_idx * other_output_stride_token
        + head_offsets_i64[:, None] * other_output_stride_head
        + value_offsets[None, :]
    )
    output = tl.load(
        output_ptr + output_offsets,
        mask=head_mask[:, None] & value_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    other_output = tl.load(
        other_output_ptr + other_output_offsets,
        mask=head_mask[:, None] & value_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    merged_output = output * output_scale[:, None] + other_output * other_scale[:, None]
    merged_output = tl.where(has_values[:, None], merged_output, 0.0)
    tl.store(
        output_ptr + output_offsets,
        merged_output,
        mask=head_mask[:, None] & value_mask[None, :],
    )

    merged_lse = tl.log(denominator) + max_lse
    merged_lse = tl.where(has_values, merged_lse, float("-inf"))
    tl.store(
        output_lse_ptr + output_lse_offsets,
        merged_lse,
        mask=head_mask,
    )


def _has_non_overlapping_layout(tensor: torch.Tensor) -> bool:
    """Return whether every logical element maps to a distinct storage offset."""

    if tensor.numel() == 0:
        return True
    required_span = 1
    for size, stride in sorted(
        (
            (size, abs(stride))
            for size, stride in zip(tensor.shape, tensor.stride())
            if size > 1
        ),
        key=lambda item: item[1],
    ):
        if stride < required_span:
            return False
        # Include the actual storage gap introduced by this dimension.  A
        # compact-size product is insufficient for padded/equal strides: for
        # example, (2, 2, 128) with strides (256, 256, 1) aliases the two
        # logical dimensions at offset 256.
        required_span += (size - 1) * stride
    return True


def _storage_byte_interval(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return a conservative half-open byte interval touched by ``tensor``."""

    if tensor.numel() == 0:
        return None
    min_offset = tensor.storage_offset()
    max_offset = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        extent = (size - 1) * stride
        min_offset += min(0, extent)
        max_offset += max(0, extent)
    storage_base = tensor.untyped_storage().data_ptr()
    element_size = tensor.element_size()
    return (
        storage_base + min_offset * element_size,
        storage_base + (max_offset + 1) * element_size,
    )


def _storage_overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    left_interval = _storage_byte_interval(left)
    right_interval = _storage_byte_interval(right)
    if left_interval is None or right_interval is None:
        return False
    return max(left_interval[0], right_interval[0]) < min(
        left_interval[1], right_interval[1]
    )


def is_merge_attention_states_in_place_supported(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    other_output: torch.Tensor,
    other_lse: torch.Tensor,
) -> bool:
    """Return whether tensors satisfy the FlashMLA state-merge contract."""

    if not all(
        isinstance(tensor, torch.Tensor)
        for tensor in (output, output_lse, other_output, other_lse)
    ):
        return False
    if not all(
        tensor.is_cuda for tensor in (output, output_lse, other_output, other_lse)
    ):
        return False
    if not (
        output.device == output_lse.device == other_output.device == other_lse.device
    ):
        return False
    if output.shape != other_output.shape or output.dim() != 3:
        return False
    if output_lse.shape != output.shape[:2] or other_lse.shape != output.shape[:2]:
        return False
    supported_output_dtypes = (torch.float16, torch.bfloat16, torch.float32)
    if output.dtype not in supported_output_dtypes:
        return False
    if other_output.dtype not in supported_output_dtypes:
        return False
    if output.dtype != torch.float32 and output.dtype != other_output.dtype:
        return False
    if output_lse.dtype != torch.float32 or other_lse.dtype != torch.float32:
        return False
    if output.stride(2) != 1 or other_output.stride(2) != 1:
        return False
    tensors = (output, output_lse, other_output, other_lse)
    if not all(_has_non_overlapping_layout(tensor) for tensor in tensors):
        return False
    dangerous_pairs = (
        (output, output_lse),
        (output, other_output),
        (output, other_lse),
        (output_lse, other_output),
        (output_lse, other_lse),
    )
    if any(_storage_overlaps(left, right) for left, right in dangerous_pairs):
        return False
    if output.shape[2] <= 0 or output.shape[2] > 256:
        return False
    return True


def merge_attention_states_in_place(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    other_output: torch.Tensor,
    other_lse: torch.Tensor,
) -> None:
    """Merge a natural-log attention state into ``(output, output_lse)``.

    Outputs have shape ``[tokens, heads, head_size]``. LSE tensors have logical
    shape ``[tokens, heads]`` and may use FlashMLA's strided transpose layout.
    The first state is updated in place.
    """

    if not is_merge_attention_states_in_place_supported(
        output, output_lse, other_output, other_lse
    ):
        raise ValueError(
            "unsupported FlashMLA attention-state merge tensors: "
            f"output(shape={tuple(output.shape)}, dtype={output.dtype}, "
            f"device={output.device}, stride={output.stride()}), "
            f"output_lse(shape={tuple(output_lse.shape)}, "
            f"dtype={output_lse.dtype}, stride={output_lse.stride()}), "
            f"other_output(shape={tuple(other_output.shape)}, "
            f"dtype={other_output.dtype}, stride={other_output.stride()}), "
            f"other_lse(shape={tuple(other_lse.shape)}, "
            f"dtype={other_lse.dtype}, stride={other_lse.stride()})"
        )
    if output.shape[0] == 0 or output.shape[1] == 0:
        return

    block_heads = 8
    padded_head_size = triton.next_power_of_2(output.shape[2])
    grid = (output.shape[0], triton.cdiv(output.shape[1], block_heads))
    _merge_attention_states_in_place_kernel[grid](
        output,
        output_lse,
        other_output,
        other_lse,
        output.stride(0),
        output.stride(1),
        other_output.stride(0),
        other_output.stride(1),
        output_lse.stride(0),
        output_lse.stride(1),
        other_lse.stride(0),
        other_lse.stride(1),
        output.shape[1],
        HEAD_SIZE=output.shape[2],
        PADDED_HEAD_SIZE=padded_head_size,
        BLOCK_HEADS=block_heads,
        num_warps=8,
    )


__all__ = [
    "is_merge_attention_states_in_place_supported",
    "merge_attention_states_in_place",
]
