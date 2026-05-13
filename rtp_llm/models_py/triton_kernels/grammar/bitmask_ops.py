# Adapted from
# https://github.com/mlc-ai/xgrammar/blob/v0.1.17/python/xgrammar/kernels/apply_token_bitmask_inplace_triton.py
# via sglang's constrained/triton_ops/bitmask_ops.py

import torch
import triton
import triton.language as tl


def _get_device_sm_count(device_id: int = 0) -> int:
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count
    return 0


@triton.jit
def _apply_token_bitmask_inplace_kernel(
    logits_ptr,
    bitmask_ptr,
    num_rows,
    vocab_size,
    logits_row_stride,
    bitmask_row_stride,
    bitmask_width,
    NUM_SMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply a bitmask to logits in-place using Triton.

    The bitmask is a 01 bitwise compressed tensor where 0 means the token is
    masked and 1 means the token is not masked.  Masked logits are set to -inf.
    """
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(vocab_size, BLOCK_SIZE)
    for work_id in tl.range(pid, num_rows * num_blocks, NUM_SMS):
        row_id = work_id // num_blocks
        block_offset = (work_id % num_blocks) * BLOCK_SIZE
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        bitmask_offsets = block_offset // 32 + tl.arange(0, BLOCK_SIZE // 32)
        vocab_mask = offsets < vocab_size
        packed_bitmask_mask = bitmask_offsets < bitmask_width
        packed_bitmask = tl.load(
            bitmask_ptr + row_id * bitmask_row_stride + bitmask_offsets,
            packed_bitmask_mask,
        )
        bitmask = ((packed_bitmask[:, None] >> (tl.arange(0, 32)[None, :])) & 1) == 0
        bitmask = bitmask.reshape(BLOCK_SIZE)

        tl.store(
            logits_ptr + row_id * logits_row_stride + offsets,
            -float("inf"),
            vocab_mask & bitmask,
        )


def apply_token_bitmask_inplace_triton(
    logits: torch.Tensor,
    bitmask: torch.Tensor,
) -> None:
    """Apply a packed int32 bitmask to logits, setting masked positions to -inf.

    Args:
        logits: Float tensor of shape ``[batch, vocab]`` (or ``[vocab]``).
        bitmask: Int32 tensor produced by ``xgrammar.allocate_token_bitmask``.
    """
    NUM_SMS = _get_device_sm_count()
    BLOCK_SIZE = 4096
    BITS_PER_BLOCK = 32

    assert bitmask.dtype == torch.int32, "bitmask must be of type int32"

    logits_shape = logits.shape
    bitmask_shape = bitmask.shape
    if logits.ndim == 1:
        logits_shape = (1, logits_shape[0])
    if bitmask.ndim == 1:
        bitmask_shape = (1, bitmask_shape[0])

    required_bitmask_width = (logits_shape[1] + BITS_PER_BLOCK - 1) // BITS_PER_BLOCK
    assert required_bitmask_width >= bitmask_shape[1], (
        f"Bitmask width too large: allow at most {required_bitmask_width} int32s for "
        f"logits' width {logits_shape[1]}, but got {bitmask_shape[1]}"
    )

    vocab_size = min(logits_shape[1], bitmask_shape[1] * BITS_PER_BLOCK)

    assert (
        logits_shape[0] == bitmask_shape[0]
    ), f"batch size mismatch: logits {logits_shape[0]} vs bitmask {bitmask_shape[0]}"
    num_rows = logits_shape[0]

    if NUM_SMS > 0:
        grid = (NUM_SMS,)
    else:
        num_blocks = triton.cdiv(vocab_size, BLOCK_SIZE)
        grid = (num_rows * num_blocks,)
        NUM_SMS = triton.next_power_of_2(grid[0])

    # Row strides come from the actual tensor strides — callers may pass a
    # sliced view (e.g. logits[:, :vocab]) whose row stride is the underlying
    # padded width, not shape[1]. Bitmask bound check still uses shape[1] (the
    # number of int32 columns), which is independent of the row stride.
    logits_row_stride = logits.stride(0) if logits.ndim >= 2 else 0
    bitmask_row_stride = bitmask.stride(0) if bitmask.ndim >= 2 else 0
    bitmask_width = bitmask_shape[1]

    _apply_token_bitmask_inplace_kernel[grid](
        logits,
        bitmask,
        num_rows,
        vocab_size,
        logits_row_stride,
        bitmask_row_stride,
        bitmask_width,
        NUM_SMS,
        BLOCK_SIZE,
        num_warps=BLOCK_SIZE // 32 // (16 // logits.element_size()),
        num_stages=3,
    )
