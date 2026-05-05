"""DSV4 CompressorFP8 prefill prelude — fused slot_mapping + freq_idx.

Replaces ~10 small CUDA launches (5 ``arange/expand/reshape`` builds in
``_forward_prefill_body`` + ``_logical_to_pool_slots``'s 5 ops) with a
single Triton kernel. Output:

  * ``slots[b*NB + p]``     — pool slot for compressed token ``p`` of
    request ``b`` (``-1`` if logical position is out of pool capacity
    or block_table entry is unallocated).
  * ``freq_idx[b*NB + p]``  — RoPE row index for that token,
    ``max(0, sp + p*ratio + 1 - ratio)``.

Inspired by vLLM's ``compressor_utils.get_compressed_slot_mapping``
(``test/_vllm_ref/compressor_utils.py``); same per-element math, fused
into one launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compressor_prelude_kernel(
    slots_out_ptr,  # [B*NB] int64
    freq_out_ptr,  # [B*NB] int64
    block_table_ptr,  # [B, max_blocks] int (whatever bt.dtype is)
    bt_stride0,
    NB: tl.constexpr,
    write_start: tl.constexpr,
    sp: tl.constexpr,
    ratio: tl.constexpr,
    eb: tl.constexpr,
    pool_capacity: tl.constexpr,  # max_blocks * eb
):
    pid = tl.program_id(0)
    b = pid // NB
    p = pid % NB

    logical = write_start + p
    in_cap = logical < pool_capacity

    # block_id lookup (clamp to 0 when out of capacity to avoid OOB load).
    block_in_seq = tl.where(in_cap, logical // eb, 0)
    in_block = logical % eb
    block_id = tl.load(
        block_table_ptr + b * bt_stride0 + block_in_seq, mask=in_cap, other=0
    ).to(tl.int64)

    valid = in_cap & (block_id > 0)
    slot = tl.where(valid, block_id * eb + in_block, -1)

    # RoPE row: max(0, sp + p*ratio + 1 - ratio).
    freq = sp + p * ratio + 1 - ratio
    freq = tl.where(freq < 0, 0, freq)

    tl.store(slots_out_ptr + pid, slot)
    tl.store(freq_out_ptr + pid, freq.to(tl.int64))


def compressor_prelude_fused(
    B: int,
    NB: int,
    write_start: int,
    sp: int,
    ratio: int,
    eb: int,
    block_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused slot_mapping + freq_idx for the FP8 compressor prefill writer.

    Args:
      B, NB:       batch and number-of-compressed-blocks per request.
      write_start: logical compressed position to start writing at
                   (= ``sp_int // ratio``).
      sp:          start_pos in tokens (for RoPE row math).
      ratio:       compress ratio (4 for CSA, 128 for HCA).
      eb:          tokens per pool block.
      block_table: ``[B, max_blocks]`` int — physical block ids per request.

    Returns: (slots, freq_idx), both ``[B*NB]`` int64.
      ``slots[i] = -1`` for out-of-capacity / unallocated positions.
    """
    device = block_table.device
    N = B * NB
    slots = torch.empty(N, dtype=torch.int64, device=device)
    freq_idx = torch.empty(N, dtype=torch.int64, device=device)
    if N == 0:
        return slots, freq_idx

    max_blocks = int(block_table.shape[1])
    pool_capacity = max_blocks * eb

    _compressor_prelude_kernel[(N,)](
        slots,
        freq_idx,
        block_table,
        bt_stride0=int(block_table.stride(0)),
        NB=NB,
        write_start=write_start,
        sp=sp,
        ratio=ratio,
        eb=eb,
        pool_capacity=pool_capacity,
    )
    return slots, freq_idx
