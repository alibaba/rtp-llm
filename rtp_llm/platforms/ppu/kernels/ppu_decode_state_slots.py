"""Fused integer mapping into cyclic PPU Decode compressor state pools."""

import torch
import triton
import triton.language as tl


@triton.jit
def _state_slots_kernel(
    Positions,
    Requests,
    Table,
    Output,
    N: tl.constexpr,
    RowStride: tl.constexpr,
    ColumnStride: tl.constexpr,
    Columns: tl.constexpr,
    Entries: tl.constexpr,
    Tokens: tl.constexpr,
    Block: tl.constexpr,
):
    i = tl.program_id(0) * Block + tl.arange(0, Block)
    valid = i < N
    pos = tl.load(Positions + i, valid, 0).to(tl.int64)
    req = tl.load(Requests + i, valid, 0).to(tl.int64)
    column = (pos // Tokens) % Columns
    block_id = tl.load(Table + req * RowStride + column * ColumnStride, valid, 0).to(
        tl.int64
    )
    slot = block_id * Entries + pos % Entries
    tl.store(Output + i, tl.where(block_id > 0, slot, -1), valid)


def update_compressor_state_slots(meta, batch, entries_by_tag):
    """Update preallocated state slots after Decode positions/table preparation.

    Positions must be the nonnegative normalized metadata positions and request
    IDs must index the active batch. The metadata allocator owns these vectors;
    only their device contents change during Graph replay. Zero/negative block
    IDs denote unallocated state and produce -1. Slot arithmetic uses int64,
    while paged block tables may have arbitrary two-dimensional strides.
    """
    n = batch * meta.q_len_per_req
    positions, requests = meta.position_ids_long, meta.req_id_per_token_long
    if not n or positions is None or requests is None:
        return
    for tensor in (positions, requests):
        if (
            tensor.ndim != 1
            or tensor.numel() < n
            or tensor.dtype != torch.int64
            or not tensor.is_cuda
            or tensor.device != positions.device
            or not tensor.is_contiguous()
        ):
            raise ValueError("State mapping requires contiguous int64 device vectors")
    for tag, output in meta.compressor_state_slot_mappings.items():
        if tag not in meta.pool_block_tables:
            continue
        table = meta.pool_block_tables[tag]
        entries = entries_by_tag[tag]
        tokens = meta.paged_pool_tokens_per_block[tag]
        if (
            table.ndim != 2
            or table.shape[0] < batch
            or table.shape[1] == 0
            or table.dtype not in (torch.int32, torch.int64)
            or table.device != positions.device
            or output.ndim != 1
            or output.numel() < n
            or output.dtype != torch.int64
            or output.device != positions.device
            or not output.is_contiguous()
            or entries <= 0
            or tokens <= 0
        ):
            raise ValueError("Invalid state-pool table, output or block geometry")
        _state_slots_kernel[(triton.cdiv(n, 128),)](
            positions,
            requests,
            table,
            output,
            n,
            *table.stride(),
            table.shape[1],
            entries,
            tokens,
            128,
        )
