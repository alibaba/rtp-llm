"""MiniMax CP-local feature addressing, without gathering feature tensors."""

import torch


def map_cp_context_rows(chunk_lengths, shuffle_indices, prefix_lengths, input_lengths):
    """Map existing CP-local rows to request ids and absolute token positions.

    ``shuffle_indices`` comes from ContextParallelProcessorBase and contains
    per-request suffix positions, not global packed indices. Runtime zigzag
    padding may contain nonnegative positions beyond the actual suffix length;
    negative sentinels are also accepted. ``input_lengths`` is the original
    global suffix length, not the padded or rank-local chunk length.
    This consumes the runtime's actual shuffle, so it does not independently
    reconstruct or assume a zigzag partition. All tensors must be colocated.
    Output order is unchanged; invalid rows are (-1, -1).
    """
    tensors = (chunk_lengths, shuffle_indices, prefix_lengths, input_lengths)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("CP context metadata must be one-dimensional")
    if any(t.device != shuffle_indices.device for t in tensors):
        raise ValueError("CP context metadata must be on one device")
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise TypeError("CP context metadata must be int32 or int64")
    batch = chunk_lengths.numel()
    if prefix_lengths.numel() != batch or input_lengths.numel() != batch:
        raise ValueError("CP context request metadata sizes differ")
    rows = shuffle_indices.numel()
    if batch == 0:
        if rows:
            raise ValueError("nonempty CP rows require request metadata")
        empty = torch.empty(0, dtype=torch.int32, device=shuffle_indices.device)
        return empty, empty
    ends = chunk_lengths.to(torch.int64).cumsum(0)
    row_ids = torch.arange(rows, device=shuffle_indices.device, dtype=torch.int64)
    request_ids = torch.searchsorted(ends, row_ids, right=True)
    safe_ids = request_ids.clamp(max=batch - 1)
    offsets = shuffle_indices.to(torch.int64)
    valid = (request_ids < batch) & (offsets >= 0)
    valid &= offsets < input_lengths[safe_ids]
    positions = prefix_lengths[safe_ids].to(torch.int64) + offsets
    return (
        torch.where(valid, request_ids, -1).to(torch.int32),
        torch.where(valid, positions, -1).to(torch.int32),
    )
