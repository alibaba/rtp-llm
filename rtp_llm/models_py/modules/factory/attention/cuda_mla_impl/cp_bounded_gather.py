"""Bound CP projection communication scratch while preserving row order."""

from typing import Callable, Optional

import torch
import triton as tr
import triton.language as tl

from rtp_llm.models_py.distributed.collective_torch import Group

# Final consumer KV is a memory floor. This bounds the temporary packed send
# and gathered receive tensors, without changing projection GEMM row tiles.
_PROJECTION_SCRATCH_BYTES = 256 * 1024 * 1024


@tr.jit(do_not_specialize=["LOCAL_ROWS", "CHUNK_ROWS", "START"])
def _restore_chunk(
    Gathered,
    Restore,
    Output,
    LOCAL_ROWS,
    CHUNK_ROWS,
    START,
    WIDTH: tl.constexpr,
    REMAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64)
    if REMAP:
        row = index
        source = tl.load(Restore + row)
    else:
        row = index // CHUNK_ROWS * LOCAL_ROWS + START + index % CHUNK_ROWS
        source = row
    owner = source // LOCAL_ROWS
    local = source % LOCAL_ROWS
    # Only this chunk's rows read/write the large KV payload. All chunks use
    # the same request-specific map, including when padding was removed.
    if (source >= 0) & (local >= START) & (local < START + CHUNK_ROWS):
        cols = tl.arange(0, BLOCK)
        values = tl.load(
            Gathered + (owner * CHUNK_ROWS + local - START) * WIDTH + cols,
            cols < WIDTH,
            other=0,
        )
        tl.store(Output + row * WIDTH + cols, values, cols < WIDTH)


def _gather_groups(groups, cp_size, restore_indices, gather_into):
    """Use one fixed send/receive backing for every chunk and dtype group."""
    first = groups[0][0]
    rows = first.shape[0]
    if cp_size < 1 or any(
        value.ndim != 2 or value.shape[0] != rows or value.device != first.device
        for group in groups for value in group
    ):
        raise ValueError("CP projection components require matching rows/device and positive CP size")
    widths = [sum(value.shape[1] for value in group) for group in groups]
    row_bytes = [width * group[0].element_size() for width, group in zip(widths, groups)]
    largest_row = max(row_bytes)
    if largest_row == 0 or (cp_size + 1) * largest_row > _PROJECTION_SCRATCH_BYTES:
        raise ValueError("one CP projection row exceeds the communication scratch budget")
    chunk_rows = min(rows, _PROJECTION_SCRATCH_BYTES // ((cp_size + 1) * largest_row))
    output_rows = rows * cp_size if restore_indices is None else restore_indices.numel()
    outputs = tuple(
        torch.empty((output_rows, width), dtype=group[0].dtype, device=first.device)
        for width, group in zip(widths, groups)
    )
    if rows == 0:
        if output_rows:
            raise ValueError("empty CP projections cannot have nonempty restore indices")
        return outputs
    # Allocate once outside both loops. This matters during graph capture:
    # scratch cannot grow with the number of captured chunks or dtype groups.
    send_bytes = torch.empty(chunk_rows * largest_row, dtype=torch.uint8, device=first.device)
    receive_bytes = torch.empty(cp_size * send_bytes.numel(), dtype=torch.uint8, device=first.device)
    for group, width, bytes_per_row, output in zip(groups, widths, row_bytes, outputs):
        for start in range(0, rows, chunk_rows):
            count = min(chunk_rows, rows - start)
            packed = send_bytes[: count * bytes_per_row].view(group[0].dtype).view(count, width)
            gathered = receive_bytes[: cp_size * count * bytes_per_row].view(group[0].dtype).view(cp_size * count, width)
            if len(group) == 1:
                packed.copy_(group[0].narrow(0, start, count))
            else:
                torch.cat(tuple(value.narrow(0, start, count) for value in group), dim=-1, out=packed)
            gather_into(gathered, packed, group=Group.TP)
            if output_rows:
                copy_rows = output_rows if restore_indices is not None else count * cp_size
                _restore_chunk[(copy_rows,)](
                    gathered,
                    restore_indices if restore_indices is not None else output,
                    output,
                    rows,
                    count,
                    start,
                    width,
                    restore_indices is not None,
                    tr.next_power_of_2(width),
                )
    return outputs


def gather_projection_chunks(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cp_size: int,
    restore_indices: Optional[torch.Tensor],
    gather_into: Callable,
) -> torch.Tensor:
    """Gather equal-dtype components into one packed final consumer tensor."""
    if compressed_kv.dtype != k_pe.dtype:
        raise ValueError("packed CP projection gather requires identical dtypes")
    return _gather_groups(((compressed_kv, k_pe),), cp_size, restore_indices, gather_into)[0]


def gather_projection_components(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cp_size: int,
    restore_indices: Optional[torch.Tensor],
    gather_into: Callable,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep mixed-dtype components separate; never promote before communication.

    Equal-dtype components share a packed collective. Mixed dtypes execute all
    cKV chunks followed by all RoPE chunks, on every rank, using the same byte
    backing. Output tensors and the immutable restore map are memory floors;
    send plus receive backing is bounded by the fixed scratch byte budget.
    All operations, including scratch reuse, remain on the caller's stream.
    """
    if compressed_kv.dtype == k_pe.dtype:
        packed = gather_projection_chunks(compressed_kv, k_pe, cp_size, restore_indices, gather_into)
        return packed.split((compressed_kv.shape[1], k_pe.shape[1]), dim=-1)
    return _gather_groups(((compressed_kv,), (k_pe,)), cp_size, restore_indices, gather_into)
