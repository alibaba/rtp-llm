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
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0).to(tl.int64)
    if INVERSE:
        source = index // CHUNK_ROWS * LOCAL_ROWS + START + index % CHUNK_ROWS
        row = tl.load(Restore + source)
    elif REMAP:
        row = index
        source = tl.load(Restore + row)
    else:
        row = index // CHUNK_ROWS * LOCAL_ROWS + START + index % CHUNK_ROWS
        source = row
    owner = source // LOCAL_ROWS
    local = source % LOCAL_ROWS
    # Only this chunk's rows read/write the large KV payload. All chunks use
    # the same request-specific map, including when padding was removed.
    if (row >= 0) & (source >= 0) & (local >= START) & (local < START + CHUNK_ROWS):
        cols = tl.arange(0, BLOCK)
        values = tl.load(
            Gathered + (owner * CHUNK_ROWS + local - START) * WIDTH + cols,
            cols < WIDTH,
            other=0,
        )
        tl.store(Output + row * WIDTH + cols, values, cols < WIDTH)


def prepare_restore_inverse(
    restore_cpu: torch.Tensor,
    source_rows: int,
    device: torch.device | str,
    current: Optional[torch.Tensor] = None,
    forbid_realloc: bool = False,
) -> Optional[torch.Tensor]:
    """Build a fresh unique-subset inverse from the CP planner's CPU snapshot.

    Duplicate maps retain the generic destination-gather fallback. The int64
    inverse costs 8 * source_rows bytes, separate from communication scratch.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if restore_cpu.is_cuda or restore_cpu.ndim != 1 or restore_cpu.dtype != torch.int64:
        raise ValueError("CP restore planning requires a CPU int64 vector")
    if source_rows < 0 or (restore_cpu.numel() and (
        int(restore_cpu.min()) < 0 or int(restore_cpu.max()) >= source_rows
    )):
        raise ValueError("CP restore source index is out of range")
    counts = torch.bincount(restore_cpu, minlength=source_rows)
    if counts.numel() and int(counts.max()) > 1:
        if forbid_realloc and current is not None:
            raise RuntimeError("captured CP inverse requires a unique restore subset")
        return None
    if current is not None and (
        current.shape != (source_rows,) or current.dtype != torch.int64
        or current.device != device
        or not current.is_contiguous()
    ):
        if forbid_realloc:
            raise RuntimeError("captured CP inverse geometry changed")
        current = None
    if forbid_realloc and current is None:
        raise RuntimeError("captured CP inverse storage is missing")
    inverse = torch.full(
        (source_rows,), -1, dtype=torch.int64,
        pin_memory=device.type == "cuda",
    )
    inverse[restore_cpu] = torch.arange(restore_cpu.numel(), dtype=torch.int64)
    if current is None:
        return inverse.to(device=device, non_blocking=True)
    current.copy_(inverse, non_blocking=True)
    return current


def _gather_groups(groups, cp_size, restore_indices, gather_into, source_to_dest=None):
    """Use one fixed send/receive backing for every chunk and dtype group."""
    first = groups[0][0]
    rows = first.shape[0]
    if cp_size < 1 or any(
        value.ndim != 2 or value.shape[0] != rows or value.device != first.device
        for group in groups for value in group
    ):
        raise ValueError("CP projection components require matching rows/device and positive CP size")
    if source_to_dest is not None and (
        restore_indices is None or source_to_dest.shape != (rows * cp_size,)
        or source_to_dest.dtype != torch.int64 or source_to_dest.device != first.device
        or not source_to_dest.is_contiguous()
    ):
        raise ValueError("CP inverse must match the padded source geometry")
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
                copy_rows = output_rows if restore_indices is not None and source_to_dest is None else count * cp_size
                _restore_chunk[(copy_rows,)](
                    gathered,
                    source_to_dest if source_to_dest is not None else (restore_indices if restore_indices is not None else output),
                    output,
                    rows,
                    count,
                    start,
                    width,
                    restore_indices is not None,
                    source_to_dest is not None,
                    tr.next_power_of_2(width),
                )
    return outputs


def gather_projection_chunks(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cp_size: int,
    restore_indices: Optional[torch.Tensor],
    gather_into: Callable,
    source_to_dest: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather equal-dtype components into one packed final consumer tensor."""
    if compressed_kv.dtype != k_pe.dtype:
        raise ValueError("packed CP projection gather requires identical dtypes")
    return _gather_groups(((compressed_kv, k_pe),), cp_size, restore_indices, gather_into, source_to_dest)[0]


def gather_projection_components(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cp_size: int,
    restore_indices: Optional[torch.Tensor],
    gather_into: Callable,
    source_to_dest: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep mixed-dtype components separate; never promote before communication.

    Equal-dtype components share a packed collective. Mixed dtypes execute all
    cKV chunks followed by all RoPE chunks, on every rank, using the same byte
    backing. Output tensors and the immutable restore map are memory floors;
    send plus receive backing is bounded by the fixed scratch byte budget.
    All operations, including scratch reuse, remain on the caller's stream.
    """
    if compressed_kv.dtype == k_pe.dtype:
        packed = gather_projection_chunks(compressed_kv, k_pe, cp_size, restore_indices, gather_into, source_to_dest)
        return packed.split((compressed_kv.shape[1], k_pe.shape[1]), dim=-1)
    return _gather_groups(((compressed_kv,), (k_pe,)), cp_size, restore_indices, gather_into, source_to_dest)
