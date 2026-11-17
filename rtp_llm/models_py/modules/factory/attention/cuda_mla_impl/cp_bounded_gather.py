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


def gather_projection_chunks(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    cp_size: int,
    restore_indices: Optional[torch.Tensor],
    gather_into: Callable,
) -> torch.Tensor:
    """Gather packed projection chunks directly into their final consumer layout.

    All ranks have the same padded local row count and dtype. Complete each
    chunk's consume on the collective stream before allocating/submitting the
    next chunk, so allocator reuse cannot race a consumer. Graph capture uses
    the caller's existing path. The final output and existing restore map are
    retained inputs/outputs, not additional chunk-local scratch.
    """
    if compressed_kv.shape[0] != k_pe.shape[0]:
        raise ValueError("CP projection components must have identical row counts")
    rows = compressed_kv.shape[0]
    width = compressed_kv.shape[1] + k_pe.shape[1]
    row_bytes = width * compressed_kv.element_size()
    chunk_rows = max(1, _PROJECTION_SCRATCH_BYTES // ((cp_size + 1) * row_bytes))
    output_rows = rows * cp_size if restore_indices is None else restore_indices.numel()
    output = torch.empty(
        (output_rows, width), dtype=compressed_kv.dtype, device=compressed_kv.device
    )
    for start in range(0, rows, chunk_rows):
        count = min(chunk_rows, rows - start)
        packed = torch.cat(
            (compressed_kv.narrow(0, start, count), k_pe.narrow(0, start, count)), dim=-1
        )
        gathered = torch.empty(
            (cp_size * count, width), dtype=packed.dtype, device=packed.device
        )
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
        # Drop previous receive/send tensors before the next collective creates
        # its buffers; retaining both chunks would double the scratch bound.
        del gathered, packed
    return output
