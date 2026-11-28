"""Generic SM120 prefill sparse-index table split and safety helpers.

``split_sm120_combined_tables`` is the exact production split extracted from
``attention.py``'s workspace prefill path so tests exercise the same code,
not a copy. ``zero_extra_sentinel_row`` and ``remap_padding_to_first_entry``
guard the SM120 kernels against uninitialized memory under masked tails.
"""

from __future__ import annotations

from typing import Tuple

import torch


_SPLIT_ROW_CHUNK = 8192


def split_sm120_combined_tables(
    combined_indices: torch.Tensor,  # [rows, 1, W] | [rows, W] int32, from combine_*
    combined_lens: torch.Tensor,  # [rows] int32
    *,
    M: int,  # per-request workspace stride
    N: int,  # compressed-region size within the stride
    window_size: int,
    extra_width: int,  # max(cmp_topk width, 1)
    ratio: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Split producer-combined tables into SM120 SWA/extra consumer tables.

    Chunked by row to avoid materializing full ``[rows, W]`` int64
    temporaries on long prefills (e.g. 262k x 8k tokens) before query chunking.
    """
    gather_width = M - N
    combined_2d = combined_indices.squeeze(1)
    rows = int(combined_2d.shape[0])
    width = int(combined_2d.shape[1])
    aligned_extra_width = (extra_width + 63) // 64 * 64
    extra_page_size = 64 if ratio == 4 else 2

    swa_indices = torch.zeros((rows, window_size), dtype=torch.int32, device=device)
    extra_indices = torch.zeros(
        (rows, aligned_extra_width), dtype=torch.int32, device=device
    )
    extra_lens = torch.empty((rows,), dtype=torch.int32, device=device)
    swa_lens = torch.empty((rows,), dtype=torch.int32, device=device)
    if rows == 0:
        return swa_indices, swa_lens, extra_indices, extra_lens, extra_page_size

    extra_cols = torch.arange(extra_width, dtype=torch.int64, device=device).unsqueeze(
        0
    )
    swa_cols = torch.arange(window_size, dtype=torch.int64, device=device).unsqueeze(0)
    combined_lens_i32 = combined_lens.to(dtype=torch.int32)
    extra_slice_cols = min(extra_width, width)

    for start in range(0, rows, _SPLIT_ROW_CHUNK):
        end = min(start + _SPLIT_ROW_CHUNK, rows)
        chunk = combined_2d[start:end]
        chunk_clamped = chunk.clamp_min(0)
        chunk_valid = chunk >= 0
        chunk_local = torch.remainder(chunk_clamped, M)
        chunk_extra_lens = (chunk_valid & (chunk_local < N)).sum(
            dim=1, dtype=torch.int32
        )
        chunk_swa_lens = combined_lens_i32[start:end] - chunk_extra_lens
        extra_lens[start:end] = chunk_extra_lens
        swa_lens[start:end] = chunk_swa_lens

        extra_src = chunk[:, :extra_slice_cols].to(torch.int64)
        extra_clamped = extra_src.clamp_min(0)
        extra_req = torch.div(extra_clamped, M, rounding_mode="floor")
        extra_loc = torch.remainder(extra_clamped, M)
        extra_chunk_out = (extra_req * N + extra_loc).to(torch.int32)
        extra_chunk_out.masked_fill_(
            extra_cols[:, :extra_slice_cols]
            >= chunk_extra_lens.to(torch.int64).unsqueeze(1),
            0,
        )
        extra_chunk_out.masked_fill_(extra_src < 0, 0)
        extra_indices[start:end, :extra_slice_cols] = extra_chunk_out

        if width > 0 and window_size > 0:
            swa_src_cols = chunk_extra_lens.to(torch.int64).unsqueeze(1) + swa_cols
            safe_cols = swa_src_cols.clamp_max(width - 1)
            swa_src = chunk.gather(1, safe_cols).to(torch.int64)
            swa_clamped = swa_src.clamp_min(0)
            swa_req = torch.div(swa_clamped, M, rounding_mode="floor")
            swa_local = torch.remainder(swa_clamped, M) - N
            swa_chunk_out = (swa_req * gather_width + swa_local).to(torch.int32)
            swa_chunk_out.masked_fill_(
                swa_cols >= chunk_swa_lens.to(torch.int64).unsqueeze(1), 0
            )
            swa_chunk_out.masked_fill_(swa_src < 0, 0)
            swa_indices[start:end] = swa_chunk_out

    return swa_indices, swa_lens, extra_indices, extra_lens, extra_page_size


def zero_extra_sentinel_row(workspace: torch.Tensor, N: int) -> None:
    """Zero request 0's first compressed row in the prefill workspace.

    ``split_sm120_combined_tables`` masks invalid extra-table tail columns to
    index 0 (and consumers clamp -1 entries to 0 as well), so the SM120
    sparse kernel's value loader gathers extra row 0 with probability 0 for
    every masked entry — ``0 * V`` must stay finite. The workspace is
    ``torch.empty``; when request 0 owns no compressed rows (e.g. HCA ratio
    128 with fewer than 128 tokens) nothing ever writes ``workspace[0, 0]``,
    so uninitialized garbage there leaks NaN into every masked row's output.
    Zero the single sentinel row before the compressed-K gather; that gather
    overwrites it whenever request 0 does own compressed rows.
    """
    if N > 0:
        workspace[0, 0].zero_()


def remap_padding_to_first_entry(indices: torch.Tensor) -> torch.Tensor:
    """Point canonical ``-1`` padding at the row's first live paged slot.

    The SM120 prefill kernels map ``idx < 0`` to pool slot 0 and still load
    it: only the QK logit is masked (by value and by position) while the
    loaded V joins the PV product with softmax weight 0. Paged pools are
    ``torch.empty`` allocations whose block 0 need never have been written,
    so the consumer's historical ``clamp_min_(0)`` can gather uninitialized
    memory and let ``0 * NaN`` poison the row. ``canonical_topk`` leaves
    every row a dense valid prefix with a ``-1`` tail, so column 0 is a
    known-initialized slot of the same row whenever that row has any tile to
    load, and rows with zero effective length load no tiles at all. Lengths
    are untouched: the kernel still masks the remapped tail positions by
    position, not by value.
    """
    if indices.dim() != 2:
        raise ValueError(
            f"sparse top-k indices must be 2D [rows, width], got {indices.shape}"
        )
    return torch.where(indices < 0, indices[:, :1], indices)
