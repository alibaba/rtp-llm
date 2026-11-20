"""Generic SM120 prefill sparse-index table split and safety helpers.

``split_sm120_combined_tables`` is the exact production split extracted from
``attention.py``'s workspace prefill path so tests exercise the same code,
not a copy. ``zero_extra_sentinel_row`` and ``remap_padding_to_first_entry``
guard the SM120 kernels against uninitialized memory under masked tails.
"""

from __future__ import annotations

from typing import Tuple

import torch


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

    This is the exact production split (extracted verbatim from
    ``attention.py``'s workspace prefill path so tests exercise the same
    code, not a copy). ``M``/``N`` are the workspace-stride geometry:
    slots < N of each request's stride belong to the compressed (extra)
    region, the rest to SWA.

    Returns ``(swa_indices, swa_lens, extra_indices, extra_lens,
    extra_page_size)``.
    """
    gather_width = M - N
    combined_2d = combined_indices.squeeze(1).to(torch.int64)
    valid = combined_2d >= 0
    request_ids = torch.div(combined_2d.clamp_min(0), M, rounding_mode="floor")
    local_slots = torch.remainder(combined_2d.clamp_min(0), M)
    is_extra = valid & (local_slots < N)
    extra_lens = is_extra.sum(dim=1, dtype=torch.int32)
    swa_lens = combined_lens.to(torch.int32) - extra_lens
    extra_cols = torch.arange(extra_width, dtype=torch.int64, device=device).unsqueeze(
        0
    )
    extra_src = combined_2d[:, :extra_width]
    extra_req = request_ids[:, :extra_width]
    extra_local = local_slots[:, :extra_width]
    extra_indices = (extra_req * N + extra_local).to(torch.int32)
    extra_indices.masked_fill_(extra_cols >= extra_lens.to(torch.int64).unsqueeze(1), 0)
    extra_indices.masked_fill_(extra_src < 0, 0)
    aligned_extra_width = (extra_width + 63) // 64 * 64
    if aligned_extra_width != extra_width:
        # P1a: only the pad tail needs zeroing, not the full buffer.
        padded_extra = torch.empty(
            (combined_2d.shape[0], aligned_extra_width),
            dtype=torch.int32,
            device=device,
        )
        padded_extra[:, :extra_width] = extra_indices
        padded_extra[:, extra_width:].zero_()
        extra_indices = padded_extra
    swa_cols = torch.arange(window_size, dtype=torch.int64, device=device).unsqueeze(0)
    swa_src_cols = extra_lens.to(torch.int64).unsqueeze(1) + swa_cols
    safe_cols = swa_src_cols.clamp_max(int(combined_2d.shape[1]) - 1)
    swa_src = combined_2d.gather(1, safe_cols)
    swa_req = torch.div(swa_src.clamp_min(0), M, rounding_mode="floor")
    swa_local = torch.remainder(swa_src.clamp_min(0), M) - N
    swa_indices = (swa_req * gather_width + swa_local).to(torch.int32)
    swa_indices.masked_fill_(swa_cols >= swa_lens.to(torch.int64).unsqueeze(1), 0)
    swa_indices.masked_fill_(swa_src < 0, 0)
    extra_page_size = 64 if ratio == 4 else 2
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
