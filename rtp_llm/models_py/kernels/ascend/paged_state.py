"""Device-side paged-state helpers shared by the GDN AscendC kernels.

Plan C (single-consumer path): eager decode and aclgraph capture run the same
fla_npu AscendC operators with device-tensor metadata.  These helpers replace
the former host-side metadata computation (``block_map.cpu().tolist()``) with
plain device ops so the whole decode step is capturable without any D2H
synchronisation:

* ``decode_state_indices`` gathers the read/write state pages from the block
  map — semantics identical to ``recurrent._resolve_state_pages``:
      read  page = block_map[b, max(len - 2, 0) // page]
      write page = block_map[b, (len - 1) // page]
  where ``len`` is ``sequence_lengths_plus_1``.
* ``seed_state_segment`` migrates the segment owned by ``seg_view`` (the ssm
  or conv slice of a page) from the read page to the write page when a block
  boundary is crossed.  The sibling segment of the destination page is left
  untouched — the other GDN operator owns it.

The pool views come from ``utils/typed_storage_view``: every page row packs
the ssm segment followed by the conv segment, so both segments are strided
views over one storage.

Perf constraints (measured on Ascend950PR / CANN 9.2.2, see the acl-graph
GDN performance notes):

* never ``index_select`` the 4D strided pool view directly — it degrades to
  element-wise copies (~3.8 ms); the 2D page-row view below keeps copies
  block-level (~0.04 ms);
* never assign into a mid-row slice (``row[:, off:off+len] = x``) — it lowers
  to aclnnInplaceCopy_Slice (element-wise again, ~5.9 ms); rebuild rows with
  ``torch.cat`` instead.
"""

from __future__ import annotations

import torch


def decode_state_indices(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather (read_page, write_page) per sequence row from ``block_map``."""

    length = sequence_lengths_plus_1.reshape(-1).to(torch.int64)
    read_col = (length - 2).clamp_min(0) // seq_size_per_block
    write_col = (length - 1).clamp_min(0) // seq_size_per_block
    read_idx = block_map.gather(1, read_col.view(-1, 1)).squeeze(1)
    write_idx = block_map.gather(1, write_col.view(-1, 1)).squeeze(1)
    return read_idx, write_idx


def paged_row_view(seg_view: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Derive a 2D page-row view ``[pages, page_stride]`` from a segment view.

    Returns ``(row_view, seg_offset, seg_len)``: page ``p``'s segment lives at
    ``row_view[p, seg_offset : seg_offset + seg_len]``.  Host-side view
    construction only — safe inside capture.
    """

    page_stride = seg_view.stride(0)
    off = seg_view.storage_offset()
    seg_offset = off % page_stride
    base = off - seg_offset
    seg_len = 1
    for s in seg_view.shape[1:]:
        seg_len *= s
    row_view = torch.empty(0, dtype=seg_view.dtype, device=seg_view.device)
    row_view.set_(seg_view.untyped_storage(), base,
                  (seg_view.shape[0], page_stride), (page_stride, 1))
    return row_view, seg_offset, seg_len


def seed_state_segment(
    seg_view: torch.Tensor,
    read_idx: torch.Tensor,
    write_idx: torch.Tensor,
) -> None:
    """Migrate ``seg_view``'s segment across a block boundary, in place.

    When ``read_idx != write_idx`` for a row the segment is copied from the
    read page into the write page (the AscendC operators read and update the
    state in place at the write page); equal indices keep the row unchanged.
    Only this segment's columns are read and written — the sibling segment
    (ssm vs conv) is never touched.

    Delegates to ``state_migration.migrate_state_rows`` (ported from PR #52):
    a triton kernel that skips ``src == dst`` rows entirely, so the common
    same-page case costs nothing but the index comparison.  Rows that need no
    migration are turned into self-copies via ``torch.where`` — static shape,
    aclgraph-capture safe (no nonzero / boolean-mask indexing).  Falls back to
    an unconditional ``index_copy_`` on CPU / without triton / with
    ``RTP_LLM_GDN_PRECOPY=0``.

    Previous implementation (full-row F.embedding gather + torch.where +
    cat rebuild + index_copy_, ~0.1 ms per GDN layer) is superseded; a
    segment-column-slice variant was also measured 40-60% SLOWER than the
    full-row version — see the perf notes in paged_row_view.
    """

    from rtp_llm.models_py.kernels.ascend.state_migration import migrate_state_rows

    same = read_idx == write_idx
    # no-migration rows become self-copies; the triton kernel skips them
    src = torch.where(same, write_idx, read_idx)
    migrate_state_rows(seg_view, src, write_idx)


__all__ = ["decode_state_indices", "paged_row_view", "seed_state_segment"]
