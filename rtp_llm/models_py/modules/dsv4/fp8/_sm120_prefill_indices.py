"""Ready-record fast path for SM120 prefill sparse-index tables.

The workspace SM120 prefill producer (`combine_topk_swa_indices{,_cp}` plus
the split in `attention.py`) already emits exactly what the SM120 consumer
needs: int32 contiguous tables whose rows are a dense valid prefix, a zero
tail, non-negative entries, and lengths bounded by the table width.  On such
tables the generic `canonical_topk` + `clamp_min_` chain is exactly the
identity copy plus an optional zero pad to the next supported width and an
in-place clamp of the (already bounded) length tensor.  This module carries
that producer/consumer contract explicitly so the eligible caller can skip
the redundant sort/gather/repair work.

Only audited private producers construct `Sm120PrefillIndices`.  Every
consumer-side metadata mismatch falls back to the generic canonical path;
unsupported means "not optimized", never "silently accepted".
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

_MODE = os.environ.get("DSV4_SM120_PREFILL_INDEX_ADAPTER", "off").strip().lower()
if _MODE not in ("off", "ready"):
    # "direct" (producer writes native-boundary layout directly, no record) is
    # NOT implemented in this revision; never let it silently follow the ready
    # branch. Fail at import so no run can mislabel its path.
    raise RuntimeError(
        "DSV4_SM120_PREFILL_INDEX_ADAPTER must be one of off|ready, "
        f"got {_MODE!r} ('direct' is not implemented in this revision)"
    )


_SPLIT_FUSION_MODE = os.environ.get("DSV4_SM120_PREFILL_SPLIT_FUSION", "0").strip()
if _SPLIT_FUSION_MODE not in ("0", "1"):
    raise RuntimeError("DSV4_SM120_PREFILL_SPLIT_FUSION must be 0|1")
_SPLIT_FUSION_ENABLED = _SPLIT_FUSION_MODE == "1"
_SPLIT_FUSION_LOGGED = False


def index_adapter_mode() -> str:
    """Mode frozen at import; off keeps the incumbent path byte-identical."""
    return _MODE


@dataclass(frozen=True)
class Sm120PrefillIndices:
    """Already normalized native-boundary tables for one forward.

    Invariants (guaranteed by the private producer, re-checked as metadata
    only at the consumer): int32, contiguous, rows == Q rows, dense valid
    prefix per row, zero tail, no negative entries, lens <= width.
    """

    swa_indices: torch.Tensor  # [rows, window] int32
    swa_lens: torch.Tensor  # [rows] int32
    extra_indices: Optional[torch.Tensor]  # [rows, aligned_extra] int32
    extra_lens: Optional[torch.Tensor]  # [rows] int32


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
    if _SPLIT_FUSION_ENABLED:
        from rtp_llm.models_py.modules.dsv4.fp8._sm120_prefill_split import (
            can_fuse_split,
            fused_split,
        )

        kwargs = dict(
            M=M,
            N=N,
            window_size=window_size,
            extra_width=extra_width,
            ratio=ratio,
            device=device,
        )
        if can_fuse_split(combined_indices, combined_lens, **kwargs):
            global _SPLIT_FUSION_LOGGED
            if not _SPLIT_FUSION_LOGGED:
                logging.getLogger(__name__).info(
                    "DSV4 fused prefill index split engaged on SM120"
                )
                _SPLIT_FUSION_LOGGED = True
            return fused_split(combined_indices, combined_lens, **kwargs)

    from rtp_llm.models_py.modules.dsv4.const_cache import cached_arange

    gather_width = M - N
    combined_2d = combined_indices.squeeze(1).to(torch.int64)
    valid = combined_2d >= 0
    request_ids = torch.div(combined_2d.clamp_min(0), M, rounding_mode="floor")
    local_slots = torch.remainder(combined_2d.clamp_min(0), M)
    is_extra = valid & (local_slots < N)
    extra_lens = is_extra.sum(dim=1, dtype=torch.int32)
    swa_lens = combined_lens.to(torch.int32) - extra_lens
    extra_cols = cached_arange(extra_width, dtype=torch.int64, device=device).unsqueeze(
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
    swa_cols = cached_arange(window_size, dtype=torch.int64, device=device).unsqueeze(0)
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


def _ready_chunk(
    indices: torch.Tensor,
    lens: torch.Tensor,
    start: int,
    end: int,
    supported_widths: Sequence[int],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Replicate canonical_topk+clamp exactly on producer-ready tables.

    Returns ``(chunk_indices, chunk_lens)`` or ``None`` when any metadata
    contract fails; ``None`` instructs the caller to use the generic path.
    """
    if indices.dim() == 3:
        indices = indices.squeeze(1)
    if (
        indices.dim() != 2
        or indices.dtype != torch.int32
        or not indices.is_contiguous()
    ):
        return None
    rows = end - start
    if int(indices.shape[0]) < end or lens.dim() != 1:
        return None
    if lens.dtype != torch.int32 or int(lens.shape[0]) != int(indices.shape[0]):
        return None
    # Layout parity with the generic path: canonical_topk's to() produces a
    # contiguous tensor on the consumer device. A mixed-device pair or a
    # non-contiguous lens view (e.g. a stride-2 slice) must fall back rather
    # than return a layout the consumer never sees on the generic path.
    if lens.device != indices.device or not lens.is_contiguous():
        return None
    width = int(indices.shape[1])
    chunk_indices = indices[start:end]
    # token_lens() today does lengths.to(int32).clamp_(0, width): for the
    # int32 producer lens this mutates the slice view in place.  Preserve the
    # observable side effect exactly; values are already bounded, so this is
    # the only kernel the ready path launches per table.
    chunk_lens = lens[start:end].clamp_(0, width)
    if width not in supported_widths:
        padded_width = next((w for w in supported_widths if w >= width), None)
        if padded_width is None:
            return None  # generic path raises its own error, as today
        # canonical_topk pads with -1 and the consumer then clamps to 0; the
        # producer tail is already zero, so a zero pad is bit-identical.
        padded = torch.zeros(
            (rows, padded_width), dtype=torch.int32, device=indices.device
        )
        padded[:, :width] = chunk_indices
        chunk_indices = padded
    return chunk_indices.contiguous(), chunk_lens
