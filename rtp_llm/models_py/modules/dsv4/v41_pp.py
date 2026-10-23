"""Validated V4.1 forward-local state at the L20/L21 PP boundary.

The explicit returned frame belongs to one forward. No module-global mHC,
top-k or candidate state is installed. Validation is synchronous prefill work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import (
        V41L20ReplicaReceiver,
        V41L20Span,
    )

_HEADER = "v41_pp_header"
_ROW_MAP = "v41_row_map"
_PRE_MIX = "v41_prev_pre_mix"
_TOPK = "v41_l20_topk"
_CANDIDATES = "v41_candidate_blocks"
_COUNTS = "v41_candidate_counts"
_FIELDS = frozenset(("hidden_states", _ROW_MAP, _PRE_MIX, _TOPK, _CANDIDATES, _COUNTS))


@dataclass(frozen=True)
class V41PPFrame:
    hidden_states: torch.Tensor
    prev_pre_mix: torch.Tensor
    row_map: torch.Tensor
    topk: torch.Tensor
    candidate_blocks: torch.Tensor
    candidate_counts: torch.Tensor


def _tensor(value, name, shape, dtype, device) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
    ):
        raise ValueError(
            f"V4.1 PP {name} requires contiguous {dtype} {shape} on {device}"
        )
    return value


def _validate_selection_rows(topk, candidates, counts, rows, spans) -> None:
    """Validate the selection rows on device, preserving row-first error order.

    ``topk``/``candidates`` are the wide [T,512] and [T,2048] int32 selection
    tensors. Staging them on the host and sorting them in numpy cost ~84MB of
    blocking D2H plus ~180ms of numpy work on *every* PP handoff at 32K, which
    serialized the two pipeline stages. The checks themselves are unchanged;
    only the accumulated per-row error codes (T int32) return to the host, so a
    failure still names the first bad row and its exact code.
    """
    count = rows.shape[0]
    if not count:
        return
    device = topk.device
    rows_device = rows.to(device)
    requests, positions, validity = (
        rows_device[:, 0],
        rows_device[:, 1],
        rows_device[:, 2],
    )
    indices = topk
    blocks = candidates
    block_counts = counts.to(device)
    real = validity == 1
    errors = torch.zeros(count, dtype=torch.int32, device=device)

    def reject(mask, code):
        nonlocal errors
        errors = torch.where(
            mask & (errors == 0), torch.full_like(errors, code), errors
        )

    reject((validity != 0) & ~real, 1)
    reject(
        (validity == 0)
        & (
            (requests != -1)
            | (positions != -1)
            | (block_counts != 0)
            | (indices != -1).any(dim=1)
            | (blocks != -1).any(dim=1)
        ),
        2,
    )
    present = (requests >= 0) & (requests < len(spans))
    reject(real & ~present, 3)
    starts = torch.tensor([s.start for s in spans], dtype=torch.int64, device=device)
    ends = torch.tensor(
        [s.start + s.count for s in spans], dtype=torch.int64, device=device
    )
    owner = requests.clamp(0, len(spans) - 1)
    in_span = (positions >= starts[owner]) & (positions < ends[owner])
    reject(real & ~in_span, 4)

    sorted_indices = torch.sort(indices, dim=1).values
    duplicates = (
        (sorted_indices[:, 1:] == sorted_indices[:, :-1]) & (sorted_indices[:, 1:] >= 0)
    ).any(dim=1)
    expected_positions = torch.where(
        positions >= 511, torch.full_like(positions, 512), positions + 1
    )
    reject(
        real
        & (
            ((indices < -1) | (indices > positions[:, None])).any(dim=1)
            | duplicates
            | ((indices >= 0).sum(dim=1) != expected_positions)
        ),
        5,
    )
    newest = positions // 8
    reject(real & (block_counts != torch.clamp(newest + 1, max=2048)), 6)
    active = torch.arange(blocks.shape[1], device=device)[None] < block_counts[:, None]
    sorted_blocks = torch.sort(blocks, dim=1).values
    duplicates = (
        (sorted_blocks[:, 1:] == sorted_blocks[:, :-1]) & (sorted_blocks[:, 1:] >= 0)
    ).any(dim=1)
    reject(
        real
        & (
            (active & ((blocks < 0) | (blocks > newest[:, None]))).any(dim=1)
            | duplicates
            | (~active & (blocks != -1)).any(dim=1)
        ),
        7,
    )
    reject(real & ~(active & (blocks == newest[:, None])).any(dim=1), 8)

    codes = errors.cpu()
    failed = codes.nonzero()
    if failed.numel():
        messages = (
            "",
            "V4.1 PP row validity must be binary",
            "V4.1 PP padding rows must have masked selections",
            "V4.1 PP row refers to an absent request",
            "V4.1 PP query position lies outside its delta span",
            "V4.1 PP top-k contains invalid, duplicate or missing positions",
            "V4.1 PP candidate count is invalid",
            "V4.1 PP candidates contain invalid or duplicate blocks",
            "V4.1 PP candidates omit the newest visible block",
        )
        raise ValueError(messages[int(codes[failed[0, 0]])])


def _validate_frame(
    frame: V41PPFrame, spans: Sequence[V41L20Span], expected_rows: torch.Tensor
) -> None:
    if not isinstance(frame, V41PPFrame) or not isinstance(
        frame.hidden_states, torch.Tensor
    ):
        raise ValueError("V4.1 PP requires a complete forward frame")
    if (
        not isinstance(expected_rows, torch.Tensor)
        or expected_rows.ndim != 2
        or expected_rows.shape[1] != 3
    ):
        raise ValueError("V4.1 PP expected row map must have shape [T, 3]")
    count = expected_rows.shape[0]
    cpu = torch.device("cpu")
    device = frame.hidden_states.device
    _tensor(expected_rows, "trusted row map", (count, 3), torch.int64, cpu)
    _tensor(frame.row_map, "row map", (count, 3), torch.int64, cpu)
    if not torch.equal(frame.row_map, expected_rows):
        raise ValueError("V4.1 PP row map disagrees with CP dispatch")
    _tensor(
        frame.hidden_states, "hidden states", (count, 4, 5120), torch.bfloat16, device
    )
    _tensor(frame.prev_pre_mix, "previous pre-mix", (count, 4), torch.float32, device)
    _tensor(frame.topk, "top-k", (count, 512), torch.int32, device)
    _tensor(
        frame.candidate_blocks, "candidate blocks", (count, 2048), torch.int32, device
    )
    _tensor(frame.candidate_counts, "candidate counts", (count,), torch.int32, device)
    # Selection and finiteness checks stay on the device: staging the wide
    # top-k/candidate tensors through the host blocked every PP handoff and
    # serialized the two stages (see _validate_selection_rows).
    _validate_selection_rows(
        frame.topk,
        frame.candidate_blocks,
        frame.candidate_counts,
        expected_rows,
        spans,
    )
    active_rows = expected_rows[:, 2].bool().to(device)
    torch._assert_async(
        torch.isfinite(frame.hidden_states[active_rows]).all(),
        "V4.1 PP active residual must be finite",
    )
    torch._assert_async(
        torch.isfinite(frame.prev_pre_mix[active_rows]).all(),
        "V4.1 PP active pre-mix must be finite",
    )
    active_map = expected_rows[expected_rows[:, 2] == 1, :2]
    if np.unique(active_map.numpy(), axis=0).shape[0] != active_map.shape[0]:
        raise ValueError("V4.1 PP CP row map repeats a query")


def pack_v41_pp_frame(
    delta_tensors: Mapping[str, torch.Tensor],
    frame: V41PPFrame,
    spans: Sequence[V41L20Span],
    expected_rows: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combine a validated frame with exported L20 deltas for PPIntermediates."""
    _validate_frame(frame, spans, expected_rows)
    if set(delta_tensors).intersection(_FIELDS | {_HEADER}):
        raise ValueError("V4.1 PP frame keys collide with the supplied delta map")
    result = dict(delta_tensors)
    count = expected_rows.shape[0]
    result[_HEADER] = torch.tensor([1, count, 4, 5120], dtype=torch.int64)
    if count:
        result.update(
            {
                "hidden_states": frame.hidden_states,
                _PRE_MIX: frame.prev_pre_mix,
                _ROW_MAP: frame.row_map,
                _TOPK: frame.topk,
                _CANDIDATES: frame.candidate_blocks,
                _COUNTS: frame.candidate_counts,
            }
        )
    return result


def validate_v41_pp_frame(
    tensors: Mapping[str, torch.Tensor],
    receiver: V41L20ReplicaReceiver,
    expected_spans: Sequence[V41L20Span],
    expected_rows: torch.Tensor,
) -> V41PPFrame:
    """Decode and validate the forward frame without importing cache deltas."""
    if (
        not isinstance(expected_rows, torch.Tensor)
        or expected_rows.ndim != 2
        or expected_rows.shape[1] != 3
    ):
        raise ValueError("V4.1 PP expected row map must have shape [T, 3]")
    count = expected_rows.shape[0]
    header = _tensor(
        tensors.get(_HEADER), "header", (4,), torch.int64, torch.device("cpu")
    )
    if header.tolist() != [1, count, 4, 5120]:
        raise ValueError("V4.1 PP header disagrees with the dispatch shape")
    device = receiver.caches["hca_kv"].raw.device
    if count:
        if not _FIELDS.issubset(tensors):
            raise ValueError(
                "V4.1 PP payload is missing delayed mHC or selection state"
            )
        frame = V41PPFrame(
            tensors["hidden_states"],
            tensors[_PRE_MIX],
            tensors[_ROW_MAP],
            tensors[_TOPK],
            tensors[_CANDIDATES],
            tensors[_COUNTS],
        )
        if (
            not isinstance(frame.hidden_states, torch.Tensor)
            or frame.hidden_states.device != device
        ):
            raise ValueError("V4.1 PP frame and replica must share a device")
    else:
        if _FIELDS.intersection(tensors):
            raise ValueError("V4.1 empty PP frame must omit zero-element payloads")
        frame = V41PPFrame(
            torch.empty(0, 4, 5120, dtype=torch.bfloat16, device=device),
            torch.empty(0, 4, dtype=torch.float32, device=device),
            torch.empty(0, 3, dtype=torch.int64),
            torch.empty(0, 512, dtype=torch.int32, device=device),
            torch.empty(0, 2048, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
        )
    _validate_frame(frame, expected_spans, expected_rows)
    for tensor in frame.__dict__.values():
        if tensor.numel() and any(
            tensor.device == cache.raw.device
            and tensor.untyped_storage().data_ptr()
            == cache.raw.untyped_storage().data_ptr()
            for cache in receiver.caches.values()
        ):
            raise ValueError("V4.1 PP frame aliases replica storage")
    return frame


def import_v41_pp_frame(
    tensors: Mapping[str, torch.Tensor],
    receiver: V41L20ReplicaReceiver,
    expected_spans: Sequence[V41L20Span],
    expected_rows: torch.Tensor,
) -> V41PPFrame:
    """Reject an incomplete handoff before any cache write, then import deltas."""
    frame = validate_v41_pp_frame(tensors, receiver, expected_spans, expected_rows)
    receiver.apply(tensors, expected_spans)
    for span in expected_spans:
        receiver.require_ready(span.identity, span.start + span.count, span.chunk_id)
    return frame
