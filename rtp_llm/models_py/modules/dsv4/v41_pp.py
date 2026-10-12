"""Validated V4.1 forward-local state at the L20/L21 PP boundary.

The explicit returned frame belongs to one forward. No module-global mHC,
top-k or candidate state is installed. Validation is synchronous prefill work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Sequence

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
    # Value checks are deliberately outside graph capture in the initial path.
    topk = frame.topk.cpu()
    candidates = frame.candidate_blocks.cpu()
    counts = frame.candidate_counts.cpu()
    for row, (request, position, valid) in enumerate(expected_rows.tolist()):
        if valid not in (0, 1):
            raise ValueError("V4.1 PP row validity must be binary")
        if not valid:
            if (
                (request, position) != (-1, -1)
                or int(counts[row]) != 0
                or bool((topk[row] != -1).any())
                or bool((candidates[row] != -1).any())
            ):
                raise ValueError("V4.1 PP padding rows must have masked selections")
            continue
        if not 0 <= request < len(spans):
            raise ValueError("V4.1 PP row refers to an absent request")
        span = spans[request]
        if not span.start <= position < span.start + span.count:
            raise ValueError("V4.1 PP query position lies outside its delta span")
        indices = topk[row]
        active = indices[indices >= 0]
        if (
            bool(((indices < -1) | (indices > position)).any())
            or active.unique().numel() != active.numel()
            or active.numel() != min(512, position + 1)
        ):
            raise ValueError(
                "V4.1 PP top-k contains invalid, duplicate or missing positions"
            )
        block_count = int(counts[row])
        if block_count != min(2048, position // 8 + 1):
            raise ValueError("V4.1 PP candidate count is invalid")
        blocks = candidates[row, :block_count]
        if (
            bool(((blocks < 0) | (blocks > position // 8)).any())
            or blocks.unique().numel() != block_count
            or bool((candidates[row, block_count:] != -1).any())
        ):
            raise ValueError("V4.1 PP candidates contain invalid or duplicate blocks")
        if not bool((blocks == position // 8).any()):
            raise ValueError("V4.1 PP candidates omit the newest visible block")
    active_rows = expected_rows[:, 2].bool().to(device)
    if not bool(torch.isfinite(frame.hidden_states[active_rows]).all()) or not bool(
        torch.isfinite(frame.prev_pre_mix[active_rows]).all()
    ):
        raise ValueError("V4.1 PP active residual and pre-mix must be finite")
    active_map = expected_rows[expected_rows[:, 2] == 1, :2]
    if active_map.unique(dim=0).shape[0] != active_map.shape[0]:
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
