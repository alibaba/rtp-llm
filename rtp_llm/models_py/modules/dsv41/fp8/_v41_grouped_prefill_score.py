"""Bounded request-local FP4 scores over adjacent, already assembled key views."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# Per-score allocation budget, fixed at import; not a total forward-memory cap.
try:
    _MAX_LOGITS_BYTES = int(
        os.environ.get("DSV41_PREFILL_SCORE_MAX_BYTES", str(256 * 1024 * 1024))
    )
except ValueError as error:
    raise ValueError(
        "DSV41_PREFILL_SCORE_MAX_BYTES must be an integer in [1, 1073741824]"
    ) from error
if not 0 < _MAX_LOGITS_BYTES <= 1024 * 1024 * 1024:
    raise ValueError(
        "DSV41_PREFILL_SCORE_MAX_BYTES must be an integer in [1, 1073741824]"
    )
_MAX_GROUP_ROWS = 4096
_MAX_LOGITS_INFLATION = 2
_MAX_REQUESTS = 128
_MIN_BOUNDS_GROUPS = 4
_MAX_BOUNDS_GROUPS = 256


@triton.jit
def _write_score_bounds(
    positions,
    request_ids,
    key_counts,
    bounds,
    ROWS,
    REQUESTS,
    FIRST,
    STOP,
    KEY_SPAN,
    WIDTH,
    BLOCK,
    RATIO: tl.constexpr,
):
    # Both tiles are fixed: batch lengths and grouping never enter the JIT key.
    ids = tl.arange(0, 128)
    counts = tl.load(key_counts + ids, ids < REQUESTS, other=0).to(tl.int64)
    padded = tl.cdiv(tl.maximum(counts, 0), 256) * 256
    offsets = tl.cumsum(padded) - padded
    origin = tl.sum(tl.where(ids == FIRST, offsets, 0))
    row = BLOCK * 128 + ids
    request = tl.load(request_ids + row, row < ROWS, other=-1).to(tl.int32)
    safe_request = tl.minimum(tl.maximum(request, 0), 127)
    count = tl.gather(counts, safe_request, 0)
    start = tl.gather(offsets, safe_request, 0) - origin
    valid = (
        (request >= FIRST)
        & (request < STOP)
        & (request < REQUESTS)
        & (count >= 0)
        & (count <= WIDTH)
        & (start >= 0)
        & (start + count <= KEY_SPAN)
    )
    position = tl.load(positions + row, row < ROWS, other=-1).to(tl.int64)
    visible = tl.minimum(tl.maximum(position + 1, 0) // RATIO, count)
    visible = tl.where(valid, visible, 0).to(tl.int32)
    start = tl.where(valid, start, 0).to(tl.int32)
    tl.store(bounds + row, start, row < ROWS)
    tl.store(bounds + ROWS + row, start + visible, row < ROWS)
    tl.store(bounds + 2 * ROWS + row, 0, row < ROWS)
    tl.store(bounds + 3 * ROWS + row, visible, row < ROWS)


@triton.jit(
    do_not_specialize=["ROWS", "REQUESTS", "FIRST", "STOP", "KEY_SPAN", "WIDTH"]
)
def _grouped_score_bounds_kernel(
    positions,
    request_ids,
    key_counts,
    bounds,
    ROWS,
    REQUESTS,
    FIRST,
    STOP,
    KEY_SPAN,
    WIDTH,
    RATIO: tl.constexpr,
):
    _write_score_bounds(
        positions,
        request_ids,
        key_counts,
        bounds,
        ROWS,
        REQUESTS,
        FIRST,
        STOP,
        KEY_SPAN,
        WIDTH,
        tl.program_id(0),
        RATIO,
    )


@triton.jit(do_not_specialize=["REQUESTS", "GROUPS"])
def _all_grouped_score_bounds_kernel(
    positions,
    request_ids,
    key_counts,
    descriptors,
    bounds,
    REQUESTS,
    GROUPS,
    RATIO: tl.constexpr,
):
    tile = tl.program_id(0)
    lower = tl.full((), 0, tl.int32)
    upper = GROUPS
    # upper_bound needs nine steps for all 257 insertion points at the cap.
    for _ in range(9):
        middle = (lower + upper) // 2
        end = tl.load(descriptors + middle * 8, middle < GROUPS, other=0)
        advance = (middle < GROUPS) & (end <= tile)
        lower = tl.where(advance, middle + 1, lower)
        upper = tl.where(advance, upper, middle)
    descriptor = descriptors + lower * 8
    first = tl.load(descriptor + 1)
    stop = tl.load(descriptor + 2)
    row_start = tl.load(descriptor + 3)
    rows = tl.load(descriptor + 4)
    span = tl.load(descriptor + 5)
    width = tl.load(descriptor + 6)
    tile_start = tl.load(descriptor + 7)
    _write_score_bounds(
        positions + row_start,
        request_ids + row_start,
        key_counts,
        bounds + 4 * row_start,
        rows,
        REQUESTS,
        first,
        stop,
        span,
        width,
        tile - tile_start,
        RATIO,
    )


def _bounds_descriptors(layout):
    """Bounded CPU-only descriptors, with exactly the original per-group CTAs."""
    if not 2 <= len(layout) <= _MAX_BOUNDS_GROUPS:
        return None
    descriptors, tiles, cursor = [], 0, 0
    for group in layout:
        count = group.rows.stop - group.rows.start
        if group.rows.start != cursor or not 0 < count <= _MAX_GROUP_ROWS:
            return None
        end = tiles + (count + 127) // 128
        descriptors.append(
            (
                end,
                group.first,
                group.stop,
                cursor,
                count,
                group.scale.numel(),
                group.width,
                tiles,
            )
        )
        tiles, cursor = end, group.rows.stop
    return descriptors, tiles, cursor


@triton.jit(do_not_specialize=["WIDTH", "STRIDE"])
def _mask_tail_kernel(logits, ends, WIDTH, STRIDE, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    end = tl.load(ends + row)
    offsets = tl.arange(0, TILE)
    for start in range(end, WIDTH, TILE):
        column = start + offsets
        tl.store(logits + row * STRIDE + column, -float("inf"), column < WIDTH)


def _slab_view(parts, *, padded=False):
    """Return a metadata-only alias, or None for nonadjacent allocations."""
    if not parts:
        return None
    first = parts[0]
    offset = first.storage_offset()
    storage = first.untyped_storage().data_ptr()
    span = 0
    for part in parts:
        if (
            part.dtype != first.dtype
            or part.device != first.device
            or part.shape[1:] != first.shape[1:]
            or not part.is_contiguous()
            or part.untyped_storage().data_ptr() != storage
            or part.storage_offset() != offset
        ):
            return None
        span = (offset - first.storage_offset()) // first.stride(0) + part.shape[0]
        count = ((part.shape[0] + 255) // 256) * 256 if padded else part.shape[0]
        offset += count * first.stride(0)
    return first.as_strided((span, *first.shape[1:]), first.stride())


@dataclass(frozen=True)
class _ScoreGroup:
    first: int
    stop: int
    rows: slice
    width: int
    payload: torch.Tensor
    scale: torch.Tensor


def _group_layout(keys, rows):
    """Group adjacent requests and chunk oversized Q ranges without K copies."""
    if len(keys) < 2 or len(rows) != len(keys) or any(n <= 0 for n in rows):
        return None
    device = keys[0].quant.device
    for key, count in zip(keys, rows):
        width = len(key)
        if (
            width <= 0
            or width * len(keys) >= 2**31
            or key.quant.shape != (width, 64)
            or key.scale.shape != (width,)
            or key.quant.dtype != torch.int8
            or key.scale.dtype != torch.int32
            or key.quant.device != device
            or key.scale.device != device
            or not key.quant.is_contiguous()
            or not key.scale.is_contiguous()
            or key.quant.data_ptr() % 16
            or key.scale.data_ptr() % 16
        ):
            return None
    groups = []
    first, row_start = 0, 0
    while first < len(keys):
        stop = first + 1
        width, group_rows = len(keys[first]), rows[first]
        useful_scores = group_rows * width
        payload, scale = keys[first].quant, keys[first].scale
        stride = (width + 255) // 256 * 256
        limit = min(_MAX_GROUP_ROWS, _MAX_LOGITS_BYTES // (stride * 4)) // 4 * 4
        if limit == 0:
            return None
        if group_rows > limit:
            for offset in range(0, group_rows, limit):
                groups.append(
                    _ScoreGroup(
                        first,
                        stop,
                        slice(
                            row_start + offset,
                            row_start + min(offset + limit, group_rows),
                        ),
                        width,
                        payload,
                        scale,
                    )
                )
            row_start += group_rows
            first = stop
            continue
        while stop < len(keys):
            next_width = max(width, len(keys[stop]))
            next_rows = group_rows + rows[stop]
            next_useful = useful_scores + rows[stop] * len(keys[stop])
            if (
                next_rows > _MAX_GROUP_ROWS
                or next_rows * next_width > _MAX_LOGITS_INFLATION * next_useful
                or ((next_rows + 3) // 4 * 4) * ((next_width + 255) // 256 * 256) * 4
                > _MAX_LOGITS_BYTES
            ):
                break
            next_payload = _slab_view((payload, keys[stop].quant), padded=True)
            next_scale = _slab_view((scale, keys[stop].scale), padded=True)
            if next_payload is None or next_scale is None:
                break
            payload, scale = next_payload, next_scale
            group_rows += rows[stop]
            width = next_width
            useful_scores = next_useful
            stop += 1
        groups.append(
            _ScoreGroup(
                first,
                stop,
                slice(row_start, row_start + group_rows),
                width,
                payload,
                scale,
            )
        )
        row_start += group_rows
        first = stop
    if len(groups) == len(keys) and all(
        group.stop - group.first == 1 for group in groups
    ):
        return None
    return tuple(groups)


class _GroupedScores:
    def __init__(
        self,
        q,
        sf,
        weights,
        positions,
        ratio,
        shared,
        rows,
        layout,
        req_ids,
        key_counts,
    ):
        self.q, self.sf, self.weights = q, sf, weights
        self.positions, self.ratio, self.shared = positions, ratio, shared
        self.rows, self.layout = rows, layout
        self.req_ids, self.key_counts = req_ids, key_counts

    def _bounds(self):
        from ._v41_prefill_metadata import try_score_bounds

        # This cache is cleared at the existing forward boundary. Identity also
        # guards replacement of positions with an equal-shaped tensor mid-forward.
        cache = self.shared.setdefault("prefill_score_bounds", {})
        layout = tuple(
            (g.first, g.stop, g.rows.start, g.rows.stop, g.width) for g in self.layout
        )
        key = ("grouped", self.rows, self.ratio, layout)
        entry = cache.get(key)
        if entry is not None and all(
            a is b
            for a, b in zip(entry[:3], (self.positions, self.req_ids, self.key_counts))
        ):
            return entry[3]
        # A captured host copy would require extending its pinned source lifetime
        # beyond this forward. Keep the existing device-only path for capture.
        if (
            self.key_counts is not None
            and _MIN_BOUNDS_GROUPS <= len(self.layout) <= _MAX_BOUNDS_GROUPS
            and not torch.cuda.is_current_stream_capturing()
        ):
            description = _bounds_descriptors(self.layout)
            if description is not None:
                descriptors, tiles, total_rows = description
                host = torch.tensor(descriptors, dtype=torch.int64, pin_memory=True)
                device = host.to(self.q.device, non_blocking=True)
                output = torch.empty(
                    4 * total_rows, dtype=torch.int32, device=self.q.device
                )
                _all_grouped_score_bounds_kernel[(tiles,)](
                    self.positions,
                    self.req_ids,
                    self.key_counts,
                    device,
                    output,
                    self.key_counts.numel(),
                    len(self.layout),
                    self.ratio,
                    num_warps=4,
                )
                bounds = tuple(
                    (
                        g.rows,
                        *output[4 * g.rows.start : 4 * g.rows.stop]
                        .view(4, g.rows.stop - g.rows.start)
                        .unbind(0),
                    )
                    for g in self.layout
                )
                cache[key] = (self.positions, self.req_ids, self.key_counts, bounds)
                return bounds
        bounds = []
        for group in self.layout:
            rows = group.rows
            if self.key_counts is not None:
                count = rows.stop - rows.start
                output = torch.empty(
                    (4, count), dtype=torch.int32, device=self.q.device
                )
                _grouped_score_bounds_kernel[(triton.cdiv(count, 128),)](
                    self.positions[rows],
                    self.req_ids[rows],
                    self.key_counts,
                    output,
                    count,
                    self.key_counts.numel(),
                    group.first,
                    group.stop,
                    group.scale.numel(),
                    group.width,
                    self.ratio,
                    num_warps=4,
                )
                starts, ends, zeros, visible = output.unbind(0)
                bounds.append((rows, starts, ends, zeros, visible))
                continue
            zeros, visible = try_score_bounds(
                self.positions[rows], group.width, self.ratio
            )
            counts = self.rows[group.first : group.stop]
            if all(n == counts[0] for n in counts):
                starts = torch.arange(
                    rows.stop - rows.start, device=self.q.device, dtype=torch.int32
                )
                starts.div_(counts[0], rounding_mode="floor").mul_(
                    (group.width + 255) // 256 * 256
                )
            else:
                # Uncommon ragged batches: native fills avoid host uploads or
                # layout-specialized JIT. Bounds are reused across source layers.
                starts = torch.empty_like(visible)
                offset = 0
                for i, count in enumerate(counts):
                    starts[offset : offset + count].fill_(
                        i * ((group.width + 255) // 256 * 256)
                    )
                    offset += count
            bounds.append((rows, starts, starts + visible, zeros, visible))
        cache[key] = (self.positions, self.req_ids, self.key_counts, tuple(bounds))
        return cache[key][3]

    def groups(self, *, mask_tail=True):
        """Yield (packed row slice, logits, visible, (zeros, visible)).

        Each FP32 output respects the configured budget (default 256 MiB).
        Invalid suffixes are -inf by default;
        callers disabling masking must use visible bounds or mask before use.
        Columns are request-local; candidate/TopK output needs no offset change.
        Consume and delete logits before advancing, including derived views:
        retaining them retains their whole group allocation. No logits are cached.
        """
        from ._indexer_score import fp8_fp4_mqa_indexer_score

        for group, bounds in zip(self.layout, self._bounds()):
            rows, starts, ends, zeros, visible = bounds
            logits = fp8_fp4_mqa_indexer_score(
                self.q[rows],
                self.sf[rows],
                group.payload,
                group.scale,
                self.weights[rows],
                starts,
                ends,
                clean_logits=False,
                max_seqlen_k=group.width,
            )
            if mask_tail:
                _mask_tail_kernel[(logits.shape[0],)](
                    logits, visible, group.width, logits.stride(0), 256
                )
            yield rows, logits, visible, (zeros, visible)
            del logits


def try_grouped_scores(
    q,
    sf,
    weights,
    globals_by_req,
    slices,
    positions,
    ratio,
    shared,
    *,
    req_ids=None,
    key_counts=None,
):
    """Return a lazy plan or None before GPU work for unsupported layouts.

    Accept already-prepared FP32 weights from the original full-M BF16 head
    projection; never reproject or fold FP4 scales into them. Candidate tables
    and CED projection in shared are intentionally allowed: callers consume
    scores after this boundary. Forward-local positions must remain immutable.
    """
    count = len(globals_by_req)
    if (
        not 2 <= count <= _MAX_REQUESTS
        or slices is None
        or len(slices) != count
        or q.device.type != "cuda"
        or torch.version.hip is not None
        or q.dtype != torch.int8
        or q.ndim != 3
        or q.shape[1:] != (32, 64)
        or not q.is_contiguous()
        or sf.shape != q.shape[:2]
        or sf.dtype != torch.int32
        or not sf.is_contiguous()
        or weights.shape != q.shape[:2]
        or weights.dtype != torch.float32
        or not weights.is_contiguous()
        or sf.device != q.device
        or weights.device != q.device
        or q.data_ptr() % 16
        or sf.data_ptr() % 16
        or weights.data_ptr() % 16
        or positions.shape != (q.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != q.device
        or not positions.is_contiguous()
        or ratio not in (1, 2)
    ):
        return None
    rows, cursor = [], 0
    for s in slices:
        if (
            not isinstance(s, slice)
            or s.start != cursor
            or s.stop is None
            or s.stop <= cursor
            or s.step not in (None, 1)
        ):
            return None
        rows.append(s.stop - cursor)
        cursor = s.stop
    if cursor != q.shape[0]:
        return None
    rows = tuple(rows)
    from ._v41_prefill_indexer import PrefillIndexerKeys

    keys = [entry[1] for entry in globals_by_req]
    if not all(isinstance(k, PrefillIndexerKeys) for k in keys):
        return None
    layout = _group_layout(keys, rows)
    if layout is None or keys[0].quant.device != q.device:
        return None
    if req_ids is not None or key_counts is not None:
        if not all(
            isinstance(value, torch.Tensor)
            and value.device == q.device
            and value.shape == shape
            and value.dtype in (torch.int32, torch.int64)
            and value.is_contiguous()
            for value, shape in ((req_ids, (q.shape[0],)), (key_counts, (count,)))
        ):
            return None
    elif len(set(len(k) for k in keys)) != 1:
        return None
    from ._indexer_score import has_fp8_fp4_mqa_logits

    if (
        torch.cuda.get_device_capability(q.device)[0] != 10
        or not has_fp8_fp4_mqa_logits()
    ):
        return None
    return _GroupedScores(
        q, sf, weights, positions, ratio, shared, rows, layout, req_ids, key_counts
    )
