"""Bounded request-local FP4 scores over adjacent, already assembled key views."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41._prefill_stream import prefill_stream_is_capturing

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
_MAX_REQUESTS = 1024
_MIN_BOUNDS_GROUPS = 4
_MAX_BOUNDS_GROUPS = 512
_BOUNDS_SEARCH_STEPS = tl.constexpr(_MAX_BOUNDS_GROUPS.bit_length())


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
    PLANES: tl.constexpr,
):
    # Each group spans at most 128 requests, independently of the batch size.
    ids = tl.arange(0, 128)
    counts = tl.load(
        key_counts + FIRST + ids,
        (FIRST + ids < STOP) & (FIRST + ids < REQUESTS),
        other=0,
    ).to(tl.int64)
    padded = tl.cdiv(tl.maximum(counts, 0), 256) * 256
    offsets = tl.cumsum(padded) - padded
    row = BLOCK * 128 + ids
    request = tl.load(request_ids + row, row < ROWS, other=-1).to(tl.int32)
    safe_request = tl.minimum(tl.maximum(request - FIRST, 0), 127)
    count = tl.gather(counts, safe_request, 0)
    start = tl.gather(offsets, safe_request, 0)
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
    if PLANES == 4:
        tl.store(bounds + 2 * ROWS + row, 0, row < ROWS)
    tl.store(bounds + (PLANES - 1) * ROWS + row, visible, row < ROWS)


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
    PLANES: tl.constexpr = 4,
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
        PLANES,
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
    PLANES: tl.constexpr = 4,
):
    tile = tl.program_id(0)
    lower = tl.full((), 0, tl.int32)
    upper = GROUPS
    # Include the insertion point after the final group.
    for _ in range(_BOUNDS_SEARCH_STEPS):
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
        bounds + PLANES * row_start,
        rows,
        REQUESTS,
        first,
        stop,
        span,
        width,
        tile - tile_start,
        RATIO,
        PLANES,
    )


def _bounds_descriptors(layout):
    """Bounded CPU-only descriptors, with exactly the original per-group CTAs."""
    if not 2 <= len(layout) <= _MAX_BOUNDS_GROUPS:
        return None
    origin = layout[0].rows.start
    descriptors, tiles, cursor = [], 0, origin
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
                cursor - origin,
                count,
                group.scale.numel(),
                group.source_width or group.width,
                tiles,
            )
        )
        tiles, cursor = end, group.rows.stop
    return descriptors, tiles, cursor - origin


@triton.jit(do_not_specialize=["WIDTH", "STRIDE"])
def _mask_tail_kernel(logits, ends, WIDTH, STRIDE, TILE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    end = tl.load(ends + row)
    offsets = tl.arange(0, TILE)
    for start in range(end, WIDTH, TILE):
        column = start + offsets
        tl.store(logits + row * STRIDE + column, -float("inf"), column < WIDTH)


def _adjacent_slab_view(first, last):
    """Alias the span already checked for adjacency by ``_group_layout``."""
    span = (last.storage_offset() - first.storage_offset()) // first.stride(0)
    return first.as_strided((span + last.shape[0], *first.shape[1:]), first.stride())


@dataclass(frozen=True)
class _ScoreGroup:
    first: int
    stop: int
    rows: slice
    width: int
    payload: torch.Tensor
    scale: torch.Tensor
    source_width: int | None = None


def _cp_causal_plan(cp_ctx, widths, rows, ratio):
    """Use host mirrors only when they describe these exact full zigzag rows."""
    if cp_ctx is None:
        return None
    chunks = getattr(cp_ctx, "chunk_lengths_per_req", None)
    fresh = getattr(cp_ctx, "input_lengths_global_host", None)
    prefixes = getattr(cp_ctx, "prefix_lengths_host", None)
    size = getattr(cp_ctx, "cp_size", 0)
    rank = getattr(cp_ctx, "cp_rank", -1)
    if (
        chunks is None
        or fresh is None
        or prefixes is None
        or len(chunks) != len(rows)
        or len(fresh) != len(rows)
        or len(prefixes) != len(rows)
        or tuple(rows) != tuple(chunks)
        or size <= 0
        or not 0 <= rank < size
        or any(
            n <= 0
            or prefix < 0
            or chunk != ((n + 2 * size - 1) // (2 * size)) * 2
            or width != (prefix + n) // ratio
            for width, n, prefix, chunk in zip(widths, fresh, prefixes, chunks)
        )
    ):
        return None
    return tuple(zip(prefixes, fresh, chunks))


def _cp_causal_width(plan, rank, size, request, end, source_width, ratio):
    prefix, fresh, chunk = plan[request]
    half = chunk // 2
    relative = (
        rank * half + end - 1
        if end <= half
        else chunk * size - (rank + 1) * half + end - half - 1
    )
    visible = (prefix + min(relative, fresh - 1) + 1) // ratio
    return min(source_width, max(512, (visible + 255) // 256 * 256))


def _plan_group_layout(widths, rows, adjacent, make_group, *, cp_ctx=None, ratio=1):
    """Plan score widths and row ranges for real or projected KV slabs."""
    request_count = len(widths)
    causal_plan = _cp_causal_plan(cp_ctx, widths, rows, ratio)
    if causal_plan is not None:
        cp_rank, cp_size = cp_ctx.cp_rank, cp_ctx.cp_size
    groups = []
    first, row_start = 0, 0
    while first < request_count:
        if widths[first] == 0:
            row_start += rows[first]
            first += 1
            continue
        stop = first + 1
        last_nonempty = first
        width, group_rows = widths[first], rows[first]
        useful_scores = group_rows * width
        stride = (width + 255) // 256 * 256
        limit = min(_MAX_GROUP_ROWS, _MAX_LOGITS_BYTES // (stride * 4)) // 4 * 4
        if limit == 0:
            return None
        if group_rows > limit:
            offset = 0
            final_probe = None
            while offset < group_rows:
                count = min(limit, group_rows - offset)
                output_width = width
                if causal_plan is not None:
                    remaining = group_rows - offset
                    probe_end = offset + min(_MAX_GROUP_ROWS, remaining)
                    if probe_end == group_rows and final_probe is not None:
                        probe_width, capacity = final_probe
                    else:
                        probe_width = _cp_causal_width(
                            causal_plan,
                            cp_rank,
                            cp_size,
                            first,
                            probe_end,
                            width,
                            ratio,
                        )
                        probe_stride = (probe_width + 255) // 256 * 256
                        capacity = _MAX_LOGITS_BYTES // (probe_stride * 4) // 4 * 4
                        if probe_end == group_rows:
                            final_probe = probe_width, capacity
                    count = min(remaining, _MAX_GROUP_ROWS, capacity)
                    output_width = (
                        probe_width
                        if offset + count == probe_end
                        else _cp_causal_width(
                            causal_plan,
                            cp_rank,
                            cp_size,
                            first,
                            offset + count,
                            width,
                            ratio,
                        )
                    )
                groups.append(
                    make_group(
                        first,
                        stop,
                        slice(row_start + offset, row_start + offset + count),
                        output_width,
                        first,
                        width if causal_plan is not None else None,
                    )
                )
                offset += count
            row_start += group_rows
            first = stop
            continue
        while stop < request_count:
            next_nonempty = stop
            while next_nonempty < request_count and widths[next_nonempty] == 0:
                next_nonempty += 1
            if next_nonempty == request_count:
                break
            if next_nonempty - first >= 128:
                break
            next_width = max(width, widths[next_nonempty])
            next_rows = group_rows + sum(rows[stop : next_nonempty + 1])
            next_useful = useful_scores + rows[next_nonempty] * widths[next_nonempty]
            if (
                next_rows > _MAX_GROUP_ROWS
                or next_rows * next_width > _MAX_LOGITS_INFLATION * next_useful
                or ((next_rows + 3) // 4 * 4) * ((next_width + 255) // 256 * 256) * 4
                > _MAX_LOGITS_BYTES
            ):
                break
            if not adjacent(last_nonempty, next_nonempty):
                break
            group_rows = next_rows
            width = next_width
            useful_scores = next_useful
            last_nonempty = next_nonempty
            stop = next_nonempty + 1
        groups.append(
            make_group(
                first,
                stop,
                slice(row_start, row_start + group_rows),
                width,
                last_nonempty,
            )
        )
        row_start += group_rows
        first = stop
    return tuple(groups)


def _group_layout(keys, rows, *, cp_ctx=None, ratio=1):
    """Group adjacent requests and chunk oversized Q ranges without K copies."""
    if (
        not 1 <= len(keys) <= _MAX_REQUESTS
        or len(rows) != len(keys)
        or any(n <= 0 for n in rows)
    ):
        return None
    layer_keys = tuple((key.quant, key.scale) for key in keys)
    widths = tuple(len(key) for key in keys)
    device = layer_keys[0][0].device
    request_count = len(keys)
    for (quant, scale), width in zip(layer_keys, widths):
        if width == 0:
            continue
        if (
            width * request_count >= 2**31
            or quant.shape != (width, 64)
            or scale.shape != (width,)
            or quant.dtype != torch.int8
            or scale.dtype != torch.int32
            or quant.device != device
            or scale.device != device
            or not quant.is_contiguous()
            or not scale.is_contiguous()
            or quant.data_ptr() % 16
            or scale.data_ptr() % 16
        ):
            return None

    def adjacent(left_idx, right_idx):
        padded = (widths[left_idx] + 255) // 256 * 256
        return all(
            left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
            and left.storage_offset() + padded * left.stride(0)
            == right.storage_offset()
            for left, right in zip(layer_keys[left_idx], layer_keys[right_idx])
        )

    def make_group(first, stop, rows, width, last, source_width=None):
        payload, scale = layer_keys[first]
        if first != last:
            payload = _adjacent_slab_view(payload, layer_keys[last][0])
            scale = _adjacent_slab_view(scale, layer_keys[last][1])
        return _ScoreGroup(first, stop, rows, width, payload, scale, source_width)

    return _plan_group_layout(
        widths, rows, adjacent, make_group, cp_ctx=cp_ctx, ratio=ratio
    )


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

    def split_rows(self, tensor):
        """Make views for this call only; never cache layer activations."""
        return (tensor[group.rows] for group in self.layout)

    def _bounds(self):
        # This cache is cleared at the existing forward boundary. Identity also
        # guards replacement of positions with an equal-shaped tensor mid-forward.
        cache = self.shared.setdefault("prefill_score_bounds", {})
        layout = tuple(
            (g.first, g.stop, g.rows.start, g.rows.stop, g.width, g.source_width)
            for g in self.layout
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
            _MIN_BOUNDS_GROUPS <= len(self.layout) <= _MAX_BOUNDS_GROUPS
            and not prefill_stream_is_capturing()
        ):
            bounds = self._fused_bounds(self.layout)
            if bounds is not None:
                cache[key] = (self.positions, self.req_ids, self.key_counts, bounds)
                return bounds
        bounds = [self._one_group_bounds(group) for group in self.layout]
        cache[key] = (self.positions, self.req_ids, self.key_counts, tuple(bounds))
        return cache[key][3]

    def _fused_bounds(self, layout):
        description = _bounds_descriptors(layout)
        if description is None:
            return None
        descriptors, tiles, total_rows = description
        origin = layout[0].rows.start
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        # Every plane needs an aligned pointer for DeepGEMM. Odd group rows
        # retain the aligned four-plane packing for this allocation.
        planes = 3 if all(d[4] % 4 == 0 for d in descriptors) else 4
        output = torch.empty(
            planes * total_rows, dtype=torch.int32, device=self.q.device
        )
        native = getattr(rtp_llm_ops, "dsv41_grouped_score_bounds", None)
        if (
            planes == 3
            and self.q.is_cuda
            and native is not None
            and self.key_counts.dtype == torch.int32
            and all(d[5] < 2**31 for d in descriptors)
        ):
            native(
                torch.tensor(descriptors, dtype=torch.int32),
                self.positions[origin:],
                self.req_ids[origin:],
                self.key_counts,
                output,
                self.ratio,
            )
        else:
            host = torch.tensor(descriptors, dtype=torch.int64, pin_memory=True)
            device = host.to(self.q.device, non_blocking=True)
            _all_grouped_score_bounds_kernel[(tiles,)](
                self.positions[origin:],
                self.req_ids[origin:],
                self.key_counts,
                device,
                output,
                self.key_counts.numel(),
                len(layout),
                self.ratio,
                planes,
                num_warps=4,
            )
        vectors = output.split(
            tuple(g.rows.stop - g.rows.start for g in layout for _ in range(planes))
        )
        return tuple(
            (g.rows, vectors[i], vectors[i + 1], None, vectors[i + planes - 1])
            for i, g in zip(range(0, len(vectors), planes), layout)
        )

    def _one_group_bounds(self, group):
        rows = group.rows
        count = rows.stop - rows.start
        planes = 3 if count % 4 == 0 else 4
        output = torch.empty((planes, count), dtype=torch.int32, device=self.q.device)
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
            group.source_width or group.width,
            self.ratio,
            planes,
            num_warps=4,
        )
        return rows, output[0], output[1], None, output[-1]

    def _schedule_cache(self):
        stream = torch.cuda.current_stream(self.q.device)
        return self.shared.setdefault("prefill_score_bounds", {}).setdefault(
            ("mqa_schedule", self.q.device, stream.cuda_stream), {}
        )

    def try_native_topk(self, out, builder):
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        from . import _indexer_score as score
        from . import _v41_prefill_deepselect as deepselect
        from .indexer import _get_topk_workspace

        submit = getattr(rtp_llm_ops, "dsv41_submit_grouped_k512", None)
        if (
            submit is None
            or not 0 < len(self.layout) <= _MAX_BOUNDS_GROUPS
            or self.weights.dtype != torch.bfloat16
            or out.shape != (self.q.shape[0], 512)
            or out.dtype != torch.int32
            or out.device != self.q.device
            or not out.is_contiguous()
            or not all(t.is_contiguous() for t in (self.q, self.sf, self.weights))
        ):
            return False
        operation = score._deep_gemm.fp8_fp4_mqa_logits
        if score._mqa_accepts_logits_dtype(operation):
            return False
        offset = 0
        for group in self.layout:
            if (
                group.rows.start != offset
                or not 512 <= group.width <= 2**31 - 1
                or not group.payload.is_contiguous()
                or not group.scale.is_contiguous()
            ):
                return False
            offset = group.rows.stop
        if offset != out.shape[0]:
            return False

        cache = self._schedule_cache()
        heads = self.q.shape[1]
        groups = [
            (
                g.payload,
                g.scale,
                starts,
                ends,
                visible,
                score.balanced_mqa_schedule(
                    builder, starts, ends, g.payload.shape[0], heads, cache
                ),
                g.rows.stop,
                g.width,
            )
            for g, (_, starts, ends, _, visible) in zip(
                self.layout, self._bounds(), strict=True
            )
        ]
        submit(
            operation,
            self.q,
            self.sf,
            self.weights,
            out,
            _get_topk_workspace(out.device),
            groups,
            deepselect.is_available(out.device, torch.bfloat16),
        )
        return True

    def groups(self, *, mask_tail=True, return_bf16=False, balance_kv_tiles=False):
        """Yield (packed row slice, logits, visible, (None, visible)).

        The conservative FP32 budget (default 256 MiB) also bounds BF16 output.
        Invalid suffixes are -inf by default;
        callers disabling masking must use visible bounds or mask before use.
        Columns are request-local; candidate/TopK output needs no offset change.
        Consume and delete logits before advancing, including derived views:
        retaining them retains their whole group allocation. No logits are cached.
        """
        from ._indexer_score import fp8_fp4_mqa_indexer_score

        score_options = {"balance_kv_tiles": True} if balance_kv_tiles else {}
        # Ratio-2 source layers share bounds. L20 is the only ratio-1 source
        # using balanced scheduling and has no later consumer of its schedule.
        if (
            balance_kv_tiles
            and self.ratio == 2
            and len(self.layout) <= _MAX_BOUNDS_GROUPS
        ):
            score_options["schedule_cache"] = self._schedule_cache()
        bounds_by_group = self._iter_bounds()
        for group, bounds, q, sf, weights in zip(
            self.layout,
            bounds_by_group,
            self.split_rows(self.q),
            self.split_rows(self.sf),
            self.split_rows(self.weights),
        ):
            rows, starts, ends, zeros, visible = bounds
            logits = fp8_fp4_mqa_indexer_score(
                q,
                sf,
                group.payload,
                group.scale,
                weights,
                starts,
                ends,
                clean_logits=False,
                max_seqlen_k=group.width,
                return_bf16=return_bf16,
                **score_options,
            )
            if mask_tail:
                _mask_tail_kernel[(logits.shape[0],)](
                    logits, visible, group.width, logits.stride(0), 256
                )
            yield rows, logits, visible, (zeros, visible)
            del logits

    def _iter_bounds(self):
        # Layouts above the cache limit stream bounds in bounded chunks.
        if len(self.layout) <= _MAX_BOUNDS_GROUPS:
            yield from self._bounds()
            return
        for first in range(0, len(self.layout), _MAX_BOUNDS_GROUPS):
            chunk = self.layout[first : first + _MAX_BOUNDS_GROUPS]
            bounds = (
                self._fused_bounds(chunk)
                if len(chunk) >= 2 and not prefill_stream_is_capturing()
                else None
            )
            if bounds is not None:
                yield from bounds
            else:
                for group in chunk:
                    yield self._one_group_bounds(group)


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
    cp_ctx=None,
):
    """Return a lazy plan or None before GPU work for unsupported layouts.

    Accept already-prepared weights from the original full-M BF16 head
    projection; never reproject or fold FP4 scales into them. Candidate tables
    and CED projection in shared are intentionally allowed: callers consume
    scores after this boundary. Forward-local positions must remain immutable.
    """
    count = len(globals_by_req)
    if (
        not 1 <= count <= _MAX_REQUESTS
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
        or weights.dtype not in (torch.float32, torch.bfloat16)
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
    layout = _group_layout(keys, rows, cp_ctx=cp_ctx, ratio=ratio)
    if layout is None or keys[0].quant.device != q.device:
        return None
    # TopK/candidate IDs are initialized by Q preparation. An optional
    # candidate bitmap is not, so it cannot leave standalone zero-key rows.
    if shared.get("prefill_candidate_mask") is not None:
        cursor = 0
        for group in layout:
            if group.rows.start != cursor:
                return None
            cursor = group.rows.stop
        if cursor != q.shape[0]:
            return None
    if not all(
        isinstance(value, torch.Tensor)
        and value.device == q.device
        and value.shape == shape
        and value.dtype in (torch.int32, torch.int64)
        and value.is_contiguous()
        for value, shape in ((req_ids, (q.shape[0],)), (key_counts, (count,)))
    ):
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
