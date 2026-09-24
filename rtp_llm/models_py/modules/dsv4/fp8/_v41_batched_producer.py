# SPDX-License-Identifier: Apache-2.0
"""Group-local V4.1 producer epilogues with the original arithmetic partitions.

Integration order: make_plan/prepare once per forward, compress_main for each
restored FP32 group, original compact-M index projections into a contiguous
group slab, then store_index. store_states may follow compress_main; all
compression reads the immutable request predecessor snapshot, never the ring.
Groups must contain whole requests and retain sequence order. All slot
mappings must be non-aliasing. Normal STATE tables have distinct physical
blocks per writable logical column; the suffix guard keeps at most one ring
per column. A pure host proof is available for callers with host tables.

No GEMM is merged here. The caller owns projection numerical equivalence and
the whole-layer launch budget. Enable only after focused GPU verification with
RTP_V41_BATCHED_PRODUCER=1. Unsupported inputs return None/False before writes.
"""

from __future__ import annotations

import ctypes
import math
import os
from array import array
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.attn_type import CSA_STATE, INDEXER_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_global as legacy
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)

_MAX_GROUP_ROWS = 65536
_MAX_PACK_PROJECTIONS = 512


@dataclass(frozen=True)
class GroupPlan:
    start: int
    end: int
    first: int
    stop: int
    ratio: int
    # (group token start/end, compact start/stop, boundary phase)
    segments: tuple

    @property
    def rows(self):
        return self.end - self.start

    @property
    def count(self):
        return self.stop - self.first


@dataclass(frozen=True)
class PreparedPlan:
    host: GroupPlan
    boundaries: torch.Tensor
    widths: torch.Tensor


def reduction_width(count):
    """The original compact-M reduction partition, including empty segments."""
    if count < 0:
        raise ValueError("Negative compact count")
    return 0 if count == 0 else min(128, 512 // min(1 << (count.bit_length() - 1), 16))


def make_plan(metadata, group_start, group_end, *, ratio, request_lengths):
    """Validate whole-request/whole-segment boundaries using host metadata only.

    ``metadata`` is the unchanged ProducerMetadata. Slice its compact vectors
    with [plan.first:plan.stop] and state_slots with [plan.start:plan.end].
    plan.segments supplies the original compact slices for index GEMMs.
    """
    if ratio not in (1, 2) or not 0 < group_end - group_start <= _MAX_GROUP_ROWS:
        return None
    request_ends = {0}
    total = 0
    for length in request_lengths:
        if length <= 0:
            return None
        total += length
        request_ends.add(total)
    if group_start not in request_ends or group_end not in request_ends:
        return None
    if metadata is None or group_start not in metadata.segments:
        return None
    cursor = group_start
    first = metadata.segments[cursor][1]
    stop = first
    segments = []
    while cursor < group_end:
        segment = metadata.segments.get(cursor)
        if segment is None:
            return None
        end, begin, compact_end, phase = segment
        count = compact_end - begin
        if (
            not cursor < end <= group_end
            or begin != stop
            or not 0 <= phase < ratio
            or count != (end - cursor + ratio - 1 - phase) // ratio
            or any(cursor < request_end < end for request_end in request_ends)
        ):
            return None
        segments.append(
            (
                cursor - group_start,
                end - group_start,
                begin - first,
                compact_end - first,
                phase,
            )
        )
        cursor, stop = end, compact_end
    return GroupPlan(group_start, group_end, first, stop, ratio, tuple(segments))


def is_supported(tensor):
    return (
        os.environ.get("RTP_V41_BATCHED_PRODUCER", "0") == "1"
        and isinstance(tensor, torch.Tensor)
        and legacy._enabled(tensor)
    )


def can_batch_groups(attention, x, cp):
    """Preflight complete CP groups before constructing the group iterator.

    The forward registers (region_names, CPU physical tables) under
    ``prefill_producer_host_tables``. A True result certifies state suffix
    uniqueness for this batch, so its stores may pass slots_are_unique=True.
    Do not reuse this result across forwards or changed cache bindings.
    Main/index additionally reject shared writable physical pages: unlike
    the old sequential segments, a group launches all writers concurrently.
    No device contents are read and no cache data is mutated here.
    """
    if not is_supported(x) or x.ndim != 2 or x.dtype != torch.bfloat16:
        return False
    if not x.is_contiguous() or (torch.is_grad_enabled() and x.requires_grad):
        return False
    lengths = getattr(cp, "input_lengths_global_host", None)
    prefixes = getattr(cp, "prefix_lengths_host", None)
    if (
        getattr(cp, "cp_size", None) != 4
        or not isinstance(lengths, (tuple, list))
        or not isinstance(prefixes, (tuple, list))
        or not 2 <= len(lengths) <= 128
        or len(prefixes) != len(lengths)
        or any(not isinstance(n, int) or n <= 0 for n in lengths)
        or any(not isinstance(p, int) or p < 0 for p in prefixes)
    ):
        return False
    ratio = getattr(attention, "compress_ratio", None)
    if ratio not in (1, 2):
        return False
    rows = sum(lengths)
    count = sum((p + n) // ratio - p // ratio for p, n in zip(prefixes, lengths))
    # ProducerMetadata needs max_segment_rows <= rows. This upper bound never
    # accepts a layout its 16MiB limit would reject, without constructing tiles.
    metadata_bytes = 8 * (
        (4 * count + 2 * rows if ratio == 2 else 3 * rows) + len(lengths)
    )
    if count == 0 or metadata_bytes > 16 * 1024**2:
        return False
    try:
        cache = attention._kv_cache
        registered = attention._shared_attention.get("prefill_producer_host_tables")
        if cache is None or registered is None:
            return False
        regions, host = registered
        regions = tuple(map(int, regions))
        if (
            regions != tuple(map(int, cache.group_region_names))
            or not isinstance(host, torch.Tensor)
            or host.device.type != "cpu"
            or host.dtype not in (torch.int32, torch.int64)
            or host.ndim != 3
            or host.shape[:2] != (len(regions), len(lengths))
            or host.shape[2] == 0
        ):
            return False
        device = x.device
        owner = attention._owner()
        if not (
            math.isfinite(attention.eps)
            and attention.eps > 0
            and legacy._vector(
                owner.global_norm, 512, device, (torch.float32, torch.bfloat16)
            )
            and legacy._vector(
                owner.index_k_norm, 128, device, (torch.float32, torch.bfloat16)
            )
            and legacy._matrix(owner.index_wk, 128, 512, torch.bfloat16, device)
            and legacy._frequencies(attention.freqs_cis, device)
            and max(p + n for p, n in zip(prefixes, lengths))
            <= attention.freqs_cis.shape[0]
        ):
            return False
        cp_size = 4 if getattr(cp, "kv_cache_sharded", False) else 1
        cp_rank = cp.cp_rank if cp_size == 4 else 0
        if not 0 <= cp_rank < cp_size:
            return False
        owner_tpb = cache.seq_size_per_block
        if owner_tpb <= 0:
            return False
        for region, width in ((attention._global_region(), 288), (INDEXER_KV, 68)):
            if regions.count(region) != 1:
                return False
            pool = attention._source_pool(region)
            table = attention._block_tables_by_type.get(region)
            if not legacy._pool(pool, device, width) or not (
                isinstance(table, torch.Tensor)
                and table.ndim == 2
                and table.shape[0] == len(lengths)
                and table.shape[1] > 0
                and table.device == device
                and table.dtype in (torch.int32, torch.int64)
            ):
                return False
            tpb = require_pool_tokens_per_block(cache, region=region)
            if tpb <= 0 or owner_tpb % tpb or tpb % ratio:
                return False
            bpk = owner_tpb // tpb
            # FULL pages expand physical ID b to b*bpk+j. Distinct writable
            # physical IDs imply distinct byte rows even when bpk > 1.
            seen = set()
            for blocks, prefix, length in zip(
                host[regions.index(region)].tolist(), prefixes, lengths
            ):
                end = prefix + length
                for column in range(prefix // owner_tpb, (end - 1) // owner_tpb + 1):
                    if column % cp_size != cp_rank:
                        continue
                    begin = max(prefix, column * owner_tpb)
                    begin += (ratio - 1 - begin) % ratio
                    if begin >= min(end, (column + 1) * owner_tpb):
                        continue
                    local_column = column // cp_size
                    if (
                        local_column >= len(blocks)
                        or (local_column + 1) * bpk > table.shape[1]
                    ):
                        return False
                    block = blocks[local_column]
                    if block <= 0:
                        continue
                    if block in seen or (block + 1) * bpk > pool.shape[0]:
                        return False
                    seen.add(block)
        if ratio == 2:
            if regions.count(CSA_STATE) != 1:
                return False
            state = attention._source_pool(CSA_STATE)
            table = attention._block_tables_by_type.get(CSA_STATE)
            if not (
                isinstance(state, torch.Tensor)
                and state.ndim == 2
                and state.shape[0] > 0
                and legacy._matrix(state, state.shape[0], 1024, torch.float32, device)
                and isinstance(table, torch.Tensor)
                and table.ndim == 2
                and table.shape[0] == len(lengths)
                and table.shape[1] >= host.shape[2]
                and table.device == device
                and table.dtype in (torch.int32, torch.int64)
            ):
                return False
            entries = attention._source_entries(CSA_STATE, state)
            tpb = require_pool_tokens_per_block(cache, region=CSA_STATE)
            state_host = host[regions.index(CSA_STATE)]
            if entries <= 0 or state.shape[0] % entries:
                return False
            if int(state_host.max()) >= state.shape[0] // entries:
                return False
            if not state_slots_are_unique(
                state_host, prefixes, lengths, entries, tpb, cp_size, cp_rank
            ):
                return False
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RuntimeError):
        # Missing/unbound descriptors are an unsupported preflight, not a
        # reason to partially execute a group and then fall back.
        return False
    return True


def state_slots_are_unique(
    table,
    prefixes,
    lengths,
    entries_per_block,
    tokens_per_block,
    cp_size=4,
    cp_rank=0,
):
    """Prove the existing suffix mapping injective without reading GPU data.

    Pass the CPU STATE table for exactly these requests, before any producer
    mutation. Allocator tables retain logical columns (including -1 holes).
    Truncated/wrapping tables and aliased writable blocks are unsupported.
    Read-only prefix blocks may be shared: only fresh suffix writes matter.
    """
    if isinstance(table, torch.Tensor):
        if (
            table.device.type != "cpu"
            or table.ndim != 2
            or table.dtype not in (torch.int32, torch.int64)
        ):
            return False
        table = table.tolist()
    if isinstance(prefixes, torch.Tensor):
        if prefixes.device.type != "cpu" or prefixes.ndim != 1:
            return False
        prefixes = prefixes.tolist()
    if isinstance(lengths, torch.Tensor):
        if lengths.device.type != "cpu" or lengths.ndim != 1:
            return False
        lengths = lengths.tolist()
    if (
        len(table) != len(prefixes)
        or len(prefixes) != len(lengths)
        or entries_per_block <= 0
        or tokens_per_block <= 0
        or cp_size <= 0
        or not 0 <= cp_rank < cp_size
    ):
        return False
    seen = set()
    ring = entries_per_block * cp_size
    for blocks, prefix, length in zip(table, prefixes, lengths):
        end = prefix + length
        if prefix < 0 or length <= 0 or end > len(blocks) * tokens_per_block:
            return False
        for column in range(
            prefix // tokens_per_block, (end - 1) // tokens_per_block + 1
        ):
            block = blocks[column]
            if block <= 0:
                continue
            effective_end = min((column + 1) * tokens_per_block, end)
            begin = max(prefix, column * tokens_per_block, effective_end - ring)
            for position in range(begin, effective_end):
                offset = position % ring
                if offset // entries_per_block != cp_rank:
                    continue
                slot = block * entries_per_block + offset % entries_per_block
                if slot in seen:
                    return False
                seen.add(slot)
    return True


@triton.jit
def _prepare_kernel(
    descriptors, boundaries, widths, RATIO: tl.constexpr, BLOCK: tl.constexpr
):
    segment = tl.program_id(1)
    start = tl.load(descriptors + segment * 5)
    first = tl.load(descriptors + segment * 5 + 1)
    count = tl.load(descriptors + segment * 5 + 2)
    phase = tl.load(descriptors + segment * 5 + 3)
    width = tl.load(descriptors + segment * 5 + 4)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(
        boundaries + first + offset, start + phase + offset * RATIO, offset < count
    )
    tl.store(widths + first + offset, width, offset < count)


def prepare(plan, device):
    """Materialize integers once; retain the result across same-forward layers."""
    if not isinstance(plan, GroupPlan):
        return None
    boundaries = torch.empty(plan.count, dtype=torch.int64, device=device)
    if not is_supported(boundaries):
        return None
    widths = torch.empty(plan.count, dtype=torch.int32, device=device)
    if plan.count:
        descriptors = [
            (start, first, stop - first, phase, reduction_width(stop - first))
            for start, _, first, stop, phase in plan.segments
        ]
        descriptors_device = torch.tensor(descriptors, dtype=torch.int64, device=device)
        maximum = max(stop - first for _, _, first, stop, _ in plan.segments)
        _prepare_kernel[(triton.cdiv(maximum, 256), len(plan.segments))](
            descriptors_device, boundaries, widths, plan.ratio, 256
        )
    return PreparedPlan(plan, boundaries, widths)


@triton.jit(do_not_specialize=["POOL_STRIDE", "BLOCKS"])
def _compress_kernel(
    values,
    scores,
    weight,
    positions,
    requests,
    starts,
    previous,
    boundaries,
    widths,
    frequencies,
    pool,
    slots,
    output,
    VALUE_STRIDE: tl.constexpr,
    SCORE_STRIDE: tl.constexpr,
    PREVIOUS_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    EPS: tl.constexpr,
    ENTRIES: tl.constexpr,
    POOL_STRIDE,
    BLOCKS,
    TRAP: tl.constexpr,
):
    width = tl.load(widths + tl.program_id(0))
    # The callee sees the same compact row program_id. Only its reduction
    # partition varies; values/boundaries address the whole immutable group.
    for i in tl.static_range(3):
        if width == (128 >> i):
            legacy._prefill_compress_main_kernel(
                values,
                scores,
                weight,
                positions,
                requests,
                starts,
                previous,
                boundaries,
                None,
                None,
                frequencies,
                pool,
                slots,
                output,
                VALUE_STRIDE,
                SCORE_STRIDE,
                PREVIOUS_STRIDE,
                False,
                RATIO,
                EPS,
                ENTRIES,
                POOL_STRIDE,
                BLOCKS,
                TRAP,
                128 >> i,
            )


def compress_main(
    values,
    scores,
    norm,
    eps,
    positions,
    req_ids,
    starts,
    previous,
    prepared,
    freqs,
    main_pool,
    main_slots,
):
    """Return compact BF16 latent with unchanged per-segment RMS arithmetic.

    values/scores and positions/req_ids are group-local; starts/previous retain
    global request indexing. There is no carry clone or state read. The first
    token of each request uses previous, all other pairs use adjacent rows.
    """
    if not is_supported(values) or not isinstance(prepared, PreparedPlan):
        return None
    plan = prepared.host
    rows, count, ratio, device = plan.rows, plan.count, plan.ratio, values.device
    if not (
        math.isfinite(eps)
        and eps > 0
        and legacy._matrix(values, rows, 512, torch.float32, device)
        and legacy._vector(norm, 512, device, (torch.float32, torch.bfloat16))
        and legacy._vector(positions, rows, device)
        and legacy._vector(prepared.boundaries, count, device)
        and legacy._vector(prepared.widths, count, device, (torch.int32,))
        and legacy._vector(main_slots, count, device)
        and legacy._frequencies(freqs, device)
        and legacy._pool(main_pool, device, 288)
    ):
        return None
    if ratio == 2 and not (
        legacy._matrix(scores, rows, 512, torch.float32, device)
        and legacy._vector(req_ids, rows, device)
        and isinstance(starts, torch.Tensor)
        and legacy._vector(starts, starts.numel(), device)
        and legacy._matrix(previous, starts.numel(), 1024, torch.float32, device)
    ):
        return None
    output = torch.empty((count, 512), dtype=torch.bfloat16, device=device)
    if count:
        _compress_kernel[(count,)](
            values,
            scores,
            norm,
            positions,
            req_ids,
            starts,
            previous,
            prepared.boundaries,
            prepared.widths,
            freqs.view(torch.float32),
            main_pool,
            main_slots,
            output,
            values.stride(0),
            scores.stride(0) if ratio == 2 else 0,
            previous.stride(0) if ratio == 2 else 0,
            ratio,
            eps,
            main_pool.shape[1],
            main_pool.stride(0),
            main_pool.shape[0],
            legacy.trap_invalid_kv_access_enabled(),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return output


def store_states(values, scores, state_slots, state_pool, *, slots_are_unique=False):
    """One original state-store launch for a proven non-aliasing suffix map.

    Check state_slots_are_unique on host metadata (or an equivalent allocator
    invariant) before compress_main writes any pool. If not proven, use the
    original segment producer for the entire group. This function does not
    inspect GPU slot values or allocate winner scratch.
    """
    if not slots_are_unique or not is_supported(values):
        return False
    return legacy.store_states(values, scores, state_slots, state_pool)


def store_index(projected, norm, eps, positions, freqs, index_pool, index_slots, ratio):
    """One group epilogue after original compact-M projections are packed.

    Require a contiguous BF16 slab, matching each old contiguous GEMM result's
    NATIVE_NORM selection even for one-row segments. No conversion is applied.
    """
    if not is_supported(projected) or not projected.is_contiguous():
        return False
    return legacy.store_index(
        projected, norm, eps, positions, freqs, index_pool, index_slots, ratio
    )


@triton.jit(do_not_specialize=["COUNT", "SEND_STRIDE"])
def _pack_projected_kernel(table, send, COUNT, SEND_STRIDE, BLOCK: tl.constexpr):
    task = tl.program_id(0)
    # Prefix tile counts schedule only useful work for skewed segment lengths.
    lo, hi = tl.full((), 0, tl.int32), COUNT
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        first = tl.load(table + mid * 5 + 4)
        if first <= task:
            lo = mid
        else:
            hi = mid
    source = tl.load(table + lo * 5).to(tl.pointer_type(tl.uint32))
    rows = tl.load(table + lo * 5 + 1)
    offset = tl.load(table + lo * 5 + 2)
    column = tl.load(table + lo * 5 + 3)
    first = tl.load(table + lo * 5 + 4)
    element = (task - first) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(source + element, element < rows * 512, other=0)
    destination = send.to(tl.pointer_type(tl.uint32))
    tl.store(
        destination + (offset + element // 512) * SEND_STRIDE + column + element % 512,
        bits,
        element < rows * 512,
    )


def pack_projected_group(send, projections, offsets, columns):
    """Pack contiguous FP32 [M,512] sources into a pre-zeroed group in one launch.

    offsets are send-relative row offsets; columns are element offsets (0 or
    512), not column ordinals. Destination rectangles must not overlap. Invalid
    host layouts return False before any upload/write. Empty sources are skipped;
    empty input succeeds without launching. Padding retains its incoming bits.

    At most 512 nonempty sources use a five-int64 descriptor each: 20KiB pinned
    host storage and one nonblocking H2D. No cache retains tensors. Callers must
    establish cross-stream input dependencies; record_stream protects lifetimes.
    """
    if not is_supported(send) or send.ndim != 2 or send.shape[1] not in (512, 1024):
        return False
    if not legacy._matrix(
        send, send.shape[0], send.shape[1], torch.float32, send.device
    ):
        return False
    if not all(isinstance(v, (tuple, list)) for v in (projections, offsets, columns)):
        return False
    if not len(projections) == len(offsets) == len(columns):
        return False
    send_rows, send_width = send.shape
    device, send_stride = send.device, send.stride(0)
    send_begin = send.data_ptr()
    send_end = (
        send_begin + ((send_rows - 1) * send_stride + send_width) * 4
        if send_rows
        else send_begin
    )
    grad_enabled = torch.is_grad_enabled()
    descriptors, ranges, tile_count = [], {0: [], 512: []}, 0
    for source, offset, column in zip(projections, offsets, columns):
        if not isinstance(source, torch.Tensor):
            return False
        shape = source.shape
        if (
            len(shape) != 2
            or shape[1] != 512
            or source.dtype != torch.float32
            or source.device != device
            or not source.is_contiguous()
            or (grad_enabled and source.requires_grad)
            or type(offset) is not int
            or type(column) is not int
            or column not in (0, 512)
            or column + 512 > send_width
            or not 0 <= offset <= offset + shape[0] <= send_rows
        ):
            return False
        rows, pointer = shape[0], source.data_ptr()
        if rows:
            if pointer < send_end and send_begin < pointer + rows * 512 * 4:
                return False
            ranges[column].append((offset, offset + rows))
            descriptors.extend((pointer, rows, offset, column, tile_count))
            tile_count += (rows * 512 + 2047) // 2048
    count = len(descriptors) // 5
    if count > _MAX_PACK_PROJECTIONS:
        return False
    for intervals in ranges.values():
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            return False
    if not descriptors:
        return True
    # Populate pinned memory as integers directly, avoiding torch.tensor's
    # per-scalar conversion overhead. Both buffers are bounded to 20KiB.
    packed = array("q", descriptors)
    host = torch.empty(
        len(descriptors), dtype=torch.int64, device="cpu", pin_memory=True
    )
    ctypes.memmove(
        host.data_ptr(), packed.buffer_info()[0], len(packed) * packed.itemsize
    )
    table = host.to(send.device, non_blocking=True)
    _pack_projected_kernel[(tile_count,)](
        table, send, count, send_stride, 2048, num_warps=4
    )
    stream = torch.cuda.current_stream(send.device)
    for source in projections:
        source.record_stream(stream)
    send.record_stream(stream)
    return True


@triton.jit(
    do_not_specialize=[
        "LOCAL_START",
        "LOCAL_ROWS",
        "FULL_CHUNK",
        "REAL_START",
        "REAL_ROWS",
        "INPUT_STRIDE",
        "OUTPUT_STRIDE",
        "MAP_STRIDE",
    ]
)
def _restore_projected_kernel(
    gathered,
    restore,
    output,
    LOCAL_START,
    LOCAL_ROWS,
    FULL_CHUNK,
    REAL_START,
    REAL_ROWS,
    INPUT_STRIDE,
    OUTPUT_STRIDE,
    MAP_STRIDE,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    element = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    row, column = element // WIDTH, element % WIDTH
    valid = row < REAL_ROWS
    index = tl.load(restore + (REAL_START + row) * MAP_STRIDE, valid, other=0).to(
        tl.int64
    )
    rank, local = index // FULL_CHUNK, index % FULL_CHUNK
    source_row = rank * LOCAL_ROWS + local - LOCAL_START
    mapped = (
        (rank >= 0)
        & (rank < 4)
        & (local >= LOCAL_START)
        & (local < LOCAL_START + LOCAL_ROWS)
    )
    if tl.sum((valid & ~mapped).to(tl.int32), 0) > 0:
        legacy._trap()
    bits = tl.load(
        gathered.to(tl.pointer_type(tl.uint32)) + source_row * INPUT_STRIDE + column,
        valid & mapped,
        other=0,
    )
    tl.store(
        output.to(tl.pointer_type(tl.uint32)) + row * OUTPUT_STRIDE + column,
        bits,
        valid,
    )


def restore_projected_group(
    gathered,
    unpad_restore,
    local_start,
    local_rows,
    full_chunk,
    real_start,
    real_rows,
    *,
    out=None,
):
    """Restore a CP4 rank-major FP32 group in one integer-addressed copy kernel.

    unpad_restore retains the full-forward rank*full_chunk+local mapping. Its
    selected entries must refer to this local interval; violations trap on GPU,
    without a host readback. Optional out supports padded strides for testing.
    Unsupported metadata returns None before mutation; an empty result launches
    nothing. Only the finite width modes 512/1024 are constexpr.
    """
    if (
        not is_supported(gathered)
        or gathered.ndim != 2
        or gathered.shape[1] not in (512, 1024)
    ):
        return None
    if not all(
        type(v) is int
        for v in (local_start, local_rows, full_chunk, real_start, real_rows)
    ):
        return None
    if not (
        full_chunk > 0
        and 0 <= local_start <= local_start + local_rows <= full_chunk
        and 0 <= real_rows <= min(4 * local_rows, _MAX_GROUP_ROWS)
        and real_start >= 0
    ):
        return None
    width, device = gathered.shape[1], gathered.device
    if not (
        legacy._matrix(gathered, 4 * local_rows, width, torch.float32, device)
        and isinstance(unpad_restore, torch.Tensor)
        and unpad_restore.ndim == 1
        and unpad_restore.device == device
        and unpad_restore.dtype in (torch.int32, torch.int64)
        and unpad_restore.stride(0) > 0
        and real_start + real_rows <= unpad_restore.numel()
    ):
        return None
    if out is not None and not (
        legacy._matrix(out, real_rows, width, torch.float32, device)
        and not torch._C._overlaps(out, gathered)
        and not torch._C._overlaps(out, unpad_restore)
    ):
        return None
    if out is None:
        out = torch.empty((real_rows, width), dtype=torch.float32, device=device)
    if real_rows:
        _restore_projected_kernel[(triton.cdiv(real_rows * width, 2048),)](
            gathered,
            unpad_restore,
            out,
            local_start,
            local_rows,
            full_chunk,
            real_start,
            real_rows,
            gathered.stride(0),
            out.stride(0),
            unpad_restore.stride(0),
            width,
            2048,
            num_warps=4,
        )
        stream = torch.cuda.current_stream(device)
        for tensor in (gathered, unpad_restore, out):
            tensor.record_stream(stream)
    return out


def warmup_projected_groups(device):
    """Warm finite pointer dtype/alignment and width modes on private buffers."""
    for width in (512, 1024):
        send = torch.zeros((9, width), dtype=torch.float32, device=device)
        if not is_supported(send):
            return False
        sources = [
            torch.zeros((n, 512), dtype=torch.float32, device=device) for n in (1, 3)
        ]
        offsets, columns = [1, 4], [0, width - 512]
        if not pack_projected_group(send, sources, offsets, columns):
            return False
        gathered = torch.zeros((16, width), dtype=torch.float32, device=device)
        for dtype in (torch.int32, torch.int64):
            for offset in (0, 1):
                mapping = torch.arange(6, dtype=dtype, device=device)[
                    offset : offset + 4
                ]
                # full_chunk=8, selected local interval [1,5), rank zero.
                mapping.fill_(2)
                if restore_projected_group(gathered, mapping, 1, 4, 8, 0, 4) is None:
                    return False
    return True
