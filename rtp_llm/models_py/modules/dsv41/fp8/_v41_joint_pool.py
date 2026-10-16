"""Startup-warmed pool readback; the FP4 codec remains the only format owner.

Unsupported common geometry returns None before allocation or transport.
Qualified but cold/invalid local state raises, never selecting a different
collective schedule. Readiness stores compiled kernels only, never tensors.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv41.fp8._kv_cache_utils import (
    require_pool_tokens_per_block,
)

_PAYLOAD = codec.FP4_INDEXER_HEAD_DIM // 2
_SCALE = codec.FP4_INDEXER_HEAD_DIM // codec.FP4_INDEXER_GROUP
_RAW = codec.FP4_GLOBAL_ENTRY_BYTES
_PACKED = _PAYLOAD + _SCALE + _RAW
# CP4 performance is qualified at B32; apply this to the batch, not group tails.
_MIN_BATCH = 32
_READY = {}
_LOCK = threading.Lock()
_INDEX_DIM = tl.constexpr(codec.FP4_INDEXER_HEAD_DIM)
_INDEX_GROUP = tl.constexpr(codec.FP4_INDEXER_GROUP)
_GLOBAL_DIM = tl.constexpr(codec.FP4_GLOBAL_HEAD_DIM)
_GLOBAL_GROUP = tl.constexpr(codec.FP4_GLOBAL_GROUP)
_RAW_C = tl.constexpr(_RAW)
_PAYLOAD_C = tl.constexpr(_PAYLOAD)


@dataclass(frozen=True)
class PoolLayout:
    tokens_per_block: int
    entries_per_block: int
    owner_tokens_per_block: int
    table_dtype: torch.dtype
    ratio: int


@triton.jit
def _slot(
    pos,
    req,
    valid,
    table,
    COLS,
    STRIDE,
    EB: tl.constexpr,
    TPB: tl.constexpr,
    OWNER: tl.constexpr,
    RATIO: tl.constexpr,
    RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
):
    owner_block = pos // OWNER
    column = owner_block // CP_SIZE * (OWNER // TPB) + pos % OWNER // TPB
    valid = valid & (owner_block % CP_SIZE == RANK) & (column < COLS)
    valid = valid & ((pos + 1) % RATIO == 0)
    block = tl.load(table + req * STRIDE + column, valid, other=0).to(tl.int64)
    return tl.where(valid & (block > 0), block * EB + pos % TPB // RATIO, -1)


@triton.jit(do_not_specialize=["ROWS", "FIRST", "STOP", "ES", "MC", "MS", "IC", "IS"])
def _metadata(
    ends,
    mt,
    it,
    slots,
    ROWS,
    FIRST,
    STOP,
    ES,
    MC,
    MS,
    IC,
    IS,
    MTPB: tl.constexpr,
    ITPB: tl.constexpr,
    MEB: tl.constexpr,
    IEB: tl.constexpr,
    OWNER: tl.constexpr,
    RATIO: tl.constexpr,
    RANK: tl.constexpr,
    CP_SIZE: tl.constexpr,
    output_offsets,
    DIRECT_OUTPUT: tl.constexpr,
):
    ix = tl.arange(0, 128)
    counts = tl.load(ends + (FIRST + ix) * ES, FIRST + ix < STOP, other=0) // RATIO
    sizes = tl.cdiv(counts, 256) * 256
    prefix = tl.cumsum(sizes)
    row = tl.program_id(0) * 128 + ix
    lower = tl.full((128,), 0, tl.int32)
    upper = tl.full((128,), STOP - FIRST, tl.int32)
    for _ in tl.static_range(8):
        active = lower < upper
        middle = (lower + upper) // 2
        cutoff = tl.gather(prefix, tl.minimum(middle, 127), 0)
        lower = tl.where(active & (row >= cutoff), middle + 1, lower)
        upper = tl.where(active & (row < cutoff), middle, upper)
    req = tl.minimum(lower, 127)
    local = row - tl.gather(prefix - sizes, req, 0)
    valid = (row < ROWS) & (FIRST + req < STOP) & (local < tl.gather(counts, req, 0))
    pos = tl.maximum((local + 1) * RATIO - 1, 0)
    req = (FIRST + req).to(tl.int64)
    mi = _slot(pos, req, valid, mt, MC, MS, MEB, MTPB, OWNER, RATIO, RANK, CP_SIZE)
    ii = _slot(pos, req, valid, it, IC, IS, IEB, ITPB, OWNER, RATIO, RANK, CP_SIZE)
    tl.store(slots + row, mi, row < ROWS)
    tl.store(slots + ROWS + row, ii, row < ROWS)
    if DIRECT_OUTPUT:
        output_row = tl.load(output_offsets + req, valid, other=0) + local
        tl.store(slots + 2 * ROWS + row, tl.where(valid, output_row, -1), row < ROWS)


@triton.jit(do_not_specialize=["ROWS", "MP", "IP"])
def _gather(
    slots,
    main,
    index,
    quant,
    scale,
    raw,
    ROWS,
    MP,
    IP,
    MEB: tl.constexpr,
    IEB: tl.constexpr,
):
    codec._fp4_indexer_gather_kernel(
        index, slots + ROWS, quant, scale, ROWS, _INDEX_DIM, _INDEX_GROUP, IEB, IP, 0
    )
    codec._fp4_global_gather_bytes_kernel(
        main, slots, raw, ROWS, _RAW_C, 512, MEB, MP, 0
    )


@triton.jit(do_not_specialize=["ROWS"])
def _dequant(raw, quant, scale, out, oq, os, ROWS, slots, DIRECT_OUTPUT: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    stride = tl.full((), _RAW_C, tl.int64)
    if DIRECT_OUTPUT:
        destination = tl.load(slots + 2 * ROWS + row)
    else:
        destination = row
    if destination >= 0:
        # Keep the FP4 codec as the only arithmetic/format owner. Adjusting its
        # base pointer changes the destination row, never the decoded values.
        codec._fp4_global_dequant_kernel(
            raw,
            raw,
            out + (destination - row) * _GLOBAL_DIM,
            ROWS,
            _GLOBAL_DIM,
            _GLOBAL_GROUP,
            _RAW_C,
            1,
            stride,
            ROWS,
            tl.bfloat16,
            True,
        )
    columns = tl.arange(0, _PAYLOAD_C)
    tl.store(
        oq + row * _PAYLOAD_C + columns, tl.load(quant + row * _PAYLOAD_C + columns)
    )
    tl.store(os + row, tl.load(scale + row))


def _layout_valid(ratio, owner, mtpb, itpb, rank, mdtype, idtype, meb, ieb, cp_size):
    return (
        type(ratio) is int
        and ratio in (1, 2)
        and type(rank) is int
        and 0 <= rank < cp_size
        and all(type(x) is int and 0 < x < 2**31 for x in (owner, mtpb, itpb))
        and owner % mtpb == owner % itpb == mtpb % ratio == itpb % ratio == 0
        and mdtype in (torch.int32, torch.int64)
        and idtype in (torch.int32, torch.int64)
        and all(type(x) is int and 0 < x < 2**31 for x in (meb, ieb))
    )


def _views(packet, rows):
    quant = packet[: rows * _PAYLOAD].view(torch.int8).view(rows, _PAYLOAD)
    scale = packet[rows * _PAYLOAD : rows * (_PAYLOAD + _SCALE)].view(torch.int32)
    raw = packet[rows * (_PAYLOAD + _SCALE) :].view(rows, _RAW)
    return quant, scale, raw


def _metadata_args(
    ends, mt, it, slots, rows, first, stop, layout, *, output_offsets=None
):
    ratio, owner, mtpb, itpb, rank, _, _, meb, ieb, cp_size = layout
    return (
        ends,
        mt,
        it,
        slots,
        rows,
        first,
        stop,
        ends.stride(0),
        mt.shape[1],
        mt.stride(0),
        it.shape[1],
        it.stride(0),
        mtpb,
        itpb,
        meb,
        ieb,
        owner,
        ratio,
        rank,
        cp_size,
        ends if output_offsets is None else output_offsets,
        output_offsets is not None,
    )


def is_supported_layout(main_layout, index_layout, *, cp_size):
    """Qualify native V4.1 capacity separately from raw-token page coverage."""
    if cp_size < 1 or not all(
        isinstance(x, PoolLayout) for x in (main_layout, index_layout)
    ):
        return False
    if not all(
        type(x.entries_per_block) is int
        and x.entries_per_block > 0
        and type(x.tokens_per_block) is int
        and x.tokens_per_block > 0
        and type(x.ratio) is int
        and x.ratio in (1, 2)
        for x in (main_layout, index_layout)
    ):
        return False
    if (
        index_layout.ratio != main_layout.ratio
        or main_layout.owner_tokens_per_block != index_layout.owner_tokens_per_block
        or main_layout.entries_per_block * main_layout.ratio
        != main_layout.tokens_per_block
        # Native INDEXER keeps full capacity; ratio2 only uses its first half.
        or index_layout.entries_per_block != index_layout.tokens_per_block
    ):
        return False
    return _layout_valid(
        main_layout.ratio,
        main_layout.owner_tokens_per_block,
        main_layout.tokens_per_block,
        index_layout.tokens_per_block,
        0,
        main_layout.table_dtype,
        index_layout.table_dtype,
        main_layout.entries_per_block,
        index_layout.entries_per_block,
        cp_size,
    )


def warmup(main_layout, index_layout, *, cp_size, cp_rank, device):
    """Compile one layout/rank signature with private buffers, on the current stream.

    Main checks is_supported_layout from common model metadata, then requires
    warmup success on every rank. Errors propagate; no production pool or
    collective is touched. No per-length variants or retained private tensors.
    """
    if not is_supported_layout(main_layout, index_layout, cp_size=cp_size):
        return False
    ratio = main_layout.ratio
    owner_tokens_per_block = main_layout.owner_tokens_per_block
    main_tokens_per_block, index_tokens_per_block = (
        main_layout.tokens_per_block,
        index_layout.tokens_per_block,
    )
    main_table_dtype, index_table_dtype = (
        main_layout.table_dtype,
        index_layout.table_dtype,
    )
    device = torch.device(device)
    layout = (
        ratio,
        owner_tokens_per_block,
        main_tokens_per_block,
        index_tokens_per_block,
        cp_rank,
        main_table_dtype,
        index_table_dtype,
        main_layout.entries_per_block,
        index_layout.entries_per_block,
        cp_size,
    )
    if device.type != "cuda" or not _layout_valid(*layout):
        return False
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.get_device_capability()[0] != 10:
            return False
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("joint pool warmup must precede CUDA graph capture")
        key = (torch.cuda.current_device(), *layout)
        if key in _READY:
            return True
        rows = 256
        # Minimal element alignment prevents accidental pointer specialization.
        ends = torch.ones(33, dtype=torch.int64, device=device)[1:]
        mt = torch.zeros(33, dtype=main_table_dtype, device=device)[1:].view(32, 1)
        it = torch.zeros(33, dtype=index_table_dtype, device=device)[1:].view(32, 1)
        slots = torch.empty((2, rows), dtype=torch.int64, device=device)
        main = torch.zeros(
            (1, main_layout.entries_per_block, _RAW), dtype=torch.uint8, device=device
        )
        index = torch.zeros(
            (1, index_layout.entries_per_block, _PAYLOAD + _SCALE),
            dtype=torch.uint8,
            device=device,
        )
        packet = torch.empty(rows * _PACKED, dtype=torch.uint8, device=device)
        q, s, raw = _views(packet, rows)
        out = torch.empty(
            (rows, codec.FP4_GLOBAL_HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        oq, os = torch.empty_like(q), torch.empty_like(s)
        args = (
            _metadata_args(ends, mt, it, slots, rows, 0, 32, layout),
            (
                slots,
                main,
                index,
                q,
                s,
                raw,
                rows,
                main.stride(0),
                index.stride(0),
                main.shape[1],
                index.shape[1],
            ),
            (raw, q, s, out, oq, os, rows, slots, False),
        )
        kernels = tuple(
            fn.warmup(*a, grid=(1,), num_warps=4)
            for fn, a in zip((_metadata, _gather, _dequant), args)
        )
        for kernel, a, grid in zip(kernels, args, (triton.cdiv(rows, 128), rows, rows)):
            kernel[(grid, 1, 1)](*a)
        direct_slots = torch.empty((3, rows), dtype=torch.int64, device=device)
        direct_ends = torch.full((33,), ratio, dtype=torch.int64, device=device)[1:]
        output_offsets = torch.arange(33, dtype=torch.int64, device=device)[1:]
        direct_args = (
            _metadata_args(
                direct_ends,
                mt,
                it,
                direct_slots,
                rows,
                0,
                32,
                layout,
                output_offsets=output_offsets,
            ),
            (raw, q, s, out, oq, os, rows, direct_slots, True),
        )
        direct_kernels = tuple(
            fn.warmup(*a, grid=(1,), num_warps=4)
            for fn, a in zip((_metadata, _dequant), direct_args)
        )
        for kernel, a, grid in zip(
            direct_kernels, direct_args, (triton.cdiv(rows, 128), rows)
        ):
            kernel[(grid, 1, 1)](*a)
        from . import _v41_owned_pool

        owned_kernels = _v41_owned_pool.warmup(layout, device, _gather)
        torch.cuda.current_stream().synchronize()
        # Preserve legacy indices 0..2; direct output uses metadata/dequant 3/4
        # with the same gather handle at index 1. Both paths are startup-warmed.
        _READY[key] = (*kernels, *direct_kernels, owned_kernels)
    return True


def _validate_workspace(workspace, counts, device):
    output, offsets = workspace.output, workspace.device_offsets
    starts = workspace.global_starts
    if (
        workspace.global_sizes != counts
        or len(starts) != len(counts)
        or output.device != device
        or output.dtype != torch.bfloat16
        or output.ndim != 2
        or output.shape[1] != codec.FP4_GLOBAL_HEAD_DIM
        or not output.is_contiguous()
        or output.requires_grad
        or output.is_conj()
        or output.is_neg()
        or offsets.device != device
        or offsets.dtype != torch.int64
        or offsets.shape != (len(counts),)
        or not offsets.is_contiguous()
    ):
        raise ValueError("Joint pool destination has incompatible storage or lengths")
    limit, previous_end = output.shape[0], 0
    for start, count in zip(starts, counts):
        if type(start) is not int or start < previous_end or start + count > limit:
            raise ValueError("Joint pool destination ranges overlap or exceed storage")
        previous_end = start + count


def _row_partitions(counts, starts, rows):
    """Describe exact request rows and gaps for a single C++ view operation.

    No per-request Python Tensor call is needed, including nonuniform padding.
    Zero-size requests retain their position and storage offset.
    """
    sizes, selected, cursor = [], [], 0
    for count, start in zip(counts, starts):
        if start > cursor:
            sizes.append(start - cursor)
        selected.append(len(sizes))
        sizes.append(count)
        cursor = start + count
    if cursor < rows:
        sizes.append(rows - cursor)
    return sizes, selected


def _split_rows(tensor, partitions):
    sizes, selected = partitions
    views = tensor.split(sizes, dim=0)
    return tuple(views[index] for index in selected)


def try_gather(
    attn,
    main,
    index,
    ends,
    seq_ends,
    *,
    groups,
    workspace=None,
    swa=None,
    output_stream=None,
):
    """Choose the schedule from common host geometry and prepared pool layouts."""
    if not 1 <= len(ends) <= 128:
        return None
    cp, cache = getattr(attn, "_cp_ctx", None), getattr(attn, "_kv_cache", None)
    tables = getattr(attn, "_block_tables_by_type", None)
    if (
        cp is None
        or cp.cp_size < 1
        or (cp.cp_size > 1 and not cp.kv_cache_sharded)
        or attn.compress_ratio not in (1, 2)
        or cache is None
    ):
        return None
    if tables is None:
        raise ValueError("Joint pool requires bound native block tables")
    region = attn._global_region()
    mt, it = tables.get(region), tables.get(INDEXER_KV)
    if not isinstance(mt, torch.Tensor) or not isinstance(it, torch.Tensor):
        raise ValueError("Joint pool requires both main and index block tables")
    mtpb = require_pool_tokens_per_block(cache, region=region)
    itpb = require_pool_tokens_per_block(cache, region=INDEXER_KV)
    ratio, owner, rank = (
        attn.compress_ratio,
        cache.get_seq_size_per_block(attn._global_region()),
        cp.cp_rank,
    )
    main_entries = attn._source_entries(region, main)
    index_entries = attn._source_entries(INDEXER_KV, index)
    if not is_supported_layout(
        PoolLayout(mtpb, main_entries, owner, mt.dtype, ratio),
        PoolLayout(itpb, index_entries, owner, it.dtype, ratio),
        cp_size=cp.cp_size,
    ):
        return None
    layout = (
        ratio,
        owner,
        mtpb,
        itpb,
        rank,
        mt.dtype,
        it.dtype,
        main_entries,
        index_entries,
        cp.cp_size,
    )
    if not isinstance(seq_ends, torch.Tensor):
        raise ValueError("Joint pool requires the prepared sequence-end tensor")
    kernels = _READY.get((seq_ends.device.index, *layout))
    if kernels is None:
        raise RuntimeError("Joint pool kernels were not prepared by startup warmup")
    from . import _v41_owned_pool

    owned = _v41_owned_pool.try_gather(
        main,
        index,
        ends,
        seq_ends,
        mt,
        it,
        layout,
        kernels[5],
        workspace=workspace,
        swa=swa,
        output_stream=output_stream,
    )
    if owned is not None:
        return owned
    # Unsupported common geometry preserves the previous collective schedule.
    # Once the owned path is selected, local errors propagate from that path.
    if len(ends) < _MIN_BATCH:
        return None
    from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import (
        PrefillIndexerKeys,
    )

    counts = tuple(x // ratio for x in ends)
    if workspace is None and swa is not None:
        from ._v41_prefill_kv_workspace import prepare_global_workspace

        workspace = prepare_global_workspace(swa, counts, output_stream=output_stream)
    if workspace is not None:
        _validate_workspace(workspace, counts, seq_ends.device)
        globals_by_request = _split_rows(
            workspace.output,
            _row_partitions(counts, workspace.global_starts, workspace.output.shape[0]),
        )
    result = []
    for first, stop, rows in groups:
        packet = torch.empty(rows * _PACKED, dtype=torch.uint8, device=seq_ends.device)
        q, s, raw = _views(packet, rows)
        if rows:
            slots = torch.empty(
                (3 if workspace is not None else 2, rows),
                dtype=torch.int64,
                device=seq_ends.device,
            )
            kernels[3 if workspace is not None else 0][(triton.cdiv(rows, 128), 1, 1)](
                *_metadata_args(
                    seq_ends,
                    mt,
                    it,
                    slots,
                    rows,
                    first,
                    stop,
                    layout,
                    output_offsets=(
                        None if workspace is None else workspace.device_offsets
                    ),
                )
            )
            kernels[1][(rows, 1, 1)](
                slots,
                main,
                index,
                q,
                s,
                raw,
                rows,
                main.stride(0),
                index.stride(0),
                main.shape[1],
                index.shape[1],
            )
            if workspace is None:
                del slots
            attn._gather_shards(packet)
        # Common groups budget 448 B/row, including the retained direct slots.
        out = (
            torch.empty(
                (rows, codec.FP4_GLOBAL_HEAD_DIM),
                dtype=torch.bfloat16,
                device=seq_ends.device,
            )
            if workspace is None
            else workspace.output
        )
        oq, os = torch.empty_like(q), torch.empty_like(s)
        if rows:
            kernels[4 if workspace is not None else 2][(rows, 1, 1)](
                raw,
                q,
                s,
                out,
                oq,
                os,
                rows,
                seq_ends if workspace is None else slots,
                workspace is not None,
            )
            if workspace is not None:
                del slots
        del packet, q, s, raw
        offset, starts = 0, []
        for count in counts[first:stop]:
            starts.append(offset)
            offset += (count + 255) // 256 * 256
        partitions = _row_partitions(counts[first:stop], starts, rows)
        global_views = (
            _split_rows(out, partitions)
            if workspace is None
            else globals_by_request[first:stop]
        )
        result.extend(
            (g, PrefillIndexerKeys(qv, sv))
            for g, qv, sv in zip(
                global_views, _split_rows(oq, partitions), _split_rows(os, partitions)
            )
        )
        del out, oq, os
    return result if workspace is None else workspace.publish(result)
