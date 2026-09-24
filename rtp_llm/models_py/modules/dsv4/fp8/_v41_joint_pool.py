"""Startup-warmed CP4 pool readback; the FP4 codec remains the only format owner.

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

from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV
from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as codec
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
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
):
    owner_block = pos // OWNER
    column = owner_block // 4 * (OWNER // TPB) + pos % OWNER // TPB
    valid = valid & (owner_block % 4 == RANK) & (column < COLS)
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
    mi = _slot(pos, req, valid, mt, MC, MS, MEB, MTPB, OWNER, RATIO, RANK)
    ii = _slot(pos, req, valid, it, IC, IS, IEB, ITPB, OWNER, RATIO, RANK)
    tl.store(slots + row, mi, row < ROWS)
    tl.store(slots + ROWS + row, ii, row < ROWS)


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
        main,
        slots,
        raw,
        ROWS,
        _RAW_C,
        _GLOBAL_DIM // 2,
        _GLOBAL_DIM // _GLOBAL_GROUP,
        512,
        MEB,
        MP,
        0,
    )


@triton.jit(do_not_specialize=["ROWS"])
def _dequant(raw, quant, scale, out, oq, os, ROWS):
    stride = tl.full((), _RAW_C, tl.int64)
    codec._fp4_global_dequant_kernel(
        raw,
        raw,
        out,
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
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, _PAYLOAD_C)
    tl.store(
        oq + row * _PAYLOAD_C + columns, tl.load(quant + row * _PAYLOAD_C + columns)
    )
    tl.store(os + row, tl.load(scale + row))


def _layout_valid(ratio, owner, mtpb, itpb, rank, mdtype, idtype, meb, ieb):
    return (
        type(ratio) is int
        and ratio in (1, 2)
        and type(rank) is int
        and 0 <= rank < 4
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


def _metadata_args(ends, mt, it, slots, rows, first, stop, layout):
    ratio, owner, mtpb, itpb, rank, _, _, meb, ieb = layout
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
    )


def is_supported_layout(main_layout, index_layout, *, cp_size):
    """Qualify native V4.1 capacity separately from raw-token page coverage."""
    if cp_size != 4 or not all(
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
            (raw, q, s, out, oq, os, rows),
        )
        kernels = tuple(
            fn.warmup(*a, grid=(1,), num_warps=4)
            for fn, a in zip((_metadata, _gather, _dequant), args)
        )
        for kernel, a, grid in zip(kernels, args, (triton.cdiv(rows, 128), rows, rows)):
            kernel[(grid, 1, 1)](*a)
        torch.cuda.current_stream().synchronize()
        _READY[key] = kernels
    return True


def _pool_valid(pool, table, device, entries, width, batch):
    return (
        isinstance(pool, torch.Tensor)
        and pool.device == device
        and pool.dtype == torch.uint8
        and pool.layout == torch.strided
        and pool.ndim == 3
        and pool.shape[0] > 0
        and pool.shape[1:] == (entries, width)
        and pool.stride(2) == 1
        and pool.stride(1) == width
        and pool.stride(0) >= entries * width
        and pool.stride(0) % 4 == 0
        and pool.stride(0) < 2**31
        and pool.data_ptr() % 16 == 0
        and pool.shape[0] * pool.stride(0) < 2**63
        and isinstance(table, torch.Tensor)
        and table.device == device
        and table.layout == torch.strided
        and table.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and table.shape[0] == batch
        and table.shape[1] > 0
        and table.stride(1) == 1
        and table.stride(0) >= table.shape[1]
        and table.shape[0] * table.stride(0) < 2**31
    )


def try_gather(attn, main, index, ends, seq_ends, *, groups):
    """Choose the schedule from common host geometry; local failures are fatal."""
    if not _MIN_BATCH <= len(ends) <= 128:
        return None
    cp, cache = getattr(attn, "_cp_ctx", None), getattr(attn, "_kv_cache", None)
    tables = getattr(attn, "_block_tables_by_type", None)
    if (
        cp is None
        or cp.cp_size != 4
        or not cp.kv_cache_sharded
        or attn.compress_ratio not in (1, 2)
        or cache is None
    ):
        return None
    if (
        tables is None
        or not isinstance(seq_ends, torch.Tensor)
        or seq_ends.device.type != "cuda"
        or seq_ends.dtype != torch.int64
        or seq_ends.layout != torch.strided
        or seq_ends.shape != (len(ends),)
        or seq_ends.stride(0) <= 0
        or seq_ends.stride(0) * len(ends) >= 2**31
        or not all(type(x) is int and 0 <= x < 2**31 for x in ends)
    ):
        raise RuntimeError("qualified joint pool has invalid local sequence metadata")
    region = attn._global_region()
    mt, it = tables.get(region), tables.get(INDEXER_KV)
    if not isinstance(mt, torch.Tensor) or not isinstance(it, torch.Tensor):
        raise RuntimeError("qualified joint pool has missing local block tables")
    try:
        mtpb = require_pool_tokens_per_block(cache, region=region)
        itpb = require_pool_tokens_per_block(cache, region=INDEXER_KV)
    except RuntimeError as error:
        raise RuntimeError("qualified joint pool has invalid model layout") from error
    ratio, owner, rank = attn.compress_ratio, cache.seq_size_per_block, cp.cp_rank
    if mt.dtype not in (torch.int32, torch.int64) or it.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise RuntimeError("qualified joint pool has invalid local block table dtype")
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
    )
    if not _layout_valid(*layout):
        raise RuntimeError("qualified joint pool layout was not accepted at startup")
    kernels = _READY.get((seq_ends.device.index, *layout))
    if kernels is None or not (
        _pool_valid(main, mt, seq_ends.device, main_entries, _RAW, len(ends))
        and _pool_valid(
            index, it, seq_ends.device, index_entries, _PAYLOAD + _SCALE, len(ends)
        )
    ):
        raise RuntimeError(
            "qualified joint pool is cold or has invalid local storage; startup warmup is required"
        )
    from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import (
        PrefillIndexerKeys,
    )

    counts = tuple(x // ratio for x in ends)
    result = []
    for first, stop, rows in groups:
        packet = torch.empty(rows * _PACKED, dtype=torch.uint8, device=seq_ends.device)
        q, s, raw = _views(packet, rows)
        if rows:
            slots = torch.empty((2, rows), dtype=torch.int64, device=seq_ends.device)
            kernels[0][(triton.cdiv(rows, 128), 1, 1)](
                *_metadata_args(seq_ends, mt, it, slots, rows, first, stop, layout)
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
            del slots
            attn._gather_shards(packet)
        # Preserve original groups; overlap is explicitly 424 B/row, not 420.
        out = torch.empty(
            (rows, codec.FP4_GLOBAL_HEAD_DIM),
            dtype=torch.bfloat16,
            device=seq_ends.device,
        )
        oq, os = torch.empty_like(q), torch.empty_like(s)
        if rows:
            kernels[2][(rows, 1, 1)](raw, q, s, out, oq, os, rows)
        del packet, q, s, raw
        padded = tuple((n + 255) // 256 * 256 for n in counts[first:stop])
        result.extend(
            (g[:n], PrefillIndexerKeys(qv[:n], sv[:n]))
            for n, g, qv, sv in zip(
                counts[first:stop],
                out.split(padded),
                oq.split(padded),
                os.split(padded),
            )
        )
        del out, oq, os
    return result
