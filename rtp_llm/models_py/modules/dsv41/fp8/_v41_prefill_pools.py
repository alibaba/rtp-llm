"""Bounded multi-request FP4 pool reads and byte-exact CP transport."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.attn_type import INDEXER_KV

# Excludes final BF16 KV and earlier groups' retained index outputs. Those
# model outputs also exist in the original path; this bounds current scratch.
# Joint direct output keeps packet (356), index outputs (68), and three int64
# slot planes (24) alive through dequant. Use this common bound for every rank
# and both output layouts, never choose collective groups from local readiness.
_MAX_TEMP_BYTES = 256 * 1024 * 1024
_BYTES_PER_ROW = 288 + 64 + 4 + 64 + 4 + 3 * 8
_MAX_ROWS = _MAX_TEMP_BYTES // _BYTES_PER_ROW
_SM90_BYTES_PER_ROW = 584 + 132 + 128 + 4 + 3 * 8
_SM90_MAX_ROWS = _MAX_TEMP_BYTES // _SM90_BYTES_PER_ROW
_KEY_ALIGNMENT = 256
_MAX_REQUESTS = 1024
_REQUESTS_PER_GROUP = 128


def _trim_rows(tensor, count):
    return tensor if count == tensor.shape[0] else tensor[:count]


@triton.jit(do_not_specialize=["FIRST", "STOP", "ROWS", "END_STRIDE"])
def _pool_row_metadata_kernel(
    seq_ends, output, FIRST, STOP, ROWS, END_STRIDE, RATIO: tl.constexpr
):
    ids = tl.arange(0, 128)
    counts = (
        tl.load(seq_ends + (FIRST + ids) * END_STRIDE, FIRST + ids < STOP, other=0)
        // RATIO
    )
    sizes = tl.cdiv(counts, 256) * 256
    ends = tl.cumsum(sizes)
    row = tl.program_id(0) * 128 + ids
    lower = tl.full((128,), 0, tl.int32)
    upper = tl.full((128,), STOP - FIRST, tl.int32)
    for _ in tl.static_range(8):
        active = lower < upper
        middle = (lower + upper) // 2
        cutoff = tl.gather(ends, tl.minimum(middle, 127), 0)
        lower = tl.where(active & (row >= cutoff), middle + 1, lower)
        upper = tl.where(active & (row < cutoff), middle, upper)
    request = tl.minimum(lower, 127)
    local = row - tl.gather(ends - sizes, request, 0)
    position = (local + 1) * RATIO - 1
    position = tl.where(local < tl.gather(counts, request, 0), position, -1)
    tl.store(output + row, position, row < ROWS)
    tl.store(output + ROWS + row, (FIRST + request).to(tl.int64), row < ROWS)


def _group_row_metadata(seq_ends, first, stop, rows, ratio):
    output = torch.empty((2, rows), dtype=torch.int64, device=seq_ends.device)
    if rows:
        _pool_row_metadata_kernel[(triton.cdiv(rows, 128),)](
            seq_ends,
            output,
            first,
            stop,
            rows,
            seq_ends.stride(0),
            ratio,
            num_warps=4,
        )
    return output.unbind(0)


def _request_groups(counts, *, max_rows: int = _MAX_ROWS):
    """Keep requests whole so returned key views preserve scorer geometry."""
    if any(count < 0 for count in counts):
        return None
    padded = tuple(
        (count + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        for count in counts
    )
    if any(count > max_rows for count in padded):
        return None
    groups = []
    first, rows = 0, 0
    for index, count in enumerate(padded):
        if rows + count > max_rows or index - first == _REQUESTS_PER_GROUP:
            groups.append((first, index, rows))
            first, rows = index, 0
        rows += count
    groups.append((first, len(counts), rows))
    return groups


def _try_gather_sm90_fp8_pools(attn, main_pool, index_pool, ends, seq_ends, counts):
    """Bounded multi-request SM90 V4+FP8 pool gather with batched CP all-reduce."""
    swa_codec = getattr(attn, "_swa_codec", None)
    if (
        getattr(attn, "_indexer_mode", None) != "fp8_dg"
        or not getattr(attn, "_use_v4_kv_cache", False)
        or swa_codec is None
        or main_pool.dtype != torch.uint8
        or index_pool.dtype != torch.uint8
        or index_pool.device != main_pool.device
        or main_pool.ndim != 3
        or main_pool.shape[-1] != 584
        or index_pool.ndim != 3
        or index_pool.shape[2] != 132
        or index_pool.shape[1] not in (64, 128)
        or not index_pool.is_contiguous()
        or not isinstance(seq_ends, torch.Tensor)
        or seq_ends.device != main_pool.device
        or seq_ends.dtype != torch.int64
        or seq_ends.shape != (len(ends),)
        or seq_ends.stride(0) <= 0
    ):
        return None
    groups = _request_groups(counts, max_rows=_SM90_MAX_ROWS)
    if groups is None:
        return None
    padded_counts = tuple(
        (count + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        for count in counts
    )

    from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_fp8_indexer_cache import dg_pages
    from rtp_llm.models_py.modules.dsv41.fp8._v41_sm90_fp8_prefill import (
        PrefillFP8IndexerKeys,
        gather_fp8_indexer,
        gather_raw_fp8_indexer,
        unpack_raw_fp8_indexer,
    )

    cp = attn._cp_ctx
    sharded_cp = cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded
    pages = dg_pages(index_pool)
    result = []
    for first, stop, rows in groups:
        positions, req_ids = _group_row_metadata(
            seq_ends, first, stop, rows, attn.compress_ratio
        )
        index_slots = attn._slots(INDEXER_KV, positions, req_ids)
        if sharded_cp:
            raw_index = gather_raw_fp8_indexer(pages, index_slots)
            attn._gather_shards(raw_index)
            index_keys = unpack_raw_fp8_indexer(raw_index)
            del raw_index
        else:
            index_keys = gather_fp8_indexer(pages, index_slots)
        del index_slots
        global_slots = attn._slots(attn._global_region(), positions, req_ids)
        raw_global = swa_codec.gather_k_cache_bytes(main_pool, global_slots)
        attn._gather_shards(raw_global)
        global_keys = swa_codec.dequantize_k_cache_bytes(raw_global)
        del raw_global, global_slots, positions, req_ids
        group_counts = padded_counts[first:stop]
        # Split into 256-aligned contiguous views sharing each group's slab so
        # grouped SM90 FP8 scoring can alias adjacent requests without copies.
        result.extend(
            (
                _trim_rows(keys, count),
                PrefillFP8IndexerKeys(_trim_rows(payload, count), _trim_rows(sf, count)),
            )
            for count, keys, payload, sf in zip(
                counts[first:stop],
                global_keys.split(group_counts, dim=0),
                index_keys.quant.split(group_counts, dim=0),
                index_keys.scale.split(group_counts, dim=0),
            )
        )
        del global_keys, index_keys
    return result


def try_gather_prefill_pools(
    attn,
    main_pool,
    index_pool,
    ends,
    seq_ends,
    *,
    workspace=None,
    swa=None,
    output_stream=None,
):
    """Return the existing [(BF16 global keys, FP4 index keys)] representation.

    Each logical slot has one CP owner. Other ranks contribute zero bytes, so
    integer SUM preserves the exact payload and packed scale bits. No scales
    are reduced as floating-point values and no cache pools are modified.
    """
    cp = attn._cp_ctx
    if (
        len(ends) < 1
        or cp is None
        or cp.cp_size < 1
        or (cp.cp_size > 1 and not cp.kv_cache_sharded)
        or attn.compress_ratio not in (1, 2)
        or len(ends) > _MAX_REQUESTS
    ):
        return None
    counts = tuple(end // attn.compress_ratio for end in ends)
    groups = _request_groups(counts)
    if any(count < 0 for count in counts):
        return None
    if main_pool.device.type != "cuda":
        return None
    major = torch.cuda.get_device_capability(main_pool.device)[0]
    if major == 9:
        return _try_gather_sm90_fp8_pools(
            attn, main_pool, index_pool, ends, seq_ends, counts
        )
    main_shape = getattr(main_pool, "shape", None)
    if major != 10 or (main_shape is not None and main_shape[-1] != 288):
        # This byte transport is explicitly FP4; unsupported pools fall back
        # to the caller's format-aware codec path.
        return None

    from rtp_llm.models_py.modules.dsv41.fp8 import _v41_joint_pool

    joint = _v41_joint_pool.try_gather(
        attn,
        main_pool,
        index_pool,
        ends,
        seq_ends,
        groups=groups,
        workspace=workspace,
        swa=swa,
        output_stream=output_stream,
    )
    if joint is not None:
        return joint

    if groups is None:
        return None

    padded_counts = tuple(
        (count + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        for count in counts
    )

    if (
        main_pool.dtype != torch.uint8
        or index_pool.dtype != torch.uint8
        or index_pool.device != main_pool.device
        or not isinstance(seq_ends, torch.Tensor)
        or seq_ends.device != main_pool.device
        or seq_ends.dtype != torch.int64
        or seq_ends.shape != (len(ends),)
        or seq_ends.stride(0) <= 0
    ):
        return None

    from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
        dequantize_k_cache_bytes_fp4,
        gather_indexer_k_fp4,
        gather_k_cache_bytes_fp4,
    )
    from rtp_llm.models_py.modules.dsv41.fp8._v41_prefill_indexer import (
        PrefillIndexerKeys,
    )

    result = []
    for first, stop, rows in groups:
        # Lengths already reside on the GPU. Do not copy the host grouping
        # metadata back to the device or synchronize to recover output sizes.
        positions, req_ids = _group_row_metadata(
            seq_ends, first, stop, rows, attn.compress_ratio
        )
        slots = attn._slots(INDEXER_KV, positions, req_ids)
        quant, scale = gather_indexer_k_fp4(index_pool, slots)
        attn._gather_shards(quant.view(torch.uint8))
        attn._gather_shards(scale.view(torch.uint8))
        slots = attn._slots(attn._global_region(), positions, req_ids)
        raw = gather_k_cache_bytes_fp4(main_pool, slots)
        attn._gather_shards(raw)
        global_keys = dequantize_k_cache_bytes_fp4(raw)
        group_counts = padded_counts[first:stop]
        # Split only the leading dimension: these contiguous views share the
        # slabs and require no GPU work or per-request device allocation.
        result.extend(
            (
                _trim_rows(keys, count),
                PrefillIndexerKeys(_trim_rows(payload, count), _trim_rows(sf, count)),
            )
            for count, keys, payload, sf in zip(
                counts[first:stop],
                global_keys.split(group_counts, dim=0),
                quant.split(group_counts, dim=0),
                scale.split(group_counts, dim=0),
            )
        )
        del (
            raw,
            global_keys,
            quant,
            scale,
            slots,
            positions,
            req_ids,
        )
    return result
