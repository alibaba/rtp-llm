"""Bounded multi-request FP4 pool reads and byte-exact CP transport."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.attn_type import INDEXER_KV

# Excludes the returned keys, which the original per-request path also retains.
_MAX_TEMP_BYTES = 256 * 1024 * 1024
_BYTES_PER_ROW = 288 + 64 + 4 + 8 * 8
_MAX_ROWS = _MAX_TEMP_BYTES // _BYTES_PER_ROW
_KEY_ALIGNMENT = 256
_MAX_REQUESTS = 128


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


def _request_groups(counts):
    """Keep requests whole so returned key views preserve scorer geometry."""
    if any(count < 0 for count in counts):
        return None
    padded = tuple(
        (count + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        for count in counts
    )
    if any(count > _MAX_ROWS for count in padded):
        return None
    groups = []
    first, rows = 0, 0
    for index, count in enumerate(padded):
        if rows + count > _MAX_ROWS:
            groups.append((first, index, rows))
            first, rows = index, 0
        rows += count
    groups.append((first, len(counts), rows))
    return groups


def try_gather_prefill_pools(attn, main_pool, index_pool, ends, seq_ends):
    """Return the existing [(BF16 global keys, FP4 index keys)] representation.

    Each logical slot has one CP owner. Other ranks contribute zero bytes, so
    integer SUM preserves the exact payload and packed scale bits. No scales
    are reduced as floating-point values and no cache pools are modified.
    """
    cp = attn._cp_ctx
    if (
        len(ends) < 2
        or cp is None
        or cp.cp_size <= 1
        or not cp.kv_cache_sharded
        or attn.compress_ratio not in (1, 2)
        or len(ends) > _MAX_REQUESTS
    ):
        return None
    counts = tuple(end // attn.compress_ratio for end in ends)
    padded_counts = tuple(
        (count + _KEY_ALIGNMENT - 1) // _KEY_ALIGNMENT * _KEY_ALIGNMENT
        for count in counts
    )
    groups = _request_groups(counts)
    if groups is None:
        return None
    if (
        main_pool.device.type != "cuda"
        or torch.cuda.get_device_capability(main_pool.device)[0] != 10
    ):
        return None

    # Joint CP4 is qualified at batch >=32; smaller batches stay on this path.
    if len(ends) >= 32:
        from rtp_llm.models_py.modules.dsv4.fp8 import _v41_joint_pool

        joint = _v41_joint_pool.try_gather(
            attn, main_pool, index_pool, ends, seq_ends, groups=groups
        )
        if joint is not None:
            return joint

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

    from rtp_llm.models_py.modules.dsv4.fp8._v41_fp4_triton import (
        dequantize_k_cache_bytes_fp4,
        gather_indexer_k_fp4,
        gather_k_cache_bytes_fp4,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._v41_prefill_indexer import (
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
            (keys[:count], PrefillIndexerKeys(payload[:count], sf[:count]))
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
