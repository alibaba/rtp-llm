"""Bounded startup warmup for V4.1 prefill attention, using private buffers.

The first model initialization can precede KV allocation. Independent kernels
are warmed then; a second call with the real cache schema warms pool layouts.
Only shapes/dtypes/strides are read from that cache. No collective or real pool
write is issued here. Request lengths remain runtime arguments to the Triton
metadata kernels, so arbitrary prefix reuse does not require an M/N grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

_WARMED: set[tuple] = set()


@dataclass(frozen=True)
class PoolLayout:
    region: int
    entries: int
    stride_bytes: int
    tokens_per_block: int
    owner_tokens_per_block: int
    ratio: int


def _attention_key(attn):
    """Host-only specialization key, independent of layer and tensor pointers."""
    return (
        int(attn.compress_ratio),
        int(attn.n_heads),
        int(attn.head_dim),
        int(attn.window_size),
        int(getattr(attn, "index_topk", 0)),
        int(getattr(attn, "index_n_heads", 0)),
        int(getattr(attn, "index_head_dim", 0)),
        float(attn.eps),
        int(attn.freqs_cis.shape[0]),
        str(getattr(getattr(attn, "global_norm", None), "dtype", None)),
        str(getattr(getattr(attn, "index_k_norm", None), "dtype", None)),
    )


def _collect_attentions(v4):
    result = {}
    for layer in getattr(v4, "layers", ()):
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "v41_config"):
            continue
        # Keep producer representatives: they also own the index norms.
        key = _attention_key(attn)
        if key not in result or getattr(attn, "is_kv_source", False):
            result[key] = attn
    return result


def _candidate_widths(max_seq_len, block_size, candidate_count):
    """One width for each reachable bitmap tile, plus the exact upper bound."""
    first = block_size * candidate_count + 1
    if first > max_seq_len:
        return ()
    widths = {first, int(max_seq_len)}
    width = 1 << (first - 1).bit_length()
    while width < max_seq_len:
        widths.add(width)
        widths.add(width + 1)
        width *= 2
    return tuple(sorted(widths))


def _collect_pool_layouts(attentions, kv_cache, cp_size, kv_cache_sharded):
    if kv_cache is None:
        return {}
    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        CSA_STATE,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )
    from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import (
        require_pool_tokens_per_block,
    )
    from rtp_llm.models_py.modules.dsv4.fp8.attention import _ATTN_TYPE_ENUM_BY_INT

    layouts = {}
    for attn in attentions.values():
        regions = [SWA_KV]
        ratio = int(attn.compress_ratio)
        if getattr(attn, "is_kv_source", False):
            regions.extend((CSA_KV if ratio == 2 else HCA_KV, INDEXER_KV))
            if ratio == 2:
                regions.append(CSA_STATE)
        for region in regions:
            cache = kv_cache.get_layer_cache(
                attn.layer_id, _ATTN_TYPE_ENUM_BY_INT[region]
            )
            base = cache.kv_cache_base
            if base is None or len(base.shape) != 2 or not base.shape[0]:
                raise RuntimeError(f"V4.1 warmup missing allocated region {region}")
            dtype, width = attn._pool_spec[region]
            stride = int(base.shape[1]) * base.element_size()
            if region == SWA_KV and kv_cache_sharded:
                stride *= cp_size  # Encoder/decoder sees reconstructed full pages.
            entries = stride // (int(width) * dtype.itemsize)
            tpb = require_pool_tokens_per_block(kv_cache, region=region)
            owner = int(kv_cache.seq_size_per_block) if kv_cache_sharded else tpb
            layout = PoolLayout(region, entries, stride, tpb, owner, ratio)
            layouts[(layout, _attention_key(attn))] = attn
    return layouts


def _private_pool(layout, width, device):
    import torch

    raw = torch.zeros((2, layout.stride_bytes), dtype=torch.uint8, device=device)
    return raw.as_strided((2, layout.entries, width), (layout.stride_bytes, width, 1))


def _require_launch(result, label, *, enabled=True):
    if result is None or result is False:
        if enabled:
            raise RuntimeError(f"V4.1 startup warmup rejected reachable {label}")
        logging.info("[DSV41 Attention] skipped disabled/unsupported %s", label)
    return result


def _private_slots(rows, entries, device):
    import torch

    # Never race duplicate writes even when STATE has fewer than 16 entries.
    slots = torch.arange(rows, device=device, dtype=torch.int64)
    return torch.where(slots < entries, slots, -1)


def _slot_metadata_layouts():
    # CP tiled slices independently offset the position/request vectors and
    # the table view. Both pointers affect Triton's alignment specialization.
    return ((0, 0), (0, 1), (1, 0), (1, 1))


def _warm_swa_metadata(attn, layout, max_batch_size, device):
    import torch

    from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
        _swa_slot_batch_block_warmup_sizes,
    )
    from rtp_llm.models_py.modules.dsv4.fp8 import _swa_ops_triton as ops

    rows = 16
    cu = torch.arange(0, rows + 1, rows, dtype=torch.int32, device=device)
    prefix = torch.full((1,), 1, dtype=torch.int32, device=device)
    lengths = torch.full((1,), rows + 1, dtype=torch.int32, device=device)
    requests = torch.zeros(rows, dtype=torch.int32, device=device)
    positions = torch.arange(1, rows + 1, dtype=torch.int64, device=device)
    ops.compute_window_topk_and_length_varlen(
        attn.window_size, cu, positions, prefix, requests
    )
    ops.compute_prefill_gather_lens(lengths, cu, 1, 0, attn.window_size)
    for table_offset in (0, 1):
        table = torch.ones(2 + table_offset, dtype=torch.int32, device=device)[
            table_offset:
        ].view(1, 2)
        ops.compute_swa_slot_mapping(
            table,
            cu,
            lengths,
            rows,
            pool_entries_per_block=layout.entries,
            tokens_per_block_for_block_table=layout.tokens_per_block,
            ring_entries=layout.entries,
        )
        ops.compute_swa_slot_mapping_from_positions(
            table,
            requests,
            positions.to(torch.int32),
            lengths,
            rows,
            layout.entries,
            layout.tokens_per_block,
            layout.entries,
        )
    # Reuse V4's bucket planner with the engine's configured concurrency bound.
    # Exact request counts are runtime; only BLOCK_B requires multiple kernels.
    for batch in _swa_slot_batch_block_warmup_sizes(max_batch_size):
        full_cu = torch.arange(0, 2 * batch + 1, 2, dtype=torch.int32, device=device)
        for dtype in (torch.int32, torch.int64):
            prefixes = torch.ones(batch, dtype=dtype, device=device)
            ops.compute_swa_slot_in_flat_from_cu(
                full_cu,
                prefixes,
                num_tokens=2 * batch,
                M=attn.window_size + 2,
                window_size=attn.window_size,
                base_offset=0,
            )


def _warm_swa_byte_slices(layout, cp_size, cp_rank, device):
    import torch

    from . import _v41_swa_triton as swa
    from ._swa_cp_byte_sliced import CPByteSlicedSlotCompaction

    if cp_size != 4 or layout.entries != 136 or layout.stride_bytes != 72192:
        return
    rows, pages, local_bytes = 16, 2, 18048
    raw = torch.zeros((pages, local_bytes), dtype=torch.uint8, device=device)
    gathered = torch.zeros(
        (cp_size * pages, local_bytes), dtype=torch.uint8, device=device
    )
    keys = torch.zeros((rows, 512), dtype=torch.bfloat16, device=device)
    out = torch.empty((1, rows + 1, 512), dtype=torch.bfloat16, device=device)
    for slot_offset, page_offset in _slot_metadata_layouts():
        slots = torch.arange(rows + slot_offset, dtype=torch.int64, device=device)[
            slot_offset:
        ]
        blocks = torch.tensor(
            [0] * page_offset + list(range(pages)), dtype=torch.int64, device=device
        )[page_offset:]
        compaction = CPByteSlicedSlotCompaction(blocks, slots)
        swa.quantize_and_insert_k_cache_cp_byte_sliced(
            keys, raw, slots, layout.entries, cp_rank, cp_size, compaction
        )
        for dtype in (torch.int32, torch.int64):
            for offset in (0, 1):
                fresh_slots = torch.arange(rows + offset, dtype=dtype, device=device)[
                    offset:
                ]
                swa.quantize_and_insert_k_cache_cp_byte_sliced(
                    keys,
                    raw,
                    slots,
                    layout.entries,
                    cp_rank,
                    cp_size,
                    compaction,
                    fresh_out=out,
                    fresh_slots=fresh_slots,
                )
        if page_offset == 0:
            for lengths in (
                None,
                torch.full((1,), rows, dtype=torch.int32, device=device),
            ):
                swa._gather_swa_rank_major(
                    out,
                    gathered,
                    slots[None],
                    lengths,
                    1,
                    pages,
                    local_bytes,
                    layout.entries,
                )


def _warm_joint_pool(main_layout, index_layout, cp_rank, device):
    import torch

    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_joint_pool as joint

    def describe(layout, dtype):
        return joint.PoolLayout(
            tokens_per_block=layout.tokens_per_block,
            entries_per_block=layout.entries,
            owner_tokens_per_block=layout.owner_tokens_per_block,
            table_dtype=dtype,
            ratio=layout.ratio,
        )

    if not joint.is_supported_layout(
        describe(main_layout, torch.int32),
        describe(index_layout, torch.int32),
        cp_size=4,
    ):
        return
    for main_dtype in (torch.int32, torch.int64):
        for index_dtype in (torch.int32, torch.int64):
            _require_launch(
                joint.warmup(
                    describe(main_layout, main_dtype),
                    describe(index_layout, index_dtype),
                    cp_size=4,
                    cp_rank=cp_rank,
                    device=device,
                ),
                "joint pool readback",
            )


def _warm_pool(attn, layout, cp_size, cp_rank, max_batch_size, device):
    import torch

    from rtp_llm.models_py.modules.dsv4.attn_type import (
        CSA_KV,
        CSA_STATE,
        HCA_KV,
        INDEXER_KV,
        SWA_KV,
    )
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_fp4_triton as fp4
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_global as producer
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata as meta
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_swa_triton as swa

    rows = 16
    starts = torch.zeros((1,), dtype=torch.int64, device=device)
    slots = _private_slots(rows, layout.entries, device)
    if layout.region != SWA_KV:
        for dtype in (torch.int32, torch.int64):
            for offset in (0, 1):
                chunk_prefix = torch.zeros(3 + offset, dtype=dtype, device=device)[
                    offset:
                ]
                chunk_lengths = torch.ones_like(chunk_prefix)
                chunk_req = torch.zeros(3 + offset, dtype=torch.int64, device=device)[
                    offset:
                ]
                for replay in (False, True):
                    _require_launch(
                        meta.try_chunk_metadata(
                            chunk_prefix,
                            chunk_lengths,
                            chunk_req,
                            layout.ratio,
                            attn.window_size,
                            replay,
                        ),
                        "chunk metadata",
                    )
        for offset, table_offset in _slot_metadata_layouts():
            metadata_positions = _indexer_warmup_vector(
                rows, torch.int64, device, 1, offset
            )
            metadata_requests = _indexer_warmup_vector(
                rows, torch.int64, device, 1, offset, fill=0
            )
            table = torch.ones(2 + table_offset, dtype=torch.int32, device=device)[
                table_offset:
            ].view(1, 2)
            for ends in (None, torch.full_like(starts, rows + offset)):
                mapping = meta.try_slot_mapping(
                    metadata_positions,
                    metadata_requests,
                    table,
                    layout.entries,
                    layout.tokens_per_block,
                    layout.ratio,
                    cp_size,
                    cp_rank,
                    owner_tokens_per_block=layout.owner_tokens_per_block,
                    state=layout.region == CSA_STATE,
                    seq_ends=ends,
                )
                _require_launch(
                    mapping,
                    f"slot mapping {layout}",
                    enabled=layout.region == CSA_STATE
                    or layout.owner_tokens_per_block % layout.tokens_per_block == 0,
                )
    if layout.region == SWA_KV:
        _warm_swa_metadata(attn, layout, max_batch_size, device)
        pool = _private_pool(layout, swa.ENTRY_BYTES, device)
        kv = torch.zeros((rows, 512), dtype=torch.bfloat16, device=device)
        out = torch.empty((1, rows + 1, 512), dtype=torch.bfloat16, device=device)
        for dtype in (torch.int32, torch.int64):
            typed_slots = slots.to(dtype)
            swa.quantize_and_insert_swa_k_cache(kv, pool, typed_slots)
            swa.dequantize_swa_k_cache(pool, typed_slots)
            for lengths in (
                None,
                torch.full((1,), rows, dtype=torch.int32, device=device),
            ):
                swa.dequantize_and_gather_k_cache_slots(
                    out, pool, typed_slots[None], lengths, 1
                )
        _warm_swa_byte_slices(layout, cp_size, cp_rank, device)
        return
    if layout.region == INDEXER_KV:
        from . import _v41_grouped_gemm

        if _v41_grouped_gemm._enabled():
            _require_launch(
                _v41_grouped_gemm.warmup(attn.index_wk), "grouped index GEMM"
            )
        pool = _private_pool(layout, 68, device)
        projected = torch.zeros((rows, 128), dtype=torch.bfloat16, device=device)
        for offset, slot_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
            index_positions = _indexer_warmup_vector(
                rows, torch.int64, device, 1, offset
            )
            index_slots = _private_slots(rows + slot_offset, layout.entries, device)[
                slot_offset:
            ]
            stored = producer.store_index(
                projected,
                attn.index_k_norm,
                attn.eps,
                index_positions,
                attn.freqs_cis,
                pool,
                index_slots,
                layout.ratio,
            )
            _require_launch(stored, "index store", enabled=producer._enabled(projected))
        fp4.quantize_indexer_k_fp4(projected, slots, pool)
        fp4.gather_indexer_k_fp4(pool, slots)
        fp4.dequantize_indexer_k_fp4(pool, slots)
        return
    if layout.region == CSA_STATE:
        pool = torch.zeros(
            (2 * layout.entries, 1024), dtype=torch.float32, device=device
        )
        for stride, slot_offset in ((512, 0), (512, 1), (1024, 0), (1024, 1)):
            values = torch.zeros((rows, stride), dtype=torch.float32, device=device)
            state_slots = _private_slots(rows + slot_offset, layout.entries, device)[
                slot_offset:
            ]
            stored = producer.store_states(
                values[:, :512], values[:, -512:], state_slots, pool
            )
            _require_launch(stored, "state store", enabled=producer._enabled(values))
        return
    if layout.region in (CSA_KV, HCA_KV):
        from ._v41_prefill_pools import _group_row_metadata

        if layout.ratio == 2:
            _warm_raw_producer_metadata(layout.ratio, device)
        pool_ends = torch.tensor(
            [0, 2 * layout.ratio, 3 * layout.ratio], dtype=torch.int64, device=device
        )
        _group_row_metadata(pool_ends, 0, 3, 512, layout.ratio)
        pool = _private_pool(layout, 288, device)
        kv = torch.zeros((rows, 512), dtype=torch.bfloat16, device=device)
        fp4.quantize_and_insert_k_cache_fp4(kv, pool, slots)
        fp4.dequantize_k_cache_slots_fp4(pool, slots)
        raw = fp4.gather_k_cache_bytes_fp4(pool, slots)
        fp4.dequantize_k_cache_bytes_fp4(raw)
        previous = (
            torch.zeros((1, 1024), dtype=torch.float32, device=device)
            if layout.ratio == 2
            else None
        )
        # REDUCE_WIDTH has only three reachable values: C<8, C<16, C>=16.
        for count in (1, 8, 16):
            for stride in (512, 1024):
                values = torch.zeros(
                    (2 * count, stride), dtype=torch.float32, device=device
                )
                scores = values[:, -512:]
                indices = torch.arange(count, device=device, dtype=torch.int64) * 2 + 1
                for offset, slot_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    pos = _indexer_warmup_vector(
                        2 * count, torch.int64, device, 1, offset
                    )
                    req = _indexer_warmup_vector(
                        2 * count, torch.int64, device, 1, offset, fill=0
                    )
                    write_slots = _private_slots(
                        count + slot_offset, layout.entries, device
                    )[slot_offset:]
                    for has_carry in (False, True) if layout.ratio == 2 else (False,):
                        carry = (
                            (values[:1, :512].contiguous(), scores[:1].contiguous())
                            if has_carry
                            else None
                        )
                        latent = producer.compress_main(
                            values[:, :512],
                            scores if layout.ratio == 2 else None,
                            attn.global_norm,
                            attn.eps,
                            pos,
                            req,
                            starts,
                            previous,
                            indices,
                            attn.freqs_cis,
                            pool,
                            write_slots,
                            layout.ratio,
                            carry,
                        )
                        _require_launch(
                            latent,
                            "main compression",
                            enabled=producer._enabled(values),
                        )
        from . import _v41_batched_producer as batched

        if batched.is_supported(kv):
            _require_launch(
                batched.warmup_projected_groups(device), "batched projection transport"
            )
            count = rows // layout.ratio
            plan = batched.GroupPlan(
                0,
                rows,
                0,
                count,
                layout.ratio,
                ((0, rows, 0, count, layout.ratio - 1),),
            )
            prepared = batched.prepare(plan, device)
            for stride in (512, 1024):
                group = torch.zeros((rows, stride), dtype=torch.float32, device=device)
                for offset, slot_offset in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    pos = _indexer_warmup_vector(rows, torch.int64, device, 1, offset)
                    req = _indexer_warmup_vector(
                        rows, torch.int64, device, 1, offset, fill=0
                    )
                    mapped = _private_slots(
                        count + slot_offset, layout.entries, device
                    )[slot_offset:]
                    _require_launch(
                        batched.compress_main(
                            group[:, :512],
                            group[:, -512:] if layout.ratio == 2 else None,
                            attn.global_norm,
                            attn.eps,
                            pos,
                            req,
                            starts,
                            previous,
                            prepared,
                            attn.freqs_cis,
                            pool,
                            mapped,
                        ),
                        "batched main compression",
                    )


def _warm_raw_producer_metadata(ratio, device):
    from types import SimpleNamespace

    import torch

    from ._v41_producer_metadata import prepare_raw

    # Slot kernels are separately warmed with the actual allocated pool schema.
    def slots(region, positions, requests, **kwargs):
        return torch.empty_like(positions)

    for lengths_host, prefixes_host in (
        ((3, 4), (0, 1)),
        ((4, 4), (0, 1)),
        ((1, 1), (0, 0)),
    ):
        cp = SimpleNamespace(
            cp_size=4,
            input_lengths_global_host=lengths_host,
            prefix_lengths_host=prefixes_host,
        )
        values = (
            [p for s, n in zip(prefixes_host, lengths_host) for p in range(s, s + n)],
            [r for r, n in enumerate(lengths_host) for _ in range(n)],
            prefixes_host,
            lengths_host,
        )
        # The raw path passes fresh contiguous position/ID and .long() vectors.
        vectors = [
            torch.tensor(host, dtype=torch.int64, device=device) for host in values
        ]
        _require_launch(
            prepare_raw(cp, *vectors, ratio, slots, (1, 2, 3), tile_rows=65536),
            "raw producer metadata",
        )


def _warm_fused_producer_metadata(layouts, cp_rank, device):
    from types import SimpleNamespace

    import torch

    from . import _v41_producer_metadata as producer

    ratio = layouts[0].ratio
    if len(layouts) != (3 if ratio == 2 else 2):
        return

    def unexpected_fallback(*args, **kwargs):
        raise RuntimeError("fused producer warmup used legacy slot callbacks")

    for table_dtype in (torch.int32, torch.int64):
        for length_dtype in (torch.int32, torch.int64):
            for lengths_host, prefixes_host in (
                ((3, 4), (0, 1)),
                ((4, 4), (0, 1)),
                ((1, 1), (0, 0)),
            ):
                cp = SimpleNamespace(
                    cp_size=4,
                    input_lengths_global_host=lengths_host,
                    prefix_lengths_host=prefixes_host,
                )
                positions = torch.tensor(
                    [
                        p
                        for s, n in zip(prefixes_host, lengths_host)
                        for p in range(s, s + n)
                    ],
                    dtype=torch.int64,
                    device=device,
                )
                requests = torch.tensor(
                    [r for r, n in enumerate(lengths_host) for _ in range(n)],
                    dtype=torch.int64,
                    device=device,
                )
                starts = torch.tensor(prefixes_host, dtype=length_dtype, device=device)
                lengths = torch.tensor(lengths_host, dtype=length_dtype, device=device)
                descriptors = tuple(
                    producer.SlotLayout(
                        torch.ones((2, 2), dtype=table_dtype, device=device),
                        layout.entries,
                        layout.tokens_per_block,
                        layout.owner_tokens_per_block,
                        4,
                        cp_rank,
                    )
                    for layout in layouts
                )
                result = _require_launch(
                    producer.prepare_raw(
                        cp,
                        positions,
                        requests,
                        starts,
                        lengths,
                        ratio,
                        unexpected_fallback,
                        (1, 2, 3),
                        tile_rows=65536,
                        slot_layouts=descriptors,
                    ),
                    "fused producer metadata",
                )
                if result.key_counts is None:
                    raise RuntimeError("fused producer warmup did not publish counts")


def _indexer_warmup_layouts():
    # Bounds returned by unbind([2,M])/[3,M] have aligned, half-word and odd
    # row offsets. These three M values cover all pointer-alignment classes
    # without specializing logical M/N or allocating real request-sized data.
    return tuple(
        (rows, stride, offset)
        for rows in (4, 2, 1)
        for stride in (1, 2)
        for offset in (0, 1)
    )


def _indexer_warmup_vector(rows, dtype, device, stride, offset, *, fill=None):
    import torch

    # Preserve the backing offset even for dtype changes: .to(dtype) on an
    # existing strided vector would allocate an aligned contiguous tensor and
    # silently miss the intended specialization. Production request slices may
    # start at a non-16B-aligned int64 position or int32 cached bound.
    size = offset + (rows - 1) * stride + 1
    storage = (
        torch.arange(size, dtype=dtype, device=device)
        if fill is None
        else torch.full((size,), fill, dtype=dtype, device=device)
    )
    return storage[offset::stride]


def _warm_indexer(attn, max_seq_len, device):
    for rows, stride, offset in _indexer_warmup_layouts():
        _warm_indexer_layout(
            attn,
            max_seq_len,
            device,
            rows=rows,
            vector_stride=stride,
            vector_offset=offset,
        )


def _warm_candidate_topk(device):
    from . import _v41_candidate_topk

    if _v41_candidate_topk._enabled() and not _v41_candidate_topk.warmup(device):
        logging.info("[DSV41 Attention] candidate TopK retains existing fallback")


def _warm_grouped_bounds(attn, device):
    import itertools

    import torch

    from ._v41_grouped_prefill_score import _all_grouped_score_bounds_kernel

    descriptors = torch.tensor(
        [(i + 1, 0, 2, 2 * i, 2, 320, 64, i) for i in range(4)],
        dtype=torch.int64,
        device=device,
    )
    output = torch.empty(32, dtype=torch.int32, device=device)
    # Query positions/IDs are long; batched selection publishes int32 counts.
    for po, ro, co in itertools.product((0, 1), repeat=3):
        positions = torch.ones(8 + po, dtype=torch.int64, device=device)[po:]
        requests = torch.zeros(8 + ro, dtype=torch.int64, device=device)[ro:]
        counts = torch.full((2 + co,), 64, dtype=torch.int32, device=device)[co:]
        _all_grouped_score_bounds_kernel[(4,)](
            positions,
            requests,
            counts,
            descriptors,
            output,
            2,
            4,
            attn.compress_ratio,
            num_warps=4,
        )


def _warm_indexer_layout(
    attn, max_seq_len, device, *, rows, vector_stride, vector_offset
):
    import torch

    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_deepselect as deepselect
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_indexer_q_triton as qfusion
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_candidates as candidates
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_indexer as indexer
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_metadata as meta
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_prefill_topk as topk
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_sparse_prefill_indexer as sparse

    if not indexer.is_supported(device, attn.index_n_heads, attn.index_head_dim):
        logging.info(
            "[DSV41 Attention] indexer unsupported on %s / H=%s D=%s",
            device,
            attn.index_n_heads,
            attn.index_head_dim,
        )
        return
    width = 1024
    heads = int(attn.index_n_heads)
    q = torch.zeros((rows, 1, heads, 128), dtype=torch.bfloat16, device=device)
    positions = _indexer_warmup_vector(
        rows, torch.int64, device, vector_stride, vector_offset
    )
    raw_weights = torch.ones((rows, 1, heads), dtype=torch.bfloat16, device=device)
    prepared = qfusion.try_fused_indexer_q(q, raw_weights, attn.freqs_cis, positions)
    _require_launch(
        prepared,
        "fused indexer Q",
        enabled=qfusion.is_supported(q, raw_weights, attn.freqs_cis, positions),
    )
    if prepared is None:
        q_fp4, q_sf = indexer.quantize_indexer_q(q.squeeze(1))
        weights = raw_weights.squeeze(1).float()
    else:
        q_fp4, q_sf, weights = (x.squeeze(1) for x in prepared)
    # Also compile the explicit quantizer used for unsupported fused-Q layouts.
    indexer.quantize_indexer_q(q.squeeze(1))
    keys = indexer.PrefillIndexerKeys(
        torch.zeros((width, 64), dtype=torch.int8, device=device),
        torch.full((width,), 0x7F7F7F7F, dtype=torch.int32, device=device),
    )
    bounds = meta.try_score_bounds(positions, width, attn.compress_ratio)
    _require_launch(bounds, "score bounds")
    visible = _indexer_warmup_vector(
        rows, torch.int32, device, vector_stride, vector_offset, fill=width
    )
    logits = indexer.score_indexer_chunk(
        q_fp4, q_sf, keys.quant, keys.scale, weights, visible, bounds=bounds
    )
    if heads == 32 and torch.cuda.get_device_capability(device)[0] == 10:
        from ._indexer_score import fp8_fp4_mqa_indexer_score
        from ._v41_grouped_prefill_score import (
            _grouped_score_bounds_kernel,
            _mask_tail_kernel,
        )

        grouped_positions = _indexer_warmup_vector(
            rows, torch.int64, device, 1, vector_offset, fill=width - 1
        )
        grouped_requests = _indexer_warmup_vector(
            rows, torch.int64, device, 1, vector_offset, fill=1
        )
        grouped_counts = torch.full((2,), width, dtype=torch.int32, device=device)
        grouped_bounds = torch.empty((4, rows), dtype=torch.int32, device=device)
        _grouped_score_bounds_kernel[(1,)](
            grouped_positions,
            grouped_requests,
            grouped_counts,
            grouped_bounds,
            rows,
            2,
            1,
            2,
            width,
            width,
            attn.compress_ratio,
            num_warps=4,
        )

        grouped_logits = fp8_fp4_mqa_indexer_score(
            q_fp4,
            q_sf,
            keys.quant,
            keys.scale,
            weights,
            *bounds,
            clean_logits=False,
            max_seqlen_k=width,
        )
        _mask_tail_kernel[(rows,)](
            grouped_logits, bounds[1], width, grouped_logits.stride(0), 256
        )
        del grouped_logits
    if attn.index_topk == 512:
        for dtype in (torch.int32, torch.int64):
            typed_visible = _indexer_warmup_vector(
                rows, dtype, device, vector_stride, vector_offset, fill=width
            )
            for cached in (bounds, None):
                selected = topk.try_select_tokens(logits, typed_visible, bounds=cached)
                _require_launch(
                    selected,
                    "dense token TopK",
                    enabled=topk.is_supported(logits, typed_visible),
                )
    config = attn.v41_config
    count = int(config.get("candidate_topk_blocks", 0))
    block = int(config.get("candidate_block_size", 0))
    if attn.compress_ratio != 1 or block <= 0 or count <= 0:
        return
    # Candidate pooling is independent of query row count. Cover the finite
    # bitmap buckets once per visible pointer/stride layout, not for every M.
    widths = _candidate_widths(max_seq_len, block, count) if rows == 4 else ()
    if rows == 4 and max_seq_len > 0:
        # Also cover the indices-only path where every candidate block fits.
        widths = tuple(sorted({*widths, min(max_seq_len, block * count)}))
    for width in widths:
        # Two rows bound the largest 512K-score startup buffer to 4 MiB.
        logits = torch.zeros((2, width + 256), dtype=torch.float32, device=device)[
            :, :width
        ]
        for dtype in (torch.int32, torch.int64):
            visible = _indexer_warmup_vector(
                2, dtype, device, vector_stride, vector_offset, fill=width
            )
            selected = candidates.select_candidates(
                logits, visible, block, count, build_bitmap=False
            )
            _require_launch(
                selected,
                "candidate selection",
                enabled=candidates.is_supported(logits, visible, block, count),
            )
            if dtype == torch.int32 and vector_stride == 1 and attn.index_topk == 512:
                token_indices = torch.zeros((2, 512), dtype=torch.int32, device=device)
                for build_bitmap in (False, True):
                    published = candidates.select_candidates(
                        logits,
                        visible,
                        block,
                        count,
                        build_bitmap=build_bitmap,
                        mask_tail=True,
                        token_indices=token_indices,
                        token_ends=visible,
                    )
                    _require_launch(
                        published,
                        "candidate publication and token filtering",
                        enabled=not build_bitmap
                        or candidates.bitmap_is_bounded(2, width, block),
                    )
            if selected is not None:
                ids = selected[0]
                dense = candidates.select_candidates(logits, visible, block, count)
                _require_launch(
                    dense,
                    "candidate bitmap",
                    enabled=candidates.bitmap_is_bounded(2, width, block),
                )
                if dense is not None:
                    _require_launch(
                        candidates.mask_candidates(logits, dense[1], block),
                        "candidate mask",
                    )
                flags = candidates.build_flags(ids, width, block)
                _require_launch(
                    flags,
                    "candidate bitmap rebuild",
                    enabled=candidates.bitmap_is_bounded(2, width, block),
                )
                if flags is not None:
                    _require_launch(
                        candidates.mask_candidates(logits, flags, block),
                        "rebuilt candidate mask",
                    )
    if (
        max_seq_len <= count * block
        or heads != 32
        or attn.index_topk != 512
        or not deepselect.is_available(device)
    ):
        return
    width = count * block + block
    ids = torch.arange(count, dtype=torch.int32, device=device)[None].repeat(rows, 1)
    for dtype in (torch.int32, torch.int64):
        visible = _indexer_warmup_vector(
            rows, dtype, device, vector_stride, vector_offset, fill=width
        )
        plan = sparse.prepare_plan(ids, visible, width, block)
        _require_launch(plan, "sparse index plan")
        if plan is not None:
            keys = indexer.PrefillIndexerKeys(
                torch.zeros((width, 64), dtype=torch.int8, device=device),
                torch.full((width,), 0x7F7F7F7F, dtype=torch.int32, device=device),
            )
            logits = sparse.score(q_fp4, q_sf, keys, weights, plan)
            _require_launch(logits, "sparse score")
            selected = deepselect.try_select_sparse_tokens(logits, plan.end)
            _require_launch(selected, "BF16 DeepSelect")
            _require_launch(
                sparse.remap(selected, plan, logits=logits), "sparse logical remap"
            )
            # The offset variant uses the same bounded sorting tile and DG
            # scorer; M and request-specific K offsets remain runtime values.
            padded_width = (width + 255) // 256 * 256
            joined_keys = indexer.PrefillIndexerKeys(
                torch.zeros(
                    (padded_width + width, 64), dtype=torch.int8, device=device
                ),
                torch.full(
                    (padded_width + width,),
                    0x7F7F7F7F,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            requests = _indexer_warmup_vector(
                rows, torch.int64, device, 1, vector_offset, fill=1
            )
            counts = torch.full((2,), width, dtype=torch.int32, device=device)
            batched_plan = sparse.prepare_plan(
                ids,
                visible,
                padded_width + width,
                block,
                request_ids=requests,
                request_key_counts=counts,
            )
            for ratio in (1, 2):
                _require_launch(
                    sparse.prepare_plan(
                        ids,
                        visible,
                        padded_width + width,
                        block,
                        request_ids=requests,
                        request_key_counts=counts,
                        positions_ratio=ratio,
                    ),
                    "batched sparse position bounds",
                )
            _require_launch(batched_plan, "batched sparse index plan")
            batched_logits = sparse.score(
                q_fp4, q_sf, joined_keys, weights, batched_plan
            )
            _require_launch(batched_logits, "batched sparse score")
            selected = deepselect.try_select_sparse_tokens(
                batched_logits, batched_plan.end
            )
            _require_launch(selected, "batched sparse selection")
            _require_launch(
                sparse.remap(selected, batched_plan, logits=batched_logits),
                "batched sparse logical remap",
            )


def _warm_flash_mla(attn, device):
    import torch
    from flash_mla import flash_mla_sparse_fwd

    from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
        _warmup_combine_topk_swa_indices_cp,
    )

    width = attn.window_size + (attn.index_topk if attn.compress_ratio else 0)
    width = (width + 63) // 64 * 64
    q = torch.zeros(
        (2, attn.n_heads, attn.head_dim), dtype=torch.bfloat16, device=device
    )
    kv = torch.zeros((4, 1, attn.head_dim), dtype=torch.bfloat16, device=device)
    indices = torch.full((2, 1, width), -1, dtype=torch.int32, device=device)
    indices[:, :, :4] = torch.arange(4, dtype=torch.int32, device=device)
    lengths = torch.full((2,), 4, dtype=torch.int32, device=device)
    if attn.compress_ratio:
        from ._v41_prefill_index_plan import try_build_index_plan

        selected = torch.full(
            (2, attn.index_topk), -1, dtype=torch.int32, device=device
        )
        for dtype in (torch.int32, torch.int64):
            positions = torch.arange(2, dtype=dtype, device=device)
            offsets = torch.zeros((2, 1), dtype=dtype, device=device)
            sizes = torch.ones((2, 1), dtype=dtype, device=device)
            _require_launch(
                try_build_index_plan(
                    selected, positions, offsets, sizes, offsets, attn.window_size
                ),
                "batched FlashMLA index plan",
            )
    flash_mla_sparse_fwd(
        q=q,
        kv=kv,
        indices=indices,
        sm_scale=attn.softmax_scale,
        attn_sink=attn.attn_sink,
        topk_length=lengths,
    )
    if not attn.compress_ratio:
        _warmup_combine_topk_swa_indices_cp(
            window_size=attn.window_size, compress_ratio=1, topk=0, device=device
        )


def _warm_q_norm_rope(attn, device):
    import torch

    from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import (
        fused_rmsnorm_rope,
    )
    from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import strided_q_rmsnorm
    from rtp_llm.models_py.modules.dsv4.fp8.attention_v41 import rope_only

    rows = 2
    frequencies = attn.freqs_cis[:rows]
    q = torch.zeros(
        (rows, attn.n_heads, attn.head_dim), dtype=torch.bfloat16, device=device
    )
    rope_only(q, frequencies, attn.rope_head_dim)
    rank = attn.q_lora_rank
    combined = torch.zeros(
        (1, rows, rank + attn.head_dim), dtype=torch.bfloat16, device=device
    )
    strided_q_rmsnorm(combined[..., :rank], attn.q_norm, attn.eps)
    fused_rmsnorm_rope(
        combined[..., rank:],
        attn.kv_norm,
        frequencies,
        attn.rope_head_dim,
        eps=attn.eps,
    )
    # The unfused projection's contiguous KV is also reachable via the flag.
    fused_rmsnorm_rope(
        combined[..., rank:].contiguous(),
        attn.kv_norm,
        frequencies,
        attn.rope_head_dim,
        eps=attn.eps,
    )


def warmup_v41_attention_jit(
    v4,
    *,
    max_seq_len,
    max_m,
    max_batch_size,
    cp_size,
    cp_rank,
    kv_cache_sharded,
    device,
    kv_cache=None,
):
    """Prewarm reachable variants; a later cache-bearing call completes pools."""
    import torch

    from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
        _assert_not_capturing,
        _run_deepgemm_warmup_launch_with_retry,
        _run_deepgemm_warmup_launches_serialized,
        _run_triton_warmup_launch_with_retry,
    )
    from rtp_llm.utils.warmup import model_warm_up_enabled

    device = torch.device(device)
    if not model_warm_up_enabled() or device.type != "cuda" or max_m <= 0:
        return
    _assert_not_capturing()
    if torch.cuda.get_device_capability(device)[0] != 10:
        return
    attentions = _collect_attentions(v4)
    stream = torch.cuda.current_stream(device)
    prefix = (device.index, stream.cuda_stream)

    def run(key, fn, deepgemm=False):
        key = prefix + key
        if key in _WARMED:
            return
        launch = partial(
            _run_triton_warmup_launch_with_retry,
            "DSV41 Attention",
            str(key),
            fn,
            device=device,
        )
        if deepgemm:
            launch = partial(
                _run_deepgemm_warmup_launch_with_retry,
                "DSV41 Attention",
                str(key),
                launch,
                device=device,
            )
            _run_deepgemm_warmup_launches_serialized("DSV41 Attention", launch)
        else:
            launch()
        torch.cuda.synchronize(device)
        _WARMED.add(key)

    with torch.inference_mode(), torch.cuda.device(device):
        for key, attn in attentions.items():
            run(("flash_mla", key), partial(_warm_flash_mla, attn, device))
            run(("q_norm_rope", key), partial(_warm_q_norm_rope, attn, device))
            if getattr(attn, "is_index_source", False) and attn.compress_ratio:
                run(("candidate_topk",), partial(_warm_candidate_topk, device))
                if attn.index_n_heads == 32:
                    run(
                        ("grouped_bounds", key),
                        partial(_warm_grouped_bounds, attn, device),
                    )
                run(
                    ("indexer", key, max_seq_len),
                    partial(_warm_indexer, attn, max_seq_len, device),
                    True,
                )
        if kv_cache is None:
            logging.info(
                "[DSV41 Attention] pool-layout warmup deferred until KV allocation"
            )
        else:
            layouts = _collect_pool_layouts(
                attentions, kv_cache, cp_size, kv_cache_sharded
            )
            for (layout, attn_key), attn in layouts.items():
                size = cp_size if kv_cache_sharded else 1
                rank = cp_rank if kv_cache_sharded else 0
                run(
                    ("pool", layout, attn_key, size, rank, max_batch_size),
                    partial(
                        _warm_pool, attn, layout, size, rank, max_batch_size, device
                    ),
                )
            if kv_cache_sharded and cp_size == 4 and max_batch_size >= 2:
                from rtp_llm.models_py.modules.dsv4.attn_type import (
                    CSA_KV,
                    CSA_STATE,
                    HCA_KV,
                    INDEXER_KV,
                )

                paired = {}
                for layout, attn_key in layouts:
                    paired.setdefault(attn_key, {})[layout.region] = layout
                for regions in paired.values():
                    index_layout = regions.get(INDEXER_KV)
                    main_layout = regions.get(CSA_KV, regions.get(HCA_KV))
                    if main_layout is None or index_layout is None:
                        continue
                    producer_layouts = (main_layout, index_layout)
                    if main_layout.ratio == 2:
                        state_layout = regions.get(CSA_STATE)
                        producer_layouts += (
                            (state_layout,) if state_layout is not None else ()
                        )
                    if len(producer_layouts) == (3 if main_layout.ratio == 2 else 2):
                        run(
                            ("fused_producer", producer_layouts, cp_rank),
                            partial(
                                _warm_fused_producer_metadata,
                                producer_layouts,
                                cp_rank,
                                device,
                            ),
                        )
                    if max_batch_size >= 32:
                        run(
                            ("joint_pool", main_layout, index_layout, cp_rank),
                            partial(
                                _warm_joint_pool,
                                main_layout,
                                index_layout,
                                cp_rank,
                                device,
                            ),
                        )
            logging.info(
                "[DSV41 Attention] warmed %d private pool layouts", len(layouts)
            )


__all__ = ["warmup_v41_attention_jit"]
