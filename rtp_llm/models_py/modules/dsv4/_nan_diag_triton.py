"""CUDA-graph-safe non-finite diagnostics for DeepSeek-V4.

Set ``DSV4_NAN_DIAG=1`` before starting the server to add read-only detector
kernels to important DSV4 numerical boundaries. A detector prints at most one
structured event per ``(batch, source, layer)`` even when an entire tensor is
non-finite. Unmapped startup and CUDA-graph-capture batches use batch id zero
and are emitted with ``trace_status=unresolved`` by the host-side reliable drain.
Triton includes the winning ``pid (row, tile, 0)`` in its auxiliary line.
The first printed integer is the model batch id. The host-side reliable event
drain emits the exact request trace id and request/query coordinates. The
second integer is this auxiliary payload for the first reported 256-element
tile:

    source * 10^12 + layer(3) | first-column-in-tile(3) | n_nan(3) | n_inf(3)

For example, ``1017007001000`` means source 1, layer 17, first bad value at
tile offset 7, one NaN, and zero Inf values.

Source ids:
    1 = MoE activation input
    2 = router linear scores
    3 = router bias
    4 = context-parallel attention LSE
    5 = final hidden state
    6 = attention query
    7 = KV value before cache quantization/write
    8 = packed SWA KV cache read
    9 = packed compressed KV cache read
   10 = attention output
   11 = packed SWA KV cache immediately after a local write
   12 = packed compressed KV cache immediately after a local write
   13 = packed indexer KV cache immediately after a local write
   14 = packed indexer KV cache immediately before score computation
   15 = indexer score output before top-k selection
   16 = indexer query before FP8 quantization
   17 = indexer projected weights before FP8 folding
   18 = indexer query after FP8 quantization
   19 = indexer projected weights after FP8 folding

For an end-to-end service test only, a guarded injector is available. Both
variables are required, otherwise startup fails instead of silently changing
model data::

    DSV4_NAN_DIAG_TEST_INJECT=2,0,0
    DSV4_NAN_DIAG_TEST_INJECT_CONFIRM=I_UNDERSTAND_THIS_CHANGES_OUTPUT

The injection tuple is ``layer,row,col`` and writes one NaN into that layer's
MoE activation before the read-only detector runs. Injection is skipped while
the diagnostic batch id is 0, so startup/warmup forwards remain unchanged and
a real request can carry the injected event's trace id.

Packed FP8 KV-cache reads have a separate guarded corruption injector for
end-to-end tests. ``layer,pool,kind`` accepts pool ``swa`` or ``compressed``
and kind ``fp8``, ``rope``, or ``scale``::

    DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE=2,swa,fp8
    DSV4_NAN_DIAG_TEST_INJECT_CONFIRM=I_UNDERSTAND_THIS_CHANGES_OUTPUT

The detector is a separate kernel: it never writes the inspected tensor or any
model output.  During CUDA graph capture it becomes part of the graph, so the
same check runs and reports on every replay without a host sync or ``.item()``.
"""

from __future__ import annotations

import logging
import os

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - CPU-only import
    triton = None
    tl = None


ENABLED = os.environ.get("DSV4_NAN_DIAG", "0") == "1"

_TEST_INJECT_SPEC = os.environ.get("DSV4_NAN_DIAG_TEST_INJECT", "").strip()
_TEST_KV_CORRUPT_SPEC = os.environ.get(
    "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE", ""
).strip()
_TEST_INJECT_CONFIRM = os.environ.get("DSV4_NAN_DIAG_TEST_INJECT_CONFIRM", "").strip()
_TEST_INJECT_CONFIRM_VALUE = "I_UNDERSTAND_THIS_CHANGES_OUTPUT"


def _parse_test_inject_spec(spec: str) -> tuple[int, int, int] | None:
    if not spec:
        return None
    if not ENABLED:
        raise RuntimeError("DSV4_NAN_DIAG_TEST_INJECT requires DSV4_NAN_DIAG=1")
    if _TEST_INJECT_CONFIRM != _TEST_INJECT_CONFIRM_VALUE:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_INJECT is test-only; set "
            "DSV4_NAN_DIAG_TEST_INJECT_CONFIRM="
            f"{_TEST_INJECT_CONFIRM_VALUE} to acknowledge output mutation"
        )
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT must be layer,row,col, got " f"{spec!r}"
        )
    layer, row, col = (int(part) for part in parts)
    if min(layer, row, col) < 0:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT coordinates must be non-negative, "
            f"got {spec!r}"
        )
    return layer, row, col


TEST_INJECT = _parse_test_inject_spec(_TEST_INJECT_SPEC)

SOURCE_MOE_INPUT = 1
SOURCE_ROUTER_SCORES = 2
SOURCE_ROUTER_BIAS = 3
SOURCE_CP_ATTENTION_LSE = 4
SOURCE_FINAL_HIDDEN = 5
SOURCE_ATTENTION_QUERY = 6
SOURCE_KV_WRITE_INPUT = 7
SOURCE_SWA_KV_CACHE_READ = 8
SOURCE_COMPRESSED_KV_CACHE_READ = 9
SOURCE_ATTENTION_OUTPUT = 10
SOURCE_SWA_KV_CACHE_WRITE = 11
SOURCE_COMPRESSED_KV_CACHE_WRITE = 12
SOURCE_INDEXER_KV_CACHE_WRITE = 13
SOURCE_INDEXER_KV_CACHE_READ = 14
SOURCE_INDEXER_SCORE = 15
SOURCE_INDEXER_QUERY = 16
SOURCE_INDEXER_WEIGHTS = 17
SOURCE_INDEXER_QUERY_FP8 = 18
SOURCE_INDEXER_WEIGHTS_FOLDED = 19

KV_KIND_UNKNOWN = 0
KV_KIND_SWA = 1
KV_KIND_CSA = 2
KV_KIND_HCA = 3
KV_KIND_INDEXER = 4

_KV_CORRUPT_POOL_TO_SOURCE = {
    "swa": SOURCE_SWA_KV_CACHE_READ,
    "compressed": SOURCE_COMPRESSED_KV_CACHE_READ,
    "indexer": SOURCE_INDEXER_KV_CACHE_READ,
}
_KV_CORRUPT_KIND = {"fp8": 1, "rope": 2, "scale": 3}


def _parse_test_kv_corrupt_spec(spec: str) -> tuple[int, int, int] | None:
    if not spec:
        return None
    if not ENABLED:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE requires DSV4_NAN_DIAG=1"
        )
    if _TEST_INJECT_CONFIRM != _TEST_INJECT_CONFIRM_VALUE:
        raise RuntimeError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE is test-only; set "
            "DSV4_NAN_DIAG_TEST_INJECT_CONFIRM="
            f"{_TEST_INJECT_CONFIRM_VALUE} to acknowledge cache mutation"
        )
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE must be layer,pool,kind, got "
            f"{spec!r}"
        )
    layer_text, pool, kind = parts
    layer = int(layer_text)
    if (
        layer < 0
        or pool not in _KV_CORRUPT_POOL_TO_SOURCE
        or kind not in _KV_CORRUPT_KIND
        or (pool == "indexer" and kind == "rope")
    ):
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_CORRUPT_KV_CACHE expects a non-negative layer, "
            "pool swa|compressed|indexer, and kind fp8|rope|scale; got "
            f"{spec!r}"
        )
    return layer, _KV_CORRUPT_POOL_TO_SOURCE[pool], _KV_CORRUPT_KIND[kind]


TEST_KV_CORRUPT = _parse_test_kv_corrupt_spec(_TEST_KV_CORRUPT_SPEC)

_BLOCK_N = 256
_DEFAULT_PRINTF_FIFO_MB = 64
_DEFAULT_EVENT_CAPACITY = 4096
_EVENT_FIELDS = 16
_MAX_SOURCE_ID = 19
_MAX_LAYER_ID = 999
_STATE_LAYERS = _MAX_LAYER_ID + 2  # layer -1 (unscoped) plus layers 0..999
_PREWARMED_DEVICES: set[str] = set()
_BATCH_ID_TENSORS: dict[str, torch.Tensor] = {}
_LAST_REPORTED_BATCH_BY_DEVICE: dict[str, torch.Tensor] = {}
_REPORT_COUNT_BY_DEVICE: dict[str, torch.Tensor] = {}
_EVENT_COUNTERS_BY_DEVICE: dict[str, torch.Tensor] = {}
_EVENT_RECORDS_BY_DEVICE: dict[str, torch.Tensor] = {}
# Uniform request layout for decode / target-verify.  The Python forward sets
# this once before the first layer.  It is process-local metadata only: all
# event coordinates themselves are still written by graph-captured kernels.
_REQUEST_LAYOUT_BY_DEVICE: dict[str, tuple[int, int]] = {}


if triton is not None:

    @triton.jit(do_not_specialize=["row", "col", "stride_row", "stride_col"])
    def _inject_nan_kernel(
        tensor_ptr,
        batch_id_ptr,
        row,
        col,
        stride_row,
        stride_col,
    ):
        if tl.load(batch_id_ptr).to(tl.int64) != 0:
            tl.store(
                tensor_ptr + row * stride_row + col * stride_col,
                float("nan"),
            )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "cols",
            "stride_row",
            "stride_col",
            "source_id",
            "layer_id",
            "state_index",
            "include_neg_inf",
            "event_capacity",
            "rows_per_request",
            "rows_per_query",
            "row_offset",
        ]
    )
    def _report_nonfinite_tiles_kernel(
        tensor_ptr,
        batch_id_ptr,
        last_reported_batch_ptr,
        report_count_ptr,
        event_counters_ptr,
        event_records_ptr,
        rows,
        cols,
        stride_row,
        stride_col,
        source_id,
        layer_id,
        state_index,
        include_neg_inf,
        event_capacity,
        rows_per_request,
        rows_per_query,
        row_offset,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        tile = tl.program_id(1).to(tl.int64)
        col_start = tile * BLOCK_N
        col_offsets = col_start + tl.arange(0, BLOCK_N)
        mask = (row < rows) & (col_offsets < cols)
        values = tl.load(
            tensor_ptr + row * stride_row + col_offsets * stride_col,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        is_nan = mask & (values != values)
        is_pos_inf = mask & (values == float("inf"))
        is_neg_inf = mask & (values == -float("inf"))
        is_inf = is_pos_inf | (is_neg_inf & (include_neg_inf != 0))
        is_bad = is_nan | is_inf
        n_nan = tl.sum(is_nan.to(tl.int32), axis=0)
        n_inf = tl.sum(is_inf.to(tl.int32), axis=0)
        first_col = tl.min(
            tl.where(is_bad, col_offsets, cols),
            axis=0,
        )

        # device_print runs once per active Triton lane unless explicitly
        # guarded. Keep the scan vectorized, but let only CUDA thread 0 emit.
        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if (n_nan + n_inf > 0) & (thread_idx == 0):
            batch_id = tl.load(batch_id_ptr).to(tl.int64)
            # Batch zero has no trace mapping. Give each forward/replay a
            # negative device epoch for deduplication, but preserve batch=0 in
            # the event so the host emits an explicit trace_status=unmapped log.
            epoch = tl.load(event_counters_ptr + 2).to(tl.int64)
            dedupe_id = tl.where(batch_id != 0, batch_id, -epoch)
            previous_batch = tl.atomic_xchg(
                last_reported_batch_ptr + state_index,
                dedupe_id,
            )
            if previous_batch != dedupe_id:
                tl.atomic_add(report_count_ptr + state_index, 1)
                safe_rows_per_request = tl.maximum(rows_per_request, 1)
                safe_rows_per_query = tl.maximum(rows_per_query, 1)
                event_row = row + row_offset
                request_index = tl.where(
                    rows_per_request > 0,
                    event_row // safe_rows_per_request,
                    -1,
                )
                row_in_request = event_row % safe_rows_per_request
                query_index = tl.where(
                    (rows_per_request > 0) & (rows_per_query > 0),
                    row_in_request // safe_rows_per_query,
                    -1,
                )
                subrow = tl.where(
                    (rows_per_request > 0) & (rows_per_query > 0),
                    row_in_request % safe_rows_per_query,
                    -1,
                )
                q_len = tl.where(
                    (rows_per_request > 0) & (rows_per_query > 0),
                    rows_per_request // safe_rows_per_query,
                    -1,
                )
                event_index = tl.atomic_add(event_counters_ptr, 1)
                if event_index < event_capacity:
                    record = event_records_ptr + event_index * 16
                    tl.store(record + 0, batch_id)
                    tl.store(record + 1, source_id.to(tl.int64))
                    tl.store(record + 2, layer_id.to(tl.int64))
                    tl.store(record + 3, first_col.to(tl.int64))
                    tl.store(record + 4, n_nan.to(tl.int64))
                    tl.store(record + 5, n_inf.to(tl.int64))
                    tl.store(record + 6, -1)
                    tl.store(record + 7, -1)
                    tl.store(record + 8, request_index.to(tl.int64))
                    tl.store(record + 9, query_index.to(tl.int64))
                    tl.store(record + 10, event_row.to(tl.int64))
                    tl.store(record + 11, first_col.to(tl.int64))
                    tl.store(record + 12, subrow.to(tl.int64))
                    tl.store(record + 13, -1)
                    tl.store(record + 14, -1)
                    tl.store(record + 15, q_len.to(tl.int64))
                else:
                    tl.atomic_add(event_counters_ptr + 1, 1)
                first_offset = first_col - col_start
                event = (
                    source_id.to(tl.int64) * 1_000_000_000_000
                    + layer_id.to(tl.int64) * 1_000_000_000
                    + first_offset * 1_000_000
                    + tl.minimum(n_nan, 999).to(tl.int64) * 1_000
                    + tl.minimum(n_inf, 999).to(tl.int64)
                )
                tl.device_print(
                    "[DSV4_NAN] batch,event=source(2d)layer(3d)"
                    "first_offset(3d)n_nan(3d)n_inf(3d):",
                    batch_id,
                    event,
                )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "width",
            "q_len",
            "num_cache_blocks",
            "cache_block_size",
            "block_stride",
            "source_id",
            "layer_id",
            "kv_kind",
            "state_index",
            "event_capacity",
        ]
    )
    def _report_packed_fp8_kv_cache_kernel(
        cache_ptr,
        indices_ptr,
        lengths_ptr,
        batch_id_ptr,
        last_reported_batch_ptr,
        report_count_ptr,
        event_counters_ptr,
        event_records_ptr,
        rows,
        width,
        q_len,
        num_cache_blocks,
        cache_block_size,
        block_stride,
        source_id,
        layer_id,
        kv_kind,
        state_index,
        event_capacity,
        HAS_LENGTHS: tl.constexpr,
        LENGTHS_PER_ROW: tl.constexpr,
        CORRUPT_KIND: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Scan the exact packed FP8 cache slots consumed by FlashMLA.

        The logical ``[..., 584]`` view is not token-contiguous inside a
        physical block. Bytes are laid out as all 576-byte token payloads,
        followed by all 8-byte scale payloads. This mirrors the production
        writer and FlashMLA reader rather than relying on tensor stride(1).
        """
        linear = tl.program_id(0).to(tl.int64)
        row = linear // width
        col = linear % width
        valid_width = width
        if HAS_LENGTHS:
            if LENGTHS_PER_ROW:
                length_idx = row
            else:
                length_idx = row // q_len
            valid_width = tl.load(lengths_ptr + length_idx).to(tl.int64)
        slot = tl.load(indices_ptr + linear).to(tl.int64)
        valid_slot = (
            (row < rows)
            & (col < valid_width)
            & (slot >= 0)
            & (slot < num_cache_blocks * cache_block_size)
        )

        block = slot // cache_block_size
        pos = slot % cache_block_size
        block_ptr = cache_ptr + block * block_stride
        token_data_ptr = block_ptr + pos * 576
        token_scale_ptr = block_ptr + cache_block_size * 576 + pos * 8

        # Test-only strong corruption. Every referenced instance of the
        # selected slot is assigned the same byte pattern, so duplicate top-k
        # indices are harmless and no auxiliary state/allocation is needed.
        if CORRUPT_KIND == 1:
            tl.store(token_data_ptr, 0x7F, mask=valid_slot)
        elif CORRUPT_KIND == 2:
            tl.store(token_data_ptr + 448, 0xC1, mask=valid_slot)
            tl.store(token_data_ptr + 449, 0x7F, mask=valid_slot)
        elif CORRUPT_KIND == 3:
            tl.store(token_scale_ptr, 0xFF, mask=valid_slot)

        lanes = tl.arange(0, BLOCK_N)

        # E4M3FN has exactly two NaN encodings: 0x7f and 0xff.
        fp8_byte = tl.load(
            token_data_ptr + lanes,
            mask=valid_slot & (lanes < 448),
            other=0,
        )
        bad_fp8 = valid_slot & (lanes < 448) & ((fp8_byte == 0x7F) | (fp8_byte == 0xFF))

        # The 64 RoPE elements are stored verbatim as little-endian BF16.
        rope_ptr = (token_data_ptr + 448).to(tl.pointer_type(tl.uint16))
        rope_bits = tl.load(
            rope_ptr + lanes,
            mask=valid_slot & (lanes < 64),
            other=0,
        )
        bad_rope = (
            valid_slot
            & (lanes < 64)
            & ((rope_bits & 0x7F80) == 0x7F80)
            & ((rope_bits & 0x007F) != 0)
        )

        # The eighth scale byte is padding. 0xff is NaN in E8M0FNU.
        scale_byte = tl.load(
            token_scale_ptr + lanes,
            mask=valid_slot & (lanes < 7),
            other=0,
        )
        bad_scale = valid_slot & (lanes < 7) & (scale_byte == 0xFF)

        n_fp8 = tl.sum(bad_fp8.to(tl.int32), axis=0)
        n_rope = tl.sum(bad_rope.to(tl.int32), axis=0)
        n_scale = tl.sum(bad_scale.to(tl.int32), axis=0)
        n_bad = n_fp8 + n_rope + n_scale
        first_bad_byte = tl.min(
            tl.where(
                bad_fp8,
                lanes,
                tl.where(
                    bad_rope,
                    448 + lanes * 2,
                    tl.where(bad_scale, 576 + lanes, 584),
                ),
            ),
            axis=0,
        )
        kind_bitmap = (
            (n_fp8 > 0).to(tl.int64)
            + (n_rope > 0).to(tl.int64) * 2
            + (n_scale > 0).to(tl.int64) * 4
        )

        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if (n_bad > 0) & (thread_idx == 0):
            batch_id = tl.load(batch_id_ptr).to(tl.int64)
            epoch = tl.load(event_counters_ptr + 2).to(tl.int64)
            dedupe_id = tl.where(batch_id != 0, batch_id, -epoch)
            previous_batch = tl.atomic_xchg(
                last_reported_batch_ptr + state_index,
                dedupe_id,
            )
            if previous_batch != dedupe_id:
                tl.atomic_add(report_count_ptr + state_index, 1)
                event_index = tl.atomic_add(event_counters_ptr, 1)
                if event_index < event_capacity:
                    record = event_records_ptr + event_index * 16
                    tl.store(record + 0, batch_id)
                    tl.store(record + 1, source_id.to(tl.int64))
                    tl.store(record + 2, layer_id.to(tl.int64))
                    tl.store(record + 3, first_bad_byte.to(tl.int64))
                    tl.store(record + 4, n_fp8.to(tl.int64))
                    tl.store(record + 5, n_rope.to(tl.int64))
                    tl.store(record + 6, n_scale.to(tl.int64))
                    tl.store(record + 7, slot)
                    tl.store(record + 8, row // q_len)
                    tl.store(record + 9, row % q_len)
                    tl.store(record + 10, row)
                    tl.store(record + 11, col)
                    tl.store(record + 12, kv_kind.to(tl.int64))
                    tl.store(record + 13, block)
                    tl.store(record + 14, pos)
                    tl.store(record + 15, q_len)
                else:
                    tl.atomic_add(event_counters_ptr + 1, 1)
                event = (
                    source_id.to(tl.int64) * 1_000_000_000_000
                    + layer_id.to(tl.int64) * 1_000_000_000
                    + tl.minimum(first_bad_byte, 999).to(tl.int64) * 1_000_000
                    + tl.minimum(n_bad, 999).to(tl.int64) * 1_000
                    + kind_bitmap
                )
                tl.device_print(
                    "[DSV4_NAN] batch,event=source(2d)layer(3d)"
                    "first_byte(3d)n_bad(3d)kind_bitmap(3d):",
                    batch_id,
                    event,
                )
                tl.device_print(
                    "[DSV4_KV_NAN] batch,slot:",
                    batch_id,
                    slot,
                )
                detail = (
                    source_id.to(tl.int64) * 1_000_000_000
                    + layer_id.to(tl.int64) * 1_000_000
                    + tl.minimum(n_fp8, 999).to(tl.int64) * 1_000
                    + tl.minimum(n_rope + n_scale, 999).to(tl.int64)
                )
                tl.device_print(
                    "[DSV4_KV_NAN] source_layer,n_fp8_n_rope_scale:",
                    detail,
                    n_rope.to(tl.int64) * 1_000 + n_scale.to(tl.int64),
                )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "width",
            "q_len",
            "num_cache_blocks",
            "cache_block_size",
            "block_stride",
            "map_stride",
            "source_id",
            "layer_id",
            "state_index",
            "event_capacity",
        ]
    )
    def _report_packed_fp8_indexer_cache_kernel(
        cache_ptr,
        map_ptr,
        lengths_ptr,
        batch_id_ptr,
        last_reported_batch_ptr,
        report_count_ptr,
        event_counters_ptr,
        event_records_ptr,
        rows,
        width,
        q_len,
        num_cache_blocks,
        cache_block_size,
        block_stride,
        map_stride,
        source_id,
        layer_id,
        state_index,
        event_capacity,
        EXPLICIT_SLOTS: tl.constexpr,
        CORRUPT_KIND: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Scan the 132-byte indexer K layout consumed by DeepGEMM."""
        linear = tl.program_id(0).to(tl.int64)
        row = linear // width
        col = linear % width
        request_index = row // q_len
        query_index = row % q_len
        if EXPLICIT_SLOTS:
            slot = tl.load(map_ptr + linear).to(tl.int64)
            valid_position = True
        else:
            valid_length = tl.load(lengths_ptr + row).to(tl.int64)
            logical_block = col // cache_block_size
            physical_block = tl.load(
                map_ptr + request_index * map_stride + logical_block,
                mask=(row < rows) & (col < valid_length),
                other=-1,
            ).to(tl.int64)
            slot = physical_block * cache_block_size + col % cache_block_size
            valid_position = col < valid_length

        block = slot // cache_block_size
        pos = slot % cache_block_size
        valid_slot = (
            (row < rows)
            & valid_position
            & (slot >= 0)
            & (block >= 0)
            & (block < num_cache_blocks)
        )
        block_ptr = cache_ptr + block * block_stride
        k_ptr = block_ptr + pos * 128
        scale_ptr = (block_ptr + cache_block_size * 128 + pos * 4).to(
            tl.pointer_type(tl.float32)
        )

        if CORRUPT_KIND == 1:
            tl.store(k_ptr, 0x7F, mask=valid_slot)
        elif CORRUPT_KIND == 3:
            tl.store(scale_ptr, float("nan"), mask=valid_slot)

        lanes = tl.arange(0, BLOCK_N)
        fp8_byte = tl.load(
            k_ptr + lanes,
            mask=valid_slot & (lanes < 128),
            other=0,
        )
        bad_fp8 = valid_slot & (lanes < 128) & ((fp8_byte == 0x7F) | (fp8_byte == 0xFF))
        scale = tl.load(scale_ptr, mask=valid_slot, other=0.0)
        bad_scale_scalar = valid_slot & (
            (scale != scale) | (scale == float("inf")) | (scale == -float("inf"))
        )
        n_fp8 = tl.sum(bad_fp8.to(tl.int32), axis=0)
        n_scale = bad_scale_scalar.to(tl.int32)
        n_bad = n_fp8 + n_scale
        first_bad_byte = tl.min(
            tl.where(bad_fp8, lanes, tl.where(bad_scale_scalar, 128, 132)),
            axis=0,
        )

        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if (n_bad > 0) & (thread_idx == 0):
            batch_id = tl.load(batch_id_ptr).to(tl.int64)
            epoch = tl.load(event_counters_ptr + 2).to(tl.int64)
            dedupe_id = tl.where(batch_id != 0, batch_id, -epoch)
            previous_batch = tl.atomic_xchg(
                last_reported_batch_ptr + state_index,
                dedupe_id,
            )
            if previous_batch != dedupe_id:
                tl.atomic_add(report_count_ptr + state_index, 1)
                event_index = tl.atomic_add(event_counters_ptr, 1)
                if event_index < event_capacity:
                    record = event_records_ptr + event_index * 16
                    tl.store(record + 0, batch_id)
                    tl.store(record + 1, source_id.to(tl.int64))
                    tl.store(record + 2, layer_id.to(tl.int64))
                    tl.store(record + 3, first_bad_byte.to(tl.int64))
                    tl.store(record + 4, n_fp8.to(tl.int64))
                    tl.store(record + 5, 0)
                    tl.store(record + 6, n_scale.to(tl.int64))
                    tl.store(record + 7, slot)
                    tl.store(record + 8, request_index.to(tl.int64))
                    tl.store(record + 9, query_index.to(tl.int64))
                    tl.store(record + 10, row.to(tl.int64))
                    tl.store(record + 11, col.to(tl.int64))
                    tl.store(record + 12, 4)
                    tl.store(record + 13, block)
                    tl.store(record + 14, pos)
                    tl.store(record + 15, q_len)
                else:
                    tl.atomic_add(event_counters_ptr + 1, 1)
                tl.device_print(
                    "[DSV4_INDEXER_KV_NAN] batch,source,layer,slot:",
                    batch_id,
                    source_id.to(tl.int64),
                    layer_id.to(tl.int64),
                    slot,
                )

    @triton.jit
    def _device_printf_canary_kernel():
        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if thread_idx == 0:
            tl.device_print("[DSV4_NAN_DIAG_CANARY] device-printf-ready:", 1)

    @triton.jit
    def _reset_event_state_kernel(event_counters_ptr):
        thread_idx = tl.inline_asm_elementwise(
            asm="mov.u32 $0, %tid.x;",
            constraints="=r",
            args=[],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        if thread_idx == 0:
            tl.store(event_counters_ptr + 0, 0)
            tl.store(event_counters_ptr + 1, 0)
            # A device-side epoch lets batch=0 forwards remain visible without
            # conflating every unmapped request with one permanent batch.
            tl.atomic_add(event_counters_ptr + 2, 1)


def _normalized_device(device: str | torch.device) -> torch.device:
    normalized = torch.device(device)
    if normalized.type == "cuda" and normalized.index is None:
        normalized = torch.device("cuda", torch.cuda.current_device())
    return normalized


def _report_state_index(source_id: int, layer_id: int) -> int:
    if not 0 <= source_id <= _MAX_SOURCE_ID:
        raise ValueError(
            f"NaN diagnostic source_id must be in [0, {_MAX_SOURCE_ID}], "
            f"got {source_id}"
        )
    if not -1 <= layer_id <= _MAX_LAYER_ID:
        raise ValueError(
            f"NaN diagnostic layer_id must be in [-1, {_MAX_LAYER_ID}], "
            f"got {layer_id}"
        )
    return source_id * _STATE_LAYERS + layer_id + 1


def _configured_event_capacity() -> int:
    capacity = int(
        os.environ.get(
            "DSV4_NAN_DIAG_EVENT_CAPACITY",
            str(_DEFAULT_EVENT_CAPACITY),
        )
    )
    if capacity <= 0:
        raise ValueError(
            f"DSV4_NAN_DIAG_EVENT_CAPACITY must be positive, got {capacity}"
        )
    return capacity


def _ensure_event_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    device = _normalized_device(device)
    device_key = str(device)
    counters = _EVENT_COUNTERS_BY_DEVICE.get(device_key)
    records = _EVENT_RECORDS_BY_DEVICE.get(device_key)
    if counters is not None and records is not None:
        return counters, records
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 NaN event state must be allocated before CUDA graph capture; "
            "call prewarm(device) first"
        )
    capacity = _configured_event_capacity()
    # counters = [events attempted, records dropped, batch-zero epoch].
    counters = torch.zeros((3,), dtype=torch.int64, device=device)
    records = torch.empty((capacity, _EVENT_FIELDS), dtype=torch.int64, device=device)
    _EVENT_COUNTERS_BY_DEVICE[device_key] = counters
    _EVENT_RECORDS_BY_DEVICE[device_key] = records
    return counters, records


def _reset_event_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    counters, records = _ensure_event_state(device)
    _reset_event_state_kernel[(1,)](counters, num_warps=1, num_stages=1)
    return counters, records


def _ensure_report_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    device = _normalized_device(device)
    device_key = str(device)
    last_reported = _LAST_REPORTED_BATCH_BY_DEVICE.get(device_key)
    report_count = _REPORT_COUNT_BY_DEVICE.get(device_key)
    if last_reported is not None and report_count is not None:
        return last_reported, report_count
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "DSV4 NaN diagnostic state must be allocated before CUDA graph capture; "
            "call prewarm(device) first"
        )
    state_size = (_MAX_SOURCE_ID + 1) * _STATE_LAYERS
    last_reported = torch.full((state_size,), -1, dtype=torch.int64, device=device)
    report_count = torch.zeros((state_size,), dtype=torch.int64, device=device)
    _LAST_REPORTED_BATCH_BY_DEVICE[device_key] = last_reported
    _REPORT_COUNT_BY_DEVICE[device_key] = report_count
    return last_reported, report_count


def set_batch_context(batch_id: torch.Tensor | None) -> None:
    """Set the graph-stable batch id and reset reliable events for this forward."""
    if not ENABLED or batch_id is None:
        return
    if not batch_id.is_cuda or batch_id.dtype != torch.int64 or batch_id.numel() < 1:
        raise ValueError(
            "DSV4 NaN diagnostic batch id must be a non-empty CUDA int64 tensor"
        )
    device_key = str(_normalized_device(batch_id.device))
    _BATCH_ID_TENSORS[device_key] = batch_id
    # The next forward must publish its own request layout before generic
    # activation checks.  This prevents a decode graph's [B, q_len] geometry
    # from being reused accidentally by a later prefill forward.
    _REQUEST_LAYOUT_BY_DEVICE.pop(device_key, None)
    # This launch is captured as the first diagnostics graph node, so every
    # replay gets a fresh event buffer without a host-side reset or allocation.
    _reset_event_state(batch_id.device)


def set_request_layout(
    device: str | torch.device,
    *,
    batch_size: int,
    q_len: int,
) -> None:
    """Set the uniform ``[B, q_len, ...]`` layout for this decode forward.

    This lets generic activation detectors turn their flattened Triton row
    back into an exact request and query position.  Packed KV detectors do not
    depend on this state because their ``indices`` tensor already has the
    explicit ``[B, q_len, topk]`` layout.
    """
    if not ENABLED:
        return
    if batch_size <= 0 or q_len <= 0:
        raise ValueError(
            "DSV4 NaN request layout requires positive batch_size and q_len, "
            f"got batch_size={batch_size} q_len={q_len}"
        )
    device_key = str(_normalized_device(device))
    _REQUEST_LAYOUT_BY_DEVICE[device_key] = (int(batch_size), int(q_len))


def attach_event_buffers(outputs):
    """Attach graph-stable diagnostic buffers to ``PyModelOutputs`` for C++."""
    if not ENABLED:
        return outputs
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None or not hidden_states.is_cuda:
        raise RuntimeError("DSV4 NaN diagnostics require CUDA model outputs")
    counters, records = _ensure_event_state(hidden_states.device)
    outputs.nan_diag_event_counters = counters
    outputs.nan_diag_events = records
    return outputs


def report_nonfinite(
    tensor: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
    include_neg_inf: bool = True,
    batch_size: int | None = None,
    q_len: int | None = None,
    map_to_request: bool = True,
    row_offset: int = 0,
    layout_rows: int | None = None,
) -> None:
    """Report non-finite values without modifying or synchronizing ``tensor``."""
    if not ENABLED or tensor.numel() == 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    if not tensor.is_cuda:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires CUDA tensors")
    if not tensor.is_floating_point():
        raise ValueError(f"NaN diagnostic requires floating tensor, got {tensor.dtype}")
    state_index = _report_state_index(int(source_id), int(layer_id))
    if tensor.dim() == 1:
        rows, cols = 1, tensor.shape[0]
        stride_row, stride_col = 0, tensor.stride(0)
    elif tensor.dim() == 2:
        rows, cols = tensor.shape
        stride_row, stride_col = tensor.stride()
    elif tensor.dim() > 2 and tensor.is_contiguous():
        cols = tensor.shape[-1]
        rows = tensor.numel() // cols
        stride_row, stride_col = cols, 1
    else:
        raise ValueError(
            "NaN diagnostic supports 1D/2D tensors and contiguous higher-rank "
            f"tensors, got shape={tuple(tensor.shape)} contiguous={tensor.is_contiguous()}"
        )

    device = _normalized_device(tensor.device)
    device_key = str(device)
    layout = _REQUEST_LAYOUT_BY_DEVICE.get(device_key) if map_to_request else None
    if map_to_request:
        if batch_size is None and layout is not None:
            batch_size = layout[0]
        if q_len is None and layout is not None:
            q_len = layout[1]
    if row_offset < 0:
        raise ValueError(
            f"NaN diagnostic row_offset must be non-negative, got {row_offset}"
        )
    rows_per_request = 0
    rows_per_query = 0
    if batch_size is not None and q_len is not None:
        mapping_rows = rows if layout_rows is None else int(layout_rows)
        if (
            batch_size <= 0
            or q_len <= 0
            or mapping_rows % batch_size != 0
            or row_offset + rows > mapping_rows
        ):
            raise ValueError(
                "NaN diagnostic tensor rows do not match request layout: "
                f"rows={rows} layout_rows={mapping_rows} row_offset={row_offset} "
                f"batch_size={batch_size} q_len={q_len} "
                f"shape={tuple(tensor.shape)}"
            )
        rows_per_request = mapping_rows // batch_size
        if rows_per_request % q_len != 0:
            raise ValueError(
                "NaN diagnostic rows per request are not divisible by q_len: "
                f"rows_per_request={rows_per_request} q_len={q_len} "
                f"shape={tuple(tensor.shape)}"
            )
        rows_per_query = rows_per_request // q_len
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        # Standalone diagnostic callers have no service trace map. Batch 0 is
        # the explicit "unmapped" value and is allocated before launch/capture.
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN diagnostic batch context must be allocated before CUDA "
                "graph capture; call prewarm(device) first"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    last_reported_batch, report_count = _ensure_report_state(device)
    event_counters, event_records = _ensure_event_state(device)
    grid = (rows, triton.cdiv(cols, _BLOCK_N))
    _report_nonfinite_tiles_kernel[grid](
        tensor,
        batch_id_tensor,
        last_reported_batch,
        report_count,
        event_counters,
        event_records,
        rows,
        cols,
        stride_row,
        stride_col,
        int(source_id),
        int(layer_id),
        state_index,
        int(include_neg_inf),
        int(event_records.shape[0]),
        int(rows_per_request),
        int(rows_per_query),
        int(row_offset),
        BLOCK_N=_BLOCK_N,
        num_warps=4,
        num_stages=1,
    )


def report_packed_fp8_kv_cache(
    cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
    kv_kind: int = KV_KIND_UNKNOWN,
    topk_length: torch.Tensor | None = None,
) -> None:
    """Report NaN encodings in the packed FP8 slots read by attention.

    ``indices`` contains global slots into ``cache`` and must have shape
    ``[B, q_len, K]``. Only the leftmost ``topk_length[B]`` entries are
    scanned when a length tensor is supplied. The cache and indices are read
    in-place without materializing or dequantizing the selected KV values.
    """
    if not ENABLED or cache.numel() == 0 or indices.numel() == 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    if kv_kind not in (KV_KIND_UNKNOWN, KV_KIND_SWA, KV_KIND_CSA, KV_KIND_HCA):
        raise ValueError(f"invalid packed DSV4 KV kind: {kv_kind}")
    if (
        not cache.is_cuda
        or cache.dtype != torch.uint8
        or cache.dim() != 3
        or cache.shape[-1] != 584
        or cache.stride(-1) != 1
    ):
        raise ValueError(
            "packed DSV4 KV diagnostic requires CUDA uint8 "
            f"[num_blocks, block_size, 584], got shape={tuple(cache.shape)} "
            f"dtype={cache.dtype} device={cache.device} stride={cache.stride()}"
        )
    if (
        not indices.is_cuda
        or indices.dtype not in (torch.int32, torch.int64)
        or indices.dim() != 3
        or not indices.is_contiguous()
        or indices.device != cache.device
    ):
        raise ValueError(
            "packed DSV4 KV diagnostic requires contiguous CUDA int32/int64 "
            f"indices [B, q_len, K] on {cache.device}, got "
            f"shape={tuple(indices.shape)} dtype={indices.dtype} "
            f"device={indices.device} contiguous={indices.is_contiguous()}"
        )
    width = int(indices.shape[-1])
    if width == 0:
        return
    rows = int(indices.numel() // width)
    q_len = int(indices.shape[-2])
    lengths_per_row = False
    if topk_length is not None:
        if (
            not topk_length.is_cuda
            or topk_length.dtype != torch.int32
            or not topk_length.is_contiguous()
            or topk_length.device != cache.device
        ):
            raise ValueError(
                "packed DSV4 KV diagnostic topk_length must be contiguous CUDA "
                f"int32 on {cache.device}, got dtype={topk_length.dtype} "
                f"device={topk_length.device} contiguous={topk_length.is_contiguous()}"
            )
        if topk_length.numel() == rows:
            lengths_per_row = True
        elif topk_length.numel() != rows // q_len:
            raise ValueError(
                "packed DSV4 KV diagnostic topk_length must have B or B*q_len "
                f"elements, got {topk_length.numel()} for indices "
                f"shape={tuple(indices.shape)}"
            )

    device = _normalized_device(cache.device)
    device_key = str(device)
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN diagnostic batch context must be allocated before CUDA "
                "graph capture; call prewarm(device) first"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    state_index = _report_state_index(int(source_id), int(layer_id))
    last_reported_batch, report_count = _ensure_report_state(device)
    event_counters, event_records = _ensure_event_state(device)
    corrupt_kind = 0
    if TEST_KV_CORRUPT is not None:
        corrupt_layer, corrupt_source, requested_kind = TEST_KV_CORRUPT
        if corrupt_layer == int(layer_id) and corrupt_source == int(source_id):
            corrupt_kind = requested_kind
    lengths_ptr = topk_length if topk_length is not None else indices
    _report_packed_fp8_kv_cache_kernel[(rows * width,)](
        cache,
        indices,
        lengths_ptr,
        batch_id_tensor,
        last_reported_batch,
        report_count,
        event_counters,
        event_records,
        rows,
        width,
        q_len,
        int(cache.shape[0]),
        int(cache.shape[1]),
        int(cache.stride(0)),
        int(source_id),
        int(layer_id),
        int(kv_kind),
        state_index,
        int(event_records.shape[0]),
        HAS_LENGTHS=topk_length is not None,
        LENGTHS_PER_ROW=lengths_per_row,
        CORRUPT_KIND=corrupt_kind,
        BLOCK_N=512,
        num_warps=4,
        num_stages=1,
    )


def _launch_packed_fp8_indexer_cache_report(
    cache: torch.Tensor,
    mapping: torch.Tensor,
    lengths: torch.Tensor,
    *,
    rows: int,
    width: int,
    q_len: int,
    map_stride: int,
    source_id: int,
    layer_id: int,
    explicit_slots: bool,
) -> None:
    device = _normalized_device(cache.device)
    device_key = str(device)
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN diagnostic batch context must be allocated before CUDA "
                "graph capture; call prewarm(device) first"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    state_index = _report_state_index(int(source_id), int(layer_id))
    last_reported_batch, report_count = _ensure_report_state(device)
    event_counters, event_records = _ensure_event_state(device)
    corrupt_kind = 0
    if TEST_KV_CORRUPT is not None:
        corrupt_layer, corrupt_source, requested_kind = TEST_KV_CORRUPT
        if corrupt_layer == int(layer_id) and corrupt_source == int(source_id):
            corrupt_kind = requested_kind
    _report_packed_fp8_indexer_cache_kernel[(rows * width,)](
        cache,
        mapping,
        lengths,
        batch_id_tensor,
        last_reported_batch,
        report_count,
        event_counters,
        event_records,
        rows,
        width,
        q_len,
        int(cache.shape[0]),
        int(cache.shape[1]),
        int(cache.stride(0)),
        map_stride,
        int(source_id),
        int(layer_id),
        state_index,
        int(event_records.shape[0]),
        EXPLICIT_SLOTS=explicit_slots,
        CORRUPT_KIND=corrupt_kind,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
    )


def _validate_indexer_cache(cache: torch.Tensor) -> None:
    if (
        not cache.is_cuda
        or cache.dtype != torch.uint8
        or cache.dim() != 3
        or cache.shape[-1] != 132
        or cache.stride(-1) != 1
    ):
        raise ValueError(
            "packed DSV4 indexer KV diagnostic requires CUDA uint8 "
            f"[num_blocks, block_size, 132], got shape={tuple(cache.shape)} "
            f"dtype={cache.dtype} device={cache.device} stride={cache.stride()}"
        )


def report_packed_fp8_indexer_slots(
    cache: torch.Tensor,
    slots: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
) -> None:
    """Scan exact indexer-K slots, normally immediately after a local write."""
    if not ENABLED or cache.numel() == 0 or slots.numel() == 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    _validate_indexer_cache(cache)
    if (
        not slots.is_cuda
        or slots.dtype not in (torch.int32, torch.int64)
        or slots.dim() != 3
        or not slots.is_contiguous()
        or slots.device != cache.device
    ):
        raise ValueError(
            "packed DSV4 indexer KV diagnostic requires contiguous CUDA "
            f"int32/int64 slots [B, q_len, K], got shape={tuple(slots.shape)} "
            f"dtype={slots.dtype} device={slots.device}"
        )
    width = int(slots.shape[-1])
    if width == 0:
        return
    rows = int(slots.numel() // width)
    q_len = int(slots.shape[-2])
    _launch_packed_fp8_indexer_cache_report(
        cache,
        slots,
        slots,
        rows=rows,
        width=width,
        q_len=q_len,
        map_stride=0,
        source_id=source_id,
        layer_id=layer_id,
        explicit_slots=True,
    )


def report_paged_fp8_indexer_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    *,
    source_id: int,
    layer_id: int,
    max_ctx_len: int,
) -> None:
    """Scan every indexer-K entry a paged score operation will consume."""
    if not ENABLED or cache.numel() == 0 or max_ctx_len <= 0:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    _validate_indexer_cache(cache)
    if (
        not block_table.is_cuda
        or block_table.dtype != torch.int32
        or block_table.dim() != 2
        or not block_table.is_contiguous()
        or block_table.device != cache.device
    ):
        raise ValueError(
            "paged DSV4 indexer diagnostic requires contiguous CUDA int32 "
            f"block_table [B, N], got shape={tuple(block_table.shape)} "
            f"dtype={block_table.dtype} device={block_table.device}"
        )
    if (
        not lengths.is_cuda
        or lengths.dtype != torch.int32
        or lengths.dim() != 2
        or not lengths.is_contiguous()
        or lengths.device != cache.device
        or lengths.shape[0] != block_table.shape[0]
    ):
        raise ValueError(
            "paged DSV4 indexer diagnostic requires contiguous CUDA int32 "
            f"lengths [B, q_len], got shape={tuple(lengths.shape)} "
            f"dtype={lengths.dtype} device={lengths.device}"
        )
    q_len = int(lengths.shape[1])
    rows = int(lengths.numel())
    width = min(
        int(max_ctx_len),
        int(block_table.shape[1]) * int(cache.shape[1]),
    )
    if q_len == 0 or rows == 0 or width == 0:
        return
    _launch_packed_fp8_indexer_cache_report(
        cache,
        block_table,
        lengths,
        rows=rows,
        width=width,
        q_len=q_len,
        map_stride=int(block_table.stride(0)),
        source_id=source_id,
        layer_id=layer_id,
        explicit_slots=False,
    )


def maybe_inject_test_nan(tensor: torch.Tensor, *, layer_id: int) -> None:
    """Inject one guarded test NaN into a mapped, non-warmup model batch."""
    if TEST_INJECT is None or TEST_INJECT[0] != layer_id:
        return
    if triton is None:
        raise RuntimeError("DSV4 NaN test injection requires Triton")
    if not tensor.is_cuda or tensor.dim() != 2:
        raise RuntimeError(
            "DSV4 NaN test injection requires a 2D CUDA activation tensor"
        )
    _, row, col = TEST_INJECT
    if row >= tensor.shape[0] or col >= tensor.shape[1]:
        raise ValueError(
            "DSV4_NAN_DIAG_TEST_INJECT is outside the activation shape: "
            f"target=(layer={layer_id},row={row},col={col}) "
            f"shape={tuple(tensor.shape)}"
        )
    device = _normalized_device(tensor.device)
    device_key = str(device)
    batch_id_tensor = _BATCH_ID_TENSORS.get(device_key)
    if batch_id_tensor is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "DSV4 NaN test injection requires batch context before CUDA "
                "graph capture"
            )
        batch_id_tensor = torch.zeros((1,), dtype=torch.int64, device=device)
        _BATCH_ID_TENSORS[device_key] = batch_id_tensor
    _inject_nan_kernel[(1,)](
        tensor,
        batch_id_tensor,
        row,
        col,
        tensor.stride(0),
        tensor.stride(1),
        num_warps=1,
        num_stages=1,
    )


def prewarm(device: str | torch.device) -> None:
    """Compile BF16/FP32 detector variants before CUDA graph capture."""
    if not ENABLED:
        return
    if triton is None:
        raise RuntimeError("DSV4_NAN_DIAG=1 requires Triton")
    device = _normalized_device(device)
    device_key = str(device)
    if device_key in _PREWARMED_DEVICES:
        return

    fifo_mb = int(
        os.environ.get(
            "DSV4_NAN_DIAG_PRINTF_FIFO_MB",
            str(_DEFAULT_PRINTF_FIFO_MB),
        )
    )
    if fifo_mb <= 0:
        raise ValueError(
            f"DSV4_NAN_DIAG_PRINTF_FIFO_MB must be positive, got {fifo_mb}"
        )
    fifo_configured = True
    try:
        # CUDA's default device-printf FIFO is small enough to lose events
        # during a widespread NaN burst. This must run before any printf kernel.
        with torch.cuda.device(device):
            triton.runtime.driver.active.utils.set_printf_fifo_size(
                fifo_mb * 1024 * 1024
            )
    except Exception as error:
        fifo_configured = False
        logging.warning(
            "[DSV4 NaN diag] failed to set CUDA printf FIFO to %d MiB; "
            "continuing with the runtime FIFO size: %s",
            fifo_mb,
            error,
        )

    event_capacity = _configured_event_capacity()
    logging.warning(
        "[DSV4_NAN_DIAG_READY] enabled on %s with reliable_event_capacity=%d "
        "and auxiliary_printf_fifo=%s; events are rate-limited per "
        "batch/source/layer; reliable host logs use prefix "
        "[DSV4_NAN_RELIABLE], while auxiliary device logs use prefixes "
        "[DSV4_NAN] and [DSV4_NAN_DIAG_CANARY]. "
        "source_id: 1=moe_input, 2=router_scores, 3=router_bias, "
        "4=cp_attention_lse, 5=final_hidden, 6=attention_query, "
        "7=kv_write_input, 8=swa_kv_cache_read, "
        "9=compressed_kv_cache_read, 10=attention_output, "
        "11=swa_kv_cache_post_write, 12=compressed_kv_cache_post_write, "
        "13=indexer_kv_cache_post_write, 14=indexer_kv_cache_read, "
        "15=indexer_score, 16-19=indexer_inputs",
        device,
        event_capacity,
        f"{fifo_mb} MiB" if fifo_configured else "runtime-default",
    )
    if TEST_INJECT is not None:
        logging.error(
            "[DSV4 NaN diag] TEST-ONLY NaN injection is active: "
            "layer=%d row=%d col=%d; model output is intentionally mutated",
            *TEST_INJECT,
        )
    if TEST_KV_CORRUPT is not None:
        logging.error(
            "[DSV4 NaN diag] TEST-ONLY KV-cache corruption is active: "
            "layer=%d source=%d kind=%d; cache and output are intentionally mutated",
            *TEST_KV_CORRUPT,
        )
    _BATCH_ID_TENSORS[device_key] = torch.zeros((1,), dtype=torch.int64, device=device)
    _ensure_report_state(device)
    _reset_event_state(device)
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.zeros((1, _BLOCK_N), dtype=dtype, device=device)
        report_nonfinite(
            probe,
            source_id=SOURCE_ROUTER_SCORES,
            layer_id=-1,
        )
    packed_probe = torch.zeros((1, 1, 584), dtype=torch.uint8, device=device)
    index_probe = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    length_probe = torch.ones((1,), dtype=torch.int32, device=device)
    report_packed_fp8_kv_cache(
        packed_probe,
        index_probe,
        source_id=SOURCE_SWA_KV_CACHE_READ,
        layer_id=-1,
        kv_kind=KV_KIND_SWA,
    )
    report_packed_fp8_kv_cache(
        packed_probe,
        index_probe,
        source_id=SOURCE_COMPRESSED_KV_CACHE_READ,
        layer_id=-1,
        kv_kind=KV_KIND_CSA,
        topk_length=length_probe,
    )
    if TEST_INJECT is not None:
        probe = torch.zeros((1, 1), dtype=torch.bfloat16, device=device)
        _inject_nan_kernel[(1,)](
            probe,
            _BATCH_ID_TENSORS[device_key],
            0,
            0,
            probe.stride(0),
            probe.stride(1),
            num_warps=1,
            num_stages=1,
        )
    _device_printf_canary_kernel[(1,)](num_warps=1, num_stages=1)
    torch.cuda.synchronize(device)
    _PREWARMED_DEVICES.add(device_key)
