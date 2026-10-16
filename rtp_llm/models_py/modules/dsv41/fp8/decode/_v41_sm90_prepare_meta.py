"""SM90-only, in-place V4.1 shared decode metadata preparation.

This is a leaf candidate, not a model hook. V4.1 deliberately gives the shared
planner no compressed ratios; its actual ratio-1/2 producers remain unchanged.
The arithmetic follows _fused_prepare_meta_triton and decode_attn_metadata.
Two launches replace table copy/zero and the small metadata/translation ops.
No source table is retained as a replacement for graph-owned output storage.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _snapshot_tables(
    sources,
    destinations,
    source_rows,
    source_cols,
    source_stride0,
    source_stride1,
    bs,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pool = tl.program_id(1)
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    for i in tl.static_range(len(ROWS)):
        if pool == i:
            row = off // COLS[i]
            col = off % COLS[i]
            nr = tl.minimum(source_rows[i], ROWS[i])
            nc = tl.minimum(source_cols[i], COLS[i])
            copy = (row < nr) & (col < nc)
            value = tl.load(
                sources[i] + row * source_stride0[i] + col * source_stride1[i],
                mask=copy & (row < ROWS[i]),
                other=0,
            )
            # Exactly mirror zero(dst[bs:]); copy(src prefix); zero(right pad).
            # In particular, rows [nr, bs) retain their old values when nr<bs.
            write = (row < ROWS[i]) & ((row < nr) | (row >= bs))
            tl.store(destinations[i] + off, value, mask=write)


@triton.jit
def _prepare_metadata(
    positions,
    start_out,
    position_out,
    position_long_out,
    seq_start_out,
    ring_slots,
    window_out,
    abs_out,
    topk_out,
    swa_length,
    tables,
    write_slots,
    state_slots,
    read_slots,
    Q: tl.constexpr,
    WINDOW: tl.constexpr,
    TOPK: tl.constexpr,
    MAX_START: tl.constexpr,
    PLUS_ONE: tl.constexpr,
    FULL_WIDTH: tl.constexpr,
    ENTRIES: tl.constexpr,
    TOKENS: tl.constexpr,
    COLS: tl.constexpr,
    WRITE_RATIO: tl.constexpr,
    HAS_STATE: tl.constexpr,
    HAS_READ: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    r = tl.program_id(0)
    q = tl.program_id(1)
    t = r * Q + q
    raw = tl.load(positions + r).to(tl.int32)
    sp = tl.minimum(tl.maximum(raw - PLUS_ONE, 0), MAX_START)
    pos = sp + q
    ring = pos % WINDOW
    if q == 0:
        tl.store(start_out + r, sp)
        tl.store(seq_start_out + r, sp.to(tl.int64))
        length = WINDOW if FULL_WIDTH else tl.minimum(sp + Q, WINDOW)
        tl.store(swa_length + r, length)
    tl.store(position_out + t, pos)
    tl.store(position_long_out + t, pos.to(tl.int64))
    tl.store(ring_slots + t, r * WINDOW + ring)
    w = tl.arange(0, BLOCK_W)
    wi = tl.where(pos >= WINDOW - 1, (ring + 1 + w) % WINDOW, tl.where(w <= pos, w, -1))
    absolute = tl.maximum(pos - WINDOW + 1, 0) + w
    absolute = tl.where(absolute <= pos, absolute, -1)
    tl.store(window_out + t * WINDOW + w, wi, mask=w < WINDOW)
    tl.store(abs_out + t * WINDOW + w, absolute, mask=w < WINDOW)
    k = tl.arange(0, BLOCK_K)
    tl.store(topk_out + t * TOPK + k, -1, mask=k < TOPK)

    for i in tl.static_range(len(ENTRIES)):
        if WRITE_RATIO[i] > 0:
            # Compatibility-only legacy fields retain their old 4/128 values.
            # Actual V4.1 global/index slots are owned by AttentionV41FP8.
            ratio = WRITE_RATIO[i]
            logical = pos if ratio == 1 else (pos + 1) // ratio - 1
            skip = False if ratio == 1 else ((pos + 1) % ratio != 0) | (logical < 0)
            safe = tl.where(skip, 0, logical)
            column = tl.minimum(
                tl.maximum(safe // (TOKENS[i] // ratio), 0), COLS[i] - 1
            )
            bid = tl.load(tables[i] + r * COLS[i] + column).to(tl.int64)
            slot = tl.where(skip | (bid <= 0), -1, bid * ENTRIES[i] + safe % ENTRIES[i])
            tl.store(write_slots[i] + t, slot)
        if HAS_STATE[i]:
            column = (pos.to(tl.int64) // TOKENS[i]) % COLS[i]
            bid = tl.load(tables[i] + r * COLS[i] + column).to(tl.int64)
            slot = tl.where(bid > 0, bid * ENTRIES[i] + pos % ENTRIES[i], -1)
            tl.store(state_slots[i] + t, slot)
        if HAS_READ[i]:
            column = absolute // TOKENS[i]
            valid = (absolute >= 0) & (column >= 0) & (column < COLS[i]) & (w < WINDOW)
            bid = tl.load(tables[i] + r * COLS[i] + column, mask=valid, other=0)
            slot = tl.where(
                valid & (bid > 0), bid * ENTRIES[i] + absolute % ENTRIES[i], -1
            )
            tl.store(read_slots[i] + t * WINDOW + w, slot, mask=w < WINDOW)


class Sm90DecodeMetadataUpdater:
    """Return False without writes for unsupported inputs; caller uses fallback.

    Construct eagerly after decoder_swa_metadata allocation. Device length
    mirrors are read on the current stream. CPU lengths use a synchronous copy
    into owned staging: this intentionally keeps pinned-source lifetime safe.
    This helper does not remove the framework's host-mirror synchronization.
    """

    def __init__(self, meta, entries_per_block, tokens_per_block):
        self.meta = meta
        self.entries = dict(entries_per_block)
        self.tokens = dict(tokens_per_block)
        self.device = meta.start_pos.device
        self.bucket = int(meta.start_pos.numel())
        self.q_len = int(meta.q_len_per_req)
        self.tags = tuple(meta.pool_block_tables)
        self.enabled = False
        self.raw_positions = None
        if self.device.type != "cuda" or torch.cuda.get_device_capability(
            self.device
        ) != (9, 0):
            return
        if any(
            getattr(meta, name)
            for name in (
                "slot_mapping_compressed",
                "compressed_lens",
                "compressed_lens_per_token",
                "compressed_topk_length_by_ratio",
                "topk_total_by_ratio",
            )
        ):
            return
        if (
            self.q_len <= 0
            or meta.window_size <= 0
            or not self.tags
            or meta.topk_buffer_compressed.shape[-1] <= 0
        ):
            return
        child = getattr(meta, "decoder_swa_metadata", None)
        if "decoder_swa_kv" in self.tags and child is None:
            return
        modes = {
            "swa_kv": 1,
            "decoder_swa_kv": 1,
            "global_kv_2": 4,
            "indexer_kv": 4,
            "global_kv_1": 128,
        }
        self.tables = tuple(meta.pool_block_tables[t] for t in self.tags)
        self.rows = tuple(int(t.shape[0]) for t in self.tables)
        self.cols = tuple(int(t.shape[1]) for t in self.tables)
        self.ratios = tuple(modes.get(t, 0) for t in self.tags)
        # Dummy pointers below are only legal for compile-time-disabled paths.
        # Never turn a missing live slot destination into a write to positions.
        if any(
            self.ratios[i] and tag not in meta.pool_write_slot_mappings
            for i, tag in enumerate(self.tags)
        ):
            return
        self.has_state = tuple(
            t in meta.compressor_state_slot_mappings for t in self.tags
        )
        self.has_read = tuple(t in ("swa_kv", "decoder_swa_kv") for t in self.tags)
        self.write_slots = tuple(
            meta.pool_write_slot_mappings.get(t, meta.position_ids_long)
            for t in self.tags
        )
        self.state_slots = tuple(
            meta.compressor_state_slot_mappings.get(t, meta.position_ids_long)
            for t in self.tags
        )
        self.read_slots = tuple(
            (
                meta.swa_global_slots
                if t == "swa_kv"
                else (
                    child.swa_global_slots
                    if t == "decoder_swa_kv"
                    else meta.topk_window_idxs
                )
            )
            for t in self.tags
        )
        if any(t not in self.entries or t not in self.tokens for t in self.tags):
            return
        self.e = tuple(int(self.entries[t]) for t in self.tags)
        self.tpb = tuple(int(self.tokens[t]) for t in self.tags)
        for i, table in enumerate(self.tables):
            if (
                table.dtype != torch.int32
                or table.device != self.device
                or not table.is_contiguous()
                or self.rows[i] < self.bucket
                or self.cols[i] <= 0
                or self.e[i] <= 0
                or self.tpb[i] <= 0
                or (self.ratios[i] and self.tpb[i] // self.ratios[i] <= 0)
            ):
                return
        count = self.bucket * self.q_len
        required = (
            (meta.start_pos, torch.int32, self.bucket),
            (meta.position_ids, torch.int32, count),
            (meta.position_ids_long, torch.int64, count),
            (meta.decode_seq_start_per_req, torch.int64, self.bucket),
            (meta.slot_mapping_swa, torch.int32, count),
            (meta.topk_window_idxs, torch.int32, count * meta.window_size),
            (meta.swa_abs_idx, torch.int32, count * meta.window_size),
            (
                meta.topk_buffer_compressed,
                torch.int32,
                count * meta.topk_buffer_compressed.shape[-1],
            ),
            (meta.swa_topk_length, torch.int32, self.bucket),
        )
        required += tuple(
            (out, torch.int64, count)
            for i, out in enumerate(self.write_slots)
            if self.ratios[i]
        )
        required += tuple(
            (out, torch.int64, count)
            for i, out in enumerate(self.state_slots)
            if self.has_state[i]
        )
        required += tuple(
            (out, torch.int32, count * meta.window_size)
            for i, out in enumerate(self.read_slots)
            if self.has_read[i]
        )
        if any(
            not isinstance(t, torch.Tensor)
            or t.dtype != dtype
            or t.device != self.device
            or not t.is_contiguous()
            or t.numel() < size
            for t, dtype, size in required
        ):
            return
        self.raw_positions = torch.empty(
            self.bucket, dtype=torch.int32, device=self.device
        )
        self.enabled = True

    def update(
        self, attention_inputs, *, paged_block_tables, capture_full_width_lengths=False
    ):
        if not self.enabled or not paged_block_tables:
            return False
        plus_one = False
        if isinstance(attention_inputs, torch.Tensor):
            source = attention_inputs
            bs = int(source.numel())
        else:
            prefix = bool(getattr(attention_inputs, "is_target_verify", False)) or (
                bool(getattr(attention_inputs, "is_prefill", False)) and self.q_len > 1
            )
            host = getattr(
                attention_inputs,
                "prefix_lengths" if prefix else "sequence_lengths",
                None,
            )
            if not isinstance(host, torch.Tensor) or host.ndim != 1:
                return False
            bs = int(host.numel())
            source = getattr(
                attention_inputs,
                "prefix_lengths_device" if prefix else "sequence_lengths_plus_1_device",
                None,
            )
            if (
                isinstance(source, torch.Tensor)
                and source.device == self.device
                and source.ndim == 1
                and source.numel() >= bs
            ):
                plus_one = not prefix
            else:
                source = host
        if (
            not isinstance(source, torch.Tensor)
            or source.ndim != 1
            or source.dtype not in (torch.int32, torch.int64)
            or not source.is_contiguous()
            or source.device.type not in ("cpu", "cuda")
            or (source.is_cuda and source.device != self.device)
            or bs <= 0
            or bs > self.bucket
        ):
            return False
        # Missing pools retain old snapshots in the generic path. Keep that
        # uncommon case on fallback rather than partially updating this helper.
        if any(t not in paged_block_tables for t in self.tags):
            return False
        sources = tuple(paged_block_tables[t] for t in self.tags)
        if any(
            not isinstance(t, torch.Tensor)
            or t.device != self.device
            or t.dtype != torch.int32
            or t.ndim != 2
            or t.stride(0) <= 0
            or t.stride(1) <= 0
            for t in sources
        ):
            return False
        # Source tables may change address every step, but may not alias the
        # graph-owned snapshots (including an offset view of that storage).
        destination_storage = {t.untyped_storage().data_ptr() for t in self.tables}
        if any(t.untyped_storage().data_ptr() in destination_storage for t in sources):
            return False
        if source.device.type == "cpu":
            if torch.cuda.is_current_stream_capturing():
                return False
            # Blocking copy is deliberate: caller may immediately reuse pinned
            # host storage. Do not make this async without an ownership contract.
            self.raw_positions[:bs].copy_(source, non_blocking=False)
            source = self.raw_positions
        _snapshot_tables[
            (
                max(triton.cdiv(r * c, 256) for r, c in zip(self.rows, self.cols)),
                len(self.tags),
            )
        ](
            sources,
            self.tables,
            tuple(t.shape[0] for t in sources),
            tuple(t.shape[1] for t in sources),
            tuple(t.stride(0) for t in sources),
            tuple(t.stride(1) for t in sources),
            bs,
            ROWS=self.rows,
            COLS=self.cols,
            BLOCK=256,
        )
        m = self.meta
        _prepare_metadata[(bs, self.q_len)](
            source,
            m.start_pos,
            m.position_ids,
            m.position_ids_long,
            m.decode_seq_start_per_req,
            m.slot_mapping_swa,
            m.topk_window_idxs,
            m.swa_abs_idx,
            m.topk_buffer_compressed,
            m.swa_topk_length,
            self.tables,
            self.write_slots,
            self.state_slots,
            self.read_slots,
            Q=self.q_len,
            WINDOW=m.window_size,
            TOPK=m.topk_buffer_compressed.shape[-1],
            MAX_START=max(0, m.max_seq_len - self.q_len),
            PLUS_ONE=plus_one,
            FULL_WIDTH=bool(capture_full_width_lengths),
            ENTRIES=self.e,
            TOKENS=self.tpb,
            COLS=self.cols,
            WRITE_RATIO=self.ratios,
            HAS_STATE=self.has_state,
            HAS_READ=self.has_read,
            BLOCK_W=triton.next_power_of_2(m.window_size),
            BLOCK_K=triton.next_power_of_2(m.topk_buffer_compressed.shape[-1]),
        )
        m.batch_size = bs
        m.total_tokens = bs * self.q_len
        m.paged_pool_tokens_per_block = dict(self.tokens)
        child = getattr(m, "decoder_swa_metadata", None)
        if child is not None:
            child.batch_size = bs
            child.total_tokens = bs * self.q_len
        return True
