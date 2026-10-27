"""Request-major packed V4.1 verification metadata with static token capacity.

The model tensor may be viewed as [B, average_width, H] for existing GEMMs.
That view never defines request boundaries: device lengths/cumulative lengths
own all positions, cache writes and attention request ids.
"""

from __future__ import annotations

from dataclasses import replace

import torch


def packed_row_mapping(lengths, prefixes, row_count, cu_seqlens=None):
    """Return (request, absolute position, cumulative lengths), without D2H."""
    if lengths.ndim != 1 or prefixes.shape != lengths.shape or not lengths.numel():
        raise ValueError("Packed verification requires one length/prefix per request")
    lengths = lengths.to(torch.int64)
    prefixes = prefixes.to(device=lengths.device, dtype=torch.int64)
    if cu_seqlens is None:
        cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    else:
        if cu_seqlens.shape != (lengths.numel() + 1,):
            raise ValueError("Packed cumulative lengths must have B+1 entries")
        cu_seqlens = cu_seqlens.to(device=lengths.device, dtype=torch.int64)
    rows = torch.arange(row_count, device=lengths.device, dtype=torch.int64)
    request = torch.searchsorted(cu_seqlens[1:].contiguous(), rows, right=True)
    # Graph padding consists of explicit dummy requests, so every row belongs
    # to a request. Clamp only to keep malformed metadata from an OOB gather;
    # the caller validates lengths and the fixed total before dispatch.
    request = request.clamp_max(lengths.numel() - 1)
    positions = prefixes[request] + rows - cu_seqlens[request]
    return request, positions, cu_seqlens


def packed_pool_slots(table, request, positions, entries, tokens_per_block):
    """Translate arbitrary query/request maps; nonallocated pages stay -1."""
    columns = positions.clamp_min(0) // tokens_per_block
    valid = (positions >= 0) & (columns < table.shape[1])
    blocks = table[request.long(), columns.clamp_max(table.shape[1] - 1)].long()
    slots = blocks * entries + positions.remainder(entries)
    return torch.where(valid & (blocks > 0), slots, torch.full_like(slots, -1))


def _validate_packed_tables(specs, tables, batch):
    if tables is None:
        raise ValueError("Packed verification requires tagged block tables")
    unknown = set(tables) - set(specs)
    if unknown:
        raise ValueError(
            f"Packed verification received unknown block table tags: {sorted(unknown)}"
        )
    for tag in specs:
        table = tables.get(tag)
        shape = None if table is None else tuple(table.shape)
        spec = specs[tag]
        capacity = (
            tuple(spec.shape) if isinstance(spec, torch.Tensor) else (batch, spec[2])
        )
        if (
            table is None
            or table.ndim != 2
            or table.shape[0] != batch
            or table.shape[1] <= 0
        ):
            raise ValueError(
                f"Packed verification requires a complete block table for {tag}: "
                f"src_shape={shape}, dst_shape={capacity}"
            )


def update_packed_decode_metadata(
    meta,
    attention_inputs,
    paged_block_tables,
    paged_pool_entries_per_block,
    paged_pool_tokens_per_block,
    *,
    valid_request_count=None,
    capture_full_width_lengths=False,
):
    """Refresh graph-stable buffers; all data-dependent work stays on device."""
    from rtp_llm.models_py.modules.dsv41.attn_type import DECODER_SWA_KV, SWA_KV

    device = meta.start_pos.device
    lengths = getattr(attention_inputs, "input_lengths_device", None)
    prefixes = getattr(attention_inputs, "prefix_lengths_device", None)
    if (
        meta.is_cuda_graph
        and device.type == "cuda"
        and (
            lengths is None
            or not lengths.is_cuda
            or prefixes is None
            or not prefixes.is_cuda
        )
    ):
        raise ValueError(
            "Packed graph preparation requires device lengths and prefixes"
        )
    if lengths is None or not lengths.numel():
        lengths = attention_inputs.input_lengths
    if prefixes is None or not prefixes.numel():
        prefixes = attention_inputs.prefix_lengths
    lengths = lengths.to(device=device, dtype=torch.int64)
    prefixes = prefixes.to(device=device, dtype=torch.int64)
    batch = int(lengths.numel())
    rows = batch * meta.q_len_per_req
    if batch != meta.batch_size or rows != meta.total_tokens:
        raise ValueError(
            "Packed metadata capacity must match the captured token/request shapes"
        )
    cu = getattr(attention_inputs, "cu_seqlens_device", None)
    if cu is not None and not cu.numel():
        cu = None
    request, positions, cu = packed_row_mapping(lengths, prefixes, rows, cu)
    live = batch if valid_request_count is None else valid_request_count
    if isinstance(live, torch.Tensor):
        if not (
            live.numel() == 1
            and live.dtype == torch.int32
            and live.device == device
            and live.is_contiguous()
        ):
            raise ValueError(
                "Packed live request count must be one contiguous device int32"
            )
        # The runner validates the count before filling this persistent device
        # input. Captured preparation must consume it without a scalar readback.
        live = live.reshape(())
    elif not 0 <= live <= batch:
        raise ValueError("Packed live request count exceeds capture capacity")
    _validate_packed_tables(meta.pool_block_tables, paged_block_tables, batch)
    for tag, dst in meta.pool_block_tables.items():
        src = paged_block_tables[tag]
        if src.shape[1] > dst.shape[1]:
            raise ValueError(
                f"Packed block table exceeds captured capacity for {tag}: "
                f"src_shape={tuple(src.shape)}, dst_shape={tuple(dst.shape)}"
            )
    meta.is_packed = True
    meta.start_pos.copy_(prefixes)
    meta.input_lengths.copy_(lengths)
    meta.position_ids.copy_(positions)
    meta.position_ids_long.copy_(positions)
    meta.req_id_per_token.copy_(request)
    meta.req_id_per_token_long.copy_(request)
    meta.decode_seq_start_per_req.copy_(prefixes)
    meta.decode_cu_seq_per_req.copy_(cu)
    meta.valid_requests.copy_(torch.arange(batch, device=device) < live)

    win = meta.window_size
    columns = torch.arange(win, device=device, dtype=torch.int64)
    candidates = (positions[:, None] - win + 1).clamp_min(0) + columns
    absolute = torch.where(candidates <= positions[:, None], candidates, -1)
    meta.swa_abs_idx.copy_(absolute.reshape_as(meta.swa_abs_idx))
    # FlashMLA bounds the same index width for S execution rows. Those rows
    # can straddle requests, so use their maximum, not a request's last row.
    if capture_full_width_lengths:
        meta.swa_topk_length.fill_(win)
    else:
        visible = (positions + 1).clamp(0, win).reshape(batch, meta.q_len_per_req)
        meta.swa_topk_length.copy_(visible.max(dim=1).values)
    meta.topk_buffer_compressed.fill_(-1)
    for tag, dst in meta.pool_block_tables.items():
        src = paged_block_tables[tag]
        dst.zero_()
        dst[:, : src.shape[1]].copy_(src)

    for tag in (SWA_KV, DECODER_SWA_KV):
        if tag not in meta.pool_block_tables:
            continue
        table = meta.pool_block_tables[tag]
        entries = paged_pool_entries_per_block[tag]
        span = paged_pool_tokens_per_block[tag]
        writes = packed_pool_slots(table, request, positions, entries, span)
        meta.pool_write_slot_mappings[tag].copy_(writes)
        read_request = request[:, None].expand_as(absolute)
        reads = packed_pool_slots(table, read_request, absolute, entries, span).int()
        if tag == SWA_KV:
            meta.swa_global_slots.copy_(reads)
        else:
            child = meta.decoder_swa_metadata
            if child is None:
                child = replace(
                    meta,
                    pool_block_tables={**meta.pool_block_tables, SWA_KV: table},
                    pool_write_slot_mappings={
                        **meta.pool_write_slot_mappings,
                        SWA_KV: meta.pool_write_slot_mappings[tag],
                    },
                    paged_pool_tokens_per_block={
                        **meta.paged_pool_tokens_per_block,
                        SWA_KV: span,
                    },
                    swa_global_slots=torch.empty_like(meta.swa_global_slots),
                )
                meta.decoder_swa_metadata = child
            child.is_packed = True
            child.swa_global_slots.copy_(reads)


def allocate_packed_decode_metadata(*, paged_block_tables=None, **kwargs):
    from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
        allocate_decode_metadata_fp8,
    )

    # V4.1 global ratio 1/2 metadata is owned by AttentionV41FP8. The shared
    # legacy metadata allocator only owns the sliding-window mappings here.
    if any(r != 0 for r in kwargs["compress_ratios"]):
        raise ValueError("Packed verification is restricted to V4.1 attention")
    if paged_block_tables is not None:
        specs = kwargs["paged_pool_specs"]
        _validate_packed_tables(specs, paged_block_tables, kwargs["max_batch_size"])
        # Native graph tables include the physical block reserve expanded to
        # kernel granularity. Its width can exceed ceil(max_seq/kernel)+1.
        # Allocate that full schema before capture; replay must never truncate
        # input columns or resize these graph-owned buffers.
        kwargs["paged_pool_specs"] = {
            tag: (entries, span, max(width, paged_block_tables[tag].shape[1]))
            for tag, (entries, span, width) in specs.items()
        }
    meta = allocate_decode_metadata_fp8(**kwargs)
    meta.is_packed = True
    meta.input_lengths = torch.empty_like(meta.start_pos)
    # The allocator can create a decoder SWA child before this field exists.
    if meta.decoder_swa_metadata is not None:
        meta.decoder_swa_metadata.is_packed = True
        meta.decoder_swa_metadata.input_lengths = meta.input_lengths
    return meta
