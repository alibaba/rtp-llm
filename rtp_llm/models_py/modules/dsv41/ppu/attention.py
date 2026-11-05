"""V4.1 sparse prefill attention for the PPU backend.

``sparse_attention`` implements the ``V41BackendOps.sparse_attention``
contract:

1. build the bounded per-request workspace exactly like rtp's
   ``fp8/_v41_prefill_kv_workspace.py`` — per request, the dequantized global
   BF16 rows ``[Tg, 1, 512]`` followed by the SWA tail (last ``window=128``
   rows) — concatenated across requests into ``[T_total, 1, 512]``;
2. assemble the combined indices ``[rows, 1, align(512 + 128, 128) = 640]``
   int32 with ``-1`` sentinels in the unused tail plus ``topk_length[rows]``
   (rtp ``_swa_ops_triton.combine_topk_swa_indices`` layout);
3. chunk the query rows by ``DSV4_FLASH_MLA_SPARSE_Q_CHUNK`` and call
   ``flash_mla_sparse_fwd(q, kv, indices, sm_scale, 512, topk_length=...)``
   through a lazy ``import flash_mla``, with the head-padding pattern from
   rtp's ``modules/dsv41/flash_mla.py`` inlined (query heads are padded to
   the kernel's 64/128-head tiles and sliced back; rtp's own module imports
   nothing at import time but is not modified here).

``sm_scale = 512 ** -0.5``. The decode path (``flash_mla_with_kvcache``
dual 528B + 288B cache) is provided as ``decode_attention`` with documented
preconditions; the V41CacheView -> paged-metadata mapping is L3's wiring.
"""

from __future__ import annotations

import torch
from rtp_llm.models_py.modules.dsv41.chunk_env import FLASH_MLA_SPARSE_Q_CHUNK
from rtp_llm.models_py.modules.dsv41.ppu import _quant, kv_codec

# Overridable for tests; mirrors chunk_env.FLASH_MLA_SPARSE_Q_CHUNK at import.
Q_CHUNK = FLASH_MLA_SPARSE_Q_CHUNK

SWA_WINDOW = 128
SPARSE_TOPK_ALIGNMENT = 128
COMBINED_WIDTH = (
    (512 + SWA_WINDOW + SPARSE_TOPK_ALIGNMENT - 1) // SPARSE_TOPK_ALIGNMENT
) * SPARSE_TOPK_ALIGNMENT  # 640
SM_SCALE = 512**-0.5
HEAD_DIM = 512

GLOBAL_KV_TAGS = kv_codec.GLOBAL_KV_TAGS
SWA_KV_TAG = kv_codec.SWA_KV_TAG


def _pad_heads(q, attn_sink):
    """Inline of rtp flash_mla._pad_heads (64-head kernel tiles)."""
    heads = q.shape[-2]
    kernel_heads = ((heads + 63) // 64) * 64
    if kernel_heads == heads:
        return q, attn_sink, heads
    q = torch.nn.functional.pad(q, (0, 0, 0, kernel_heads - heads))
    if attn_sink is not None:
        attn_sink = torch.nn.functional.pad(
            attn_sink, (0, kernel_heads - heads), value=float("inf")
        )
    return q, attn_sink, heads


def flash_mla_sparse_fwd_ppu(
    q, kv, indices, sm_scale, d_v=HEAD_DIM, attn_sink=None, topk_length=None
):
    """Lazy-importing ``flash_mla_sparse_fwd`` with rtp's head padding."""
    from flash_mla import flash_mla_sparse_fwd as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, maximum, lse = native(
        q, kv, indices, sm_scale, d_v=d_v, attn_sink=sink, topk_length=topk_length
    )
    return output[:, :heads], maximum[:, :heads], lse[:, :heads]


def _resolve_global_pool(cache, layer_id: int):
    """(pool_tag, ratio, descriptor) for this layer's global KV pool.

    Producers are layers 2/8/14 (kv_2) and 20 (kv_1); the mapping is read
    from the cache descriptors when they carry a source/producer layer id,
    otherwise the single present global tag wins, else ratio 2 is assumed
    (matching the majority producer set).
    """
    descriptors = cache.descriptors or {}
    candidates = []
    for tag in GLOBAL_KV_TAGS:
        desc = descriptors.get(tag)
        if desc is None:
            continue
        producer = None
        for attr in ("source_layer_id", "producer_layer_id", "layer_id"):
            producer = getattr(desc, attr, None)
            if producer is not None:
                break
        candidates.append((tag, desc, producer))
    if not candidates:
        return "global_kv_2", 2, None
    for tag, desc, producer in candidates:
        if producer is not None and int(producer) == int(layer_id):
            ratio = int(getattr(desc, "compression_ratio", 2) or 2)
            return tag, ratio, desc
    if len(candidates) == 1:
        tag, desc, _ = candidates[0]
        return tag, int(getattr(desc, "compression_ratio", 2) or 2), desc
    return "global_kv_2", 2, None


def _entries_per_block(pool: torch.Tensor, descriptor, entry_bytes: int) -> int:
    entries = pool.shape[1] // entry_bytes if pool.ndim == 2 else pool.shape[1]
    if entries <= 0:
        entries = int(
            getattr(descriptor, "tokens_per_block", 0)
            or getattr(descriptor, "block_entries", 0)
            or 128
        )
    return entries


def _request_groups(request_ids: torch.Tensor):
    """Ordered (request_id, row indices) groups from rank-local rows."""
    seen: dict = {}
    order = []
    ids = request_ids.tolist()
    for row, rid in enumerate(ids):
        if rid not in seen:
            seen[rid] = []
            order.append(rid)
        seen[rid].append(row)
    return [(rid, seen[rid]) for rid in order]


def _slots_for_entries(
    block_table: torch.Tensor, entries: int, count: int, request: int
):
    """Absolute pool slots for compressed entries ``[0, count)`` of one request."""
    slots = torch.zeros(count, dtype=torch.int64, device=block_table.device)
    blocks_needed = (count + entries - 1) // entries
    table = block_table[request, :blocks_needed].to(torch.int64)
    for entry in range(count):
        block = entry // entries
        offset = entry % entries
        slots[entry] = table[block] * entries + offset
    return slots


def _slots_for_positions(
    block_table: torch.Tensor, entries: int, positions, request: int
):
    """Absolute pool slots for RAW positions (the SWA ring mapping).

    ``slot(p) = table[p // entries] * entries + p % entries`` — the caller's
    block table owns the ring; the linear position arithmetic stays here.
    """
    table = block_table[request]
    pos = positions.to(torch.int64)
    blocks = table[(pos // entries).to(torch.long)].to(torch.int64)
    return blocks * entries + (pos % entries)


def _dequantize_global_slots(pool2d, slots):
    return _quant.dequantize_global_kv_slots(pool2d, slots)


def _dequantize_swa_slots(pool2d, slots, entries):
    """Gather + decode 528B SWA rows into fresh bf16 rows."""
    rows = slots.shape[0]
    out = torch.empty((rows, _quant.KV_DIM), dtype=torch.bfloat16, device=pool2d.device)
    desc = kv_codec.PoolDescriptor(
        entry_bytes=_quant.SWA_KV_ENTRY_BYTES,
        block_entries=entries,
        window=SWA_WINDOW,
        kind="fp8_e4m3_ue8m0",
    )
    packed = kv_codec.gather_rows(pool2d, SWA_KV_TAG, desc, slots)
    return kv_codec.decode_kv(packed, pool_tag=SWA_KV_TAG, descriptor=desc, out=out)


def build_sparse_workspace(
    selected_blocks: torch.Tensor,
    *,
    cache,
    layer_id: int,
    context,
    window: int = SWA_WINDOW,
    global_pool_tag: str | None = None,
    ratio: int | None = None,
    swa_starts: dict | None = None,
):
    """Assemble the bounded sparse-attention workspace.

    Returns ``(kv [T_total, 1, 512] bf16, indices [rows, 1, 640] int32,
    topk_length [rows] int32)``. Per request the workspace is
    ``[global rows (Tg) | SWA tail (<= window)]``; per token the combined
    index row is ``[topk global indices | SWA window indices | -1 padding]``
    with ``topk_length`` counting the written entries.

    ``selected_blocks`` is ``[rows, topk]`` int32 of compressed indices per
    token (rtp's indexer top-k output; ``-1`` entries are skipped).
    Positions and request ids come from ``context``; the request's total
    length is ``max(positions of its rows) + 1``.

    SWA workspace span: every SWA position any query row of the request can
    see, ``[max(0, min_pos - (window - 1)), max_pos]`` — NOT just the last
    ``window`` tokens of the request. The two coincide when the query rows
    are the request's tail (decode / chunk-tail); under CP8 the rank-local
    query rows sit at scattered global positions and the old tail-only span
    both missed early rows and indexed out of range.
    """
    positions = context.positions.to(torch.int64).reshape(-1)
    request_ids = context.request_ids.reshape(-1)
    rows = positions.shape[0]
    tag, resolved_ratio, gdesc = _resolve_global_pool(cache, layer_id)
    if global_pool_tag is not None:
        tag = global_pool_tag
    if ratio is not None:
        resolved_ratio = int(ratio)
    ratio = max(int(resolved_ratio), 1)

    global_pool = cache.pools[tag]
    swa_pool = cache.pools[SWA_KV_TAG]
    g_entries = _entries_per_block(global_pool, gdesc, _quant.FP4_GLOBAL_ENTRY_BYTES)
    s_entries = _entries_per_block(
        swa_pool, cache.descriptors.get(SWA_KV_TAG), _quant.SWA_KV_ENTRY_BYTES
    )
    global_table = cache.block_tables[tag]
    swa_table = cache.block_tables[SWA_KV_TAG]

    topk = selected_blocks.shape[1] if selected_blocks is not None else 0
    combined = torch.full(
        (rows, 1, COMBINED_WIDTH), -1, dtype=torch.int32, device=positions.device
    )
    lengths = torch.zeros(rows, dtype=torch.int32, device=positions.device)

    global_tensors = []
    swa_tensors = []
    offset = 0
    for request, row_indices in _request_groups(request_ids):
        seq_len = int(positions[row_indices].max()) + 1
        total_global = seq_len // ratio
        # Global rows: dequantize the request's compressed entries.
        if total_global > 0:
            gslots = _slots_for_entries(global_table, g_entries, total_global, request)
            grows = _dequantize_global_slots(
                global_pool.reshape(global_pool.shape[0], -1), gslots
            )
        else:
            grows = torch.zeros(
                (0, _quant.KV_DIM), dtype=torch.bfloat16, device=positions.device
            )
        # SWA span: every position any query row of this request can see
        # (wave2-fix: the old tail-only span broke rank-local CP query rows).
        min_pos = int(positions[row_indices].min())
        max_pos = seq_len - 1
        replay_start = int((swa_starts or {}).get(request, 0))
        swa_start = max(0, replay_start, min_pos - (window - 1))
        tail = max_pos + 1 - swa_start
        if tail > 0:
            tail_positions = torch.arange(swa_start, seq_len, device=positions.device)
            sslots = _slots_for_positions(swa_table, s_entries, tail_positions, request)
            srows = _dequantize_swa_slots(
                swa_pool.reshape(swa_pool.shape[0], -1), sslots, s_entries
            )
        else:
            srows = torch.zeros(
                (0, _quant.KV_DIM), dtype=torch.bfloat16, device=positions.device
            )
        global_tensors.append(grows)
        swa_tensors.append(srows)
        gbase = offset
        offset += grows.shape[0]
        sbase = offset
        offset += srows.shape[0]

        for row in row_indices:
            pos = int(positions[row])
            # topk part: valid selected compressed indices (skip -1 padding).
            topk_len = min((pos + 1) // ratio, topk)
            written = 0
            if topk_len > 0:
                entries = selected_blocks[row, :topk_len].to(torch.int64)
                keep = entries[entries >= 0]
                if keep.numel():
                    combined[row, 0, written : written + keep.numel()] = (
                        gbase + keep
                    ).to(torch.int32)
                    written += int(keep.numel())
            # SWA part: positions [pos - swa_len + 1, pos].
            swa_len = max(0, min(pos + 1, window, pos - replay_start + 1))
            if swa_len > 0:
                first = pos - swa_len + 1
                offsets = torch.arange(first, pos + 1, device=positions.device)
                swa_rows = (sbase + (offsets - swa_start)).to(torch.int32)
                combined[row, 0, written : written + swa_len] = swa_rows
                written += swa_len
            lengths[row] = written

    tensors = [t for g, s in zip(global_tensors, swa_tensors) for t in (g, s)]
    kv = (
        torch.cat(tensors).reshape(-1, 1, _quant.KV_DIM)
        if tensors
        else torch.zeros(
            (0, 1, _quant.KV_DIM), dtype=torch.bfloat16, device=positions.device
        )
    )
    return kv, combined, lengths


def sparse_attention(
    q: torch.Tensor,
    selected_blocks: torch.Tensor,
    *,
    cache,
    layer_id: int,
    context,
    attn_sink: torch.Tensor | None = None,
    sm_scale: float = SM_SCALE,
    window: int = SWA_WINDOW,
    ratio: int | None = None,
    swa_starts: dict | None = None,
):
    """Sparse prefill attention over the V41CacheView pools.

    ``q`` is the rank-local BF16 ``[rows, n_heads, 512]``; ``selected_blocks``
    is the indexer top-k output ``[rows, topk]`` int32. Returns the kernel's
    attention output for the real heads (``[rows, n_heads, ...]``, sliced
    back from the padded head tile). The output projection and all-reduce
    stay with the model.
    """
    kv, indices, topk_length = build_sparse_workspace(
        selected_blocks,
        cache=cache,
        layer_id=layer_id,
        context=context,
        window=window,
        ratio=ratio,
        swa_starts=swa_starts,
    )
    outputs = []
    rows = q.shape[0]
    chunk = int(Q_CHUNK) if Q_CHUNK and Q_CHUNK > 0 else rows or 1
    # attn_sink is per-HEAD ([h_q]), not per query row: slicing it by the
    # query-row chunk truncates it below the head count on decode/CP8
    # chunk shapes ("attn_sink must have shape (h_q)") and silently drops
    # heads on multi-chunk prefill. Pass it through unsliced.
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        output, _, _ = flash_mla_sparse_fwd_ppu(
            q[start:end].contiguous(),
            kv,
            indices[start:end],
            sm_scale,
            HEAD_DIM,
            attn_sink=attn_sink,
            topk_length=topk_length[start:end],
        )
        outputs.append(output)
    if not outputs:
        return q.new_empty((0,) + tuple(q.shape[1:]))
    return torch.cat(outputs, dim=0)


def decode_attention(
    q: torch.Tensor,
    *,
    cache=None,
    layer_id: int = 0,
    context=None,
    block_table: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    sm_scale: float = SM_SCALE,
    attn_sink: torch.Tensor | None = None,
    **kwargs,
):
    """Decode-path call structure for the dual 528B SWA + 288B global cache.

    Preconditions (L3 wiring supplies them from V41CacheView + device
    metadata): ``block_table`` is the paged block table of the FP8 SWA cache
    and ``cache_seqlens`` the per-request visible lengths, both in the exact
    form ``flash_mla_with_kvcache`` expects, plus any paged cache tensors in
    ``kwargs``. Without them the dual-cache metadata cannot be derived from
    the cache view alone, so this raises ``NotImplementedError``.
    """
    if block_table is None or cache_seqlens is None:
        raise NotImplementedError(
            "decode dual-cache metadata is wired by L3: pass the paged "
            "block_table and cache_seqlens (plus paged cache tensors) that "
            "flash_mla_with_kvcache expects for the 528B SWA + 288B global "
            "dual cache; V41CacheView alone does not carry them."
        )
    from flash_mla import flash_mla_with_kvcache as native

    q, sink, heads = _pad_heads(q, attn_sink)
    output, lse = native(
        q=q,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        sm_scale=sm_scale,
        attn_sink=sink,
        **kwargs,
    )
    return output[..., :heads, :], lse[:, :heads, :]
