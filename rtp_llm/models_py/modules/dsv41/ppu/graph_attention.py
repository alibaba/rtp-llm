"""Fixed-shape V4.1 decode selection and bounded attention workspaces.

The framework cache owns every byte. This module reads Indexer K for scoring
and gathers only the selected global entries plus the sliding window for
attention. No tensor value controls Python flow, allocation sizes or loops.
"""

from __future__ import annotations

import os

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.indexer import _bf16_rounding_reduce

_VERIFY_FP4_DEQUANT_ENV = "RTP_V41_PPU_VERIFY_FUSED_INDEXER_DEQUANT"


def score_indexer_entries(
    query, head_weights, *, cache, source_layer, ratio, positions, valid_mask,
    entries, quantized_query=None, score_reducer=None,
):
    """Score explicit logical key entries through the cache's current view."""
    batch, columns = query.shape[0], entries.shape[-1]
    fused_read = None
    if score_reducer is not None:
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal
        from rtp_llm.models_py.modules.dsv41.ppu.verify_ops import try_read_verify_indexer_entries

        if type(cache) is VerifyGraphJournal:
            native = cache.native_cache
            tag = native._table_tag(source_layer, "indexer_kv")
            key = source_layer, "indexer_kv"
            fused_read = try_read_verify_indexer_entries(
                entries, positions, valid_mask,
                row_tables=cache.row_tables[tag], pool=native.pool(*key),
                payloads=cache.payloads[key], generations=cache.generations[key],
                requests=cache.requests, starts=cache.starts, prefixes=cache.prefixes,
                valid=cache.valid, span=native.spans[tag],
                physical_entries=native.entries[tag], ratio=ratio,
            )
    if fused_read is None:
        raw_positions = entries.expand(batch, columns) * ratio
        visible = (positions + 1) // ratio
        valid = valid_mask[:, None] & (entries < visible[:, None])
        slots = cache.slots(source_layer, "indexer_kv", torch.where(valid, raw_positions, -1))
        packed = cache.read_rows(source_layer, "indexer_kv", slots)
        effective_valid = valid & (slots >= 0)
    else:
        packed, effective_valid = fused_read
    decoded = None
    if (
        score_reducer is not None
        and quantized_query is not None
        and batch == 32 and 1 <= columns <= 4 and ratio in (1, 2)
        and os.environ.get(_VERIFY_FP4_DEQUANT_ENV, "1") == "1"
    ):
        from rtp_llm.models_py.modules.dsv41.ppu.dspark_graph import VerifyGraphJournal

        if type(cache) is VerifyGraphJournal:
            decoded = _quant.try_dequantize_verify_indexer(*quantized_query, packed)
    if decoded is None:
        keys = _quant.dequantize_mx_fp4_rows(
            packed[..., :64].contiguous().view(torch.int8),
            _quant._bytes_to_sf(packed[..., 64:].reshape(-1, 4).contiguous()).reshape(batch, columns),
        )
        payload, scales = (
            _quant.quantize_indexer_q(query) if quantized_query is None else quantized_query
        )
        query = _quant.dequantize_mx_fp4_rows(payload, scales)
    else:
        query, keys = decoded
    dots = torch.bmm(query.float(), keys.float().transpose(1, 2))
    weights = head_weights.to(torch.bfloat16)
    if score_reducer is not None:
        return score_reducer(dots, weights, effective_valid)
    scores = _bf16_rounding_reduce(dots, weights)
    return scores.masked_fill(~effective_valid, float("-inf"))


def indexer_topk(
    query: torch.Tensor,
    head_weights: torch.Tensor,
    *,
    cache,
    source_layer: int,
    ratio: int,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
    topk: int,
    max_seq_len: int,
    scores_only: bool = False,
) -> torch.Tensor:
    """Score each request's own fixed-capacity planar Indexer K bank.

    Preserve the existing PPU indexer's FP4 query boundary and four-lane BF16
    rounding chain. Invalid/padded keys never become selected entries. The
    fixed capacity is a Graph geometry bound, not this step's visible length.
    """
    capacity = max(1, int(max_seq_len) // int(ratio))
    entries = torch.arange(capacity, dtype=torch.int64, device=query.device)
    scores = score_indexer_entries(
        query, head_weights, cache=cache, source_layer=source_layer, ratio=ratio,
        positions=positions, valid_mask=valid_mask, entries=entries,
    )
    if scores_only:
        return scores
    width = min(int(topk), capacity)
    values, indices = torch.topk(scores, width, dim=-1)
    selected = torch.where(
        torch.isfinite(values), indices.to(torch.int32), -1
    )
    if width < topk:
        selected = torch.nn.functional.pad(selected, (0, topk - width), value=-1)
    return selected


def bounded_workspace(
    selected: torch.Tensor | None,
    *,
    cache,
    layer_id: int,
    source_layer: int | None,
    ratio: int,
    positions: torch.Tensor,
    valid_mask: torch.Tensor,
    window: int,
):
    """Gather [selected global top-k | SWA window] with a fixed row stride.

    The sparse kernel accepts -1 entries anywhere in its index row. Effective
    length is the fixed row width; holes remain masked by their -1 index.
    This avoids device-count-dependent compaction and never reads an invalid
    cache slot. An empty row receives one zero-valued dummy key and its output
    is masked after attention, so the native kernel never sees a zero-row bank.
    """
    batch = positions.shape[0]
    device = positions.device
    global_width = 0 if selected is None else selected.shape[1]
    width = max(640, global_width + int(window))
    chunks = []
    masks = []
    if selected is not None:
        if source_layer is None or ratio not in (1, 2):
            raise ValueError("global decode attention requires its ratio-1/2 source")
        visible = ((positions + 1) // ratio).clamp_min(0)
        valid = (
            valid_mask[:, None]
            & (selected >= 0)
            & (selected < visible[:, None])
        )
        raw = selected.to(torch.int64) * ratio
        slots = cache.slots(
            source_layer,
            "global_kv_2" if ratio == 2 else "global_kv_1",
            torch.where(valid, raw, -1),
        )
        packed = cache.read_rows(
            source_layer, "global_kv_2" if ratio == 2 else "global_kv_1", slots
        )
        decoded = _quant.dequantize_global_kv_rows(packed.reshape(-1, 288))
        chunks.append(decoded.reshape(batch, global_width, _quant.KV_DIM))
        masks.append(valid & (slots >= 0))

    offsets = torch.arange(window, dtype=torch.int64, device=device)
    raw = positions[:, None] - (window - 1) + offsets[None, :]
    valid = valid_mask[:, None] & (raw >= 0)
    slots = cache.slots(layer_id, "swa_kv", torch.where(valid, raw, -1))
    packed = cache.read_rows(layer_id, "swa_kv", slots)
    decoded = _quant.dequantize_swa_kv_rows(packed.reshape(-1, 528))
    chunks.append(decoded.reshape(batch, window, _quant.KV_DIM))
    masks.append(valid & (slots >= 0))
    values = torch.cat(chunks, dim=1)
    live = torch.cat(masks, dim=1)
    padding = width - values.shape[1]
    if padding:
        values = torch.nn.functional.pad(values, (0, 0, 0, padding))
        live = torch.nn.functional.pad(live, (0, padding), value=False)
    values = torch.where(live[..., None], values, torch.zeros_like(values))

    columns = torch.arange(width, dtype=torch.int32, device=device)[None, :]
    bases = torch.arange(batch, dtype=torch.int32, device=device)[:, None] * width
    indices = torch.where(live, bases + columns, -1)
    # Native sparse attention may have an undefined all-empty reduction. A
    # padded request can read its own zero row; it can never read another row.
    first = torch.where(valid_mask, indices[:, 0], bases[:, 0])
    indices = torch.cat((first[:, None], indices[:, 1:]), dim=1)
    lengths = torch.full((batch,), width, dtype=torch.int32, device=device)
    return values.reshape(batch * width, 1, _quant.KV_DIM), indices[:, None], lengths


def _attention_torch(q, kv, indices, scale, attn_sink):
    """Independent torch implementation for CPU contract tests only."""
    batch, _, width = indices.shape
    safe = indices[:, 0].clamp_min(0).long()
    keys = kv[:, 0][safe]
    logits = torch.einsum("bhd,bkd->bhk", q.float(), keys.float()) * scale
    logits = logits.masked_fill(indices[:, 0, None] < 0, float("-inf"))
    if attn_sink is not None:
        sink = attn_sink.float()[None, :, None].expand(batch, -1, 1)
        normalizer = torch.logsumexp(
            torch.cat((logits, sink), dim=-1), dim=-1, keepdim=True
        )
        weights = torch.exp(logits - normalizer)
    else:
        weights = torch.softmax(logits, dim=-1)
    return torch.einsum("bhk,bkd->bhd", weights, keys.float()).to(q.dtype)


def decode_attention(
    q,
    selected,
    *,
    cache,
    layer_id,
    source_layer,
    ratio,
    positions,
    valid_mask,
    window,
    sm_scale,
    attn_sink,
):
    kv, indices, lengths = bounded_workspace(
        selected,
        cache=cache,
        layer_id=layer_id,
        source_layer=source_layer,
        ratio=ratio,
        positions=positions,
        valid_mask=valid_mask,
        window=window,
    )
    if q.is_cuda:
        output, _, _ = flash_mla_sparse_fwd_ppu(
            q.contiguous(), kv, indices, sm_scale, attn_sink=attn_sink,
            topk_length=lengths,
        )
    else:
        output = _attention_torch(q, kv, indices, sm_scale, attn_sink)
    return torch.where(valid_mask[:, None, None], output, torch.zeros_like(output))
