"""Tensor-only PPU Attention device methods over shared projection weights.

Global/CSA production and selection are local device operations. Block mHC,
Engram, MoE and the complete layer loop remain in their shared modules.
"""

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import (
    _attention_torch,
    bounded_workspace,
    indexer_topk,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _compress_pairs, _rms_norm as _norm, _rope_tail
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import mask_candidate_logits, select_candidate_blocks


def _compact_workspace_indices(indices):
    """Stable live prefix and device length over the unchanged fixed KV bank."""
    row = indices[:, 0]
    live = row >= 0
    ranks = live.to(torch.int64).cumsum(dim=-1)
    lengths = live.sum(dim=-1, dtype=torch.int32)
    columns = torch.arange(row.shape[-1], dtype=torch.int64, device=row.device)[None]
    # Both live and invalid entries have unique destinations. Preserve each
    # subset's order without boolean indexing or device-dependent allocation.
    destinations = torch.where(
        live, ranks - 1, lengths[:, None].to(torch.int64) + columns - ranks
    )
    compacted = torch.empty_like(row).scatter_(1, destinations, row)
    return compacted[:, None], lengths


def _shared_decode_attention(
    q, selected, *, cache, layer_id, source_layer, ratio, positions,
    valid_mask, window, sm_scale, attn_sink,
):
    """Use the eager live-prefix layout with fixed-capacity Graph storage."""
    kv, indices, _ = bounded_workspace(
        selected, cache=cache, layer_id=layer_id, source_layer=source_layer,
        ratio=ratio, positions=positions, valid_mask=valid_mask, window=window,
    )
    # Keep the existing own-row dummy index for padded requests. Compaction
    # counts that index; the original output mask still zeros the request.
    indices, lengths = _compact_workspace_indices(indices)
    if q.is_cuda:
        output, _, _ = flash_mla_sparse_fwd_ppu(
            q.contiguous(), kv, indices, sm_scale, attn_sink=attn_sink,
            topk_length=lengths,
        )
    else:
        output = _attention_torch(q, kv, indices, sm_scale, attn_sink)
    return torch.where(valid_mask[:, None, None], output, torch.zeros_like(output))


def produce_global(attn, x, state):
    g, cache = attn.geometry, state.cache
    positions, valid = state.metadata.positions, state.metadata.valid_mask
    layer_id, ratio = attn.layer_id, attn.compress_ratio
    freqs = state.freqs[layer_id]
    values = attn.global_wkv(x)
    if ratio == 1:
        latent = _norm(values, attn.global_norm, g.norm_eps).bfloat16()
        entry_positions, commit = positions, valid
    elif ratio == 2:
        scores = attn.global_wgate(x)
        commit = valid & ((positions & 1) == 1)
        csa_slots = cache.slots(layer_id, "csa_state", torch.where(valid, positions, -1))
        csa_rows = torch.cat((values, scores), -1).float()
        if getattr(state, "verify", False):
            # The journal keeps every raw generation. An odd row in this
            # bucket may need the immediately preceding even row's payload.
            cache.write_rows(layer_id, "csa_state", csa_slots, csa_rows)
        previous = cache.read_rows(layer_id, "csa_state", cache.slots(
            layer_id, "csa_state", torch.where(commit, positions - 1, -1)
        ))
        latent = _compress_pairs(
            torch.stack((previous[:, :g.head_dim].to(values.dtype), values), 1),
            torch.stack((previous[:, g.head_dim:].to(scores.dtype), scores), 1),
            attn.global_norm, g.norm_eps,
        )
        entry_positions = positions - 1
        if not getattr(state, "verify", False):
            cache.write_rows(layer_id, "csa_state", csa_slots, csa_rows)
    else:
        raise ValueError("shared Graph producer requires compression ratio1/2")
    rope = freqs[entry_positions.clamp(min=0, max=freqs.shape[0] - 1)]
    keys = _rope_tail(latent, rope)
    tag = "global_kv_2" if ratio == 2 else "global_kv_1"
    raw = torch.where(commit, entry_positions, -1)
    cache.write_rows(layer_id, tag, cache.slots(layer_id, tag, raw),
                     attn.ops.encode_kv(keys, pool_tag=tag))
    index = _norm(F.linear(latent, attn.index_wk.weight), attn.index_k_norm, g.norm_eps)
    index = _rope_tail(index, rope)
    cache.write_rows(layer_id, "indexer_kv", cache.slots(layer_id, "indexer_kv", raw),
                     attn.ops.encode_kv(index, pool_tag="indexer_kv"))


def select_indices(attn, x, qr, state, freqs):
    g = attn.geometry
    source = attn.kv_source_layer_id
    ratio = g.ratio(source)
    positions, valid = state.metadata.positions, state.metadata.valid_mask
    query = _rope_tail(attn.index_wq(qr).reshape(-1, g.index_n_heads, g.index_head_dim), freqs)
    weights = F.linear(x, attn.index_weights).float() * (g.index_head_dim * g.index_n_heads) ** -0.5
    scores = indexer_topk(
        query, weights, cache=state.cache, source_layer=source, ratio=ratio,
        positions=positions, valid_mask=valid, topk=g.index_topk,
        max_seq_len=state.cache.max_seq_len, scores_only=True,
    )
    visible = (positions + 1) // ratio
    config = attn.v41_config
    candidate_source = int(config.get("candidate_source_layer_id", -1))
    block = int(config.get("candidate_block_size", 0))
    candidate_k = int(config.get("candidate_topk_blocks", 0))
    if attn.layer_id == candidate_source and block > 0 and candidate_k > 0:
        state.shared["candidates"] = select_candidate_blocks(scores, visible, block, candidate_k)
        state.shared["candidate_active"] = (visible > block * candidate_k) & (visible > g.index_topk)
    elif attn.layer_id > candidate_source and "candidates" in state.shared:
        masked = mask_candidate_logits(scores.clone(), state.shared["candidates"], block)
        scores = torch.where(state.shared["candidate_active"][:, None], masked, scores)
    width = min(g.index_topk, scores.shape[-1])
    values, indices = scores.topk(width, dim=-1)
    selected = torch.where(torch.isfinite(values), indices.int(), -1)
    if width < g.index_topk:
        selected = F.pad(selected, (0, g.index_topk - width), value=-1)
    # The eager shared selector preserves ordinal order until top-k is needed.
    # Keep that same order via a device condition, including across replay.
    ordinal = torch.arange(g.index_topk, dtype=torch.int32, device=x.device)[None]
    ordered = torch.where(valid[:, None] & (ordinal < visible[:, None]), ordinal, -1)
    return torch.where((visible <= g.index_topk)[:, None], ordered, selected)


def forward_graph_attention(attn, x, state):
    g, cache = attn.geometry, state.cache
    shape = x.shape
    x = x.reshape(-1, g.hidden_size)
    positions, valid = state.metadata.positions, state.metadata.valid_mask
    freqs = state.freqs[attn.layer_id][positions]
    qr = attn.query_latent(x)
    q = _rope_tail(attn.wq_b(qr).reshape(-1, g.n_heads, g.head_dim), freqs)
    kv = _rope_tail(attn.kv_norm(attn.wkv(x)), freqs)
    cache.write_rows(attn.layer_id, "swa_kv", cache.slots(
        attn.layer_id, "swa_kv", torch.where(valid, positions, -1)
    ), attn.ops.encode_kv(kv, pool_tag="swa_kv"))
    if attn.is_kv_source:
        produce_global(attn, x, state)
    selected = None
    if attn.compress_ratio:
        if attn.is_index_source:
            selected = select_indices(attn, x, qr, state, freqs)
            state.shared[attn.layer_id] = selected
        else:
            selected = state.shared[attn.index_source_layer_id]
    output = _shared_decode_attention(
        q, selected, cache=cache, layer_id=attn.layer_id,
        source_layer=attn.kv_source_layer_id, ratio=attn.compress_ratio,
        positions=positions, valid_mask=valid, window=g.window,
        sm_scale=g.sm_scale(), attn_sink=attn.attn_sink,
    )
    output = _rope_tail(output, freqs, inverse=True).reshape(-1, g.o_groups, g.n_heads * g.head_dim // g.o_groups)
    projected = torch.stack([
        F.linear(output[:, i].contiguous(), attn.wo_a.weight[i * g.o_lora_rank:(i + 1) * g.o_lora_rank].contiguous())
        for i in range(g.o_groups)
    ], 1)
    return attn.wo_b(projected.flatten(1)).reshape(shape)
