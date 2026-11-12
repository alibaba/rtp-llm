"""Tensor-only PPU Attention device methods over shared projection weights.

Global/CSA production and selection are local device operations. Block mHC,
Engram, MoE and the complete layer loop remain in their shared modules.
"""

import os

import torch
import torch.nn.functional as F

from rtp_llm.config.v41_runtime import env_bool
from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.attention import flash_mla_sparse_fwd_ppu
from rtp_llm.models_py.modules.dsv41.ppu.graph_attention import (
    _attention_torch,
    bounded_workspace,
    indexer_topk,
)
from rtp_llm.models_py.modules.dsv41.ppu.graph_packed_attention import GraphPackedAttention
from rtp_llm.models_py.modules.dsv41.ppu.graph_swa_write import (
    try_compress_csa,
    try_prepare_verify_csa_pairs,
    try_rope_,
    try_rotate_q_and_write_swa,
    try_write_csa_state,
    try_write_packed_kv,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill import _compress_pairs, _rms_norm as _norm, _rope_tail
from rtp_llm.models_py.modules.dsv41.fp8.attention_v41 import mask_candidate_logits, select_candidate_blocks


GRAPH_BATCHED_WO_A_ENV = "RTP_V41_PPU_GRAPH_BATCHED_WO_A"


def _try_batched_wo_a(
    output: torch.Tensor, weight: torch.Tensor, *,
    verify_q4: bool = False, proposal_q3: bool = False,
) -> torch.Tensor | None:
    """Project qualified BF16 Graph buckets directly into the wo_b layout."""
    rows = (1, 8)
    if verify_q4:
        rows = (32,)
    elif proposal_q3:
        rows = (24,)
    if (
        output.ndim != 3
        or output.shape[0] not in rows
        or output.shape[1:] != (8, 4096)
        or weight.shape != (8192, 4096)
        or not output.is_cuda
        or output.device != weight.device
        or output.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not output.is_contiguous()
        or not weight.is_contiguous()
        or output.data_ptr() % 16
        or weight.data_ptr() % 16
        or (torch.is_grad_enabled() and (output.requires_grad or weight.requires_grad))
        or not is_ppu()
        or not env_bool(os.environ, GRAPH_BATCHED_WO_A_ENV, default=True)
    ):
        return None

    # Like SGLang's token-major BF16 wo_a, make both transposes and the
    # batched destination views. The native GEMM rounds once into BF16.
    grouped_weight = weight.view(8, 1024, 4096)
    projected = output.new_empty((output.shape[0], 8, 1024))
    torch.bmm(
        output.transpose(0, 1),
        grouped_weight.transpose(1, 2),
        out=projected.transpose(0, 1),
    )
    return projected


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


def _rotate_graph_value(
    value, freqs, *, inplace=False, inverse=False, valid_mask=None, preserve_signed_zero=False
):
    if inplace:
        rotated = try_rope_(
            value, freqs, inverse=inverse, valid_mask=valid_mask,
            preserve_signed_zero=preserve_signed_zero,
        )
        if rotated is not None:
            return rotated
    if valid_mask is not None:
        value = torch.where(valid_mask[:, None, None], value, torch.zeros_like(value))
        # An unsupported provider layout can become contiguous after masking.
        if inplace:
            rotated = try_rope_(
                value, freqs, inverse=inverse, preserve_signed_zero=preserve_signed_zero
            )
            if rotated is not None:
                return rotated
    return _rope_tail(value, freqs, inverse=inverse)


def _normalize_graph_value(value, weight, eps, *, fused=False):
    if fused:
        from rtp_llm.models_py.modules.dsv41.ppu.layers import try_graph_rms_norm

        normalized = try_graph_rms_norm(value, weight, eps)
        if normalized is not None:
            return normalized
    return _norm(value, weight, eps)


def _store_graph_compressed_key(attn, cache, tag, slots, value, rope, *, fused_graph, verify_rope):
    layer_id = attn.layer_id
    if verify_rope and try_write_packed_kv(
        value, cache, layer_id, tag, slots, journal=True, freqs=rope
    ):
        return
    keys = _rotate_graph_value(
        value, rope, inplace=fused_graph or verify_rope, preserve_signed_zero=verify_rope
    )
    if not fused_graph or not try_write_packed_kv(keys, cache, layer_id, tag, slots):
        cache.write_rows(layer_id, tag, slots, attn.ops.encode_kv(keys, pool_tag=tag))


def produce_global(attn, x, state, *, fused_graph=False, verify_rope=False):
    from rtp_llm.models_py.modules.dsv41.ppu.layers import full_graph_rmsnorm_enabled

    g, cache = attn.geometry, state.cache
    positions, valid = state.metadata.positions, state.metadata.valid_mask
    layer_id, ratio = attn.layer_id, attn.compress_ratio
    freqs = state.freqs[layer_id]
    full_norm = full_graph_rmsnorm_enabled(state)
    values = attn.global_wkv(x)
    compressed = None
    verify_positions = None
    if ratio == 1:
        latent = _normalize_graph_value(
            values, attn.global_norm, g.norm_eps, fused=fused_graph or full_norm
        ).bfloat16()
        entry_positions, commit = positions, valid
    elif ratio == 2:
        scores = attn.global_wgate(x)
        verify = getattr(state, "verify", False)
        if fused_graph and not verify and attn.global_norm is not None:
            compressed = try_compress_csa(
                values, scores, attn.global_norm, g.norm_eps, freqs, positions, valid,
                cache, layer_id, state.shared,
            )
        if compressed is not None:
            latent, csa_slots, main_slots, index_slots, rope = compressed
        else:
            if verify and verify_rope:
                from rtp_llm.models_py.modules.dsv41.ppu.verify_ops import try_prepare_verify_positions

                key = ("verify_global_positions", id(positions), id(valid), ratio)
                if key not in state.shared:
                    state.shared[key] = try_prepare_verify_positions(positions, valid, ratio=ratio)
                verify_positions = state.shared[key]
            if verify_positions is None:
                commit = valid & ((positions & 1) == 1)
                entry_positions = positions - 1
                current_raw = torch.where(valid, positions, -1)
                previous_raw = torch.where(commit, entry_positions, -1)
            else:
                entry_positions, commit, current_raw, previous_raw = verify_positions
            csa_slots = cache.slots(layer_id, "csa_state", current_raw)
            if verify and (
                not verify_rope or not try_write_csa_state(
                    values, scores, cache, layer_id, csa_slots, journal=True
                )
            ):
                # The journal keeps every raw generation. An odd row in this
                # bucket may need the immediately preceding even row's payload.
                cache.write_rows(layer_id, "csa_state", csa_slots, torch.cat((values, scores), -1).float())
            previous_slots = cache.slots(
                layer_id, "csa_state", previous_raw
            )
            pairs = (
                try_prepare_verify_csa_pairs(values, scores, cache, layer_id, previous_slots)
                if verify_rope else None
            )
            if pairs is None:
                previous = cache.read_rows(layer_id, "csa_state", previous_slots)
                pairs = (
                    torch.stack((previous[:, :g.head_dim].to(values.dtype), values), 1),
                    torch.stack((previous[:, g.head_dim:].to(scores.dtype), scores), 1),
                )
            if full_norm:
                pair_values, pair_scores = pairs
                weights = torch.softmax(pair_scores.float(), dim=-2)
                # Keep the BF16 pooled boundary before normalization.
                pooled = (pair_values.float() * weights).sum(-2).to(torch.bfloat16)
                latent = _normalize_graph_value(
                    pooled, attn.global_norm, g.norm_eps, fused=True
                )
            else:
                latent = _compress_pairs(*pairs, attn.global_norm, g.norm_eps)
        # Reads for every request must finish before any current-row write
        # can overwrite a previous slot when the CSA ring wraps.
        if not verify and (
            not fused_graph or not try_write_csa_state(values, scores, cache, layer_id, csa_slots)
        ):
            cache.write_rows(layer_id, "csa_state", csa_slots, torch.cat((values, scores), -1).float())
    else:
        raise ValueError("shared Graph producer requires compression ratio1/2")
    if verify_rope:
        key = ("verify_global_rope", id(freqs), ratio)
        if key not in state.shared:
            state.shared[key] = freqs[entry_positions.clamp(min=0, max=freqs.shape[0] - 1)]
        rope = state.shared[key]
    elif compressed is None:
        rope = freqs[entry_positions.clamp(min=0, max=freqs.shape[0] - 1)]
    inplace_rope = fused_graph or verify_rope
    if inplace_rope:
        # The Indexer consumes the unrotated latent on this same stream.
        index = _normalize_graph_value(
            F.linear(latent, attn.index_wk.weight), attn.index_k_norm, g.norm_eps,
            fused=fused_graph or full_norm,
        )
    tag = "global_kv_2" if ratio == 2 else "global_kv_1"
    if compressed is None:
        raw = (
            verify_positions[3] if verify_positions is not None
            else torch.where(commit, entry_positions, -1)
        )
        main_slots = cache.slots(layer_id, tag, raw)
    _store_graph_compressed_key(
        attn, cache, tag, main_slots, latent, rope, fused_graph=fused_graph, verify_rope=verify_rope
    )
    if not inplace_rope:
        index = _normalize_graph_value(
            F.linear(latent, attn.index_wk.weight),
            attn.index_k_norm,
            g.norm_eps,
            fused=full_norm,
        )
    if compressed is None:
        index_slots = cache.slots(layer_id, "indexer_kv", raw)
    _store_graph_compressed_key(
        attn, cache, "indexer_kv", index_slots, index, rope,
        fused_graph=fused_graph, verify_rope=verify_rope,
    )


def select_indices(
    attn, x, qr, state, freqs, *, fused_graph=False, verify_rope=False,
    qr_quantized=None,
):
    g = attn.geometry
    source = attn.kv_source_layer_id
    ratio = g.ratio(source)
    positions, valid = state.metadata.positions, state.metadata.valid_mask
    query = _rotate_graph_value(
        attn.project_indexer_query(qr, quantized_input=qr_quantized).reshape(
            -1, g.index_n_heads, g.index_head_dim
        ),
        freqs,
        inplace=fused_graph or verify_rope,
        preserve_signed_zero=verify_rope,
    )
    projected_weights = F.linear(x, attn.index_weights)
    weight_scale = (g.index_head_dim * g.index_n_heads) ** -0.5
    use_visible_indexer = getattr(state, "visible_indexer", False)
    config = attn.v41_config
    candidate_source = int(config.get("candidate_source_layer_id", -1))
    block = int(config.get("candidate_block_size", 0))
    candidate_k = int(config.get("candidate_topk_blocks", 0))
    publish = attn.layer_id == candidate_source and block > 0 and candidate_k > 0
    consume = attn.layer_id > candidate_source and "candidates" in state.shared
    if use_visible_indexer and fused_graph:
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import select_visible_indices

        end_key = ("indexer_visible", 1)
        if end_key not in state.shared:
            state.shared[end_key] = positions + 1
        visible_key = ("indexer_visible", ratio)
        if visible_key not in state.shared:
            state.shared[visible_key] = state.shared[end_key] // ratio
        visible = state.shared[visible_key]
        # The visible scorer consumes BF16. Its 32 x 128 head scale is 1/64,
        # so the scalar is exact in both FP32 and BF16.
        weights = projected_weights * weight_scale
        candidate_inactive_limit = (
            state.shared["candidate_inactive_raw_limit"] // ratio
            if consume and "candidate_inactive_raw_limit" in state.shared
            and getattr(state, "grouped_moe", None) is None
            else None
        )
        selected, published = select_visible_indices(
            query, weights, cache=state.cache, source_layer=source, ratio=ratio,
            positions=positions, valid_mask=valid, topk=g.index_topk,
            candidate_block_size=block, candidate_topk_blocks=candidate_k if publish else 0,
            candidates=state.shared["candidates"] if consume else None,
            candidate_active=state.shared["candidate_active"] if consume else None,
            visible=visible, candidate_flags_cache=state.shared if consume else None,
            candidate_inactive_limit=candidate_inactive_limit,
        )
        if publish:
            state.shared["candidates"] = published
            state.shared["candidate_active"] = (visible > block * candidate_k) & (visible > g.index_topk)
            state.shared["candidate_inactive_raw_limit"] = (
                (max(block * candidate_k, g.index_topk) + 1) * ratio - 1
            )
        state.attention_calls["visible_indexer"] += 1
        return selected
    weights = projected_weights.float() * weight_scale
    if use_visible_indexer and verify_rope:
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import select_verify_indices

        result = select_verify_indices(
            query, weights, cache=state.cache, source_layer=source, ratio=ratio,
            positions=positions, valid_mask=valid, topk=g.index_topk,
            candidate_block_size=block, candidate_topk_blocks=candidate_k if publish else 0,
            candidates=state.shared["candidates"] if consume else None,
            candidate_active=state.shared["candidate_active"] if consume else None,
            candidate_inactive_limit=(
                state.shared["candidate_inactive_raw_limit"] // ratio
                if consume and "candidate_inactive_raw_limit" in state.shared
                and getattr(state, "grouped_moe", None) is None else None
            ),
            shared=state.shared,
        )
        if result is not None:
            selected, published, visible = result
            if publish:
                state.shared["candidates"] = published
                state.shared["candidate_active"] = (
                    (visible > block * candidate_k) & (visible > g.index_topk)
                )
                state.shared["candidate_inactive_raw_limit"] = (
                    (max(block * candidate_k, g.index_topk) + 1) * ratio - 1
                )
            state.attention_calls["visible_indexer"] += 1
            return selected
    visible = (positions + 1) // ratio
    if use_visible_indexer:
        from rtp_llm.models_py.modules.dsv41.ppu.graph_indexer import (
            indexer_scores, select_topk, verify_indexer_scores,
        )

        score = verify_indexer_scores if getattr(state, "verify", False) else indexer_scores
        scores = score(
            query, weights, cache=state.cache, source_layer=source, ratio=ratio,
            positions=positions, valid_mask=valid,
        )
        state.attention_calls["visible_indexer"] += 1
    else:
        scores = indexer_topk(
            query, weights, cache=state.cache, source_layer=source, ratio=ratio,
            positions=positions, valid_mask=valid, topk=g.index_topk,
            max_seq_len=state.cache.max_seq_len, scores_only=True,
        )
    if publish:
        state.shared["candidates"] = select_candidate_blocks(scores, visible, block, candidate_k)
        state.shared["candidate_active"] = (visible > block * candidate_k) & (visible > g.index_topk)
    elif consume:
        masked = mask_candidate_logits(scores.clone(), state.shared["candidates"], block)
        scores = torch.where(state.shared["candidate_active"][:, None], masked, scores)
    if use_visible_indexer:
        return select_topk(scores, visible, valid, g.index_topk)
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
    packed = getattr(state, "packed_attention", None)
    verify = getattr(state, "verify", False)
    ordinary_packed = packed is not None and not verify
    native_packed = ordinary_packed and type(packed) is GraphPackedAttention
    verify_packed = (
        verify and type(packed) is GraphPackedAttention and packed.journal is cache
    )
    inplace_rope = native_packed or verify_packed
    table = state.freqs[attn.layer_id]
    if inplace_rope:
        # The runtime interns tables by RoPE base. Gather each one once
        # from the replay's current positions and reuse across layers.
        key = ("packed_rope", id(table))
        if key not in state.shared:
            state.shared[key] = table[positions]
        freqs = state.shared[key]
    else:
        freqs = table[positions]
    qr, q, kv, qr_quantized = attn.project_query_kv(
        x, prepare_indexer=bool(attn.compress_ratio and attn.is_index_source),
        reuse_ptpc=verify, fuse_qr_norm=verify,
    )
    q = q.reshape(-1, g.n_heads, g.head_dim)
    if native_packed:
        swa_slots = packed.swa_write_slots(attn.layer_id, positions, valid, state.shared)
    elif verify_packed:
        domain = packed.swa_domains[attn.layer_id]
        key = ("verify_swa_slots", domain)
        if key not in state.shared:
            state.shared[key] = cache.slots(
                attn.layer_id, "swa_kv", torch.where(valid, positions, -1)
            )
        swa_slots = state.shared[key]
    else:
        swa_slots = cache.slots(
            attn.layer_id, "swa_kv", torch.where(valid, positions, -1)
        )
    q_rotated = (
        try_rotate_q_and_write_swa(
            q, kv, freqs, cache, attn.layer_id, swa_slots, journal=verify_packed
        )
        if ordinary_packed or verify_packed else None
    )
    if q_rotated is None:
        q_rotated = _rotate_graph_value(
            q, freqs, inplace=verify_packed, preserve_signed_zero=verify_packed
        )
        kv = _rope_tail(kv, freqs)
        cache.write_rows(
            attn.layer_id, "swa_kv", swa_slots, attn.ops.encode_kv(kv, pool_tag="swa_kv")
        )
    else:
        counter = "direct_swa_journal" if verify_packed else "direct_swa_write"
        state.attention_calls[counter] += 1
    q = q_rotated
    if attn.is_kv_source:
        produce_global(attn, x, state, fused_graph=ordinary_packed, verify_rope=verify_packed)
    selected = None
    if attn.compress_ratio:
        if attn.is_index_source:
            selected = select_indices(
                attn, x, qr, state, freqs, fused_graph=ordinary_packed,
                verify_rope=verify_packed, qr_quantized=qr_quantized,
            )
            state.shared[attn.layer_id] = selected
        else:
            selected = state.shared[attn.index_source_layer_id]
    if packed is None:
        output = _shared_decode_attention(
            q, selected, cache=cache, layer_id=attn.layer_id,
            source_layer=attn.kv_source_layer_id, ratio=attn.compress_ratio,
            positions=positions, valid_mask=valid, window=g.window,
            sm_scale=g.sm_scale(), attn_sink=attn.attn_sink,
        )
    else:
        if inplace_rope:
            output = packed.forward(
                attn, q, selected, state.metadata, state.shared,
                mask_output=False,
                share_scheduler=native_packed and getattr(state, "grouped_moe", None) is None,
            )
        else:
            output = packed.forward(attn, q, selected, state.metadata, state.shared)
        state.attention_calls["packed_attention"] += 1
    rotated = _rotate_graph_value(
        output, freqs, inplace=ordinary_packed or verify_packed, inverse=True,
        valid_mask=valid if inplace_rope else None,
        preserve_signed_zero=verify_packed,
    )
    output = rotated.reshape(-1, g.o_groups, g.n_heads * g.head_dim // g.o_groups)
    if hasattr(attn.wo_a, "forward_grouped"):
        projected = attn.wo_a.forward_grouped(output)
        return attn.wo_b(projected.flatten(1)).reshape(shape)
    weight = attn.wo_a.weight
    verify_q4 = (
        verify_packed
        and state.metadata.batch_size == 8
        and state.metadata.q_len_per_req == 4
    )
    projected = (
        _try_batched_wo_a(output, weight, verify_q4=verify_q4)
        if (native_packed or verify_q4)
        and state.grouped_moe is None and g.o_lora_rank == 1024
        else None
    )
    if projected is not None:
        state.attention_calls["batched_wo_a"] += 1
    elif (ordinary_packed or verify_packed) and weight.is_contiguous() and not (
        torch.is_grad_enabled() and (output.requires_grad or weight.requires_grad)
    ):
        # PPU GEMM accepts row strides on both sides of the contraction.
        # Write the exact layout wo_b needs while retaining each BF16 GEMM.
        projected = output.new_empty((output.shape[0], g.o_groups, g.o_lora_rank))
        for i, group_weight in enumerate(weight.view(g.o_groups, g.o_lora_rank, -1)):
            torch.mm(output[:, i], group_weight.T, out=projected[:, i])
    else:
        projected = torch.stack([
            F.linear(output[:, i].contiguous(), weight[i * g.o_lora_rank:(i + 1) * g.o_lora_rank].contiguous())
            for i in range(g.o_groups)
        ], 1)
    return attn.wo_b(projected.flatten(1)).reshape(shape)
