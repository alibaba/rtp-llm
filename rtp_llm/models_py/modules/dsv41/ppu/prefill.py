"""Single-rank (CP1) V4.1 prefill orchestration for the PPU backend.

Ports the operator order of the CUDA reference
(``dsv41/block.py::forward_prefill_fast`` /
``dsv41/prefill/forward.py::forward_layers`` /
``dsv41/fp8/attention_v41.py``) with every kernel call replaced by a
``PPUOps`` contract op (or an L1 torch reference path):

per layer::

    [Engram inject]                (layers in engram_layer_ids)
    attn mHC pre  -> attn RMSNorm  -> attention  -> attn mHC post
    ffn  mHC pre  -> ffn  RMSNorm  -> MoE        -> ffn  mHC post

Attention internals per layer, all semantic roles preserved:

* every layer writes its own SWA KV (``attn.wkv`` -> kv_norm -> RoPE tail ->
  ``ops.encode_kv("swa_kv")`` -> cache);
* kv-source layers additionally produce the compressed global KV
  (``compressor.wkv``/``wgate`` pair pooling at odd absolute positions, the
  boundary state carried by ``csa_state``), the 68B indexer K
  (``indexer.wk`` -> k_norm -> RoPE) and their CSA states;
* index-source layers compute the indexer top-k candidates
  (``indexer.wq_b`` + ``weights_proj`` scoring over the source layer's
  indexer K via ``ops.index_score``); consumers REUSE their index source's
  top-k (cross-layer sharing, no copy);
* sparse attention combines the global compressed KV entries selected by the
  top-k with the SWA window. On an accelerator device with ratio != 0 this
  goes through ``ops.sparse_attention`` (L1 workspace + flash_mla);
  everywhere else a pure-torch reference attention runs over an identically
  assembled workspace (the correctness baseline; flash_mla is not importable
  on CPU hosts, and ratio-0 layers have no global pool to feed the contract
  op — see the module TODO for the SWA-only device path).

The output projection applies the inverse RoPE on the rope tail, the grouped
``wo_a`` (per ``o_groups`` head group) and ``wo_b``.

Chunk/prefix semantics: the Engram n-gram windows cross chunk boundaries
through ``lookback_token_ids`` (most-recent-first predecessor ids per
request, ``-1`` before the sequence starts); ratio-2 pairing aligns to
absolute-position parity and reads the missing left token from the
``csa_state`` pool when a chunk starts at an odd position.

CP8 (wave 2): pass a :class:`~rtp_llm.models_py.modules.dsv41.ppu.cp.
CPPrefillContext` as ``parallel_context`` (built from
``cp.build_zigzag_plan``). The input tensors are then the rank's zigzag
chunk (padding rows clamped to valid positions; see ``ppu/cp.py``): query
tokens, MoE routing, mHC and the output projection run rank-local; the
compressor values/scores and the SWA KV rows are all-gathered and restored
to original model order before pooling, so every rank's cache holds the
full request state (v1: CP KV sharding off); CP padding rows never enter
the pools, the Engram hash windows or the expert dispatch ids; the final
hidden is restored to logical token order on every rank.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor, V41StepContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.attention import COMBINED_WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache, PPURequestSlots
from rtp_llm.models_py.modules.dsv41.ppu.ced import PPUCEDPlan, decoder_replay_starts
from rtp_llm.models_py.modules.dsv41.ppu.cp import CPPrefillContext, cp_context_of
from rtp_llm.models_py.modules.dsv41.ppu.engram import (
    build_token_windows,
    compute_engram_hash_ids,
    engram_gate_inject,
    wkv_project,
)
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.weight import PPUWeightBundle, V41Geometry

__all__ = ["PrefillResult", "run_prefill", "build_step_context"]


@dataclass
class PrefillResult:
    """Outputs of one prefill run."""

    hidden: torch.Tensor  # [T, hidden] final-normed, pre-lm-head
    logits: Optional[torch.Tensor]  # [T, vocab] when apply_lm_head
    cache: PPUKVCache
    slots: Dict[int, PPURequestSlots]  # request id -> handles
    ced_info: Optional[dict] = None
    mtp_target_hidden_states: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Small shared helpers (reference math, CPU-safe)
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: Optional[torch.Tensor], eps: float):
    """Reference RMSNorm (fp32 math, single cast back to the input dtype)."""
    if weight is None:
        weight = torch.ones(x.shape[-1], dtype=torch.float32, device=x.device)
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * weight.float()
    return y.to(x.dtype)


def _rope_tail(x: torch.Tensor, freqs: torch.Tensor, inverse: bool = False):
    """Rotate the LAST ``2 * freqs.shape[-1]`` dims of ``x`` with ``freqs``.

    ``x`` is ``[T, D]`` or ``[T, H, D]``; ``freqs`` is complex64
    ``[T, rope_head_dim // 2]``. Matches ``dsv41/rope.py`` (fp32 rotation,
    one cast back to the input dtype). Returns a new tensor.
    """
    half = int(freqs.shape[-1])
    rope_dim = half * 2
    if x.shape[-1] < rope_dim:
        raise ValueError(
            f"rope tail {rope_dim} exceeds the tensor's last dim {x.shape[-1]}"
        )
    tokens = x.shape[0]
    if freqs.shape[0] != tokens:
        raise ValueError(f"freqs rows {freqs.shape[0]} != token rows {tokens}")
    tail = x[..., -rope_dim:].float()
    tail = tail.reshape(tokens, *x.shape[1:-1], half, 2)
    complex_tail = torch.view_as_complex(tail)  # [T, ..., half]
    table = freqs.conj() if inverse else freqs  # [T, half]
    if complex_tail.dim() == 2:
        rotated = complex_tail * table
    else:
        broadcast = (tokens,) + (1,) * (complex_tail.dim() - 2) + (half,)
        rotated = complex_tail * table.reshape(broadcast)
    real = torch.view_as_real(rotated).reshape(x.shape[:-1] + (rope_dim,))
    out = x.clone()
    out[..., -rope_dim:] = real.to(x.dtype)
    return out


def build_step_context(
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    parallel_context: Optional[object] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> V41StepContext:
    """The frozen-contract step context (rank-local rows, global positions).

    ``valid_mask`` marks real rows — under CP8 the rank-local chunk carries
    zigzag padding rows whose outputs the restore drops; CP1 keeps the
    default all-ones mask.
    """
    rows = positions.numel()
    if valid_mask is None:
        valid_mask = torch.ones(rows, dtype=torch.bool, device=positions.device)
    return V41StepContext(
        positions=positions.to(torch.int64).reshape(-1),
        valid_mask=valid_mask,
        request_ids=request_ids.reshape(-1),
        owner_ranks=torch.zeros(rows, dtype=torch.int64, device=positions.device),
        request_offsets=cu_seqlens.to(torch.int64),
        parallel_context=parallel_context,
    )


# ---------------------------------------------------------------------------
# Reference sparse attention (torch math over the assembled workspace)
# ---------------------------------------------------------------------------


def _torch_sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    lengths: torch.Tensor,
    sm_scale: float,
    attn_sink: Optional[torch.Tensor],
) -> torch.Tensor:
    """Softmax attention over the workspace rows selected by ``indices``.

    Mirrors the flash_mla sparse call: per row the valid indices are
    ``indices[row, 0, :lengths[row]]`` into ``kv [R, 1, 512]``. Each head's
    ``attn_sink`` is a separate logit with a zero value: it contributes to
    the softmax denominator without contributing to the weighted values.
    """
    rows = q.shape[0]
    out = torch.zeros_like(q)
    sink = None if attn_sink is None else attn_sink.to(q.device).float().reshape(-1)
    for row in range(rows):
        count = int(lengths[row])
        if count <= 0:
            continue
        valid = indices[row, 0, :count].to(torch.int64)
        keys = kv[valid, 0].float()  # [n, 512]
        scores = torch.einsum("hd,kd->hk", q[row].float(), keys) * sm_scale
        if sink is None:
            probs = torch.softmax(scores, dim=-1)
        else:
            normalizer = torch.logsumexp(
                torch.cat((scores, sink[:, None]), dim=-1), dim=-1, keepdim=True
            )
            probs = torch.exp(scores - normalizer)
        out[row] = (probs @ keys).to(q.dtype)
    return out


def _request_rows(request_ids: torch.Tensor) -> Dict[int, Tuple[int, ...]]:
    """request id -> its row indices, first-appearance order."""
    rows: Dict[int, Tuple[int, ...]] = {}
    for row, rid in enumerate(request_ids.tolist()):
        rows.setdefault(int(rid), []).append(row)
    return {rid: tuple(index) for rid, index in rows.items()}


def _request_id_list(request_ids: torch.Tensor, cu: torch.Tensor) -> Tuple[int, ...]:
    """Per-request ids in first-appearance order, validated against ``cu``.

    Explicit ``request_ids`` may carry arbitrary int ids (wave2 fix ③):
    slot allocation and the cache bookkeeping use THESE ids, not positional
    ``range(B)``. Each ``cu`` segment must carry exactly one request id and
    each id exactly one segment — anything else fails loudly instead of
    silently mis-mapping slots.
    """
    batch = int(cu.numel()) - 1
    ids: List[int] = []
    seen = set()
    for value in request_ids.tolist():
        rid = int(value)
        if rid not in seen:
            seen.add(rid)
            ids.append(rid)
    if len(ids) != batch:
        raise ValueError(
            f"cu_seqlens describes {batch} requests but request_ids carries "
            f"{len(ids)} distinct ids"
        )
    for index, rid in enumerate(ids):
        segment = request_ids[int(cu[index]) : int(cu[index + 1])]
        if segment.numel() == 0 or not bool((segment == rid).all()):
            raise ValueError(
                "request_ids must be constant within each cu_seqlens segment "
                f"(segment {index} does not carry only id {rid})"
            )
    return tuple(ids)


def _build_reference_workspace(
    *,
    ops,
    cache: PPUKVCache,
    layer_id: int,
    geometry: V41Geometry,
    context: V41StepContext,
    selected_blocks: Optional[torch.Tensor],
    window: int,
    swa_starts: Optional[dict] = None,
):
    """CPU reference workspace: global rows + the queries' full SWA range.

    Unlike L1's ``build_sparse_workspace`` (whose SWA tail assumes the query
    rows are the request's last window — the decode/chunk-tail shape), this
    builder materializes every SWA position any query row can see:
    ``[max(0, min_pos - (window - 1)), max_pos + 1]``, plus the request's
    global compressed entries. Per token the combined index row is
    ``[top-k global entries | SWA window | -1 padding]`` with the row's
    written-entry count in ``lengths`` — the flash_mla sparse layout.
    """
    positions = context.positions
    request_ids = context.request_ids
    rows = positions.shape[0]
    device = positions.device
    ratio = geometry.ratio(layer_id)
    source = geometry.kv_source_for(layer_id)
    descriptors = cache.descriptors(layer_id if source is None else source)
    swa_desc = cache.descriptors(layer_id)["swa_kv"]

    kv_rows = []
    spans = {}
    row_total = 0
    for rid, row_indices in _request_rows(request_ids).items():
        if ratio and source is not None:
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            count = cache.request_length(rid) // ratio
            packed = cache.read_source(source, rid, global_entries=count)[0]
            decoded = torch.empty(
                (packed.shape[0], geometry.head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            ops.decode_kv(
                packed, pool_tag=tag, descriptor=descriptors[tag], out=decoded
            )
        else:
            decoded = torch.zeros(
                (0, geometry.head_dim), dtype=torch.bfloat16, device=device
            )
        gbase = row_total
        row_total += decoded.shape[0]
        kv_rows.append(decoded)
        min_pos = int(min(int(positions[row]) for row in row_indices))
        max_pos = int(max(int(positions[row]) for row in row_indices))
        replay_start = int((swa_starts or {}).get(rid, 0))
        swa_first = max(0, replay_start, min_pos - (window - 1))
        swa_positions = torch.arange(
            swa_first, max_pos + 1, dtype=torch.int64, device=device
        )
        packed_swa = cache.read_swa(layer_id, rid, swa_positions)
        swa_decoded = torch.empty(
            (packed_swa.shape[0], geometry.head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        ops.decode_kv(
            packed_swa, pool_tag="swa_kv", descriptor=swa_desc, out=swa_decoded
        )
        sbase = row_total
        row_total += swa_decoded.shape[0]
        kv_rows.append(swa_decoded)
        spans[rid] = (gbase, decoded.shape[0], sbase, swa_first)

    kv = (
        torch.cat(kv_rows, dim=0).reshape(-1, 1, geometry.head_dim)
        if kv_rows
        else torch.zeros((0, 1, geometry.head_dim), dtype=torch.bfloat16, device=device)
    )
    indices = torch.full(
        (rows, 1, COMBINED_WIDTH), -1, dtype=torch.int32, device=device
    )
    lengths = torch.zeros(rows, dtype=torch.int32, device=device)
    for rid, row_indices in _request_rows(request_ids).items():
        gbase, gcount, sbase, swa_first = spans[rid]
        for row in row_indices:
            pos = int(positions[row])
            written = 0
            if selected_blocks is not None and gcount:
                topk = selected_blocks.shape[1]
                topk_len = min((pos + 1) // ratio, topk)
                if topk_len > 0:
                    entries = selected_blocks[row, :topk_len].to(torch.int64)
                    keep = entries[(entries >= 0) & (entries < gcount)]
                    if keep.numel():
                        indices[row, 0, written : written + keep.numel()] = (
                            gbase + keep
                        ).to(torch.int32)
                        written += int(keep.numel())
            replay_start = int((swa_starts or {}).get(rid, 0))
            swa_len = max(0, min(pos + 1, window, pos - replay_start + 1))
            if swa_len > 0:
                first = pos - swa_len + 1
                local = torch.arange(first, pos + 1, device=device)
                indices[row, 0, written : written + swa_len] = (
                    sbase + local - swa_first
                ).to(torch.int32)
                written += swa_len
            lengths[row] = written
    return kv, indices, lengths


def _layer_attention(
    q: torch.Tensor,
    selected_blocks: torch.Tensor,
    *,
    ops,
    cache: PPUKVCache,
    layer_id: int,
    geometry: V41Geometry,
    context: V41StepContext,
    attn_sink: Optional[torch.Tensor],
    window: int,
    swa_starts: Optional[dict] = None,
) -> torch.Tensor:
    """Sparse attention for one layer (contract op on device, reference on CPU).

    Device + ratio != 0 goes through ``ops.sparse_attention`` (L1 workspace +
    flash_mla). CPU hosts — and ratio-0 layers, which own no global pool to
    feed the contract op (wave 2 wires the SWA-only flash path) — run the
    torch reference below over an identically-shaped workspace.
    """
    ratio = geometry.ratio(layer_id)
    if q.is_cuda and ratio != 0:
        # Integrated PPU path: L1 builds the bounded workspace and calls
        # flash_mla_sparse_fwd through the contract op.
        view = cache.cache_view(layer_id)
        return ops.sparse_attention(
            q,
            selected_blocks,
            cache=view,
            layer_id=layer_id,
            context=context,
            attn_sink=attn_sink,
            ratio=ratio,
            window=window,
            swa_starts=swa_starts,
        )
    # CPU reference path (and ratio-0 everywhere: those layers have no global
    # pool to feed the contract op — see the wave-2 TODO above).
    kv, indices, lengths = _build_reference_workspace(
        ops=ops,
        cache=cache,
        layer_id=layer_id,
        geometry=geometry,
        context=context,
        selected_blocks=selected_blocks,
        window=window,
        swa_starts=swa_starts,
    )
    return _torch_sparse_attention(
        q, kv, indices, lengths, geometry.sm_scale(), attn_sink
    )


# ---------------------------------------------------------------------------
# KV production (kv-source layers)
# ---------------------------------------------------------------------------


def _compress_pairs(
    values: torch.Tensor, scores: torch.Tensor, norm, eps: float
) -> torch.Tensor:
    """Pair pooling: softmax over the pair per channel, then RMSNorm."""
    weights = torch.softmax(scores.float(), dim=-2)
    # The compressor emits BF16 pooled values before RMSNorm. Its C2
    # projections and unfinished-pair state retain FP32 separately.
    latent = (values.float() * weights).sum(-2).to(torch.bfloat16)
    return _rms_norm(latent, norm, eps).to(torch.bfloat16)


def _produce_global(
    *,
    ops,
    layer,
    geometry: V41Geometry,
    bundle: PPUWeightBundle,
    values: torch.Tensor,
    scores: Optional[torch.Tensor],
    positions: torch.Tensor,
    request_ids: torch.Tensor,
    cache: PPUKVCache,
    slots: Dict[int, PPURequestSlots],
    layer_id: int,
    append_starts: Optional[dict] = None,
) -> None:
    """Produce + write global KV, indexer K and CSA states for one layer.

    ``values`` / ``scores`` are the compressor projections of the batch's
    rows in the SAME order as ``positions`` / ``request_ids`` — rank-local
    under CP1, original-order (restored) under CP8 (the caller gathered
    them; the pooling below then runs identically on every rank).

    Ratio 2 pairs are aligned to absolute-position parity (pair = even|odd
    absolute positions, rope position = the pair's even start); a chunk that
    begins at an odd position reads its left token from ``csa_state``.
    """
    ratio = geometry.ratio(layer_id)
    tag = "global_kv_2" if ratio == 2 else "global_kv_1"
    freqs = bundle.freqs_for(layer_id)
    per_request = _request_rows(request_ids)

    for rid, row_indices in per_request.items():
        handle = slots[rid]
        start = int(positions[row_indices[0]])
        end = int(max(int(positions[row]) for row in row_indices)) + 1
        begin, stop = start // ratio, end // ratio
        if ratio == 1:
            latent = _rms_norm(
                values[list(row_indices)], layer.compressor_norm, geometry.norm_eps
            ).to(torch.bfloat16)
            entry_positions = positions[list(row_indices)]
        else:
            boundaries = [p for p in range(start, end) if p % 2 == 1]
            left_values, right_values = [], []
            left_scores, right_scores = [], []
            left_positions = []
            for boundary in boundaries:
                left_pos = boundary - 1
                if left_pos >= start:
                    index = row_indices[left_pos - start]
                    left_values.append(values[index])
                    left_scores.append(scores[index])
                else:
                    state = cache.read_csa(
                        layer_id, rid, torch.tensor([left_pos], dtype=torch.int64)
                    )
                    if not bool(state.any()):
                        raise RuntimeError(
                            f"ratio-2 chunk of request {rid} starts at odd "
                            f"position {start} without a prior csa_state entry "
                            "at "
                            f"{left_pos}; the compression phase is missing"
                        )
                    left_values.append(state[0, :512].to(values.dtype))
                    left_scores.append(state[0, 512:].to(scores.dtype))
                right_index = row_indices[boundary - start]
                right_values.append(values[right_index])
                right_scores.append(scores[right_index])
                left_positions.append(left_pos)
            if not boundaries:
                latent = values.new_zeros((0, geometry.head_dim), dtype=torch.bfloat16)
                entry_positions = positions.new_zeros(0)
            else:
                pair_values = torch.stack(
                    (torch.stack(left_values, 0), torch.stack(right_values, 0)), dim=1
                )
                pair_scores = torch.stack(
                    (torch.stack(left_scores, 0), torch.stack(right_scores, 0)), dim=1
                )
                latent = _compress_pairs(
                    pair_values,
                    pair_scores,
                    layer.compressor_norm,
                    geometry.norm_eps,
                )
                entry_positions = torch.tensor(
                    left_positions, dtype=torch.int64, device=positions.device
                )
        entry_count = stop - begin
        if latent.shape[0] != entry_count:
            raise RuntimeError(
                f"layer {layer_id} produced {latent.shape[0]} entries for "
                f"request {rid}, expected {entry_count} (start={start}, end={end})"
            )
        safe_positions = entry_positions.clamp_max(freqs.shape[0] - 1)
        global_keys = _rope_tail(latent, freqs[safe_positions])
        packed_global = ops.encode_kv(global_keys, pool_tag=tag)
        first_write = max(begin, int((append_starts or {}).get(rid, start)) // ratio)
        write_offset = first_write - begin
        cache.insert_rows(
            layer_id,
            tag,
            handle.global_slots[layer_id][first_write:stop],
            packed_global[write_offset:],
        )
        # Indexer K comes from the same latents.
        projected = F.linear(latent, layer.index_wk.weight)
        projected = _rms_norm(projected, layer.index_k_norm, geometry.norm_eps)
        index_keys = _rope_tail(projected, freqs[safe_positions])
        packed_index = ops.encode_kv(index_keys, pool_tag="indexer_kv")
        cache.insert_rows(
            layer_id,
            "indexer_kv",
            handle.indexer_slots[layer_id][first_write:stop],
            packed_index[write_offset:],
        )
        if ratio == 2:
            # CSA states at RAW positions: [values | scores] fp32.
            state_indices = [
                row
                for row in row_indices
                if int(positions[row]) >= int((append_starts or {}).get(rid, start))
            ]
            if not state_indices:
                continue
            state = torch.cat(
                (
                    values[state_indices],
                    scores[state_indices],
                ),
                dim=-1,
            ).float()
            state_slots = cache.slots_at_positions(
                layer_id, "csa_state", rid, positions[state_indices]
            )
            cache.insert_rows(layer_id, "csa_state", state_slots, state)


# ---------------------------------------------------------------------------
# Indexer top-k
# ---------------------------------------------------------------------------


def _quantized_indexer_query(query: torch.Tensor) -> V41QuantizedTensor:
    """Apply the post-RoPE group-32 FP4/UE8M0 query boundary before scoring."""
    payload, scales = _quant.quantize_indexer_q(query.contiguous())
    return V41QuantizedTensor(payload, scales, "fp4_ue8m0")


def _indexer_topk(
    *,
    ops,
    layer,
    geometry: V41Geometry,
    bundle: PPUWeightBundle,
    x_pre: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    context: V41StepContext,
    cache: PPUKVCache,
    slots: Dict[int, PPURequestSlots],
    layer_id: int,
) -> torch.Tensor:
    """Score + select the top-k compressed entries for one index-source layer.

    The indexer K pool belongs to this layer's KV SOURCE, not the layer
    itself: only kv-source layers carry the indexer K projection (the CUDA
    reference loads ``index_wk`` under ``is_kv_source``), and
    ``_select_indices`` scores index sources against
    ``shared["global"][kv_source_layer_id]`` — pure index sources (24/28/32/36
    in the real config, none of them kv sources) read their source's pool.
    """
    heads = geometry.index_n_heads
    source = geometry.kv_source_for(layer_id)
    if source is None:
        raise RuntimeError(
            f"index-source layer {layer_id} has no kv-source layer below it; "
            "the config is inconsistent"
        )
    source_ratio = geometry.ratio(source)
    # Query side: reuse the q_lora representation, project + rope.
    index_q = layer.index_wq(qr).reshape(-1, heads, geometry.index_head_dim)
    index_q = _rope_tail(index_q, bundle.freqs_for(layer_id)[positions])
    # Head weights: projection of the normed input, softmax-scale prescaled.
    raw_weights = F.linear(x_pre, layer.index_weights_proj).float()
    head_weights = raw_weights * (geometry.index_head_dim * heads) ** -0.5
    # Key side: the KV SOURCE's indexer pool, dequantized through the op.
    keys = []
    offsets = torch.zeros_like(positions, dtype=torch.int64)
    limits = torch.zeros_like(offsets)
    running = 0
    for rid in slots:
        handle = slots[rid]
        count = handle.global_slots[source].shape[0]
        packed = cache.read_source(source, rid, global_entries=count)[1]
        decoded = torch.empty(
            (packed.shape[0], geometry.index_head_dim),
            dtype=torch.bfloat16,
            device=x_pre.device,
        )
        ops.decode_kv(
            packed,
            pool_tag="indexer_kv",
            descriptor=cache.descriptors(source)["indexer_kv"],
            out=decoded,
        )
        keys.append(decoded)
        rows = context.request_ids == rid
        offsets[rows] = running
        visible = ((positions[rows] + 1) // source_ratio).clamp_max(count)
        limits[rows] = running + visible
        running += count
    keys = (
        torch.cat(keys, dim=0)
        if keys
        else x_pre.new_zeros((0, geometry.index_head_dim))
    )
    total = keys.shape[0]
    key_positions = (
        torch.arange(total, device=x_pre.device, dtype=torch.int64) * source_ratio
    )
    logits = ops.index_score(
        _quantized_indexer_query(index_q),
        V41QuantizedTensor(keys, None, "bf16"),
        head_weights,
        layer_id=layer_id,
        candidate_blocks=None,
        key_positions=key_positions,
        context=replace(context, positions=limits * source_ratio - 1),
        ratio=source_ratio,
    )
    # The op's causal count spans the concatenated bank. Restrict each row to
    # its own request and convert selected columns back to local entry IDs.
    columns = torch.arange(total, device=x_pre.device)
    keep = (columns[None, :] >= offsets[:, None]) & (columns[None, :] < limits[:, None])
    logits = logits.masked_fill(~keep, float("-inf"))
    k = min(geometry.index_topk, total)
    if k == 0:
        return torch.full(
            (x_pre.shape[0], 1), -1, dtype=torch.int32, device=x_pre.device
        )
    top_scores, ids = logits.topk(k, dim=-1)
    return torch.where(torch.isfinite(top_scores), ids - offsets[:, None], -1).to(
        torch.int32
    )


# ---------------------------------------------------------------------------
# The prefill driver
# ---------------------------------------------------------------------------


def run_prefill(
    bundle: PPUWeightBundle,
    ops,
    cache: PPUKVCache,
    input_ids: torch.Tensor,
    *,
    positions: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    request_ids: Optional[torch.Tensor] = None,
    lookback_token_ids: Optional[torch.Tensor] = None,
    apply_lm_head: bool = False,
    parallel_context: Optional[object] = None,
    policy: Optional[PrefillPolicy] = None,
    allow_ced: bool = True,
    compact_ced: bool = True,
    replay_prefix_lengths=None,
    capture_layer_ids=(),
) -> PrefillResult:
    """Run the full single-rank prefill over one flat token batch.

    Args:
        bundle: the rank's weights (``load_ppu_weights``).
        ops: a bound ``PPUOps`` (``bundle.bind(ops)`` must have run).
        cache: the rank's :class:`PPUKVCache`.
        input_ids: flat ``[T]`` int token ids.
        positions: per-token global absolute positions; default continues
            each request from its cache-tracked length (created on the input
            device).
        cu_seqlens: ``[B + 1]`` request boundaries; default one request.
        request_ids: per-token request ids; default derived from cu_seqlens.
            Arbitrary int ids are honored (wave2 fix ③): each cu_seqlens
            segment must carry exactly one id and slot allocation uses the
            ids themselves. Under CP the ids must match the zigzag plan's.
        lookback_token_ids: ``[B, ngram - 1]`` most-recent-first predecessor
            token ids for the Engram hash (``-1`` before the sequence
            starts); default fresh-prefill (all ``-1``).
        apply_lm_head: also return ``[T, vocab]`` logits.
        parallel_context: a ``CPPrefillContext`` (see ``ppu/cp.py``) turns
            this call into one CP rank of a zigzag CP8 forward: the input
            tensors must be the rank's chunk (``plan.local_*``), the KV
            inputs are all-gathered/restored across the CP group, and the
            returned hidden/logits are in logical global order (identical
            on every rank).
    """
    geometry = bundle.geometry
    capture_layer_ids = tuple(map(int, capture_layer_ids))
    if len(set(capture_layer_ids)) != len(capture_layer_ids) or any(
        layer < 0 or layer >= geometry.num_layers for layer in capture_layer_ids
    ):
        raise ValueError("target capture layers must be unique valid layer ids")
    # DSpark consumes real features for every committed prompt row. CED's
    # scatter-to-original output contains holes before its retained tail.
    if capture_layer_ids:
        allow_ced = False
    captured = {}
    policy = (policy or cache.prefill_policy).validate()
    cp_ctx: Optional[CPPrefillContext] = cp_context_of(parallel_context)
    input_ids = input_ids.reshape(-1)
    tokens = int(input_ids.numel())
    if cp_ctx is not None:
        if positions is None or cu_seqlens is None:
            raise ValueError(
                "CP prefill requires the rank-local positions and cu_seqlens "
                "from cp.build_zigzag_plan (plan.local_positions / plan.cu_local)"
            )
        if tokens != cp_ctx.chunk_length:
            raise ValueError(
                f"CP prefill got {tokens} local rows but the plan's chunk "
                f"length is {cp_ctx.chunk_length}"
            )
    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, tokens], dtype=torch.int64)
    # Device discipline (wave2 fix ②): every bookkeeping tensor derived or
    # normalized here lives on the INPUT device — cuda inputs must not mix
    # with CPU request_ids/positions (the default branches used to build CPU
    # tensors unconditionally, crashing only after the long weight load).
    device = input_ids.device
    cu = cu_seqlens.to(device=device, dtype=torch.int64).reshape(-1)
    if request_ids is None:
        request_ids = torch.repeat_interleave(
            torch.arange(cu.numel() - 1, dtype=torch.int64, device=cu.device),
            cu[1:] - cu[:-1],
        )
    request_ids = request_ids.to(device=device, dtype=torch.int64).reshape(-1)
    request_id_list = _request_id_list(request_ids, cu)
    if positions is None:
        positions = torch.empty(tokens, dtype=torch.int64, device=device)
        for index, rid in enumerate(request_id_list):
            start_row, end_row = int(cu[index]), int(cu[index + 1])
            base = 0 if replay_prefix_lengths is not None else cache.request_length(rid)
            positions[start_row:end_row] = base + torch.arange(
                end_row - start_row, dtype=torch.int64, device=device
            )
    positions = positions.to(device=device, dtype=torch.int64).reshape(-1)
    append_starts = None
    if replay_prefix_lengths is not None:
        if not policy.encoder_replay:
            raise ValueError("query replay requires encoder SWA recovery policy")
        if len(replay_prefix_lengths) != len(request_id_list):
            raise ValueError("one replay prefix length is required per request")
        append_starts = dict(zip(request_id_list, map(int, replay_prefix_lengths)))
        for index, rid in enumerate(request_id_list):
            prefix = append_starts[rid]
            if prefix < 0 or prefix != cache.request_length(rid):
                raise ValueError(
                    "replay prefix differs from the restored Global KV prefix"
                )
            if cp_ctx is None:
                begin, end = int(cu[index]), int(cu[index + 1])
                query_start, query_length = int(positions[begin]), end - begin
            else:
                split = cp_ctx.plan.requests[index]
                query_start, query_length = split.prefix, split.length
            if query_start != 0 or query_length < prefix:
                raise ValueError(
                    "encoder recovery requires complete history from token zero"
                )

    if policy.bounded_replay:
        for index, rid in enumerate(request_id_list):
            if cp_ctx is None:
                begin, end = int(cu[index]), int(cu[index + 1])
                prefix, fresh = int(positions[begin]), end - begin
            else:
                split = cp_ctx.plan.requests[index]
                prefix, fresh = split.prefix, split.length
            if prefix > 0 and fresh < geometry.window:
                raise ValueError(
                    f"request {rid} needs at least {geometry.window} replay tokens "
                    "when decoder SWA is not restored; reduce the reused prefix"
                )

    # Allocate pool space per request and validate position bookkeeping.
    # Under CP8 the cache holds the FULL request (v1: no CP KV sharding); the
    # rank-local chunk's zigzag positions are intentionally non-contiguous.
    slots: Dict[int, PPURequestSlots] = {}
    for index, rid in enumerate(request_id_list):
        start_row, end_row = int(cu[index]), int(cu[index + 1])
        n_new = end_row - start_row
        if cp_ctx is not None:
            split = cp_ctx.plan.requests[index]
            if rid != split.request_id:
                raise ValueError(
                    f"request id {rid} (segment {index}) disagrees with the CP "
                    f"plan's request id {split.request_id}; the zigzag plan "
                    "owns the request ids under CP"
                )
            slots[rid] = cache.alloc(
                rid, split.length - (append_starts or {}).get(rid, 0)
            )
            continue
        expected_start = cache.request_length(rid)
        actual_start = int(positions[start_row])
        if actual_start != (0 if append_starts is not None else expected_start):
            raise ValueError(
                f"request {rid} positions start at {actual_start} but the cache "
                f"tracks {expected_start} existing tokens"
            )
        slots[rid] = cache.alloc(rid, n_new - (append_starts or {}).get(rid, 0))
        if slots[rid].total_tokens != int(positions[end_row - 1]) + 1:
            raise ValueError(
                f"request {rid} positions are inconsistent with its length"
            )
    context = build_step_context(
        positions,
        request_ids,
        cu,
        parallel_context,
        valid_mask=(
            cp_ctx.plan.local_is_real.to(positions.device)
            if cp_ctx is not None
            else None
        ),
    )

    # Engram hashes (once; per-layer slices below). Under CP8 the windows are
    # built over the FULL original-order sequence (per-token hash math is
    # unaffected by which rank executes it) and sliced to the rank's rows by
    # global position; zigzag padding rows are masked out of the token mask.
    hashes = None
    token_mask = None
    if geometry.engram_enabled():
        if cp_ctx is not None:
            plan = cp_ctx.plan
            lookback = plan.full_lookback
            if lookback is None:
                lookback = torch.full(
                    (
                        int(plan.full_cu_seqlens.numel()) - 1,
                        geometry.engram_max_ngram_size - 1,
                    ),
                    -1,
                    dtype=torch.int64,
                )
            windows = build_token_windows(
                plan.full_input_ids, plan.full_cu_seqlens, lookback
            )
            hashes_full = compute_engram_hash_ids(windows, bundle.engram_hash_params)
            mask_full = (windows[:, 0] >= 0) & (
                windows[:, 0] != geometry.image_token_id
            )
            prefixes = torch.tensor(
                [split.prefix for split in plan.requests],
                dtype=torch.int64,
                device=positions.device,
            )
            full_row = (
                plan.full_cu_seqlens.to(positions.device)[request_ids]
                + positions
                - prefixes[request_ids]
            )
            hashes = hashes_full.to(positions.device)[full_row]
            token_mask = mask_full.to(full_row.device)[full_row] & (
                plan.local_is_real.to(full_row.device)
            )
        else:
            if lookback_token_ids is None:
                lookback_token_ids = torch.full(
                    (int(cu.numel()) - 1, geometry.engram_max_ngram_size - 1),
                    -1,
                    dtype=torch.int64,
                )
            windows = build_token_windows(input_ids, cu, lookback_token_ids)
            # The hash math runs over the host-normalized windows; move the
            # ids to the compute device so the engram lookup + WKV
            # projection below mix with the device-resident weights (and the
            # lookup takes the pinned-shard gather fast path on the PPU).
            hashes = compute_engram_hash_ids(windows, bundle.engram_hash_params).to(
                input_ids.device
            )
            token_mask = (windows[:, 0] >= 0) & (
                windows[:, 0] != geometry.image_token_id
            )

    # Embedding + mHC lane expansion.
    hidden = bundle.embedding[input_ids.to(torch.int64)]  # [T, dim] bf16
    h = (
        hidden.unsqueeze(1)
        .expand(tokens, geometry.hc_mult, geometry.hidden_size)
        .contiguous()
    )  # [T, hc, dim]
    prev_mix = None  # delayed-mHC chain state (sublayer N's pre result)
    shared_topk: Dict[int, torch.Tensor] = {}
    decoder_start = policy.decoder_start(geometry)
    if cp_ctx is None:
        lengths = [int(cu[i + 1] - cu[i]) for i in range(cu.numel() - 1)]
        prefixes = [int(positions[int(cu[i])]) for i in range(cu.numel() - 1)]
    else:
        lengths = [split.length for split in cp_ctx.plan.requests]
        prefixes = [split.prefix for split in cp_ctx.plan.requests]
    swa_starts = None
    if policy.bounded_replay:
        swa_starts = (
            decoder_replay_starts(request_id_list, lengths, prefixes, geometry.window)
            if allow_ced
            else dict(zip(request_id_list, prefixes))
        )
    ced = (
        PPUCEDPlan.create(
            geometry, policy, input_ids, positions, request_ids, cu, cp_ctx
        )
        if allow_ced
        else None
    )
    compacted = ced is not None and compact_ced

    for layer_id in range(geometry.num_layers):
        if layer_id == decoder_start and compacted:
            h = ced.compact(h)
            prev_mix = ced.compact_mix(prev_mix)
            source = geometry.index_source_for(layer_id)
            if source in shared_topk:
                shared_topk = {source: ced.compact(shared_topk[source])}
            input_ids, positions = ced.input_ids, ced.positions
            request_ids, cu = ced.request_ids, ced.cu_seqlens
            cp_ctx = ced.context
            tokens = int(input_ids.numel())
            context = build_step_context(
                positions,
                request_ids,
                cu,
                cp_ctx,
                valid_mask=(
                    cp_ctx.plan.local_is_real.to(positions.device)
                    if cp_ctx is not None
                    else None
                ),
            )
            logging.getLogger(__name__).info("PPU CED compacted %s", ced.info)
        if layer_id in capture_layer_ids:
            captured[layer_id] = h.mean(dim=-2)
        layer = bundle.layers[layer_id]
        freqs = bundle.freqs_for(layer_id)
        rope_positions = positions.clamp_max(freqs.shape[0] - 1)

        # ---- Engram inject (before the delayed mHC projection).
        if layer_id in geometry.engram_layer_ids:
            hash_index = geometry.engram_layer_ids.index(layer_id)
            rows = hashes[:, hash_index]  # [T, n_hash_cols]
            embedded = ops.engram_lookup(rows, layer_id=layer_id)
            flat = embedded.reshape(tokens, -1)
            kv = wkv_project(flat, layer.engram_wkv.weight)
            h = engram_gate_inject(
                h,
                layer.engram_q_weight,
                layer.engram_k_weight,
                kv,
                token_mask=token_mask,
            )

        # ---- Attention sublayer.
        residual = h
        step_attn = ops.mhc(
            residual,
            None,
            phase="pre",
            parameters=layer.mhc_parameters("attn"),
            state=prev_mix,
        )
        prev_mix = step_attn
        x_pre = _rms_norm(step_attn.hidden, layer.attn_norm, geometry.norm_eps)

        # Query path: wq_a -> q_norm -> wq_b -> RoPE tail per head.
        qr = _rms_norm(layer.wq_a(x_pre), layer.q_norm, geometry.norm_eps)
        q = layer.wq_b(qr).reshape(tokens, geometry.n_heads, geometry.head_dim)
        q = _rope_tail(q, freqs[rope_positions])

        # SWA KV (every layer): wkv -> kv_norm -> RoPE -> 528B pool. Under CP8
        # the rank-local rows are all-gathered and restored to original order
        # first, so every rank's pool holds the FULL request (padding rows are
        # dropped by the restore permutation and never reach the pool).
        kv_latent = layer.wkv(x_pre)
        kv_latent = _rms_norm(kv_latent, layer.kv_norm, geometry.norm_eps)
        kv_latent = _rope_tail(kv_latent, freqs[rope_positions])
        if cp_ctx is None:
            packed_swa = ops.encode_kv(kv_latent, pool_tag="swa_kv")
            for rid, row_indices in _request_rows(request_ids).items():
                swa_slots = cache.slots_at_positions(
                    layer_id, "swa_kv", rid, positions[list(row_indices)]
                )
                cache.insert_rows(
                    layer_id, "swa_kv", swa_slots, packed_swa[list(row_indices)]
                )
        else:
            swa_full = cp_ctx.all_gather_restore(kv_latent)
            packed_swa = ops.encode_kv(swa_full, pool_tag="swa_kv")
            for rid, row_indices in _request_rows(cp_ctx.plan.full_request_ids).items():
                full_positions = cp_ctx.plan.full_positions[list(row_indices)]
                swa_slots = cache.slots_at_positions(
                    layer_id, "swa_kv", rid, full_positions
                )
                cache.insert_rows(
                    layer_id, "swa_kv", swa_slots, packed_swa[list(row_indices)]
                )

        # Global KV / indexer K / CSA state production. Under CP8 the
        # compressor values/scores are all-gathered + restored to original
        # order before the pair pooling (the reference "all-gather kv/score
        # before S-pool"), so the production below runs identically on every
        # rank over the full request.
        if layer.is_kv_source:
            produce_values = layer.compressor_wkv(x_pre)
            produce_scores = (
                layer.compressor_wgate(x_pre) if geometry.ratio(layer_id) == 2 else None
            )
            if cp_ctx is not None:
                produce_values = cp_ctx.all_gather_restore(produce_values)
                if produce_scores is not None:
                    produce_scores = cp_ctx.all_gather_restore(produce_scores)
                # The plan's full-order views are CPU tensors; move them to
                # the compute device so the rope table indexing below works.
                produce_request_ids = cp_ctx.plan.full_request_ids.to(
                    produce_values.device
                )
                produce_positions = cp_ctx.plan.full_positions.to(produce_values.device)
            else:
                produce_request_ids = request_ids
                produce_positions = positions
            _produce_global(
                ops=ops,
                layer=layer,
                geometry=geometry,
                bundle=bundle,
                values=produce_values,
                scores=produce_scores,
                positions=produce_positions,
                request_ids=produce_request_ids,
                cache=cache,
                slots=slots,
                layer_id=layer_id,
                append_starts=append_starts,
            )

        # Indexer top-k: sources compute, consumers reuse (no copy); SWA-only
        # layers (ratio 0) have no global entries to select.
        if layer.is_index_source:
            selected = _indexer_topk(
                ops=ops,
                layer=layer,
                geometry=geometry,
                bundle=bundle,
                x_pre=x_pre,
                qr=qr,
                positions=positions,
                context=context,
                cache=cache,
                slots=slots,
                layer_id=layer_id,
            )
            shared_topk[layer_id] = selected
        elif geometry.ratio(layer_id):
            source = geometry.index_source_for(layer_id)
            if source is None or source not in shared_topk:
                raise RuntimeError(
                    f"layer {layer_id} has ratio {layer.ratio} but no index "
                    "source result to reuse; the config is inconsistent"
                )
            selected = shared_topk[source]
        else:
            selected = None

        # Sparse attention over global top-k entries + the SWA window.
        attn_out = _layer_attention(
            q,
            selected,
            ops=ops,
            cache=cache,
            layer_id=layer_id,
            geometry=geometry,
            context=context,
            attn_sink=layer.attn_sink,
            window=geometry.window,
            swa_starts=swa_starts if layer_id >= decoder_start else None,
        )  # [T, n_heads, head_dim]

        # Output projection: inverse RoPE tail, grouped wo_a, wo_b.
        attn_out = _rope_tail(attn_out, freqs[rope_positions], inverse=True)
        grouped = attn_out.reshape(tokens, geometry.o_groups, -1)
        group_rank = geometry.o_lora_rank
        wo_a_weight = layer.wo_a.weight
        projected = torch.stack(
            [
                F.linear(
                    grouped[:, group].contiguous(),
                    wo_a_weight[
                        group * group_rank : (group + 1) * group_rank
                    ].contiguous(),
                )
                for group in range(geometry.o_groups)
            ],
            dim=1,
        )
        attn_branch = layer.wo_b(projected.reshape(tokens, -1))

        transition = (
            ops.try_mhc_ffn_transition(
                attn_branch,
                residual,
                step_attn,
                layer.mhc_parameters("ffn"),
                layer.ffn_norm,
                geometry.norm_eps,
                phase=(
                    "prefill"
                    if not capture_layer_ids
                    and not policy.ced
                    and not policy.bounded_replay
                    else "prefill_replay"
                ),
            )
            if hasattr(ops, "try_mhc_ffn_transition")
            else None
        )
        if transition is None:
            step_attn_post = ops.mhc(
                residual, attn_branch, phase="post", state=step_attn
            )
            residual = step_attn_post.hidden
            step_ffn = ops.mhc(
                residual,
                None,
                phase="pre",
                parameters=layer.mhc_parameters("ffn"),
                state=prev_mix,
            )
            x_ffn = _rms_norm(step_ffn.hidden, layer.ffn_norm, geometry.norm_eps)
        else:
            residual = transition.residual
            step_ffn = transition
            x_ffn = transition.normalized
        prev_mix = step_ffn

        ids, route_weights = sqrtsoftplus_gate(
            x_ffn,
            layer.router_weight,
            layer.router_bias,
            top_k=geometry.n_activated_experts,
            routed_scaling_factor=geometry.routed_scaling_factor,
        )
        ids_p, weights_p = pad_topk_for_dispatch(ids, route_weights)
        if cp_ctx is not None:
            # Zigzag padding rows never enter the expert dispatch (plan 3.3):
            # their route ids are invalidated and their weights zeroed, so
            # they contribute no expert load and no combine traffic.
            real = cp_ctx.plan.local_is_real.to(ids_p.device)
            ids_p = torch.where(real[:, None], ids_p, torch.full_like(ids_p, -1))
            weights_p = torch.where(
                real[:, None], weights_p, torch.zeros_like(weights_p)
            )
            if cp_ctx.simulate_moe:
                # CPU simulation: EP8 dispatch/combine through the CP group.
                routed = cp_ctx.moe(x_ffn, ids_p, weights_p, layer_id)
            else:
                routed = ops.moe(x_ffn, ids_p, weights_p, layer_id=layer_id)
        else:
            routed = ops.moe(x_ffn, ids_p, weights_p, layer_id=layer_id)
        if layer.shared_w13 is not None:
            shared = shared_expert_forward(
                x_ffn, layer.shared_w13, layer.shared_w2, geometry.swiglu_limit
            )
            ffn_branch = combine_routed_and_shared(routed, shared)
        else:
            ffn_branch = routed

        step_ffn_post = ops.mhc(residual, ffn_branch, phase="post", state=step_ffn)
        h = step_ffn_post.hidden

    # Final delayed-mHC head + final norm. Under CP8 the rank-local rows
    # (padding included) are all-gathered and restored to LOGICAL token
    # order here; the returned hidden/logits are the global sequence,
    # identical on every rank.
    step_head = ops.mhc(h, None, phase="head", state=prev_mix)
    hidden = _rms_norm(step_head.hidden, bundle.final_norm, geometry.norm_eps)
    if cp_ctx is not None:
        hidden = cp_ctx.all_gather_restore(hidden)
    aux = None
    if capture_layer_ids:
        aux = torch.cat([captured[layer] for layer in capture_layer_ids], dim=-1)
        if cp_ctx is not None:
            aux = cp_ctx.all_gather_restore(aux)
    if compacted:
        hidden = ced.restore_global(hidden)
    logits = None
    if apply_lm_head:
        if bundle.lm_head is None:
            raise ValueError("apply_lm_head requires the lm_head weight")
        logits = F.linear(hidden, bundle.lm_head).float()
    info = (
        dict(ced.info, compacted=compacted)
        if ced is not None
        else {
            "compacted": False,
            "bounded_replay": policy.bounded_replay,
            "original_rows": sum(lengths),
            "query_rows": sum(lengths),
            "first_decoder_layer": decoder_start,
        }
    )
    return PrefillResult(
        hidden=hidden,
        logits=logits,
        cache=cache,
        slots=slots,
        ced_info=info,
        mtp_target_hidden_states=aux,
    )
