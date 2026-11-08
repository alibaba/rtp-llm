"""PPU projection, rotary embedding and sparse-cache helpers for shared Prefill."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor, V41StepContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.attention import COMBINED_WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache, PPURequestSlots
from rtp_llm.models_py.modules.dsv41.ppu.weight import V41Geometry


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


def _request_rows(request_ids: torch.Tensor) -> Dict[int, Tuple[int, ...]]:
    """request id -> its row indices, first-appearance order."""
    rows: Dict[int, Tuple[int, ...]] = {}
    for row, rid in enumerate(request_ids.tolist()):
        rows.setdefault(int(rid), []).append(row)
    return {rid: tuple(index) for rid, index in rows.items()}


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
    freqs: torch.Tensor,
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


def _quantized_indexer_query(query: torch.Tensor) -> V41QuantizedTensor:
    """Apply the post-RoPE group-32 FP4/UE8M0 query boundary before scoring."""
    payload, scales = _quant.quantize_indexer_q(query.contiguous())
    return V41QuantizedTensor(payload, scales, "fp4_ue8m0")
