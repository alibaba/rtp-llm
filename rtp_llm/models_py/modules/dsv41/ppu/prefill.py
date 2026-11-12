"""PPU projection, rotary embedding and sparse-cache helpers for shared Prefill."""

from __future__ import annotations

import os
from typing import Callable, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.backend import V41QuantizedTensor, V41StepContext
from rtp_llm.models_py.modules.dsv41.ppu import _quant
from rtp_llm.models_py.modules.dsv41.ppu.attention import COMBINED_WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    CP8_WORKSPACE_DEQUANT_ENV,
    SharedNativeCache,
    _LocalSwaStageRanges,
    _source_tensor_key,
)
from rtp_llm.models_py.modules.dsv41.ppu.packed_attention import compact_prefix
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
    if os.environ.get("RTP_V41_PPU_FUSED_BF16_ROPE", "0") == "1":
        from rtp_llm.models_py.modules.dsv41.ppu.fused_ops import try_rope_tail

        out = try_rope_tail(x, freqs, inverse)
        if out is not None:
            return out
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
            packed = (
                cache.read_global(source, rid, global_entries=count)
                if isinstance(cache, SharedNativeCache)
                else cache.read_source(source, rid, global_entries=count)[0]
            )
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


LOCAL_PREFILL_WORKSPACE_FUSED_ENV = "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE_FUSED"


def _local_prefill_workspace_fusion_supported(cache, geometry, context, cp):
    """Admit existing row kernels without enabling CP placement or writers."""
    positions, requests = context.positions, context.request_ids
    chunks = getattr(cp, "chunk_lengths_per_req", None)
    return (
        os.environ.get(LOCAL_PREFILL_WORKSPACE_FUSED_ENV, "0") == "1"
        and os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE", "0") == "1"
        and isinstance(cache, SharedNativeCache)
        and cache._is_prefill
        and not cache.sharded
        and getattr(cp, "cp_size", 1) == 1
        and not getattr(cp, "kv_cache_sharded", False)
        and geometry.head_dim == _quant.KV_DIM
        and positions.is_cuda
        and positions.ndim == 1
        and positions.dtype == torch.int64
        and requests.shape == positions.shape
        and requests.device == positions.device
        and chunks is not None
        and len(chunks) == len(cache.lengths)
        and sum(chunks) == positions.numel()
        and _quant.is_ppu()
    )


def _local_prefill_workspace_bounds(cache, context, cp, *, full_layout=False):
    """Use only the adapter's full token or the bound CP1 replay layout.

    Replay starts are fresh-relative; CP1 pads an odd retained count with
    the request's last real position, so padding does not extend its bounds.
    Exact selected layouts carry no replay starts and retain the GPU oracle.
    """
    chunks = getattr(cp, "chunk_lengths_per_req", None)
    prefixes = getattr(cp, "prefix_lengths_host", None)
    positions = context.positions
    if (
        chunks is None or prefixes is None
        or len(chunks) != len(prefixes) or len(chunks) != len(cache.lengths)
        or any(type(n) is not int or n < 0 for n in (*chunks, *prefixes))
        or positions.ndim != 1 or positions.dtype != torch.int64
        or not positions.is_contiguous() or positions.numel() != sum(chunks)
        or context.request_ids.shape != positions.shape
    ):
        return None
    if not hasattr(cp, "cp_size"):
        # Only shared["local_prefill_workspace_layout"] proves these chunks
        # came from state.rows(None), which emits prefix + arange(fresh).
        if not full_layout or tuple(cache.lengths) != tuple(p + n for p, n in zip(prefixes, chunks)):
            return None
        return [(rid, p, p + n - 1) for rid, (p, n) in enumerate(zip(prefixes, chunks)) if n]

    lengths = getattr(cp, "input_lengths_global_host", None)
    starts = getattr(cp, "swa_replay_starts_host", None)
    if starts is None and len(chunks) == 1 and getattr(cp, "swa_replay_start", None) is not None:
        starts = (cp.swa_replay_start,)
    source = getattr(cp, "global_positions", None)
    selected = getattr(cp, "gather_restore_positions", None)
    if (
        cp.cp_size != 1 or getattr(cp, "cp_rank", None) != 0 or getattr(cp, "kv_cache_sharded", False)
        or lengths is None or starts is None
        or len(lengths) != len(chunks) or len(starts) != len(chunks)
        or any(type(n) is not int or n < 0 for n in lengths)
        or any(type(s) is not int or not 0 <= s <= n for n, s in zip(lengths, starts))
        or tuple(cache.lengths) != tuple(p + n for p, n in zip(prefixes, lengths))
        or tuple(chunks) != tuple((n - s + 1) // 2 * 2 for n, s in zip(lengths, starts))
        or getattr(cp, "seq_len_full", None) != sum(lengths)
        or getattr(cp, "chunk_length", None) != positions.numel()
        or getattr(cp, "padded_seq_len", None) != positions.numel()
        or not isinstance(source, torch.Tensor) or source.shape != positions.shape
        or source.device != positions.device or source.dtype != positions.dtype
        or not source.is_contiguous() or source.data_ptr() != positions.data_ptr()
        or context.valid_mask is not getattr(cp, "local_is_real", None)
        or not isinstance(selected, torch.Tensor) or selected.ndim != 1
        or selected.dtype != torch.int64 or selected.device != positions.device
        or not selected.is_contiguous()
        or selected.numel() != sum(n - s for n, s in zip(lengths, starts))
    ):
        return None
    return [(rid, p + s, p + n - 1)
            for rid, (p, n, s) in enumerate(zip(prefixes, lengths, starts)) if n > s]


class _CPPrefillWorkspace:
    """One CP query layout and immutable global bank within a forward."""

    def __init__(self, ops, cache, geometry, context, cp, source, ratio, window, starts, *, full_local_layout=False):
        positions = context.positions
        cp8 = isinstance(cache, SharedNativeCache) and cache.cp8_row_prep_enabled()
        local = _local_prefill_workspace_fusion_supported(cache, geometry, context, cp)
        self.fused_dequant = local or (cp8 and os.environ.get(CP8_WORKSPACE_DEQUANT_ENV, "1") == "1")
        self.fused_indices = local or (cp8 and os.environ.get("RTP_V41_PPU_CP8_FUSED_WORKSPACE_INDICES", "0") == "1")
        bounds = _full_cp_workspace_bounds(cp, positions) if cp8 and self.fused_indices else None
        if bounds is None and local:
            bounds = _local_prefill_workspace_bounds(cache, context, cp, full_layout=full_local_layout)
        if bounds is None:
            chunks = positions.split(cp.chunk_lengths_per_req)
            active = [rid for rid, chunk in enumerate(chunks) if chunk.numel()]
            limits = torch.stack([
                torch.stack((chunks[rid].min(), chunks[rid].max())) for rid in active
            ]).tolist() if active else []
            bounds = [(rid, first, last) for rid, (first, last) in zip(active, limits)]

        layout = [(0, 0, 0, 0)] * len(cp.chunk_lengths_per_req)
        self.swa_ranges = []
        self.global_ranges = []
        offset = 0
        for rid, first, last in bounds:
            count = cache.request_length(rid) // ratio if ratio else 0
            swa_first = max(0, starts.get(rid, 0), first - window + 1)
            swa_count = max(0, last + 1 - swa_first)
            swa_offset = offset + count
            layout[rid] = (offset, count, swa_offset, swa_first)
            if count:
                self.global_ranges.append((rid, offset, count))
            swa_rows = None if self.fused_indices else torch.arange(
                swa_first, swa_first + swa_count, device=positions.device,
            )
            self.swa_ranges.append((rid, swa_offset, swa_first, swa_count, swa_rows))
            offset = swa_offset + swa_count

        self.kv = torch.empty((offset, geometry.head_dim), dtype=torch.bfloat16, device=positions.device)
        self.kv8 = None
        plans = getattr(cache, "prefill_plans", None)
        if plans is None:
            meta = torch.tensor(layout, dtype=torch.int64, device=positions.device).reshape(-1, 4)
        else:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

            key = ("workspace_layout", tuple(layout), positions.device)
            meta = plans.get(key)
            if meta is None:
                meta = _upload(layout, positions.device, torch.int64).reshape(-1, 4)
                plans[key] = meta
        self.row_meta = meta.index_select(0, context.request_ids.long())
        self.selection = None
        self.index_plan = None
        if source is not None:
            cache.prepare_source(source)
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            descriptor = cache.descriptors(source)[tag]
            for rid, begin, count in self.global_ranges:
                packed = (
                    cache.read_global(source, rid, global_entries=count)
                    if isinstance(cache, SharedNativeCache)
                    else cache.read_source(source, rid, global_entries=count)[0]
                )
                out = self.kv[begin:begin + count]
                if not (self.fused_dequant and _quant.try_dequantize_prefill_rows(packed, out, global_pool=True)):
                    ops.decode_kv(packed, pool_tag=tag, descriptor=descriptor, out=out)

    def indices(self, positions, selected, ratio, window):
        if self.index_plan is None or self.selection is not selected:
            if self.fused_indices:
                from rtp_llm.models_py.modules.dsv41.ppu.graph_cache_kernels import try_assemble_cp_prefill_indices

                result = try_assemble_cp_prefill_indices(
                    self.row_meta, positions, selected, ratio, window, COMBINED_WIDTH,
                )
                if result is not None:
                    self.index_plan, self.selection = result, selected
                    return result
            gbase, count, sbase, swa_first = self.row_meta.unbind(dim=1)
            swpos = positions[:, None] - window + 1 + torch.arange(window, device=positions.device)
            swa = torch.where(swpos >= swa_first[:, None], sbase[:, None] + swpos - swa_first[:, None], -1)
            if selected is not None:
                columns = torch.arange(selected.shape[1], device=positions.device)
                live = (columns < ((positions + 1) // ratio)[:, None])
                live = live & (selected >= 0) & (selected < count[:, None])
                global_indices = torch.where(live, gbase[:, None] + selected, -1)
                swa = torch.cat((global_indices, swa), dim=1)
            indices, lengths = compact_prefix(swa.int())
            self.index_plan = F.pad(indices, (0, COMBINED_WIDTH - indices.shape[-1]), value=-1), lengths
            self.selection = selected
        return self.index_plan

    def refresh_swa(self, ops, cache, layer_id):
        descriptor = cache.descriptors(layer_id)["swa_kv"]
        for index, (rid, begin, first, count, positions) in enumerate(self.swa_ranges):
            out = self.kv[begin:begin + count]
            if self.fused_dequant and cache.try_decode_swa_range(layer_id, rid, first, out):
                continue
            if positions is None:
                positions = torch.arange(first, first + count, device=out.device)
                self.swa_ranges[index] = (rid, begin, first, count, positions)
            packed = cache.read_swa(layer_id, rid, positions)
            if not (self.fused_dequant and _quant.try_dequantize_prefill_rows(packed, out, global_pool=False)):
                ops.decode_kv(packed, pool_tag="swa_kv", descriptor=descriptor, out=out)

    def q8_kv(self):
        # The global bank stays immutable until the workspace changes source.
        # Update just the uniformly-strided SWA stripes after the first layer.
        requests = len(self.swa_ranges)
        if not requests or self.kv.shape[0] % requests:
            return self.kv
        stride = self.kv.shape[0] // requests
        count = self.swa_ranges[0][3]
        global_count = stride - count
        if any(
            begin != rid * stride + global_count or length != count
            for rid, begin, _, length, _ in self.swa_ranges
        ):
            return self.kv
        if self.kv8 is None:
            self.kv8 = self.kv.clamp(-448, 448).to(torch.float8_e4m3fn)
        else:
            width = self.kv.shape[1]
            swa = self.kv.view(requests, stride, width)[:, global_count:]
            self.kv8.view(requests, stride, width)[:, global_count:].copy_(swa.clamp(-448, 448))
        return self.kv8


def _full_cp_workspace_bounds(cp, positions):
    """Reuse the full zigzag split's host geometry to size its SWA intervals."""
    chunks = cp.chunk_lengths_per_req
    lengths = cp.input_lengths_global_host
    prefixes = cp.prefix_lengths_host
    if (
        cp.gather_restore_positions is not None
        or lengths is None or prefixes is None
        or len(chunks) != len(lengths) or len(chunks) != len(prefixes)
        or any(chunk % 2 for chunk in chunks)
        or not positions.is_contiguous()
        or positions.numel() != cp.global_positions.numel()
        or positions.data_ptr() != cp.global_positions.data_ptr()
    ):
        return None
    bounds = []
    for rid, (chunk, length, prefix) in enumerate(zip(chunks, lengths, prefixes)):
        if chunk:
            first = cp.cp_rank * (chunk // 2)
            last = (2 * cp.cp_size - cp.cp_rank) * (chunk // 2) - 1
            last_real = max(0, length - 1)
            bounds.append((rid, prefix + min(first, last_real), prefix + min(last, last_real)))
    return bounds


def build_cp_prefill_workspace(
    *, ops, cache, layer_id, geometry, context, selected_blocks, window, swa_starts, cp, shared,
    q8=False,
):
    """Reuse global rows and indices, refreshing only this layer's SWA rows."""
    ratio = geometry.ratio(layer_id)
    source = geometry.kv_source_for(layer_id) if ratio else None
    starts = swa_starts or {}
    layout = (source, ratio, window, starts)
    cached = shared.get("prefill_kv_workspace")
    if cached is None or cached[0] is not cp or cached[1] != layout:
        workspace = _CPPrefillWorkspace(
            ops, cache, geometry, context, cp, source, ratio, window, starts,
            full_local_layout=shared.get("local_prefill_workspace_layout") is cp,
        )
        shared["prefill_kv_workspace"] = (cp, layout, workspace)
    else:
        workspace = cached[2]
    workspace.refresh_swa(ops, cache, layer_id)
    indices, lengths = workspace.indices(context.positions, selected_blocks, ratio, window)
    kv = workspace.q8_kv() if q8 else workspace.kv
    return kv.unsqueeze(1), indices, lengths


def _compress_pairs(
    values: torch.Tensor, scores: torch.Tensor, norm, eps: float
) -> torch.Tensor:
    """Pair pooling: softmax over the pair per channel, then RMSNorm."""
    weights = torch.softmax(scores.float(), dim=-2)
    # The compressor emits BF16 pooled values before RMSNorm. Its C2
    # projections and unfinished-pair state retain FP32 separately.
    latent = (values.float() * weights).sum(-2).to(torch.bfloat16)
    return _rms_norm(latent, norm, eps).to(torch.bfloat16)


def _global_write_ranges(positions, request_ids, append_starts, prefill_layout):
    """Return each request's input range, absolute bounds and append rows."""
    if prefill_layout is not None:
        prefixes, lengths = prefill_layout
        if len(prefixes) != len(lengths) or sum(lengths) != positions.numel():
            raise ValueError("global KV Prefill layout does not match the restored rows")
        offset = 0
        for rid, (start, length) in enumerate(zip(prefixes, lengths)):
            if length:
                end = start + length
                append = append_starts.get(rid, start)
                rows = slice(offset, offset + length)
                state_rows = slice(offset + max(0, append - start), offset + length)
                yield rid, start, end, append, rows, state_rows
            offset += length
        return

    # Eager decode and compacted layouts do not have the full Prefill ranges.
    host_positions = positions.tolist()
    for rid, indices in _request_rows(request_ids).items():
        start = host_positions[indices[0]]
        end = max(host_positions[row] for row in indices) + 1
        append = append_starts.get(rid, start)
        state_rows = [row for row in indices if host_positions[row] >= append]
        yield rid, start, end, append, list(indices), state_rows


_PRODUCER_TILE_ENTRIES = 8192
_DIRECT_PRODUCER_WRITE_ENV = "RTP_V41_PPU_CP8_DIRECT_PRODUCER_WRITE"
LOCAL_PREFILL_ROTARY_PACK_ENV = "RTP_V41_PPU_LOCAL_PREFILL_ROTARY_PACK"


class _LocalProducerRotaryProof(NamedTuple):
    """Read-only row provenance, independent of the preceding SWA payload."""
    cache: object
    layer_id: int
    positions: torch.Tensor
    request_ids: torch.Tensor
    input_keys: tuple
    prefixes: tuple
    lengths: tuple
    ranges: tuple
    guard: Callable[[], bool]


def _valid_local_producer_rotary_proof(
    proof, *, cache, layer_id, prefill_layout, positions=None, request_ids=None,
):
    if (
        os.environ.get(LOCAL_PREFILL_ROTARY_PACK_ENV, "0") != "1"
        or not isinstance(proof, _LocalProducerRotaryProof)
        or type(cache) is not SharedNativeCache or proof.cache is not cache
        or not cache._is_prefill or cache.sharded or proof.layer_id != layer_id
        or prefill_layout is None or len(prefill_layout) != 2 or not callable(proof.guard)
    ):
        return False
    prefixes, lengths = tuple(prefill_layout[0]), tuple(prefill_layout[1])
    if (prefixes != proof.prefixes or lengths != proof.lengths
            or len(prefixes) != len(lengths) or len(prefixes) != len(cache.lengths)
            or (positions is not None and positions is not proof.positions)
            or (request_ids is not None and request_ids is not proof.request_ids)):
        return False
    offset, ranges, totals = 0, [], []
    for prefix, length in zip(prefixes, lengths):
        if (type(prefix) is not int or type(length) is not int
                or prefix < 0 or length < 0 or prefix + length > 2**63 - 1):
            return False
        ranges.append((offset, prefix, length))
        totals.append(prefix + length)
        offset += length
    if tuple(totals) != cache.lengths or tuple(ranges) != proof.ranges:
        return False
    owners = proof.positions, proof.request_ids
    if not all(
        isinstance(value, torch.Tensor) and value.layout == torch.strided
        and not value.is_neg() and not value.is_conj()
        and value.dtype == torch.int64 and value.shape == (offset,) and value.is_contiguous()
        and value.device == torch.device(cache.device)
        for value in owners
    ):
        return False
    # Keep the original creation keys, including normal Tensor versions.
    # Inference metadata follows A08's read-only-within-forward contract.
    return (tuple(_source_tensor_key(value) for value in owners) == proof.input_keys
            and proof.guard())


def _make_local_producer_rotary_proof(ticket, *, cache, layer_id, prefixes, lengths):
    if (
        os.environ.get(LOCAL_PREFILL_ROTARY_PACK_ENV, "0") != "1"
        or type(cache) is not SharedNativeCache
        or not isinstance(ticket, _LocalSwaStageRanges) or ticket.cache is not cache
        or not ticket.retain_prefix_tail or ticket.bounded_starts is not None
        or len(ticket.inputs) != 3 or len(ticket.input_keys) != 3
        or tuple(prefixes) != ticket.prefixes or ticket.totals != cache.lengths
    ):
        return None
    proof = _LocalProducerRotaryProof(
        cache, layer_id, ticket.inputs[0], ticket.inputs[1], ticket.input_keys[:2],
        tuple(prefixes), tuple(lengths), ticket.ranges, ticket.guard,
    )
    # No payload/pool key is read or rebaselined: stage_swa may legitimately
    # change its own pool before this metadata capability is consumed.
    return proof if _valid_local_producer_rotary_proof(
        proof, cache=cache, layer_id=layer_id, prefill_layout=(prefixes, lengths),
    ) else None


def _producer_tiles(requests, ratio):
    """Group adjacent whole requests without crossing a compression pair."""
    tile, count = [], 0
    for request in requests:
        _, start, end, _, _, _ = request
        entries = (end - start) // ratio
        aligned = start % ratio == 0 and end % ratio == 0
        if tile and (not aligned or count + entries > _PRODUCER_TILE_ENTRIES):
            yield tile
            tile, count = [], 0
        if not aligned or entries > _PRODUCER_TILE_ENTRIES:
            yield [request]
        else:
            tile.append(request)
            count += entries
    if tile:
        yield tile


def _tiled_producer_inputs(
    requests, ops, layer, eps, freqs, values, scores, positions, ratio, tag
):
    pack_rotary = getattr(ops, "try_encode_rotary_kv", None)
    for tile in _producer_tiles(requests, ratio):
        if len(tile) == 1:
            # Odd prefixes, unpaired tails, and oversized requests use the
            # original producer, including its CSA carry and error checks.
            yield tile[0], None
            continue

        with record_function_range("dsv41.ppu.prefill.compressor.tile"):
            begin, end = tile[0][4].start, tile[-1][4].stop
            tile_values = values[begin:end]
            if ratio == 1:
                latent = _rms_norm(tile_values, layer.compressor_norm, eps).to(torch.bfloat16)
            else:
                tile_scores = scores[begin:end]
                latent = _compress_pairs(
                    tile_values.reshape(-1, 2, tile_values.shape[-1]),
                    tile_scores.reshape(-1, 2, tile_scores.shape[-1]),
                    layer.compressor_norm,
                    eps,
                )
            entry_positions = positions[begin:end:ratio]
            rotation = None
            packed_global = (
                pack_rotary(latent, freqs, entry_positions, global_pool=True)
                if pack_rotary is not None else None
            )
            if packed_global is None:
                rotation = freqs[entry_positions.clamp_max(freqs.shape[0] - 1)]
                packed_global = ops.encode_kv(_rope_tail(latent, rotation), pool_tag=tag)
            counts = tuple((request[2] - request[1]) // ratio for request in tile)
            # Keep the GEMM shape of each request. Tile the row-local norm,
            # rotary embedding and codec around that BF16 rounding boundary.
            weight = layer.index_wk.weight
            if isinstance(layer.index_wk, PPUPTPCLinear):
                if (
                    os.environ.get(_DIRECT_PRODUCER_WRITE_ENV, "0") == "1"
                    and not torch.is_grad_enabled()
                ):
                    projected = latent.new_empty((latent.shape[0], weight.shape[0]))
                    for rows, out in zip(latent.split(counts), projected.split(counts)):
                        layer.index_wk.forward_into(rows, out)
                else:
                    projected = torch.cat([layer.index_wk(rows) for rows in latent.split(counts)])
            elif (
                os.environ.get(_DIRECT_PRODUCER_WRITE_ENV, "0") == "1"
                and not torch.is_grad_enabled()
                and weight.is_contiguous()
                and weight.dtype == latent.dtype
            ):
                projected = latent.new_empty((latent.shape[0], weight.shape[0]))
                for rows, out in zip(latent.split(counts), projected.split(counts)):
                    torch.mm(rows, weight.T, out=out)
            else:
                projected = torch.cat([F.linear(rows, weight) for rows in latent.split(counts)])
            projected = _rms_norm(projected, layer.index_k_norm, eps)
            packed_index = (
                pack_rotary(projected, freqs, entry_positions, global_pool=False)
                if pack_rotary is not None else None
            )
            if packed_index is None:
                if rotation is None:
                    rotation = freqs[entry_positions.clamp_max(freqs.shape[0] - 1)]
                packed_index = ops.encode_kv(_rope_tail(projected, rotation), pool_tag="indexer_kv")
            del latent, projected, rotation
        yield from zip(
            tile,
            zip(entry_positions.split(counts), packed_global.split(counts), packed_index.split(counts)),
        )


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
    cache: SharedNativeCache,
    layer_id: int,
    append_starts: Optional[dict] = None,
    prefill_layout=None,
    local_rotary_proof=None,
) -> None:
    """Produce + write global KV, indexer K and CSA states for one layer.

    ``values`` / ``scores`` are the compressor projections of the batch's
    rows in the SAME order as ``positions`` / ``request_ids`` — rank-local
    under CP1, original-order (restored) under CP8 (the caller gathered
    them; the pooling below then runs identically on every rank).

    Ratio 2 pairs are aligned to absolute-position parity (pair = even|odd
    absolute positions, rope position = the pair's even start); a chunk that
    begins at an odd position reads its left token from ``csa_state``.
    Full Prefill supplies host (prefixes, lengths) in restored request order.
    """
    ratio = geometry.ratio(layer_id)
    tag = "global_kv_2" if ratio == 2 else "global_kv_1"
    pack_rotary = (
        getattr(ops, "try_encode_rotary_kv", None)
        if ratio in (1, 2) and _valid_local_producer_rotary_proof(
            local_rotary_proof, cache=cache, layer_id=layer_id, prefill_layout=prefill_layout,
            positions=positions, request_ids=request_ids,
        ) else None
    )
    request_local = (
        prefill_layout is not None
        and isinstance(cache, SharedNativeCache)
        and cache.cp8_row_prep_enabled()
    )
    local_store = (
        os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_WRITE", "0") == "1"
        and getattr(cache, "_local_source_native_write_enabled", False)
        and _valid_local_producer_rotary_proof(
            local_rotary_proof, cache=cache, layer_id=layer_id, prefill_layout=prefill_layout,
            positions=positions, request_ids=request_ids,
        )
    )
    ranges = _global_write_ranges(positions, request_ids, append_starts or {}, prefill_layout)
    prepared = (
        _tiled_producer_inputs(
            ranges, ops, layer, geometry.norm_eps, freqs, values, scores, positions, ratio, tag,
        )
        if request_local and os.environ.get("RTP_V41_PPU_CP8_TILED_COMPRESSOR", "0") == "1"
        else ((request, None) for request in ranges)
    )
    for (rid, start, end, append, rows, state_rows), packed in prepared:
        handle = None if request_local or local_store else cache.slots[rid]
        begin, stop = start // ratio, end // ratio
        if packed is not None:
            entry_positions, packed_global, packed_index = packed
        elif ratio == 1:
            latent = _rms_norm(
                values[rows], layer.compressor_norm, geometry.norm_eps
            ).to(torch.bfloat16)
            entry_positions = positions[rows]
        elif begin == stop:
            latent = values.new_zeros((0, geometry.head_dim), dtype=torch.bfloat16)
            entry_positions = positions.new_zeros(0)
        else:
            request_values, request_scores = values[rows], scores[rows]
            odd_start = start % 2
            left = slice(odd_start, end - start - 1, 2)
            right = slice(1 - odd_start, end - start, 2)
            left_values, left_scores = request_values[left], request_scores[left]
            if odd_start:
                left_pos = start - 1
                state = cache.read_csa(
                    layer_id, rid, torch.tensor([left_pos], dtype=torch.int64)
                )
                if not bool(state.any()):
                    raise RuntimeError(
                        f"ratio-2 chunk of request {rid} starts at odd "
                        f"position {start} without a prior csa_state entry "
                        f"at {left_pos}; the compression phase is missing"
                    )
                previous_values, previous_scores = state.split(geometry.head_dim, dim=-1)
                left_values = torch.cat((previous_values.to(values.dtype), left_values))
                left_scores = torch.cat((previous_scores.to(scores.dtype), left_scores))
            latent = _compress_pairs(
                torch.stack((left_values, request_values[right]), dim=1),
                torch.stack((left_scores, request_scores[right]), dim=1),
                layer.compressor_norm,
                geometry.norm_eps,
            )
            entry_positions = torch.arange(
                begin * ratio, stop * ratio, ratio, device=positions.device
            )
        if packed is None:
            entry_count = stop - begin
            if latent.shape[0] != entry_count:
                raise RuntimeError(
                    f"layer {layer_id} produced {latent.shape[0]} entries for "
                    f"request {rid}, expected {entry_count} (start={start}, end={end})"
                )
            # Ratio1 is a private slice of the proof's creation-stamped rows;
            # ratio2 above creates its own arange from the same host bounds.
            # Both are nonnegative without a new device-value reduction.
            safe_positions = None
            packed_global = (
                pack_rotary(latent, freqs, entry_positions, global_pool=True)
                if pack_rotary is not None else None
            )
            if packed_global is None:
                safe_positions = entry_positions.clamp_max(freqs.shape[0] - 1)
                global_keys = _rope_tail(latent, freqs[safe_positions])
                packed_global = ops.encode_kv(global_keys, pool_tag=tag)
        first_write = max(begin, append // ratio)
        write_offset = first_write - begin
        write_positions = entry_positions[write_offset:]
        if local_store and cache.try_insert_local_source_rows(
            layer_id, tag, rid, first_write * ratio, packed_global[write_offset:],
            proof=local_rotary_proof, prefill_layout=prefill_layout,
        ):
            pass
        elif request_local:
            cache.insert_request_rows(
                layer_id, tag, rid, write_positions, packed_global[write_offset:],
                start=first_write * ratio,
            )
        else:
            if handle is None:
                handle = cache.slots[rid]
            cache.insert_rows(
                layer_id,
                tag,
                handle.global_slots[layer_id][first_write:stop],
                packed_global[write_offset:],
            )
        if packed is None:
            # Indexer K comes from the same latents.
            projected = (
                layer.index_wk(latent) if isinstance(layer.index_wk, PPUPTPCLinear)
                else F.linear(latent, layer.index_wk.weight)
            )
            projected = _rms_norm(projected, layer.index_k_norm, geometry.norm_eps)
            packed_index = (
                pack_rotary(projected, freqs, entry_positions, global_pool=False)
                if pack_rotary is not None else None
            )
            if packed_index is None:
                if safe_positions is None:
                    safe_positions = entry_positions.clamp_max(freqs.shape[0] - 1)
                index_keys = _rope_tail(projected, freqs[safe_positions])
                packed_index = ops.encode_kv(index_keys, pool_tag="indexer_kv")
        if local_store and cache.try_insert_local_source_rows(
            layer_id, "indexer_kv", rid, first_write * ratio, packed_index[write_offset:],
            proof=local_rotary_proof, prefill_layout=prefill_layout,
        ):
            pass
        elif request_local:
            cache.insert_request_rows(
                layer_id, "indexer_kv", rid, write_positions, packed_index[write_offset:],
                start=first_write * ratio,
            )
        else:
            if handle is None:
                handle = cache.slots[rid]
            cache.insert_rows(
                layer_id,
                "indexer_kv",
                handle.indexer_slots[layer_id][first_write:stop],
                packed_index[write_offset:],
            )
        if ratio == 2:
            # CSA states at RAW positions: [values | scores] fp32.
            state_positions = positions[state_rows]
            if not state_positions.numel():
                continue
            state_values, state_scores = values[state_rows], scores[state_rows]
            state_start = max(start, append)
            if (
                os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_CSA_NATIVE_WRITE", "0") == "1"
                and isinstance(state_rows, slice)
                and isinstance(cache, SharedNativeCache)
                and _valid_local_producer_rotary_proof(
                    local_rotary_proof, cache=cache, layer_id=layer_id, prefill_layout=prefill_layout,
                    positions=positions, request_ids=request_ids,
                )
                and cache.try_insert_local_csa_fields(
                    layer_id, rid, state_start, state_values, state_scores,
                    proof=local_rotary_proof, prefill_layout=prefill_layout,
                )
            ):
                continue
            if (
                request_local
                and isinstance(state_rows, slice)
                and isinstance(cache, SharedNativeCache)
                and os.environ.get(_DIRECT_PRODUCER_WRITE_ENV, "0") == "1"
                and cache.try_insert_csa_fields(layer_id, rid, state_start, state_values, state_scores)
            ):
                continue
            state = torch.cat((state_values, state_scores), dim=-1).float()
            if request_local:
                cache.insert_request_rows(
                    layer_id, "csa_state", rid, state_positions, state, start=state_start
                )
            else:
                state_slots = cache.slots_at_positions(layer_id, "csa_state", rid, state_positions)
                cache.insert_rows(layer_id, "csa_state", state_slots, state)


def _quantized_indexer_query(query: torch.Tensor) -> V41QuantizedTensor:
    """Apply the post-RoPE group-32 FP4/UE8M0 query boundary before scoring."""
    payload, scales = _quant.quantize_indexer_q(query.contiguous())
    return V41QuantizedTensor(payload, scales, "fp4_ue8m0")
