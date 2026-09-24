"""V4.1 decoder prefill compaction with an optional bounded replay domain.

L0--L20 retain the complete fresh input and decoder global KV. L21--L39
only evaluate the dependency windows of allocated SWA checkpoints and the
request end. The original cache geometry and wire format stay unchanged.
In opt-in bounded replay, only the final 128 rows are evaluated, with SWA
truncated to that domain. Those approximate decoder/draft pools are private
live state, excluded from prefix caching by the native cache policy.
Query rows are redistributed within CP; fresh KV is scattered back before
ordinary cache writes. No omitted row may be exposed as a prompt output.

All metadata decisions use the engine's rank-consistent host input contract.
Transport and scratch live for one forward, including the original-layout
indexer projection needed to preserve BF16 head-weight rounding.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.attn_type import DECODER_SWA_KV, SWA_KV
from rtp_llm.models_py.modules.dsv4.cp import CPContext, cp_swa_replay_starts

_CP_SIZE = 4
_TAIL_TOKENS = 3072
_MIN_TOKENS = 32768
_MIN_L20_SPLIT_TOKENS = 65536
_DERIVED_KEYS = (
    "prefill_chunk_meta",
    "prefill_kv_workspace",
    "prefill_index_plan",
    "prefill_candidate_mask",
    "candidate_mask",
    "prefill_sparse_plans",
    "prefill_score_bounds",
    "prefill_meta_common",
)


def permits_ced(inputs) -> bool:
    """Missing/old native contracts conservatively keep all prompt rows."""
    return (
        getattr(inputs, "need_all_logits", None) is False
        and getattr(inputs, "need_all_hidden_states", None) is False
    )


def checkpoint_positions(prefix, length, block_ids, block_span, ring_entries):
    """Return fresh-concat row indices, covering every allocated ring write.

    Physical SWA columns cover block_span raw tokens (including CP scaling).
    Reserve blocks beyond the current end and prefix-only blocks do not write.
    The 19 causal windows contribute 19*127 predecessor rows. Keeping at least
    3072 also bounds padding variations to the previously validated geometry.
    """
    if prefix < 0 or length <= 0 or block_span <= 0 or ring_entries <= 0:
        raise ValueError("invalid CED checkpoint geometry")
    end = prefix + length
    halo = max(_TAIL_TOKENS, 19 * 127 + ring_entries)
    endpoints = [end]
    for column, physical_id in enumerate(block_ids):
        start = column * block_span
        stop = min(start + block_span, end)
        if physical_id > 0 and start < end and stop > prefix:
            endpoints.append(stop)
    intervals = []
    for stop in sorted(set(endpoints)):
        start = max(prefix, stop - halo)
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], stop)
        else:
            intervals.append((start, stop))
    return torch.cat(
        [
            torch.arange(start - prefix, stop - prefix, dtype=torch.int64)
            for start, stop in intervals
        ]
    )


def _supported_model(v4) -> bool:
    args = getattr(v4, "args", None)
    layers = getattr(v4, "layers", ())
    captures = tuple(getattr(v4, "capture_aux_hidden_layer_ids", ()))
    if (
        args is None
        or getattr(args, "v41_config", None) is None
        or not getattr(v4, "fp8_kv_cache", False)
        or getattr(args, "n_layers", 0) != 40
        or len(layers) != 40
        or getattr(args, "ep_size", 0) != _CP_SIZE
        or getattr(args, "window_size", 0) != 128
        or getattr(args, "dim", 0) != 5120
        or getattr(args, "hc_mult", 0) != 4
        or getattr(args, "n_hash_layers", -1) != 0
        or len(set(captures)) != len(captures)
        or any(layer not in (37, 38, 39) for layer in captures)
        # Ordinary MTP has a different full-hidden handoff contract.
        or (not captures and getattr(v4, "_mtp_hidden_buffer", None) is not None)
    ):
        return False
    for index in range(20, 40):
        layer = layers[index]
        attn = getattr(layer, "attn", None)
        if (
            attn is None
            or getattr(attn, "layer_id", -1) != index
            or getattr(attn, "kv_source_layer_id", None) != 20
            or getattr(attn, "index_source_layer_id", None)
            != 20 + (index - 20) // 4 * 4
            or getattr(attn, "compress_ratio", 0) != 1
            or getattr(attn, "window_size", 0) != 128
            or bool(getattr(attn, "is_kv_source", False)) != (index == 20)
            or bool(getattr(attn, "is_index_source", False)) != (index % 4 == 0)
            or getattr(layer, "engram", None) is not None
        ):
            return False
    return True


@dataclass
class _RowExchange:
    """A bijection on real selected rows; padding is never sent back."""

    original_rows: int
    compact_rows: int
    send_indices: torch.Tensor
    receive_positions: torch.Tensor
    send_sizes: Tuple[int, ...]
    receive_sizes: Tuple[int, ...]
    group: Any
    candidate_rows_host: Optional[Tuple[int, ...]] = None
    projection_groups_host: Tuple = ()
    receive_is_identity: bool = False

    def compact(self, value):
        if value.shape[0] != self.original_rows:
            raise ValueError("CED source row count changed")
        send = value.index_select(0, self.send_indices).contiguous()
        received = value.new_empty((sum(self.receive_sizes), *value.shape[1:]))
        torch.distributed.all_to_all_single(
            received,
            send,
            output_split_sizes=list(self.receive_sizes),
            input_split_sizes=list(self.send_sizes),
            group=self.group,
        )
        if self.receive_is_identity:
            return received
        output = value.new_zeros((self.compact_rows, *value.shape[1:]))
        return output.index_copy_(0, self.receive_positions, received)

    def restore(self, value, out=None):
        if value.shape[0] != self.compact_rows:
            raise ValueError("CED compact row count changed")
        send = value.index_select(0, self.receive_positions).contiguous()
        received = value.new_empty((sum(self.send_sizes), *value.shape[1:]))
        torch.distributed.all_to_all_single(
            received,
            send,
            output_split_sizes=list(self.send_sizes),
            input_split_sizes=list(self.receive_sizes),
            group=self.group,
        )
        if out is None:
            out = value.new_empty((self.original_rows, *value.shape[1:]))
        if out.shape != (self.original_rows, *value.shape[1:]):
            raise ValueError("CED restore destination shape changed")
        # received is independent even when value and out alias the aux buffer.
        out.zero_()
        return out.index_copy_(0, self.send_indices, received)


def _query_layout(original, selected, group, *, keep_candidate_rows=False):
    """Derive transport from the actual engine inverse map, not rank0 ownership."""
    device = original.global_positions.device
    cp, rank = original.cp_size, original.cp_rank
    lengths = original.input_lengths_global_host or (original.seq_len_full,)
    prefixes = original.prefix_lengths_host or (original.prefix_length,)
    original_chunks = original.chunk_lengths_per_req or (original.chunk_length,)
    if (
        len(lengths) != len(prefixes)
        or len(lengths) != len(original_chunks)
        or sum(lengths) != original.seq_len_full
        or selected.ndim != 1
        or selected.numel() == 0
        or selected[0] < 0
        or selected[-1] >= original.seq_len_full
        or (selected[1:] <= selected[:-1]).any()
    ):
        raise ValueError("invalid per-request CED selection")
    owners, local, local_positions, absolute_positions, request_ids = [], [], [], [], []
    local_real, chunks = [], []
    original_start, padded_start, local_start = 0, 0, 0
    for request, (length, prefix, original_chunk) in enumerate(
        zip(lengths, prefixes, original_chunks)
    ):
        chosen = selected[
            (selected >= original_start) & (selected < original_start + length)
        ]
        count = chosen.numel()
        if count == 0:
            raise ValueError("CED must preserve the end of every request")
        padded = ((count + 2 * cp - 1) // (2 * cp)) * (2 * cp)
        chunk, half = padded // cp, padded // (2 * cp)
        canonical = torch.arange(count, dtype=torch.int64)
        pair = canonical // half
        owner = torch.where(pair < cp, pair, 2 * cp - 1 - pair)
        row = canonical % half + torch.where(pair < cp, 0, half)
        own = owner == rank
        positions = torch.full((chunk,), length - 1, dtype=torch.int64)
        positions[row[own]] = chosen[own] - original_start
        real = torch.zeros(chunk, dtype=torch.bool)
        real[row[own]] = True
        owners.append(owner)
        local.append(row + local_start)
        local_positions.append(positions + padded_start)
        absolute_positions.append(positions + prefix)
        request_ids.append(torch.full((chunk,), request, dtype=torch.int32))
        local_real.append(real)
        chunks.append(chunk)
        original_start += length
        padded_start += original_chunk * cp
        local_start += chunk
    owners, local = torch.cat(owners), torch.cat(local)
    rows = sum(chunks)
    padded = rows * cp
    info = original.cp_info
    restore = info.prefill_qkv_restore_indice.detach().cpu().long()
    mask = info.prefill_qkv_padding_mask.detach().cpu().bool()
    if restore.shape != mask.shape or int(mask.sum()) != original.seq_len_full:
        raise ValueError("CED requires the engine's complete CP inverse map")
    source_flat = restore[mask].index_select(0, selected)
    source_owners = source_flat // original.chunk_length
    source_rows = source_flat % original.chunk_length
    own = owners == rank
    sends, receives, send_sizes, receive_sizes, indexer_groups = [], [], [], [], []
    projection_groups_host = []
    for peer in range(cp):
        outgoing = (source_owners == rank) & (owners == peer)
        incoming = (source_owners == peer) & own
        sends.append(source_rows[outgoing])
        receives.append(local[incoming])
        send_sizes.append(int(outgoing.sum()))
        receive_sizes.append(int(incoming.sum()))
        if incoming.any():
            indexer_groups.append(
                (local[incoming].to(device), source_rows[incoming].to(device))
            )
            if keep_candidate_rows:
                projection_groups_host.append(
                    (
                        tuple(local[incoming].tolist()),
                        tuple(source_rows[incoming].tolist()),
                    )
                )
    receive_positions_host = torch.cat(receives)
    exchange = _RowExchange(
        original.chunk_length,
        rows,
        torch.cat(sends).to(device),
        receive_positions_host.to(device),
        tuple(send_sizes),
        tuple(receive_sizes),
        group,
        candidate_rows_host=(
            tuple(sorted(source_rows[source_owners == rank].tolist()))
            if keep_candidate_rows
            else None
        ),
        projection_groups_host=tuple(projection_groups_host),
        receive_is_identity=(
            sum(receive_sizes) == rows
            and torch.equal(
                receive_positions_host, torch.arange(rows, dtype=torch.int64)
            )
        ),
    )
    # Only query geometry changes. All write-side lengths and true prefixes
    # remain original; cp.py scatters gathered KV into that original fresh view.
    context = replace(
        original,
        chunk_length=rows,
        padded_seq_len=padded,
        relative_positions=torch.cat(local_positions).to(device),
        global_positions=torch.cat(absolute_positions).to(device),
        local_is_real=torch.cat(local_real).to(device),
        unpad_restore=(owners * rows + local).to(device),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=tuple(chunks),
        req_id_per_token=torch.cat(request_ids).to(device),
        gather_restore_positions=selected.to(device),
    )
    return context, exchange, tuple(indexer_groups)


@dataclass
class CEDPlan:
    """One forward's query domain and reversible CP transport."""

    original_context: CPContext
    context: CPContext
    cu_seqlens: torch.Tensor
    exchange: _RowExchange
    indexer_groups: Tuple
    _capture_ids: Tuple[int, ...]
    _compacted: bool = field(default=False, init=False)
    _aux_restored: bool = field(default=False, init=False)
    _router_groups: Optional[Tuple] = field(default=None, init=False)
    _router_chunk_rows: Optional[int] = field(default=None, init=False)
    _router_gate_groups: Optional[Tuple] = field(default=None, init=False)
    _router_gate_rows: int = field(default=0, init=False)

    @classmethod
    @torch.inference_mode()
    def create(
        cls,
        v4,
        cp_ctx,
        attn_inputs,
        kv_cache,
        *,
        prepare_hidden_fn=None,
        bounded_replay=False,
    ):
        # Host metadata and the request gate are broadcast by the engine to all
        # TP ranks. No extra device->host voting in the prefill critical path.
        if (
            not isinstance(cp_ctx, CPContext)
            or cp_ctx.cp_size != _CP_SIZE
            or not cp_ctx.kv_cache_sharded
            or cp_ctx.seq_len_full < (128 if bounded_replay else _MIN_TOKENS)
            or not cp_ctx.global_positions.is_cuda
            or prepare_hidden_fn is not None
            or kv_cache is None
            or not bool(getattr(attn_inputs, "is_prefill", False))
            or bool(getattr(attn_inputs, "is_target_verify", False))
            or bool(getattr(attn_inputs, "is_cuda_graph", False))
            or not _supported_model(v4)
            or not torch.distributed.is_initialized()
            or torch.cuda.is_current_stream_capturing()
        ):
            return None
        lengths = cp_ctx.input_lengths_global_host
        prefixes = cp_ctx.prefix_lengths_host
        chunks = cp_ctx.chunk_lengths_per_req
        if (
            not lengths
            or prefixes is None
            or chunks is None
            or len(prefixes) != len(lengths)
            or len(chunks) != len(lengths)
            or any(n <= 0 for n in lengths)
            or any(p < 0 for p in prefixes)
            or (len(lengths) == 1 and prefixes != (cp_ctx.prefix_length,))
            or sum(lengths) != cp_ctx.seq_len_full
            or sum(chunks) != cp_ctx.chunk_length
            or (not bounded_replay and len(lengths) != 1)
        ):
            return None
        regions = tuple(getattr(kv_cache, "group_region_names", ()))
        region = DECODER_SWA_KV if bounded_replay else SWA_KV
        groups = [i for i, r in enumerate(regions) if int(r) == int(region)]
        if bounded_replay and len(groups) != 1:
            raise ValueError("bounded replay requires a native decoder SWA pool")
        host = getattr(attn_inputs, "kv_cache_block_id_host", None)
        spans = tuple(getattr(kv_cache, "group_seq_size_per_block", ()))
        if (
            len(groups) != 1
            or not isinstance(host, torch.Tensor)
            or host.device.type != "cpu"
            or host.ndim != 3
            or host.shape[1] != len(lengths)
            or host.shape[0] != len(regions)
            or len(spans) != len(regions)
            or spans[groups[0]] <= 0
        ):
            return None
        group_id = groups[0]
        base = kv_cache.get_layer_cache(21, regions[group_id]).kv_cache_base
        # V4.1 SWA is a byte-sliced 528-byte entry across CP ranks. The native
        # descriptor includes speculative lookahead in the actual ring stride.
        if base is None or base.ndim != 2 or base.numel() == 0:
            return None
        full_stride = base.shape[1] * base.element_size() * cp_ctx.cp_size
        # Native block strides include up to 511 bytes of TMA padding. Match
        # AttentionV41FP8._swa_entries_per_block, which excludes that padding.
        if full_stride < 528:
            return None
        if bounded_replay:
            # Only live decode state is produced. Native cache policy must
            # exclude these decoder/draft SWA pools from prefix reuse.
            starts = tuple(max(n - 128, 0) for n in lengths)
            selected_parts, offset = [], 0
            for length, start in zip(lengths, starts):
                selected_parts.append(
                    torch.arange(offset + start, offset + length, dtype=torch.int64)
                )
                offset += length
            selected = torch.cat(selected_parts)
        else:
            selected = checkpoint_positions(
                cp_ctx.prefix_length,
                cp_ctx.seq_len_full,
                host[group_id, 0].tolist(),
                spans[group_id],
                full_stride // 528,
            )
        # Exact CED avoids transport for dense checkpoints. Approximate replay
        # uses the same 128-row domain even immediately above that boundary.
        if selected.numel() == cp_ctx.seq_len_full or (
            not bounded_replay and selected.numel() * 2 >= cp_ctx.seq_len_full
        ):
            return None
        group = collective_torch._get_group(Group.TP)
        if (
            torch.distributed.get_world_size(group) != cp_ctx.cp_size
            or torch.distributed.get_rank(group) != cp_ctx.cp_rank
        ):
            raise ValueError("CED CP metadata does not match its process group")
        context, exchange, indexer_groups = _query_layout(
            cp_ctx, selected, group, keep_candidate_rows=bounded_replay
        )
        if bounded_replay:
            if len(lengths) == 1:
                context.swa_replay_start = starts[0]
            else:
                context.swa_replay_starts_host = starts
        cumulative = [0]
        for chunk in context.chunk_lengths_per_req:
            cumulative.append(cumulative[-1] + chunk)
        return cls(
            cp_ctx,
            context,
            torch.tensor(
                cumulative,
                dtype=torch.int32,
                device=cp_ctx.global_positions.device,
            ),
            exchange,
            indexer_groups,
            tuple(v4.capture_aux_hidden_layer_ids),
        )

    @contextmanager
    def candidate_publication(self, shared):
        """Limit L20 publication only when this forward will compact its consumers."""
        rows = self.exchange.candidate_rows_host
        if cp_swa_replay_starts(self.context) is None or rows is None:
            yield
            return
        shared["ced_candidate_rows"] = rows
        try:
            yield
        finally:
            shared.pop("ced_candidate_rows", None)

    def can_split_l20(self, v4):
        """Keep Stage0 publication/compaction if a split interface is absent."""
        if self.original_context.seq_len_full < _MIN_L20_SPLIT_TOKENS:
            return False
        block = v4.layers[20]
        interfaces = (
            cp_swa_replay_starts(self.context) is not None
            and callable(getattr(block, "forward_prefill_ced_l20", None))
            and callable(getattr(block.attn, "_prefill_produce", None))
            and callable(getattr(block.attn, "_prefill_query", None))
            and hasattr(getattr(block, "attn_hc", None), "pre_mix_out")
            and callable(getattr(getattr(block, "ffn", None), "_should_chunk", None))
            and callable(
                getattr(
                    getattr(getattr(block, "ffn", None), "gate", None),
                    "_project_scores",
                    None,
                )
            )
        )
        if not interfaces:
            return False
        # The callback maps all compact rows. A separately chunked compact FFN
        # would need its own call-local row offsets, so retain Stage0 there.
        if block.ffn._should_chunk(self.context.chunk_length):
            return False
        from rtp_llm.models_py.modules.dsv4.hc.v41_mega_mhc import (
            can_preserve_compact_mhc,
        )

        if not can_preserve_compact_mhc(
            block.attn_hc,
            getattr(block, "ffn_hc", None),
            getattr(block, "ffn_norm", None),
            original_tokens=self.original_context.chunk_length,
            compact_tokens=self.context.chunk_length,
        ):
            return False
        self._prepare_router_projection(block.ffn)
        return True

    def _prepare_router_projection(self, ffn):
        """Prepare only eligible Stage1 maps during the pre-L0 admission check."""
        if ffn._should_chunk(self.original_context.chunk_length):
            rows = int(ffn.max_tokens_per_rank)
            if self._router_groups is None or self._router_chunk_rows != rows:
                self._router_groups = self._router_chunk_groups(rows)
                self._router_chunk_rows = rows
        else:
            self._router_groups = None
            self._router_chunk_rows = None
        gate_rows = int(getattr(ffn.gate, "_prefill_gate_chunk_rows", 0))
        outer_rows = self._router_chunk_rows or self.original_context.chunk_length
        self._router_gate_rows = gate_rows
        self._router_gate_groups = (
            self._router_chunk_groups(outer_rows, gate_rows=gate_rows)
            if gate_rows > 0
            and min(outer_rows, self.original_context.chunk_length) > gate_rows
            else None
        )

    @contextmanager
    def router_projection(self, ffn):
        """Scope original-row router arithmetic to the compact L20 FFN."""
        gate = getattr(ffn, "gate", None)
        if (
            not self._compacted
            or cp_swa_replay_starts(self.context) is None
            or not callable(getattr(gate, "_project_scores", None))
            or not callable(getattr(ffn, "_should_chunk", None))
            or ffn._should_chunk(self.context.chunk_length)
        ):
            raise ValueError("L20 router projection requires a supported compact plan")
        projection = self.project_indexer_weights
        if ffn._should_chunk(self.original_context.chunk_length):
            if self._router_groups is None or self._router_chunk_rows != int(
                ffn.max_tokens_per_rank
            ):
                raise ValueError("CED router chunk mapping was not prepared before L0")
            projection = partial(
                self._project_original_rows, groups=self._router_groups
            )
        elif self._router_groups is not None:
            raise ValueError("CED router chunk policy changed after admission")
        if int(getattr(gate, "_prefill_gate_chunk_rows", 0)) != self._router_gate_rows:
            raise ValueError("CED gate chunk policy changed after admission")
        if self._router_gate_groups is not None:
            projection = self._project_router_weights
        missing = object()
        previous = getattr(gate, "_ced_row_projection", missing)
        gate._ced_row_projection = projection
        try:
            yield
        finally:
            if previous is missing:
                del gate._ced_row_projection
            else:
                gate._ced_row_projection = previous

    @torch.inference_mode()
    def compact_l20(self, block, residual, post, comb, input_ids):
        """Move the current Block state after full L20 source production."""
        if (
            self._compacted
            or cp_swa_replay_starts(self.context) is None
            or block.layer_id != 20
        ):
            raise ValueError("invalid L20 CED transition")
        shared = block.attn._shared_attention
        if "ced_candidate_rows" in shared:
            raise ValueError(
                "L20 compact queries cannot use original-owner candidate rows"
            )
        residual = self.exchange.compact(residual)
        post = self.exchange.compact(post)
        comb = self.exchange.compact(comb)
        input_ids = self.exchange.compact(input_ids)
        block.attn_hc.pre_mix_out = self.exchange.compact(block.attn_hc.pre_mix_out)
        shared["ced_indexer_projection"] = self.project_indexer_weights
        for key in _DERIVED_KEYS:
            shared.pop(key, None)
        self._compacted = True
        return residual, post, comb, input_ids

    def l20_query_metadata(self, common):
        """Replace only query fields; L20 keeps exact SWA/global visibility."""
        if not self._compacted or cp_swa_replay_starts(self.context) is None:
            raise ValueError("L20 query metadata requires a completed transition")
        query_context = replace(
            self.context, swa_replay_start=None, swa_replay_starts_host=None
        )
        rows = query_context.chunk_length
        slices, start = [], 0
        for count in query_context.chunk_lengths_per_req:
            slices.append(slice(start, start + count))
            start += count
        return common._replace(
            seqlen=rows,
            cp_ctx=query_context,
            freqs_cis=self.exchange.compact(common.freqs_cis),
            cu_seqlens=self.cu_seqlens,
            input_lengths=self.cu_seqlens[1:] - self.cu_seqlens[:-1],
            position_ids=query_context.global_positions,
            req_id_per_token=query_context.req_id_per_token,
            max_seqlen_q=max(query_context.chunk_lengths_per_req),
            request_row_slices=tuple(slices),
            swa_meta=None,
        )

    @torch.inference_mode()
    def compact(self, v4, hidden, input_ids, shared):
        if self._compacted:
            raise ValueError("CED plan reused across forwards")
        pre_mix = v4.layers[20].ffn_hc.pre_mix_out
        topk = shared["topk"][20]
        candidates = shared["candidates"]
        compact_hidden = self.exchange.compact(hidden)
        compact_ids = self.exchange.compact(input_ids)
        v4.layers[20].ffn_hc.pre_mix_out = self.exchange.compact(pre_mix)
        shared["topk"] = {20: self.exchange.compact(topk)}
        shared["candidates"] = (
            self.exchange.compact(candidates) if candidates is not None else None
        )
        shared["ced_indexer_projection"] = self.project_indexer_weights
        for key in _DERIVED_KEYS:
            shared.pop(key, None)
        self._compacted = True
        return compact_hidden, compact_ids

    def _router_chunk_groups(self, rows_per_chunk, *, gate_rows=0):
        if rows_per_chunk <= 0 or not self.exchange.projection_groups_host:
            raise ValueError("missing original router chunk geometry")
        device = self.context.global_positions.device
        groups = []
        for compact_rows, original_rows in self.exchange.projection_groups_host:
            # Keep original owners separate even when their chunk-local row
            # indices coincide. All grouping uses metadata retained on CPU.
            chunks = {}
            for dst, src in zip(compact_rows, original_rows):
                outer_start = src // rows_per_chunk * rows_per_chunk
                stop = min(
                    outer_start + rows_per_chunk, self.original_context.chunk_length
                )
                start = outer_start
                if gate_rows > 0:
                    start += (src - outer_start) // gate_rows * gate_rows
                    stop = min(stop, start + gate_rows)
                destinations, sources = chunks.setdefault((start, stop), ([], []))
                destinations.append(dst)
                sources.append(src - start)
            for (start, stop), (destinations, sources) in chunks.items():
                groups.append(
                    (
                        torch.tensor(destinations, dtype=torch.long, device=device),
                        torch.tensor(sources, dtype=torch.long, device=device),
                        stop - start,
                    )
                )
        return tuple(groups)

    def _project_router_weights(self, x, weight):
        # CED's gate callback precedes Gate's own chunk policy. Reproduce that
        # policy in the original owner layout, including each outer MoE tail.
        if (
            not torch.is_grad_enabled()
            and x.ndim == weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_cuda
            and weight.is_cuda
        ):
            groups = self._router_gate_groups
        else:
            groups = self._router_groups
        if groups is None:
            return self.project_indexer_weights(x, weight)
        return self._project_original_rows(x, weight, groups)

    def project_indexer_weights(self, x, weight):
        return self._project_original_rows(
            x,
            weight,
            (
                (dst, src, self.original_context.chunk_length)
                for dst, src in self.indexer_groups
            ),
        )

    def _project_original_rows(self, x, weight, groups):
        # Original CP owners can share the same local row index. Group them
        # separately: scattering all compact rows into one M-sized matrix loses
        # values at those collisions and changes sparse attention selection.
        if x.shape[0] != self.context.chunk_length:
            raise ValueError("CED projection query row count changed")
        output = x.new_zeros((x.shape[0], weight.shape[0]))
        for compact_rows, original_rows, rows in groups:
            full_x = x.new_zeros((rows, x.shape[1]))
            full_x.index_copy_(0, original_rows, x.index_select(0, compact_rows))
            projected = F.linear(full_x, weight).index_select(0, original_rows)
            output.index_copy_(0, compact_rows, projected)
            del full_x
        return output

    @torch.inference_mode()
    def restore_rows(self, tensor):
        if not self._compacted:
            raise ValueError("CED restore before compact")
        return self.exchange.restore(tensor)

    @torch.inference_mode()
    def restore_aux(self, v4):
        if not self._capture_ids:
            return
        if not self._compacted or self._aux_restored:
            raise ValueError("CED auxiliary state restored out of order")
        buffer = v4._mtp_hidden_buffer
        if tuple(v4.capture_aux_hidden_layer_ids) != self._capture_ids:
            raise ValueError("CED auxiliary capture contract changed")
        self.exchange.restore(
            buffer[: self.context.chunk_length],
            out=buffer[: self.original_context.chunk_length],
        )
        v4._note_aux_hidden_rows(
            self.original_context.chunk_length, is_cuda_graph=False
        )
        self._aux_restored = True
