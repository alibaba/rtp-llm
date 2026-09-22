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

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.attn_type import DECODER_SWA_KV, SWA_KV
from rtp_llm.models_py.modules.dsv4.cp import CPContext

_CP_SIZE = 4
_TAIL_TOKENS = 3072
_MIN_TOKENS = 32768
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


def _query_layout(original, selected, group):
    """Derive transport from the actual engine inverse map, not rank0 ownership."""
    device = original.global_positions.device
    cp, rank = original.cp_size, original.cp_rank
    count = selected.numel()
    padded = ((count + 2 * cp - 1) // (2 * cp)) * (2 * cp)
    rows, half = padded // cp, padded // (2 * cp)
    canonical = torch.arange(count, dtype=torch.int64)
    pair = canonical // half
    owners = torch.where(pair < cp, pair, 2 * cp - 1 - pair)
    local = canonical % half + torch.where(pair < cp, 0, half)
    info = original.cp_info
    restore = info.prefill_qkv_restore_indice.detach().cpu().long()
    mask = info.prefill_qkv_padding_mask.detach().cpu().bool()
    if restore.shape != mask.shape or int(mask.sum()) != original.seq_len_full:
        raise ValueError("CED requires the engine's complete CP inverse map")
    source_flat = restore[mask].index_select(0, selected)
    source_owners = source_flat // original.chunk_length
    source_rows = source_flat % original.chunk_length
    own = owners == rank
    local_positions = torch.full((rows,), original.seq_len_full - 1, dtype=torch.int64)
    local_positions[local[own]] = selected[own]
    local_real = torch.zeros(rows, dtype=torch.bool)
    local_real[local[own]] = True
    sends, receives, send_sizes, receive_sizes, indexer_groups = [], [], [], [], []
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
    exchange = _RowExchange(
        original.chunk_length,
        rows,
        torch.cat(sends).to(device),
        torch.cat(receives).to(device),
        tuple(send_sizes),
        tuple(receive_sizes),
        group,
    )
    # Only query geometry changes. All write-side lengths and true prefixes
    # remain original; cp.py scatters gathered KV into that original fresh view.
    context = replace(
        original,
        chunk_length=rows,
        padded_seq_len=padded,
        relative_positions=local_positions.to(device),
        global_positions=(local_positions + original.prefix_length).to(device),
        local_is_real=local_real.to(device),
        unpad_restore=(owners * rows + local).to(device),
        unpad_restore_is_prefix=False,
        chunk_lengths_per_req=(rows,),
        req_id_per_token=torch.zeros(rows, dtype=torch.int32, device=device),
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
            or cp_ctx.input_lengths_global_host != (cp_ctx.seq_len_full,)
            or cp_ctx.prefix_lengths_host != (cp_ctx.prefix_length,)
            or cp_ctx.chunk_lengths_per_req != (cp_ctx.chunk_length,)
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
            or host.shape[1] != 1
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
            selected = torch.arange(
                cp_ctx.seq_len_full - 128, cp_ctx.seq_len_full, dtype=torch.int64
            )
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
        context, exchange, indexer_groups = _query_layout(cp_ctx, selected, group)
        if bounded_replay:
            context.swa_replay_start = cp_ctx.seq_len_full - 128
        return cls(
            cp_ctx,
            context,
            torch.tensor(
                [0, context.chunk_length],
                dtype=torch.int32,
                device=cp_ctx.global_positions.device,
            ),
            exchange,
            indexer_groups,
            tuple(v4.capture_aux_hidden_layer_ids),
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

    def project_indexer_weights(self, x, weight):
        # Original CP owners can share the same local row index. Group them
        # separately: scattering all compact rows into one M-sized matrix loses
        # values at those collisions and changes sparse attention selection.
        output = x.new_zeros((x.shape[0], weight.shape[0]))
        for compact_rows, original_rows in self.indexer_groups:
            full_x = x.new_zeros((self.original_context.chunk_length, x.shape[1]))
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
