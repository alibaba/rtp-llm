"""Per-forward PPU row/state adapter around the unchanged public executor."""

import logging
import os
import weakref
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from itertools import accumulate
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext, cp_swa_replay_starts
from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.dsv41.ppu.ced import _LocalRowExchange, bounded_plan
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    CP8_ROW_PREP_ENV,
    LOCAL_PREFILL_SG_ALIGN_ENV,
    LOCAL_PREFILL_SWA_SEGMENTS_ENV,
    SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import (
    RowLayout,
    RowScope,
    RowView,
    is_plain_tensor,
    tensor_metadata_key as _row_tensor_key,
)
from rtp_llm.models_py.modules.dsv41.prefill.ced import permits_ced
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _DERIVED_KEYS, _MIN_TOKENS


LOCAL_PREFILL_ROWS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_ROWS"
LOCAL_PREFILL_FULL_ROWS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_FULL_ROWS"
LOCAL_PREFILL_COMPACT_ROWS_ENV = "RTP_V41_PPU_LOCAL_PREFILL_COMPACT_ROWS"
LOCAL_PREFILL_CROSS_LAYER_MHC_ENV = "RTP_V41_PPU_LOCAL_PREFILL_CROSS_LAYER_MHC"


def bind_cross_mhc_layers(layers, geometry):
    """Bind only real adjacent modules; Engram is an entry barrier."""
    if (len(layers) != getattr(geometry, "num_layers", None)
            or any(getattr(layer, "layer_id", None) != i for i, layer in enumerate(layers))
            or tuple(i for i, layer in enumerate(layers) if getattr(layer, "engram", None) is not None)
            != tuple(getattr(geometry, "engram_layer_ids", ()) or ())):
        return {}
    result = {}
    for i, (source, target) in enumerate(zip(layers, layers[1:])):
        if (getattr(target, "engram", None) is None and getattr(source, "ffn_hc", None) is not None
                and getattr(target, "attn_hc", None) is not None
                and getattr(target, "attn_norm", None) is not None):
            try:
                result[i] = (weakref.ref(source.ffn_hc), weakref.ref(target),
                             weakref.ref(target.attn_hc), weakref.ref(target.attn_norm))
            except TypeError:
                return {}
    return result


def _cross_normal_tensor(value):
    return (value is None or (isinstance(value, torch.Tensor) and value.layout == torch.strided
                             and not value.is_neg() and not value.is_conj()))


def _cross_mix_snapshot(state):
    if state is None:
        return None
    values = tuple(getattr(state, name, None) for name in (
        "hidden", "pre_mix", "post_mix", "residual_mix",
    ))
    if not all(_cross_normal_tensor(value) for value in values):
        return None
    return tuple(_row_tensor_key(value) for value in values), values


def _cross_mix_keys(state):
    snapshot = _cross_mix_snapshot(state)
    return snapshot[0] if snapshot is not None else None


def _cross_target_snapshot(target):
    hc, norm = target.attn_hc, target.attn_norm
    values = tuple(hc.parameters_map[name] for name in ("fn", "base", "scale")) + (norm.weight,)
    if not all(isinstance(value, torch.Tensor) and _cross_normal_tensor(value) for value in values):
        return None
    return ((tuple(_row_tensor_key(value) for value in values[:3]),
             _row_tensor_key(values[3]), norm.variance_epsilon), values)


def _cross_target_key(target):
    snapshot = _cross_target_snapshot(target)
    return snapshot[0] if snapshot is not None else None


def _cross_transition_keys(transition):
    return tuple(_row_tensor_key(getattr(transition, name)) for name in (
        "residual", "normalized", "pre_mix", "post_mix", "residual_mix",
    ))


@dataclass(frozen=True)
class _NextAttnMHCTicket:
    token: object
    producer: object
    target: object
    source_state: object
    source_keys: tuple
    transition: object
    result_keys: tuple
    target_key: tuple
    epoch: tuple
    owners: tuple


class _Overlay:
    def __init__(self, original, **values):
        self._original = original
        self.__dict__.update(values)

    def __getattr__(self, name):
        return getattr(self._original, name)


def encoder_replay_inputs(inputs, attn, lengths, prefixes, cp_size, cp_rank):
    """Recover encoder state from full scheduler history; never rewrite KV prefix.

    This changes only input/CP metadata. The same public executor performs
    the single forward; the caller restores its fresh output layout afterward.
    """
    from rtp_llm.models_py.modules.dsv41.cp import build_cp_context, cp_all_gather_full
    from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_attention_inputs_by_tag
    from rtp_llm.models_py.modules.dsv41.ppu.cp import build_zigzag_plan
    from rtp_llm.models_py.modules.dsv41.ppu.engram import build_token_windows

    device = inputs.input_ids.device
    totals = tuple(p + n for p, n in zip(prefixes, lengths))
    history = getattr(inputs, "prefill_replay_token_ids", None)
    if history is None or history.numel() != sum(totals):
        raise ValueError("encoder recovery requires complete scheduler token history")
    history = history.reshape(-1).cpu().long()
    features = getattr(getattr(inputs, "multimodal_inputs", None), "multimodal_features", None)
    if features:
        raise ValueError("encoder state replay requires full-history multimodal feature metadata; fresh-only features cannot reconstruct a cached image prefix")
    full_cu = torch.tensor([0, *torch.tensor(totals).cumsum(0).tolist()])
    if cp_size > 1:
        original = build_cp_context(attn.context_parallel_info, cp_size, cp_rank, inputs.input_ids.numel(), device,
                                    position_offset=torch.tensor(prefixes, device=device), position_offset_host=torch.tensor(prefixes))
        fresh = cp_all_gather_full(inputs.input_ids.reshape(-1, 1), original).flatten().cpu()
    else:
        original = None
        fresh = inputs.input_ids.cpu().reshape(-1)
    expected = torch.cat([history[int(full_cu[i]) + p:int(full_cu[i + 1])] for i, p in enumerate(prefixes)])
    if not torch.equal(fresh.long(), expected):
        raise ValueError("encoder recovery history disagrees with fresh scheduler tokens")
    windows = build_token_windows(history, full_cu, torch.full((len(totals), 3), -1, dtype=torch.int64))
    if original is not None:
        plan = build_zigzag_plan(history, full_cu, cp_size=cp_size, cp_rank=cp_rank)
        restore = torch.empty(plan.padded_seq_len, dtype=torch.int64)
        for rank in range(cp_size):
            peer = build_zigzag_plan(history, full_cu, cp_size=cp_size, cp_rank=rank)
            restore[peer.relative_positions] = rank * plan.chunk_length + torch.arange(plan.chunk_length)
        mask = torch.cat([torch.arange(split.padded) < split.length for split in plan.requests])
        info = SimpleNamespace(
            prefill_qkv_restore_indice=restore.to(device), prefill_qkv_restore_indice_cpu=restore,
            prefill_qkv_padding_mask=mask.to(device), prefill_qkv_padding_mask_cpu=mask,
            prefill_actual_input_lengths_cpu=torch.tensor(totals, dtype=torch.int32),
            prefill_prefix_lengths_cpu=torch.zeros(len(totals), dtype=torch.int32),
            prefill_cp_chunk_lengths=torch.tensor([split.chunk for split in plan.requests], dtype=torch.int32),
        )
        local_ids, local_positions, local_cu = plan.local_input_ids, plan.local_positions, plan.cu_local
        windows = windows[full_cu[plan.local_request_ids] + local_positions]
    else:
        info = None
        local_ids, local_cu = history, full_cu
        local_positions = torch.cat([torch.arange(length) for length in totals])
    local_lengths = (local_cu[1:] - local_cu[:-1]).int()
    tagged = {
        tag: _Overlay(meta, context_parallel_info=info,
                      cu_seqlens=local_cu.int().to(device), cu_seqlens_device=local_cu.int().to(device),
                      input_lengths=local_lengths, input_lengths_device=local_lengths.to(device),
                      prefix_lengths=torch.zeros(len(totals), dtype=torch.int32),
                      prefix_lengths_device=torch.zeros(len(totals), dtype=torch.int32, device=device))
        for tag, meta in as_attention_inputs_by_tag(inputs.attention_inputs).items()
    }
    adapted = _Overlay(inputs, input_ids=local_ids.to(device), combo_position_ids=local_positions.to(device),
                       attention_inputs=tagged, engram_token_windows=windows.to(device))
    def restore_output(value, replay_context):
        if original is not None:
            full = cp_all_gather_full(value, replay_context)
            indices = full_cu.to(device)[original.req_id_per_token.long()] + original.global_positions
            return full.index_select(0, indices)
        return torch.cat([value[int(full_cu[i]) + p:int(full_cu[i + 1])] for i, p in enumerate(prefixes)])
    return adapted, totals, restore_output


class SharedForwardState:
    def __init__(self, runtime, inputs, attn, cache, *, prefill, idle, prefixes, lengths):
        self.runtime, self.inputs, self.attn, self.cache = runtime, inputs, attn, cache
        self.prefill, self.idle = bool(prefill), bool(idle)
        self.query_prefixes = tuple(map(int, prefixes))
        self.lengths = tuple(map(int, lengths))
        self._restored_rows = None
        self._local_cp_rows = None
        self._local_prefill_rows = {}
        self._local_prefill_full_rows = {}
        self._local_compact_row_proof = None
        self._l20_query_projection_proof = None
        self._local_compact_full_rows = None
        self.original_context = self.context = None
        self.prefill_mega_buffer = None
        self.dp_prefill_mega_buffer = None
        self.plan = None
        self.compacted = False
        self.compact_ids = None
        self.shared = {"layers": {i: layer.attn for i, layer in enumerate(runtime.model.v4.layers)}, "topk": {}, "candidates": None}
        self.append_starts = None
        self.restoring_aux = False
        self.plan_reason = "public_exact_executor" if runtime.policy.mode == "exact" else "ordinary_full_prefill"
        self.valid_output_rows = None
        self.head_restored = False
        self.exact_context = None
        self.public_returned = False
        self.exact_short_fallback_observed = None
        self.executed_layers = []
        self.routed_rows = {}
        self._cross_mhc_token = object()
        self._cross_mhc_ticket = self._cross_mhc_block = None
        requested = os.environ.get(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
        self._cross_mhc_enabled = requested and self.prefill and not self.idle
        self._cross_mhc_invalid = False
        self._cross_mhc_full_stamp = None
        self._cross_mhc_compact_stamp = None
        self._cross_mhc_initial = self._cross_mhc_original() if self._cross_mhc_enabled else None
        self._cross_mhc_evidence = {
            "enabled": self._cross_mhc_enabled,
            "decision": "pending" if self._cross_mhc_enabled else "disabled",
            "published": 0, "consumed": 0, "edges": [], "fallbacks": {}, "complete": False,
        }
        self._local_swa_invalid = False
        self._local_swa_full_stamp = self._local_swa_selected_stamp = None
        self._local_swa_initial = (
            (self._local_swa_full_source(), self.attn.cu_seqlens)
            if os.environ.get(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") == "1"
            and self.prefill and not self.idle and isinstance(self.attn.cu_seqlens, torch.Tensor)
            else None
        )
        from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_attention_provider import LocalPrefillAttentionProvider

        self.local_attention_provider = LocalPrefillAttentionProvider(self)
        model = runtime.model
        self.local_prefill_sg_align = (
            os.environ.get(LOCAL_PREFILL_SG_ALIGN_ENV, "0") == "1"
            and self.prefill and not self.idle and len(self.lengths) == 1
            and type(cache) is SharedNativeCache and cache._is_prefill and not cache.sharded
            and getattr(cache, "_local_sg_source_enabled", False)
            and not torch.is_grad_enabled() and not getattr(attn, "is_cuda_graph", False)
            and not getattr(attn, "is_target_verify", False)
            and not getattr(attn, "is_packed_target_verify", False)
            and not any(getattr(owner, flag, False) for owner in (inputs, attn) for flag in (
                "is_graph", "is_graph_commit", "verify", "proposal", "is_proposal", "is_speculative_decode",
            ))
            and isinstance(attn.cu_seqlens, torch.Tensor) and attn.cu_seqlens.layout == torch.strided
            and not attn.cu_seqlens.is_neg() and not attn.cu_seqlens.is_conj()
            and not getattr(model, "_is_decode_role", True)
            and not getattr(model, "_is_speculative", False)
            and getattr(model, "_prefill_cp_size", None) == 1
            and getattr(getattr(model, "parallelism_config", None), "tp_size", None) == 1
        )
        if self.local_prefill_sg_align:
            from rtp_llm.models_py.modules.dsv4 import _record_tensor, _forward_tensor_debug
            from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled

            self.local_prefill_sg_align = (
                not _record_tensor.ENABLED and not _forward_tensor_debug.enabled()
                and not optimized_diagnostics_enabled()
            )
        self._sg_rows_invalid = False
        self._sg_rows_epoch = object()
        self._sg_rows_stamps = {}
        self._sg_rows_initial = (
            (self._local_swa_full_source(), inputs, attn, cache, model, attn.cu_seqlens)
            if self.local_prefill_sg_align else None
        )

    @property
    def v4(self):
        return self.runtime.model.v4

    def discard_next_attn_mhc(self, reason):
        if self._cross_mhc_ticket is not None:
            reasons = self._cross_mhc_evidence["fallbacks"]
            reasons[reason] = reasons.get(reason, 0) + 1
        self._cross_mhc_ticket = None

    def begin_cross_mhc_block(self, block, *, allowed=True):
        self._cross_mhc_block = block if allowed else None
        ticket = self._cross_mhc_ticket
        if ticket is not None and (not allowed or ticket.target is not block):
            self.discard_next_attn_mhc("entry_order_or_metadata")

    def _cross_mhc_phase(self):
        if (not self._cross_mhc_enabled or self._cross_mhc_invalid
                or os.environ.get(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") != "1"
                or getattr(self.runtime, "current", None) is not self):
            return False
        from rtp_llm.models_py.modules.dsv41.ppu.layers import _local_prefill_business_phase

        policy = self.runtime.policy
        return (
            _local_prefill_business_phase(self)
            and policy.mode == "compact" and policy.keep_tokens == 140 and policy.window == 128
            and policy.max_rollback_tokens == 0 and not policy.encoder_replay
        )

    def _cross_mhc_original(self):
        owners = (self.inputs.input_ids, getattr(self.inputs, "combo_position_ids", None),
                  getattr(self.attn, "cu_seqlens", None), getattr(self.attn, "input_lengths", None),
                  getattr(self.attn, "prefix_lengths", None))
        if not all(_cross_normal_tensor(value) for value in owners):
            return None
        return ((self.lengths, self.query_prefixes,
                 tuple(_row_tensor_key(value) for value in owners)), owners)

    def _cross_mhc_original_valid(self):
        current = self._cross_mhc_original()
        if self._cross_mhc_initial is None:
            return False
        if current is None or current[0] != self._cross_mhc_initial[0]:
            self._cross_mhc_invalid = True
            return False
        return not self._cross_mhc_invalid

    def _cross_mhc_compact_descriptor(self):
        if not self.compacted:
            return None
        cp, plan, proof = self.context, self.plan, self._local_compact_row_proof
        exchange = getattr(plan, "exchange", None)
        ids, cu = self.compact_ids, getattr(plan, "cu_seqlens", None)
        sources = tuple(getattr(cp, name, None) for name in (
            "global_positions", "req_id_per_token", "local_is_real", "unpad_restore", "gather_restore_positions",
        )) + (getattr(exchange, "send_indices", None), getattr(exchange, "receive_positions", None), ids, cu)
        if (not all(_cross_normal_tensor(t) for t in sources)
                or not isinstance(ids, torch.Tensor) or not isinstance(cu, torch.Tensor)):
            return None
        if os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1":
            current = self._validated_local_compact_rows_key(cp)
            if current is None:
                return None
        else:
            if not self.use_local_compact_rows(cp):
                return None
            current = self._local_compact_rows_key(cp)
        key = (id(cp), id(plan), id(proof), current[0],
               _row_tensor_key(ids), _row_tensor_key(cu))
        return key, (cp, plan, proof, ids, cu, *current[1])

    def _cross_mhc_full_valid(self):
        if not self._cross_mhc_original_valid() or self._cross_mhc_full_stamp is None:
            return False
        keys, owners = self._cross_mhc_full_stamp
        cached = self._local_prefill_rows.get(id(None))
        if (keys != tuple(_row_tensor_key(t) for t in owners)
                or cached is None or cached[1] is not owners):
            self._cross_mhc_invalid = True
            return False
        return True

    def _cross_mhc_epoch(self, rows):
        if not self._cross_mhc_full_valid():
            return None
        full_keys, full_owners = self._cross_mhc_full_stamp
        host = self._cross_mhc_initial[0], full_keys
        cp = self.context
        if self.compacted:
            current = self._cross_mhc_compact_descriptor()
            if self._cross_mhc_compact_stamp is None:
                return None
            if current is None or current[0] != self._cross_mhc_compact_stamp[0]:
                self._cross_mhc_invalid = True
                return None
            if rows != 140:
                return None
            return ((host, True, current[0]),
                    (*self._cross_mhc_initial[1], *full_owners, *self._cross_mhc_compact_stamp[1]))
        if rows != self.lengths[0]:
            return None
        if cp is None:
            # L19 creates a CED plan without changing the full-row layout.
            # A future plan is not a full-row epoch or a license to transport.
            return (host, False, None), (*self._cross_mhc_initial[1], *full_owners)
        if not isinstance(cp, CPContext) or cp.gather_restore_positions is not None:
            return None
        sources = cp.global_positions, cp.req_id_per_token, cp.local_is_real, cp.unpad_restore
        if not all(_cross_normal_tensor(value) for value in sources):
            return None
        return ((host, False, id(cp), tuple(_row_tensor_key(t) for t in sources)),
                (*self._cross_mhc_initial[1], *full_owners, cp, *sources))

    def try_ffn_to_next_attn(self, unit, branch, residual, post, comb):
        if (not self._cross_mhc_phase() or self._cross_mhc_block is None
                or not isinstance(branch, torch.Tensor) or not _cross_normal_tensor(branch)
                or not isinstance(residual, torch.Tensor) or not _cross_normal_tensor(residual)):
            return None
        binding = getattr(self.runtime, "_cross_mhc_bindings", {}).get(unit.layer_id)
        if binding is None:
            return None
        producer, target = binding[0](), binding[1]()
        if (producer is not unit or target is None or self._cross_mhc_block.ffn_hc is not unit
                or binding[2]() is not target.attn_hc or binding[3]() is not target.attn_norm
                or target.layer_id != unit.layer_id + 1 or target.engram is not None
                or not self.executed_layers or self.executed_layers[-1] != unit.layer_id
                or self._cross_mhc_ticket is not None
                or residual.ndim != 3 or tuple(residual.shape[1:]) != (4, 5120)
                or (unit.layer_id == 20 and not self.compacted)):
            return None
        previous = target.attn_hc._previous_ref
        state = unit.state
        if (previous is None or previous() is not unit or state is None
                or post is not state.post_mix or comb is not state.residual_mix
                or unit.pre_mix_out is not state.pre_mix):
            return None
        epoch = self._cross_mhc_epoch(residual.shape[0])
        if epoch is None:
            return None
        source_snapshot, target_snapshot = _cross_mix_snapshot(state), _cross_target_snapshot(target)
        if source_snapshot is None or target_snapshot is None:
            return None
        source_keys, source_owners = source_snapshot
        target_key, target_owners = target_snapshot
        raw = (branch, residual, *source_owners, *target_owners)
        if any(not isinstance(t, torch.Tensor) or t.layout != torch.strided
               or t.is_neg() or t.is_conj() for t in raw if t is not None):
            return None
        from rtp_llm.models_py.modules.dsv41.ppu.mhc import PPUStepMix

        native_state = PPUStepMix(hidden=source_owners[0], pre_mix=source_owners[1],
                                  post_mix=source_owners[2], residual_mix=source_owners[3])
        with record_function_range(f"dsv41.ppu.cross_mhc.L{unit.layer_id:02d}_to_L{target.layer_id:02d}.produce"):
            transition = unit.ops.try_mhc_ffn_transition(
                branch, residual, native_state,
                dict(zip(("fn", "base", "scale"), target_owners[:3])),
                target_owners[3], target_key[2], phase="prefill",
            )
        if transition is None:
            return None
        self._cross_mhc_ticket = _NextAttnMHCTicket(
            self._cross_mhc_token, unit, target, state, source_keys, transition,
            _cross_transition_keys(transition), target_key, epoch[0],
            (*epoch[1], *source_owners, *target_owners),
        )
        evidence = self._cross_mhc_evidence
        evidence["decision"] = "active"
        evidence["published"] += 1
        evidence["edges"].append({"src": unit.layer_id, "dst": target.layer_id,
                                  "rows": residual.shape[0], "compacted": self.compacted})
        return transition.residual

    def take_next_attn_mhc(self, block, residual):
        ticket, self._cross_mhc_ticket = self._cross_mhc_ticket, None
        if ticket is None:
            return None
        if not isinstance(residual, torch.Tensor) or not _cross_normal_tensor(residual):
            self._cross_mhc_evidence["fallbacks"]["consumer_layout"] = (
                self._cross_mhc_evidence["fallbacks"].get("consumer_layout", 0) + 1)
            return None
        epoch = self._cross_mhc_epoch(residual.shape[0]) if residual.ndim == 3 else None
        binding = getattr(self.runtime, "_cross_mhc_bindings", {}).get(ticket.producer.layer_id)
        updated = ticket.transition.residual
        valid = (
            self._cross_mhc_phase() and self._cross_mhc_block is block and ticket.target is block
            and ticket.token is self._cross_mhc_token and block.engram is None
            and binding is not None and binding[0]() is ticket.producer and binding[1]() is block
            and binding[2]() is block.attn_hc and binding[3]() is block.attn_norm
            and self.executed_layers[-2:] == [ticket.producer.layer_id, block.layer_id]
            and ticket.source_state is ticket.producer.state
            and ticket.producer.pre_mix_out is ticket.source_state.pre_mix
            and ticket.source_keys == _cross_mix_keys(ticket.producer.state)
            and ticket.target_key == _cross_target_key(block)
            and ticket.result_keys == _cross_transition_keys(ticket.transition)
            and epoch is not None and ticket.epoch == epoch[0]
            and torch._C._is_alias_of(updated, residual)
            and _row_tensor_key(updated)[1:] == _row_tensor_key(residual)[1:]
        )
        if not valid:
            reasons = self._cross_mhc_evidence["fallbacks"]
            reasons["consumer_invalid"] = reasons.get("consumer_invalid", 0) + 1
            return None
        with record_function_range(f"dsv41.ppu.cross_mhc.L{block.layer_id:02d}.consume"):
            block.attn_hc.accept_transition(ticket.transition)
        self._cross_mhc_evidence["consumed"] += 1
        return ticket.transition

    def bind_context(self, context):
        if self.original_context is None and context is not None:
            self.original_context = context
            for layer in self.v4.layers:
                if layer.engram is not None and layer.engram_token_mask is not None:
                    layer.engram_token_mask = layer.engram_token_mask & context.local_is_real
        self.context = context
        if isinstance(self.cache, SharedNativeCache):
            from rtp_llm.models_py.modules.dsv41.ppu.layers import cp8_yoco_fast_prefill_enabled

            full_cp8 = (
                self.prefill
                and context is not None
                and context.cp_size == 8
                and context.kv_cache_sharded
                and context.gather_restore_positions is None
            )
            keep_plans = (
                context is not None and context.gather_restore_positions is not None
                and os.environ.get("RTP_V41_PPU_CP8_YOCO_SHARED_PREFIX_READBACK", "0") == "1"
                and self.cache.shared_prefix_stripes is not None
            )
            self.cache.bind_swa_fresh_lengths(
                self.lengths if full_cp8 else None, keep_plans=keep_plans,
            )
            if (
                full_cp8 and self.cache.shared_prefix_stripes is None
                and os.environ.get("RTP_V41_PPU_CP8_YOCO_SHARED_PREFIX_READBACK", "0") == "1"
                and cp8_yoco_fast_prefill_enabled(self)
            ):
                self.cache.bind_shared_source_prefix(self.query_prefixes)
            if (
                full_cp8
                and (self.runtime.policy.mode == "full" or cp8_yoco_fast_prefill_enabled(self))
                and context.input_lengths_global_host is not None
                and self.prefill_mega_buffer is None
                and os.environ.get("RTP_V41_PPU_CP8_FORWARD_MEGA_CAPACITY", "1") == "1"
            ):
                # Every CP rank has the same padded chunk from the host split.
                self.prefill_mega_buffer = self.runtime.model._shared_ops.moe_engine.prepare_full_cp8(
                    context.global_positions.device, context.chunk_length
                )
        if context is not None and context.gather_restore_positions is not None and self.runtime.policy.mode == "exact":
            self.exact_context = context
            self.plan_reason = "public_exact_compaction"

    def _local_row_cache_key(self, context):
        if (
            os.environ.get(LOCAL_PREFILL_ROWS_ENV, "0") != "1"
            or not self.prefill or self.idle or self.cache is None or self.cache.sharded
            or getattr(self, "is_graph", False) or getattr(self.attn, "is_cuda_graph", False)
            or (context is not None and (
                context.cp_size != 1 or context.kv_cache_sharded
                or context.global_positions.device != self.inputs.input_ids.device
            ))
        ):
            return None
        sources = (self.attn.cu_seqlens,) if context is None else (
            context.global_positions, context.req_id_per_token, context.local_is_real,
            context.gather_restore_positions,
        )
        key = (
            self.lengths, self.query_prefixes, self.inputs.input_ids.device,
            tuple(context.chunk_lengths_per_req) if context is not None else None,
            tuple(_row_tensor_key(tensor) for tensor in sources),
        )
        # Native per-forward metadata is read-only, including inference
        # tensors without version counters. Ordinary mutable metadata also
        # invalidates on version changes. Retain its sources to avoid reuse
        # of a destroyed TensorImpl's identity during this forward.
        return key, sources

    def _local_swa_full_source(self):
        return (self.lengths, self.query_prefixes, self.inputs.input_ids.device,
                _row_tensor_key(self.attn.cu_seqlens))

    def _local_swa_birth_valid(self):
        initial = self._local_swa_initial
        if os.environ.get(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") != "1" or initial is None or self._local_swa_invalid:
            return False
        if self._local_swa_full_source() != initial[0]:
            self._local_swa_invalid = True
            return False
        return True

    def _local_swa_ancestors_valid(self):
        if not self._local_swa_birth_valid() or self._local_swa_full_stamp is None:
            return False
        keys, owners = self._local_swa_full_stamp
        if keys != tuple(_row_tensor_key(value) for value in owners):
            self._local_swa_invalid = True
            return False
        return True

    def _local_swa_stage_rows(self, context, positions, requests):
        """Read creation-time provenance; never authenticate a cached result."""
        if not self._local_swa_ancestors_valid():
            return None
        if context is None:
            keys, owners = self._local_swa_full_stamp
            starts = (0,) * len(self.lengths)
        else:
            stamp = self._local_swa_selected_stamp
            if (stamp is None or stamp[0] is not context or stamp[1] is not self.plan
                    or stamp[2] is not self._local_compact_row_proof
                    or not self.use_local_compact_rows(context)
                    or stamp[3] != self._local_compact_rows_key(context)[0]):
                return None
            keys, owners = stamp[4:]
            if keys != tuple(_row_tensor_key(value) for value in owners):
                self._local_swa_invalid = True
                return None
            starts = cp_swa_replay_starts(context)
        if tuple(_row_tensor_key(value) for value in (positions, requests)) != keys[:2]:
            return None
        offset, ranges = 0, []
        for prefix, length, start in zip(self.query_prefixes, self.lengths, starts):
            ranges.append((offset, prefix + start, length - start))
            offset += length - start
        return tuple(ranges)

    def _sg_rows_original_valid(self):
        birth = self._sg_rows_initial
        if not self.local_prefill_sg_align or self._sg_rows_invalid or birth is None:
            return False
        if (self.inputs is not birth[1] or self.attn is not birth[2] or self.cache is not birth[3]
                or self.runtime.model is not birth[4] or self._local_swa_full_source() != birth[0]
                or not self.prefill or self.idle or self.cache.sharded
                or torch.is_grad_enabled()
                or any(getattr(self, flag, False) for flag in ("is_graph", "is_graph_commit", "verify", "proposal"))
                or getattr(self.attn, "is_cuda_graph", False)):
            self._sg_rows_invalid = True
            return False
        return True

    @staticmethod
    def _sg_rows_keys(rows):
        if (not isinstance(rows, tuple) or len(rows) != 4
                or any(not is_plain_tensor(value) for value in rows)):
            return None
        return tuple(_row_tensor_key(value) for value in rows)

    def _remember_sg_rows(self, context, rows):
        if not self._sg_rows_original_valid() or id(context) in self._sg_rows_stamps:
            return
        if context is not None and (not self.use_local_compact_rows(context)
                                    or not self._sg_rows_stamp_valid(None)):
            return
        keys = self._sg_rows_keys(rows)
        local = self._local_row_cache_key(context)
        if keys is not None and local is not None:
            self._sg_rows_stamps[id(context)] = (
                context, self.plan if context is not None else None, self._sg_rows_epoch,
                local[0], local[1], keys, rows,
                self._local_compact_row_proof if context is not None else None,
            )

    def _sg_rows_stamp_valid(self, context):
        stamp = self._sg_rows_stamps.get(id(context))
        local = self._local_row_cache_key(context)
        if (stamp is None or local is None or stamp[0] is not context
                or stamp[2] is not self._sg_rows_epoch or stamp[3] != local[0]
                or stamp[5] != self._sg_rows_keys(stamp[6])
                or (context is not None and (stamp[1] is not self.plan
                    or stamp[7] is not self._local_compact_row_proof
                    or not self.use_local_compact_rows(context)))):
            return False
        return True

    def local_indexer_request_slices(self, context, rows, query_rows):
        """B1 request-major slices from actual row-construction stamps only."""
        if not self._sg_rows_original_valid():
            return None
        if not self._sg_rows_stamp_valid(None) or not self._sg_rows_stamp_valid(context):
            self._sg_rows_invalid = True
            return None
        stamp = self._sg_rows_stamps[id(context)]
        if (not isinstance(rows, tuple) or len(rows) != 4
                or any(value is not owner for value, owner in zip(rows, stamp[6]))):
            return None
        lengths = self.lengths if context is None else context.chunk_lengths_per_req
        if (len(lengths) != 1 or type(lengths[0]) is not int or lengths[0] < 0
                or type(query_rows) is not int or lengths[0] != query_rows
                or tuple(value.shape for value in rows) != ((query_rows,), (query_rows,), (2,), (query_rows,))):
            return None
        return (slice(0, query_rows),)

    def _describe_rows(self, rows, context, *, query, selected_only=False):
        # This is a description, not a new activation/eligibility boundary.
        scope = RowScope(
            self, context, getattr(self, "plan", None),
            getattr(self, "_local_compact_row_proof", None),
            self.lengths, self.query_prefixes,
            tuple(context.chunk_lengths_per_req) if context is not None else (),
        )
        if context is None:
            sources = (getattr(getattr(self, "attn", None), "cu_seqlens", None),)
        else:
            sources = tuple(getattr(context, name, None) for name in (
                "global_positions", "req_id_per_token", "local_is_real",
                "gather_restore_positions", "unpad_restore",
            ))
        if query:
            layout = RowLayout.LOCAL_QUERY if context is None else RowLayout.CP_QUERY
            if (context is not None and getattr(self, "compacted", False)
                    and context is getattr(self, "context", None)):
                layout = RowLayout.COMPACT_QUERY
        else:
            layout = RowLayout.SELECTED if selected_only and context is not None else RowLayout.FULL
        return RowView(rows, layout, scope, query, sources)

    def row_view(self, context=None):
        return self._describe_rows(self._rows_legacy(context), context, query=True)

    def rows(self, context=None):
        return self.row_view(context).as_rows()

    def _rows_legacy(self, context=None):
        local = self._local_row_cache_key(context)
        slot = id(context)
        if local is not None:
            cached = self._local_prefill_rows.get(slot)
            if cached is not None and cached[0] == local[0]:
                return cached[1]
        if context is not None:
            chunks = context.chunk_lengths_per_req
            reuse = (
                self.prefill
                and context.cp_size == 8
                and context.kv_cache_sharded
                and (
                    context.gather_restore_positions is None
                    or self.use_compact_cp8_ops(context)
                )
                and os.environ.get("RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS", "0") == "1"
            )
            if reuse and self._local_cp_rows is not None and self._local_cp_rows[0] is context:
                return self._local_cp_rows[1]
            cu = torch.tensor(
                (0, *accumulate(chunks)), dtype=torch.int64 if reuse else torch.int32,
                device=self.inputs.input_ids.device,
            )
            rows = context.global_positions, context.req_id_per_token.long(), cu, context.local_is_real
            self._remember_sg_rows(context, rows)
            if reuse:
                self._local_cp_rows = context, rows
            if local is not None:
                self._local_prefill_rows[slot] = local[0], rows, local[1], context
            return rows
        device = self.inputs.input_ids.device
        if self.prefill:
            cu = self.attn.cu_seqlens.to(device=device, dtype=torch.int64)
            requests = torch.repeat_interleave(torch.arange(len(self.lengths), device=device), cu[1:] - cu[:-1])
            positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
        else:
            sizes = torch.tensor(self.lengths, device=device)
            requests = torch.repeat_interleave(torch.arange(len(self.lengths), device=device), sizes)
            positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
            cu = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
        rows = positions.long(), requests.long(), cu.int(), torch.ones_like(positions, dtype=torch.bool)
        self._remember_sg_rows(None, rows)
        if (self._cross_mhc_enabled and os.environ.get(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
                and local is not None and self._cross_mhc_full_stamp is None
                and self._cross_mhc_original_valid()):
            # Only the real construction boundary certifies derived row owners.
            # A cache hit or invalid ancestor never acquires a replacement stamp.
            self._cross_mhc_full_stamp = tuple(_row_tensor_key(t) for t in rows), rows
        if (self._local_swa_birth_valid() and local is not None
                and self._local_swa_full_stamp is None):
            # Only this real construction boundary may mint a full-row stamp.
            self._local_swa_full_stamp = tuple(_row_tensor_key(value) for value in rows), rows
        if local is not None:
            self._local_prefill_rows[slot] = local[0], rows, local[1], context
        return rows

    def use_compact_cp8_ops(self, context):
        return (
            self.compacted
            and context is not None
            and context is self.context
            and context.cp_size == 8
            and context.kv_cache_sharded
            and os.environ.get("RTP_V41_PPU_CP8_COMPACT_PREFILL_OPS", "1") == "1"
        )

    def _local_compact_rows_key(self, context, *, before_l20=False):
        """Recognize the bounded CP1 plan at its trusted activation boundary."""
        plan, policy = self.plan, self.runtime.policy
        exchange = getattr(plan, "exchange", None)
        if (
            not self.prefill or self.idle or self.compacted == before_l20 or self.cache.sharded
            or getattr(self, "is_graph", False) or getattr(self.attn, "is_cuda_graph", False)
            or getattr(self.runtime.model, "_is_speculative", False)
            or policy.mode != "compact" or getattr(policy, "encoder_replay", False)
            or getattr(policy, "max_rollback_tokens", 0) != 0
            or self.plan_reason != "compact140"
            or not isinstance(context, CPContext)
            or (not before_l20 and context is not self.context)
            or getattr(plan, "context", None) is not context
            or not isinstance(exchange, _LocalRowExchange) or exchange.group is not None
            or context.cp_size != 1 or context.cp_rank != 0 or context.kv_cache_sharded
            or context.input_lengths_global_host != self.lengths
            or context.prefix_lengths_host != self.query_prefixes
        ):
            return None
        starts = cp_swa_replay_starts(context)
        keep = getattr(policy, "keep_tokens", None)
        if keep != 140 or starts != tuple(max(n - keep, 0) for n in self.lengths):
            return None
        use_host_geometry = os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1"
        geometry_key = self.lengths, starts
        saved = getattr(self, "_local_compact_host_geometry", None) if use_host_geometry else None
        if saved is not None and saved[0] == geometry_key:
            chunks, selected, receive = saved[1]
        else:
            chunks = tuple((n - s + 1) // 2 * 2 for n, s in zip(self.lengths, starts))
            selected, receive = [], []
            original_offset = compact_offset = 0
            for n, start, chunk in zip(self.lengths, starts, chunks):
                selected.extend(range(original_offset + start, original_offset + n))
                receive.extend(range(compact_offset, compact_offset + n - start))
                original_offset += n
                compact_offset += chunk
            selected, receive = tuple(selected), tuple(receive)
            if use_host_geometry:
                self._local_compact_host_geometry = geometry_key, (chunks, selected, receive)
        if (
            not selected or context.chunk_lengths_per_req != chunks
            or context.seq_len_full != sum(self.lengths)
            or context.chunk_length != sum(chunks) or context.padded_seq_len != sum(chunks)
            or exchange.original_rows != sum(self.lengths) or exchange.compact_rows != sum(chunks)
            or exchange.send_sizes != (len(selected),) or exchange.receive_sizes != (len(selected),)
            or exchange.candidate_rows_host != selected
            or exchange.projection_groups_host != ((receive, selected),)
        ):
            return None
        sources = (
            context.global_positions, context.req_id_per_token, context.local_is_real,
            context.unpad_restore, context.gather_restore_positions,
            exchange.send_indices, exchange.receive_positions,
        )
        sizes = (sum(chunks),) * 3 + (len(selected),) * 4
        dtypes = (torch.int64, torch.int32, torch.bool) + (torch.int64,) * 4
        device = self.inputs.input_ids.device
        if any(not isinstance(t, torch.Tensor) or t.device != device
               or t.shape != (size,) or t.dtype != dtype or not t.is_contiguous()
               for t, size, dtype in zip(sources, sizes, dtypes)):
            return None
        key = (self.query_prefixes, self.lengths, chunks, starts,
               tuple(_row_tensor_key(tensor) for tensor in sources))
        return key, sources

    def local_l20_query_slice(self, rows):
        """Prove a B1 suffix against the plan's creation-stamped transport.

        This only changes query consumers. Full L20 KV producers and the
        existing compact-state callback retain their original row epoch.
        """
        proof = self._l20_query_projection_proof
        if (
            os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_DENSE_PROJECTIONS", "0") != "1"
            or not self.local_prefill_sg_align or proof is None
            or self.compacted or len(self.lengths) != 1 or self.lengths != (rows,)
            or rows <= 140 or self.runtime.policy.window != 128
            or proof[0] is not self.plan or proof[1] is not self.plan.context
        ):
            return None
        current = self._local_compact_rows_key(self.plan.context, before_l20=True)
        if current is None or current[0] != proof[2]:
            return None
        return slice(rows - 140, rows)

    def use_local_compact_rows(self, context):
        return self._validated_local_compact_rows_key(context) is not None

    def _validated_local_compact_rows_key(self, context):
        if os.environ.get(LOCAL_PREFILL_COMPACT_ROWS_ENV, "0") != "1":
            return None
        proof = self._local_compact_row_proof
        if proof is None or proof[0] is not context or proof[1] is not self.plan:
            return None
        current = self._local_compact_rows_key(context)
        # Ordinary version changes invalidate even before the first consumer.
        # Inference metadata follows the existing read-only-within-forward
        # contract; every new forward creates a fresh state/proof/storage owner.
        return current if current is not None and current[0] == proof[2] else None

    def full_row_view(self, context, *, selected_only=False):
        return self._describe_rows(
            self._full_rows_legacy(context, selected_only=selected_only), context,
            query=False, selected_only=selected_only,
        )

    def full_rows(self, context, *, selected_only=False):
        return self.full_row_view(context, selected_only=selected_only).as_full_rows()

    def _full_rows_legacy(self, context, *, selected_only=False):
        if selected_only and self.use_local_compact_rows(context):
            if self._local_compact_full_rows is None:
                selected = context.unpad_restore
                self._local_compact_full_rows = (
                    context.global_positions.index_select(0, selected),
                    context.req_id_per_token.long().index_select(0, selected), slice(None),
                )
                if self._local_swa_ancestors_valid() and self._local_swa_selected_stamp is None:
                    # Existing full ancestors and the live A06 proof must both
                    # remain valid before authenticating newly derived rows.
                    owners = self._local_compact_full_rows[:2]
                    self._local_swa_selected_stamp = (
                        context, self.plan, self._local_compact_row_proof,
                        self._local_compact_rows_key(context)[0],
                        tuple(_row_tensor_key(value) for value in owners), owners,
                    )
            return self._local_compact_full_rows
        local = self._local_row_cache_key(context) if context is not None else None
        if local is None:
            return self._build_full_rows(context, selected_only=selected_only)
        slot = id(context), bool(selected_only)
        cached = self._local_prefill_full_rows.get(slot)
        if cached is not None and cached[0] == local[0]:
            return cached[1]
        # The original selected-only cache keys on layout identity. A local
        # mutable metadata update must not reuse that earlier result after
        # the stronger source/version key above has changed.
        self._restored_rows = None
        rows = self._build_full_rows(context, selected_only=selected_only)
        self._local_prefill_full_rows[slot] = local[0], rows, local[1], context
        return rows

    def _build_full_rows(self, context, *, selected_only=False):
        if context is None:
            p, r, _, valid = self.rows()
            if (
                os.environ.get(LOCAL_PREFILL_FULL_ROWS_ENV, "0") == "1"
                and self._local_row_cache_key(None) is not None
            ):
                # rows(None) constructs exactly the real request rows. The
                # restored CP path already uses this selector ABI; retaining
                # its all-ones mask here would force dynamic boolean gathers
                # for KV, producer fields and positions on every local layer.
                return p, r, slice(None)
            return p, r, valid
        device = context.global_positions.device
        selected = context.gather_restore_positions
        reuse = selected_only or (
            selected is None
            and self.prefill
            and context.cp_size == 8
            and context.kv_cache_sharded
            and os.environ.get(CP8_ROW_PREP_ENV, "1") == "1"
        )
        if reuse:
            key = (
                self.query_prefixes, self.lengths, device,
                id(context) if selected_only else None,
            )
            if self._restored_rows is not None and self._restored_rows[0] == key:
                return self._restored_rows[1], self._restored_rows[2], slice(None)

        from rtp_llm.models_py.modules.dsv41.ppu.layers import cp8_yoco_operator_eligible

        if (
            reuse
            and cp8_yoco_operator_eligible(self)
            and self.lengths
            and self.lengths[0] > 0
            and all(length == self.lengths[0] for length in self.lengths)
            and all(prefix == self.query_prefixes[0] for prefix in self.query_prefixes)
        ):
            length = self.lengths[0]
            flat = torch.arange(len(self.lengths) * length, device=device)
            requests = flat // length
            positions = flat % length + self.query_prefixes[0]
        else:
            positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
            requests = torch.repeat_interleave(
                torch.arange(len(self.lengths), device=device),
                torch.tensor(self.lengths, device=device),
                output_size=positions.numel() if reuse else None,
            )
        if reuse:
            if selected_only:
                positions = positions.index_select(0, selected)
                requests = requests.index_select(0, selected)
            # The CP gather already restored every real row, in request order.
            self._restored_rows = key, positions, requests
            return positions, requests, slice(None)

        valid = torch.ones(positions.numel(), dtype=torch.bool, device=device)
        if selected is not None:
            valid.zero_()
            valid[selected] = True
        return positions, requests, valid

    def identity_context(self):
        positions, requests, _, valid = self.rows()
        device = positions.device
        rows = positions.numel()
        indices = torch.arange(rows, device=device)
        info = SimpleNamespace(
            prefill_qkv_restore_indice_cpu=indices.cpu(),
            prefill_qkv_padding_mask_cpu=valid.cpu(),
        )
        return CPContext(
            cp_size=1, cp_rank=0, chunk_length=rows, padded_seq_len=rows,
            seq_len_full=rows, relative_positions=indices, prefix_length=self.query_prefixes[0],
            global_positions=positions, local_is_real=valid, unpad_restore=indices,
            seq_len_total=self.query_prefixes[0] + self.lengths[0], cp_info=info,
            req_id_per_token=requests.int(), prefix_lengths=torch.tensor(self.query_prefixes, device=device),
            input_lengths_global=torch.tensor(self.lengths, device=device),
            cu_seqlens_global=torch.tensor([0, *torch.tensor(self.lengths).cumsum(0).tolist()], device=device),
            chunk_lengths_per_req=self.lengths, input_lengths_global_host=self.lengths,
            prefix_lengths_host=self.query_prefixes, first_position_host=self.query_prefixes[0],
        )

    def prepare_plan(self):
        policy = self.runtime.policy
        if policy.mode != "compact":
            return
        if self.idle or not permits_ced(self.inputs):
            self.plan_reason = "full_output_required_or_idle"
            return
        from rtp_llm.models_py.modules.dsv4 import _forward_tensor_debug as debug
        from rtp_llm.models_py.modules.dsv4 import _record_tensor as record

        if record.ENABLED or debug.enabled():
            self.plan_reason = "tensor_debug_requires_original_rows"
            return
        context = self.original_context or self.identity_context()
        self.original_context = context
        if os.environ.get("RTP_V41_SHARED_VALIDATE_STATE", "0") == "1" and context.seq_len_full < _MIN_TOKENS:
            self.exact_short_fallback_observed = CEDPlan.create(self.v4, context, self.attn, self.cache.native, bounded_replay=False) is None
        self.plan, self.plan_reason = bounded_plan(self.v4, context, self.attn, self.cache.native, policy)
        if (self.local_prefill_sg_align and self.plan is not None
                and os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_DENSE_PROJECTIONS", "0") == "1"):
            proof = self._local_compact_rows_key(self.plan.context, before_l20=True)
            self._l20_query_projection_proof = (
                (self.plan, self.plan.context, proof[0], proof[1]) if proof is not None else None
            )

    def before_block(self, block, hidden, ids, positions, cu):
        layer = block.layer_id
        self.executed_layers.append(layer)
        if not self.prefill:
            return hidden, ids, positions, cu
        if layer == 0 and hasattr(self, "window_current_tokens"):
            real = self.context.local_is_real if self.context is not None else torch.ones_like(ids, dtype=torch.bool)
            if not torch.equal(self.window_current_tokens[real].long(), ids[real].long()):
                raise ValueError("real CP Engram windows disagree with input tokens")
        if layer == 19:
            self.prepare_plan()
        if layer == 21 and self.plan is not None and not self.compacted:
            self.discard_next_attn_mhc("compaction")
            hc = self.v4.layers[20].ffn_hc
            carry = hc.state
            hidden, self.compact_ids = self.plan.compact(self.v4, hidden, ids, self.shared)
            # Public compact updates pre_mix_out; PPU's actual successor reads
            # previous.state. Transport every live field from the original once.
            hc.state = replace(carry, **{
                field.name: self.plan.exchange.compact(getattr(carry, field.name))
                if getattr(carry, field.name) is not None else None
                for field in fields(carry)
            })
            hc.pre_mix_out = hc.state.pre_mix
            self._activate_compact(hidden)
        if self.compacted:
            return hidden, self.compact_ids, self.context.global_positions, self.plan.cu_seqlens
        return hidden, ids, positions, cu

    def compact_l20_queries(self, hidden, input_ids):
        # L20 will publish its selection directly in compact row order.
        self.compact_ids = input_ids
        self.shared["topk"] = {}
        self.shared["candidates"] = None
        self.shared["ced_indexer_projection"] = self.plan.project_indexer_weights
        for key in _DERIVED_KEYS:
            self.shared.pop(key, None)
        self._activate_compact(hidden)

    def _activate_compact(self, hidden):
        self.discard_next_attn_mhc("compaction")
        close_swa_reads = getattr(self.cache, "close_local_swa_native_reads", None)
        if close_swa_reads is not None:
            # Metadata plans belong to the full-row epoch. Staged payload
            # snapshots remain owned by the cache for the L20 consumer.
            close_swa_reads()
        self.context = self.plan.context
        self.v4._propagate_cp_ctx(self.context)
        self.compacted = True
        exchange = self.plan.exchange
        # The local source indices already encode every restored real row.
        self.valid_output_rows = hidden.new_zeros(exchange.original_rows, dtype=torch.bool)
        self.valid_output_rows.index_fill_(0, exchange.send_indices, True)
        original = self.original_context
        ends_host = [p + n - 1 for p, n in zip(self.query_prefixes, self.lengths)]
        if original.cp_size == 8 and original.kv_cache_sharded:
            from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

            ends = _upload(ends_host, original.global_positions.device, torch.int64)
        else:
            ends = torch.tensor(ends_host, device=original.global_positions.device)
        last = original.local_is_real & (original.global_positions == ends[original.req_id_per_token.long()])
        if not bool(self.valid_output_rows[last].all()):
            raise RuntimeError("YOCO removed a framework first-token output row")
        logging.info("PPU shared YOCO compact %s", self.evidence())
        if os.environ.get(LOCAL_PREFILL_COMPACT_ROWS_ENV, "0") == "1":
            proof = self._local_compact_rows_key(self.context)
            self._local_compact_row_proof = (
                (self.context, self.plan, proof[0], proof[1]) if proof is not None else None
            )
            self._local_compact_full_rows = None
        if (self._cross_mhc_enabled and self._cross_mhc_compact_stamp is None
                and os.environ.get(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
                and self._cross_mhc_full_valid()):
            # A compact epoch is born only here, never at a cache consumer.
            self._cross_mhc_compact_stamp = self._cross_mhc_compact_descriptor()

    def publication(self, layer):
        return self.plan.candidate_publication(self.shared) if layer == 20 and self.plan is not None else nullcontext()

    def restore_head(self, reduced):
        # Head reduction consumes compact delayed carry before transport.
        if self.compacted:
            reduced = self.plan.restore_rows(reduced)
            self.restore_aux()
            self.head_restored = True
            logging.info("PPU shared YOCO restored %s", self.evidence())
        return reduced

    def restore_aux(self):
        if self.compacted and not self.restoring_aux and self.v4.capture_aux_hidden_layer_ids:
            self.restoring_aux = True
            try:
                self.plan.restore_aux(self.v4)
            finally:
                self.restoring_aux = False

    def evidence(self):
        context = self.plan.context if self.plan is not None else self.exact_context or self.original_context
        return {
            "mode": self.runtime.policy.mode, "keep_tokens": self.runtime.policy.keep_tokens,
            "window": self.runtime.policy.window, "max_rollback_tokens": self.runtime.policy.max_rollback_tokens,
            "plan_reason": self.plan_reason, "input_lengths": list(self.lengths),
            "prefix_lengths": list(self.query_prefixes), "compacted": self.compacted,
            "logical_retained_rows": int(context.unpad_restore.numel()) if context is not None else sum(self.lengths),
            "cp_padding_rows": int(context.padded_seq_len - context.unpad_restore.numel()) if context is not None else 0,
            "cp_size": int(context.cp_size) if context is not None else 1,
            "cp_rank": int(context.cp_rank) if context is not None else 0,
            "local_positions": context.global_positions.tolist() if context is not None else self.rows()[0].tolist(),
            "executed_layers": list(self.executed_layers),
            "restored": self.head_restored or (self.exact_context is not None and self.public_returned),
            "output_requirements": {"need_all_logits": getattr(self.inputs, "need_all_logits", None), "need_all_hidden_states": getattr(self.inputs, "need_all_hidden_states", None)},
            "exact_short_fallback_observed": self.exact_short_fallback_observed,
            "exact_compacted": self.exact_context is not None,
            "encoder_append_starts": self.append_starts,
            "local_valid_output_rows": int(self.valid_output_rows.sum()) if self.valid_output_rows is not None else None,
        }
