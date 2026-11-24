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
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import env_value as optimization_env_value


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
        self._mega_all_live_masks = {}
        self._indexer_lookup_stamp = None
        self._indexer_lookup_compact_birth = None
        self._indexer_lookup_compact_rows = None
        self._compact_validation_original = None
        self._compact_validation_output = None
        self._indexer_lookup_invalid = False
        self._local_metadata_entries = {}
        self._local_metadata_stream = None
        self._local_metadata_invalid = False
        self._local_compact_row_proof = None
        self._l20_query_projection_proof = None
        self._cp_l20_projection_proof = None
        self._cp_compact_host_native_birth = None
        self._cp_compact_host_birth = None
        self._cp_compact_host_output_birth = None
        self._cp_compact_host_geometry_evidence = {
            "plan_constructs": 0, "output_coverage": 0, "workspace_bounds": 0,
            "workspace_checks": 0, "fallbacks": 0,
        }
        self._local_compact_full_rows = None
        self.original_context = self.context = None
        self._entry_plan_stamp = None
        self.prefill_mega_buffer = None
        self.dp_prefill_mega_buffer = None
        self.plan = None
        self.compacted = False
        self.compact_ids = None
        self.shared = {"layers": {i: layer.attn for i, layer in enumerate(runtime.model.v4.layers)}, "topk": {}, "candidates": None}
        self.shared["cp_compact_host_geometry"] = self._cp_compact_host_geometry_evidence
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
        requested = optimization_env_value(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
        self._cross_mhc_enabled = requested and self.prefill and not self.idle
        self._cross_mhc_invalid = False
        self._cross_mhc_full_stamp = None
        self._cross_mhc_compact_stamp = None
        self._cp_full_rows_birth = None
        self._cp_full_rows_invalid = False
        self._cp_full_rows_bound = False
        self._cp_metadata_invalid = False
        self._cp_metadata_stream = None
        self._cp_metadata_compact_birth = None
        self._cp_row_records = {}
        self._cp_metadata_entries = {}
        self._cp_metadata_evidence = {
            "row_constructs": 0, "row_hits": 0, "step_constructs": 0, "step_hits": 0,
            "positions32_constructs": 0, "positions32_hits": 0,
            "window_all_live_checks": 0, "invalidations": 0,
        }
        # The receipt retains scalar counters only, never the forward/state.
        runtime._cp_metadata_evidence = self._cp_metadata_evidence
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
            if optimization_env_value(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") == "1"
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
                or optimization_env_value(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") != "1"
                or getattr(self.runtime, "current", None) is not self):
            return False
        from rtp_llm.models_py.modules.dsv41.ppu.layers import _local_prefill_business_phase

        policy = self.runtime.policy
        return (
            (_local_prefill_business_phase(self, require_single_request=False)
             or (self._cp_rows_phase() and self._cp_full_rows_valid()))
            and policy.mode == "compact" and policy.keep_tokens == 140 and policy.window == 128
            and policy.max_rollback_tokens == 0 and not policy.encoder_replay
        )

    def _cp_rows_phase(self):
        """The CP8 rowwise seam has its own phase proof, not the local B1 guard."""
        if (not self.prefill or self.idle or torch.is_grad_enabled()
                or type(self.cache) is not SharedNativeCache or not self.cache._is_prefill
                or not self.cache.sharded or getattr(self.runtime, "current", None) is not self
                or any(getattr(owner, flag, False) for owner in (self, self.inputs, self.attn)
                       for flag in ("is_graph", "is_graph_commit", "verify", "proposal", "is_proposal",
                                    "is_speculative_decode", "is_cuda_graph", "is_target_verify",
                                    "is_packed_target_verify"))):
            return False
        model, policy = self.runtime.model, self.runtime.policy
        if (getattr(model, "_is_decode_role", True) or getattr(model, "_is_speculative", False)
                or getattr(model, "_prefill_cp_size", None) != 8
                or getattr(getattr(model, "parallelism_config", None), "tp_size", None) != 8
                or (policy.mode, policy.keep_tokens, policy.window,
                    policy.max_rollback_tokens, policy.encoder_replay) != ("compact", 140, 128, 0, False)
                or getattr(self.cache, "lengths", None)
                != tuple(p + n for p, n in zip(self.query_prefixes, self.lengths))):
            return False
        from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
        from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled
        from rtp_llm.models_py.modules.dsv4 import _record_tensor, _forward_tensor_debug

        return (not _record_tensor.ENABLED and not _forward_tensor_debug.enabled()
                and not optimized_diagnostics_enabled() and is_ppu()
                and not torch.cuda.is_current_stream_capturing())

    def _cp_full_rows_descriptor(self):
        """Snapshot the actual native context at its existing bind boundary.

        Native original tensors follow the framework's read-only-within-forward
        contract, including inference tensors without counters. Strong owners,
        rebinding, geometry and every available version are checked. No device
        values or hypothetical contiguous request-local order certify CP rows.
        """
        cp = self.original_context
        lengths, prefixes = self.lengths, self.query_prefixes
        if (not isinstance(cp, CPContext) or cp.cp_size != 8 or not cp.kv_cache_sharded
                or type(cp.cp_rank) is not int or not 0 <= cp.cp_rank < 8
                or not lengths or len(lengths) != len(prefixes)
                or any(n <= 0 for n in lengths) or any(p < 0 for p in prefixes)
                or cp.input_lengths_global_host != lengths or cp.prefix_lengths_host != prefixes
                or cp.chunk_lengths_per_req != tuple((n + 15) // 16 * 2 for n in lengths)
                or cp.chunk_length != sum(cp.chunk_lengths_per_req)
                or cp.padded_seq_len != cp.chunk_length * 8 or cp.seq_len_full != sum(lengths)
                or cp.gather_restore_positions is not None
                or not is_plain_tensor(self.inputs.input_ids)
                or self.inputs.input_ids.numel() != cp.chunk_length):
            return None
        sources = tuple(getattr(cp, name, None) for name in (
            "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
            "relative_positions", "prefix_lengths", "input_lengths_global", "cu_seqlens_global",
        ))
        sizes = ((cp.chunk_length,) * 3 + (sum(lengths), cp.chunk_length,
                                         len(lengths), len(lengths), len(lengths) + 1))
        dtypes = ((torch.int64,), (torch.int32, torch.int64), (torch.bool,), (torch.int64,),
                  (torch.int64,), (torch.int32, torch.int64), (torch.int32, torch.int64),
                  (torch.int32, torch.int64))
        device = self.inputs.input_ids.device
        if any(not is_plain_tensor(value) or value.shape != (size,) or value.dtype not in dtype
               or value.device != device for value, size, dtype in zip(sources, sizes, dtypes)):
            return None
        original = self._cross_mhc_original()
        if original is None:
            return None
        owners = (self.inputs, self.attn, self.cache, self.runtime.model, cp, cp.cp_info,
                  *sources, *original[1])
        key = (tuple(id(owner) for owner in owners[:6]), original[0],
               lengths, prefixes, cp.cp_rank, cp.chunk_lengths_per_req,
               cp.chunk_length, cp.padded_seq_len, cp.seq_len_full,
               tuple(_row_tensor_key(value) for value in sources))
        return key, owners, sum(lengths) == cp.padded_seq_len

    def _remember_cp_full_rows(self):
        # A cache hit, changed context or a later consumer cannot mint a birth.
        if (self._cp_full_rows_bound or not isinstance(self.original_context, CPContext)
                or self.original_context.cp_size != 8):
            return
        self._cp_full_rows_bound = True
        if (self._cp_full_rows_birth is not None or self._cp_full_rows_invalid
                or self.compacted or self.context is not self.original_context
                or not self._cp_rows_phase()):
            return
        descriptor = self._cp_full_rows_descriptor()
        if descriptor is None:
            return
        self._cp_full_rows_birth = descriptor
        cp = self.original_context
        from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import (
            CP8_COMPACT_HOST_GEOMETRY_ENV, cp8_native_host_metadata,
        )

        if optimization_env_value(CP8_COMPACT_HOST_GEOMETRY_ENV, "0") == "1":
            self._cp_compact_host_native_birth = cp8_native_host_metadata(cp)
        # Bind the native source stream at the original context boundary.
        # A later consumer cannot certify an unknown source stream for itself.
        if cp.global_positions.is_cuda:
            stream = torch.cuda.current_stream(cp.global_positions.device)
            if stream.device == cp.global_positions.device:
                self._cp_metadata_stream = ((stream.device, int(stream.cuda_stream)), stream)
        self._cross_mhc_evidence["cp_row_proof"] = {
            "full_rows": cp.chunk_length, "full_real_rows_global": sum(self.lengths),
            "full_padded_rows_global": cp.padded_seq_len, "full_all_live": descriptor[2],
            "compact_rows": None, "compact_real_rows_local": None, "compact_all_live": None,
            "full_mask_omitted": 0, "compact_mask_omitted": 0, "compact_mask_kept": 0,
        }

    def _cp_full_rows_valid(self):
        birth = self._cp_full_rows_birth
        if birth is None or self._cp_full_rows_invalid:
            return False
        current = self._cp_full_rows_descriptor()
        if (current is None or current[0] != birth[0]
                or any(old is not new for old, new in zip(birth[1], current[1]))):
            self._cp_full_rows_invalid = True
            return False
        return True

    def _remember_cp_compact_host_geometry(self):
        """Bind the factory's host result only at this forward's plan birth."""
        from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import (
            CP8_COMPACT_HOST_GEOMETRY_ENV, CP8L20ProjectionProof,
        )

        if (optimization_env_value(CP8_COMPACT_HOST_GEOMETRY_ENV, "0") != "1"
                or self._cp_compact_host_birth is not None):
            return
        proof = self._cp_l20_projection_proof
        native = self._cp_compact_host_native_birth
        geometry = getattr(getattr(proof, "birth", None), "host_geometry", None)
        if (not isinstance(proof, CP8L20ProjectionProof) or not proof.current(self)
                or not self._cp_full_rows_valid() or native is None or geometry is None
                or geometry.native[1] != native[1] or not geometry.current(proof.birth)
                or not permits_ced(self.inputs) or self.append_starts is not None or self._cp_metadata_stream is None
                or geometry.stream_key != self._cp_metadata_stream[0]):
            return
        # No state/runtime reference is added: the existing L20 proof already
        # has that lifetime; this record only binds its exact creation identity.
        self._cp_compact_host_birth = (proof, geometry, self._cp_full_rows_birth)
        self._cp_compact_host_geometry_evidence["plan_constructs"] += 1

    def _cp_compact_host_current(self, *, require_output=True, positions=None):
        from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import (
            CP8_COMPACT_HOST_GEOMETRY_ENV, cp8_native_host_metadata,
        )

        saved = self._cp_compact_host_birth
        if (optimization_env_value(CP8_COMPACT_HOST_GEOMETRY_ENV, "0") != "1"
                or saved is None or not self.compacted):
            return None
        proof, geometry, _ = saved
        birth = proof.birth
        source = birth.compact.global_positions
        if (type(source) is not torch.Tensor or not is_plain_tensor(source)
                or not source.is_cuda or self._cp_metadata_stream is None):
            return None
        stream = torch.cuda.current_stream(source.device)
        stream_key = stream.device, int(stream.cuda_stream)
        if stream_key != geometry.stream_key or stream_key != self._cp_metadata_stream[0]:
            return None
        # Read the actual stream BEFORE the phase, owner and metadata checks.
        # Even a same-stream hook may mutate a source or replace the plan.
        if (self._cp_l20_projection_proof is not proof or self.plan is not birth.plan
                or getattr(self.plan, "_ppu_cp8_query_birth", None) is not birth
                or self.context is not birth.compact or self.original_context is not birth.original
                or self.plan.exchange is not birth.exchange or self.plan_reason != "compact140"
                or not self.compacted or self._cp_full_rows_birth is not saved[2]
                or self.append_starts is not None or not permits_ced(self.inputs)
                or not self._cp_rows_phase()):
            return None
        native = cp8_native_host_metadata(self.original_context)
        if (not self._cp_full_rows_valid() or native is None or self._cp_compact_host_native_birth is None
                or native[1] != self._cp_compact_host_native_birth[1]
                or birth.host_geometry is not geometry or not geometry.current(birth)
                or not birth.current()):
            return None
        if positions is not None:
            source = self.context.global_positions
            if (type(positions) is not torch.Tensor or not is_plain_tensor(positions)
                    or positions.requires_grad or positions.shape != source.shape
                    or positions.stride() != source.stride() or positions.dtype != source.dtype
                    or positions.device != source.device or positions.storage_offset() != source.storage_offset()
                    or positions.data_ptr() != source.data_ptr()
                    or not torch._C._is_alias_of(positions, source)):
                return None
        if require_output:
            output = self._cp_compact_host_output_birth
            if (output is None or self.valid_output_rows is not output[0]
                    or _row_tensor_key(self.valid_output_rows) != output[1]
                    or self.compact_ids is not output[2]
                    or _row_tensor_key(self.compact_ids) != output[3]):
                return None
        return geometry

    def cp_compact_workspace_bounds(self, context, positions):
        """Exact public-upload bounds, or None before the unchanged fallback."""
        if not self.compacted or context is not self.context:
            return None
        geometry = self._cp_compact_host_current(positions=positions)
        if geometry is None:
            if self._cp_compact_host_birth is not None:
                self._cp_compact_host_geometry_evidence["fallbacks"] += 1
            return None
        self._cp_compact_host_geometry_evidence["workspace_bounds"] += 1
        return geometry.bounds

    def cp_compact_workspace_bounds_current(self, context, positions):
        """Recheck a workspace's admitted birth before every cached use.

        Callers which already sized a workspace from these bounds must fail
        before refresh/index work when False, rather than re-minting a birth.
        """
        current = context is self.context and self._cp_compact_host_current(positions=positions) is not None
        if current:
            self._cp_compact_host_geometry_evidence["workspace_checks"] += 1
        return current

    def _cp_compact_rows_descriptor(self):
        if not self.compacted or not self._cp_full_rows_valid():
            return None
        from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import CP8QueryPlanBirth

        plan, cp = self.plan, self.context
        birth = getattr(plan, "_ppu_cp8_query_birth", None)
        ids = self.compact_ids
        if (not isinstance(birth, CP8QueryPlanBirth) or not birth.current()
                or birth.original is not self.original_context or birth.plan is not plan
                or birth.compact is not cp or not is_plain_tensor(ids)
                or ids.shape != (cp.chunk_length,) or ids.dtype not in (torch.int32, torch.int64)
                or ids.device != cp.global_positions.device):
            return None
        # Actual compact activation ids are read-only within this epoch. The
        # replacement map/context tensors themselves have strict birth versions.
        key = (id(cp), id(plan), id(birth), birth.geometry, birth.keys,
               _row_tensor_key(ids), _row_tensor_key(plan.cu_seqlens))
        return key, (cp, plan, birth, ids, plan.cu_seqlens, *birth.transport, *birth.original_sources)

    def _cp_cross_mhc_epoch(self, rows):
        if not self._cross_mhc_original_valid() or not self._cp_full_rows_valid():
            return None
        full = self._cp_full_rows_birth
        host = self._cross_mhc_initial[0], full[0]
        if not self.compacted:
            if self.context is not self.original_context or rows != self.original_context.chunk_length:
                return None
            # L19's future plan is deliberately absent from the full-row epoch.
            return (host, False, id(self.original_context)), (*self._cross_mhc_initial[1], *full[1])
        current, stamp = self._cp_compact_rows_descriptor(), self._cross_mhc_compact_stamp
        if stamp is None or current is None or current[0] != stamp[0]:
            self._cross_mhc_invalid = True
            return None
        if rows != self.context.chunk_length:
            return None
        return ((host, True, current[0]), (*self._cross_mhc_initial[1], *full[1], *stamp[1]))

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
        if self._cp_full_rows_birth is not None:
            return self._cp_compact_rows_descriptor()
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
        if optimization_env_value("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1":
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
        if self._cp_full_rows_birth is not None:
            return self._cross_mhc_original_valid() and self._cp_full_rows_valid()
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
        if self._cp_full_rows_birth is not None:
            return self._cp_cross_mhc_epoch(rows)
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
            if rows != 140 * len(self.lengths) or rows != cp.chunk_length:
                return None
            return ((host, True, current[0]),
                    (*self._cross_mhc_initial[1], *full_owners, *self._cross_mhc_compact_stamp[1]))
        if rows != sum(self.lengths):
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
        self._remember_cp_full_rows()
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
            optimization_env_value(LOCAL_PREFILL_ROWS_ENV, "0") != "1"
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
        if optimization_env_value(LOCAL_PREFILL_SWA_SEGMENTS_ENV, "0") != "1" or initial is None or self._local_swa_invalid:
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
                    or stamp[2] is not self._local_compact_row_proof):
                return None
            current = self._validated_local_compact_rows_key(context)
            if current is None or stamp[3] != current[0]:
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

    def _indexer_lookup_phase(self, context=None):
        return (
            self.prefill
            and optimization_env_value("RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE", "0") == "1"
            and optimization_env_value("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1"
            and is_plain_tensor(getattr(self.inputs, "input_ids", None))
            and is_plain_tensor(getattr(self.attn, "cu_seqlens", None))
            and self._mega_all_live_phase()
            and not self.local_prefill_sg_align
            and ((context is None and not self.compacted and self.context is None)
                 or (isinstance(context, CPContext) and self.compacted and self.context is context))
            and bool(self.lengths) and len(self.lengths) == len(self.query_prefixes)
            and all(length > 0 for length in self.lengths)
            and (len(self.lengths) == 1 or self.attn.cu_seqlens.device.type == "cpu")
            and self._local_row_cache_key(context) is not None
        )

    def _remember_indexer_lookup_rows(self, rows):
        # Stamp only the original repeat_interleave construction. A batched
        # owner additionally proves the exact lengths from its native host CU.
        if (not self._indexer_lookup_phase() or self._indexer_lookup_invalid
                or self._indexer_lookup_stamp is not None):
            return
        count = sum(self.lengths)
        batch = len(self.lengths)
        if (any(not is_plain_tensor(value) for value in rows)
                or tuple(value.shape for value in rows) != ((count,), (count,), (batch + 1,), (count,))
                or rows[1].is_inference() or rows[1].dtype != torch.int64
                or not rows[1].is_contiguous() or self.inputs.input_ids.numel() != count):
            return
        sources = self.inputs.input_ids, self.attn.cu_seqlens
        if batch > 1 and not self._local_host_cu_matches_lengths():
            return
        self._indexer_lookup_stamp = SimpleNamespace(
            rows=rows, row_keys=tuple(_row_tensor_key(value) for value in rows),
            lengths=self.lengths, prefixes=self.query_prefixes,
            owners=(self.inputs, self.attn, self.cache, self.runtime.model),
            sources=sources, source_keys=tuple(_row_tensor_key(value) for value in sources),
            cache_key=self._local_row_cache_key(None)[0],
        )

    def _local_host_cu_matches_lengths(self):
        cu = self.attn.cu_seqlens
        return cu.device.type == "cpu" and tuple(cu.tolist()) == (0, *accumulate(self.lengths))

    def _indexer_lookup_original_valid(self):
        stamp = self._indexer_lookup_stamp
        if stamp is None or self._indexer_lookup_invalid:
            return False
        cached = self._local_prefill_rows.get(id(None))
        sources = self.inputs.input_ids, self.attn.cu_seqlens
        local = self._local_row_cache_key(None)
        if (tuple(_row_tensor_key(value) for value in stamp.rows) != stamp.row_keys
                or self.lengths != stamp.lengths or self.query_prefixes != stamp.prefixes
                or any(old is not new for old, new in zip(
                    stamp.owners, (self.inputs, self.attn, self.cache, self.runtime.model)))
                or tuple(_row_tensor_key(value) for value in sources) != stamp.source_keys
                or local is None or local[0] != stamp.cache_key
                or cached is None or cached[1] is not stamp.rows):
            self._indexer_lookup_invalid = True
            return False
        return True

    def _remember_indexer_lookup_compact_plan(self, context, proof):
        # Only the actual bounded-plan construction may certify compact request ids.
        # Activation and later row-cache misses must not mint a new plan birth.
        if (self._indexer_lookup_compact_birth is not None
                or not self._indexer_lookup_phase() or not self._indexer_lookup_original_valid()
                or proof is None or context.chunk_lengths_per_req != tuple(min(n, 140) for n in self.lengths)
                or any(not is_plain_tensor(value) or value.is_inference() for value in proof[1])):
            return
        self._indexer_lookup_compact_birth = self.plan, context, proof[0], proof[1]

    def _indexer_lookup_compact_valid(self, context):
        if context is None or not self._indexer_lookup_phase(context) or self._indexer_lookup_invalid:
            return False
        birth = self._indexer_lookup_compact_birth
        if birth is None or not self._indexer_lookup_original_valid():
            return False
        current = self._validated_local_compact_rows_key(context)
        if (birth[0] is not self.plan or birth[1] is not context or current is None
                or current[0] != birth[2]
                or any(not is_plain_tensor(value) or value.is_inference() for value in current[1])):
            self._indexer_lookup_invalid = True
            return False
        return True

    def local_indexer_lookup_slices(self, context, rows, query_rows):
        """Describe authenticated request ranges for the original score/TopK path."""
        if not self._indexer_lookup_phase(context) or not self._indexer_lookup_original_valid():
            return None
        if context is None:
            stamp = self._indexer_lookup_stamp
            owners, keys, lengths = stamp.rows, stamp.row_keys, stamp.lengths
        else:
            if not self._indexer_lookup_compact_valid(context):
                return None
            stamp = self._indexer_lookup_compact_rows
            if stamp is None or stamp[0] is not context:
                return None
            _, owners, keys = stamp
            lengths = context.chunk_lengths_per_req
        cached = self._local_prefill_rows.get(id(context))
        if (type(query_rows) is not int or query_rows != sum(lengths)
                or not isinstance(rows, tuple) or len(rows) != 4
                or any(value is not owner for value, owner in zip(rows, owners))
                or any(not is_plain_tensor(value) for value in rows)
                or tuple(_row_tensor_key(value) for value in rows) != keys
                or cached is None or cached[1] is not owners):
            self._indexer_lookup_invalid = True
            return None
        boundaries = tuple(accumulate(lengths, initial=0))
        return tuple(slice(start, stop) for start, stop in zip(boundaries, boundaries[1:]))

    def _local_metadata_snapshot(self, context, rows):
        # Callers authenticate the original read-only consumer/builder chain.
        # Here only the existing full/compact row birth and current state matter.
        if (type(self) is not SharedForwardState or getattr(self, "_local_metadata_invalid", True)
                or getattr(self.runtime, "current", None) is not self
                or self._indexer_lookup_invalid or not self._mega_all_live_phase()
                or self.local_prefill_sg_align
                or any(optimization_env_value(name, "0") != "1" for name in (
                    LOCAL_PREFILL_ROWS_ENV, "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE",
                    "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA"))
                or type(rows) is not tuple or len(rows) != 4
                or any(type(value) is not torch.Tensor or value.is_inference()
                       or not is_plain_tensor(value) or not value.is_contiguous() for value in rows)
                or not rows[0].is_cuda):
            return None
        full = self._indexer_lookup_stamp
        cached = self._local_prefill_rows.get(id(context))
        full_cached = self._local_prefill_rows.get(id(None))
        if (full is None or cached is None
                or any(value is not owner for value, owner in zip(rows, cached[1]))
                or full_cached is None
                or full_cached[1] is not full.rows
                or len(self.lengths) != len(self.query_prefixes)):
            return None
        sources = rows + (self.inputs.input_ids, self.attn.cu_seqlens)
        identity_sources = rows
        owners = (self.runtime, self.inputs, self.attn, self.cache, self.runtime.model,
                  self.runtime.policy, self.plan, context, full, cached, full_cached,
                  self.original_context)
        policy = self.runtime.policy
        geometry = (self.lengths, self.query_prefixes, self.plan_reason, policy.mode,
                    getattr(policy, "keep_tokens", None), getattr(policy, "encoder_replay", False),
                    getattr(policy, "max_rollback_tokens", 0))
        count = sum(self.lengths)
        if context is None:
            if self.compacted or self.context is not None or cached[1] is not full.rows:
                return None
        else:
            birth, derived = self._indexer_lookup_compact_birth, self._indexer_lookup_compact_rows
            if (type(context) is not CPContext or not self.compacted or self.context is not context
                    or birth is None or derived is None or derived[0] is not context or derived[1] is not cached[1]
                    or self.plan is not birth[0] or context is not birth[1]
                    or optimization_env_value(LOCAL_PREFILL_COMPACT_ROWS_ENV, "0") != "1"):
                return None
            exchange, policy = self.plan.exchange, self.runtime.policy
            transport = (context.global_positions, context.req_id_per_token, context.local_is_real,
                         context.unpad_restore, context.gather_restore_positions,
                         exchange.send_indices, exchange.receive_positions)
            if any(type(value) is not torch.Tensor or value.is_inference() for value in (*full.rows, *transport)):
                return None
            sources += full.rows + transport
            identity_sources += full.rows + transport
            owners += (birth, derived, self._local_compact_row_proof, exchange,
                       self.plan.context, getattr(self.plan, "original_context", None), exchange.group)
            geometry += (
                context.cp_size, context.cp_rank, context.kv_cache_sharded, context.chunk_length,
                context.padded_seq_len, context.seq_len_full, context.chunk_lengths_per_req,
                context.input_lengths_global_host, context.prefix_lengths_host,
                getattr(context, "swa_replay_starts_host", None), getattr(context, "swa_replay_start", None),
                exchange.original_rows, exchange.compact_rows, exchange.send_sizes, exchange.receive_sizes,
                exchange.candidate_rows_host, exchange.projection_groups_host,
            )
            count = context.chunk_length
        if any(type(value) not in (int, bool, str, tuple, type(None)) for value in geometry):
            return None
        # Host tuples are immutable birth metadata. Pin and compare their
        # identity instead of traversing the 140-row transport every hit.
        owners += geometry
        geometry = tuple((tuple, id(value)) if type(value) is tuple else value for value in geometry)
        if (tuple(value.shape for value in rows) != ((count,), (count,), (len(self.lengths) + 1,), (count,))
                or tuple(value.dtype for value in rows) != (torch.int64, torch.int64, torch.int32, torch.bool)
                or any(value.device != rows[0].device for value in rows)):
            return None
        stream = torch.cuda.current_stream(rows[0].device)
        if stream.device != rows[0].device:
            return None
        # Native input getters may wrap the same TensorImpl again. Only the
        # four known row owners/CP tensors require Python object identity.
        key = (tuple(id(value) for value in (*owners, *identity_sources)), geometry,
               tuple(_row_tensor_key(value) for value in sources))
        return key, (*owners, *sources), (rows[0].device, int(stream.cuda_stream)), stream

    @staticmethod
    def _local_metadata_result_key(value, *, step):
        if step:
            fields = vars(value)
            if fields.get("parallel_context", object()) is not None:
                return None
            tensors = tuple(fields.get(name) for name in (
                "positions", "valid_mask", "request_ids", "owner_ranks", "request_offsets",
            ))
        else:
            tensors = (value,)
        if any(type(tensor) is not torch.Tensor or tensor.is_inference()
               or not is_plain_tensor(tensor) for tensor in tensors):
            return None
        return (type(value), id(value), tuple((id(tensor), _row_tensor_key(tensor)) for tensor in tensors))

    def _local_metadata_entry(self, context, rows):
        snapshot = self._local_metadata_snapshot(context, rows)
        if snapshot is None:
            self._local_metadata_invalid = True
            return None
        key, owners, stream_key, stream = snapshot
        if self._local_metadata_stream is None:
            self._local_metadata_stream = stream_key, stream
        elif self._local_metadata_stream[0] != stream_key:
            self._local_metadata_invalid = True
            return None
        slot = context is not None
        entry = self._local_metadata_entries.get(slot)
        if entry is None:
            # Full qualification occurs only at the first actual consumer of
            # each epoch. Hits compare the stamped sources/owners directly.
            if self.local_indexer_lookup_slices(context, rows, rows[0].numel()) is None:
                self._local_metadata_invalid = True
                return None
            entry = SimpleNamespace(key=key, owners=owners, values={}, retained=[], builder=None)
            self._local_metadata_entries[slot] = entry
        elif entry.key != key or any(
            self._local_metadata_result_key(value, step=kind == "step") != saved
            for kind, (value, saved) in entry.values.items()
        ):
            self._local_metadata_invalid = True
            return None
        return entry

    def _local_metadata_remember(self, entry, context, rows, kind, value):
        # A later field rebind must not release an earlier in-flight buffer.
        entry.retained.extend((value, *vars(value).values()) if kind == "step" else (value,))
        result_key = self._local_metadata_result_key(value, step=kind == "step")
        entry.values[kind] = value, result_key
        current = self._local_metadata_snapshot(context, rows)
        if (result_key is None or current is None or current[0] != entry.key
                or current[2] != self._local_metadata_stream[0]):
            self._local_metadata_invalid = True
        # A completed first construction is still this call's original result.
        # Never return None here and make the caller repeat a dispatched cast.
        return value

    def local_prefill_step_context(self, context, rows, *, builder):
        """Reuse only a current, same-stream main context from a known row epoch."""
        entry = self._local_metadata_entry(context, rows)
        if entry is None:
            return None
        if entry.builder is not None and entry.builder is not builder:
            self._local_metadata_invalid = True
            return None
        if "step" in entry.values:
            return entry.values["step"][0]
        entry.builder = builder
        try:
            with torch.inference_mode(False), torch.no_grad():
                value = builder(*rows[:3], valid_mask=rows[3])
        except BaseException:
            self._local_metadata_invalid = True
            raise
        return self._local_metadata_remember(entry, context, rows, "step", value)

    def local_prefill_positions32(self, context, rows):
        """A private int32 view of the epoch, for the admitted Mega consumer only."""
        entry = self._local_metadata_entry(context, rows)
        if entry is None:
            return None
        if "positions32" in entry.values:
            return entry.values["positions32"][0]
        try:
            with torch.inference_mode(False), torch.no_grad():
                value = rows[0].to(torch.int32)
        except BaseException:
            self._local_metadata_invalid = True
            raise
        return self._local_metadata_remember(entry, context, rows, "positions32", value)

    def _invalidate_cp_metadata(self):
        if not self._cp_metadata_invalid:
            self._cp_metadata_evidence["invalidations"] += 1
        self._cp_metadata_invalid = True

    def _cp_metadata_epoch(self, context):
        """Separate CP admission; neither local B1 guard is widened here."""
        if type(self) is not SharedForwardState or self._cp_metadata_invalid:
            return None
        if self._cp_full_rows_birth is None:
            return None
        if (os.environ.get("RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS", "0") != "1"
                or not self._cp_rows_phase() or not self._cp_full_rows_valid()
                or context is not self.context or not isinstance(context, CPContext)
                or not context.global_positions.is_cuda or self._cp_metadata_stream is None):
            self._invalidate_cp_metadata()
            return None
        stream = torch.cuda.current_stream(context.global_positions.device)
        if (stream.device != context.global_positions.device
                or (stream.device, int(stream.cuda_stream)) != self._cp_metadata_stream[0]):
            self._invalidate_cp_metadata()
            return None
        full = self._cp_full_rows_birth[0]
        if not self.compacted:
            if context is not self.original_context:
                self._invalidate_cp_metadata()
                return None
            # A CED plan created at L19 has not changed any full-row owner.
            return (False, full), (context,)
        current = self._cp_compact_rows_descriptor()
        birth = self._cp_metadata_compact_birth
        if birth is None or current is None or current[0] != birth[0]:
            self._invalidate_cp_metadata()
            return None
        return (True, full, current[0]), current[1]

    @staticmethod
    def _cp_metadata_current_stream(rows):
        if (type(rows) is not tuple or len(rows) != 4
                or any(type(value) is not torch.Tensor or not is_plain_tensor(value)
                       or not value.is_contiguous() or value.requires_grad for value in rows)
                or not rows[0].is_cuda):
            return None
        stream = torch.cuda.current_stream(rows[0].device)
        if stream.device != rows[0].device:
            return None
        return (rows[0].device, int(stream.cuda_stream)), stream

    def _cp_metadata_stream_valid(self, rows):
        current = self._cp_metadata_current_stream(rows)
        if current is None:
            self._invalidate_cp_metadata()
            return False
        if self._cp_metadata_stream is None or self._cp_metadata_stream[0] != current[0]:
            self._invalidate_cp_metadata()
            return False
        return True

    def _remember_cp_rows(self, context, rows, epoch):
        """Only the original long()/cu construction may certify derived rows.

        Original native CP positions/masks may be inference tensors under the
        existing read-only contract. New long-request/cu owners are versioned;
        no cache hit or consumer certifies an unknown versionless replacement.
        """
        if epoch is None or self._cp_metadata_invalid:
            return
        count, requests = context.chunk_length, context.req_id_per_token
        if (tuple(value.shape for value in rows) != ((count,), (count,),
                                                     (len(self.lengths) + 1,), (count,))
                or rows[0] is not context.global_positions or rows[3] is not context.local_is_real
                or rows[1].dtype != torch.int64 or rows[2].dtype not in (torch.int32, torch.int64)
                or rows[2].is_inference()
                or any(value.device != rows[0].device for value in rows)
                or (requests.dtype == torch.int64 and rows[1] is not requests)
                or (requests.dtype != torch.int64 and (
                    rows[1].is_inference() or torch._C._is_alias_of(rows[1], requests)))
                or not self._cp_metadata_stream_valid(rows)):
            self._invalidate_cp_metadata()
            return
        slot = bool(self.compacted)
        if slot in self._cp_row_records:
            self._invalidate_cp_metadata()
            return
        # No state/runtime/model object is retained by these additional caches.
        self._cp_row_records[slot] = SimpleNamespace(
            context=context, epoch=epoch[0], owners=epoch[1], rows=rows,
            keys=tuple(_row_tensor_key(value) for value in rows),
        )
        self._cp_metadata_evidence["row_constructs"] += 1

    def _cp_row_record(self, context, rows=None, *, epoch=None):
        epoch = self._cp_metadata_epoch(context) if epoch is None else epoch
        if epoch is None:
            return None
        record = self._cp_row_records.get(bool(self.compacted))
        if record is None:
            return None
        if (record.context is not context or record.epoch != epoch[0]
                or record.keys != tuple(_row_tensor_key(value) for value in record.rows)
                or (rows is not None and (type(rows) is not tuple or len(rows) != 4
                    or any(old is not new for old, new in zip(record.rows, rows))))
                or not self._cp_metadata_stream_valid(record.rows)):
            self._invalidate_cp_metadata()
            return None
        return record

    @staticmethod
    def _cp_metadata_result_key(value, rows, *, step):
        if step:
            from rtp_llm.models_py.modules.dsv41.backend import V41StepContext

            if type(value) is not V41StepContext or vars(value).get("parallel_context", object()) is not None:
                return None
            tensors = tuple(vars(value).get(name) for name in (
                "positions", "valid_mask", "request_ids", "owner_ranks", "request_offsets",
            ))
            sizes = (rows[0].numel(),) * 4 + (rows[2].numel(),)
            dtypes = (torch.int64, torch.bool, torch.int64, torch.int64, torch.int64)
            if any(type(t) is not torch.Tensor or not is_plain_tensor(t) or not t.is_contiguous()
                   or t.shape != (n,) or t.dtype != dtype or t.device != rows[0].device
                   or t.requires_grad for t, n, dtype in zip(tensors, sizes, dtypes)):
                return None
            # The standard builder borrows precisely these native/derived row
            # owners. Only those known aliases may remain inference tensors.
            for tensor, source in zip(tensors[:3], (rows[0], rows[3], rows[1])):
                if (not torch._C._is_alias_of(tensor, source)
                        or _row_tensor_key(tensor)[1:] != _row_tensor_key(source)[1:]):
                    return None
            if (any(t.is_inference() for t in tensors[3:])
                    or any(torch._C._is_alias_of(tensors[3], source) for source in rows)):
                return None
        else:
            tensors = (value,)
            if (type(value) is not torch.Tensor or not is_plain_tensor(value) or value.is_inference()
                    or value.dtype != torch.int32 or value.shape != rows[0].shape
                    or value.device != rows[0].device or not value.is_contiguous() or value.requires_grad
                    or any(torch._C._is_alias_of(value, source) for source in rows)):
                return None
        return type(value), id(value), tuple((id(tensor), _row_tensor_key(tensor)) for tensor in tensors)

    def _cp_metadata_entry(self, context, rows):
        record = self._cp_row_record(context, rows)
        if record is None:
            return None
        slot = bool(self.compacted)
        entry = self._cp_metadata_entries.get(slot)
        if entry is None:
            entry = SimpleNamespace(record=record, values={}, retained=[], builder=None)
            self._cp_metadata_entries[slot] = entry
        elif entry.record is not record or any(
            self._cp_metadata_result_key(value, record.rows, step=kind == "step") != key
            for kind, (value, key) in entry.values.items()
        ):
            self._invalidate_cp_metadata()
            return None
        return entry

    def _remember_cp_metadata(self, entry, context, rows, kind, value):
        tensors = tuple(vars(value).values()) if kind == "step" else (value,)
        entry.retained.extend(tensor for tensor in tensors if isinstance(tensor, torch.Tensor))
        key = self._cp_metadata_result_key(value, entry.record.rows, step=kind == "step")
        entry.values[kind] = value, key
        if key is None or self._cp_row_record(context, rows) is not entry.record:
            self._invalidate_cp_metadata()
        # The first real construction has already dispatched. Return it once,
        # even if a custom builder changed a source; never trigger a retry.
        return value

    def cp_prefill_step_context(self, context, rows, *, builder):
        """Called only by the authenticated read-only CP main-context consumer."""
        entry = self._cp_metadata_entry(context, rows)
        if entry is None:
            return None
        if entry.builder is not None and entry.builder is not builder:
            self._invalidate_cp_metadata()
            return None
        if "step" in entry.values:
            self._cp_metadata_evidence["step_hits"] += 1
            return entry.values["step"][0]
        entry.builder = builder
        try:
            with torch.inference_mode(False), torch.no_grad():
                value = builder(*rows[:3], valid_mask=rows[3])
        except BaseException:
            self._invalidate_cp_metadata()
            raise
        self._cp_metadata_evidence["step_constructs"] += 1
        return self._remember_cp_metadata(entry, context, rows, "step", value)

    def cp_prefill_cached_attention_rows(self, context, *, builder):
        """Borrow a row tuple and its StepContext from one validated entry."""
        if type(self) is not SharedForwardState:
            return None
        if (RowScope is not _CP_ROW_SCOPE or RowScope.__init__ is not _CP_ROW_SCOPE_INIT
                or RowView is not _CP_ROW_VIEW or RowView.__init__ is not _CP_ROW_VIEW_INIT
                or RowView.as_rows is not _CP_ROW_VIEW_AS_ROWS
                or any(name in vars(self) or vars(SharedForwardState).get(name) is not original
                       for name, original in _CP_CACHED_ATTENTION_METHODS)):
            return None
        cached = self._cp_metadata_entries.get(bool(self.compacted))
        if cached is None or cached.builder is not builder or "step" not in cached.values:
            return None
        entry = self._cp_metadata_entry(context, None)
        if entry is not cached:
            return None
        self._cp_metadata_evidence["row_hits"] += 1
        self._cp_metadata_evidence["step_hits"] += 1
        hits = self._cp_metadata_evidence.get("attention_pair_hits", 0)
        self._cp_metadata_evidence["attention_pair_hits"] = hits + 1
        return entry.record.rows, entry.values["step"][0]

    def cp_prefill_positions32(self, context, rows):
        """A same-stream private cast for the authenticated all-live Mega reader."""
        entry = self._cp_metadata_entry(context, rows)
        if entry is None:
            return None
        # Padding retains the existing masked positions/where/cast path.
        all_live = (self._cp_full_rows_birth[2] if not self.compacted else
                    sum(self.plan.exchange.receive_sizes) == context.chunk_length)
        if not all_live:
            return None
        if "positions32" in entry.values:
            self._cp_metadata_evidence["positions32_hits"] += 1
            return entry.values["positions32"][0]
        try:
            with torch.inference_mode(False), torch.no_grad():
                value = rows[0].to(torch.int32)
        except BaseException:
            self._invalidate_cp_metadata()
            raise
        self._cp_metadata_evidence["positions32_constructs"] += 1
        return self._remember_cp_metadata(entry, context, rows, "positions32", value)

    def _cp_window_all_live_rows(self, ids, window, comparison_ids, mask):
        cp = self.original_context
        if (self.compacted or self.context is not cp or self._cp_full_rows_birth is None
                or not self._cp_full_rows_birth[2] or not self._cp_rows_phase()
                or not self._cp_full_rows_valid() or mask is not cp.local_is_real):
            return False
        if any(type(value) is not torch.Tensor or not is_plain_tensor(value)
               or value.ndim != 1 or value.shape != (cp.chunk_length,)
               or value.dtype not in (torch.int32, torch.int64) or not value.is_contiguous()
               or value.device != cp.global_positions.device for value in (ids, comparison_ids)):
            return False
        # The native Engram frame retains windows[:, 0], a read-only column
        # view (normally stride3). An all-live mask selects every logical row,
        # so torch.equal(window.long(), ...) preserves its ordering and cast
        # semantics without first gathering that column into another tensor.
        # Keep the native row operands contiguous and unknown window layouts
        # on the existing masked path; no new owner or transport is certified.
        if (window is not getattr(self, "window_current_tokens", None)
                or type(window) is not torch.Tensor or not is_plain_tensor(window)
                or window.ndim != 1 or window.shape != (cp.chunk_length,)
                or window.dtype not in (torch.int32, torch.int64)
                or window.device != cp.global_positions.device or window.stride(0) <= 0):
            return False
        self._cp_metadata_evidence["window_all_live_checks"] += 1
        return True

    def _compact_host_geometry(self):
        """Return the exact compact row geometry only while its birth holds.

        This describes values produced by the original arange/identity and
        bounded-layout constructors. It does not infer values from tensor
        shapes or accept versionless transport as immutable.
        """
        context = self.context
        original_birth = self._compact_validation_original
        birth = self._indexer_lookup_compact_birth
        if (original_birth is None or birth is None
                or not self._indexer_lookup_phase(context)
                or not self._indexer_lookup_original_valid()
                or original_birth[0] is not self.original_context
                or original_birth[1] is not self._indexer_lookup_stamp
                or birth[0] is not self.plan or birth[1] is not context
                or getattr(self.plan, "original_context", None) is not self.original_context):
            return None
        original, full_rows = self.original_context, self._indexer_lookup_stamp.rows
        count = sum(self.lengths)
        owners = original.global_positions, original.req_id_per_token, original.local_is_real
        if (any(not is_plain_tensor(value) or value.is_inference() for value in (*full_rows, *owners))
                or any(value is not owner for value, owner in zip(owners, original_birth[2]))
                or tuple(_row_tensor_key(value) for value in owners) != original_birth[3]
                or original.global_positions is not full_rows[0] or original.local_is_real is not full_rows[3]
                or (original.cp_size, original.cp_rank, original.kv_cache_sharded) != (1, 0, False)
                or (original.chunk_length, original.padded_seq_len, original.seq_len_full) != (count,) * 3
                or original.input_lengths_global_host != self.lengths
                or original.chunk_lengths_per_req != self.lengths
                or original.prefix_lengths_host != self.query_prefixes):
            return None
        current = self._local_compact_rows_key(context)
        if (current is None or current[0] != birth[2]
                or any(not is_plain_tensor(value) or value.is_inference() for value in current[1])):
            return None
        selected = self.plan.exchange.candidate_rows_host
        # The certified transport lists each request's retained suffix in
        # request order. Convert physical rows back to per-request positions.
        positions = tuple(
            position for prefix, length in zip(self.query_prefixes, self.lengths)
            for position in range(prefix + max(length - 140, 0), prefix + length)
        )
        covered = all(end - 1 in selected for end in accumulate(self.lengths))
        return positions, len(selected), covered

    def _compact_host_evidence(self):
        stamp = self._compact_validation_output
        if stamp is None or self.valid_output_rows is not stamp[0]:
            return None
        value = self.valid_output_rows
        if (not is_plain_tensor(value) or value.is_inference()
                or _row_tensor_key(value) != stamp[1]):
            return None
        geometry = self._compact_host_geometry()
        return geometry if geometry is not None and geometry[2] else None

    def _mega_all_live_phase(self):
        """Only eager, local Prefill may omit the native invalid-row mask."""
        if (not self.prefill or self.idle or self.cache is None or self.cache.sharded
                or torch.is_grad_enabled()
                or getattr(self.runtime.model, "_is_decode_role", False)
                or getattr(self.runtime.model, "_is_speculative", False)
                or (self.inputs.input_ids.is_cuda and torch.cuda.is_current_stream_capturing())):
            return False
        flags = (
            "is_graph", "is_graph_commit", "verify", "proposal", "is_proposal",
            "is_speculative_decode", "is_cuda_graph", "is_target_verify", "is_packed_target_verify",
        )
        return not any(getattr(owner, flag, False)
                       for owner in (self, self.inputs, self.attn) for flag in flags)

    def _remember_mega_all_live_mask(self, context, mask, *, plan=None):
        # Called only at an all-ones construction or the bounded-plan birth
        # boundary. A cache hit/consumer must never certify an arbitrary mask.
        if (not self._mega_all_live_phase() or not is_plain_tensor(mask)
                or mask.is_inference() or mask.dtype != torch.bool
                or mask.ndim != 1 or not mask.is_contiguous()):
            return
        geometry = None if context is None else (
            context.cp_size, context.cp_rank, context.kv_cache_sharded,
            context.chunk_length, context.padded_seq_len, context.chunk_lengths_per_req,
        )
        self._mega_all_live_masks[id(context)] = (
            context, plan, mask, _row_tensor_key(mask), geometry,
            self.lengths, self.query_prefixes,
            (self.inputs, self.attn, self.cache, self.runtime.model),
            tuple(_row_tensor_key(value) for value in (self.inputs.input_ids, self.attn.cu_seqlens)),
            (self.inputs.input_ids, self.attn.cu_seqlens),
        )

    def mega_attention_valid_mask(self, context, mask, query_rows):
        """Return None only for a still-unmodified construction-born all-live mask.

        No device values, host transfers or expanded compact-layout proof are
        consulted here. Versionless external/inference masks fail closed.
        """
        if self._cp_full_rows_birth is not None:
            return self._cp_mega_attention_valid_mask(context, mask, query_rows)
        if not self._mega_all_live_phase():
            return mask
        stamp = self._mega_all_live_masks.get(id(context))
        if (stamp is None or stamp[0] is not context or stamp[2] is not mask
                or mask.is_inference() or mask.shape != (query_rows,)
                or stamp[3] != _row_tensor_key(mask)
                or stamp[5:7] != (self.lengths, self.query_prefixes)
                or any(old is not new for old, new in zip(
                    stamp[7], (self.inputs, self.attn, self.cache, self.runtime.model)))
                or stamp[8] != tuple(_row_tensor_key(value) for value in
                                     (self.inputs.input_ids, self.attn.cu_seqlens))):
            return mask
        if context is not None:
            geometry = (
                context.cp_size, context.cp_rank, context.kv_cache_sharded,
                context.chunk_length, context.padded_seq_len, context.chunk_lengths_per_req,
            )
            if (geometry != stamp[4] or geometry[:3] != (1, 0, False)
                    or context.local_is_real is not mask
                    or context.chunk_length != query_rows):
                return mask
            if stamp[1] is not None and (
                self.plan is not stamp[1] or self.context is not context or not self.compacted
            ):
                return mask
        elif self.context is not None:
            # A local all-ones tuple cannot authorize an unrelated CP epoch.
            return mask
        return None

    def _cp_mega_attention_valid_mask(self, context, mask, query_rows):
        if (not self._cp_rows_phase() or not self._cp_full_rows_valid()
                or context is not self.context or not isinstance(context, CPContext)
                or mask is not context.local_is_real or not is_plain_tensor(mask)
                or mask.shape != (query_rows,) or context.chunk_length != query_rows):
            return mask
        evidence = self._cross_mhc_evidence["cp_row_proof"]
        if not self.compacted:
            if context is self.original_context and self._cp_full_rows_birth[2]:
                evidence["full_mask_omitted"] += 1
                return None
            return mask
        current = self._cp_compact_rows_descriptor()
        if current is None:
            return mask
        # CP8's140 retained rows pad to144 for B8. Rank0 receives112 real
        # rows; other ranks receive144. Only the actual private transport's
        # complete receive layout certifies an all-live compact mask.
        if sum(self.plan.exchange.receive_sizes) != query_rows:
            evidence["compact_mask_kept"] += 1
            return mask
        evidence["compact_mask_omitted"] += 1
        return None

    def _rows_legacy(self, context=None):
        local = self._local_row_cache_key(context)
        slot = id(context)
        if local is not None:
            cached = self._local_prefill_rows.get(slot)
            if cached is not None and cached[0] == local[0]:
                return cached[1]
        if context is not None:
            chunks = context.chunk_lengths_per_req
            cp_epoch = self._cp_metadata_epoch(context)
            if cp_epoch is not None:
                record = self._cp_row_record(context, epoch=cp_epoch)
                if record is not None:
                    self._cp_metadata_evidence["row_hits"] += 1
                    return record.rows
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
            compact_lookup = self._indexer_lookup_compact_valid(context)
            with torch.inference_mode(False) if compact_lookup or cp_epoch is not None else nullcontext():
                if compact_lookup and self.inputs.input_ids.is_cuda:
                    from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata import _upload

                    # Keep the original independent int32 row buffer and its
                    # normal version counter; only its upload is asynchronous.
                    cu = _upload((0, *accumulate(chunks)), self.inputs.input_ids.device, torch.int32)
                else:
                    cu = torch.tensor(
                        (0, *accumulate(chunks)), dtype=torch.int64 if reuse else torch.int32,
                        device=self.inputs.input_ids.device,
                    )
                requests = context.req_id_per_token.long()
            rows = context.global_positions, requests, cu, context.local_is_real
            self._remember_cp_rows(context, rows, cp_epoch)
            if compact_lookup and self._indexer_lookup_compact_rows is None:
                self._indexer_lookup_compact_rows = context, rows, tuple(_row_tensor_key(value) for value in rows)
            self._remember_sg_rows(context, rows)
            if reuse:
                self._local_cp_rows = context, rows
            if local is not None:
                self._local_prefill_rows[slot] = local[0], rows, local[1], context
            return rows
        device = self.inputs.input_ids.device
        lookup = False
        if self.prefill:
            cu = self.attn.cu_seqlens.to(device=device, dtype=torch.int64)
            # Preserve the original operation/allocation, but retain a version
            # counter when its request IDs will authorize host lookup.
            lookup = self._indexer_lookup_phase()
            if lookup:
                size = sum(self.lengths) if len(self.lengths) > 1 and self._local_host_cu_matches_lengths() else None
                with torch.inference_mode(False), torch.no_grad():
                    requests = torch.repeat_interleave(
                        torch.arange(len(self.lengths), device=device), cu[1:] - cu[:-1], output_size=size,
                    )
                    positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
            else:
                requests = torch.repeat_interleave(torch.arange(len(self.lengths), device=device), cu[1:] - cu[:-1])
                positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
        else:
            sizes = torch.tensor(self.lengths, device=device)
            requests = torch.repeat_interleave(torch.arange(len(self.lengths), device=device), sizes)
            positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
            cu = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
        # Keep this original allocation versioned even under engine inference
        # mode, so an in-place mask write revokes all-live eligibility.
        with torch.inference_mode(False):
            valid = torch.ones_like(positions, dtype=torch.bool)
        with torch.inference_mode(False) if lookup else nullcontext():
            rows = positions.long(), requests.long(), cu.int(), valid
        self._remember_indexer_lookup_rows(rows)
        self._remember_mega_all_live_mask(None, valid)
        self._remember_sg_rows(None, rows)
        if (self._cross_mhc_enabled and optimization_env_value(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
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
        use_host_geometry = optimization_env_value("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") == "1"
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

    def _l20_query_projection_enabled(self):
        from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
            DENSE_PROJECTIONS, enabled,
        )

        return self._mega_all_live_phase() and (
            enabled(DENSE_PROJECTIONS) or (
                self.local_prefill_sg_align
                and os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_DENSE_PROJECTIONS", "0") == "1"
            )
        )

    def local_l20_query_rows(self, rows, *, allow_batch=False):
        """Prove the retained rows against the plan's creation-stamped transport.

        This only changes query consumers. Full L20 KV producers and the
        existing compact-state callback retain their original row epoch.
        """
        proof = self._l20_query_projection_proof
        if (
            not self._l20_query_projection_enabled() or proof is None
            or self.compacted or rows != sum(self.lengths)
            or rows <= 140 * len(self.lengths) or self.runtime.policy.window != 128
            or proof[0] is not self.plan or proof[1] is not self.plan.context
        ):
            return None
        current = self._local_compact_rows_key(self.plan.context, before_l20=True)
        if current is None or current[0] != proof[2]:
            return None
        if not self.local_prefill_sg_align and any(torch.is_inference(value) for value in current[1]):
            # The general fused route requires the versioned metadata created
            # by prepare_plan. Unknown inference-only transport cannot prove
            # that the retained row order has not been mutated.
            return None
        if len(self.lengths) == 1:
            return slice(rows - 140, rows)
        exchange = self.plan.exchange
        if (allow_batch and exchange.receive_is_identity
                and self.plan.context.chunk_lengths_per_req == (140,) * len(self.lengths)
                and min(self.lengths) >= 140
                and not any(torch.is_inference(value) for value in current[1])):
            return exchange.send_indices
        return None

    def use_local_compact_rows(self, context):
        return self._validated_local_compact_rows_key(context) is not None

    def _validated_local_compact_rows_key(self, context):
        if optimization_env_value(LOCAL_PREFILL_COMPACT_ROWS_ENV, "0") != "1":
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
                optimization_env_value(LOCAL_PREFILL_FULL_ROWS_ENV, "0") == "1"
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
        all_live = self.mega_attention_valid_mask(None, valid, rows) is None
        info = SimpleNamespace(
            # indices is a just-created arange, with no intervening consumer.
            prefill_qkv_restore_indice_cpu=torch.arange(rows, dtype=indices.dtype, device="cpu"),
            prefill_qkv_padding_mask_cpu=(
                torch.ones(rows, dtype=torch.bool, device="cpu") if all_live else valid.cpu()
            ),
        )
        host_geometry = (
            self._indexer_lookup_phase() and self._indexer_lookup_original_valid()
            and all(not value.is_inference() for value in self._indexer_lookup_stamp.rows)
        )
        # Keep this existing cast versioned at its real construction point.
        # The host proof must reject a later mutation of the identity request
        # ids independently of their original int64 source.
        with torch.inference_mode(False) if host_geometry else nullcontext():
            identity_requests = requests.int()
        context = CPContext(
            cp_size=1, cp_rank=0, chunk_length=rows, padded_seq_len=rows,
            seq_len_full=rows, relative_positions=indices, prefix_length=self.query_prefixes[0],
            global_positions=positions, local_is_real=valid, unpad_restore=indices,
            seq_len_total=self.query_prefixes[0] + self.lengths[0], cp_info=info,
            req_id_per_token=identity_requests, prefix_lengths=torch.tensor(self.query_prefixes, device=device),
            input_lengths_global=torch.tensor(self.lengths, device=device),
            cu_seqlens_global=torch.tensor([0, *torch.tensor(self.lengths).cumsum(0).tolist()], device=device),
            chunk_lengths_per_req=self.lengths, input_lengths_global_host=self.lengths,
            prefix_lengths_host=self.query_prefixes, first_position_host=self.query_prefixes[0],
        )
        # The identity context aliases the very same freshly certified mask.
        if all_live:
            self._remember_mega_all_live_mask(context, valid)
        if host_geometry and self._compact_validation_original is None:
            owners = context.global_positions, context.req_id_per_token, context.local_is_real
            self._compact_validation_original = (
                context, self._indexer_lookup_stamp, owners,
                tuple(_row_tensor_key(value) for value in owners),
            )
        return context

    def _entry_plan_key(self):
        original, plan = self.original_context, self.plan
        compact = getattr(plan, "context", None)
        exchange = getattr(plan, "exchange", None)
        info = getattr(original, "cp_info", None)
        proof = self._local_compact_rows_key(compact, before_l20=True) if plan is not None else None
        owners = (self.inputs, self.attn, self.cache, self.runtime.model,
                  original, plan, compact, exchange, info)
        tensors = (self.inputs.input_ids, self.attn.cu_seqlens) + tuple(
            getattr(original, field, None) for field in (
                "global_positions", "req_id_per_token", "local_is_real", "unpad_restore",
                "prefix_lengths", "input_lengths_global", "cu_seqlens_global",
            )
        ) + tuple(getattr(info, field, None) for field in (
            "prefill_qkv_restore_indice_cpu", "prefill_qkv_padding_mask_cpu",
        ))
        policy = self.runtime.policy
        key = (
            tuple(id(owner) for owner in owners),
            tuple(_row_tensor_key(value) for value in tensors),
            self.lengths, self.query_prefixes, self.plan_reason,
            tuple(getattr(original, field, None) for field in (
                "cp_size", "cp_rank", "chunk_length", "padded_seq_len", "seq_len_full",
                "kv_cache_sharded", "chunk_lengths_per_req", "input_lengths_global_host", "prefix_lengths_host",
            )),
            (policy.mode, policy.keep_tokens, policy.window,
             policy.max_rollback_tokens, policy.encoder_replay),
            permits_ced(self.inputs), self.local_prefill_sg_align,
            (getattr(self.v4, "_mtp_hidden_buffer", None) is not None,
             tuple(getattr(self.v4, "capture_aux_hidden_layer_ids", ()))),
            None if proof is None else proof[0],
            self._mega_all_live_phase(), self.context is None, self.compacted,
        )
        return key, owners, tensors

    def prepare_local_plan_at_entry(self):
        """Prepare local CP1 metadata before any layer work is queued.

        Do not propagate the compact context: L0--L20 still see the original
        full rows, and the existing publication/compact callback owns L20.
        """
        policy, model = self.runtime.policy, self.runtime.model
        if (
            self._entry_plan_stamp is not None or self.executed_layers
            or optimization_env_value("RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA", "0") != "1"
            or not self._mega_all_live_phase() or self.local_prefill_sg_align
            or getattr(model, "_prefill_cp_size", None) != 1
            or getattr(getattr(model, "parallelism_config", None), "tp_size", None) != 1
            or (policy.mode, policy.keep_tokens, policy.window,
                policy.max_rollback_tokens, policy.encoder_replay) != ("compact", 140, 128, 0, False)
            or self.context is not None or self.original_context is not None
            or not self.lengths or min(self.lengths) < 140 or max(self.lengths) <= 140
            or self.inputs.input_ids.numel() != sum(self.lengths)
        ):
            return
        if len(self.lengths) > 1:
            if not self._indexer_lookup_phase() or not self._local_host_cu_matches_lengths():
                return
            self.rows()
            if not self._indexer_lookup_original_valid():
                return
        self.prepare_plan()
        # Retain source owners as well as their keys. L19 may reuse only this
        # exact creation result, including an unchanged admission rejection.
        self._entry_plan_stamp = self._entry_plan_key()

    def prepare_plan(self):
        if self._entry_plan_stamp is not None:
            if self._entry_plan_key()[0] != self._entry_plan_stamp[0]:
                raise RuntimeError("local Prefill plan metadata changed after entry preparation")
            return
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
        # This call constructs metadata only. Preserve its original mask
        # allocation/reference while retaining version counters for consumers.
        with torch.inference_mode(False), torch.no_grad():
            self.plan, self.plan_reason = bounded_plan(self.v4, context, self.attn, self.cache.native, policy)
        if self.plan is not None:
            compact = self.plan.context
            proof = self._local_compact_rows_key(compact, before_l20=True)
            if (proof is not None and compact.chunk_lengths_per_req
                    == tuple(min(n, 140) for n in self.lengths)):
                # The bounded planner creates exactly these real retained rows;
                # any odd-length padding makes this equality fail.
                self._remember_mega_all_live_mask(compact, compact.local_is_real, plan=self.plan)
                self._remember_indexer_lookup_compact_plan(compact, proof)
        if self.plan is not None and self._l20_query_projection_enabled():
            proof = self._local_compact_rows_key(self.plan.context, before_l20=True)
            self._l20_query_projection_proof = (
                (self.plan, self.plan.context, proof[0], proof[1]) if proof is not None else None
            )
        if self.plan is not None and getattr(self.plan, "_ppu_cp8_query_birth", None) is not None:
            from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import (
                CP8L20ProjectionProof, _cp8_l20_host_domain,
            )

            if _cp8_l20_host_domain(self):
                self._cp_l20_projection_proof = CP8L20ProjectionProof(
                    self, self.plan._ppu_cp8_query_birth,
                )
                self._remember_cp_compact_host_geometry()

    def before_block(self, block, hidden, ids, positions, cu):
        layer = block.layer_id
        self.executed_layers.append(layer)
        if not self.prefill:
            return hidden, ids, positions, cu
        if layer == 0 and hasattr(self, "window_current_tokens"):
            def plain_ids(value):
                return (type(value) is torch.Tensor and value.layout == torch.strided
                        and value.ndim == 1 and value.dtype in (torch.int32, torch.int64)
                        and not value.is_neg() and not value.is_conj())

            unmasked = self.context is None and plain_ids(ids)
            real = None if unmasked else (
                self.context.local_is_real if self.context is not None
                else torch.ones_like(ids, dtype=torch.bool)
            )
            comparison_ids = ids
            image_rows = getattr(self.v4, "_image_token_mask", None)
            if (
                image_rows is not None
                and getattr(self.runtime.model, "_image_plan", None) is not None
            ):
                comparison_ids = self.runtime.model._engram_input_ids(ids, image_rows)
            window = self.window_current_tokens
            cp_all_live = self._cp_window_all_live_rows(ids, window, comparison_ids, real)
            if (unmasked or cp_all_live) and all(
                plain_ids(value) and value.shape == ids.shape and value.device == ids.device
                for value in (window, comparison_ids)
            ):
                # The omitted mask was constructed here as all true. These
                # temporary aliases reach only the original read-only check.
                matches = torch.equal(window.long(), comparison_ids.long())
            else:
                if unmasked:
                    real = torch.ones_like(ids, dtype=torch.bool)
                # Preserve indexing, cast order and exceptions for other layouts.
                matches = torch.equal(window[real].long(), comparison_ids[real].long())
            if not matches:
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
        host_geometry = self._compact_host_geometry()
        cp_host_geometry = self._cp_compact_host_current(require_output=False)
        # The local source indices already encode every restored real row.
        with torch.inference_mode(False) if host_geometry is not None or cp_host_geometry is not None else nullcontext():
            self.valid_output_rows = hidden.new_zeros(exchange.original_rows, dtype=torch.bool)
            self.valid_output_rows.index_fill_(0, exchange.send_indices, True)
        if cp_host_geometry is not None:
            if not cp_host_geometry.output_covered:
                raise RuntimeError("YOCO removed a framework first-token output row")
            self._cp_compact_host_output_birth = (
                self.valid_output_rows, _row_tensor_key(self.valid_output_rows),
                self.compact_ids, _row_tensor_key(self.compact_ids),
            )
            self._cp_compact_host_geometry_evidence["output_coverage"] += 1
        elif host_geometry is not None:
            if not host_geometry[2]:
                raise RuntimeError("YOCO removed a framework first-token output row")
            self._compact_validation_output = self.valid_output_rows, _row_tensor_key(self.valid_output_rows)
        else:
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
        if logging.getLogger().isEnabledFor(logging.INFO):
            logging.info("PPU shared YOCO compact %s", self.evidence())
        if optimization_env_value(LOCAL_PREFILL_COMPACT_ROWS_ENV, "0") == "1":
            proof = self._local_compact_rows_key(self.context)
            self._local_compact_row_proof = (
                (self.context, self.plan, proof[0], proof[1]) if proof is not None else None
            )
            self._local_compact_full_rows = None
        if (self._cross_mhc_enabled and self._cross_mhc_compact_stamp is None
                and optimization_env_value(LOCAL_PREFILL_CROSS_LAYER_MHC_ENV, "0") == "1"
                and self._cross_mhc_full_valid()):
            # A compact epoch is born only here, never at a cache consumer.
            self._cross_mhc_compact_stamp = self._cross_mhc_compact_descriptor()
        if self._cp_full_rows_birth is not None:
            current = self._cp_compact_rows_descriptor()
            if current is not None:
                if self._cp_metadata_compact_birth is None:
                    self._cp_metadata_compact_birth = current
                elif self._cp_metadata_compact_birth[0] != current[0]:
                    self._invalidate_cp_metadata()
                evidence = self._cross_mhc_evidence["cp_row_proof"]
                real = sum(self.plan.exchange.receive_sizes)
                evidence.update(compact_rows=self.context.chunk_length, compact_real_rows_local=real,
                                compact_all_live=real == self.context.chunk_length)

    def publication(self, layer):
        return self.plan.candidate_publication(self.shared) if layer == 20 and self.plan is not None else nullcontext()

    def restore_head(self, reduced):
        # Head reduction consumes compact delayed carry before transport.
        if self.compacted:
            reduced = self.plan.restore_rows(reduced)
            self.restore_aux()
            self.head_restored = True
            if logging.getLogger().isEnabledFor(logging.INFO):
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
        host_geometry = self._compact_host_evidence()
        return {
            "mode": self.runtime.policy.mode, "keep_tokens": self.runtime.policy.keep_tokens,
            "window": self.runtime.policy.window, "max_rollback_tokens": self.runtime.policy.max_rollback_tokens,
            "plan_reason": self.plan_reason, "input_lengths": list(self.lengths),
            "prefix_lengths": list(self.query_prefixes), "compacted": self.compacted,
            "logical_retained_rows": int(context.unpad_restore.numel()) if context is not None else sum(self.lengths),
            "cp_padding_rows": int(context.padded_seq_len - context.unpad_restore.numel()) if context is not None else 0,
            "cp_size": int(context.cp_size) if context is not None else 1,
            "cp_rank": int(context.cp_rank) if context is not None else 0,
            "local_positions": list(host_geometry[0]) if host_geometry is not None else (
                context.global_positions.tolist() if context is not None else self.rows()[0].tolist()
            ),
            "executed_layers": list(self.executed_layers),
            "restored": self.head_restored or (self.exact_context is not None and self.public_returned),
            "output_requirements": {"need_all_logits": getattr(self.inputs, "need_all_logits", None), "need_all_hidden_states": getattr(self.inputs, "need_all_hidden_states", None)},
            "exact_short_fallback_observed": self.exact_short_fallback_observed,
            "exact_compacted": self.exact_context is not None,
            "encoder_append_starts": self.append_starts,
            "local_valid_output_rows": host_geometry[1] if host_geometry is not None else (
                int(self.valid_output_rows.sum()) if self.valid_output_rows is not None else None
            ),
        }


_CP_ROW_SCOPE = RowScope
_CP_ROW_SCOPE_INIT = RowScope.__init__
_CP_ROW_VIEW = RowView
_CP_ROW_VIEW_INIT = RowView.__init__
_CP_ROW_VIEW_AS_ROWS = RowView.as_rows
_CP_CACHED_ATTENTION_METHODS = tuple(
    (name, vars(SharedForwardState)[name])
    for name in (
        "rows", "row_view", "_describe_rows", "_rows_legacy", "_local_row_cache_key",
        "cp_prefill_step_context", "_cp_metadata_entry", "_cp_row_record", "_cp_metadata_epoch",
    )
)
