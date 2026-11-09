"""Per-forward PPU row/state adapter around the unchanged public executor."""

import logging
import os
from contextlib import nullcontext
from dataclasses import fields, replace
from itertools import accumulate
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu.ced import bounded_plan
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    CP8_ROW_PREP_ENV,
    SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.prefill.ced import permits_ced
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _MIN_TOKENS


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
        self.original_context = self.context = None
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

    @property
    def v4(self):
        return self.runtime.model.v4

    def bind_context(self, context):
        if self.original_context is None and context is not None:
            self.original_context = context
            for layer in self.v4.layers:
                if layer.engram is not None and layer.engram_token_mask is not None:
                    layer.engram_token_mask = layer.engram_token_mask & context.local_is_real
        self.context = context
        if isinstance(self.cache, SharedNativeCache):
            full_cp8 = (
                self.prefill
                and context is not None
                and context.cp_size == 8
                and context.kv_cache_sharded
                and context.gather_restore_positions is None
            )
            self.cache.bind_swa_fresh_lengths(self.lengths if full_cp8 else None)
        if context is not None and context.gather_restore_positions is not None and self.runtime.policy.mode == "exact":
            self.exact_context = context
            self.plan_reason = "public_exact_compaction"

    def rows(self, context=None):
        if context is not None:
            chunks = context.chunk_lengths_per_req
            reuse = (
                self.prefill
                and context.cp_size == 8
                and context.kv_cache_sharded
                and context.gather_restore_positions is None
                and os.environ.get("RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS", "0") == "1"
            )
            if reuse and self._local_cp_rows is not None and self._local_cp_rows[0] is context:
                return self._local_cp_rows[1]
            cu = torch.tensor(
                (0, *accumulate(chunks)), dtype=torch.int64 if reuse else torch.int32,
                device=self.inputs.input_ids.device,
            )
            rows = context.global_positions, context.req_id_per_token.long(), cu, context.local_is_real
            if reuse:
                self._local_cp_rows = context, rows
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
        return positions.long(), requests.long(), cu.int(), torch.ones_like(positions, dtype=torch.bool)

    def full_rows(self, context):
        if context is None:
            p, r, _, valid = self.rows()
            return p, r, valid
        device = context.global_positions.device
        selected = context.gather_restore_positions
        reuse = (
            selected is None
            and self.prefill
            and context.cp_size == 8
            and context.kv_cache_sharded
            and os.environ.get(CP8_ROW_PREP_ENV, "1") == "1"
        )
        if reuse:
            key = self.query_prefixes, self.lengths, device
            if self._restored_rows is not None and self._restored_rows[0] == key:
                return self._restored_rows[1], self._restored_rows[2], slice(None)

        positions = torch.cat([torch.arange(p, p + n, device=device) for p, n in zip(self.query_prefixes, self.lengths)])
        requests = torch.repeat_interleave(
            torch.arange(len(self.lengths), device=device),
            torch.tensor(self.lengths, device=device),
            output_size=positions.numel() if reuse else None,
        )
        if reuse:
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
        if layer == 21 and self.plan is not None:
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
            self.context = self.plan.context
            self.v4._propagate_cp_ctx(self.context)
            self.compacted = True
            mask = hidden.new_ones((hidden.shape[0], 1), dtype=torch.bool)
            self.valid_output_rows = self.plan.restore_rows(mask).flatten()
            original = self.original_context
            ends = torch.tensor([p + n - 1 for p, n in zip(self.query_prefixes, self.lengths)], device=original.global_positions.device)
            last = original.local_is_real & (original.global_positions == ends[original.req_id_per_token.long()])
            if not bool(self.valid_output_rows[last].all()):
                raise RuntimeError("YOCO removed a framework first-token output row")
            logging.info("PPU shared YOCO compact %s", self.evidence())
        if self.compacted:
            return hidden, self.compact_ids, self.context.global_positions, self.plan.cu_seqlens
        return hidden, ids, positions, cu

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
