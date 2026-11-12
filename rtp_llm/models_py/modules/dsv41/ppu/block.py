"""PPU construction of the existing V4.1 Block execution surface."""

import os
from dataclasses import replace

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu.layers import (
    PPUAttentionProjections,
    PPUSharedEngram,
    PPUSharedHCUnit,
    PPUSharedMoE,
    PPUSharedRMSNorm,
    _ordinary_decode_graph,
    cp8_yoco_fast_prefill_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.mhc import _prefer_triton
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.utils.model_weight import W


LOCAL_PREFILL_MEGA_MHC_ENV = "RTP_V41_PPU_LOCAL_PREFILL_MEGA_MHC"
LOCAL_PREFILL_SPLIT_L20_ENV = "RTP_V41_PPU_LOCAL_PREFILL_SPLIT_L20"
LOCAL_PREFILL_ENGRAM_GATE_ENV = "RTP_V41_PPU_LOCAL_PREFILL_ENGRAM_GATE"


def _local_prefill_engram_gate_enabled(state):
    if (
        os.environ.get(LOCAL_PREFILL_ENGRAM_GATE_ENV, "0") != "1"
        or not getattr(state, "prefill", False) or getattr(state, "idle", True)
        or any(getattr(state, name, False) for name in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
        or torch.is_grad_enabled()
    ):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled
    from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache

    runtime = getattr(state, "runtime", None)
    model, cache = getattr(runtime, "model", None), getattr(state, "cache", None)
    context = getattr(state, "context", None)
    return (
        type(cache) is SharedNativeCache and cache._is_prefill and not cache.sharded
        and (context is None or (getattr(context, "cp_size", 0) == 1
                                 and not getattr(context, "kv_cache_sharded", True)))
        and model is not None and not getattr(model, "_is_decode_role", True)
        and not getattr(model, "_is_speculative", False)
        and getattr(model, "_prefill_cp_size", None) == 1
        and getattr(getattr(model, "parallelism_config", None), "tp_size", None) == 1
        and not optimized_diagnostics_enabled() and is_ppu()
        and not torch.cuda.is_current_stream_capturing()
    )


def _local_prefill_mega_mhc_enabled(state):
    """Admit rowwise transitions without changing local CED/cache transport."""
    if (
        os.environ.get(LOCAL_PREFILL_MEGA_MHC_ENV, "0") != "1"
        or state is None
        or not state.prefill
        or state.idle
        or getattr(state, "is_graph", False)
        or getattr(state, "verify", False)
        or torch.is_grad_enabled()
    ):
        return False
    cache = getattr(state, "cache", None)
    context = getattr(state, "context", None)
    runtime = getattr(state, "runtime", None)
    policy = getattr(runtime, "policy", None)
    model = getattr(runtime, "model", None)
    return (
        cache is not None
        and not getattr(cache, "sharded", True)
        and (context is None or getattr(context, "cp_size", 0) == 1)
        and getattr(policy, "mode", None) == "compact"
        and getattr(policy, "keep_tokens", None) == 140
        and getattr(policy, "window", None) == 128
        and getattr(policy, "max_rollback_tokens", None) == 0
        and getattr(policy, "encoder_replay", True) is False
        and model is not None
        and not getattr(model, "_is_speculative", False)
    )


def _local_prefill_split_l20_enabled(state):
    """Reuse the full-producer/compact-query seam for a local bounded plan."""
    if (
        os.environ.get(LOCAL_PREFILL_SPLIT_L20_ENV, "0") != "1"
        or not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or getattr(state, "compacted", True)
        or any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or torch.is_grad_enabled()
    ):
        return False
    runtime = getattr(state, "runtime", None)
    policy, model = getattr(runtime, "policy", None), getattr(runtime, "model", None)
    context, cache = getattr(state, "context", None), getattr(state, "cache", None)
    query = getattr(getattr(state, "plan", None), "context", None)
    return (
        cache is not None and not getattr(cache, "sharded", True)
        and (context is None or (
            getattr(context, "cp_size", 0) == 1
            and not getattr(context, "kv_cache_sharded", True)
            and getattr(context, "gather_restore_positions", None) is None
        ))
        and getattr(query, "cp_size", 0) == 1
        and not getattr(query, "kv_cache_sharded", True)
        and getattr(policy, "mode", None) == "compact"
        and getattr(policy, "keep_tokens", None) == 140
        and getattr(policy, "window", None) == 128
        and getattr(policy, "max_rollback_tokens", None) == 0
        and getattr(policy, "encoder_replay", True) is False
        and model is not None and not getattr(model, "_is_speculative", False)
    )


class SharedPPUBlock(Block):
    """Keep Block methods; replace only construction and the device mHC seam.

    query_only is a real, explicitly partial weight consumer. It exercises
    the same attention-pre and query-latent modules without loading experts,
    Engram or any other layer. It must never be placed in a full Transformer.
    """

    def __init__(self, views, layer_id, ops, *, query_only=False, enpm_group=None):
        # The public constructor directly binds CUDA Attention/MoE/HC modules.
        # Construct only PPU modules here; do not alter its factories/globals.
        nn.Module.__init__(self)
        self.layer_id = int(layer_id)
        self.query_only = bool(query_only)
        self.ops = ops
        self.fp8_kv_cache = True
        self.engram = None
        self.engram_hashes = None
        self.engram_token_mask = None
        self._cp_sync_after_attn_done = False
        self._mhc_phase = "unbound"
        g = views.geometry
        if query_only:
            self.attn = PPUAttentionProjections(views, layer_id, query_only=True)
        else:
            from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import SharedPPUAttention

            self.attn = SharedPPUAttention(views, layer_id, ops)
        self.attn_norm = PPUSharedRMSNorm(
            views.tensor(layer_id, W.v4_attn_norm), g.norm_eps
        )
        self.ffn_norm = PPUSharedRMSNorm(
            views.tensor(layer_id, W.v4_ffn_norm), g.norm_eps
        )
        self.attn_hc = PPUSharedHCUnit(views, layer_id, "attn", ops)
        self.ffn_hc = PPUSharedHCUnit(views, layer_id, "ffn", ops)
        self.ffn_hc.set_previous(self.attn_hc)
        self.ffn = PPUSharedMoE(views, layer_id, ops, routed_enabled=not query_only)
        if not query_only and layer_id in g.engram_layer_ids:
            self.engram = PPUSharedEngram(views, layer_id, ops, enpm_group=enpm_group)
        self._prefill_fast_hc_impls_cached = self._resolve_prefill_fast_hc_impls()

    def forward(self, x, input_ids, positions, cu_seqlens, expert_load=None, **kwargs):
        state = self.attn.runtime.current
        x, input_ids, positions, cu_seqlens = state.before_block(
            self, x, input_ids, positions, cu_seqlens
        )
        begin = getattr(state, "begin_cross_mhc_block", None)
        if begin is not None:
            begin(self, allowed=(kwargs.get("routing_metadata") is None and not (
                set(kwargs) - {"routing_metadata", "kv_cache", "block_tables_by_type"}
            )))
        fast_yoco = cp8_yoco_fast_prefill_enabled(state)
        self.bind_execution_context(
            "prefill", ced=(
                state.runtime.policy.mode != "full"
                and not fast_yoco
                and not _local_prefill_mega_mhc_enabled(state)
            ),
        )
        if self.engram is not None:
            context = state.context
            uniform_rows = (
                fast_yoco and context is not None and context.cp_size == 8
                and context.gather_restore_positions is None
                and os.environ.get("RTP_V41_PPU_CP8_YOCO_ENGRAM_RS", "0") == "1"
            )
            schedule = getattr(state, "engram_round_schedule", None)
            gate = _local_prefill_engram_gate_enabled(state)
            self.engram.prefetch(self.engram_hashes, uniform_rank_rows=uniform_rows,
                                 **({"round_schedule": schedule} if schedule is not None else {}),
                                 **({"local_fused_mask": True} if gate else {}))
        if (
            self.layer_id == 20 and state.plan is not None
            and (
                (fast_yoco and state.cache.sharded
                 and os.environ.get("RTP_V41_PPU_CP8_YOCO_SPLIT_L20", "1") == "1")
                or _local_prefill_split_l20_enabled(state)
            )
            and self.attn.is_kv_source and self.attn.is_index_source
            and input_ids is not None and kwargs.get("routing_metadata") is None
            and not torch.is_grad_enabled()
        ):
            from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled

            if not optimized_diagnostics_enabled():
                return self._forward_compact_l20(x, input_ids, state, expert_load)
        with state.publication(self.layer_id):
            return super().forward(
                x, input_ids, positions, cu_seqlens, expert_load=expert_load, **kwargs
            )

    def _forward_compact_l20(self, x, input_ids, state, expert_load):
        residual = self._inject_engram(x)
        cached = self._try_cached_attn_pre(residual)
        if cached is None:
            hidden, _, _ = self.attn_hc.pre(residual)
            normalized = self.attn_norm(hidden)
        else:
            normalized, _, _ = cached
        exchange = state.plan.exchange

        def compact_after_producer():
            nonlocal residual
            residual = exchange.compact(residual)
            mix = self.attn_hc.state
            compact_mix = replace(
                mix, hidden=None,
                pre_mix=exchange.compact(mix.pre_mix),
                post_mix=exchange.compact(mix.post_mix),
                residual_mix=exchange.compact(mix.residual_mix),
            )
            self.attn_hc.pre_mix_out = compact_mix.pre_mix
            self.attn_hc.state = compact_mix
            state.compact_l20_queries(residual, exchange.compact(input_ids))

        attn_out = self.attn(normalized, compact_after_producer=compact_after_producer)
        mix = self.attn_hc.state
        transition = self._try_mega_mhc(attn_out, residual, mix.post_mix, mix.residual_mix)
        if transition is None:
            residual = self.attn_hc.post(attn_out, residual, mix.post_mix, mix.residual_mix)
            hidden, post, comb = self.ffn_hc.pre(residual)
            hidden = self.ffn_norm(hidden)
        else:
            residual, hidden, post, comb = transition
        ffn_out = self.ffn(hidden, state.compact_ids, expert_load=expert_load)
        return self.ffn_hc.post(ffn_out, residual, post, comb)

    def _try_cached_attn_pre(self, residual):
        state = self.attn.runtime.current
        take = getattr(state, "take_next_attn_mhc", None)
        transition = take(self, residual) if take is not None else None
        if transition is None:
            return None
        return transition.normalized, transition.post_mix, transition.residual_mix

    def forward_decode(self, x, attn_metadata, input_ids, expert_load=None, **kwargs):
        state = self.attn.runtime.current
        if getattr(state, "is_graph", False):
            self.bind_execution_context(
                "decode", is_cuda_graph=True,
                speculative=self.attn.runtime.model._is_speculative,
            )
            if not getattr(state, "bulk_counters", False):
                state.layer_calls[self.layer_id].add_(1)
            return super().forward_decode(
                x, attn_metadata, input_ids, expert_load=expert_load, **kwargs
            )
        state.executed_layers.append(self.layer_id)
        self.bind_execution_context("decode", speculative=state.runtime.model._is_speculative)
        if self.layer_id in state.v4.capture_aux_hidden_layer_ids:
            state.v4.capture_aux_hidden(self.layer_id, x)
        if self.engram is not None:
            schedule = getattr(state, "engram_round_schedule", None)
            self.engram.prefetch(self.engram_hashes,
                                 **({"round_schedule": schedule} if schedule is not None else {}))
        return super().forward_decode(
            x, attn_metadata, input_ids, expert_load=expert_load, **kwargs
        )

    def bind_execution_context(
        self, phase, *, ced=False, speculative=False, is_cuda_graph=False
    ):
        """Report the actual execution phase to the PPU mHC adapter."""
        self._mhc_phase = (
            "dspark" if speculative else
            "graph" if is_cuda_graph else
            "ced" if ced else phase
        )

    def _try_decode_ffn_post(self, hidden, input_ids, expert_load, residual, post, comb):
        runtime = self.attn.runtime
        if (
            not _ordinary_decode_graph(runtime)
            or runtime.current.grouped_moe is not None
            or self.ffn.shared_expert is None
            or torch.is_grad_enabled()
            or residual.shape[1:] != (1, 4, 5120)
            or hidden.shape != (residual.shape[0], 1, 5120)
            or residual.dtype != torch.bfloat16
            or not _prefer_triton(self.ops._mhc.config, residual)
        ):
            return None
        routed, shared = self.ffn(
            hidden, input_ids, expert_load=expert_load, _return_shared=True
        )
        return self.ffn_hc.post_graph_moe(routed, shared, residual, post, comb)

    def _try_mega_mhc(
        self, attn_out, residual, post, comb, *, original_tokens=None, moe_inputs=None
    ):
        state = self.attn_hc.state
        if state is None or post is not state.post_mix or comb is not state.residual_mix:
            raise RuntimeError("Mega-mHC requires this Block's live attention pre state")
        phase = self._mhc_phase
        if original_tokens is not None or moe_inputs is not None:
            phase = "unqualified_metadata"
        original_shape = residual.shape
        decode_view = residual.ndim == 4 and residual.shape[1] == 1
        native_state = replace(
            state, hidden=None,
            pre_mix=state.pre_mix.reshape(-1, state.pre_mix.shape[-1]),
            post_mix=state.post_mix.reshape(-1, *state.post_mix.shape[-2:]),
            residual_mix=state.residual_mix.reshape(-1, *state.residual_mix.shape[-2:]),
        ) if decode_view else state
        transition = self.ops.try_mhc_ffn_transition(
            attn_out.reshape(-1, attn_out.shape[-1]) if decode_view else attn_out,
            residual.reshape(-1, *residual.shape[-2:]) if decode_view else residual,
            native_state,
            self.ffn_hc.parameters_map,
            self.ffn_norm.weight,
            self.ffn_norm.variance_epsilon,
            phase=phase,
        )
        if transition is None:
            return None
        if decode_view:
            transition = replace(
                transition, residual=transition.residual.reshape(original_shape),
                normalized=transition.normalized.reshape(*original_shape[:-2], original_shape[-1]),
                pre_mix=transition.pre_mix.reshape(*original_shape[:-2], original_shape[-2]),
                post_mix=transition.post_mix.reshape(*original_shape[:-2], original_shape[-2], 1),
                residual_mix=transition.residual_mix.reshape(*original_shape[:-2], original_shape[-2], original_shape[-2]),
            )
        self.ffn_hc.accept_transition(transition)
        # Existing Block.forward consumes this normalized tensor directly in
        # the fused arm, skipping ffn_hc.pre and ffn_norm exactly once.
        return (
            transition.residual,
            transition.normalized,
            transition.post_mix,
            transition.residual_mix,
        )
