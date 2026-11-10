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
from rtp_llm.utils.model_weight import W


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
        fast_yoco = cp8_yoco_fast_prefill_enabled(state)
        self.bind_execution_context(
            "prefill", ced=state.runtime.policy.mode != "full" and not fast_yoco
        )
        if self.engram is not None:
            context = state.context
            uniform_rows = (
                fast_yoco and context is not None and context.cp_size == 8
                and context.gather_restore_positions is None
                and os.environ.get("RTP_V41_PPU_CP8_YOCO_ENGRAM_RS", "0") == "1"
            )
            self.engram.prefetch(self.engram_hashes, uniform_rank_rows=uniform_rows)
        if (
            self.layer_id == 20 and fast_yoco and state.plan is not None and state.cache.sharded
            and os.environ.get("RTP_V41_PPU_CP8_YOCO_SPLIT_L20", "1") == "1"
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
        hidden, _, _ = self.attn_hc.pre(residual)
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

        attn_out = self.attn(
            self.attn_norm(hidden), compact_after_producer=compact_after_producer
        )
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
            self.engram.prefetch(self.engram_hashes)
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
