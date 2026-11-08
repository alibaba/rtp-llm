"""PPU construction of the existing V4.1 Block execution surface."""

from dataclasses import replace

from torch import nn

from rtp_llm.models_py.modules.dsv41.block import Block
from rtp_llm.models_py.modules.dsv41.ppu.shared_modules import (
    PPUAttentionProjections,
    PPUSharedEngram,
    PPUSharedHCUnit,
    PPUSharedMoE,
    PPUSharedRMSNorm,
)
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
            from rtp_llm.models_py.modules.dsv41.ppu.shared_attention import SharedPPUAttention

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
        self.bind_execution_context("prefill", ced=state.runtime.policy.mode != "full")
        if self.engram is not None:
            self.engram.prefetch(self.engram_hashes)
        with state.publication(self.layer_id):
            return super().forward(
                x, input_ids, positions, cu_seqlens, expert_load=expert_load, **kwargs
            )

    def forward_decode(self, x, attn_metadata, input_ids, expert_load=None, **kwargs):
        state = self.attn.runtime.current
        if getattr(state, "is_graph", False):
            self.bind_execution_context("decode", is_cuda_graph=True)
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
        """N2+ must name actual execution, without broadening Mega eligibility."""
        self._mhc_phase = (
            "graph" if is_cuda_graph else
            "dspark" if speculative else
            "ced" if ced else phase
        )

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
