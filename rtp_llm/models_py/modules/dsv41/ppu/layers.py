"""Device modules bound to the unchanged public Block method interfaces.

Paged Attention orchestration is intentionally an N2 boundary. This file
provides its real projection/normalization inputs, rather than constructing
a CUDA Attention object or presenting a placeholder as AttentionFP8.
"""

import importlib.util
import weakref
from dataclasses import replace

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.models_py.modules.base.common.norm import BaseNorm
from rtp_llm.models_py.modules.dsv41.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv41.ppu.engram import engram_gate_inject, wkv_project
from rtp_llm.models_py.modules.dsv41.ppu.mhc import PPUStepMix
from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    combine_routed_and_shared,
    pad_topk_for_dispatch,
    shared_expert_forward,
    sqrtsoftplus_gate,
)
from rtp_llm.models_py.modules.dsv41.ppu.weight import (
    _CheckpointSource,
    _load_dense,
    _load_engram_shard,
)
from rtp_llm.utils.model_weight import W


def _ordinary_decode_graph(runtime):
    state = runtime.current if runtime is not None else None
    return (
        getattr(state, "is_graph", False)
        and not getattr(state, "verify", False)
        and not runtime.model._is_speculative
    )


def try_graph_rms_norm(hidden_states, weight, eps) -> torch.Tensor | None:
    """Reuse the strided RMSNorm for ordinary Graph BF16 activations."""
    if (
        weight is None
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
        or hidden_states.ndim not in (2, 3)
        or not hidden_states.is_contiguous()
        or weight.dtype not in (torch.bfloat16, torch.float32)
        or weight.ndim != 1
        or not weight.is_contiguous()
        or weight.device != hidden_states.device
        or not 0 < hidden_states.shape[-1] == weight.numel() <= 8192
        or torch.is_grad_enabled()
        or importlib.util.find_spec("triton") is None
    ):
        return None
    from rtp_llm.models_py.modules.dsv41._v41_fused_qkv import strided_q_rmsnorm

    return strided_q_rmsnorm(hidden_states, weight, eps)


class PPUSharedEmbedding(EmbeddingTorch):
    """Keep public V4.1 feature-row injection at the embedding boundary."""

    def bind_owner(self, owner):
        self._owner_ref = weakref.ref(owner)

    def forward(self, input):
        hidden = super().forward(input)
        owner = self._owner_ref() if hasattr(self, "_owner_ref") else None
        return owner._inject_image_rows(hidden, input) if owner is not None else hidden


class PPUSharedRMSNorm(BaseNorm):
    """Stored gamma, no offset transform; one FP32 norm and BF16 landing."""

    def __init__(self, weight, eps=1e-6):
        if weight.ndim != 1 or weight.dtype != torch.bfloat16:
            raise TypeError("shared PPU RMSNorm requires a BF16 gamma vector")
        super().__init__(weight, eps)
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        if _ordinary_decode_graph(getattr(self, "runtime", None)):
            normalized = try_graph_rms_norm(
                hidden_states, self.weight, self.variance_epsilon
            )
            if normalized is not None:
                return normalized
        x = hidden_states.float()
        return (
            x
            * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
            * self.weight.float()
        ).to(hidden_states.dtype)


class PPUSharedHCUnit(HCUnitBase):
    """Inherit public pre/post wrappers; delegate kernels/state to PPUOps."""

    def __init__(self, views, layer_id, which, ops):
        g = views.geometry
        super().__init__(
            *(views.tensor(layer_id, getattr(W, f"v4_hc_{which}_{part}"))
              for part in ("fn", "base", "scale")),
            dim=g.hidden_size,
            hc_mult=g.hc_mult,
            hc_sinkhorn_iters=g.hc_sinkhorn_iters,
            norm_eps=g.norm_eps,
            hc_eps=g.hc_eps,
            layer_id=layer_id,
            name=which,
        )
        if any(t.dtype != torch.float32 for t in (self.fn, self.base, self.scale)):
            raise TypeError("PPU delayed mHC requires FP32 coefficient weights")
        if tuple(self.fn.shape) != (g.hc_mult * (g.hc_mult + 2), g.hc_mult * g.hidden_size):
            raise ValueError("PPU delayed mHC fn shape differs from checkpoint geometry")
        self.base = self.base.reshape(-1)
        self.scale = self.scale.reshape(3)
        self.ops = ops
        self._previous_ref = None
        self.state = None
        self.pre_mix_out = None

    @property
    def pre_mix_out(self):
        return self._ppu_pre_mix_out

    @pre_mix_out.setter
    def pre_mix_out(self, value):
        self._ppu_pre_mix_out = value
        state = self.state
        if state is not None and value is not None:
            # The public exact CED primitive assigns this property. Only the
            # carried pre coefficients survive a completed FFN boundary.
            changed_rows = state.pre_mix is not None and state.pre_mix.shape != value.shape
            self.state = replace(
                state, pre_mix=value,
                **({"hidden": None, "post_mix": None, "residual_mix": None} if changed_rows else {}),
            )

    @property
    def parameters_map(self):
        return {"fn": self.fn, "base": self.base, "scale": self.scale}

    def set_previous(self, previous):
        self._previous_ref = weakref.ref(previous) if previous is not None else None

    def _pre_impl(self, x, dbg_tag=None):
        previous = self._previous_ref() if self._previous_ref is not None else None
        if previous is not None and previous.state is None:
            raise RuntimeError("previous delayed mHC state was not produced")
        state = self.ops.mhc(
            x,
            None,
            phase="pre",
            parameters=self.parameters_map,
            state=previous.state if previous is not None else None,
        )
        # Native pre may flatten token axes; public HC consumers preserve them.
        leading_shape = tuple(x.shape[:-2])
        self.state = replace(
            state,
            pre_mix=state.pre_mix.reshape(*leading_shape, self.hc_mult),
            post_mix=state.post_mix.reshape(*leading_shape, self.hc_mult, 1),
            residual_mix=state.residual_mix.reshape(
                *leading_shape, self.hc_mult, self.hc_mult
            ),
        )
        self.pre_mix_out = self.state.pre_mix
        return self.state.hidden, self.state.post_mix, self.state.residual_mix

    def _post_impl(self, x, residual, post, comb):
        if (
            _ordinary_decode_graph(getattr(self, "runtime", None))
            and residual.is_cuda
            and residual.ndim == 4
            and residual.shape[1] == 1
            and residual.shape[-2:] == (4, 5120)
            and x.dtype == residual.dtype == torch.bfloat16
            and post.dtype == comb.dtype == torch.float32
            and all(t.is_contiguous() for t in (x, residual, post, comb))
        ):
            # The singleton decode axis is a view. The existing flat-token
            # mHC kernel loads BF16 inputs and accumulates in FP32.
            mix = PPUStepMix(
                hidden=None, post_mix=post.squeeze(1), residual_mix=comb.squeeze(1)
            )
            return self.ops.mhc(
                residual.squeeze(1), x.squeeze(1), phase="post", state=mix
            ).hidden.unsqueeze(1)
        return self.ops.mhc(
            residual,
            x,
            phase="post",
            state=PPUStepMix(
                hidden=None, post_mix=post, residual_mix=comb, pre_mix=self.pre_mix_out
            ),
        ).hidden

    def accept_transition(self, transition):
        # The native transition has already normalized its separate readout.
        # Keep only unnormalized mHC state here, never re-norm that output.
        self.pre_mix_out = transition.pre_mix
        self.state = PPUStepMix(
            hidden=None,
            pre_mix=transition.pre_mix,
            post_mix=transition.post_mix,
            residual_mix=transition.residual_mix,
        )

    def post_graph_moe(self, routed, shared, residual, post, comb):
        if (
            routed.shape != shared.shape
            or routed.dtype != torch.bfloat16
            or shared.dtype != torch.bfloat16
            or post.dtype != torch.float32
            or comb.dtype != torch.float32
            or not all(
                t.is_contiguous() and t.device == residual.device
                for t in (routed, shared, residual, post, comb)
            )
        ):
            return self.post(
                combine_routed_and_shared(routed, shared), residual, post, comb
            )
        from rtp_llm.models_py.modules.dsv41.ppu._mhc_triton import mhc_post

        with record_function_range(f"dsv4.hc.L{self.layer_id:02d}.{self.name}.post"):
            return mhc_post(
                routed.squeeze(1),
                residual.squeeze(1),
                post.squeeze(1).squeeze(-1),
                comb.squeeze(1),
                shared=shared.squeeze(1),
            ).unsqueeze(1)


class PPUSharedHCHead(nn.Module):
    def __init__(self, previous, ops):
        super().__init__()
        self._previous_ref = weakref.ref(previous)
        self.ops = ops

    def head(self, x):
        return self.ops.mhc(x, None, phase="head", state=self._previous_ref().state).hidden


class PPUAttentionProjections(nn.Module):
    """Weight-bearing PPU input to N2's paged Attention adapter.

    query_only binds the exact same Wq_a/q_norm consumer for the narrow N1
    calculation. It is explicitly partial and cannot execute Attention.
    """

    def __init__(self, views, layer_id, *, query_only=False):
        super().__init__()
        self.layer_id = layer_id
        self.query_only = bool(query_only)
        self.wq_a = views.linear(layer_id, W.v4_attn_wq_a_w)
        g = views.geometry
        if tuple(self.wq_a.weight.shape) != (g.q_lora_rank, g.hidden_size):
            raise ValueError("Wq_a shape differs from checkpoint query geometry")
        self.q_norm = PPUSharedRMSNorm(
            views.tensor(layer_id, W.v4_attn_q_norm), views.geometry.norm_eps
        )
        self._cp_ctx = None
        if query_only:
            return
        self.v41_config = views.config
        self.wq_b = views.linear(layer_id, W.v4_attn_wq_b_w)
        self.wkv = views.linear(layer_id, W.v4_attn_wkv_w)
        self.wo_a = views.linear(layer_id, W.v4_attn_wo_a_w)
        self.wo_b = views.linear(layer_id, W.v4_attn_wo_b_w)
        self.kv_norm = PPUSharedRMSNorm(
            views.tensor(layer_id, W.v4_attn_kv_norm), views.geometry.norm_eps
        )
        self.attn_sink = views.tensor(layer_id, W.v4_attn_sink)
        g = views.geometry
        self.compress_ratio = g.ratio(layer_id)
        self.is_kv_source = layer_id in g.kv_source_layer_ids
        self.is_index_source = layer_id in g.index_source_layer_ids
        self.kv_source_layer_id = g.kv_source_for(layer_id)
        self.index_source_layer_id = g.index_source_for(layer_id)
        if self.is_kv_source:
            dtype = torch.float32 if self.compress_ratio == 2 else torch.bfloat16
            self.global_wkv = views.linear(
                layer_id, W.v4_compressor_wkv, output_dtype=dtype
            )
            self.global_norm = views.tensor(layer_id, W.v4_compressor_norm)
            self.global_wgate = (
                views.linear(layer_id, W.v4_compressor_wgate, output_dtype=torch.float32)
                if self.compress_ratio == 2 else None
            )
            self.index_wk = views.linear(layer_id, W.v41_indexer_wk)
            self.index_k_norm = views.tensor(layer_id, W.v41_indexer_k_norm)
        if self.is_index_source:
            self.index_wq = views.linear(layer_id, W.v4_indexer_wq_b_w)
            self.index_weights = views.tensor(layer_id, W.v4_indexer_weights_proj_w)

    def query_latent(self, normalized_hidden):
        return self.q_norm(self.wq_a(normalized_hidden))

    def set_cp_ctx(self, cp_ctx):
        self._cp_ctx = cp_ctx

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "N2 must bind PPU paged Attention to the public Prefill/CED/KV flow; "
            "N1 supplies real projection weights only"
        )


class PPUSharedGate(nn.Module):
    def __init__(self, views, layer_id):
        super().__init__()
        self.weight = views.tensor(layer_id, W.v4_router_w)
        self.bias = views.tensor(layer_id, W.v4_router_bias)
        self.top_k = views.geometry.n_activated_experts
        self.route_scale = views.geometry.routed_scaling_factor

    def forward(self, hidden, *, graph_routes=None):
        return sqrtsoftplus_gate(
            hidden, self.weight, self.bias,
            top_k=self.top_k, routed_scaling_factor=self.route_scale,
            graph_routes=graph_routes,
        )


class PPUSharedExpert(nn.Module):
    """The actual shared FFN consumer used by full MoE and the local smoke."""

    def __init__(self, views, layer_id):
        super().__init__()
        self.w13 = views.linear(layer_id, W.v4_shared_w13_w)
        self.w2 = views.linear(layer_id, W.v4_shared_w2_w)
        self.swiglu_limit = views.geometry.swiglu_limit

    def forward(self, hidden, *, fused_activation=False):
        return shared_expert_forward(
            hidden, self.w13, self.w2, self.swiglu_limit,
            fused_activation=fused_activation,
        )


class PPUSharedMoE(nn.Module):
    def __init__(self, views, layer_id, ops, *, routed_enabled=True):
        super().__init__()
        self.routed_enabled = bool(routed_enabled)
        self.layer_id = layer_id
        self.ops = ops
        self.shared_expert = (
            PPUSharedExpert(views, layer_id) if views.geometry.n_shared_experts else None
        )
        self.gate = PPUSharedGate(views, layer_id) if routed_enabled else None
        self.calls = 0
        self.eplb_layer = None
        if routed_enabled:
            ops.bind_expert_weights(layer_id, *views.routed(layer_id))
            mapping, counts = views.eplb_maps(layer_id)
            if mapping is not None:
                from rtp_llm.models_py.modules.dsv41.ppu.eplb import bind_layer

                self.eplb_layer = bind_layer(
                    ops.moe_engine, layer_id, mapping, counts,
                    raw_tensors=views.eplb_raw_tensors(layer_id),
                )

    def forward(
        self,
        hidden,
        input_ids=None,
        expert_load=None,
        routing_metadata=None,
        *,
        _return_shared=False,
    ):
        self.calls += 1
        if not self.routed_enabled:
            raise RuntimeError("partial N1 consumer cannot execute full routed MoE")
        if routing_metadata is not None:
            raise NotImplementedError("shared PPU does not consume external routing metadata")
        shape = hidden.shape
        flat = hidden.reshape(-1, shape[-1])
        runtime = getattr(self, "runtime", None)
        state = runtime.current if runtime is not None else None
        if getattr(state, "is_graph", False):
            real = state.metadata.valid_mask
            routed_rows = state.routed_rows_device[self.layer_id]
            ordinary = _ordinary_decode_graph(runtime)
            if ordinary and state.grouped_moe is None:
                ids, weights = self.gate(flat, graph_routes=(real, routed_rows))
            else:
                ids, weights = pad_topk_for_dispatch(*self.gate(flat))
                ids = torch.where(real[:, None], ids, -1)
                weights = torch.where(real[:, None], weights, 0)
                routed_rows.copy_((ids >= 0).any(-1).sum())
            if not getattr(state, "bulk_counters", False):
                state.moe_calls[self.layer_id].add_(1)
            engine = self.ops.moe_engine
            if state.grouped_moe is None:
                routed = engine.forward_shared_graph(
                    flat,
                    ids,
                    weights,
                    layer_id=self.layer_id,
                    expert_load=expert_load,
                    fused_expert_load=ordinary,
                )
            else:
                physical = engine._map_expert_ids(ids, weights, self.layer_id)
                routed = state.grouped_moe.forward(
                    flat, physical, weights, layer_id=self.layer_id, expert_load=expert_load
                )
            if self.shared_expert is not None:
                shared = self.shared_expert(flat, fused_activation=ordinary)
                if _return_shared:
                    return routed.reshape(shape), shared.reshape(shape)
                routed = combine_routed_and_shared(routed, shared)
            return routed.reshape(shape)
        ids, weights = pad_topk_for_dispatch(*self.gate(flat))
        if state is not None:
            if state.idle:
                real = torch.zeros(flat.shape[0], dtype=torch.bool, device=flat.device)
            elif state.prefill and state.context is not None:
                real = state.context.local_is_real
            elif getattr(state, "active_token_mask", None) is not None:
                real = state.active_token_mask
            else:
                real = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
            ids = torch.where(real[:, None], ids, torch.full_like(ids, -1))
            weights = torch.where(real[:, None], weights, torch.zeros_like(weights))
            state.routed_rows[self.layer_id] = int((ids >= 0).any(-1).sum())
        routed = self.ops.moe(
            flat, ids, weights, layer_id=self.layer_id, expert_load=expert_load
        )
        if self.shared_expert is None:
            return routed.reshape(shape)
        return combine_routed_and_shared(routed, self.shared_expert(flat)).reshape(shape)


class PPUSharedEngram(nn.Module):
    """Use the existing rank-owned host shard loader, never generic GPU tables."""

    def __init__(self, views, layer_id, ops, *, enpm_group):
        super().__init__()
        self.layer_id = layer_id
        self.ops = ops
        source = _CheckpointSource(views.config["checkpoint_path"])
        shard, layout = _load_engram_shard(
            source, layer_id, views.geometry, views.ep_rank, views.ep_size
        )
        ops.bind_engram_shard(layer_id, shard, layout, group=enpm_group)
        prefix = f"layers.{layer_id}.engram"
        device = views.tensor(layer_id, W.v4_attn_norm).device
        self.wkv = _load_dense(source, prefix + ".wkv")
        self.wkv.weight = self.wkv.weight.to(device)
        self.q_weight = source.get(prefix + ".q_weight").to(device, torch.bfloat16)
        self.k_weight = source.get(prefix + ".k_weight").to(device, torch.bfloat16)
        self._prefetched = None

    def prefetch(self, hashes):
        if hashes is None:
            raise ValueError("shared Engram requires scheduler token-window hashes")
        if self._prefetched is not None:
            raise RuntimeError("unconsumed Engram prefetch from an earlier call")
        self._prefetched = (hashes, self.ops.engram_lookup(hashes, layer_id=self.layer_id))

    def clear(self):
        self._prefetched = None

    def forward(self, hidden, hashes, token_mask=None):
        if hashes is None:
            raise RuntimeError("N2 must supply original-row Engram hashes")
        state = getattr(self, "_graph_state", None)
        if state is not None:
            fused_mask = getattr(state, "bulk_counters", False)
            if not fused_mask:
                state.engram_calls[self.layer_id].add_(1)
            embedded = state.engram.lookup(
                hashes, layer_id=self.layer_id, valid_mask=state.metadata.valid_mask
            )
            value = wkv_project(embedded.flatten(1), self.wkv.weight)
            return engram_gate_inject(
                hidden, self.q_weight, self.k_weight, value,
                token_mask=token_mask, fused_mask=fused_mask,
            )
        pending, self._prefetched = self._prefetched, None
        if pending is not None and pending[0] is not hashes:
            raise RuntimeError("Engram prefetch belongs to different token rows")
        embedded = pending[1] if pending is not None else self.ops.engram_lookup(hashes, layer_id=self.layer_id)
        value = wkv_project(embedded.flatten(1), self.wkv.weight)
        return engram_gate_inject(
            hidden, self.q_weight, self.k_weight, value, token_mask=token_mask
        )
