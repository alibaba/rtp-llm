"""Device modules bound to the unchanged public Block method interfaces.

Paged Attention orchestration is intentionally an N2 boundary. This file
provides its real projection/normalization inputs, rather than constructing
a CUDA Attention object or presenting a placeholder as AttentionFP8.
"""

import os
import weakref
from dataclasses import replace

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41._profiler import record_function_range
from rtp_llm.models_py.modules.base.common.embedding import EmbeddingTorch
from rtp_llm.models_py.modules.base.common.norm import BaseNorm
from rtp_llm.models_py.modules.dsv41.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.engram import engram_gate_inject, wkv_project
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUPTPCLinear,
    prepare_shared_ptpc_input,
)
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
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    DIRECT_MOE_ROUTES,
    enabled as optimization_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.quant_mode import MXFP4_FP8_PTPC, quant_mode
from rtp_llm.models_py.modules.dsv41.ppu.rmsnorm import rms_norm, try_rms_norm_quant
from rtp_llm.utils.model_weight import W


def _ordinary_decode_graph(runtime):
    state = runtime.current if runtime is not None else None
    return (
        getattr(state, "is_graph", False)
        and not getattr(state, "verify", False)
        and not runtime.model._is_speculative
    )


def _verify_packed_attention_graph(state):
    return (
        getattr(state, "is_graph", False)
        and getattr(state, "verify", False)
        and getattr(state, "packed_attention", None) is not None
    )


def _proposal_exact_ops(state):
    return (
        getattr(state, "is_graph", False)
        and getattr(state, "proposal", False)
        and state.grouped_moe is None
        and os.environ.get("RTP_V41_PPU_PROPOSAL_EXACT_OPS", "1") == "1"
    )


def _full_cp8_prefill_context(state):
    cp = state.context if getattr(state, "prefill", False) else None
    if (
        cp is not None
        and not state.idle
        and cp.cp_size == 8
        and cp.kv_cache_sharded
        and cp.gather_restore_positions is None
    ):
        return cp
    return None


def cp8_yoco_operator_eligible(state):
    """Existing compact CP8 operator domain, independent of rollout profiles."""
    policy = getattr(getattr(state, "runtime", None), "policy", None)
    return (
        getattr(state, "prefill", False)
        and not state.idle
        and getattr(policy, "mode", None) == "compact"
        and (
            _full_cp8_prefill_context(state) is not None
            or state.use_compact_cp8_ops(state.context)
        )
    )


def cp8_yoco_fast_prefill_enabled(state):
    """Retained framework/communication profile; never an operator selector."""
    return (
        os.environ.get("RTP_V41_PPU_CP8_YOCO_FAST_PREFILL", "0") == "1"
        and cp8_yoco_operator_eligible(state)
    )


LOCAL_PREFILL_AIU_PRENORM_ENV = "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM"


def _local_prefill_business_phase(state):
    """Metadata-only qualification shared by local Prefill consumers."""
    if (not getattr(state, "prefill", False) or getattr(state, "idle", True)
            or torch.is_grad_enabled()
            or any(getattr(state, flag, False) for flag in (
                "is_graph", "is_graph_commit", "verify", "proposal",
            ))
            or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
            or len(getattr(state, "lengths", ())) != 1):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import SharedNativeCache
    from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled
    from rtp_llm.models_py.modules.dsv4 import _record_tensor, _forward_tensor_debug

    cache, cp = getattr(state, "cache", None), getattr(state, "context", None)
    model = getattr(getattr(state, "runtime", None), "model", None)
    return (
        type(cache) is SharedNativeCache and cache._is_prefill and not cache.sharded
        and (cp is None or (getattr(cp, "cp_size", 0) == 1
                           and not getattr(cp, "kv_cache_sharded", True)))
        and model is not None and not getattr(model, "_is_decode_role", True)
        and not getattr(model, "_is_speculative", False)
        and getattr(model, "_prefill_cp_size", None) == 1
        and getattr(getattr(model, "parallelism_config", None), "tp_size", None) == 1
        and not _record_tensor.ENABLED and not _forward_tensor_debug.enabled()
        and not optimized_diagnostics_enabled() and is_ppu()
        and not torch.cuda.is_current_stream_capturing()
    )


def _local_prefill_aiu_prenorm_enabled(state, residual):
    """Admit the existing rowwise AIU projection without a CP8 context."""
    if (
        not optimization_enabled(LOCAL_PREFILL_AIU_PRENORM_ENV)
        or not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or torch.is_grad_enabled()
    ):
        return False
    cache = getattr(state, "cache", None)
    context = getattr(state, "context", None)
    model = getattr(getattr(state, "runtime", None), "model", None)
    if (
        cache is None or getattr(cache, "sharded", True)
        or (context is not None and (
            getattr(context, "cp_size", 0) != 1
            or getattr(context, "kv_cache_sharded", True)
        ))
        or model is None or getattr(model, "_is_speculative", False)
        or residual.ndim != 3 or residual.shape[1:] != (4, 5120)
        or not 1024 <= residual.shape[0] <= 16384
        or residual.dtype != torch.bfloat16 or residual.device.type != "cuda"
        or not residual.is_contiguous()
    ):
        return False
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    return (
        is_ppu()
        and torch.cuda.get_device_capability(residual.device) == (8, 9)
        and not torch.cuda.is_current_stream_capturing()
    )


LOCAL_PREFILL_DIRECT_MOE_ROUTES_ENV = "RTP_V41_PPU_LOCAL_PREFILL_DIRECT_MOE_ROUTES"


def _local_prefill_direct_moe_real(state, flat):
    """Return the current local layout's mask for the existing direct gate."""
    if (
        not optimization_enabled(LOCAL_PREFILL_DIRECT_MOE_ROUTES_ENV)
        or not getattr(state, "prefill", False)
        or getattr(state, "idle", True)
        or any(getattr(state, flag, False) for flag in (
            "is_graph", "is_graph_commit", "verify", "proposal",
        ))
        or getattr(getattr(state, "attn", None), "is_cuda_graph", False)
        or torch.is_grad_enabled()
    ):
        return None
    cache = getattr(state, "cache", None)
    context = getattr(state, "context", None)
    model = getattr(getattr(state, "runtime", None), "model", None)
    if (
        cache is None or getattr(cache, "sharded", True)
        or (context is not None and (
            getattr(context, "cp_size", 0) != 1
            or getattr(context, "kv_cache_sharded", True)
        ))
        or model is None or getattr(model, "_is_speculative", False)
        or getattr(model, "_is_decode_role", False)
        or flat.ndim != 2 or flat.shape[1] != 5120
        or flat.dtype != torch.bfloat16 or flat.device.type != "cuda"
        or not flat.is_contiguous()
    ):
        return None
    from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu

    if (
        not is_ppu()
        or torch.cuda.get_device_capability(flat.device) != (8, 9)
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    # Match the legacy mask precedence, including compact CP1 padding. An
    # unpadded local layout without an active mask consists entirely of real
    # rows. Keep this call-local so layout transitions cannot reuse old masks.
    real = (
        getattr(context, "local_is_real", None) if context is not None
        else getattr(state, "active_token_mask", None)
    )
    if real is None:
        if context is not None:
            return None
        return torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
    if (
        not isinstance(real, torch.Tensor) or real.shape != (flat.shape[0],)
        or real.dtype != torch.bool or real.device != flat.device
        or not real.is_contiguous()
    ):
        return None
    return real


class PPUSharedEmbedding(EmbeddingTorch):
    """Keep public V4.1 feature-row injection at the embedding boundary."""

    def bind_owner(self, owner):
        self._owner_ref = weakref.ref(owner)

    def forward(self, input):
        hidden = super().forward(input)
        owner = self._owner_ref() if hasattr(self, "_owner_ref") else None
        return owner._inject_image_rows(hidden, input) if owner is not None else hidden


class PPUSharedRMSNorm(BaseNorm):
    """Standalone norm policy shared by Prefill, Decode and speculative calls."""

    def __init__(self, weight, eps=1e-6):
        if weight.ndim != 1 or weight.dtype != torch.bfloat16:
            raise TypeError("shared PPU RMSNorm requires a BF16 gamma vector")
        super().__init__(weight, eps)
        self.calls = 0
        self._ptpc = quant_mode() == MXFP4_FP8_PTPC

    def forward_with_ptpc_input(self, hidden_states, *, allow_strided=False):
        """Prefer a fused producer only when an explicit PTPC consumer asks."""
        if self._ptpc:
            result = try_rms_norm_quant(
                hidden_states, self.weight, self.variance_epsilon,
            )
            if result is not None:
                self.calls += 1
                normalized, payload, scale = result
                return normalized, (payload, scale)
        return self(hidden_states, allow_strided=allow_strided), None

    def forward(self, hidden_states, *, allow_strided=False):
        # Retain the QKV keyword ABI; actual strides select the provider.
        self.calls += 1
        return rms_norm(hidden_states, self.weight, self.variance_epsilon)


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
        previous_state = previous.state if previous is not None else None
        reduce_block_m = getattr(self, "_graph_mhc_reduce_block_m", None)
        state = getattr(getattr(self, "runtime", None), "current", None)
        aiu_prenorm = (
            self.name == "attn"
            and optimization_enabled("RTP_V41_PPU_CP8_YOCO_AIU_PRENORM")
            and cp8_yoco_operator_eligible(state)
            and _full_cp8_prefill_context(state) is not None
            and not torch.is_grad_enabled()
            and not torch.cuda.is_current_stream_capturing()
        )
        aiu_prenorm = aiu_prenorm or (
            self.name == "attn" and _local_prefill_aiu_prenorm_enabled(state, x)
        )
        if reduce_block_m is None and not aiu_prenorm:
            state = self.ops.mhc(
                x, None, phase="pre", parameters=self.parameters_map, state=previous_state,
            )
        else:
            state = self.ops._mhc.pre(
                x, self.parameters_map, state=previous_state,
                reduce_block_m=reduce_block_m, aiu_prenorm=aiu_prenorm,
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
        current = getattr(getattr(self, "runtime", None), "current", None)
        if self.name == "ffn" and current is not None:
            produce = getattr(current, "try_ffn_to_next_attn", None)
            updated = produce(self, x, residual, post, comb) if produce is not None else None
            if updated is not None:
                return updated
        graph_post = getattr(self, "_graph_mhc_post", None)
        if graph_post is not None:
            from rtp_llm.models_py.modules.dsv41.ppu._mhc_triton import try_verify_mhc_post

            mode, proposal_q3 = graph_post
            updated = try_verify_mhc_post(
                x, residual, post, comb, mode=mode, proposal_q3=proposal_q3,
            )
            if updated is not None:
                return updated
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
        self._local_prefill_qkv = None
        if os.environ.get("RTP_V41_PPU_LOCAL_PREFILL_DENSE_PROJECTIONS", "0") == "1":
            from rtp_llm.models_py.modules.dsv41.ppu.local_prefill_qkv import PreparedQKV

            self._local_prefill_qkv = PreparedQKV.try_prepare(self.wq_a, self.wkv)
        self.wo_a = views.linear(layer_id, W.v4_attn_wo_a_w)
        self.wo_b = views.linear(layer_id, W.v4_attn_wo_b_w)
        if hasattr(self.wo_a, "prepare_grouped"):
            self.wo_a.prepare_grouped(g.o_groups, g.o_lora_rank)
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

    def project_query_kv(
        self, x, *, prepare_indexer=False, reuse_ptpc=False, fuse_qr_norm=False,
        fused_qkv=False, query_rows=None,
    ):
        """Keep projection order while sharing identical PTPC input rows.

        The optional pairs are local to this invocation. Graph capture records
        their producers before every consumer; replay recomputes their values.
        """
        if query_rows is not None and not (
            isinstance(query_rows, slice) and query_rows.step in (None, 1)
            and type(query_rows.start) is int and type(query_rows.stop) is int
            and 0 <= query_rows.start < query_rows.stop == x.shape[0]
            and x.ndim == 2
        ):
            raise ValueError("query projection requires an admitted contiguous row suffix")
        prepared = getattr(self, "_local_prefill_qkv", None) if fused_qkv else None
        joined = prepared.project(x) if prepared is not None else None
        x_pair = (
            prepare_shared_ptpc_input(x, (self.wq_a, self.wkv))
            if reuse_ptpc and joined is None else None
        )
        raw_qr = joined[0] if joined is not None else (
            self.wq_a(x) if x_pair is None else self.wq_a.forward_quantized(*x_pair)
        )
        if query_rows is not None:
            raw_qr = raw_qr[query_rows]
        qr_consumers = (self.wq_b, self.index_wq) if prepare_indexer else (self.wq_b,)
        can_prepare_qr = (
            fuse_qr_norm and joined is None
            and reuse_ptpc
            and raw_qr.shape[-1] == 1280
            and all(
                isinstance(op, PPUPTPCLinear)
                and op.weight.device == raw_qr.device
                and op.weight.shape[1] == raw_qr.shape[-1]
                for op in qr_consumers
            )
        )
        if can_prepare_qr:
            qr, qr_pair = self.q_norm.forward_with_ptpc_input(raw_qr)
        else:
            qr = (
                self.q_norm(raw_qr, allow_strided=True)
                if joined is not None else self.q_norm(raw_qr)
            )
            qr_pair = None
        if qr_pair is None and reuse_ptpc and prepare_indexer:
            qr_pair = prepare_shared_ptpc_input(qr, qr_consumers)
        q = self.wq_b(qr) if qr_pair is None else self.wq_b.forward_quantized(*qr_pair)
        raw_kv = joined[1] if joined is not None else (
            self.wkv(x) if x_pair is None else self.wkv.forward_quantized(*x_pair)
        )
        kv = (
            self.kv_norm(raw_kv, allow_strided=True)
            if joined is not None else self.kv_norm(raw_kv)
        )
        self._last_local_prefill_qkv_joined = joined is not None
        return qr, q, kv, qr_pair

    def project_indexer_query(self, qr, *, quantized_input=None):
        if quantized_input is None:
            return self.index_wq(qr)
        return self.index_wq.forward_quantized(*quantized_input)

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

    def forward(self, hidden, *, graph_routes=None, allow_top3_graph_routes=False):
        return sqrtsoftplus_gate(
            hidden, self.weight, self.bias,
            top_k=self.top_k, routed_scaling_factor=self.route_scale,
            graph_routes=graph_routes,
            allow_top3_graph_routes=allow_top3_graph_routes,
        )


class PPUSharedExpert(nn.Module):
    """The actual shared FFN consumer used by full MoE and the local smoke."""

    def __init__(self, views, layer_id):
        super().__init__()
        self.w13 = views.linear(layer_id, W.v4_shared_w13_w)
        self.w2 = views.linear(layer_id, W.v4_shared_w2_w)
        self.swiglu_limit = views.geometry.swiglu_limit

    def forward(self, hidden, *, allow_quant=True):
        return shared_expert_forward(
            hidden, self.w13, self.w2, self.swiglu_limit,
            allow_quant=allow_quant,
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
            verify = _verify_packed_attention_graph(state) and state.grouped_moe is None
            proposal = _proposal_exact_ops(state)
            overlap = getattr(state, "verify_moe_overlap", None)
            if verify and overlap is not None and overlap.supported(state, flat, self):
                return overlap.forward(
                    self, flat, shape, state,
                    expert_load=expert_load, return_shared=_return_shared,
                )
            if state.grouped_moe is None and (
                (
                    (ordinary or proposal)
                    and optimization_enabled(DIRECT_MOE_ROUTES, default=True)
                )
                or (
                    verify
                    and optimization_enabled(
                        "RTP_V41_PPU_VERIFY_DIRECT_MOE_ROUTES", default=True,
                    )
                )
            ):
                ids, weights = self.gate(
                    flat,
                    graph_routes=(real, routed_rows),
                    allow_top3_graph_routes=proposal,
                )
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
                # SG cannot be captured: warmup must select the same fast
                # activation as capture/replay for every Graph phase.
                shared = self.shared_expert(flat, allow_quant=False)
                if _return_shared:
                    return routed.reshape(shape), shared.reshape(shape)
                routed = combine_routed_and_shared(routed, shared)
            return routed.reshape(shape)
        cp = _full_cp8_prefill_context(state)
        routing_cp = cp
        if (
            cp is None
            and getattr(state, "compacted", False)
            and state.use_compact_cp8_ops(state.context)
        ):
            routing_cp = state.context
        direct_routes = (
            routing_cp is not None
            and flat.is_cuda
            and optimization_enabled("RTP_V41_PPU_CP8_DIRECT_MOE_ROUTES")
        )
        direct_real = (
            routing_cp.local_is_real if direct_routes
            else _local_prefill_direct_moe_real(state, flat)
        )
        direct_routes = direct_routes or direct_real is not None
        if direct_routes:
            routed_rows = torch.empty((), dtype=torch.int64, device=flat.device)
            ids, weights = self.gate(flat, graph_routes=(direct_real, routed_rows))
            if os.environ.get("RTP_V41_SHARED_EVIDENCE_DIR"):
                state.routed_rows[self.layer_id] = int(routed_rows)
        else:
            ids, weights = pad_topk_for_dispatch(*self.gate(flat))
        if state is not None and not direct_routes:
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
        # The compact query layout fits the full CP8 workspace reserved at
        # entry. Reusing it also avoids a capacity all-reduce on each layer.
        prefill_buffer = getattr(state, "dp_prefill_mega_buffer", None)
        if prefill_buffer is None:
            prefill_buffer = (
                getattr(state, "prefill_mega_buffer", None)
                if cp is not None or (routing_cp is not None and cp8_yoco_fast_prefill_enabled(state))
                else None
            )

        def run_routed(value):
            return self.ops.moe(
                value, ids, weights, layer_id=self.layer_id, expert_load=expert_load,
                **({"prepared_buffer": prefill_buffer} if prefill_buffer is not None else {}),
            )

        if self.shared_expert is None:
            return run_routed(flat).reshape(shape)
        def run_shared(value):
            return self.shared_expert(value)

        if (
            cp is not None and prefill_buffer is not None
            and os.environ.get("RTP_V41_PPU_CP8_YOCO_OVERLAP_SHARED", "0") == "1"
            and cp8_yoco_fast_prefill_enabled(state)
            and not torch.is_grad_enabled() and not torch.cuda.is_current_stream_capturing()
            and torch.cuda.current_stream(flat.device) == torch.cuda.default_stream(flat.device)
        ):
            from rtp_llm.models_py.modules.dsv41.ppu.projection_streams import QKVStreams

            if runtime.qkv_streams is None:
                runtime.qkv_streams = QKVStreams(flat.device)
            routed, shared = runtime.qkv_streams.run(
                flat, run_routed, run_shared,
            )
        else:
            routed = run_routed(flat)
            shared = run_shared(flat)
        return combine_routed_and_shared(routed, shared).reshape(shape)


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

    def prefetch(self, hashes, *, uniform_rank_rows=False, round_schedule=None, local_fused_mask=False):
        if hashes is None:
            raise ValueError("shared Engram requires scheduler token-window hashes")
        if self._prefetched is not None:
            raise RuntimeError("unconsumed Engram prefetch from an earlier call")
        options = {"uniform_rank_rows": True} if uniform_rank_rows else {}
        if round_schedule is not None:
            options["round_schedule"] = round_schedule
        embedded = self.ops.engram_lookup(hashes, layer_id=self.layer_id, **options)
        self._prefetched = (hashes, embedded, True) if local_fused_mask is True else (hashes, embedded)

    def clear(self):
        self._prefetched = None

    def forward(self, hidden, hashes, token_mask=None):
        if hashes is None:
            pending = self._prefetched
            if pending is not None and len(pending) == 3 and pending[2] is True:
                self._prefetched = None
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
        local_mask = pending is not None and len(pending) == 3 and pending[2] is True
        if local_mask:
            # The existing raw-pointer gate cannot interpret deferred view
            # transforms. Decline before launch and keep the masked legacy
            # expression; read-only aliases remain valid inputs.
            local_mask = all(
                isinstance(tensor, torch.Tensor) and tensor.layout == torch.strided
                and not tensor.is_neg() and not tensor.is_conj()
                for tensor in (hidden, value, self.q_weight, self.k_weight, token_mask)
                if tensor is not None
            )
        return engram_gate_inject(
            hidden, self.q_weight, self.k_weight, value, token_mask=token_mask,
            **({"fused_mask": True} if local_mask else {}),
        )
