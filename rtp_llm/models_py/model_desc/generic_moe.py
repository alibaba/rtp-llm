import logging
from typing import Any, Dict, List, NamedTuple, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_fmha_impl_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FakeBalanceExpert,
    FMHAImplBase,
    FusedMoeFactory,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    MultimodalEmbeddingInjector,
    RMSNorm,
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

try:
    from rtp_llm.ops.compute_ops import (
        cuda_graph_capture_forward_enabled,
        cuda_graph_warmup_forward_enabled,
    )
except ImportError:

    def cuda_graph_capture_forward_enabled() -> bool:
        return False

    def cuda_graph_warmup_forward_enabled() -> bool:
        return False


try:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
except ImportError:
    CudaFp8GEMMLinear = None

try:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.mxfp8_linear import (
        CudaMxfp8Linear,
    )
except ImportError:
    CudaMxfp8Linear = None

try:
    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        fused_add_rmsnorm_fp8_quant,
        fused_add_rmsnorm_fp8_quant_with_bf16_output,
    )
except ImportError:
    fused_add_rmsnorm_fp8_quant = None
    fused_add_rmsnorm_fp8_quant_with_bf16_output = None


class _FusedFp8QuantParams(NamedTuple):
    group_size: int
    scale_ue8m0: bool
    round_to_pow2: bool


def _get_fused_fp8_quant_params(linear: Any) -> Optional[_FusedFp8QuantParams]:
    if CudaFp8GEMMLinear is not None and isinstance(linear, CudaFp8GEMMLinear):
        return _FusedFp8QuantParams(
            group_size=getattr(linear, "input_quant_group_size", 128),
            scale_ue8m0=getattr(linear, "input_quant_scale_ue8m0", linear.scale_ue8m0),
            round_to_pow2=getattr(linear, "input_quant_round_to_pow2", False),
        )
    if CudaMxfp8Linear is not None and isinstance(linear, CudaMxfp8Linear):
        return _FusedFp8QuantParams(
            group_size=getattr(linear, "input_quant_group_size", 32),
            scale_ue8m0=getattr(linear, "input_quant_scale_ue8m0", False),
            round_to_pow2=getattr(linear, "input_quant_round_to_pow2", True),
        )
    return None



def _resolve_swiglu_oai_params(config: ModelConfig):
    """Return (alpha, limit) tuple iff config asks for SwiGLU-OAI, else None.

    Used by MiniMax M3 / GPT-OSS-style models. ``swiglu_alpha`` is a Python-only
    config field (added via _python_fields); ``swiglu_limit`` already lives on
    the C++ side. We trigger OAI when ``swiglu_alpha > 0`` *and* a positive
    limit is set, to avoid grabbing DeepSeek-V4's silu-clamp variant (which
    sets only swiglu_limit).
    """
    alpha = getattr(config, "swiglu_alpha", 0.0) or 0.0
    limit = getattr(config, "swiglu_limit", 0.0) or 0.0
    if alpha > 0.0 and limit > 0.0:
        return (float(alpha), float(limit))
    return None


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        self.ep_size = parallelism_config.ep_size

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config, hw_kernel_config
        )
        self.select_topk = SelectTopk(config=config)
        if moe_config.fake_balance_expert:
            self.fake_balance_expert = FakeBalanceExpert(
                expert_num=config.expert_num,
                moe_k=config.moe_k,
                dp_rank=parallelism_config.dp_rank,
                dp_size=parallelism_config.dp_size,
                ep_size=parallelism_config.ep_size,
            )
        else:
            self.fake_balance_expert = None
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        config_adapter.has_shared_expert_gate = W.shared_expert_gate in weights
        # Executors with per-layer state (mega kernels key their JIT warmup and
        # logs on it) and decode-batch-sized staging buffers need these; the
        # adapter itself is neither layer- nor batch-bound.
        config_adapter.layer_id = layer_idx
        config_adapter.max_generate_batch_size = max_generate_batch_size
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)
        router = self.fused_moe.router
        router_tp_size = router.tp_collective_size

        self.num_local_experts = self.num_experts // max(self.ep_size, 1)
        self.add_shared_expert = (
            config.moe_style == 2 and not self.fused_moe.includes_shared_expert
        )
        if self.add_shared_expert:
            self.shared_expert = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
                swiglu_oai_params=_resolve_swiglu_oai_params(config),
            )
        else:
            self.shared_expert = None
        # Overlap executor: runs shared expert on an auxiliary CUDA stream
        # concurrently with the routed-expert dispatch/combine pipeline.
        # Controlled by MOE_SHARED_EXPERT_OVERLAP env var (default: off).
        if self.shared_expert is not None:
            from rtp_llm.models_py.modules.shared_expert_overlap import (
                SharedExpertOverlapExecutor,
            )

            self._shared_overlap = SharedExpertOverlapExecutor()
            # Pre-create the auxiliary stream so it exists before any CUDA
            # graph capture.  prepare() is a no-op when overlap is disabled.
            device = next(self.shared_expert.parameters(), None)
            if device is not None:
                self._shared_overlap.prepare(device.device)
        else:
            self._shared_overlap = None

        if weights.get(W.shared_expert_gate, None) is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights,
                W.shared_expert_gate,
                None,
                None,
                quant_config=quant_config,
                # For ROCm devices shared_expert_gate is not pre-swizzled during weight
                # loading and its single output column does not satisfy the SwizzleA
                # layout. Keep this scalar projection on the no-swizzle backend.
                hw_kernel_config=None,
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        self.use_ep_shared_allreduce = (
            self.shared_expert is not None and self.ffn_tp_size > 1 and self.ep_size > 1
        )
        self.use_unified_tp_allreduce = (
            self.shared_expert is not None
            and self.ffn_tp_size > 1
            and self.ep_size == 1
            and self.ffn_tp_size == router_tp_size
            and router.supports_skip_tp_allreduce
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "GenericMoE unified TP all-reduce %s "
                "(router=%s, ffn_tp_size=%d, router_tp_size=%d, ep_size=%d)",
                "enabled" if self.use_unified_tp_allreduce else "disabled",
                type(router).__name__,
                self.ffn_tp_size,
                router_tp_size,
                self.ep_size,
            )

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def _merge_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
        experts_output: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
            self.sigmoid_gate_scale_add(
                gate_output, shared_expert_output, experts_output
            )
            return experts_output
        return experts_output + shared_expert_output

    def clone_for_cuda_graph(self) -> "GenericMoeLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)

        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.hidden_dim = self.hidden_dim
        clone.ffn_dim = self.ffn_dim
        clone.num_experts = self.num_experts
        clone.top_k = self.top_k
        clone.gate = self.gate
        clone.select_topk = self.select_topk
        clone.fake_balance_expert = self.fake_balance_expert
        if hasattr(self.fused_moe, "clone_for_cuda_graph"):
            clone.fused_moe = self.fused_moe.clone_for_cuda_graph()
        else:
            clone.fused_moe = self.fused_moe
        clone.w1 = self.w1
        clone.w2 = self.w2
        clone.num_local_experts = self.num_local_experts
        clone.add_shared_expert = self.add_shared_expert
        clone.ffn_tp_size = self.ffn_tp_size
        clone.ep_size = self.ep_size
        clone.shared_expert = self.shared_expert
        clone._shared_overlap = self._shared_overlap
        clone.shared_expert_gate = self.shared_expert_gate
        clone.sigmoid_gate_scale_add = self.sigmoid_gate_scale_add
        clone.correction_bias = self.correction_bias
        clone.use_ep_shared_allreduce = self.use_ep_shared_allreduce
        clone.use_unified_tp_allreduce = self.use_unified_tp_allreduce
        return clone

    def _gate_shared_expert_output(
        self,
        hidden_states: torch.Tensor,
        shared_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_gate is not None:
            gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
            return torch.sigmoid(gate_output) * shared_expert_output
        return shared_expert_output

    def _compute_router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_weight = getattr(self.gate, "weight", None)
        router_input = (
            hidden_states.float()
            if gate_weight is not None
            and gate_weight.dtype == torch.float32
            and hidden_states.dtype != torch.float32
            else hidden_states
        )
        return self.gate(router_input)

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        # Some architectures (MiniMax-M3) deliberately store and evaluate the
        # router in FP32.  Match the input to that contract before GEMM instead
        # of rounding the FP32 checkpoint weight down to the activation dtype.
        router_logits = self._compute_router_logits(hidden_states)

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        # different executor may need different topk_ids dtype
        topk_ids_dtype = self.fused_moe.topk_ids_dtype
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=topk_ids_dtype,
            device=hidden_states.device,
        )

        if self.correction_bias is not None:
            self.group_topk = GroupTopK()
            self.renormalize = self.config.has_moe_norm
            self.num_expert_group = self.config.moe_n_group

            self.topk_group = self.config.moe_topk_group
            self.n_routed_experts = self.config.expert_num  # config.n_routed_experts
            self.routed_scaling_factor = self.config.routed_scaling_factor
            self.group_topk(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                scores=router_logits,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            self.select_topk(router_logits, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        # SwiGLU-OAI routing: when config asks for OAI math (MiniMax-M3 /
        # GPT-OSS), tell the FusedMoE backend to use the OAI kernel variant.
        # Alpha/limit are carried through ``extra_expert_args`` so executors
        # that don't know OAI can ignore them and we get a clear error in
        # those that do.
        _moe_act = "SiGLU"
        _moe_extra: Optional[Dict[str, Any]] = None
        _oai = _resolve_swiglu_oai_params(self.config)
        if _oai is not None:
            _moe_act = "swiglu_oai"
            _moe_extra = {"swiglu_alpha": _oai[0], "swiglu_limit": _oai[1]}

        # In pure-TP mode both the routed experts and the shared expert produce
        # TP-partial outputs.  Reduce their sum once instead of reducing each
        # path separately.  This is especially important for decode, where the
        # hidden dimension is small enough that collective launch latency
        # dominates the payload transfer.
        skip_shared_allreduce = (
            self.use_ep_shared_allreduce or self.use_unified_tp_allreduce
        )

        # Launch shared expert on auxiliary stream before routed expert work.
        # When overlap is disabled (env var / CUDA graph capture / non-CUDA),
        # SharedExpertOverlapExecutor.start() runs synchronously and caches
        # the result for finish().
        if self.shared_expert is not None and self._shared_overlap is not None:
            self._shared_overlap.start(
                self.shared_expert,
                hidden_states,
                x_fp8=x_fp8,
                x_scale=x_scale,
                skip_allreduce=skip_shared_allreduce,
            )

        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=_moe_act,
            extra_expert_args=_moe_extra,
            skip_tp_allreduce=self.use_unified_tp_allreduce,
        )
        if self.shared_expert is not None:
            # Collect shared expert output — from overlap executor or direct call.
            if self._shared_overlap is not None:
                shared_expert_output = self._shared_overlap.finish()
            else:
                shared_expert_output = self.shared_expert(
                    hidden_states,
                    x_fp8=x_fp8,
                    x_scale=x_scale,
                    skip_allreduce=skip_shared_allreduce,
                )
            if self.use_unified_tp_allreduce:
                # Both paths are still TP-partial.  The shared-expert gate is
                # rank-consistent because hidden_states are replicated across
                # TP ranks, so it is safe to apply it before the single
                # all-reduce.
                experts_output = self._merge_shared_expert_output(
                    hidden_states, experts_output, shared_expert_output
                )
                experts_output = all_reduce(experts_output, group=Group.TP)
            elif self.use_ep_shared_allreduce:
                # EP mode: routed expert output is already complete
                # (EP combine via all_to_all / all_gather aggregated across ranks).
                # Only the shared expert output is TP-partial and needs all_reduce.
                shared_expert_output = self._gate_shared_expert_output(
                    hidden_states, shared_expert_output
                )
                shared_expert_output = all_reduce(shared_expert_output, group=Group.TP)
                experts_output = experts_output + shared_expert_output
            else:
                # Fallback path: each path is already complete independently.
                # This includes ffn_tp_size == 1 and routers that retain their
                # own finalize reduction, so only local merging remains.
                experts_output = self._merge_shared_expert_output(
                    hidden_states, experts_output, shared_expert_output
                )

        return experts_output


class DecodeLayerOutput:
    def __init__(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        topk_indices: Optional[torch.Tensor] = None,
    ):
        self.hidden_states = hidden_states
        self.residual = residual
        # MoE routing indices produced by an MLA indexer layer, forwarded so a
        # later layer can reuse them instead of recomputing topk.
        self.topk_indices = topk_indices


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.self_attn = self._create_attention(
            config,
            parallelism_config,
            weights,
            global_weights,
            layer_idx,
            quant_config,
            hw_kernel_config,
        )
        self._is_msa_attn = self.self_attn.__class__.__name__ == "MSAAttention"

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
                swiglu_oai_params=_resolve_swiglu_oai_params(config),
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )

        _prefetch_gate = getattr(self.self_attn, "cp_prefix_prefetch_enabled", None)
        self._join_cp_prefix_prefetch = (
            getattr(self.self_attn, "join_cp_prefix_prefetch", None)
            if _prefetch_gate is not None
            and _prefetch_gate()
            and isinstance(self.mlp, GenericMoeLayer)
            else None
        )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        # Fuse input_layernorm + fp8_quant → pass fp8 directly to the attention
        # projection, AND emit a bf16 normed output so downstream consumers
        # still see the normed feature vector.
        #
        # Only MSAAttention takes part: it is the one attention module whose
        # ``forward`` accepts the ``x_fp8``/``x_scale`` pair. CausalAttention
        # and MlaAttention keep the unfused ``input_layernorm`` path below.
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        _fuse_on = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_input_norm_quant = False
        self._fuse_input_norm_quant_params = None
        if _fuse_on and (fused_add_rmsnorm_fp8_quant_with_bf16_output is not None):
            projection = self._input_quant_projection()
            params = _get_fused_fp8_quant_params(projection)
            if params is not None:
                self._fuse_input_norm_quant = True
                self._fuse_input_norm_quant_params = params

        # Fuse post_attention_layernorm + fp8_quant for DenseMLP
        self._fuse_post_norm_quant_params = (
            _get_fused_fp8_quant_params(getattr(self.mlp, "up_proj", None))
            if isinstance(self.mlp, DenseMLP)
            else None
        )
        self._fuse_post_norm_quant = (
            _fuse_on
            and fused_add_rmsnorm_fp8_quant is not None
            and isinstance(self.mlp, DenseMLP)
            and self._fuse_post_norm_quant_params is not None
        )

        # Fuse post_attention_layernorm + dual output (bf16+fp8) for MoE
        self._fuse_post_norm_quant_moe_params = None
        if isinstance(self.mlp, GenericMoeLayer) and self.mlp.shared_expert is not None:
            self._fuse_post_norm_quant_moe_params = _get_fused_fp8_quant_params(
                getattr(self.mlp.shared_expert, "up_proj", None)
            )
        self._fuse_post_norm_quant_moe = (
            _fuse_on
            and fused_add_rmsnorm_fp8_quant_with_bf16_output is not None
            and isinstance(self.mlp, GenericMoeLayer)
            and self.mlp.shared_expert is not None
            and self._fuse_post_norm_quant_moe_params is not None
        )

    def _create_attention(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Any,
        hw_kernel_config: Optional["HWKernelConfig"],
    ) -> nn.Module:
        if config.attn_config.use_mla:
            return MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
                has_indexer=dsa_layer_has_indexer(config, layer_idx),
                reuse_topk_indices=dsa_layer_skips_topk(config, layer_idx),
            )
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        # MiniMax-M3 sparse layers (MSA): route to the Triton sparse attention
        # path when this layer is sparse AND its index-branch weights are
        # present (gated by M3_LOAD_MSA_INDEX at load time). Otherwise fall
        # back to dense CausalAttention, which is numerically equivalent to MSA
        # for short prompts.
        msa_cfg = getattr(config, "msa_sparse_config", None)
        is_msa_layer = (
            msa_cfg is not None
            and layer_idx in set(msa_cfg.get("sparse_layer_ids", []))
            and W.msa_idx_q_w in weights
        )
        if is_msa_layer:
            from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention

            return MSAAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                msa_cfg,
                layer_idx,
                quant_config,
                hw_kernel_config,
            )
        return CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )

    def _input_quant_projection(self) -> Optional[nn.Module]:
        if self._is_msa_attn:
            # MSA's qkv_proj is built by LinearFactory and is FP8 under the M3
            # quant config; the idx_q/idx_k branches stay bf16 and consume the
            # bf16_normed output from the fused kernel.
            return getattr(self.self_attn, "qkv_proj", None)
        if isinstance(self.self_attn, CausalAttention):
            return getattr(self.self_attn, "qkv_proj", None)
        if isinstance(self.self_attn, MlaAttention):
            return getattr(self.self_attn, "fused_qkv_a_proj", None) or getattr(
                self.self_attn, "fused_qkv_proj", None
            )
        return None

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool,
        attn_inputs: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        quantized_inputs = {}
        if x_fp8 is not None:
            quantized_inputs = {"x_fp8": x_fp8, "x_scale": x_scale}

        if self._is_msa_attn:
            # Sparse MSA bypasses the shared FMHA impl: the module consumes
            # PyAttentionInputs directly and runs its own index branch plus
            # Triton sparse kernels. It produces no MoE topk indices.
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                attn_inputs=attn_inputs,
                kv_cache=kv_cache,
                **quantized_inputs,
            )
            return hidden_states, None

        if isinstance(self.self_attn, MlaAttention):
            hidden_states, topk_indices = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                prev_topk_indices=prev_topk_indices,
                force_reuse_topk_indices=force_reuse_topk_indices,
                return_topk=True,
                **quantized_inputs,
            )
            return hidden_states, topk_indices

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            **quantized_inputs,
        )
        return hidden_states, None

    def clone_for_cuda_graph(self) -> "GenericMoeDecoderLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.layer_idx = self.layer_idx
        clone.self_attn = self.self_attn
        if hasattr(self.mlp, "clone_for_cuda_graph"):
            clone.mlp = self.mlp.clone_for_cuda_graph()
        else:
            clone.mlp = self.mlp
        clone.input_layernorm = self.input_layernorm
        clone.post_attention_layernorm = self.post_attention_layernorm
        clone._is_msa_attn = self._is_msa_attn
        clone._fuse_input_norm_quant = self._fuse_input_norm_quant
        clone._fuse_input_norm_quant_params = self._fuse_input_norm_quant_params
        clone._fuse_post_norm_quant = self._fuse_post_norm_quant
        clone._fuse_post_norm_quant_params = self._fuse_post_norm_quant_params
        clone._fuse_post_norm_quant_moe = self._fuse_post_norm_quant_moe
        clone._fuse_post_norm_quant_moe_params = self._fuse_post_norm_quant_moe_params
        clone._join_cp_prefix_prefetch = None
        return clone

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        force_reuse_topk_indices: bool = False,
        attn_inputs: Optional[Any] = None,
    ) -> DecodeLayerOutput:
        if self._fuse_input_norm_quant and hidden_states.dim() == 2:
            params = self._fuse_input_norm_quant_params
            assert params is not None
            bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states,
                residual,
                self.input_layernorm.weight.data,
                self.input_layernorm.variance_epsilon,
                group_size=params.group_size,
                scale_ue8m0=params.scale_ue8m0,
                round_to_pow2=params.round_to_pow2,
            )
            hidden_states, topk_indices = self._forward_attention(
                bf16_hs,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices,
                attn_inputs,
                fp8_hs,
                scale,
            )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states, topk_indices = self._forward_attention(
                hidden_states,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices,
                attn_inputs,
            )

        if self._join_cp_prefix_prefetch is not None:
            self._join_cp_prefix_prefetch()

        if self._fuse_post_norm_quant and hidden_states.dim() == 2:
            _params = self._fuse_post_norm_quant_params
            assert _params is not None
            fp8_hs, scale = fused_add_rmsnorm_fp8_quant(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight.data,
                self.post_attention_layernorm.variance_epsilon,
                group_size=_params.group_size,
                scale_ue8m0=_params.scale_ue8m0,
                round_to_pow2=_params.round_to_pow2,
            )
            hidden_states = self.mlp(hidden_states, x_fp8=fp8_hs, x_scale=scale)
        elif self._fuse_post_norm_quant_moe and hidden_states.dim() == 2:
            _params = self._fuse_post_norm_quant_moe_params
            assert _params is not None
            bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight.data,
                self.post_attention_layernorm.variance_epsilon,
                group_size=_params.group_size,
                scale_ue8m0=_params.scale_ue8m0,
                round_to_pow2=_params.round_to_pow2,
            )
            hidden_states = self.mlp(bf16_hs, x_fp8=fp8_hs, x_scale=scale)
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
        return DecodeLayerOutput(hidden_states, residual, topk_indices)


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

    decoder_layer_cls = GenericMoeDecoderLayer

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = (
            MultimodalEmbeddingInjector()
            if bool(
                getattr(
                    getattr(model_config, "mm_model_config", None),
                    "is_multimodal",
                    False,
                )
            )
            else None
        )
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._cuda_graph_layers: Optional[nn.ModuleList] = None
        self._prefix_prefetch_hooks: Dict[int, Optional[List[Optional[Any]]]] = {}

    def _resolve_prefix_prefetch_hooks(
        self, layers: nn.ModuleList
    ) -> Optional[List[Optional[Any]]]:
        key = id(layers)
        if key not in self._prefix_prefetch_hooks:
            hooks: List[Optional[Any]] = []
            for layer in layers[: self.layer_num]:
                attn = getattr(layer, "self_attn", None)
                gate = getattr(attn, "cp_prefix_prefetch_enabled", None)
                hooks.append(
                    attn.maybe_prefetch_cp_prefix
                    if gate is not None and gate()
                    else None
                )
            self._prefix_prefetch_hooks[key] = (
                hooks if any(hook is not None for hook in hooks) else None
            )
        return self._prefix_prefetch_hooks[key]

    def _begin_mtp_target_hidden_capture(
        self, hidden_states: torch.Tensor
    ) -> Optional[torch.Tensor]:
        return None

    def _capture_mtp_target_hidden(
        self,
        capture: torch.Tensor,
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> None:
        pass

    def _finish_mtp_target_hidden_capture(self, capture: torch.Tensor) -> None:
        pass

    def _finish_mtp_target_hidden_capture_after_norm(
        self, final_residual: torch.Tensor
    ) -> None:
        pass

    def embedding(self, inputs: PyModelInputs) -> torch.Tensor:
        """Build token embeddings and inject features for an MM request.

        The model-level flag controls whether the injector is constructed;
        the request-level feature list controls whether this invocation uses
        the multimodal path. This keeps text-only requests on the original
        embedding kernel even when a VL model serves a mixed workload.
        """
        multimodal_inputs = getattr(inputs, "multimodal_inputs", None)
        multimodal_features = getattr(multimodal_inputs, "multimodal_features", None)
        injector = getattr(self, "multimodal_embedding_injector", None)
        if injector is None or not multimodal_features:
            return self.embed_tokens(inputs.input_ids)

        inputs_embeds = self.embed_tokens(
            inputs.input_ids,
            inputs.combo_position_ids,
            inputs.embedding_inputs.combo_tokens_type_ids,
            inputs.embedding_inputs.text_tokens_mask,
        )
        return injector(
            inputs_embeds,
            multimodal_features,
            multimodal_inputs.mm_features_locs,
        )

    def _layers_for_forward(self) -> nn.ModuleList:
        use_cuda_graph_layers = (
            cuda_graph_capture_forward_enabled() or cuda_graph_warmup_forward_enabled()
        )
        if not use_cuda_graph_layers:
            return self.layers
        if self._cuda_graph_layers is None:
            self._cuda_graph_layers = nn.ModuleList(
                [
                    (
                        layer.clone_for_cuda_graph()
                        if hasattr(layer, "clone_for_cuda_graph")
                        else layer
                    )
                    for layer in self.layers
                ]
            )
        return self._cuda_graph_layers

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embedding(inputs)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(
                inputs
            )  # pyright: ignore[reportUnreachable]

        from rtp_llm.models_py.modules.dsv4 import _record_tensor as _rt

        _rt_on = _rt.ENABLED
        if _rt_on:
            _rt.begin(
                seqlen=(
                    int(input_ids.size(0))
                    if input_ids.dim() == 1
                    else int(input_ids.size(-1))
                )
            )
            if _rt._get_buf() is None:
                _rt_on = False
        if _rt_on:
            _rt.record("embed_out", hidden_states)

        residual = torch.zeros_like(hidden_states)
        mtp_target_hidden_capture = self._begin_mtp_target_hidden_capture(hidden_states)
        prev_topk_indices = None
        layers = self._layers_for_forward()
        prefetch_hooks = (
            self._resolve_prefix_prefetch_hooks(layers)
            if inputs.attention_inputs.is_prefill and self.kv_cache is not None
            else None
        )
        for i, decoder_layer in enumerate(layers[: self.layer_num]):
            if prefetch_hooks is not None and i + 1 < self.layer_num:
                next_prefetch = prefetch_hooks[i + 1]
                if next_prefetch is not None:
                    next_prefetch(
                        self.kv_cache.get_layer_cache(i + 1), inputs.attention_inputs
                    )
            layer_fmha_impl = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                prev_topk_indices=prev_topk_indices,
                attn_inputs=inputs.attention_inputs,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            prev_topk_indices = output.topk_indices
            if mtp_target_hidden_capture is not None:
                self._capture_mtp_target_hidden(
                    mtp_target_hidden_capture,
                    i + 1,
                    hidden_states,
                    residual,
                )
            if _rt_on:
                _rt.record(f"layer{i:02d}_hidden", hidden_states)
                _rt.record(f"layer{i:02d}_residual", residual)
                _rt.record(f"layer{i:02d}_combined", hidden_states + residual)

        if mtp_target_hidden_capture is not None:
            self._finish_mtp_target_hidden_capture(mtp_target_hidden_capture)

        hidden_states, _ = self.norm(hidden_states, residual)
        self._finish_mtp_target_hidden_capture_after_norm(residual)
        if _rt_on:
            _rt.record("final_norm", hidden_states)
            extra: dict = {
                "input_ids_shape": tuple(input_ids.shape),
                "input_ids": input_ids.detach().cpu(),
            }
            _rt.dump(step=getattr(self, "_dbg_step", 0), extra=extra)
            self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        return PyModelOutputs(hidden_states)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
