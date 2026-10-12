import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import get_layer_caches_for_groups
from rtp_llm.models_py.model_desc.cp_mtp_hidden_handoff import CpMtpHiddenHandoff
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer, GenericMoeModel
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    MlaAttention,
    RMSResNorm,
)
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.glm5_mega_moe.factory import (
    MEGA_MOE_STRATEGIES,
    create_mega_moe_wrapper,
)
from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import MegaMoeWrapper
from rtp_llm.models_py.modules.hybrid.glm5_cmp import (
    Glm5Cmp,
    resolve_glm5_cmp_enabled,
    should_enable_glm5_cmp,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig, RoleType
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer, dsa_layer_skips_topk
from rtp_llm.utils.model_weight import W

try:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        fused_add_rmsnorm_fp8_quant,
        fused_add_rmsnorm_fp8_quant_with_bf16_output,
    )
except ImportError:
    CudaFp8GEMMLinear = None
    fused_add_rmsnorm_fp8_quant = None
    fused_add_rmsnorm_fp8_quant_with_bf16_output = None


class Glm5MoeLayer(GenericMoeLayer):
    """Common MoE execution with GLM5 gate sizing and graph/prepacked support."""

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
        self.layer_idx = layer_idx
        cp_prefill_enabled = (
            parallelism_config.role_type != RoleType.DECODE
            and parallelism_config.prefill_cp_config.is_enabled()
        )
        default_gate_chunk_rows = (
            8192 if config.model_type == "glm_5" and cp_prefill_enabled else 0
        )
        self.gate_chunk_rows = int(
            os.environ.get("GLM5_MOE_GATE_CHUNK_ROWS", str(default_gate_chunk_rows))
        )
        if self.gate_chunk_rows < 0:
            raise ValueError("GLM5_MOE_GATE_CHUNK_ROWS must be non-negative")
        super().__init__(
            config,
            parallelism_config,
            weights,
            moe_config,
            max_generate_batch_size,
            enable_cuda_graph,
            hw_kernel_config,
        )

    def _create_fused_moe(
        self,
        weights: Dict[str, torch.Tensor],
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        enable_cuda_graph: bool,
    ):
        if moe_config.moe_strategy not in MEGA_MOE_STRATEGIES:
            return super()._create_fused_moe(
                weights, moe_config, max_generate_batch_size, enable_cuda_graph
            )
        return create_mega_moe_wrapper(
            self.config,
            self.parallelism_config,
            weights,
            moe_config,
            layer_idx=self.layer_idx,
            max_generate_batch_size=max_generate_batch_size,
        )

    def _supports_unified_tp_allreduce(self) -> bool:
        return (
            not isinstance(self.fused_moe, MegaMoeWrapper)
            and super()._supports_unified_tp_allreduce()
        )

    def _gate_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate_chunk_rows > 0 and hidden_states.shape[0] > 0:
            return fixed_m_linear(
                self.gate, hidden_states, self.gate_chunk_rows
            ).float()
        return self.gate(hidden_states).float()

    def clone_for_cuda_graph(self, *, share_mega_buf: bool = False) -> "Glm5MoeLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.layer_idx = self.layer_idx
        clone.hidden_dim = self.hidden_dim
        clone.ffn_dim = self.ffn_dim
        clone.num_experts = self.num_experts
        clone.top_k = self.top_k
        clone.gate_chunk_rows = self.gate_chunk_rows
        clone.gate = self.gate
        clone.select_topk = self.select_topk
        clone.fake_balance_expert = self.fake_balance_expert
        if isinstance(self.fused_moe, MegaMoeWrapper):
            clone.fused_moe = self.fused_moe.clone_for_cuda_graph(
                share_mega_buf=share_mega_buf
            )
        elif share_mega_buf:
            raise RuntimeError(
                "MoE implementation does not support shared MegaMoE buffers"
            )
        else:
            clone.fused_moe = self.fused_moe
        clone.num_local_experts = self.num_local_experts
        clone.add_shared_expert = self.add_shared_expert
        clone.ffn_tp_size = self.ffn_tp_size
        clone.ep_size = self.ep_size
        clone.shared_expert = self.shared_expert
        clone.shared_expert_gate = self.shared_expert_gate
        clone.sigmoid_gate_scale_add = self.sigmoid_gate_scale_add
        clone.correction_bias = self.correction_bias
        clone.group_topk = self.group_topk
        clone.use_ep_shared_allreduce = self.use_ep_shared_allreduce
        clone.use_unified_tp_allreduce = self.use_unified_tp_allreduce
        return clone

    def forward_prepacked(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Consume router/quant resources already prepared by GLM5 CMP."""
        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)
        return self._add_shared_expert(
            hidden_states, self.fused_moe.forward_prepacked(hidden_states)
        )


@dataclass(slots=True)
class DecodeLayerOutput:
    hidden_states: torch.Tensor
    residual: torch.Tensor
    topk_indices: Optional[torch.Tensor] = None


class Glm5MoeDecoderLayer(nn.Module):
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
        if config.attn_config.use_mla:
            self.self_attn = MlaAttention(
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
                stable_topk=True,
            )
        else:
            attn_configs = config.getAttentionConfigs(
                parallelism_config.get_attn_tp_size()
            )
            self.self_attn = CausalAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                layer_idx,
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type, parallelism_config, weights, quant_config
            )
        elif moe_config.moe_strategy in MEGA_MOE_STRATEGIES:
            self.mlp = Glm5MoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )
        else:
            # Retain main's auto dispatch, fused shared-expert ownership and
            # unified TP all-reduce for strategies outside the GLM wrappers.
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # 使用 RMSResNorm 来 fuse residual add 和 layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.cmp = (
            Glm5Cmp(
                layer_idx=layer_idx,
                config=config,
                parallelism_config=parallelism_config,
                self_attn=self.self_attn,
                input_layernorm=self.input_layernorm,
                mlp=self.mlp,
                post_attention_layernorm=self.post_attention_layernorm,
            )
            if resolve_glm5_cmp_enabled()
            else None
        )

        # Fuse input_layernorm + fp8_quant → pass fp8 directly to first linear,
        # AND emit a bf16 normed output so downstream consumers (e.g. Indexer)
        # still see the normed feature vector. Single-output variant cannot be
        # used here because Indexer reads hidden_states directly.
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        _fuse_on = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_input_norm_quant = False
        self._fuse_input_scale_ue8m0 = False
        if _fuse_on and (
            fused_add_rmsnorm_fp8_quant_with_bf16_output is not None
            and CudaFp8GEMMLinear is not None
        ):
            if isinstance(self.self_attn, CausalAttention):
                _qkv = getattr(self.self_attn, "qkv_proj", None)
                if isinstance(_qkv, CudaFp8GEMMLinear):
                    self._fuse_input_norm_quant = True
                    self._fuse_input_scale_ue8m0 = _qkv.scale_ue8m0
            elif isinstance(self.self_attn, MlaAttention):
                _proj = getattr(self.self_attn, "fused_qkv_a_proj", None) or getattr(
                    self.self_attn, "fused_qkv_proj", None
                )
                if isinstance(_proj, CudaFp8GEMMLinear):
                    self._fuse_input_norm_quant = True
                    self._fuse_input_scale_ue8m0 = _proj.scale_ue8m0

        # Fuse post_attention_layernorm + fp8_quant for DenseMLP
        self._fuse_post_norm_quant = (
            _fuse_on
            and fused_add_rmsnorm_fp8_quant is not None
            and isinstance(self.mlp, DenseMLP)
            and self.mlp.accepts_fp8_input
        )

        # Fuse post_attention_layernorm + dual output (bf16+fp8) for MoE
        self._fuse_post_norm_quant_moe = (
            _fuse_on
            and fused_add_rmsnorm_fp8_quant_with_bf16_output is not None
            and isinstance(self.mlp, Glm5MoeLayer)
            and self.mlp.shared_expert is not None
            and self.mlp.shared_expert.accepts_fp8_input
        )

    def clone_for_cuda_graph(
        self, *, draft_prefill: bool = False, share_mega_buf: bool = False
    ) -> "Glm5MoeDecoderLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.layer_idx = self.layer_idx
        clone.self_attn = self.self_attn
        if share_mega_buf:
            if not isinstance(self.mlp, Glm5MoeLayer):
                raise RuntimeError("Shared MegaMoE buffers require a MoE layer")
            clone.mlp = self.mlp.clone_for_cuda_graph(share_mega_buf=True)
        elif hasattr(self.mlp, "clone_for_cuda_graph"):
            clone.mlp = self.mlp.clone_for_cuda_graph()
        else:
            clone.mlp = self.mlp
        clone.input_layernorm = self.input_layernorm
        clone.post_attention_layernorm = self.post_attention_layernorm
        clone._fuse_input_norm_quant = self._fuse_input_norm_quant
        clone._fuse_input_scale_ue8m0 = self._fuse_input_scale_ue8m0
        clone._fuse_post_norm_quant = self._fuse_post_norm_quant
        clone._fuse_post_norm_quant_moe = self._fuse_post_norm_quant_moe
        clone.cmp = (
            None
            if self.cmp is None
            else self.cmp.clone_for_cuda_graph(
                mlp=clone.mlp,
                draft_prefill=draft_prefill,
            )
        )
        return clone

    def _fwd_mlp_or_moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run residual add, RMSNorm, then the dense MLP or MoE."""
        # Dense MLP: fuse add + RMSNorm + FP8 quant; up_proj consumes FP8 directly.
        if self._fuse_post_norm_quant and hidden_states.dim() == 2:
            fp8_hs, scale = fused_add_rmsnorm_fp8_quant(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight.data,
                self.post_attention_layernorm.variance_epsilon,
                group_size=128,
                scale_ue8m0=self.mlp.up_proj.scale_ue8m0,
            )
            hidden_states = self.mlp(hidden_states, x_fp8=fp8_hs, x_scale=scale)
        # MoE: keep BF16 for routing/routed experts and FP8 for the shared expert.
        elif self._fuse_post_norm_quant_moe and hidden_states.dim() == 2:
            bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states,
                residual,
                self.post_attention_layernorm.weight.data,
                self.post_attention_layernorm.variance_epsilon,
                group_size=128,
                scale_ue8m0=self.mlp.shared_expert.up_proj.scale_ue8m0,
            )
            hidden_states = self.mlp(bf16_hs, x_fp8=fp8_hs, x_scale=scale)
        # Fallback: use the standard norm path and let MLP/MoE prepare its inputs.
        else:
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_cmp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool = False,
    ) -> DecodeLayerOutput:
        """Run the GLM5 CMP attention and MoE preparation path."""
        cmp = self.cmp
        if cmp is None:
            raise RuntimeError("GLM5 CMP execution was not initialized")
        # Prepare MLA inputs with internal streams and PDL. MTP reuse layers
        # consume the TopK indices produced by the seed model.
        output_residual, mla_query, topk_indices = cmp.mla_prologue(
            hidden_states,
            residual,
            fmha_impl["default"] if isinstance(fmha_impl, Mapping) else fmha_impl,
            kv_cache["default"] if isinstance(kv_cache, Mapping) else kv_cache,
            prev_topk_indices,
            reuse_topk_indices=force_reuse_topk_indices,
            indexer_fmha_impl=fmha_impl["indexer_kv"],
            indexer_kv_cache=(
                kv_cache.get("indexer_kv") if kv_cache is not None else None
            ),
        )

        # Run RTP's existing FlashMLA interface explicitly.
        mla_output = cmp.sparse_mla(
            mla_query,
            topk_indices,
            fmha_impl["default"] if isinstance(fmha_impl, Mapping) else fmha_impl,
            kv_cache["default"] if isinstance(kv_cache, Mapping) else kv_cache,
        )

        # Dense or unsupported MoE prepacking returns four None values and
        # falls back to the regular MLP/MoE path below.
        moe_activation, moe_scale, routed_indices, routed_weights = (
            cmp.moe_prepacked_input_views(int(mla_output.size(0)))
        )
        # Project attention output and prepare routed MoE input only when its
        # stable buffers match RTP-kernel's FP8/group-32 input ABI.
        mla_post = cmp.mla_post_moe_pre(
            mla_output,
            output_residual,
            fmha_impl["default"] if isinstance(fmha_impl, Mapping) else fmha_impl,
            moe_activation=moe_activation,
            moe_scale=moe_scale,
            routed_indices=routed_indices,
            routed_weights=routed_weights,
        )
        if moe_activation is not None:
            moe_hidden_states, output_residual, routed_indices, routed_weights = (
                mla_post
            )
            hidden_states = self.mlp.forward_prepacked(
                moe_hidden_states,
                routed_indices,
                routed_weights,
            )
        else:
            # Dense layers and unsupported MoE strategies use RTP's MLP path.
            hidden_states, output_residual = mla_post
            hidden_states, output_residual = self._fwd_mlp_or_moe(
                hidden_states, output_residual
            )

        return DecodeLayerOutput(hidden_states, output_residual, topk_indices)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        enable_cmp: bool = False,
        force_reuse_topk_indices: bool = False,
    ) -> DecodeLayerOutput:
        if enable_cmp:
            return self._forward_cmp(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices=force_reuse_topk_indices,
            )

        topk_indices = None
        if self._fuse_input_norm_quant and hidden_states.dim() == 2:
            bf16_hs, fp8_hs, scale = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states,
                residual,
                self.input_layernorm.weight.data,
                self.input_layernorm.variance_epsilon,
                group_size=128,
                scale_ue8m0=self._fuse_input_scale_ue8m0,
            )
            if isinstance(self.self_attn, MlaAttention):
                hidden_states, topk_indices = self.self_attn(
                    hidden_states=bf16_hs,
                    fmha_impl=fmha_impl,
                    kv_cache=kv_cache,
                    x_fp8=fp8_hs,
                    x_scale=scale,
                    prev_topk_indices=prev_topk_indices,
                    force_reuse_topk_indices=force_reuse_topk_indices,
                    return_topk=True,
                )
            else:
                hidden_states = self.self_attn(
                    hidden_states=bf16_hs,
                    fmha_impl=fmha_impl,
                    kv_cache=kv_cache,
                    x_fp8=fp8_hs,
                    x_scale=scale,
                )
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            if isinstance(self.self_attn, MlaAttention):
                hidden_states, topk_indices = self.self_attn(
                    hidden_states=hidden_states,
                    fmha_impl=fmha_impl,
                    kv_cache=kv_cache,
                    prev_topk_indices=prev_topk_indices,
                    force_reuse_topk_indices=force_reuse_topk_indices,
                    return_topk=True,
                )
            else:
                hidden_states = self.self_attn(
                    hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
                )

        hidden_states, residual = self._fwd_mlp_or_moe(hidden_states, residual)

        return DecodeLayerOutput(hidden_states, residual, topk_indices)


class Glm5ModelBase(CpMtpHiddenHandoff, GenericMoeModel):
    """Share main's FMHA contract while constructing GLM decoder layers."""

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
        # GenericMoeModel's constructor builds different decoder layers.
        GptModelBase.__init__(
            self,
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.moe_config = moe_config
        self.max_generate_batch_size = max_generate_batch_size
        self.device_resource_config = device_resource_config
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )

    def _initialize_decoder_layers(self) -> None:
        self.layers = nn.ModuleList(
            Glm5MoeDecoderLayer(
                self.config,
                self.parallelism_config,
                self.weight.weights[idx],
                self.weight.global_weights,
                idx,
                self.moe_config,
                self.max_generate_batch_size,
                enable_cuda_graph=(
                    self.py_hw_kernel_config.enable_cuda_graph
                    if self.py_hw_kernel_config is not None
                    else False
                ),
                hw_kernel_config=self.py_hw_kernel_config,
            )
            for idx in range(self.layer_num)
        )

    def _forward_decoder_layers(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        prev_topk_indices: Optional[torch.Tensor] = None,
        *,
        force_reuse_topk_indices: bool = False,
    ) -> DecodeLayerOutput:
        default_fmha = (
            fmha_impl["default"] if isinstance(fmha_impl, Mapping) else fmha_impl
        )
        enable_cmp = should_enable_glm5_cmp(
            self.layers,
            self.layer_num,
            hidden_states,
            default_fmha,
            self.kv_cache,
            force_reuse_topk_indices=force_reuse_topk_indices,
        )
        residual = torch.zeros_like(hidden_states)
        for idx, layer in enumerate(self.layers[: self.layer_num]):
            caches = None
            if self.kv_cache is not None:
                tags = (
                    ("default", "indexer_kv")
                    if isinstance(layer.self_attn, MlaAttention)
                    and layer.self_attn.has_indexer
                    else ("default",)
                )
                caches = get_layer_caches_for_groups(self.kv_cache, idx, tags)
            result = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=caches,
                prev_topk_indices=prev_topk_indices,
                enable_cmp=enable_cmp,
                force_reuse_topk_indices=force_reuse_topk_indices,
            )
            hidden_states, residual = result.hidden_states, result.residual
            prev_topk_indices = result.topk_indices
        return DecodeLayerOutput(hidden_states, residual, prev_topk_indices)


class Glm5MoeModel(Glm5ModelBase):
    """GLM5 model with shared DSA selection and optional CMP execution."""

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
            moe_config,
            max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self._initialize_decoder_layers()
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        hidden_states = self.embed_tokens(inputs.input_ids)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        if not isinstance(fmha_impl, Mapping) or set(fmha_impl) != {
            "default",
            "indexer_kv",
        }:
            raise RuntimeError("GLM5 requires default and indexer_kv attention routes")
        result = self._forward_decoder_layers(hidden_states, fmha_impl)
        hidden_states, _ = self.norm(result.hidden_states, result.residual)
        self.capture_cp_mtp_hidden(hidden_states)
        return PyModelOutputs(hidden_states)


__all__ = [
    "Glm5MoeLayer",
    "Glm5MoeDecoderLayer",
    "Glm5MoeModel",
]
