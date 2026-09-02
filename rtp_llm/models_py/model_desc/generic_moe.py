import os
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
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
    RMSResNorm,
    SelectTopk,
    SigmoidGateScaleAdd,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear
from rtp_llm.models_py.modules.hybrid.glm5_cmp import (
    Glm5Cmp,
    resolve_glm5_cmp_enabled,
    should_enable_glm5_cmp,
)
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
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


class _FusedSharedExpertSentinel(nn.Module):
    """Marker for a shared expert folded into a MegaMoE kernel."""

    accepts_fp8_input = False

    def forward(self, *args, **kwargs):
        raise RuntimeError("shared expert is fused into MegaMoE")


def _validate_hy4_mxfp8_moe_strategy(
    config: ModelConfig, moe_config: MoeConfig
) -> None:
    """Reject MXFP8 HY4 expert backends that silently drop SwiGLU clamp.

    HY4 applies ``swiglu_limit`` only to routed experts. The dedicated
    ``mega_moe_fp8`` wrapper forwards that limit to DeepGEMM while leaving the
    separately evaluated shared expert unclamped. The generic and fused-shared
    executors currently accept ``extra_expert_args`` but do not implement this
    asymmetric clamp, so allowing them would produce plausible-shaped but
    numerically different output.
    """
    if config.model_type not in ("hy_v4", "hy_v4_mtp"):
        return
    quant_config = config.quant_config
    quant_method = quant_config.get_method() if quant_config is not None else None
    if quant_method != "MXFP8" or float(config.swiglu_limit) <= 0:
        return
    if moe_config.moe_strategy != "mega_moe_fp8":
        raise ValueError(
            "HY V4 MXFP8 routed experts require moe_strategy=mega_moe_fp8: "
            f"moe_strategy={moe_config.moe_strategy!r} does not preserve the "
            "routed-only SwiGLU clamp"
        )


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

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # GLM5 CP prefill changes the rank-local token count when a reused long
        # request is mixed with short requests (666 -> 697 rows in the
        # reproducer).  NVJet's BF16 GEMM changes its K-reduction at that M
        # boundary, and one-ULP router-logit drift can flip the last top-k
        # expert.  Decode does not use this padding path.
        is_decode_role = False
        try:
            from rtp_llm.ops import RoleType

            is_decode_role = (
                getattr(parallelism_config, "role_type", None) == RoleType.DECODE
            )
        except Exception:
            pass
        cp_prefill_enabled = False
        cp_config = getattr(parallelism_config, "prefill_cp_config", None)
        if not is_decode_role and cp_config is not None:
            try:
                cp_prefill_enabled = bool(cp_config.is_enabled())
            except Exception:
                pass
        default_gate_chunk_rows = (
            8192
            if getattr(config, "model_type", "") == "glm_5" and cp_prefill_enabled
            else 0
        )
        self.gate_chunk_rows = int(
            os.environ.get("GLM5_MOE_GATE_CHUNK_ROWS", str(default_gate_chunk_rows))
        )
        if self.gate_chunk_rows < 0:
            raise ValueError(
                "GLM5_MOE_GATE_CHUNK_ROWS must be non-negative, got "
                f"{self.gate_chunk_rows}"
            )

        # Get quant_config from model_config
        quant_config = config.quant_config
        _validate_hy4_mxfp8_moe_strategy(config, moe_config)
        self._hy4_fp32_router = getattr(config, "model_type", "") in (
            "hy_v4",
            "hy_v4_mtp",
        )
        if self._hy4_fp32_router:
            self.gate = None
            self.gate_weight = weights[W.moe_gate]
            if self.gate_weight.dtype != torch.float32:
                raise TypeError(
                    f"HY V4 router weight must be fp32, got {self.gate_weight.dtype}"
                )
            expected_router_shape = (self.hidden_dim, config.expert_num)
            if tuple(self.gate_weight.shape) != expected_router_shape:
                raise ValueError(
                    "HY V4 router weight must have runtime shape "
                    f"{expected_router_shape}, got {tuple(self.gate_weight.shape)}"
                )
        else:
            self.gate = LinearFactory.create_linear_from_weights(
                weights, W.moe_gate, None, None, quant_config, hw_kernel_config
            )
            self.gate_weight = None
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
        self.add_shared_expert = config.moe_style == 2
        self.ffn_tp_size = parallelism_config.get_ffn_tp_size()
        self.ep_size = parallelism_config.ep_size
        shared_expert_gate_weight = weights.get(W.shared_expert_gate, None)
        is_ep_mode = self.ep_size > 1
        use_ep_shared_allreduce_at_init = (
            self.add_shared_expert and self.ffn_tp_size > 1 and is_ep_mode
        )
        fused_shared_strategies = (
            "mega_moe_se",
            "mega_moe_fused",
            "mega_moe_fp8_se",
        )
        self._use_mega_moe_fused_shared = (
            moe_config.moe_strategy in fused_shared_strategies
        )
        if self._use_mega_moe_fused_shared:
            if not self.add_shared_expert:
                raise ValueError(
                    f"moe_strategy={moe_config.moe_strategy} requires shared experts"
                )
            if shared_expert_gate_weight is not None:
                raise ValueError(
                    f"moe_strategy={moe_config.moe_strategy} does not support "
                    "shared_expert_gate"
                )
            if use_ep_shared_allreduce_at_init:
                raise ValueError(
                    f"moe_strategy={moe_config.moe_strategy} does not support EP "
                    "shared-expert all-reduce with ffn_tp_size > 1"
                )

        if moe_config.moe_strategy in (
            "mega_moe",
            "mega_moe_se",
            "mega_moe_fp8",
            "mega_moe_fp8_se",
            "mega_moe_fused",
        ):
            if moe_config.moe_strategy == "mega_moe_se":
                from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_se_wrapper import (
                    MegaMoeSEWrapper,
                )

                wrapper_cls = MegaMoeSEWrapper
            elif moe_config.moe_strategy == "mega_moe_fused":
                from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fused_wrapper import (
                    MegaMoeFusedWrapper,
                )

                wrapper_cls = MegaMoeFusedWrapper
            elif moe_config.moe_strategy == "mega_moe_fp8_se":
                from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_se_wrapper import (
                    MegaMoeFp8SEWrapper,
                )

                wrapper_cls = MegaMoeFp8SEWrapper
            elif moe_config.moe_strategy == "mega_moe_fp8":
                from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_fp8_wrapper import (
                    MegaMoeFp8Wrapper,
                )

                wrapper_cls = MegaMoeFp8Wrapper
            else:
                from rtp_llm.models_py.modules.glm5_mega_moe.mega_moe_wrapper import (
                    MegaMoeWrapper,
                )

                wrapper_cls = MegaMoeWrapper

            self.fused_moe = wrapper_cls(
                config,
                parallelism_config,
                weights,
                moe_config,
                layer_idx=layer_idx,
                max_generate_batch_size=max_generate_batch_size,
            )
        else:
            config_adapter = MoEConfigAdapter(
                model_config=config,
                parallelism_config=parallelism_config,
                moe_config=moe_config,
                quant_config=quant_config,
                enable_cuda_graph=enable_cuda_graph,
            )
            self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        if self.w1 is not None:
            self.num_local_experts = self.w1.shape[0]
        elif hasattr(self.fused_moe, "expert_num"):
            self.num_local_experts = self.fused_moe.expert_num
        else:
            raise ValueError(
                "Cannot determine num_local_experts: no w1 weight and fused_moe has no expert_num"
            )
        if self.add_shared_expert:
            if self._use_mega_moe_fused_shared:
                self.shared_expert = _FusedSharedExpertSentinel()
            else:
                self.shared_expert = DenseMLP(
                    config.activation_type,
                    parallelism_config,
                    weights,
                    quant_config,
                    hw_kernel_config=hw_kernel_config,
                )
        else:
            self.shared_expert = None
        if shared_expert_gate_weight is not None:
            self.shared_expert_gate = LinearFactory.create_linear_from_weights(
                weights, W.shared_expert_gate, None, None, config
            )
            self.sigmoid_gate_scale_add = SigmoidGateScaleAdd()
        else:
            self.shared_expert_gate = None
            self.sigmoid_gate_scale_add = None

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)
        if self._hy4_fp32_router:
            if self.correction_bias is None:
                raise KeyError("HY V4 MoE requires e_score_correction_bias")
            if self.correction_bias.dtype != torch.float32:
                raise TypeError(
                    "HY V4 correction bias must be fp32, got "
                    f"{self.correction_bias.dtype}"
                )
            if self.correction_bias.numel() != config.expert_num:
                raise ValueError(
                    "HY V4 correction bias must contain one value per expert, got "
                    f"{self.correction_bias.numel()} for {config.expert_num} experts"
                )

    def clone_for_cuda_graph(self) -> "GenericMoeLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)

        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.hidden_dim = self.hidden_dim
        clone.ffn_dim = self.ffn_dim
        clone.num_experts = self.num_experts
        clone.top_k = self.top_k
        clone.gate_chunk_rows = self.gate_chunk_rows
        clone.gate = self.gate
        clone.gate_weight = self.gate_weight
        clone._hy4_fp32_router = self._hy4_fp32_router
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
        clone.shared_expert_gate = self.shared_expert_gate
        clone.sigmoid_gate_scale_add = self.sigmoid_gate_scale_add
        clone.correction_bias = self.correction_bias
        clone._use_mega_moe_fused_shared = self._use_mega_moe_fused_shared
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
        experts_output = self.fused_moe.forward_prepacked(hidden_states)
        if self._use_mega_moe_fused_shared:
            return experts_output
        return experts_output + self.shared_expert(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        x_fp8: "Optional[torch.Tensor]" = None,
        x_scale: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        if self._hy4_fp32_router:
            router_logits = torch.matmul(hidden_states.float(), self.gate_weight)
        elif self.gate_chunk_rows > 0 and num_tokens > 0:
            router_logits = fixed_m_linear(
                self.gate, hidden_states, self.gate_chunk_rows
            )
        else:
            router_logits = self.gate(
                hidden_states
            )  # fuse kernel: nvjet_tst_64x8_64x16_2x4_h_bz_NNT (bf16 nn.Linear router, every layer)
        router_logits_fp32 = (
            router_logits.float()
        )  # fuse kernel: at::native::unrolled_elementwise_kernel<direct_copy_kernel_cuda> (bf16 -> fp32 cast)

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
                scores=router_logits_fp32,
                correction_bias=self.correction_bias,
                n_group=self.num_expert_group,
                topk_group=self.topk_group,
                topk=self.top_k,
                renormalize=self.renormalize,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            # Top-K selection using C++ SelectTopkOp
            self.select_topk(router_logits_fp32, topk_ids, topk_weights)

        if self.fake_balance_expert is not None:
            self.fake_balance_expert(topk_ids, topk_weights)

        is_ep_mode = self.ep_size > 1
        use_ep_shared_allreduce = (
            self.shared_expert is not None and self.ffn_tp_size > 1 and is_ep_mode
        )
        use_mega_moe_fused_shared = (
            self.shared_expert is not None
            and self.shared_expert_gate is None
            and not use_ep_shared_allreduce
            and self._use_mega_moe_fused_shared
        )

        experts_output = self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
            extra_expert_args={
                "swiglu_limit": (
                    float(getattr(self.config, "swiglu_limit", 0.0))
                    if getattr(self.config, "model_type", "")
                    in ("hy_v4", "hy_v4_mtp")
                    else 0.0
                )
            },
        )
        if use_mega_moe_fused_shared:
            return experts_output
        if self.shared_expert is not None:
            shared_expert_output = self.shared_expert(
                hidden_states,
                x_fp8=x_fp8,
                x_scale=x_scale,
                skip_allreduce=use_ep_shared_allreduce,
            )
            if use_ep_shared_allreduce:
                # EP mode: routed expert output is already complete
                # (EP combine via all_to_all / all_gather aggregated across ranks).
                # Only the shared expert output is TP-partial and needs all_reduce.
                # Cannot use sigmoid_gate_scale_add here — all_reduce must run
                # between the gate-apply and the add-to-experts steps.
                if self.shared_expert_gate is not None:
                    gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
                    shared_expert_output = (
                        torch.sigmoid(gate_output) * shared_expert_output
                    )
                shared_expert_output = all_reduce(shared_expert_output, group=Group.TP)
                experts_output = experts_output + shared_expert_output
            else:
                if self.shared_expert_gate is not None:
                    gate_output = self.shared_expert_gate(hidden_states)  # [T, 1]
                    # Fused: experts_output += sigmoid(gate_output) * shared_expert_output
                    self.sigmoid_gate_scale_add(
                        gate_output, shared_expert_output, experts_output
                    )
                else:
                    experts_output = experts_output + shared_expert_output
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
                indexer_layernorm_eps=getattr(
                    config, "indexer_layernorm_eps", None
                ),
                indexer_scale_fmt=getattr(config, "indexer_scale_fmt", None),
                indexer_use_hadamard=getattr(
                    config, "indexer_use_hadamard", True
                ),
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
            and isinstance(self.mlp, GenericMoeLayer)
            and self.mlp.shared_expert is not None
            and self.mlp.shared_expert.accepts_fp8_input
        )

    def clone_for_cuda_graph(
        self, *, draft_prefill: bool = False
    ) -> "GenericMoeDecoderLayer":
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
            fmha_impl,
            kv_cache,
            prev_topk_indices,
            reuse_topk_indices=force_reuse_topk_indices,
        )

        # Run RTP's existing FlashMLA interface explicitly.
        mla_output = cmp.sparse_mla(
            mla_query,
            topk_indices,
            fmha_impl,
            kv_cache,
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
            fmha_impl,
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


class GenericMoeModel(GptModelBase):
    """Generic MoE model supporting Qwen3-MoE, internal model, and other MoE architectures."""

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
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
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

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
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
        prev_topk_indices = None
        enable_cmp = should_enable_glm5_cmp(
            self.layers,
            self.layer_num,
            hidden_states,
            fmha_impl,
            self.kv_cache,
        )
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                prev_topk_indices=prev_topk_indices,
                enable_cmp=enable_cmp,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            prev_topk_indices = output.topk_indices
            if _rt_on:
                _rt.record(f"layer{i:02d}_hidden", hidden_states)
                _rt.record(f"layer{i:02d}_residual", residual)
                _rt.record(f"layer{i:02d}_combined", hidden_states + residual)

        hidden_states, _ = self.norm(hidden_states, residual)
        if _rt_on:
            _rt.record("final_norm", hidden_states)
            extra: dict = {
                "input_ids_shape": tuple(input_ids.shape),
                "input_ids": input_ids.detach().cpu(),
            }
            _rt.dump(step=getattr(self, "_dbg_step", 0), extra=extra)
            self._dbg_step = getattr(self, "_dbg_step", 0) + 1
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
