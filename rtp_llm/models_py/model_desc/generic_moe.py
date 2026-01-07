import logging
from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    CausalAttention,
    Embedding,
    FMHAImplBase,
    FusedMoeFactory,
    FusedSiluActDenseMLP,
    GroupTopK,
    LinearFactory,
    MlaAttention,
    RMSNorm,
    SelectTopk,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


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
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.inter_size
        self.num_experts = config.eplb_config.phy_exp_num(config.expert_num)
        self.top_k = config.moe_k

        # Get quant_config from model_config
        quant_config = config.quant_config
        self.gate = LinearFactory.create_linear_from_weights(
            weights, W.moe_gate, None, None, quant_config
        )
        self.select_topk = SelectTopk(
            config, moe_config.fake_balance_expert, parallelism_config.dp_rank
        )
        config_adapter = MoEConfigAdapter(
            model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            max_generate_batch_size=max_generate_batch_size,
            quant_config=quant_config,
            enable_cuda_graph=enable_cuda_graph,
        )
        self.fused_moe = FusedMoeFactory().create_fused_moe(config_adapter, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

        # for eplb log2phy conversion
        self.log2phy = weights.get(W.log2phy, None)
        self.logic_expert_cnt = weights.get(W.logic_expert_cnt, None)
        self.phy_exp_num = config.eplb_config.phy_exp_num(config.expert_num)
        self.ep_rank = parallelism_config.ep_rank

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
        router_logits_fp32 = router_logits.float()

        topk_weights = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.empty(
            (num_tokens, self.top_k),
            dtype=torch.int64,
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
            self.select_topk(
                router_logits_fp32,
                topk_ids,
                topk_weights,
            )

        # Convert logical expert IDs to physical expert IDs using log2phy mapping
        if (
            self.log2phy is not None
            and self.logic_expert_cnt is not None
            and self.phy_exp_num > 0
        ):
            # Ensure tensors are contiguous and on the correct device
            log2phy = self.log2phy.contiguous()
            logic_expert_cnt = self.logic_expert_cnt.contiguous()
            topk_ids = topk_ids.contiguous()

            # Validate tensor dtypes
            if log2phy.dtype != torch.int32:
                raise RuntimeError("log2phy must be int32 tensor")
            if logic_expert_cnt.dtype != torch.int32:
                raise RuntimeError("logic_expert_cnt must be int32 tensor")

            # Call C++ kernel for log2phy conversion
            # convert_logical_to_physical_experts is a method of SelectTopkOp class
            self.select_topk.select_topk_op.convert_logical_to_physical_experts(
                topk_ids,
                log2phy,
                logic_expert_cnt,
                self.config.expert_num,  # log_exp_num
                self.phy_exp_num,  # phy_exp_num
                self.ep_rank,  # ep_rank
            )

        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
        )


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
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
            )
        else:
            self.self_attn = CausalAttention(
                config, parallelism_config, weights, quant_config
            )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
            )

        self.add_shared_expert = config.moe_style == 2

        # Try to create shared_mlp and catch errors if weights don't exist
        self.shared_mlp = None
        if self.is_dense_layer or self.add_shared_expert:
            try:
                self.shared_mlp = FusedSiluActDenseMLP(
                    config.activation_type, parallelism_config, weights, quant_config
                )
            except (KeyError, AssertionError) as e:
                # If weights don't exist, shared_mlp remains None
                logging.warning(
                    f"[GenericMoeDecoderLayer] Layer {self.layer_idx}: Failed to create shared_mlp: {e}"
                )

        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # MLP (Dense or MoE with optional shared experts)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_dense_layer:
            # Dense layer uses shared_mlp (must exist)
            assert self.shared_mlp is not None, "Dense layer must have shared_mlp"
            hidden_states = self.shared_mlp(hidden_states)
        else:
            # MoE layer
            experts_output = self.moe_mlp(hidden_states)

            if self.shared_mlp is not None:
                shared_mlp_output = self.shared_mlp(hidden_states)
                hidden_states = experts_output + shared_mlp_output
            else:
                hidden_states = experts_output

        hidden_states = residual + hidden_states

        return hidden_states


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
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
        )

        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states = self.norm(hidden_states)

        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "GenericMoeLayer",
    "GenericMoeDecoderLayer",
    "GenericMoeModel",
]
