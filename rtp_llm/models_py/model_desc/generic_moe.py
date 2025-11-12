import logging
from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import ParallelismConfig, MoeConfig, RuntimeConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.models_py.modules.moe import FusedMoe
from rtp_llm.models_py.modules.moe.config_adapter import MoEConfigAdapter
from rtp_llm.models_py.modules.moe.fused_moe_factory import FusedMoeFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

try:
    from librtp_compute_ops.rtp_llm_ops import SelectTopkOp
except ImportError:
    logging.error("SelectTopkOp is required but not available")


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self, config: ModelConfig, parallelism_config: ParallelismConfig, weights: Dict[str, torch.Tensor], quant_config: Optional[object] = None
    ):
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k

        self.gate = Linear(weights[W.moe_gate], None)
        # ModelConfig inherits from CppModelConfig (ModelConfig), so can be passed directly
        self.select_topk_op = SelectTopkOp(config)
        
                # Use MoEConfigAdapter to provide unified interface
        # Get MoeConfig and RuntimeConfig from ops (they have defaults)
        moe_config = MoeConfig()
        runtime_config = RuntimeConfig()
        # Create adapter that provides shortcut access to config fields
        config_adapter = MoEConfigAdapter(
            py_model_config=config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
            runtime_config=runtime_config,
            quant_config=quant_config,
        )
        self.fused_moe: FusedMoe = FusedMoeFactory.create_fused_moe(config_adapter, weights)

        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.expert_map = self.build_expert_map()

    def build_expert_map(self):
        """Build expert mapping for EP (Expert Parallelism)."""
        num_local_experts = self.num_local_experts
        global_num_experts = self.num_experts
        expert_map = torch.full((global_num_experts,), fill_value=-1, dtype=torch.int32)
        start_id = self.parallelism_config.ep_rank * num_local_experts
        end_id = start_id + num_local_experts
        expert_map[start_id:end_id] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(), dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)

        # Top-K selection using C++ SelectTopkOp
        router_logits_fp32 = router_logits.float()
        topk_weights = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)

        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
            expert_map=self.expert_map,
        )


class GenericMoeDecoderLayer(nn.Module):
    """Generic MoE decoder layer supporting Dense/MoE hybrid and shared experts."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Optional[object] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = CausalAttention(config, parallelism_config, weights, quant_config)

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = GenericMoeLayer(config, parallelism_config, weights, quant_config)

        self.add_shared_expert = config.moe_style == 2

        # Try to create shared_mlp and catch errors if weights don't exist
        self.shared_mlp = None
        if self.is_dense_layer or self.add_shared_expert:
            try:
                self.shared_mlp = FusedSiluActDenseMLP(config, parallelism_config, weights, quant_config)
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
        py_model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        device_resource_config,
        weights: ModelWeights,
        vocab_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
    ):
        super().__init__(py_model_config, parallelism_config, device_resource_config, weights, vocab_size, fmha_config=fmha_config, py_hw_kernel_config=py_hw_kernel_config)
        self.embed_tokens = Embedding(py_model_config, parallelism_config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(py_model_config, parallelism_config, weights.weights[idx], idx, quant_config)
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=py_model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = self.get_fmha_impl(attention_inputs)

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
