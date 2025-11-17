import logging
from typing import Callable, Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import RMSNorm, SelectTopk
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.factory.fused_moe import FusedMoeFactory
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.mla.mla_attention import MlaAttention
from rtp_llm.models_py.modules.mlp import FusedSiluActDenseMLP
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class AttentionFactory:
    """Factory class for creating attention modules based on key_str."""

    # Attention type registry - maps key_str to attention class and creation function
    ATTENTION_REGISTRY: Dict[str, Dict[str, any]] = {
        "causal": {
            "class": CausalAttention,
            "create_func": lambda config, weights, layer_idx: CausalAttention(
                config, weights
            ),
        },
        "mla": {
            "class": MlaAttention,
            "create_func": lambda config, weights, layer_idx: MlaAttention(
                config, weights, layer_idx
            ),
        },
    }

    @classmethod
    def create_attention(
        cls,
        key_str: str,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int = 0,
    ) -> nn.Module:
        """
        Create attention module based on key_str.

        Args:
            key_str: String key identifying the attention type
            config: Model configuration
            weights: Model weights
            layer_idx: Layer index (used by some attention types)

        Returns:
        Attention module instance

        Raises:
            ValueError: If key_str is not supported
        """
        if key_str not in cls.ATTENTION_REGISTRY:
            available_types = list(cls.ATTENTION_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{key_str}'. Available types: {available_types}"
            )

        attention_info = cls.ATTENTION_REGISTRY[key_str]
        attention_class = attention_info["class"]

        if attention_class is None:
            raise ImportError(
                f"Attention class for '{key_str}' is not available. Please check imports."
            )

        create_func = attention_info["create_func"]
        return create_func(config, weights, layer_idx)

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported attention types."""
        return list(cls.ATTENTION_REGISTRY.keys())

    @classmethod
    def register_attention_type(
        cls, key_str: str, attention_class: Type[nn.Module], create_func: Callable
    ):
        """
        Register a new attention type.

        Args:
            key_str: String key for the attention type
            attention_class: Attention module class
            create_func: Function to create the attention instance
        """
        cls.ATTENTION_REGISTRY[key_str] = {
            "class": attention_class,
            "create_func": create_func,
        }


class FMHAImplFactory:
    """Factory class for creating FMHA implementations based on attention_type."""

    # FMHA implementation registry - maps attention_type to impl method
    FMHA_IMPL_REGISTRY: Dict[str, str] = {
        "causal": "get_fmha_impl",
        "mla": "get_mla_impl",
    }

    @classmethod
    def get_fmha_impl_method(cls, attention_type: str) -> str:
        """
        Get the appropriate FMHA implementation method based on attention_type.

        Args:
            attention_type: String identifying the attention type

        Returns:
            Method name to call for getting FMHA implementation

        Raises:
            ValueError: If attention_type is not supported
        """
        if attention_type not in cls.FMHA_IMPL_REGISTRY:
            available_types = list(cls.FMHA_IMPL_REGISTRY.keys())
            raise ValueError(
                f"Unsupported attention type '{attention_type}'. Available types: {available_types}"
            )

        return cls.FMHA_IMPL_REGISTRY[attention_type]

    @classmethod
    def register_fmha_impl(cls, attention_type: str, impl_method: str):
        """
        Register a new FMHA implementation method for an attention type.

        Args:
            attention_type: String key for the attention type
            impl_method: Method name to call for getting FMHA implementation
        """
        cls.FMHA_IMPL_REGISTRY[attention_type] = impl_method

    @classmethod
    def get_supported_types(cls) -> list:
        """Get list of supported attention types."""
        return list(cls.FMHA_IMPL_REGISTRY.keys())


class GenericMoeLayer(nn.Module):
    """Generic MoE layer supporting both Qwen3 and internal model."""

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__()
        self.config = config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k

        self.gate = Linear(weights[W.moe_gate], None)
        self.select_topk = SelectTopk(config)
        self.fused_moe = FusedMoeFactory().create_fused_moe(config, weights)
        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert (
            self.w1 is not None and self.w2 is not None
        ), "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]
        self.expert_map = self.build_expert_map()

        # for group topk
        self.correction_bias = weights.get(W.e_score_correction_b, None)

    def build_expert_map(self):
        """Build expert mapping for EP (Expert Parallelism)."""
        num_local_experts = self.num_local_experts
        global_num_experts = self.num_experts
        expert_map = torch.full((global_num_experts,), fill_value=-1, dtype=torch.int32)
        start_id = self.config.ep_rank * num_local_experts
        end_id = start_id + num_local_experts
        expert_map[start_id:end_id] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(), dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)
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

        if self.correction_bias is not None:
            from rtp_llm.models_py.modules.select_topk import GroupTopK

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
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        attention_type: str = "causal",
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = AttentionFactory.create_attention(
            attention_type, config, weights, layer_idx
        )

        # Determine if this is a Dense layer (before first MoE layer or dense only)
        if layer_idx not in config.moe_layer_index:
            self.is_dense_layer = True
        else:
            self.is_dense_layer = False
            self.moe_mlp = GenericMoeLayer(config, weights)

        self.add_shared_expert = getattr(config, "moe_style", 1) == 2

        # Try to create shared_mlp and catch errors if weights don't exist
        self.shared_mlp = None
        if self.is_dense_layer or self.add_shared_expert:
            try:
                self.shared_mlp = FusedSiluActDenseMLP(config, weights)
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
        config: GptInitModelParameters,
        weights: ModelWeights,
        attention_type: str = "causal",  # Default attention type
    ):
        super().__init__(config, weights)
        self.attention_type = attention_type
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    config, weights.weights[idx], idx, attention_type
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        impl_method = FMHAImplFactory.get_fmha_impl_method(self.attention_type)
        fmha_impl = getattr(self, impl_method)(attention_inputs)

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
