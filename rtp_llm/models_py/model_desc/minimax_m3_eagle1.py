from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import CausalAttention, DenseMLP, Embedding, RMSNorm
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MiniMaxM3Eagle1DecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=model_config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=model_config.layernorm_eps
        )
        self.self_attn = CausalAttention(
            model_config.attn_config,
            parallelism_config,
            weights,
            model_config.layernorm_eps,
            model_config.quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            model_config.activation_type,
            parallelism_config,
            weights,
            model_config.quant_config,
            hw_kernel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class MiniMaxM3Eagle1Model(GptModelBase):
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
        if self.layer_num != 1:
            raise ValueError(
                f"MiniMax-M3 EAGLE1 draft expects one layer, got {self.layer_num}"
            )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.weights[0],
            W.multi_tokens_predict_eh_proj,
            quant_config=model_config.quant_config,
        )
        fc_weight = weights.weights[0][W.multi_tokens_predict_eh_proj]
        self.fc_input_width = int(fc_weight.shape[0])
        self.hidden_size = int(model_config.hidden_size)
        if self.fc_input_width != self.hidden_size * 2:
            raise RuntimeError(
                "MiniMax-M3 EAGLE1 HASS fc input width must be 2x hidden size, "
                f"got {self.fc_input_width} for hidden size {self.hidden_size}"
            )
        attn_input_width = int(weights.weights[0][W.attn_qkv_w].shape[0])
        if attn_input_width != self.hidden_size:
            raise RuntimeError(
                "MiniMax-M3 EAGLE1 HASS attention input width must be hidden size, "
                f"got {attn_input_width} for hidden size {self.hidden_size}"
            )
        self.embedding_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_enorm],
            eps=model_config.layernorm_eps,
        )
        self.hidden_norm = RMSNorm(
            weights.weights[0][W.multi_tokens_predict_hnorm],
            eps=model_config.layernorm_eps,
        )
        self.layers = nn.ModuleList(
            [
                MiniMaxM3Eagle1DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[0],
                    0,
                    hw_kernel_config=py_hw_kernel_config,
                )
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def clone_for_cuda_graph(self) -> "MiniMaxM3Eagle1Model":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.weight = self.weight
        clone.fmha_config = self.fmha_config
        clone.py_hw_kernel_config = self.py_hw_kernel_config
        clone.micro_batch_size = self.micro_batch_size
        clone.layer_num = self.layer_num
        clone.vocab_size = self.vocab_size
        clone.kv_cache = None
        clone.device_type = self.device_type
        clone.params_dict = {}
        clone.embed_tokens = self.embed_tokens
        clone.fc = self.fc
        clone.fc_input_width = self.fc_input_width
        clone.hidden_size = self.hidden_size
        clone.embedding_norm = self.embedding_norm
        clone.hidden_norm = self.hidden_norm
        clone.layers = self.layers
        clone.norm = self.norm
        return clone

    def _build_fc_input(
        self, input_embeds: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        if int(target_hidden.shape[-1]) != self.hidden_size:
            raise RuntimeError(
                "MiniMax-M3 EAGLE1 HASS draft expected target hidden width "
                f"{self.hidden_size}, got {int(target_hidden.shape[-1])}"
            )
        return torch.cat(
            [self.embedding_norm(input_embeds), self.hidden_norm(target_hidden)],
            dim=-1,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        target_hidden = inputs.input_hiddens
        if target_hidden.numel() == 0:
            raise RuntimeError("MiniMax-M3 EAGLE1 draft requires target hidden states")
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(self._build_fc_input(input_embeds, target_hidden))
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
