from typing import Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AddBiasResLayerNorm,
    AttnImplFactory,
    CausalAttention,
    DenseMLP,
    EmbeddingBert,
    FMHAImplBase,
    LayerNorm,
    MultimodalEmbeddingInjector,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class BertDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        attn_configs = config.getAttentionConfigs(parallelism_config.get_attn_tp_size())
        self.self_attn = CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = AddBiasResLayerNorm(
            weights[W.post_ln_gamma],
            beta=weights[W.post_ln_beta],
            eps=config.layernorm_eps,
        )
        self.post_attention_layernorm = AddBiasResLayerNorm(
            weights[W.post_ffn_ln_gamma],
            beta=weights[W.post_ffn_ln_beta],
            eps=config.layernorm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        empty_bias = torch.empty(
            0, device=hidden_states.device, dtype=hidden_states.dtype
        )

        residual = hidden_states
        hidden_states, attention_bias = self.self_attn.forward_without_output_bias(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = self.input_layernorm(
            hidden_states,
            residual,
            (
                attention_bias.to(hidden_states.dtype)
                if attention_bias is not None
                else empty_bias
            ),
        )

        residual = hidden_states
        hidden_states, ffn_bias = self.mlp.forward_without_output_bias(hidden_states)
        hidden_states = self.post_attention_layernorm(
            hidden_states,
            residual,
            ffn_bias.to(hidden_states.dtype) if ffn_bias is not None else empty_bias,
        )
        return hidden_states


class BertModel(GptModelBase):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        quant_config: Optional[object] = None,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embed_tokens = EmbeddingBert(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.pre_decoder_layernorm = LayerNorm(
            weight=weights.get_global_weight(W.pre_decoder_ln_gamma),
            beta=weights.get_global_weight(W.pre_decoder_ln_beta),
            eps=config.layernorm_eps,
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.layers = nn.ModuleList(
            [
                BertDecoderLayer(
                    config,
                    parallelism_config,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )

    def forward(
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        # MultimodalProcessor writes arbitrary signed int32 feature hashes into
        # image slots for prefix-cache keys. The image features replace these
        # rows below, but embedding lookup runs first and must only see valid IDs.
        multimodal_features = inputs.multimodal_inputs.multimodal_features
        multimodal_locs = inputs.multimodal_inputs.mm_features_locs
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        has_text_tokens_mask = (
            text_tokens_mask is not None and text_tokens_mask.numel() != 0
        )
        has_multimodal_features = bool(multimodal_features)
        has_multimodal_locs = (
            multimodal_locs is not None and multimodal_locs.numel() != 0
        )
        # This is Bert-specific: image slots hold feature hashes before the
        # word-table lookup, while feature/location pairing belongs to the
        # shared injector below. Keep the check here, before any device lookup.
        if not (has_multimodal_features == has_multimodal_locs == has_text_tokens_mask):
            raise ValueError(
                "multimodal features, locations, and text_tokens_mask must be "
                "provided together: "
                f"features={len(multimodal_features)}, "
                f"locations={multimodal_locs.numel() if has_multimodal_locs else 0}, "
                f"mask_tokens={text_tokens_mask.numel() if has_text_tokens_mask else 0}"
            )

        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
            text_tokens_mask if has_text_tokens_mask else None,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)

        # Producer contract: every multimodal feature is already projected to
        # hidden_size and normalized in the first decoder layer's input space.
        # Replace image rows after the text pre-LN so producer-owned features do
        # not inherit text position/type embeddings or get normalized twice.
        hidden_states = self.multimodal_embedding_injector(
            hidden_states,
            multimodal_features,
            multimodal_locs,
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        return PyModelOutputs(hidden_states)
