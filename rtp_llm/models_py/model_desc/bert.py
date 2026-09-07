from typing import Any, Dict, Optional, Sequence

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


def _validate_bert_uqi_runtime(
    config: ModelConfig, device_resource_config: Any
) -> None:
    """Reject execution modes that cannot preserve UQI request metadata."""
    if (
        config.bert_uqi_config.enabled
        and device_resource_config is not None
        and device_resource_config.enable_layer_micro_batch != 0
    ):
        raise ValueError(
            "BERT UQI attention does not support layer micro-batching; "
            "set enable_layer_micro_batch=0"
        )


def _prepare_multimodal_input_ids(
    input_ids: torch.Tensor,
    multimodal_features: Sequence[torch.Tensor],
    multimodal_locs: Optional[torch.Tensor],
    text_tokens_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[torch.Tensor] | list[int]]:
    """Replace only multimodal placeholder IDs before word embedding lookup.

    Multimodal IDs may be negative or unbounded feature hashes. Zeroing their
    exact spans avoids an invalid embedding-table access without silently
    clamping malformed text token IDs elsewhere in the request.
    """
    if not multimodal_features:
        return input_ids, multimodal_locs
    if multimodal_locs is None or multimodal_locs.numel() != len(multimodal_features):
        raise ValueError(
            "multimodal feature and location counts must match before BERT embedding"
        )
    if input_ids.dim() != 1:
        raise ValueError("BERT multimodal input_ids must be a packed 1-D tensor")

    locs = multimodal_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
    for index, (feature, loc) in enumerate(zip(multimodal_features, locs)):
        if feature is None or feature.numel() == 0:
            continue
        if feature.dim() != 2:
            raise ValueError(
                f"multimodal feature[{index}] must have shape [tokens, hidden_size]"
            )
        length = feature.size(0)
        if loc < 0 or loc + length > input_ids.numel():
            raise IndexError(
                f"multimodal feature[{index}] span [{loc}, {loc + length}) is "
                f"outside {input_ids.numel()} packed BERT tokens"
            )
    if text_tokens_mask is None or text_tokens_mask.shape != input_ids.shape:
        raise ValueError("BERT multimodal text mask must match the packed input IDs")
    # The multimodal processor supplies a binary int32 mask (text=1, vision=0).
    return input_ids * text_tokens_mask, locs


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
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states = self.input_layernorm(hidden_states, residual, empty_bias)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_attention_layernorm(
            hidden_states, residual, empty_bias
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
        _validate_bert_uqi_runtime(config, device_resource_config)
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
        self._uqi_attention_op = None

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> FMHAImplBase:
        if not self.config.bert_uqi_config.enabled:
            return super().prepare_fmha_impl(inputs, is_cuda_graph)
        if is_cuda_graph:
            raise ValueError("BERT UQI attention does not support CUDA graph execution")

        attn_inputs = inputs.attention_inputs
        uqi_mask = attn_inputs.bert_uqi_mask
        if uqi_mask is None:
            raise ValueError(
                "BERT UQI attention is enabled but its batch metadata is missing"
            )
        if not attn_inputs.is_prefill:
            raise ValueError("BERT UQI attention is only valid for prefill batches")
        if uqi_mask.numel() == 0:
            return super().prepare_fmha_impl(inputs, is_cuda_graph)

        from rtp_llm.models_py.modules.factory.attention.cuda_impl.bert_uqi import (
            BertUqiAttention,
        )

        if self._uqi_attention_op is None:
            attn_configs = self.config.getAttentionConfigs(
                self.parallelism_config.get_attn_tp_size()
            )
            self._uqi_attention_op = BertUqiAttention(attn_configs)
        self._uqi_attention_op.prepare(attn_inputs, inputs.input_ids.device)
        return self._uqi_attention_op

    def forward(
        self, inputs: PyModelInputs, fmha_impl: FMHAImplBase = None
    ) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        bert_embedding_inputs = inputs.bert_embedding_inputs
        multimodal_inputs = inputs.multimodal_inputs
        # The embedding executor already owns CPU locations. Reuse them rather
        # than downloading the device mirror just to index feature spans.
        multimodal_locs = multimodal_inputs.mm_features_locs_host
        if multimodal_locs is None:
            multimodal_locs = multimodal_inputs.mm_features_locs
        input_ids, multimodal_locs = _prepare_multimodal_input_ids(
            input_ids,
            multimodal_inputs.multimodal_features,
            multimodal_locs,
            inputs.embedding_inputs.text_tokens_mask,
        )
        inputs_embeds = self.embed_tokens(
            input_ids,
            bert_embedding_inputs.combo_position_ids,
            bert_embedding_inputs.position_encoding,
            bert_embedding_inputs.combo_tokens_type_ids,
            bert_embedding_inputs.token_type_embedding,
            bert_embedding_inputs.input_embedding_scalar,
        )
        hidden_states = self.pre_decoder_layernorm(inputs_embeds)
        hidden_states = self.multimodal_embedding_injector(
            hidden_states,
            multimodal_inputs.multimodal_features,
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
