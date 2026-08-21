from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3ModelConfig
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import DenseMLP, Embedding, LinearFactory, MlaAttention, RMSNorm
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    MultimodalEmbeddingInjector,
)
from rtp_llm.models_py.modules.kimi_k3.utils import (
    mask_multimodal_token_ids,
    sequence_offsets,
)
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class _GatedEagle3MLA(MlaAttention):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx=0,
            layernorm_eps=config.layernorm_eps,
            quant_config=config.quant_config,
        )
    def _project_qkv_a_input(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        fused_qkv_gate = self.fused_qkv_a_proj(hidden_states)
        qkv_width = self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim
        return fused_qkv_gate[..., :qkv_width], fused_qkv_gate[..., qkv_width:]

    def _apply_output_gate(
        self, attn_output: torch.Tensor, output_gate: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if output_gate is None:
            raise RuntimeError("Kimi K3 Eagle3 MLA requires an output gate")
        return attn_output * torch.sigmoid(output_gate.reshape_as(attn_output))


class _KimiK3Eagle3Layer(nn.Module):
    def __init__(
        self,
        config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
    ) -> None:
        super().__init__()
        self.embedding_norm = RMSNorm(
            weights[W.eagle3_input_norm_gamma], eps=config.layernorm_eps
        )
        self.hidden_norm = RMSNorm(
            weights[W.eagle3_fc_norm_gamma], eps=config.layernorm_eps
        )
        self.attention = _GatedEagle3MLA(config, parallelism_config, weights)
        self.post_attention_norm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            config.quant_config,
        )

    def forward(self, embedding, hidden_states, fmha_impl, kv_cache):
        attention_input = torch.cat(
            (self.embedding_norm(embedding), self.hidden_norm(hidden_states)), dim=-1
        )
        hidden_states = hidden_states + self.attention(
            attention_input, fmha_impl, kv_cache
        )
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class KimiK3Eagle3Model(GptModelBase):
    """Runtime graph matching ``KimiK3MLASWAEagle3`` from the training code."""

    def __init__(
        self,
        model_config: KimiK3ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ) -> None:
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        self.embedding = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.aux_projection = LinearFactory.create_linear_from_weights(
            weights.weights[0], W.eagle3_fc_proj
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.hidden_size = model_config.hidden_size
        self.layer = _KimiK3Eagle3Layer(
            model_config, parallelism_config, weights.weights[0]
        )
        self.final_norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def _embed_shifted_multimodal(self, inputs: PyModelInputs) -> torch.Tensor:
        input_ids = inputs.input_ids
        multimodal_inputs = inputs.multimodal_inputs
        multimodal_features = multimodal_inputs.multimodal_features
        if not multimodal_features:
            return self.embedding(input_ids)

        mm_features_locs = multimodal_inputs.mm_features_locs_host
        if mm_features_locs is None or mm_features_locs.numel() != len(
            multimodal_features
        ):
            raise ValueError(
                "Kimi K3 EAGLE-3 multimodal feature locations must match the "
                "feature count"
            )

        # MTP prefill shifts each packed request one token left and appends the target
        # sample, so image features must move with it (draft reaches here on prefill).
        ranges = sequence_offsets(
            inputs.attention_inputs.cu_seqlens,
            input_ids.numel(),
            cu_seqlens_host=inputs.attention_inputs.cu_seqlens_host,
        )
        shifted_features, shifted_locs = [], []
        for feature, loc in zip(multimodal_features, mm_features_locs.tolist()):
            # loc - 1 must stay in its own request; the last row anchors the lookup
            # since loc may precede the request start across a reused prefix.
            start = max(s for s, _ in ranges if s <= loc + feature.size(0) - 1)
            dropped = max(0, start - loc + 1)  # rows with no draft slot
            shifted_features.append(feature[dropped:])
            shifted_locs.append(max(loc - 1, start))
        shifted_locs = torch.tensor(shifted_locs, dtype=torch.int32)
        input_ids = mask_multimodal_token_ids(input_ids, shifted_features, shifted_locs)
        embedding = self.embedding(input_ids)
        return self.multimodal_embedding_injector(
            embedding, shifted_features, shifted_locs
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Optional[Any] = None):
        if inputs.input_hiddens is None:
            raise ValueError("Kimi K3 EAGLE-3 requires merged auxiliary hidden states")
        embedding = self._embed_shifted_multimodal(inputs)
        hidden_width = inputs.input_hiddens.shape[-1]
        if hidden_width == self.hidden_size * 3:
            # The teacher/target pass supplies the three selected target-layer
            # states concatenated along the hidden dimension.  This is the
            # only step that goes through EAGLE-3's fc projection.
            hidden_states = self.aux_projection(inputs.input_hiddens)
        elif hidden_width == self.hidden_size:
            # Autoregressive draft steps consume the previous draft layer's
            # pre-norm hidden directly, matching KimiK3MLASWAEagle3.decode_step.
            hidden_states = inputs.input_hiddens
        else:
            raise ValueError(
                "Kimi K3 EAGLE-3 expected either three concatenated target "
                f"hidden states ({self.hidden_size * 3}) or one recurrent draft "
                f"hidden state ({self.hidden_size}), got {hidden_width}"
            )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        select_block_map_for_layer(inputs.attention_inputs, 0)
        hidden_states = self.layer(
            embedding,
            hidden_states,
            fmha_impl,
            self.kv_cache.get_layer_cache(0) if self.kv_cache else None,
        )
        return PyModelOutputs(self.final_norm(hidden_states), fmha_impl.fmha_params)


__all__ = ["KimiK3Eagle3Model"]
