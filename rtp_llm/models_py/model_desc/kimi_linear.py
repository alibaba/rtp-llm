"""Kimi Linear model description.

Hybrid architecture:
  - KDA (Kimi Delta Attention) linear attention layers
  - MLA (Multi-head Latent Attention) full attention layers
  - MoE FFN with sigmoid routing (layer 1+), Dense FFN (layer 0)
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    KdaBase as KimiLinearKDABase,
    KdaDecode as KimiLinearKDADecode,
    KdaMetadata as KimiLinearMetadata,
    KdaPrefill as KimiLinearKDAPrefill,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MlaAttention,
    RMSResNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.ops import (
    AttentionConfigs,
    HWKernelConfig,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) module.

    Forward flow:
      1. qkv = in_proj_qkv(hidden)           # merged q+k+v projection
      2. beta_input = in_proj_b(hidden)       # beta gate input [token, H]
      3. forget_gate = f_b_proj(f_a_proj(hidden))  # LoRA forget gate [token, H*D]
      4. g_proj = g_b_proj(g_a_proj(hidden))  # LoRA output gate [token, H*D]
      5. conv1d on qkv
      6. KDA attention (cuLA chunk prefill, shared recurrent decode)
         - The internal forget gate applies: g = -exp(A_log) * softplus(forget_gate + dt_bias)
         - beta = sigmoid(beta_input)
      7. o_norm(attn_out, g_proj) with sigmoid activation
      8. o_proj(result) + all_reduce
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config

        # Projections
        self.in_proj_qkv = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_qkv_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.in_proj_b = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # LoRA forget gate
        self.f_a_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_f_a_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.f_b_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_f_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # LoRA output gate
        self.g_a_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_g_a_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.g_b_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_g_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads // parallelism_config.tp_size
        )

        self.prefill_kda = KimiLinearKDAPrefill(
            linear_attn_config, parallelism_config, weights
        )
        self.decode_kda = KimiLinearKDADecode(
            linear_attn_config, parallelism_config, weights
        )
        # o_norm with sigmoid activation (not SwiGLU)
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
            activation="sigmoid",
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_out_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
            or attn_meta.prefill_paged_conv_meta is not None
        ), "convolution metadata is required for prefill"

        # 1. Projections
        projected_qkv = self.in_proj_qkv(hidden_states)
        beta_input = self.in_proj_b(hidden_states)  # [token, H]
        forget_gate = self.f_b_proj(self.f_a_proj(hidden_states))  # [token, H*D]
        g_proj = self.g_b_proj(self.g_a_proj(hidden_states))  # [token, H*D]

        # 2. Prefill or decode
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            attn_output = self.prefill_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )
        else:
            attn_output = self.decode_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )

        # 3. o_norm with sigmoid gating: y = RMSNorm(attn_out) * sigmoid(g_proj)
        attn_output = self.norm(
            attn_output.reshape(-1, self.head_v_dim),
            g_proj.reshape(-1, self.head_v_dim),
        )

        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)

        # 4. Output projection + all_reduce
        attn_output = self.out_proj(attn_output)

        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)

        return attn_output


class KimiLinearDecoderLayer(nn.Module):
    """Kimi Linear decoder layer.

    Hybrid KDA/MLA attention + MoE/Dense FFN.
    Uses RMSResNorm (fused residual add + layernorm) like GenericMoeDecoderLayer.
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]

        quant_config = config.quant_config

        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = KimiLinearKDA(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            # Full MLA attention layer
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        # FFN: Dense (layer 0) or MoE (layer 1+)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
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
            )

        # RMSResNorm: fused residual add + layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: KimiLinearMetadata = KimiLinearMetadata(),
    ) -> DecodeLayerOutput:
        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention (KDA or MLA)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP (Dense or MoE)
        hidden_states = self.mlp(hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class KimiLinearModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
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
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                KimiLinearDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual capture requires a residual tensor")
        return hidden_states + residual

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual finalization requires a residual tensor")
        return self.norm(hidden_states, residual)[0]

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        full_attention_layers = (
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.layer_type != HybridAttentionType.LINEAR
        )
        return get_group_tags_for_layers(self.kv_cache, full_attention_layers)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            cu_seqlen_without_padding = attention_inputs.cu_seqlens_device
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
            )

        attn_meta = KimiLinearMetadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        capture = self.capture_context(inputs.capture_hidden_states)
        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            layer_attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, i
            )
            layer_fmha_impl = (
                None
                if decoder_layer.layer_type == HybridAttentionType.LINEAR
                else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            )
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            capture.capture_layer(i, hidden_states, residual)

        return capture.finalize(hidden_states, residual)
