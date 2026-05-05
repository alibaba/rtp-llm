from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    FMHAImplBase,
    RMSNorm,
)
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        layer_idx: int,
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
            layer_idx,
        )
        self.mlp = DenseMLP(
            config.activation_type,
            parallelism_config,
            weights,
            quant_config,
            hw_kernel_config,
        )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        # === ROCm fused path detection (kept None on cuda; fully isolated) ===
        # Enable only if ALL 4 Linear projections are RocmFp8PTPCLinear, so
        # the upstream RMSNorm + per-token quant fusion pays off end-to-end
        # without any duplicate quant or BF16 fallback in the middle.
        self._use_rocm_fused = False
        self._input_layernorm_fused = None
        self._post_attn_layernorm_fused = None
        if get_device_type() == DeviceType.ROCm:
            try:
                from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
                    RocmFp8PTPCLinear,
                )
                from rtp_llm.models_py.modules.base.rocm.norm import (
                    RMSNormFusedQuant,
                    RMSResNormFusedQuant,
                )
                if (
                    isinstance(self.self_attn.qkv_proj, RocmFp8PTPCLinear)
                    and isinstance(self.self_attn.o_proj, RocmFp8PTPCLinear)
                    and isinstance(self.mlp.up_proj, RocmFp8PTPCLinear)
                    and isinstance(self.mlp.down_proj, RocmFp8PTPCLinear)
                ):
                    self._use_rocm_fused = True
                    self._input_layernorm_fused = RMSNormFusedQuant(
                        weights[W.pre_ln_gamma], eps=config.layernorm_eps
                    )
                    self._post_attn_layernorm_fused = RMSResNormFusedQuant(
                        weights[W.post_ln_gamma], eps=config.layernorm_eps
                    )
            except ImportError:
                # ROCm Linear impl or new fused norm classes missing -> fall back
                pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        if self._use_rocm_fused:
            return self._forward_rocm_fused(hidden_states, fmha_impl, kv_cache)
        return self._forward_legacy(hidden_states, fmha_impl, kv_cache)

    def _forward_legacy(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def _forward_rocm_fused(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:
        # ROCm-only fused path (selected by self._use_rocm_fused).
        # Replaces 4 separate kernels with 2 fused ones per layer:
        #   - input_layernorm + per-token fp8 quant -> 2-in-1
        #   - residual_add + post_attn_layernorm + per-token fp8 quant -> 3-in-1
        # Self-Attention and MLP submodules are inlined so qkv_proj / up_proj
        # can consume pre-quantized fp8 inputs (skipping their internal quant).

        # === 1. input_layernorm + per-token fp8 quant (2-in-1) ===
        residual = hidden_states  # bf16 residual carried through this layer
        h_fp8, h_scale = self._input_layernorm_fused(hidden_states)

        # === 2. Self-Attention (inlined to feed pre-quantized input) ===
        attn_module = self.self_attn
        input_shape = hidden_states.shape[:-1]
        qkv = attn_module.qkv_proj.forward_prequantized(h_fp8, h_scale)
        if attn_module.qk_fuse_norm is not None:
            qkv = attn_module.qk_fuse_norm(qkv)
        attn_out = fmha_impl.forward(qkv, kv_cache, attn_module.layer_idx)
        attn_out = attn_out.reshape(*input_shape, -1).contiguous()
        attn_out = attn_module.o_proj(attn_out)
        if attn_module.tp_size > 1:
            attn_out = all_reduce(attn_out, group=Group.TP)

        # === 3. residual_add + post_attn_layernorm + per-token fp8 quant (3-in-1) ===
        # residual_out_bf16 = residual + attn_out (carried as next residual)
        h_fp8_2, h_scale_2, residual = self._post_attn_layernorm_fused(
            attn_out, residual
        )

        # === 4. MLP (inlined to feed pre-quantized input) ===
        mlp_module = self.mlp
        up = mlp_module.up_proj.forward_prequantized(h_fp8_2, h_scale_2)
        activated = mlp_module.act_fn(up)
        mlp_out = mlp_module.down_proj(activated)
        if mlp_module.parallelism_config.get_ffn_tp_size() > 1:
            mlp_out = all_reduce(mlp_out, group=Group.TP)

        # === 5. final residual add (layer boundary, cannot be fused) ===
        return residual + mlp_out


class Qwen3Model(GptModelBase):
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

        self.embed_tokens = Embedding(
            config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    parallelism_config,
                    idx,
                    weights.weights[idx],
                    quant_config,
                    py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = [
    "Qwen3Model",
]
