"""HY V4 four-channel decoder implemented on RTP-LLM MLA/MoE modules."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import DenseMLP, Embedding, FMHAImplBase, MlaAttention, RMSNorm
from rtp_llm.models_py.modules.hy_v4 import Hy4IHCHead, Hy4IHCUnit
from rtp_llm.ops import HWKernelConfig, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.dsa_indexing import dsa_layer_has_indexer, dsa_layer_skips_topk
from rtp_llm.utils.model_weight import W


_IHC_MXFP8_QUANT_ENV = "RTP_LLM_HY4_IHC_MXFP8_QUANT"


def _ihc_mxfp8_quant_enabled(hw_kernel_config: Optional[HWKernelConfig]) -> bool:
    from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

    if not fuse_kernels_enabled(hw_kernel_config):
        return False
    requested = os.environ.get(_IHC_MXFP8_QUANT_ENV, "auto").strip().lower()
    if requested in ("0", "false", "off", "no"):
        return False
    if requested not in ("", "auto", "1", "true", "on", "yes"):
        raise ValueError(
            f"invalid {_IHC_MXFP8_QUANT_ENV}={requested!r}; expected auto, 0, or 1"
        )
    return True


@dataclass
class Hy4LayerOutput:
    channels: torch.Tensor
    topk_indices: Optional[torch.Tensor]


class Hy4DecoderLayer(nn.Module):
    """One HY V4 layer with independent attention and MLP iHC boundaries."""

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
        hw_kernel_config: Optional[HWKernelConfig] = None,
    ) -> None:
        super().__init__()
        if not config.enable_ihc:
            raise ValueError("Hy4DecoderLayer is only valid with iHC enabled")
        required_attention = (W.attn_gate_w, W.hy4_attn_sink)
        missing_attention = [key for key in required_attention if key not in weights]
        if missing_attention:
            raise KeyError(
                f"HY V4 layer {layer_idx} is missing required gated/sink MLA "
                f"weights: {missing_attention}"
            )
        self.layer_idx = layer_idx
        self.self_attn = MlaAttention(
            config.attn_config,
            parallelism_config,
            weights,
            layer_idx,
            config.layernorm_eps,
            config.quant_config,
            hw_kernel_config,
            global_weights=global_weights,
            has_indexer=dsa_layer_has_indexer(config, layer_idx),
            reuse_topk_indices=dsa_layer_skips_topk(config, layer_idx),
            indexer_layernorm_eps=config.indexer_layernorm_eps,
            indexer_scale_fmt=config.indexer_scale_fmt,
            indexer_use_hadamard=config.indexer_use_hadamard,
        )
        if layer_idx in config.moe_layer_index:
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
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )
        ihc_args = dict(
            hidden_size=config.hidden_size,
            hc_mult=config.hc_mult,
            magnitude=config.hc_magnitude,
            hc_eps=config.hc_eps,
            norm_eps=config.layernorm_eps,
        )
        self.attn_ihc = Hy4IHCUnit(weights, kind="attn", **ihc_args)
        self.mlp_ihc = Hy4IHCUnit(weights, kind="mlp", **ihc_args)
        fuse_ihc_mxfp8 = _ihc_mxfp8_quant_enabled(hw_kernel_config)
        self._fuse_attn_ihc_mxfp8 = bool(
            fuse_ihc_mxfp8 and self.self_attn.accepts_mxfp8_input
        )
        mlp_mxfp8_consumer = self.mlp
        if isinstance(self.mlp, GenericMoeLayer):
            mlp_mxfp8_consumer = self.mlp.shared_expert
        self._fuse_mlp_ihc_mxfp8 = bool(
            fuse_ihc_mxfp8
            and mlp_mxfp8_consumer is not None
            and getattr(mlp_mxfp8_consumer, "accepts_mxfp8_input", False)
        )

    def clone_for_cuda_graph(self) -> "Hy4DecoderLayer":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.layer_idx = self.layer_idx
        clone.self_attn = self.self_attn
        clone.mlp = (
            self.mlp.clone_for_cuda_graph()
            if hasattr(self.mlp, "clone_for_cuda_graph")
            else self.mlp
        )
        clone.input_layernorm = self.input_layernorm
        clone.post_attention_layernorm = self.post_attention_layernorm
        clone.attn_ihc = self.attn_ihc
        clone.mlp_ihc = self.mlp_ihc
        clone._fuse_attn_ihc_mxfp8 = self._fuse_attn_ihc_mxfp8
        clone._fuse_mlp_ihc_mxfp8 = self._fuse_mlp_ihc_mxfp8
        return clone

    def forward(
        self,
        channels: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ) -> Hy4LayerOutput:
        attn_input_fp8 = None
        attn_input_scale = None
        if getattr(self, "_fuse_attn_ihc_mxfp8", False):
            (
                attn_input,
                attn_post_gate,
                attn_input_fp8,
                attn_input_scale,
            ) = self.attn_ihc.pre_normed_mxfp8(channels, self.input_layernorm)
        else:
            attn_input, attn_post_gate = self.attn_ihc.pre_normed(
                channels, self.input_layernorm
            )
        attn_kwargs = {}
        if attn_input_fp8 is not None and attn_input_scale is not None:
            attn_kwargs = {
                "x_fp8": attn_input_fp8,
                "x_scale": attn_input_scale,
            }
        attn_output, topk_indices = self.self_attn(
            hidden_states=attn_input,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
            prev_topk_indices=prev_topk_indices,
            return_topk=True,
            **attn_kwargs,
        )
        channels = self.attn_ihc.post(attn_output, channels, attn_post_gate)

        mlp_input_fp8 = None
        mlp_input_scale = None
        if getattr(self, "_fuse_mlp_ihc_mxfp8", False):
            (
                mlp_input,
                mlp_post_gate,
                mlp_input_fp8,
                mlp_input_scale,
            ) = self.mlp_ihc.pre_normed_mxfp8(
                channels, self.post_attention_layernorm
            )
        else:
            mlp_input, mlp_post_gate = self.mlp_ihc.pre_normed(
                channels, self.post_attention_layernorm
            )
        if mlp_input_fp8 is not None and mlp_input_scale is not None:
            mlp_output = self.mlp(
                mlp_input, x_fp8=mlp_input_fp8, x_scale=mlp_input_scale
            )
        else:
            mlp_output = self.mlp(mlp_input)
        channels = self.mlp_ihc.post(mlp_output, channels, mlp_post_gate)
        return Hy4LayerOutput(channels, topk_indices)


class Hy4Model(GptModelBase):
    """HY V4 backbone; the residual state remains four-channel until the head."""

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
        if not model_config.enable_ihc or model_config.hc_mult != 4:
            raise ValueError("HY V4 backbone requires four-channel iHC")
        if int(getattr(parallelism_config, "pp_size", 1)) > 1:
            raise ValueError(
                "HY V4 pipeline parallelism is unsupported until four-channel "
                "iHC residuals are transported between pipeline stages"
            )
        self.hc_mult = model_config.hc_mult
        self.hidden_size = model_config.hidden_size
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        enable_cuda_graph = bool(
            py_hw_kernel_config is not None
            and py_hw_kernel_config.enable_cuda_graph
        )
        self.layers = nn.ModuleList(
            [
                Hy4DecoderLayer(
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
        self.ihc_head = Hy4IHCHead(
            weights.global_weights,
            hidden_size=model_config.hidden_size,
            hc_mult=model_config.hc_mult,
            hc_eps=model_config.hc_eps,
            norm_eps=model_config.layernorm_eps,
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma),
            eps=model_config.layernorm_eps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)
        if hidden_states.dim() != 2:
            hidden_states = hidden_states.reshape(-1, self.hidden_size)
        channels = (
            hidden_states.unsqueeze(1)
            .expand(-1, self.hc_mult, -1)
            .contiguous()
        )
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        prev_topk_indices = None
        for idx, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, idx)
            output = decoder_layer(
                channels,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(idx) if self.kv_cache else None,
                prev_topk_indices=prev_topk_indices,
            )
            channels = output.channels
            prev_topk_indices = output.topk_indices

        hidden_states = self.ihc_head(channels)
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Hy4DecoderLayer", "Hy4LayerOutput", "Hy4Model"]
