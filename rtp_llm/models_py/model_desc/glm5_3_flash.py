"""Framework-facing GLM-5.3-Flash text decoder."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.base import RMSNorm
from rtp_llm.models_py.modules.base.common.embedding import Embedding
from rtp_llm.models_py.modules.base.common.kvcache_store import (
    create_write_cache_store_impl,
)
from rtp_llm.models_py.modules.dsv4.hc import build_hc_unit
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.modules.hybrid.mla_attention import MlaAttention
from rtp_llm.models_py.modules.kimi_k3.chunk_prefill import kda_materialized_block_maps
from rtp_llm.models_py.modules.kimi_k3.kda import KDAExecutionMode, KimiK3KDA
from rtp_llm.models_py.modules.kimi_k3.kda.prefill import (
    KimiKDAPrefillMetadata,
    prepare_kimi_kda_prefill_metadata,
)
from rtp_llm.models_py.modules.kimi_k3.utils import resolve_cu_seqlens
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    LayerKVCache,
    PyAttentionInputs,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W


class Glm53FlashSwiGLU(nn.Module):
    """Released GLM SwiGLU with asymmetric clamp at 10."""

    def __init__(self, limit: float) -> None:
        super().__init__()
        self.limit = float(limit)

    def forward(self, merged: torch.Tensor) -> torch.Tensor:
        gate, up = merged.chunk(2, dim=-1)
        if self.limit > 0:
            gate = torch.clamp(gate, max=self.limit)
            up = torch.clamp(up, min=-self.limit, max=self.limit)
        return F.silu(gate) * up


@dataclass(frozen=True)
class Glm53FlashDecoderMetadata:
    cu_seqlens: torch.Tensor
    mode: KDAExecutionMode
    kda_prefill_metadata: Optional[KimiKDAPrefillMetadata]


class Glm53FlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        *,
        max_generate_batch_size: int,
        enable_cuda_graph: bool,
        hw_kernel_config: Optional[Any],
    ) -> None:
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        runtime = config.glm5_3_flash_runtime_config
        self.attn_norm = RMSNorm(weights[W.pre_ln_gamma], config.layernorm_eps)
        self.ffn_norm = RMSNorm(weights[W.post_ln_gamma], config.layernorm_eps)
        self.attn_hc = build_hc_unit(
            weights[W.v4_hc_attn_fn],
            weights[W.v4_hc_attn_base],
            weights[W.v4_hc_attn_scale],
            dim=config.hidden_size,
            hc_mult=runtime.hc_mult,
            hc_sinkhorn_iters=runtime.hc_sinkhorn_iters,
            norm_eps=config.layernorm_eps,
            hc_eps=runtime.hc_eps,
            layer_id=layer_idx,
            name="glm5_attn",
        )
        self.ffn_hc = build_hc_unit(
            weights[W.v4_hc_ffn_fn],
            weights[W.v4_hc_ffn_base],
            weights[W.v4_hc_ffn_scale],
            dim=config.hidden_size,
            hc_mult=runtime.hc_mult,
            hc_sinkhorn_iters=runtime.hc_sinkhorn_iters,
            norm_eps=config.layernorm_eps,
            hc_eps=runtime.hc_eps,
            layer_id=layer_idx,
            name="glm5_ffn",
        )
        if self.is_kda:
            self.self_attn = KimiK3KDA(config, parallelism_config, weights, layer_idx)
        else:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                config.quant_config,
                hw_kernel_config,
                global_weights=global_weights,
            )

        if layer_idx in set(config.moe_layer_index):
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
            # Shared expert uses the same bounded SwiGLU. Routed-expert clamp is
            # propagated through the fused-MoE call below once supported by its
            # selected executor.
            if self.mlp.shared_expert is not None:
                self.mlp.shared_expert.act_fn = Glm53FlashSwiGLU(runtime.swiglu_limit)
        else:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                config.quant_config,
                hw_kernel_config=hw_kernel_config,
                gated_activation=Glm53FlashSwiGLU(runtime.swiglu_limit),
            )

    @property
    def is_kda(self) -> bool:
        return self.layer_type == HybridAttentionType.LINEAR

    def prepare_kda_cache_store(self, kv_cache: LayerKVCache) -> None:
        if not self.is_kda:
            raise RuntimeError("only KDA layers publish recurrent cache segments")
        kv_cache.cache_store_segment_sizes = list(
            self.self_attn.cache_store_segment_sizes
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: Glm53FlashDecoderMetadata,
        fmha_impl: Any,
        kv_cache: LayerKVCache,
        attention_inputs: PyAttentionInputs,
        global_kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        residual = hidden_states
        attn_input, post, combine = self.attn_hc.pre(hidden_states)
        attn_input = self.attn_norm(attn_input)
        if self.is_kda:
            attn_output = self.self_attn(
                attn_input,
                metadata.cu_seqlens,
                mode=metadata.mode,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                sequence_parallel=False,
                prefill_metadata=metadata.kda_prefill_metadata,
            )
        else:
            # The GLM KPool indexer uses typed side regions from global_kv_cache;
            # MlaAttention keeps the ordinary MLA cache in kv_cache.
            setattr(self.self_attn, "global_kv_cache", global_kv_cache)
            attn_output = self.self_attn(attn_input, fmha_impl, kv_cache)
        hidden_states = self.attn_hc.post(attn_output, residual, post, combine)

        residual = hidden_states
        ffn_input, post, combine = self.ffn_hc.pre(hidden_states)
        ffn_input = self.ffn_norm(ffn_input)
        ffn_output = self.mlp(ffn_input)
        return self.ffn_hc.post(ffn_output, residual, post, combine)


class Glm53FlashModel(GptModelBase):
    def __init__(
        self,
        model_config,
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
        self.embed_tokens = Embedding(
            model_config,
            parallelism_config,
            weights.get_global_weight(W.embedding),
        )
        enable_cuda_graph = bool(
            py_hw_kernel_config is not None and py_hw_kernel_config.enable_cuda_graph
        )
        if moe_config.moe_strategy == "auto":
            moe_config.moe_strategy = "mega_moe"
        logging.info("GLM-5.3 MoE strategy: %s", moe_config.moe_strategy)
        self.layers = nn.ModuleList(
            [
                Glm53FlashDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[layer],
                    weights.global_weights,
                    layer,
                    moe_config,
                    max_generate_batch_size=max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for layer in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), model_config.layernorm_eps
        )
        first_kda = next(layer.self_attn for layer in self.layers if layer.is_kda)
        self._kda_local_heads = int(first_kda.local_heads)
        self._kda_head_dim = int(first_kda.head_dim)
        self._layer_group_ids: Optional[tuple[int, ...]] = None
        self._max_generate_batch_size = int(max_generate_batch_size)

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        super().initialize(init_resource)
        if not torch.cuda.is_available():
            return True

        from rtp_llm.models_py.modules.dsv4.dsv4_kernel_jit_warmup import (
            _collect_dsv4_mhc_prenorm_shapes,
            warmup_mhc_prenorm_gemm_jit,
        )

        shapes = _collect_dsv4_mhc_prenorm_shapes(self)
        if not shapes:
            return True
        device = next(iter(shapes.values()))["fn"].device
        warmup_mhc_prenorm_gemm_jit(
            shapes,
            max_m=max(self._max_generate_batch_size, 1),
            device=device,
        )
        logging.info(
            "GLM-5.3 mHC TileLang JIT prewarm done: shapes=%s max_m=%d",
            sorted(shapes),
            max(self._max_generate_batch_size, 1),
        )
        return True

    def _prepare_kda_metadata(
        self,
        inputs: PyModelInputs,
        mode: KDAExecutionMode,
    ) -> Optional[KimiKDAPrefillMetadata]:
        attention = inputs.attention_inputs
        if mode != "prefill" or self.kv_cache is None:
            return None
        required = (
            attention.cu_seqlens_host,
            attention.input_lengths_host,
            attention.prefix_lengths_host,
        )
        if any(value is None or not value.numel() for value in required):
            raise RuntimeError("GLM-5.3 KDA Prefill requires host sequence metadata")
        maps = kda_materialized_block_maps(
            attention,
            layer_group_ids=self._layer_group_ids,
            kda_layer_indices=[
                layer.layer_idx for layer in self.layers if layer.is_kda
            ],
        )
        return prepare_kimi_kda_prefill_metadata(
            attention.cu_seqlens_host,
            attention.input_lengths_host,
            attention.prefix_lengths_host,
            page_size=int(self.kv_cache.seq_size_per_block),
            local_heads=self._kda_local_heads,
            head_dim=self._kda_head_dim,
            device=inputs.input_ids.device,
            materialized_block_maps_host=maps,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        attention = inputs.attention_inputs
        if attention is None:
            raise ValueError("GLM-5.3 requires PyAttentionInputs")
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        input_ids = inputs.input_ids.reshape(-1)
        hidden = self.embed_tokens(input_ids)
        hc_mult = self.config.glm5_3_flash_runtime_config.hc_mult
        hidden = hidden.unsqueeze(1).expand(-1, hc_mult, -1).contiguous()

        if self._layer_group_ids is None:
            host_map = getattr(attention, "kv_cache_layer_to_group_host", None)
            if host_map is not None and host_map.numel():
                self._layer_group_ids = tuple(int(value) for value in host_map.tolist())
        mode: KDAExecutionMode = "prefill" if attention.is_prefill else "decode"
        metadata = Glm53FlashDecoderMetadata(
            cu_seqlens=resolve_cu_seqlens(attention, input_ids),
            mode=mode,
            kda_prefill_metadata=self._prepare_kda_metadata(inputs, mode),
        )
        cache_writer = create_write_cache_store_impl(attention, self.kv_cache)
        for layer_idx, layer in enumerate(self.layers):
            group_id = (
                self._layer_group_ids[layer_idx]
                if self._layer_group_ids is not None
                else None
            )
            select_block_map_for_layer(attention, layer_idx, group_id)
            layer_cache = (
                self.kv_cache.get_layer_cache(layer_idx)
                if self.kv_cache is not None
                else None
            )
            if layer_cache is None:
                raise RuntimeError("GLM-5.3 requires an initialized hybrid KV cache")
            hidden = layer(
                hidden,
                metadata=metadata,
                fmha_impl=fmha_impl,
                kv_cache=layer_cache,
                attention_inputs=attention,
                global_kv_cache=self.kv_cache,
            )
            if layer.is_kda and cache_writer is not None:
                layer.prepare_kda_cache_store(layer_cache)
                cache_writer(layer_cache)

        hidden = self.norm(hidden.mean(dim=1))
        params = getattr(fmha_impl, "fmha_params", None)
        return (
            PyModelOutputs(hidden, params)
            if params is not None
            else PyModelOutputs(hidden)
        )


__all__ = ["Glm53FlashDecoderLayer", "Glm53FlashModel", "Glm53FlashSwiGLU"]
