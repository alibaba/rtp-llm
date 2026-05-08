"""Ascend attention implementations using PyTorch native SDPA."""

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, KvCacheDataType, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


class AscendSDPAPrefillImpl(FMHAImplBase):
    """Ascend prefill FMHA using PyTorch native scaled_dot_product_attention."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: Any,
        cos_sin_cache: Optional[torch.Tensor] = None,
        fmha_config: Optional[Any] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.parallelism_config = parallelism_config
        self.head_num = attn_configs.head_num
        self.head_num_kv = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.is_causal = attn_configs.is_causal

    @staticmethod
    def support(
        attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return attn_inputs.is_prefill

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        batch_size = q.shape[0]
        n_q_heads = self.head_num
        n_kv_heads = self.head_num_kv
        head_dim = self.head_dim

        q = q.view(batch_size, -1, n_q_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, n_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, n_kv_heads, head_dim).transpose(1, 2)

        attn_mask = None
        if self.is_causal:
            attn_mask = None  # SDPA handles causal internally

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=self.is_causal
        )
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, n_q_heads * head_dim)
        return output


class AscendSDPADecodeImpl(FMHAImplBase):
    """Ascend decode FMHA using simple attention (no paged attention).

    This uses a simple contiguous KV-cache approach for decode.
    Performance-senstive deployments should replace with torch_npu paged attention.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: Any,
        cos_sin_cache: Optional[torch.Tensor] = None,
        fmha_config: Optional[Any] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
        parallelism_config: Optional[ParallelismConfig] = None,
    ):
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.head_num = attn_configs.head_num
        self.head_num_kv = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head

    @staticmethod
    def support(
        attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> bool:
        return not attn_inputs.is_prefill

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        q, k, v = qkv.chunk(3, dim=-1)
        batch_size = q.shape[0]
        n_q_heads = self.head_num
        n_kv_heads = self.head_num_kv
        head_dim = self.head_dim

        q = q.view(batch_size, -1, n_q_heads, head_dim).transpose(1, 2)

        if kv_cache is not None:
            kv_cache.store_kv(layer_idx, k, v)
            k, v = kv_cache.fetch_kv(layer_idx)
        else:
            k = k.view(batch_size, -1, n_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, n_kv_heads, head_dim).transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, n_q_heads * head_dim)
        return output
