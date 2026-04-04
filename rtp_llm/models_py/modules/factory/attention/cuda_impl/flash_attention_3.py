"""Dao-AILab Flash Attention 3 prefill implementation for SM90+ (Hopper).

Uses flash_attn_interface.flash_attn_varlen_func from the standalone FA3 package.
"""

import logging
from typing import Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.cuda_impl.flashinfer_rotary_emb import (
    MhaRotaryEmbeddingOp,
)
from rtp_llm.models_py.modules.factory.attention.cuda_impl.kv_cache_write_op import (
    KVCacheWriteOp,
)
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, FMHAType, ParallelismConfig, RopeStyle
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

_HAS_FLASH_ATTN_3 = False
try:
    from flash_attn_interface import flash_attn_varlen_func as fa3_varlen_func

    _HAS_FLASH_ATTN_3 = True
except ImportError:
    pass


def _is_sm90_or_above() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major >= 9


class FlashAttention3PrefillImpl(FMHAImplBase):
    """Dao-AILab Flash Attention 3 prefill backend (SM90+ Hopper).

    Directly calls flash_attn_interface.flash_attn_varlen_func for
    variable-length causal attention on Hopper GPUs.
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.head_num = attn_configs.head_num
        self.kv_head_num = attn_configs.kv_head_num
        self.head_dim = attn_configs.size_per_head
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache

        if self.need_rope_kv_cache and attn_configs.rope_config.style != RopeStyle.No:
            self.rope_impl = MhaRotaryEmbeddingOp(attn_configs)
        else:
            self.rope_impl = None

        self.kv_cache_write_op = KVCacheWriteOp(
            num_kv_heads=attn_configs.kv_head_num,
            head_size=attn_configs.size_per_head,
            token_per_block=attn_configs.kernel_tokens_per_block,
        )
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    @staticmethod
    def fmha_type() -> FMHAType:
        return FMHAType.FLASH_ATTENTION_3

    @staticmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        if not _HAS_FLASH_ATTN_3:
            return False
        if not _is_sm90_or_above():
            return False
        if not attn_inputs.is_prefill:
            return False
        if attn_configs.size_per_head not in (64, 128, 192):
            return False
        if attn_configs.use_mla:
            return False
        return True

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # Split packed QKV into Q, K, V
        qkv_flat = qkv.reshape(qkv.shape[0], -1)
        q, k, v = torch.split(
            qkv_flat,
            [
                self.head_dim * self.head_num,
                self.head_dim * self.kv_head_num,
                self.head_dim * self.kv_head_num,
            ],
            dim=-1,
        )
        query = q.reshape(q.shape[0], self.head_num, self.head_dim)
        key = k.reshape(k.shape[0], self.kv_head_num, self.head_dim)
        value = v.reshape(v.shape[0], self.kv_head_num, self.head_dim)

        # Apply RoPE if needed
        if self.need_rope_kv_cache and self.rope_impl is not None:
            query, key, value = self.rope_impl.forward(qkv)

        # Write KV to cache
        if self.need_rope_kv_cache:
            self.kv_cache_write_op.forward(key, value, kv_cache)

        # Apply write cache store if needed
        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        # Build cu_seqlens
        batch_size = self.attn_inputs.input_lengths.size(0)
        cu_seqlens = self.attn_inputs.cu_seqlens[: batch_size + 1].to(torch.int32)
        max_seqlen = self.attn_inputs.input_lengths.max().item()

        # Call FA3
        output = fa3_varlen_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=True,
        )
        # fa3_varlen_func returns (output, softmax_lse) or just output depending on version
        if isinstance(output, tuple):
            output = output[0]

        # Reshape output: [total_tokens, num_heads, head_dim] -> [total_tokens, num_heads * head_dim]
        return output.reshape(output.shape[0], -1)
