from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import FMHAType
from rtp_llm.ops.compute_ops import KVCache, ParamsBase, PyAttentionInputs


class FMHAImplBase(object):
    """Flash Multi-Head Attention 实现的基础类。

    该类封装了 FMHA 的核心功能，包括 RoPE、KV Cache 和注意力计算。
    """
    fmha_impl: Any
    fmha_params: ParamsBase
    rope_params: Any
    rope_kvcache_impl: Any
    write_cache_store_impl: Any
    attn_inputs: PyAttentionInputs
    support_: bool = False

    def __init__(
        self,
        fmha_impl: Any,
        rope_kvcache_impl: Any,
        attn_inputs: PyAttentionInputs,
        init_params: bool = True,
    ) -> None:
        """初始化 FMHA 实现。

        Args:
            fmha_impl: FMHA 的具体实现对象
            rope_kvcache_impl: RoPE 和 KV Cache 的实现对象
            attn_inputs: 注意力计算的输入参数
            init_params: 是否初始化参数，默认为 True
        """
        self.fmha_impl = fmha_impl
        self.input_lengths = attn_inputs.input_lengths
        self.cu_seq_lens = attn_inputs.cu_seqlens
        self.support_: bool = self.fmha_impl.support(attn_inputs)
        self.fmha_params = None
        self.rope_params = None
        self.write_cache_store_impl = None
        if self.support_ and init_params:
            self.rope_kvcache_impl = rope_kvcache_impl
            self.attn_inputs = attn_inputs
            if self.attn_inputs.is_prefill and self.attn_inputs.cache_store_inputs:
                self.write_cache_store_impl = WriteCacheStoreOp(
                    self.attn_inputs.input_lengths,
                    self.attn_inputs.prefix_lengths,
                    self.attn_inputs.kv_cache_block_id_host,
                    self.attn_inputs.cache_store_inputs,
                )
            self.create_params(attn_inputs)

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        """执行前向传播计算。

        执行注意力计算的前向传播，包括：
        1. 可选的 RoPE 和 KV Cache 处理
        2. 预填充阶段的缓存存储（如果启用）
        3. FMHA 核心计算

        Args:
            qkv: 输入的 QKV 张量，形状为 [batch_size, seq_len, hidden_size * 3]
            kv_cache: 可选的 KV Cache，用于存储历史键值对
            need_rope_kv_cache: 是否需要应用 RoPE 和 KV Cache 处理，默认为 True

        Returns:
            计算后的注意力输出张量
        """
        assert self.rope_kvcache_impl is not None and self.rope_params is not None
        if need_rope_kv_cache:
            fmha_input = self.rope_kvcache_impl.forward(qkv, kv_cache, self.rope_params)
        else:
            fmha_input = qkv
        if (
            self.attn_inputs.is_prefill
            and self.attn_inputs.cache_store_inputs
            and self.write_cache_store_impl is not None
        ):
            self.write_cache_store_impl(kv_cache)
        assert self.fmha_impl is not None
        res = self.fmha_impl.forward(fmha_input, kv_cache, self.fmha_params)
        return res

    @staticmethod
    def fmha_type() -> FMHAType:
        """返回 FMHA 类型。

        Returns:
            FMHA 类型，基类默认返回 NONE
        """
        return FMHAType.NONE

    def create_params(self, attn_inputs: PyAttentionInputs):
        """创建 FMHA 和 RoPE 的计算参数。

        根据输入参数准备 FMHA 和 RoPE 所需的计算参数。

        Args:
            attn_inputs: 注意力计算的输入参数
        """
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)

    def support(self):
        """检查当前实现是否支持给定的输入。
           self.support_ is set in __init__
        Returns:
            bool: 如果支持则返回 True，否则返回 False
        """
        return self.support_

    def support_cuda_graph(self) -> bool:
        """检查是否支持 CUDA Graph 优化。

        通过检查是否存在 prepare_cuda_graph 方法来判断是否支持 CUDA Graph。

        Returns:
            bool: 如果支持 CUDA Graph 则返回 True，否则返回 False
        """
        attr = getattr(self, "prepare_cuda_graph", None)
        return callable(attr)

    def _update_trt_params(self, attn_inputs: PyAttentionInputs):
        """更新 trt 相关的参数。

        根据新的输入参数更新 FMHA 和 RoPE 的参数，并保持 KV Cache offset 的一致性。
        主要用于 cuda graph 的参数更新场景。

        Args:
            attn_inputs: 新的注意力计算输入参数
        """
        new_fmha_params = self.fmha_impl.prepare(attn_inputs)
        new_offset = new_fmha_params.kv_cache_offset
        old_offset = self.fmha_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

        new_rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
        new_offset = new_rope_params.kv_cache_offset
        old_offset = self.rope_params.kv_cache_offset
        self.copy_kv_cache_offset(old_offset, new_offset)

    def copy_kv_cache_offset(self, old_offset: torch.Tensor, new_offset: torch.Tensor):
        """复制 KV Cache offset 数据。
        用于cuda graph 的参数更新场景。
        将新的 offset 数据复制到旧的 offset 张量中。如果形状相同则直接复制，
        否则只复制能够匹配的部分（从第一个维度开始切片）。

        Args:
            old_offset: 目标 offset 张量，数据将被更新
            new_offset: 源 offset 张量，提供新的数据
        """
        if new_offset.shape == old_offset.shape:
            old_offset.copy_(new_offset, non_blocking=True)
        else:
            # Build slice indices dynamically
            slice_indices = [
                slice(0, new_offset.size(dim)) for dim in range(new_offset.dim())
            ]
            target_slice = old_offset[tuple(slice_indices)]
            target_slice.copy_(new_offset, non_blocking=True)
