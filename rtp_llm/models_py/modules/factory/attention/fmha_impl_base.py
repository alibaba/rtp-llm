from abc import ABC, abstractmethod
from typing import Optional

import torch

from rtp_llm.ops import AttentionConfigs
from rtp_llm.ops.compute_ops import KVCache, PyAttentionInputs


class FMHAImplBase(ABC):
    """Flash Multi-Head Attention 实现的接口基类。

    该类定义了 FMHA 实现必须提供的接口方法。
    所有具体的实现类都应该继承此类并实现这些方法。
    """

    @abstractmethod
    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[KVCache],
        need_rope_kv_cache: bool = True,
    ) -> torch.Tensor:
        """执行前向传播计算。

        Args:
            qkv: 输入的 QKV 张量
            kv_cache: 可选的 KV Cache，用于存储历史键值对
            need_rope_kv_cache: 是否需要应用 RoPE 和 KV Cache 处理

        Returns:
            计算后的注意力输出张量
        """
        raise NotImplementedError("Subclass must implement forward method")

    @staticmethod
    @abstractmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        """检查当前实现是否支持给定的输入。

        Args:
            attn_configs: 注意力配置
            attn_inputs: 注意力输入参数

        Returns:
            bool: 如果支持则返回 True，否则返回 False
        """
        return False

    @abstractmethod
    def support_cuda_graph(self) -> bool:
        """检查是否支持 CUDA Graph 优化。

        Returns:
            bool: 如果支持 CUDA Graph 则返回 True，否则返回 False
            如果想支持cuda graph需要子类支持prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):这个函数
        """
        return callable(getattr(self, "prepare_cuda_graph", None))

    # def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
    #     pass
    
