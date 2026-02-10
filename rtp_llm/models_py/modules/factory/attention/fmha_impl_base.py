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

    def support_cuda_graph(self) -> bool:
        """检查是否支持 CUDA Graph 优化。

        Returns:
            bool: 如果支持 CUDA Graph 则返回 True，否则返回 False
            如果想支持cuda graph需要子类支持prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):这个函数
        """
        return callable(getattr(self, "prepare_cuda_graph", None))

    def prepare(self, attn_inputs: PyAttentionInputs):
        """基于当前 `attn_inputs` 初始化/刷新算子运行参数。
        
        在 hybrid attention / hybrid kv-cache 场景下，`attn_inputs.kv_cache_block_id_{host,device}`
        可能会按 layer 切换到不同的 block table（例如按 group 分配）。当 block_ids 所引用的
        tensor 被替换或需要重新绑定时，应通过该流程让底层算子拿到更新后的 block_ids。
        """
        assert self.fmha_impl is not None
        self.fmha_params = self.fmha_impl.prepare(attn_inputs)
        assert self.rope_kvcache_impl is not None
        self.rope_params = self.rope_kvcache_impl.prepare(attn_inputs)
