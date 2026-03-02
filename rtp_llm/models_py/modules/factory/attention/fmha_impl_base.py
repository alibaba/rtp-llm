from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from rtp_llm.models_py.modules.base.common.kvcache_store import WriteCacheStoreOp
from rtp_llm.ops import AttentionConfigs, FMHAConfig, FMHAType
from rtp_llm.ops.compute_ops import KVCache, ParamsBase, PyAttentionInputs


class MlaImplBase(object):
    """Base class for MLA attention implementations."""

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        weights: List[Dict[str, torch.Tensor]],
        cos_sin_cache: torch.Tensor,
        fmha_config: Optional[FMHAConfig] = None,
        use_trt_fmha: bool = False,
        quant_config: Optional[object] = None,
        max_seq_len: int = 0,
        is_cuda_graph: bool = False,
    ) -> None:
        """Initialize MLA implementation base class.

        Args:
            attn_configs: Attention configuration
            attn_inputs: Attention input tensors
            weights: Model weights
            cos_sin_cache: Cosine and sine cache for RoPE
            fmha_config: FMHA configuration
            use_trt_fmha: Whether to use TensorRT FMHA
            quant_config: Quantization configuration
            max_seq_len: Maximum sequence length
            is_cuda_graph: Whether CUDA graph is enabled
        """
        self.attn_configs = attn_configs
        self.attn_inputs = attn_inputs
        self.weights = weights
        self.cos_sin_cache = cos_sin_cache
        self.fmha_config = fmha_config
        self.use_trt_fmha = use_trt_fmha
        self.quant_config = quant_config
        self.max_seq_len = max_seq_len
        self.is_cuda_graph = is_cuda_graph
        self.fmha_params: Any = None

    @staticmethod
    def is_sparse() -> bool:
        return False

    @staticmethod
    @abstractmethod
    def support(attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        """Check if CUDA graph is supported."""
        return callable(getattr(self, "prepare_cuda_graph", None))

    def prepare(self, attn_inputs: PyAttentionInputs):
        """Prepare for attention computation."""
        pass

    def forward(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        layer_id: int,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention computation."""
        raise NotImplementedError("forward method must be implemented by subclass")


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

    # def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs):
    #     pass
