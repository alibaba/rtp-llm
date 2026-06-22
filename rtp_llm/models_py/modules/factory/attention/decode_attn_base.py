"""Decode Attention 后端统一模板基类。

所有 decode attention 实现共享相同的 forward 流程：
1. 若需要 RoPE，调用 rope_impl 处理 qkv
2. 若需要 cache store，写入 kv_cache
3. 调用 fmha_impl 执行 attention forward

子类只需实现差异化的 hook 方法，无需重复 forward/init 逻辑。
"""

from abc import abstractmethod
from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.attention import common
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.ops import AttentionConfigs, ParallelismConfig
from rtp_llm.ops.compute_ops import (
    FusedRopeKVCacheDecodeOp,
    LayerKVCache,
    PyAttentionInputs,
)


class DecodeAttnImplBase(FMHAImplBase):
    """Decode attention 后端统一模板基类。

    Template Method 模式：forward 和 prepare_cuda_graph 由基类驱动，
    子类通过以下 hook 方法注入差异化逻辑：

    必须实现：
        _create_fmha_impl(attn_configs, attn_inputs) -> fmha算子实例
        _init_fmha_params(attn_inputs) -> fmha参数对象
        _prepare_cuda_graph_impl(attn_inputs) -> None

    可选覆写：
        _create_rope_impl(attn_configs) -> rope算子实例
        _update_rope_offset(attn_inputs) -> None
        support_cuda_graph() -> bool
    """

    def __init__(
        self,
        attn_configs: AttentionConfigs,
        attn_inputs: PyAttentionInputs,
        parallelism_config: Optional[ParallelismConfig] = None,
    ) -> None:
        self.need_rope_kv_cache = attn_configs.need_rope_kv_cache
        self.attn_inputs = attn_inputs

        # 子类 hook：创建底层算子
        self.fmha_impl = self._create_fmha_impl(attn_configs, attn_inputs)
        self.rope_impl = self._create_rope_impl(attn_configs)

        # 统一的参数初始化（rope_params 先于 fmha_params，因为部分后端的
        # _init_fmha_params 需要引用 rope_params.kv_cache_offset）
        self.rope_params = (
            self.rope_impl.prepare(attn_inputs) if self.rope_impl is not None else None
        )
        self.fmha_params = self._init_fmha_params(attn_inputs)
        self.write_cache_store_impl = common.create_write_cache_store_impl(attn_inputs)

    # ─── Template Methods (不要覆写) ────────────────────────────────────────

    def forward(
        self,
        qkv: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """统一的 forward 模板 - 所有 decode 后端共享。"""
        if self.need_rope_kv_cache and self.rope_impl is not None:
            qkv = self.rope_impl.forward(qkv, kv_cache, self.rope_params)

        common.apply_write_cache_store(
            self.write_cache_store_impl, self.attn_inputs, kv_cache
        )

        return self.fmha_impl.forward(qkv, kv_cache, self.fmha_params)

    def prepare_cuda_graph(self, attn_inputs: PyAttentionInputs) -> None:
        """统一的 CUDA graph 准备入口。"""
        self._prepare_cuda_graph_impl(attn_inputs)
        self._update_rope_offset(attn_inputs)

    # ─── 子类必须实现的 Hook ─────────────────────────────────────────────────

    @abstractmethod
    def _create_fmha_impl(
        self, attn_configs: AttentionConfigs, attn_inputs: PyAttentionInputs
    ) -> Any:
        """创建底层 FMHA 算子实例。"""
        ...

    @abstractmethod
    def _init_fmha_params(self, attn_inputs: PyAttentionInputs) -> Any:
        """初始化 FMHA 参数对象。在 __init__ 中被调用。"""
        ...

    @abstractmethod
    def _prepare_cuda_graph_impl(self, attn_inputs: PyAttentionInputs) -> None:
        """CUDA graph 重放时的参数更新逻辑（不含 rope offset 更新）。"""
        ...

    # ─── 可选覆写的 Hook ─────────────────────────────────────────────────────

    def _create_rope_impl(self, attn_configs: AttentionConfigs) -> Any:
        """创建 RoPE 算子实例。默认使用 FusedRopeKVCacheDecodeOp。"""
        return FusedRopeKVCacheDecodeOp(attn_configs)

    def _update_rope_offset(self, attn_inputs: PyAttentionInputs) -> None:
        """更新 RoPE 的 kv_cache_offset。默认通过 in-place copy 实现。

        某些后端（如 FlashInferTRTLLMDecodeImpl）的 kv_cache_offset 由 Triton
        kernel 在 _prepare_cuda_graph_impl 中统一更新，无需再单独 copy，
        可覆写此方法为空操作。
        """
        if self.rope_impl is None:
            return
        new_rope_params = self.rope_impl.prepare(attn_inputs)
        common.copy_kv_cache_offset(
            self.rope_params.kv_cache_offset, new_rope_params.kv_cache_offset
        )
