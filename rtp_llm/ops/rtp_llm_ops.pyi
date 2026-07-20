"""
rtp llm custom ops
"""

from __future__ import annotations

from typing import Any, Optional

import libth_transformer
import torch

from rtp_llm.ops import KVCache, PyCacheStoreInputs

__all__ = [
    "SelectTopkOp",
    "XQAAttnOp",
    "XQAParams",
    "embedding",
    "fused_add_layernorm",
    "fused_add_rmsnorm",
    "fused_qk_rmsnorm",
    "write_cache_store",
    "FlashInferMlaAttnParams",
    "layernorm",
    "rmsnorm",
    "silu_and_mul",
]

class FlashInferMlaAttnParams:
    def __init__(self) -> None: ...
    def fill_decode_cuda_graph_params(
        self,
        sequence_lengths_plus_1_d: torch.Tensor,
        kv_cache_block_id_device: torch.Tensor,
        seq_size_per_block: int,
    ) -> None: ...

class SelectTopkOp:
    def __init__(
        self, attn_configs: typing.Any
    ) -> None: ...
    def forward(
        self,
        router_logits: torch.Tensor,
        expert_ids: torch.Tensor,
        expert_scales: torch.Tensor,
    ) -> None: ...

class XQAAttnOp:
    def __init__(
        self, attn_configs: typing.Any
    ) -> None: ...
    def forward(
        self,
        input: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        params: XQAParams,
    ) -> torch.Tensor: ...
    def prepare(
        self, attn_inputs: libth_transformer.PyAttentionInputs
    ) -> XQAParams: ...
    def update(
        self, params: XQAParams, attn_inputs: libth_transformer.PyAttentionInputs
    ) -> None: ...
    def update_kv_cache_offset(
        self, kv_cache_offset: torch.Tensor, kv_cache_block_id_device: torch.Tensor
    ) -> None: ...

class XQAParams:
    def __init__(self) -> None: ...

def embedding(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    cuda_stream: int = 0,
) -> None:
    """
    Embedding lookup kernel
    """

def fused_add_layernorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    cuda_stream: int = 0,
) -> None:
    """
    Fused Add LayerNorm kernel
    """

def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cuda_stream: int = 0,
) -> None:
    """
    Fused Add RMSNorm kernel
    """

def fused_qk_rmsnorm(
    IO: torch.Tensor,
    q_gamma: torch.Tensor,
    k_gamma: torch.Tensor,
    layernorm_eps: float,
    q_group_num: int,
    k_group_num: int,
    m: int,
    n: int,
    norm_size: int,
    cuda_stream: int = 0,
) -> None:
    """
    Fused QK RMSNorm kernel
    """

def write_cache_store(
    input_lengths: torch.Tensor,
    prefix_lengths: torch.Tensor,
    kv_cache_block_id_host: torch.Tensor,
    cache_store_member: Optional[PyCacheStoreInputs],
    kv_cache: Optional[KVCache],
    cuda_stream: int = 0,
) -> None:
    """
    WriteCacheStoreOp kernel
    """

def layernorm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    cuda_stream: int = 0,
) -> None:
    """
    LayerNorm kernel
    """

def rmsnorm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    cuda_stream: int = 0,
) -> None:
    """
    RMSNorm kernel
    """

def silu_and_mul(
    output: torch.Tensor, input: torch.Tensor, cuda_stream: int = 0
) -> None:
    """
    SiLU and Multiply kernel
    """
