"""CUDA XQA decode op backed by rtp_kernel.cuda_xqa (DeepJIT)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Optional

import torch

from librtp_compute_ops import LayerKVCache, PyAttentionInputs
from libth_transformer_config import AttentionConfigs, KvCacheDataType
from rtp_llm.models_py.utils.arch import get_sm, is_sm12x


@cache
def _cuda_xqa():
    from rtp_kernel.cuda_xqa import run_xqa, support_xqa
    from rtp_kernel.fused_rope_kvcache import convert_offset_to_block_array

    return run_xqa, support_xqa, convert_offset_to_block_array


def _sm() -> int:
    major, minor = get_sm()
    return major * 10 + minor


def _max_seq_len(attn_configs: AttentionConfigs, sequence_lengths: torch.Tensor) -> int:
    if sequence_lengths is None or sequence_lengths.numel() == 0:
        return 0
    if sequence_lengths.is_cuda:
        return int(attn_configs.max_seq_len)
    return int(sequence_lengths.max().item())


@dataclass
class XQAParams:
    kv_cache_offset: torch.Tensor
    sequence_lengths: torch.Tensor
    batch_size: int
    max_seq_len: int
    max_blocks_per_seq: int
    is_kv_cache_fp8: bool


class XQAAttnOp:
    """Tensor-form of the former C++ XQAAttnOp, calling rtp_kernel.cuda_xqa."""

    def __init__(self, attn_configs: AttentionConfigs) -> None:
        self.attn_configs = attn_configs

    def support(self, attn_inputs: PyAttentionInputs) -> bool:
        if is_sm12x() or _sm() < 90:
            return False
        group_size = self.attn_configs.head_num // self.attn_configs.kv_head_num
        try:
            _, support_xqa, _ = _cuda_xqa()
            return bool(
                support_xqa(
                    group_size,
                    self.attn_configs.size_per_head,
                    self.attn_configs.kernel_tokens_per_block,
                    input_dtype=torch.bfloat16,
                    output_dtype=torch.bfloat16,
                    kv_dtype=torch.float8_e4m3fn,
                )
            )
        except Exception:
            # CUDA 12 wheels and hosts without DeepJIT still need a support=False
            # path rather than crashing factory dispatch.
            return False

    def prepare(self, attn_inputs: PyAttentionInputs) -> XQAParams:
        block_ids = attn_inputs.kv_cache_kernel_block_id_device
        if block_ids is None or not block_ids.is_cuda:
            raise ValueError("XQAAttnOp expects CUDA kv_cache_kernel_block_id_device")
        _, _, convert_offset = _cuda_xqa()
        kv_cache_offset = convert_offset(block_ids)
        use_fp8 = self.attn_configs.kv_cache_dtype == KvCacheDataType.FP8
        return XQAParams(
            kv_cache_offset=kv_cache_offset,
            sequence_lengths=attn_inputs.sequence_lengths,
            batch_size=int(block_ids.size(0)),
            max_seq_len=_max_seq_len(self.attn_configs, attn_inputs.sequence_lengths),
            max_blocks_per_seq=int(block_ids.size(1)),
            is_kv_cache_fp8=use_fp8,
        )

    def update(self, params: XQAParams, attn_inputs: PyAttentionInputs) -> None:
        block_ids = attn_inputs.kv_cache_kernel_block_id_device
        self.update_kv_cache_offset(params.kv_cache_offset, block_ids)
        params.batch_size = int(block_ids.size(0))
        params.sequence_lengths = attn_inputs.sequence_lengths

    def update_kv_cache_offset(
        self, kv_cache_offset: torch.Tensor, kv_cache_block_id_device: torch.Tensor
    ) -> None:
        _, _, convert_offset = _cuda_xqa()
        filled = convert_offset(kv_cache_block_id_device)
        kv_cache_offset.copy_(filled)

    def forward(
        self,
        input: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        params: XQAParams,
    ) -> torch.Tensor:
        if kv_cache is None:
            raise ValueError("decode should have kv cache.")
        run_xqa, _, _ = _cuda_xqa()
        return run_xqa(
            input,
            kv_cache.kv_cache_base,
            params.kv_cache_offset,
            params.sequence_lengths,
            num_kv_heads=self.attn_configs.kv_head_num,
            page_size=self.attn_configs.kernel_tokens_per_block,
            is_kv_cache_fp8=params.is_kv_cache_fp8,
            num_q_heads=self.attn_configs.head_num,
            max_seq_len=params.max_seq_len + 1,
        )
