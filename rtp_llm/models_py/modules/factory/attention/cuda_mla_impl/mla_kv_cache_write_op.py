"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch

from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import KVCache, rtp_llm_ops


class MlaKVCacheWriteOp:
    """Write compressed KV cache for Multi-Latent Attention."""

    def __init__(
        self,
        kv_cache_dtype: KvCacheDataType,
    ) -> None:
        self.kv_cache_type = (
            "fp8_ds_mla" if kv_cache_dtype == KvCacheDataType.FP8 else "auto"
        )
        # Scale tensor is required for concat_and_cache_mla even in non-FP8 mode
        self.scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

    def forward(
        self,
        append_ckv_t: torch.Tensor,
        key_pe: torch.Tensor,
        kv_cache: Optional[KVCache],
        fmha_params: rtp_llm_ops.SparseMlaParams,
    ) -> None:
        """Write compressed KV and position-encoded key to MLA cache.

        Args:
            append_ckv_t: Compressed KV tensor to append [num_tokens, kv_lora_rank]
            key_pe: Position-encoded key tensor [num_tokens, rope_head_dim]
            kv_cache: MLA KV cache with compressed layout
        """
        if kv_cache is not None:
            compute_ops.concat_and_cache_mla(
                append_ckv_t,
                key_pe,
                kv_cache.kv_cache_base,
                fmha_params.slot_mapping,
                self.kv_cache_type,
                self.scale,
            )
