"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Any, Optional

import torch

from rtp_llm.models.kimi_k3.mla_cache_tp import (
    kimi_k3_mla_cache_layout,
    mla_cache_tp_enabled,
)
from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache


class MlaKVCacheWriteOp:
    """Write compressed KV cache for Multi-Latent Attention.

    ``clear_page_on_boundary`` clears one kernel-visible page before writing
    its first token. Graph decode backends that use a static capture shape can
    read the unwritten tail of a newly allocated page, so they must opt in.
    """

    def __init__(
        self,
        kv_cache_dtype: KvCacheDataType,
        clear_page_on_boundary: bool = False,
        parallelism_config: Any = None,
    ) -> None:
        self.kv_cache_type = (
            "fp8_ds_mla" if kv_cache_dtype == KvCacheDataType.FP8 else "auto"
        )
        # Scale tensor is required for concat_and_cache_mla even in non-FP8 mode.
        # Initialize it directly on the device: torch.tensor(1.0, device="cuda")
        # stages the Python scalar through pageable host memory and synchronizes
        # the current stream on every transient MLA implementation build.
        self.scale = torch.ones((), dtype=torch.float32, device="cuda")
        self.clear_page_on_boundary = clear_page_on_boundary
        self.parallelism_config = parallelism_config
        if (
            kv_cache_dtype == KvCacheDataType.FP8
            and mla_cache_tp_enabled(parallelism_config)
        ):
            raise ValueError(
                "KIMI_K3_MLA_CACHE_TP flat-72 ABI currently supports BF16 only"
            )

    def forward(
        self,
        append_ckv_t: torch.Tensor,
        key_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: Any,
        total_global_ids: torch.Tensor = None,
        slot_mapping_override: Optional[torch.Tensor] = None,
    ) -> None:
        """Write compressed KV and position-encoded key to MLA cache.

        Args:
            append_ckv_t: Compressed KV tensor to append [num_tokens, kv_lora_rank]
            key_pe: Position-encoded key tensor [num_tokens, rope_head_dim]
            kv_cache: MLA KV cache with compressed layout
        """
        if kv_cache is not None:
            if mla_cache_tp_enabled(self.parallelism_config):
                layout = kimi_k3_mla_cache_layout(self.parallelism_config)
                full_cache = torch.cat((append_ckv_t, key_pe), dim=-1)
                append_ckv_t = layout.shard_full_cache(full_cache)
                key_pe = append_ckv_t.new_empty(
                    (append_ckv_t.shape[0], 0)
                )
                if kv_cache.kv_cache_base.shape[-1] != layout.local_width:
                    raise RuntimeError(
                        "K3 MLA cache TP physical width mismatch: "
                        f"cache={kv_cache.kv_cache_base.shape[-1]} "
                        f"expected={layout.local_width}"
                    )
            slot_mapping = (
                slot_mapping_override
                if slot_mapping_override is not None
                else fmha_params.slot_mapping
            )
            compute_ops.concat_and_cache_mla(
                append_ckv_t,
                key_pe,
                kv_cache.kv_cache_base,
                (
                    slot_mapping
                    if total_global_ids is None
                    else slot_mapping[total_global_ids]
                ),
                self.kv_cache_type,
                self.scale,
                self.clear_page_on_boundary,
            )
