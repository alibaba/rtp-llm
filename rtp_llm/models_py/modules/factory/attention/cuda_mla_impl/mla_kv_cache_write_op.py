"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Any, Optional

import torch

from rtp_llm.models_py.modules.factory.linear.quantized_activation import retained_bf16
from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache

from .mla_fp8_kernels import _FP8_DIAGNOSTICS, observe_fp8_input


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
        fp8_compute: bool = False,
        kv_scale: float = 1.0,
    ) -> None:
        if fp8_compute and kv_cache_dtype != KvCacheDataType.FP8:
            raise ValueError("FP8 MLA compute requires ordinary FP8 cache")
        self.kv_cache_type = (
            "fp8_ds_mla" if kv_cache_dtype == KvCacheDataType.FP8 else "auto"
        )
        if fp8_compute:
            self.kv_cache_type = "fp8"
        # Scale tensor is required for concat_and_cache_mla even in non-FP8 mode.
        # Initialize it directly on the device: torch.tensor(1.0, device="cuda")
        # stages the Python scalar through pageable host memory and synchronizes
        # the current stream on every transient MLA implementation build.
        self.scale = torch.full((), kv_scale, dtype=torch.float32, device="cuda")
        self.clear_page_on_boundary = clear_page_on_boundary
        self.fp8_diagnostics = fp8_compute and _FP8_DIAGNOSTICS
        self.kv_scale = kv_scale

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
        append_ckv_t = retained_bf16(append_ckv_t)
        if kv_cache is not None:
            if self.fp8_diagnostics:
                observe_fp8_input(append_ckv_t, self.kv_scale, "cache_latent")
                observe_fp8_input(key_pe, self.kv_scale, "cache_suffix")
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
