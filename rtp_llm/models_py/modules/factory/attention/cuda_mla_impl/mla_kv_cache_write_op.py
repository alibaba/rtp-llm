"""KV cache write operation for Multi-Latent Attention (MLA).

This module provides the KV cache writing operation specifically for MLA architecture,
which uses a compressed KV cache layout.
"""

from typing import Any, Optional, Tuple

import flashinfer.page as page
import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.offload.runtime import KVWrite
from rtp_llm.ops import KvCacheDataType, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache, rtp_llm_ops


@triton.jit
def _pack_write_ids(
    Source, Output, N: tl.constexpr, BLOCK_ROWS: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    slot = tl.load(Source + i, i < N, other=-1)
    # Graph dummy rows use reserved block 0. Their staging is private and their
    # writes must never reach backing storage or alias another dummy row.
    tl.store(Output + i, tl.where(slot >= BLOCK_ROWS, slot, -1).to(tl.int32), i < N)


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

    def configure_offload(self, storage, max_write_rows: int) -> None:
        if len(storage.resident_planes) != 1 or max_write_rows <= 0:
            raise ValueError(
                "MLA packing requires one native plane and positive write capacity"
            )
        native = storage.resident_planes[0]
        pages = triton.cdiv(max_write_rows, storage.rows_per_page)
        self._pack_staging = torch.empty(
            (pages, *native.shape[1:]), dtype=native.dtype, device=native.device
        )
        self._pack_slots = torch.arange(
            max_write_rows, dtype=torch.int64, device=native.device
        )
        self._pack_backing_ids = torch.empty(
            max_write_rows, dtype=torch.int32, device=native.device
        )
        self._pack_block_rows = storage.rows_per_block

    def pack(
        self, append_ckv_t: torch.Tensor, key_pe: torch.Tensor, fmha_params
    ) -> KVWrite:
        n = append_ckv_t.shape[0]
        if n > self._pack_slots.numel():
            raise ValueError("MLA write exceeds configured decode batch size")
        if n:
            compute_ops.concat_and_cache_mla(
                append_ckv_t,
                key_pe,
                self._pack_staging,
                self._pack_slots[:n],
                self.kv_cache_type,
                self.scale,
            )
            _pack_write_ids[(triton.cdiv(n, 256),)](
                fmha_params.slot_mapping,
                self._pack_backing_ids,
                n,
                self._pack_block_rows,
                256,
            )
        rows = self._pack_staging.view(-1, *self._pack_staging.shape[2:])[:n]
        return KVWrite(self._pack_backing_ids[:n], (rows,))

    def forward(
        self,
        append_ckv_t: torch.Tensor,
        key_pe: torch.Tensor,
        kv_cache: Optional[LayerKVCache],
        fmha_params: rtp_llm_ops.SparseMlaParams,
        total_global_ids: torch.Tensor = None,
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
                (
                    fmha_params.slot_mapping
                    if total_global_ids is None
                    else fmha_params.slot_mapping[total_global_ids]
                ),
                self.kv_cache_type,
                self.scale,
            )
