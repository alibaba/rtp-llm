"""Gather DSv4 indexer K/scale rows from the per-block grouped FP8 cache.

The CP hot path reads ``block_table`` directly into the padded rank-local NCCL
layout, including zero-filled padding rows. The older slot-mapping entry point
remains available for callers that already materialize absolute cache slots.

Cache layout (per block of ``block_size`` tokens):

  bytes [0,                 block_size * 128)        = FP8 K (token-major)
  bytes [block_size * 128,  block_size * 132)        = fp32 scales (one/token)

Outputs:

  k_quant  [N, 128]  float8_e4m3fn  (contiguous; padded slots → 0)
  k_scale  [N]       float32        (contiguous; padded slots → 0)

Used by the FP8 prefill path in ``Indexer.forward`` to feed
``deep_gemm.fp8_mqa_logits``.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    _trap_invalid_kv_access,
)
from rtp_llm.models_py.modules.dsv4.fp8._trap_utils import (
    trap_invalid_kv_access_enabled,
    validate_slot_mapping,
)


@triton.jit(do_not_specialize=["N"])
def _cp_gather_indexer_k_kernel(
    cache_ptr,  # raw byte ptr into [num_blocks, block_size*132] uint8
    slot_mapping_ptr,  # [N] int64; -1 = write zeros
    k_quant_ptr,  # [N, D] float8_e4m3fn (uint8 view)
    k_scale_ptr,  # [N]    float32
    N,
    D: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_stride_b: tl.constexpr,
    num_cache_blocks: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= N:
        return

    d_off = tl.arange(0, D)
    out_k_ptr = (k_quant_ptr + pid * D + d_off).to(tl.pointer_type(tl.uint8))

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        tl.store(out_k_ptr, tl.zeros([D], dtype=tl.uint8))
        tl.store(k_scale_ptr + pid, 0.0)
        return

    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size
    if block_idx < 0:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)
    if block_idx >= num_cache_blocks:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    block_base = cache_ptr + block_idx * cache_stride_b

    # K bytes
    k_src = (block_base + block_off * D + d_off).to(tl.pointer_type(tl.uint8))
    tl.store(out_k_ptr, tl.load(k_src))

    # fp32 scale
    scale_src = (block_base + cache_block_size * D + block_off * 4).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(k_scale_ptr + pid, tl.load(scale_src))


@triton.jit(
    do_not_specialize=[
        "batch_size",
        "total_local_tokens",
        "block_table_stride_b",
        "max_blocks_per_request",
        "cache_block_size",
        "cache_stride_b",
        "num_cache_blocks",
    ]
)
def _cp_gather_indexer_k_to_padded_kernel(
    cache_ptr,
    block_table_ptr,
    padded_lens_ptr,
    actual_lens_ptr,
    out_k_ptr,
    out_scale_ptr,
    batch_size,
    total_local_tokens,
    block_table_stride_b,
    max_blocks_per_request,
    cache_block_size,
    cache_stride_b,
    num_cache_blocks,
    D: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
    BATCH_BLOCK: tl.constexpr,
    TRAP_INVALID_KV_ACCESS: tl.constexpr,
):
    """Read actual CP-owned rows directly into the padded NCCL layout."""
    token_idx = tl.program_id(0).to(tl.int64)
    if token_idx >= total_local_tokens:
        return

    batch_offsets = tl.arange(0, BATCH_BLOCK)
    batch_mask = batch_offsets < batch_size
    padded_lens = tl.load(
        padded_lens_ptr + batch_offsets, mask=batch_mask, other=0
    ).to(tl.int64)
    padded_ends = tl.cumsum(padded_lens, axis=0)
    batch_idx = tl.sum((token_idx >= padded_ends).to(tl.int32), axis=0)
    padded_start = tl.sum(
        tl.where(batch_offsets == batch_idx, padded_ends - padded_lens, 0),
        axis=0,
    )
    token_in_request = token_idx - padded_start
    actual_len = tl.sum(
        tl.where(
            batch_offsets == batch_idx,
            tl.load(actual_lens_ptr + batch_offsets, mask=batch_mask, other=0).to(
                tl.int64
            ),
            0,
        ),
        axis=0,
    )

    is_actual = token_in_request < actual_len
    logical_block_idx = token_in_request // cache_block_size
    valid_table_lookup = is_actual & (logical_block_idx < max_blocks_per_request)
    cache_block_idx = tl.load(
        block_table_ptr
        + batch_idx.to(tl.int64) * block_table_stride_b
        + logical_block_idx,
        mask=valid_table_lookup,
        other=-1,
    ).to(tl.int64)
    valid_cache_block = (
        valid_table_lookup
        & (cache_block_idx >= 0)
        & (cache_block_idx < num_cache_blocks)
    )
    if valid_table_lookup & ~valid_cache_block:
        _trap_invalid_kv_access(TRAP_INVALID_KV_ACCESS)

    block_offset = token_in_request % cache_block_size
    cache_block_base = cache_ptr + cache_block_idx * cache_stride_b
    d_offsets = tl.arange(0, D)
    cache_k_ptr = (cache_block_base + block_offset * D + d_offsets).to(
        tl.pointer_type(tl.uint8)
    )
    k_bytes = tl.load(cache_k_ptr, mask=valid_cache_block, other=0)
    out_k_row = (out_k_ptr + token_idx * D + d_offsets).to(
        tl.pointer_type(tl.uint8)
    )
    tl.store(out_k_row, k_bytes)

    cache_scale_ptr = (
        cache_block_base + cache_block_size * D + block_offset * SCALE_BYTES
    ).to(tl.pointer_type(tl.float32))
    scale = tl.load(cache_scale_ptr, mask=valid_cache_block, other=0.0)
    out_scale_row = (out_scale_ptr + token_idx * SCALE_BYTES).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(out_scale_row, scale)


def cp_indexer_padded_gather_enabled() -> bool:
    value = os.environ.get("DSV4_CP_INDEXER_GATHER_TRITON", "1").strip().lower()
    return value not in ("0", "false", "off", "no")


def try_gather_indexer_k_to_padded(
    kv_cache_packed: torch.Tensor,
    block_table: torch.Tensor,
    per_req_padded_lens: torch.Tensor,
    per_req_actual_lens: torch.Tensor,
    out_k_quant: torch.Tensor,
    out_k_scale: torch.Tensor,
    *,
    total_actual_tokens: int,
) -> bool:
    """Fuse paged-cache gather, CP padding, and the two output scatters.

    The outputs are the rank-local padded tensors consumed directly by the two
    NCCL all-gathers. Unsupported layouts return ``False`` for the C++/PyTorch
    fallback in :class:`IndexerFP8`.
    """
    if not cp_indexer_padded_gather_enabled():
        return False

    batch_size = int(per_req_padded_lens.numel())
    total_local_tokens = int(out_k_quant.shape[0]) if out_k_quant.dim() == 2 else -1
    total_actual_tokens = int(total_actual_tokens)
    tensors = (
        kv_cache_packed,
        block_table,
        per_req_padded_lens,
        per_req_actual_lens,
        out_k_quant,
        out_k_scale,
    )
    if (
        batch_size <= 0
        or batch_size > 64
        or total_local_tokens < 0
        or total_actual_tokens < 0
        or total_actual_tokens > total_local_tokens
        or any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.device != out_k_quant.device for tensor in tensors[:-1])
        or out_k_scale.device != out_k_quant.device
        or kv_cache_packed.dim() != 3
        or kv_cache_packed.dtype != torch.uint8
        or int(kv_cache_packed.shape[-1]) != INDEXER_ENTRY_BYTES
        or int(kv_cache_packed.stride(2)) != 1
        or int(kv_cache_packed.stride(1)) != INDEXER_ENTRY_BYTES
        or block_table.dim() != 2
        or block_table.dtype != torch.int32
        or int(block_table.shape[0]) < batch_size
        or int(block_table.stride(1)) != 1
        or per_req_padded_lens.dim() != 1
        # IndexerCPChunkPlan materializes both length vectors as int64. Keep
        # this fused ABI fixed so startup warmup covers its pointer signature.
        or per_req_padded_lens.dtype != torch.int64
        or not per_req_padded_lens.is_contiguous()
        or per_req_actual_lens.dim() != 1
        or int(per_req_actual_lens.numel()) != batch_size
        or per_req_actual_lens.dtype != torch.int64
        or not per_req_actual_lens.is_contiguous()
        or out_k_quant.dim() != 2
        or out_k_quant.dtype != torch.float8_e4m3fn
        or int(out_k_quant.shape[1]) != INDEXER_HEAD_DIM
        or not out_k_quant.is_contiguous()
        or out_k_scale.dim() != 2
        or out_k_scale.dtype != torch.uint8
        or tuple(out_k_scale.shape) != (total_local_tokens, 4)
        or not out_k_scale.is_contiguous()
    ):
        return False
    if total_local_tokens == 0:
        return True
    if (
        int(kv_cache_packed.shape[0]) <= 0
        or int(kv_cache_packed.shape[1]) <= 0
        or (total_actual_tokens > 0 and int(block_table.shape[1]) <= 0)
    ):
        return False

    batch_block = triton.next_power_of_2(batch_size)
    _cp_gather_indexer_k_to_padded_kernel[(total_local_tokens,)](
        kv_cache_packed,
        block_table,
        per_req_padded_lens,
        per_req_actual_lens,
        out_k_quant,
        out_k_scale,
        batch_size,
        total_local_tokens,
        int(block_table.stride(0)),
        int(block_table.shape[1]),
        int(kv_cache_packed.shape[1]),
        int(kv_cache_packed.stride(0)),
        int(kv_cache_packed.shape[0]),
        D=INDEXER_HEAD_DIM,
        SCALE_BYTES=4,
        BATCH_BLOCK=batch_block,
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
        num_warps=4,
    )
    return True


def gather_indexer_k_for_prefill(
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 132] uint8
    slot_mapping: torch.Tensor,  # [N] int64; -1 = pad
    *,
    head_dim: int = INDEXER_HEAD_DIM,
):
    """Single-pass gather: reads the per-block grouped FP8 cache via
    ``slot_mapping``, writes contiguous ``(k_quant [N, 128] fp8e4m3fn,
    k_scale [N] fp32)``. Padded slots (``slot < 0``) write zero K bytes
    and zero scale."""
    assert head_dim == INDEXER_HEAD_DIM, f"head_dim={head_dim}"
    assert (
        kv_cache_packed.dim() == 3
        and kv_cache_packed.shape[-1] == INDEXER_ENTRY_BYTES
        and kv_cache_packed.dtype == torch.uint8
    ), (
        f"kv_cache_packed expected [num_blocks, block_size, 132] uint8, "
        f"got {tuple(kv_cache_packed.shape)}/{kv_cache_packed.dtype}"
    )
    assert (
        slot_mapping.dim() == 1
    ), f"slot_mapping must be 1-D, got {slot_mapping.shape}"
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()

    N = slot_mapping.shape[0]
    device = kv_cache_packed.device
    k_quant = torch.empty(N, head_dim, dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.empty(N, dtype=torch.float32, device=device)
    if N == 0:
        return k_quant, k_scale

    cache_block_size = kv_cache_packed.shape[1]
    cache_stride_b = cache_block_size * INDEXER_ENTRY_BYTES
    validate_slot_mapping(
        "indexer.cp_gather_k.slot_mapping",
        slot_mapping,
        block_size=int(cache_block_size),
        num_blocks=int(kv_cache_packed.shape[0]),
        negative_mode="skip_any",
    )

    _cp_gather_indexer_k_kernel[(N,)](
        kv_cache_packed,
        slot_mapping,
        k_quant,
        k_scale,
        N=N,
        D=head_dim,
        cache_block_size=cache_block_size,
        cache_stride_b=cache_stride_b,
        num_cache_blocks=int(kv_cache_packed.shape[0]),
        TRAP_INVALID_KV_ACCESS=trap_invalid_kv_access_enabled(),
        num_warps=4,
    )
    return k_quant, k_scale
