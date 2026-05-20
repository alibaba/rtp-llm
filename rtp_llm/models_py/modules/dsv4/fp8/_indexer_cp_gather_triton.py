"""Gather contiguous (k_quant, k_scale) for DSv4 indexer prefill from the
per-block grouped FP8 cache.

Mirrors vLLM's ``cp_gather_indexer_k_quant_cache_kernel`` (csrc/cache_kernels.cu),
but driven by an explicit ``slot_mapping`` (one absolute slot per output token,
``-1`` to skip / write zeros) instead of ``cu_seq_lens + block_table`` —
RTP-LLM's pool helper (`_compute_pool_slots`) already resolves the
``(b, t) → absolute_slot`` indirection.

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

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    _trap_invalid_kv_access,
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
        _trap_invalid_kv_access()
    if block_idx >= num_cache_blocks:
        _trap_invalid_kv_access()

    block_base = cache_ptr + block_idx * cache_stride_b

    # K bytes
    k_src = (block_base + block_off * D + d_off).to(tl.pointer_type(tl.uint8))
    tl.store(out_k_ptr, tl.load(k_src))

    # fp32 scale
    scale_src = (block_base + cache_block_size * D + block_off * 4).to(
        tl.pointer_type(tl.float32)
    )
    tl.store(k_scale_ptr + pid, tl.load(scale_src))


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
        num_warps=4,
    )
    return k_quant, k_scale
