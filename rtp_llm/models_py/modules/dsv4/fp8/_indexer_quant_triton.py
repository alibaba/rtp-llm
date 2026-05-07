"""DSv4 indexer FP8 KV-cache quantize + dequantize (Triton).

Layout matches vLLM's ``indexer_k_quant_and_cache`` (csrc/cache_kernels.cu:515)
so that DeepGEMM's ``fp8_paged_mqa_logits`` can read the cache directly:

    Per block (= ``block_size`` tokens, 132 bytes/token total):
      bytes [0                         : block_size * 128)        = FP8 K
            (token-major: tok0_k0..k127, tok1_k0..k127, ...)
      bytes [block_size * 128          : block_size * (128 + 4))  = fp32 scales
            (one per token, little-endian)

Pool storage is the same ``[num_blocks, block_size, 132] uint8`` 3-D view
the framework allocates (``_pool_spec[INDEXER_KV] = (uint8, 132)``). The
per-slot ``[K|scale]`` interpretation is **not** the physical layout —
within a block, all K bytes come first, then all scale bytes. That makes
the 4-byte scale region naturally aligned and matches what DeepGEMM
expects.

Algorithm matches vLLM byte-for-byte (with quant_block_size = head_dim,
ue8m0=False — the FP8 path):

    scale[t]   = max(|k[t,:]|) / 448.0          # fp32 (clamped to avoid /0)
    fp8_q[t,d] = round_e4m3(k[t,d] / scale[t])
    K bytes    → block region [0, block_size*128)
    scale bytes→ block region [block_size*128, block_size*132)

Public API:

  * :func:`quantize_indexer_k` — fast path. K[T,128] bf16 + slot_mapping
    [T] int64 → write into kv_cache_packed in-place. Skips slot==-1.
  * :func:`dequantize_indexer_k` — read packed slots back to [T, 128]
    fp32 / bf16. Used in the bf16 fallback indexer path; the FP8
    DeepGEMM path consumes the cache directly without dequant.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

INDEXER_HEAD_DIM = 128
INDEXER_ENTRY_BYTES = 132  # 128 FP8 + 4 fp32 scale (per token, total)
FP8_E4M3_MAX = 448.0


# ---------------------------------------------------------------------------
# Quantize: K[T, 128] bf16 -> kv_cache_packed[num_blocks, block_size, 132]
#                              (per-block layout: K_all_tokens || scale_all_tokens)
# ---------------------------------------------------------------------------
@triton.jit
def _indexer_k_quant_kernel(
    k_ptr,  # [T, D] bf16
    slot_mapping_ptr,  # [T] int64; -1 = skip
    cache_ptr,  # raw byte ptr into [num_blocks, block_size*(D+4)] uint8
    # geometry
    T: tl.constexpr,
    D: tl.constexpr,  # head_dim = 128
    cache_block_size: tl.constexpr,
    cache_stride_b: tl.constexpr,  # bytes per block = block_size * (D+4) = block_size * 132
    fp8_max: tl.constexpr,
):
    """One program per token. Loads the [D] vector, computes per-vector
    absmax + fp32 scale, casts to FP8, and writes into the per-block
    K region + per-token scale slot. Layout is vLLM/DeepGEMM-compatible."""
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        return

    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size

    d_off = tl.arange(0, D)
    k_vals = tl.load(k_ptr + pid * D + d_off).to(tl.float32)  # [D]

    # Per-vector absmax → fp32 scale = absmax / 448 (clamp to avoid /0)
    absmax = tl.max(tl.abs(k_vals), axis=0)
    scale = tl.maximum(absmax / fp8_max, 1e-12)

    # Quantize.  Triton's ``.to(tl.float8e4nv)`` clamps + RNE-rounds the
    # same way PyTorch's ``.to(torch.float8_e4m3fn)`` does.
    q_f32 = k_vals / scale
    q_fp8 = q_f32.to(tl.float8e4nv)

    # Per-block K region: bytes [0, block_size * D); per-token offset = block_off * D
    block_base = cache_ptr + block_idx * cache_stride_b
    k_dst = (block_base + block_off * D + d_off).to(tl.pointer_type(tl.uint8))
    tl.store(k_dst, q_fp8.to(tl.uint8, bitcast=True))

    # Per-block scale region: bytes [block_size * D, block_size * (D+4))
    # Per-token scale offset within scale region = block_off * 4 bytes.
    scale_region_base = block_base + cache_block_size * D
    scale_dst = (scale_region_base + block_off * 4).to(tl.pointer_type(tl.float32))
    tl.store(scale_dst, scale)


def quantize_indexer_k(
    k_bf16: torch.Tensor,  # [T, 128] bf16, contiguous
    slot_mapping: torch.Tensor,  # [T] int64 (or convertible); -1 = skip
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 132] uint8
) -> None:
    """Quantize ``k_bf16`` per-vector and write into ``kv_cache_packed``
    at the positions given by ``slot_mapping``. In-place on the cache;
    no return value. Layout matches vLLM/DeepGEMM (per-block grouped)."""
    assert (
        k_bf16.dim() == 2
        and k_bf16.shape[1] == INDEXER_HEAD_DIM
        and k_bf16.dtype == torch.bfloat16
    ), f"k_bf16 expected [T, 128] bf16, got {tuple(k_bf16.shape)}/{k_bf16.dtype}"
    assert k_bf16.is_contiguous(), "k_bf16 must be contiguous"
    assert (
        kv_cache_packed.dim() == 3
        and kv_cache_packed.shape[-1] == INDEXER_ENTRY_BYTES
        and kv_cache_packed.dtype == torch.uint8
    ), (
        f"kv_cache_packed expected [num_blocks, block_size, 132] uint8, "
        f"got {tuple(kv_cache_packed.shape)}/{kv_cache_packed.dtype}"
    )
    assert slot_mapping.dim() == 1 and slot_mapping.shape[0] == k_bf16.shape[0], (
        f"slot_mapping shape {tuple(slot_mapping.shape)} doesn't match "
        f"k_bf16 T={k_bf16.shape[0]}"
    )
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()

    T = k_bf16.shape[0]
    if T == 0:
        return
    cache_block_size = kv_cache_packed.shape[1]
    cache_stride_b = cache_block_size * INDEXER_ENTRY_BYTES

    _indexer_k_quant_kernel[(T,)](
        k_bf16,
        slot_mapping,
        kv_cache_packed,
        T=T,
        D=INDEXER_HEAD_DIM,
        cache_block_size=cache_block_size,
        cache_stride_b=cache_stride_b,
        fp8_max=FP8_E4M3_MAX,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Dequantize: gather packed slots back to [T, 128] fp32/bf16
# (Mirrors the quant layout above. Used by the bf16-fallback indexer path
# when DeepGEMM's FP8 logits kernel isn't applicable.)
# ---------------------------------------------------------------------------
@triton.jit
def _indexer_k_dequant_kernel(
    cache_ptr,  # raw byte ptr into [num_blocks, block_size*(D+4)]
    slot_mapping_ptr,  # [T] int64; -1 = output zeros
    out_ptr,  # [T, D] OUT_DTYPE
    T: tl.constexpr,
    D: tl.constexpr,
    cache_block_size: tl.constexpr,
    cache_stride_b: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    if pid >= T:
        return

    d_off = tl.arange(0, D)
    out_row_ptr = out_ptr + pid * D + d_off

    slot = tl.load(slot_mapping_ptr + pid).to(tl.int64)
    if slot < 0:
        # Padded — write zeros.
        tl.store(out_row_ptr, tl.zeros([D], dtype=tl.float32).to(OUT_DTYPE))
        return

    block_idx = slot // cache_block_size
    block_off = slot % cache_block_size
    block_base = cache_ptr + block_idx * cache_stride_b

    k_src = (block_base + block_off * D + d_off).to(tl.pointer_type(tl.uint8))
    q_bytes = tl.load(k_src)
    q_fp8 = q_bytes.to(tl.float8e4nv, bitcast=True)
    q_f32 = q_fp8.to(tl.float32)

    scale_src = (block_base + cache_block_size * D + block_off * 4).to(
        tl.pointer_type(tl.float32)
    )
    scale = tl.load(scale_src)

    out = q_f32 * scale
    tl.store(out_row_ptr, out.to(OUT_DTYPE))


def dequantize_indexer_k(
    kv_cache_packed: torch.Tensor,  # [num_blocks, block_size, 132] uint8
    slot_mapping: torch.Tensor,  # [T] int64
    *,
    out_dtype: torch.dtype = torch.float32,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gather packed FP8 slots and dequantize to ``[T, 128] out_dtype``.
    Padded entries (slot==-1) get zeros."""
    assert (
        kv_cache_packed.dim() == 3
        and kv_cache_packed.shape[-1] == INDEXER_ENTRY_BYTES
        and kv_cache_packed.dtype == torch.uint8
    )
    assert slot_mapping.dim() == 1
    if slot_mapping.dtype != torch.int64:
        slot_mapping = slot_mapping.to(torch.int64)
    slot_mapping = slot_mapping.contiguous()

    T = slot_mapping.shape[0]
    if out is None:
        out = torch.empty(
            T, INDEXER_HEAD_DIM, dtype=out_dtype, device=kv_cache_packed.device
        )
    else:
        assert out.shape == (T, INDEXER_HEAD_DIM) and out.dtype == out_dtype
    if T == 0:
        return out

    cache_block_size = kv_cache_packed.shape[1]
    cache_stride_b = cache_block_size * INDEXER_ENTRY_BYTES

    _OUT_DTYPE = {
        torch.float32: tl.float32,
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
    }[out_dtype]
    _indexer_k_dequant_kernel[(T,)](
        kv_cache_packed,
        slot_mapping,
        out,
        T=T,
        D=INDEXER_HEAD_DIM,
        cache_block_size=cache_block_size,
        cache_stride_b=cache_stride_b,
        OUT_DTYPE=_OUT_DTYPE,
        num_warps=4,
    )
    return out
