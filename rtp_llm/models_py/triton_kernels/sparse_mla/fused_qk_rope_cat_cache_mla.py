"""Fused RoPE + concat + paged KV cache write for MLA decode/prefill.

Replaces the two-kernel chain in
``flashmla_sparse_impl.py``::

    self.rope_impl.forward(q_pe, k_pe, self.rope_params)
    self.kv_cache_write_op.forward(compressed_kv, k_pe, kv_cache, self.rope_params)

with a single Triton kernel.  The two original kernels were:
  1. flashinfer ``apply_rope_pos_ids_cos_sin_cache`` (in-place RoPE on q_pe and k_pe)
  2. rtp_llm CUDA ``concat_and_cache_mla`` (cat compressed_kv || k_pe, write to paged cache)

Supports both KV cache layouts:
  - "auto":       BF16 cache, kv_cache dtype=bf16
  - "fp8_ds_mla": FP8 cache,  kv_cache dtype=uint8 (656B per slot)

``slot_mapping[t] == -1`` is legal padding. Match the unfused path by still
applying RoPE to q/k for that token while skipping only the KV-cache write.

Note: q.shape[-1] = nope_head_dim + rope_head_dim (per-head Q dim),
      which differs from kv_lora_rank (compressed KV dim for the cache).
      The kernel uses Q_ROPE_OFFSET (=nope_head_dim) for Q-side indexing,
      and KV_LORA (=kv_lora_rank) for K-side cache writes.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_BLOCK_H_TILE = 16
_NUM_FP8_K_BLOCKS = 5

_FP8_SCALE_MIN: dict[torch.device, torch.Tensor] = {}


def _get_fp8_scale_min(device: torch.device) -> torch.Tensor:
    t = _FP8_SCALE_MIN.get(device)
    if t is None:
        t = torch.tensor(
            torch.finfo(torch.float32).tiny,
            dtype=torch.float32,
            device=device,
        )
        _FP8_SCALE_MIN[device] = t
    return t


# ---- BF16 KV cache path ("auto") ------------------------------------------
@triton.jit
def _fused_qk_rope_cat_cache_mla_bf16_kernel(
    Q,
    Q_ROPE_OUT,
    K_PE,
    KV_CACHE,
    COMPRESSED_KV,
    KV_NORM_WEIGHT,
    SLOT_MAPPING,
    POSITIONS,
    COS_SIN_CACHE,
    stride_q_t,
    stride_q_h,
    stride_qo_t,
    stride_qo_h,
    stride_kpe_t,
    stride_ck_t,
    stride_kvc_page,
    stride_kvc_slot,
    kv_norm_eps,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    Q_ROPE_OFFSET: tl.constexpr,  # nope_head_dim — offset into q for rope slice
    KV_LORA: tl.constexpr,  # kv_lora_rank — compressed KV size for cache
    ROPE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_H_TILE: tl.constexpr,
    H_TILES: tl.constexpr,
    HAS_Q_ROPE_OUT: tl.constexpr,
    HAS_KV_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h_blk = tl.program_id(1)

    pos = tl.load(POSITIONS + t).to(tl.int64)
    rh = tl.arange(0, HALF_ROPE)
    c = tl.load(COS_SIN_CACHE + pos * ROPE + rh)
    s = tl.load(COS_SIN_CACHE + pos * ROPE + HALF_ROPE + rh)

    if h_blk < H_TILES:
        h_start = h_blk * BLOCK_H_TILE
        h_offs = h_start + tl.arange(0, BLOCK_H_TILE)
        h_mask = h_offs < H
        q_base = t * stride_q_t
        m2d = h_mask[:, None]

        if IS_NEOX:
            addrs_lo = (
                q_base + h_offs[:, None] * stride_q_h + Q_ROPE_OFFSET + rh[None, :]
            )
            addrs_hi = (
                q_base
                + h_offs[:, None] * stride_q_h
                + Q_ROPE_OFFSET
                + HALF_ROPE
                + rh[None, :]
            )
            x1 = tl.load(Q + addrs_lo, mask=m2d, other=0.0).to(tl.float32)
            x2 = tl.load(Q + addrs_hi, mask=m2d, other=0.0).to(tl.float32)
            y1 = tl.extra.libdevice.fma_rn(x1, c[None, :], -x2 * s[None, :])
            y2 = tl.extra.libdevice.fma_rn(x2, c[None, :], x1 * s[None, :])
            if HAS_Q_ROPE_OUT:
                qo_base = t * stride_qo_t + h_offs[:, None] * stride_qo_h
                tl.store(
                    Q_ROPE_OUT + qo_base + rh[None, :],
                    y1.to(tl.bfloat16),
                    mask=m2d,
                )
                tl.store(
                    Q_ROPE_OUT + qo_base + HALF_ROPE + rh[None, :],
                    y2.to(tl.bfloat16),
                    mask=m2d,
                )
            else:
                tl.store(Q + addrs_lo, y1.to(tl.bfloat16), mask=m2d)
                tl.store(Q + addrs_hi, y2.to(tl.bfloat16), mask=m2d)
        else:
            addrs_e = (
                q_base + h_offs[:, None] * stride_q_h + Q_ROPE_OFFSET + 2 * rh[None, :]
            )
            addrs_o = addrs_e + 1
            xe = tl.load(Q + addrs_e, mask=m2d, other=0.0).to(tl.float32)
            xo = tl.load(Q + addrs_o, mask=m2d, other=0.0).to(tl.float32)
            ye = tl.extra.libdevice.fma_rn(xe, c[None, :], -xo * s[None, :])
            yo = tl.extra.libdevice.fma_rn(xo, c[None, :], xe * s[None, :])
            if HAS_Q_ROPE_OUT:
                qo_base = t * stride_qo_t + h_offs[:, None] * stride_qo_h
                tl.store(
                    Q_ROPE_OUT + qo_base + 2 * rh[None, :],
                    ye.to(tl.bfloat16),
                    mask=m2d,
                )
                tl.store(
                    Q_ROPE_OUT + qo_base + 2 * rh[None, :] + 1,
                    yo.to(tl.bfloat16),
                    mask=m2d,
                )
            else:
                tl.store(Q + addrs_e, ye.to(tl.bfloat16), mask=m2d)
                tl.store(Q + addrs_o, yo.to(tl.bfloat16), mask=m2d)
    else:
        slot = tl.load(SLOT_MAPPING + t).to(tl.int64)
        slot_valid = slot >= 0
        safe_slot = tl.where(slot_valid, slot, 0)
        page_idx = safe_slot // BLOCK_SIZE
        slot_offset = safe_slot % BLOCK_SIZE
        kvc_off = page_idx * stride_kvc_page + slot_offset * stride_kvc_slot

        kv_lora_idx = tl.arange(0, KV_LORA)
        ck = tl.load(COMPRESSED_KV + t * stride_ck_t + kv_lora_idx).to(
            tl.float32
        )
        if HAS_KV_NORM:
            sq_sum = tl.sum(ck * ck)
            inv_rms = tl.rsqrt(sq_sum / KV_LORA + kv_norm_eps)
            norm_weight = tl.load(KV_NORM_WEIGHT + kv_lora_idx).to(tl.float32)
            # Preserve the old ``RMSNorm -> BF16 cache writer`` boundary.
            ck = (ck * inv_rms * norm_weight).to(tl.bfloat16)
        tl.store(KV_CACHE + kvc_off + kv_lora_idx, ck, mask=slot_valid)

        kpe_off = t * stride_kpe_t
        if IS_NEOX:
            k1 = tl.load(K_PE + kpe_off + rh).to(tl.float32)
            k2 = tl.load(K_PE + kpe_off + HALF_ROPE + rh).to(tl.float32)
            k_y1 = tl.extra.libdevice.fma_rn(k1, c, -k2 * s)
            k_y2 = tl.extra.libdevice.fma_rn(k2, c, k1 * s)
            k_y1_bf = k_y1.to(tl.bfloat16)
            k_y2_bf = k_y2.to(tl.bfloat16)
            tl.store(K_PE + kpe_off + rh, k_y1_bf)
            tl.store(K_PE + kpe_off + HALF_ROPE + rh, k_y2_bf)
            tl.store(KV_CACHE + kvc_off + KV_LORA + rh, k_y1_bf, mask=slot_valid)
            tl.store(
                KV_CACHE + kvc_off + KV_LORA + HALF_ROPE + rh,
                k_y2_bf,
                mask=slot_valid,
            )
        else:
            ke = tl.load(K_PE + kpe_off + 2 * rh).to(tl.float32)
            ko = tl.load(K_PE + kpe_off + 2 * rh + 1).to(tl.float32)
            k_ye = tl.extra.libdevice.fma_rn(ke, c, -ko * s)
            k_yo = tl.extra.libdevice.fma_rn(ko, c, ke * s)
            k_ye_bf = k_ye.to(tl.bfloat16)
            k_yo_bf = k_yo.to(tl.bfloat16)
            tl.store(K_PE + kpe_off + 2 * rh, k_ye_bf)
            tl.store(K_PE + kpe_off + 2 * rh + 1, k_yo_bf)
            tl.store(KV_CACHE + kvc_off + KV_LORA + 2 * rh, k_ye_bf, mask=slot_valid)
            tl.store(
                KV_CACHE + kvc_off + KV_LORA + 2 * rh + 1,
                k_yo_bf,
                mask=slot_valid,
            )


# ---- FP8 KV cache path ("fp8_ds_mla") -------------------------------------
@triton.jit
def _fused_qk_rope_cat_cache_mla_fp8_kernel(
    Q,
    Q_ROPE_OUT,
    K_PE,
    KV_CACHE_FP8,
    KV_CACHE_FP32,
    KV_CACHE_BF16,
    COMPRESSED_KV,
    KV_NORM_WEIGHT,
    SLOT_MAPPING,
    POSITIONS,
    COS_SIN_CACHE,
    SCALE_MIN_PTR,
    stride_q_t,
    stride_q_h,
    stride_qo_t,
    stride_qo_h,
    stride_kpe_t,
    stride_ck_t,
    stride_kvc_u8_page,
    stride_kvc_u8_slot,
    stride_kvc_fp32_page,
    stride_kvc_fp32_slot,
    stride_kvc_bf16_page,
    stride_kvc_bf16_slot,
    kv_norm_eps,
    BLOCK_SIZE: tl.constexpr,
    H: tl.constexpr,
    Q_ROPE_OFFSET: tl.constexpr,
    KV_LORA: tl.constexpr,
    ROPE: tl.constexpr,
    HALF_ROPE: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_H_TILE: tl.constexpr,
    H_TILES: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    HAS_Q_ROPE_OUT: tl.constexpr,
    HAS_KV_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    h_blk = tl.program_id(1)

    if h_blk < H_TILES:
        pos = tl.load(POSITIONS + t).to(tl.int64)
        rh = tl.arange(0, HALF_ROPE)
        c = tl.load(COS_SIN_CACHE + pos * ROPE + rh)
        s = tl.load(COS_SIN_CACHE + pos * ROPE + HALF_ROPE + rh)

        h_start = h_blk * BLOCK_H_TILE
        h_offs = h_start + tl.arange(0, BLOCK_H_TILE)
        h_mask = h_offs < H
        q_base = t * stride_q_t
        m2d = h_mask[:, None]

        if IS_NEOX:
            addrs_lo = (
                q_base + h_offs[:, None] * stride_q_h + Q_ROPE_OFFSET + rh[None, :]
            )
            addrs_hi = (
                q_base
                + h_offs[:, None] * stride_q_h
                + Q_ROPE_OFFSET
                + HALF_ROPE
                + rh[None, :]
            )
            x1 = tl.load(Q + addrs_lo, mask=m2d, other=0.0).to(tl.float32)
            x2 = tl.load(Q + addrs_hi, mask=m2d, other=0.0).to(tl.float32)
            y1 = tl.extra.libdevice.fma_rn(x1, c[None, :], -x2 * s[None, :])
            y2 = tl.extra.libdevice.fma_rn(x2, c[None, :], x1 * s[None, :])
            if HAS_Q_ROPE_OUT:
                qo_base = t * stride_qo_t + h_offs[:, None] * stride_qo_h
                tl.store(
                    Q_ROPE_OUT + qo_base + rh[None, :],
                    y1.to(tl.bfloat16),
                    mask=m2d,
                )
                tl.store(
                    Q_ROPE_OUT + qo_base + HALF_ROPE + rh[None, :],
                    y2.to(tl.bfloat16),
                    mask=m2d,
                )
            else:
                tl.store(Q + addrs_lo, y1.to(tl.bfloat16), mask=m2d)
                tl.store(Q + addrs_hi, y2.to(tl.bfloat16), mask=m2d)
        else:
            addrs_e = (
                q_base + h_offs[:, None] * stride_q_h + Q_ROPE_OFFSET + 2 * rh[None, :]
            )
            addrs_o = addrs_e + 1
            xe = tl.load(Q + addrs_e, mask=m2d, other=0.0).to(tl.float32)
            xo = tl.load(Q + addrs_o, mask=m2d, other=0.0).to(tl.float32)
            ye = tl.extra.libdevice.fma_rn(xe, c[None, :], -xo * s[None, :])
            yo = tl.extra.libdevice.fma_rn(xo, c[None, :], xe * s[None, :])
            if HAS_Q_ROPE_OUT:
                qo_base = t * stride_qo_t + h_offs[:, None] * stride_qo_h
                tl.store(
                    Q_ROPE_OUT + qo_base + 2 * rh[None, :],
                    ye.to(tl.bfloat16),
                    mask=m2d,
                )
                tl.store(
                    Q_ROPE_OUT + qo_base + 2 * rh[None, :] + 1,
                    yo.to(tl.bfloat16),
                    mask=m2d,
                )
            else:
                tl.store(Q + addrs_e, ye.to(tl.bfloat16), mask=m2d)
                tl.store(Q + addrs_o, yo.to(tl.bfloat16), mask=m2d)

    elif h_blk < H_TILES + NUM_K_BLOCKS - 1:
        slot = tl.load(SLOT_MAPPING + t).to(tl.int64)
        if slot < 0:
            return

        page_idx = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE
        scale_min = tl.load(SCALE_MIN_PTR)
        ck_base = t * stride_ck_t
        kvc_base = page_idx * stride_kvc_u8_page + slot_offset * stride_kvc_u8_slot
        scale_base = (
            page_idx * stride_kvc_fp32_page
            + slot_offset * stride_kvc_fp32_slot
            + (KV_LORA // 4)
        )

        if HAS_KV_NORM:
            # One program owns all four group-128 KV tiles. This lets the
            # row-wide RMS reduction happen once, keeps the normalized BF16
            # values in registers, and writes only the final FP8 cache bytes.
            elem_off = tl.arange(0, KV_LORA)
            ck = tl.load(COMPRESSED_KV + ck_base + elem_off).to(tl.float32)
            sq_sum = tl.sum(ck * ck)
            inv_rms = tl.rsqrt(sq_sum / KV_LORA + kv_norm_eps)
            norm_weight = tl.load(KV_NORM_WEIGHT + elem_off).to(tl.float32)
            normed = (ck * inv_rms * norm_weight).to(tl.bfloat16).to(tl.float32)

            num_groups: tl.constexpr = KV_LORA // QUANT_BLOCK
            normed_2d = tl.reshape(normed, (num_groups, QUANT_BLOCK))
            max_abs = tl.max(tl.abs(normed_2d), axis=1)
            tile_scale = tl.maximum(max_abs / 448.0, scale_min)
            scale_full = tl.broadcast_to(
                tl.reshape(tile_scale, (num_groups, 1)),
                (num_groups, QUANT_BLOCK),
            )
            fp8_vals = (normed_2d / scale_full).to(tl.float8e4nv)
            tl.store(
                KV_CACHE_FP8 + kvc_base + elem_off,
                tl.reshape(fp8_vals, (KV_LORA,)),
            )
            group_off = tl.arange(0, num_groups)
            tl.store(KV_CACHE_FP32 + scale_base + group_off, tile_scale)
        else:
            tile_id = h_blk - H_TILES
            tile_off = tile_id * QUANT_BLOCK
            elem_off = tl.arange(0, QUANT_BLOCK)
            ck_tile = tl.load(
                COMPRESSED_KV + ck_base + tile_off + elem_off
            ).to(tl.float32)
            max_abs = tl.max(tl.abs(ck_tile))
            tile_scale = tl.maximum(max_abs / 448.0, scale_min)
            fp8_vals = (ck_tile / tile_scale).to(tl.float8e4nv)
            tl.store(KV_CACHE_FP8 + kvc_base + tile_off + elem_off, fp8_vals)
            tl.store(KV_CACHE_FP32 + scale_base + tile_id, tile_scale)

    else:
        slot = tl.load(SLOT_MAPPING + t).to(tl.int64)
        slot_valid = slot >= 0
        safe_slot = tl.where(slot_valid, slot, 0)
        page_idx = safe_slot // BLOCK_SIZE
        slot_offset = safe_slot % BLOCK_SIZE
        pos = tl.load(POSITIONS + t).to(tl.int64)
        rh = tl.arange(0, HALF_ROPE)
        c = tl.load(COS_SIN_CACHE + pos * ROPE + rh)
        s = tl.load(COS_SIN_CACHE + pos * ROPE + HALF_ROPE + rh)

        rope_bf16_base = (
            page_idx * stride_kvc_bf16_page
            + slot_offset * stride_kvc_bf16_slot
            + (KV_LORA // 2)
            + (4 * 4 // 2)
        )
        kpe_off = t * stride_kpe_t
        if IS_NEOX:
            k1 = tl.load(K_PE + kpe_off + rh).to(tl.float32)
            k2 = tl.load(K_PE + kpe_off + HALF_ROPE + rh).to(tl.float32)
            k_y1 = tl.extra.libdevice.fma_rn(k1, c, -k2 * s)
            k_y2 = tl.extra.libdevice.fma_rn(k2, c, k1 * s)
            k_y1_bf = k_y1.to(tl.bfloat16)
            k_y2_bf = k_y2.to(tl.bfloat16)
            tl.store(K_PE + kpe_off + rh, k_y1_bf)
            tl.store(K_PE + kpe_off + HALF_ROPE + rh, k_y2_bf)
            tl.store(KV_CACHE_BF16 + rope_bf16_base + rh, k_y1_bf, mask=slot_valid)
            tl.store(
                KV_CACHE_BF16 + rope_bf16_base + HALF_ROPE + rh,
                k_y2_bf,
                mask=slot_valid,
            )
        else:
            ke = tl.load(K_PE + kpe_off + 2 * rh).to(tl.float32)
            ko = tl.load(K_PE + kpe_off + 2 * rh + 1).to(tl.float32)
            k_ye = tl.extra.libdevice.fma_rn(ke, c, -ko * s)
            k_yo = tl.extra.libdevice.fma_rn(ko, c, ke * s)
            k_ye_bf = k_ye.to(tl.bfloat16)
            k_yo_bf = k_yo.to(tl.bfloat16)
            tl.store(K_PE + kpe_off + 2 * rh, k_ye_bf)
            tl.store(K_PE + kpe_off + 2 * rh + 1, k_yo_bf)
            tl.store(KV_CACHE_BF16 + rope_bf16_base + 2 * rh, k_ye_bf, mask=slot_valid)
            tl.store(
                KV_CACHE_BF16 + rope_bf16_base + 2 * rh + 1,
                k_yo_bf,
                mask=slot_valid,
            )


def fused_qk_rope_cat_cache_mla(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_lora_rank: int,
    rope_head_dim: int,
    is_neox_style: bool,
    kv_cache_type: str = "auto",
    q_rope_output: Optional[torch.Tensor] = None,
    kv_norm_weight: Optional[torch.Tensor] = None,
    kv_norm_eps: float = 0.0,
) -> None:
    """Fused RoPE + paged KV cache write for sparse MLA.

    Args:
        q:             [T, H, nope_head_dim + rope_head_dim]  bf16
        compressed_kv: [T, kv_lora_rank]                      bf16
        k_pe:          [T, rope_head_dim]                     bf16
        kv_cache:      [num_blocks, block_size, D]
                       - "auto":       dtype=bf16, D = kv_lora_rank + rope_head_dim
                       - "fp8_ds_mla": dtype=uint8, D = 656
        slot_mapping:  [T]  int64, -1 skips KV-cache write for that token
        positions:     [T]  int32
        cos_sin_cache: [max_pos, rope_head_dim]  fp32
        kv_lora_rank:  compressed KV dimension (e.g. 512)
        rope_head_dim: RoPE dimension (e.g. 64)
        is_neox_style: True for NEOX rotation
        kv_cache_type: "auto" or "fp8_ds_mla"
        q_rope_output: optional ``[T, H, rope_head_dim]`` BF16 destination.
                       When present, rotated Q-RoPE is written here instead
                       of back to ``q`` so an absorbed-query buffer can avoid
                       a following strided slice-copy launch.
        kv_norm_weight: optional BF16 ``[kv_lora_rank]`` RMSNorm weight. When
                        present, ``compressed_kv`` is the raw strided KV-A
                        projection and RMSNorm is fused into the cache writer.
        kv_norm_eps: RMSNorm epsilon used with ``kv_norm_weight``.
    """
    assert q.dim() == 3 and q.dtype == torch.bfloat16
    assert compressed_kv.dim() == 2 and compressed_kv.dtype == torch.bfloat16
    assert k_pe.dim() == 2 and k_pe.dtype == torch.bfloat16
    assert kv_cache.dim() == 3
    assert cos_sin_cache.dtype == torch.float32

    T, H, qk_head_dim = q.shape
    nope_head_dim = qk_head_dim - rope_head_dim
    block_size = kv_cache.size(1)
    assert nope_head_dim > 0
    assert k_pe.size(0) == T and k_pe.size(1) == rope_head_dim
    assert compressed_kv.size(0) == T and compressed_kv.size(1) == kv_lora_rank
    assert slot_mapping.size(0) == T and slot_mapping.dtype == torch.int64
    assert positions.size(0) == T
    if q_rope_output is not None:
        assert q_rope_output.dim() == 3
        assert q_rope_output.dtype == torch.bfloat16
        assert q_rope_output.device == q.device
        assert q_rope_output.shape == (T, H, rope_head_dim)
        assert q_rope_output.stride(-1) == 1
    if kv_norm_weight is not None:
        assert kv_lora_rank == 512
        assert kv_norm_weight.shape == (kv_lora_rank,)
        assert kv_norm_weight.dtype == torch.bfloat16
        assert kv_norm_weight.device == compressed_kv.device
        assert kv_norm_weight.is_contiguous()

    if T == 0:
        return

    half_rope = rope_head_dim // 2
    h_tiles = (H + _BLOCK_H_TILE - 1) // _BLOCK_H_TILE

    if kv_cache_type == "auto":
        assert kv_cache.dtype == torch.bfloat16
        assert kv_cache.size(2) == kv_lora_rank + rope_head_dim
        grid = (T, h_tiles + 1)
        _fused_qk_rope_cat_cache_mla_bf16_kernel[grid](
            q,
            q if q_rope_output is None else q_rope_output,
            k_pe,
            kv_cache,
            compressed_kv,
            compressed_kv if kv_norm_weight is None else kv_norm_weight,
            slot_mapping,
            positions,
            cos_sin_cache,
            q.stride(0),
            q.stride(1),
            q.stride(0) if q_rope_output is None else q_rope_output.stride(0),
            q.stride(1) if q_rope_output is None else q_rope_output.stride(1),
            k_pe.stride(0),
            compressed_kv.stride(0),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_norm_eps,
            BLOCK_SIZE=block_size,
            H=H,
            Q_ROPE_OFFSET=nope_head_dim,
            KV_LORA=kv_lora_rank,
            ROPE=rope_head_dim,
            HALF_ROPE=half_rope,
            IS_NEOX=is_neox_style,
            BLOCK_H_TILE=_BLOCK_H_TILE,
            H_TILES=h_tiles,
            HAS_Q_ROPE_OUT=q_rope_output is not None,
            HAS_KV_NORM=kv_norm_weight is not None,
            # Match fused_strided_rmsnorm(H=512)'s reduction order when KV
            # normalization is folded into this launch.
            num_warps=2 if kv_norm_weight is not None else 4,
            num_stages=2,
        )
    elif kv_cache_type == "fp8_ds_mla":
        assert kv_lora_rank == 512
        assert rope_head_dim == 64
        slot_bytes = kv_cache.size(2) * kv_cache.element_size()
        assert (
            slot_bytes == 656
        ), f"fp8_ds_mla requires 656 bytes per slot, got {slot_bytes}"
        kvc_fp8 = kv_cache.view(torch.float8_e4m3fn)
        kvc_fp32 = kv_cache.view(torch.float32)
        kvc_bf16 = kv_cache.view(torch.bfloat16)
        num_fp8_k_blocks = 2 if kv_norm_weight is not None else _NUM_FP8_K_BLOCKS
        fp8_grid = (T, h_tiles + num_fp8_k_blocks)
        _fused_qk_rope_cat_cache_mla_fp8_kernel[fp8_grid](
            q,
            q if q_rope_output is None else q_rope_output,
            k_pe,
            kvc_fp8,
            kvc_fp32,
            kvc_bf16,
            compressed_kv,
            compressed_kv if kv_norm_weight is None else kv_norm_weight,
            slot_mapping,
            positions,
            cos_sin_cache,
            _get_fp8_scale_min(kv_cache.device),
            q.stride(0),
            q.stride(1),
            q.stride(0) if q_rope_output is None else q_rope_output.stride(0),
            q.stride(1) if q_rope_output is None else q_rope_output.stride(1),
            k_pe.stride(0),
            compressed_kv.stride(0),
            kvc_fp8.stride(0),
            kvc_fp8.stride(1),
            kvc_fp32.stride(0),
            kvc_fp32.stride(1),
            kvc_bf16.stride(0),
            kvc_bf16.stride(1),
            kv_norm_eps,
            BLOCK_SIZE=block_size,
            H=H,
            Q_ROPE_OFFSET=nope_head_dim,
            KV_LORA=kv_lora_rank,
            ROPE=rope_head_dim,
            HALF_ROPE=half_rope,
            QUANT_BLOCK=128,
            IS_NEOX=is_neox_style,
            BLOCK_H_TILE=_BLOCK_H_TILE,
            H_TILES=h_tiles,
            NUM_K_BLOCKS=num_fp8_k_blocks,
            HAS_Q_ROPE_OUT=q_rope_output is not None,
            HAS_KV_NORM=kv_norm_weight is not None,
            num_warps=2,
            num_stages=2,
        )
    else:
        raise ValueError(
            f"Unsupported kv_cache_type {kv_cache_type!r}; expected 'auto' or 'fp8_ds_mla'"
        )
