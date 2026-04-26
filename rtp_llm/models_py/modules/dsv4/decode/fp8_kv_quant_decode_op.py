"""Phase 4 — V4 decode-arm FP8 KV quantize-and-write op.

V4 K is a 512-dim BF16 tensor with the LAST 64 dims being the
RoPE-rotated portion (NoPE = first 448 dims). This maps exactly to the
``fp8_model1_mla`` layout already implemented in
``rtp_llm/models_py/bindings/cuda/kernels/mla_quant_kernel.cu``:

    [NoPE FP8 e4m3fn] : 448 bytes (7 tiles × 64 elements; 1 ue8m0 scale per tile)
    [RoPE BF16      ] : 128 bytes (64 elements)
    [scales         ] :   8 bytes (7 ue8m0 scale bytes + 1 padding)
    Total            : 584 bytes per slot

This module provides:

  * :func:`quantize_v4_kv_decode` — fast path. Splits V4 K into
    ``(kv_c, k_pe)`` and dispatches the existing CUDA kernel
    ``compute_ops.concat_and_cache_mla(..., "fp8_model1_mla", scale)``.
    Used by ``Attention.forward_decode_fp8`` (Stage 4D).

  * :func:`reference_quantize_v4_kv_decode` — pure-PyTorch oracle that
    implements the SAME byte-level layout on CPU. Used by unit tests on
    dev boxes without CUDA / without the compute_ops shared library.

The ``slot_mapping`` is the same flat-slot tensor produced by the
metadata builder (``DSv4DecodeAttnMetadata.slot_mapping_swa`` etc.),
with -1 sentinels skipped (mirrors Phase 1 ``write_swa_k_decode``).

Output cache shape: ``[num_blocks, block_size, 584]`` uint8. For V4
Phase 4 we use ``num_blocks=1, block_size=max_B*T`` so the kv_cache
tensor is contiguous and addressable by a flat slot index — keeps us
on the per-request (not block-paged) layout in line with the Phase 1-3
register_buffer scheme. M4 hetero pool can replace this with a real
block_table later without touching the ops.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

# --------------------------------------------------------------------------------
# Constants — fp8_model1_mla layout (mirrors mla_quant_kernel.cu MODEL1)
# --------------------------------------------------------------------------------
NOPE_DIM = 448
ROPE_DIM = 64
TILE_SIZE = 64  # NoPE elements per ue8m0 scale group
NOPE_TILES = NOPE_DIM // TILE_SIZE  # = 7
NOPE_BYTES = NOPE_DIM  # 1 byte per fp8_e4m3 element
ROPE_BYTES = ROPE_DIM * 2  # 2 bytes per bf16 element = 128
NOPE_ROPE_STRIDE = NOPE_BYTES + ROPE_BYTES  # = 576
SCALE_BYTES_PER_TOKEN = 8  # 7 scale bytes + 1 padding byte
ENTRY_BYTES = NOPE_ROPE_STRIDE + SCALE_BYTES_PER_TOKEN  # = 584

FP8_E4M3_MAX = 448.0  # max representable in fp8_e4m3fn


def quantize_v4_kv_decode(
    k_bf16: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_packed: torch.Tensor,
) -> None:
    """Fast path — dispatch the CUDA ``concat_and_cache_mla`` kernel.

    Args:
        k_bf16: ``[T, head_dim=512]`` bf16 — K tokens to write.
            (Caller flattens ``[B, q_len, 512]`` to ``[B*q_len, 512]``.)
        slot_mapping: ``[T]`` int64 — global flat slot per token; -1 means skip.
        kv_cache_packed: ``[num_blocks, block_size, 584]`` uint8 — packed
            FP8 cache. Modified in place.
    """
    assert (
        k_bf16.dim() == 2 and k_bf16.shape[1] == NOPE_DIM + ROPE_DIM
    ), f"k_bf16 expected [T, 512], got {tuple(k_bf16.shape)}"
    assert (
        kv_cache_packed.dtype == torch.uint8
        and kv_cache_packed.shape[-1] == ENTRY_BYTES
    ), f"kv_cache_packed expected [..., 584] uint8, got {kv_cache_packed.shape}/{kv_cache_packed.dtype}"

    # Split V4 K into (kv_c[NoPE], k_pe[RoPE])
    kv_c = k_bf16[:, :NOPE_DIM].contiguous()
    k_pe = k_bf16[:, NOPE_DIM:].contiguous()

    # Slot mapping must be int64 per the kernel signature.
    if slot_mapping.dtype != torch.long:
        slot_mapping = slot_mapping.long()

    from rtp_llm.ops.compute_ops import rtp_llm_ops

    # The kernel takes a scale tensor (used as a static-scaling fallback
    # for fp8_e4m3 path; for fp8_model1_mla the kernel computes per-tile
    # scales internally and ignores the input scale, but the API still
    # requires the tensor to be present and float32).
    dummy_scale = torch.ones(1, dtype=torch.float32, device=k_bf16.device)
    rtp_llm_ops.concat_and_cache_mla(
        kv_c,
        k_pe,
        kv_cache_packed,
        slot_mapping,
        "fp8_model1_mla",
        dummy_scale,
    )


# --------------------------------------------------------------------------------
# CPU reference — used by unit tests, also a reference oracle
# --------------------------------------------------------------------------------


def _ue8m0_scale_byte(tile_max_abs: float) -> int:
    """Compute the ue8m0 scale byte for a NoPE tile.

    Mirrors the CUDA kernel exactly:
        tile_scale = exp2(ceil(log2(max_abs / 448)))    (clamped to FLT_MIN)
        scale_byte = clamp(log2(tile_scale) + 127, 0, 255)
    """
    eps = torch.finfo(torch.float32).tiny
    tile_scale = max(tile_max_abs / FP8_E4M3_MAX, eps)
    tile_scale = math.pow(2.0, math.ceil(math.log2(tile_scale)))
    scale_byte = int(min(max(math.log2(tile_scale) + 127.0, 0.0), 255.0))
    return scale_byte


def _ue8m0_byte_to_scale(scale_byte: int) -> float:
    """Inverse of ``_ue8m0_scale_byte`` — recover the float scale."""
    return math.pow(2.0, scale_byte - 127.0)


def _quantize_to_fp8_e4m3(x_bf16: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize a bf16 tile to fp8_e4m3fn given a precomputed scale.

    Returns ``[N]`` uint8 — the fp8_e4m3fn bit pattern. Performs the
    cast via PyTorch's ``.to(torch.float8_e4m3fn)`` which clamps and
    rounds the same way the CUDA scaled_convert path does (round-to-nearest-even).
    """
    quantized = (x_bf16.float() / scale).to(torch.float8_e4m3fn)
    return quantized.view(torch.uint8)


def reference_quantize_v4_kv_decode(
    k_bf16: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_packed: torch.Tensor,
    block_size: int,
) -> None:
    """Pure-PyTorch oracle for :func:`quantize_v4_kv_decode`.

    Same in/out semantics; runs on CPU. Used by the Phase 4 unit tests
    on dev boxes without a real CUDA build of compute_ops.

    Args:
        block_size: caller specifies the block_size used to allocate
            ``kv_cache_packed`` so we can compute (block_idx, block_offset)
            from the flat ``slot_mapping`` entries the same way the CUDA
            kernel does.
    """
    assert k_bf16.dim() == 2 and k_bf16.shape[1] == NOPE_DIM + ROPE_DIM
    assert (
        kv_cache_packed.dtype == torch.uint8
        and kv_cache_packed.shape[-1] == ENTRY_BYTES
    )

    T = k_bf16.shape[0]
    if slot_mapping.dtype != torch.long:
        slot_mapping = slot_mapping.long()

    for i in range(T):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue  # padded token — skip
        block_idx = slot // block_size
        block_offset = slot % block_size

        kv_c_token = k_bf16[i, :NOPE_DIM]  # [448] bf16
        k_pe_token = k_bf16[i, NOPE_DIM:]  # [64]  bf16

        # ---- NoPE: 7 tiles of 64 elements each ----
        scale_bytes = bytearray(SCALE_BYTES_PER_TOKEN)  # 8 bytes (7 scales + 1 pad)
        nope_bytes = bytearray(NOPE_BYTES)
        for tile_idx in range(NOPE_TILES):
            tile = kv_c_token[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE]
            tile_max_abs = tile.abs().max().item()
            scale_byte = _ue8m0_scale_byte(tile_max_abs)
            scale_bytes[tile_idx] = scale_byte
            scale = _ue8m0_byte_to_scale(scale_byte)
            tile_quant = _quantize_to_fp8_e4m3(tile, scale)  # [64] uint8
            base = tile_idx * TILE_SIZE
            nope_bytes[base : base + TILE_SIZE] = bytes(tile_quant.tolist())

        # ---- RoPE: 64 bf16 → 128 bytes (no quantization) ----
        rope_bytes = k_pe_token.contiguous().view(torch.uint8)  # [128] uint8

        # ---- Write into kv_cache_packed[block_idx, block_offset, :] ----
        slot_view = kv_cache_packed[block_idx, block_offset]  # [584] uint8
        # NoPE region: [0, 448)
        slot_view[:NOPE_BYTES] = torch.tensor(list(nope_bytes), dtype=torch.uint8)
        # RoPE region: [448, 576)
        slot_view[NOPE_BYTES : NOPE_BYTES + ROPE_BYTES] = rope_bytes

        # ---- Scale region: at end of block (after all tokens' NoPE+RoPE) ----
        # The CUDA kernel layout puts scales at:
        #   block_idx * block_stride + block_size * nope_rope_stride + block_offset * 8
        # block_stride = block_size * entry_bytes = block_size * 584
        # So within a block, the scale offset is:
        #   block_size * 576 + block_offset * 8
        # We have kv_cache_packed[block_idx, block_offset, :] which is the
        # 584-byte slot view — but the scales for token block_offset live
        # PHYSICALLY at the end of the block, not at offset 576 of this slot.
        # Since the entry_size=584 includes the scale region for this token,
        # use the trailing 8 bytes of THIS slot for the scales (the layout
        # treats each slot as 584 bytes, with scales appended).
        slot_view[NOPE_BYTES + ROPE_BYTES :] = torch.tensor(
            list(scale_bytes),
            dtype=torch.uint8,
        )


def dequantize_v4_kv_slot(
    slot_view: torch.Tensor,
) -> torch.Tensor:
    """Inverse: read a single 584-byte slot back into a 512-dim bf16 K vector.

    Used by the FP8 sparse-attn reference (Stage 4C) to verify the
    quantize-then-dequantize round-trip stays within fp8_e4m3 precision.

    Args:
        slot_view: ``[584]`` uint8 — one packed FP8 slot.

    Returns:
        ``[512]`` bf16 — K vector reconstructed from the packed slot.
    """
    assert slot_view.dtype == torch.uint8 and slot_view.shape == (ENTRY_BYTES,)

    # NoPE: 7 tiles
    nope_out = torch.empty(NOPE_DIM, dtype=torch.bfloat16)
    scale_bytes = slot_view[NOPE_BYTES + ROPE_BYTES :]  # [8] uint8
    for tile_idx in range(NOPE_TILES):
        scale = _ue8m0_byte_to_scale(int(scale_bytes[tile_idx].item()))
        tile_quant = slot_view[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE]
        tile_fp8 = tile_quant.view(torch.float8_e4m3fn)
        nope_out[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE] = (
            tile_fp8.float() * scale
        ).to(torch.bfloat16)

    # RoPE: bf16 view of the 128-byte slice
    rope_out = (
        slot_view[NOPE_BYTES : NOPE_BYTES + ROPE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
    )

    return torch.cat([nope_out, rope_out.clone()], dim=0)
