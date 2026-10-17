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

  * :func:`quantize_v4_kv_decode` — generic CUDA wrapper. Splits V4 K into
    ``(kv_c, k_pe)`` and dispatches the existing CUDA kernel
    ``compute_ops.concat_and_cache_mla(..., "fp8_model1_mla", scale)``.
    The current paged AttentionFP8 path uses its dedicated SWA writer instead;
    this wrapper remains a platform-neutral MODEL1 contract utility.

  * :func:`reference_quantize_v4_kv_decode` — pure-PyTorch reference that
    implements the same byte-level layout on CPU. Independent golden fixtures,
    rather than this implementation itself, are the byte-contract oracle.

The ``slot_mapping`` is the same flat-slot tensor produced by the
metadata builder (``DSv4DecodeAttnMetadataFP8.slot_mapping_swa`` etc.),
with -1 sentinels skipped (mirrors Phase 1 ``write_swa_k_decode``).

Output cache shape: ``[num_blocks, block_size, 584]`` uint8. Slot ids are
flat in token space: ``block_idx = slot // block_size`` and
``block_offset = slot % block_size``. The tensor shape is a kernel ABI; do
not parse a slot via ``kv_cache_packed[b, t, :]`` when ``block_size > 1``.
The MODEL1 byte layout inside each block is split as ``block_size * 576``
bytes of NoPE+RoPE followed by ``block_size * 8`` scale bytes, and the
batch/block stride may be padded to a multiple of 576 for FlashMLA's TMA
path.
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


def model1_block_payload_bytes(block_size: int) -> int:
    """Logical MODEL1 payload bytes before any physical block-tail padding."""
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"MODEL1 block_size must be positive, got {block_size}")
    return block_size * ENTRY_BYTES


def model1_slot_physical_offsets(
    block_size: int, block_offset: int
) -> tuple[int, int]:
    """Return the data and scale offsets for one slot within a MODEL1 block."""
    block_size = int(block_size)
    block_offset = int(block_offset)
    if block_size <= 0:
        raise ValueError(f"MODEL1 block_size must be positive, got {block_size}")
    if block_offset < 0 or block_offset >= block_size:
        raise IndexError(
            "MODEL1 block_offset out of range: "
            f"offset={block_offset}, block_size={block_size}"
        )
    return (
        block_offset * NOPE_ROPE_STRIDE,
        block_size * NOPE_ROPE_STRIDE
        + block_offset * SCALE_BYTES_PER_TOKEN,
    )


def _validate_model1_cache_tensor(
    kv_cache_packed: torch.Tensor, block_size: Optional[int] = None
) -> tuple[int, int]:
    if kv_cache_packed.dtype != torch.uint8 or kv_cache_packed.dim() != 3:
        raise ValueError(
            "MODEL1 cache must be a 3D uint8 tensor, "
            f"got shape={tuple(kv_cache_packed.shape)}, dtype={kv_cache_packed.dtype}"
        )
    if kv_cache_packed.shape[0] <= 0 or kv_cache_packed.shape[1] <= 0:
        raise ValueError(
            f"MODEL1 cache dimensions must be non-empty, got {tuple(kv_cache_packed.shape)}"
        )
    if kv_cache_packed.shape[2] != ENTRY_BYTES:
        raise ValueError(
            f"MODEL1 logical entry must be {ENTRY_BYTES} bytes, "
            f"got {kv_cache_packed.shape[2]}"
        )
    actual_block_size = int(kv_cache_packed.shape[1])
    if block_size is not None and int(block_size) != actual_block_size:
        raise ValueError(
            "MODEL1 block_size must match cache shape[1], "
            f"got block_size={block_size}, shape={tuple(kv_cache_packed.shape)}"
        )
    if (
        kv_cache_packed.stride(1) != ENTRY_BYTES
        or kv_cache_packed.stride(2) != 1
    ):
        raise ValueError(
            "MODEL1 logical shape must use strides (physical_block, 584, 1), "
            f"got strides={tuple(kv_cache_packed.stride())}"
        )
    block_stride = int(kv_cache_packed.stride(0))
    payload_bytes = model1_block_payload_bytes(actual_block_size)
    if block_stride < payload_bytes:
        raise ValueError(
            "MODEL1 physical block stride is smaller than its logical payload: "
            f"stride={block_stride}, payload={payload_bytes}"
        )
    storage_elems = (
        kv_cache_packed.untyped_storage().nbytes()
        // kv_cache_packed.element_size()
    )
    required_storage_end = (
        int(kv_cache_packed.storage_offset())
        + (int(kv_cache_packed.shape[0]) - 1) * block_stride
        + block_stride
    )
    if required_storage_end > storage_elems:
        raise ValueError(
            "MODEL1 storage does not cover the final physical block stride: "
            f"required_end={required_storage_end}, storage_elems={storage_elems}, "
            f"storage_offset={kv_cache_packed.storage_offset()}, "
            f"num_blocks={kv_cache_packed.shape[0]}, stride={block_stride}"
        )
    return actual_block_size, block_stride


def _model1_block_bytes(kv_cache_packed: torch.Tensor, block_idx: int) -> torch.Tensor:
    """Return a 1D byte view for one MODEL1 block.

    ``concat_and_cache_ds_model1_kernel`` uses ``kv_cache.stride(0)`` as the
    physical block stride and computes all intra-block offsets manually. The
    logical ``[block_size, 584]`` dimensions are not a true per-slot layout.
    """
    _, block_stride = _validate_model1_cache_tensor(kv_cache_packed)
    block_idx = int(block_idx)
    if block_idx < 0 or block_idx >= kv_cache_packed.shape[0]:
        raise IndexError(
            "MODEL1 block_idx out of range: "
            f"index={block_idx}, num_blocks={kv_cache_packed.shape[0]}"
        )
    blocks = kv_cache_packed.as_strided(
        (kv_cache_packed.shape[0], block_stride),
        (block_stride, 1),
    )
    return blocks[block_idx]


def _model1_slot_views(
    kv_cache_packed: torch.Tensor,
    block_idx: int,
    block_offset: int,
    block_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(nope_rope[576], scales[8])`` views for one MODEL1 slot."""
    actual_block_size, _ = _validate_model1_cache_tensor(
        kv_cache_packed, block_size
    )
    block = _model1_block_bytes(kv_cache_packed, block_idx)
    nope_rope_start, scale_start = model1_slot_physical_offsets(
        actual_block_size, block_offset
    )
    return (
        block[nope_rope_start : nope_rope_start + NOPE_ROPE_STRIDE],
        block[scale_start : scale_start + SCALE_BYTES_PER_TOKEN],
    )


def read_model1_kv_slot_bytes(
    kv_cache_packed: torch.Tensor,
    block_idx: int,
    block_offset: int,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Materialize one logical MODEL1 slot as contiguous ``[584]`` bytes.

    This is the safe Python-side reader for tests/reference code. It stitches
    the split physical layout back into the logical ``[NoPE+RoPE][scales]``
    order expected by the dequantizer.
    """
    nope_rope, scale_bytes = _model1_slot_views(
        kv_cache_packed, block_idx, block_offset, block_size
    )
    slot = torch.empty(ENTRY_BYTES, dtype=torch.uint8, device=kv_cache_packed.device)
    slot[:NOPE_ROPE_STRIDE] = nope_rope
    slot[NOPE_ROPE_STRIDE:] = scale_bytes
    return slot


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
    if k_bf16.dim() != 2 or k_bf16.shape[1] != NOPE_DIM + ROPE_DIM:
        raise ValueError(f"k_bf16 expected [T, 512], got {tuple(k_bf16.shape)}")
    _validate_model1_cache_tensor(kv_cache_packed)

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
    if k_bf16.dim() != 2 or k_bf16.shape[1] != NOPE_DIM + ROPE_DIM:
        raise ValueError(f"k_bf16 expected [T, 512], got {tuple(k_bf16.shape)}")
    _validate_model1_cache_tensor(kv_cache_packed, block_size)

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
        scale_bytes[-1] = 0

        # ---- RoPE: 64 bf16 → 128 bytes (no quantization) ----
        rope_bytes = k_pe_token.contiguous().view(torch.uint8)  # [128] uint8

        nope_rope_view, scale_view = _model1_slot_views(
            kv_cache_packed, block_idx, block_offset, block_size
        )
        nope_rope_view[:NOPE_BYTES] = torch.tensor(
            list(nope_bytes),
            dtype=torch.uint8,
            device=kv_cache_packed.device,
        )
        nope_rope_view[NOPE_BYTES : NOPE_BYTES + ROPE_BYTES] = rope_bytes
        scale_view[:] = torch.tensor(
            list(scale_bytes),
            dtype=torch.uint8,
            device=kv_cache_packed.device,
        )


def dequantize_v4_kv_slot(
    slot_or_cache: torch.Tensor,
    block_idx: Optional[int] = None,
    block_offset: Optional[int] = None,
    block_size: Optional[int] = None,
) -> torch.Tensor:
    """Inverse: read a single 584-byte slot back into a 512-dim bf16 K vector.

    Used by the FP8 sparse-attn reference (Stage 4C) to verify the
    quantize-then-dequantize round-trip stays within fp8_e4m3 precision.

    Args:
        slot_or_cache: either a materialized ``[584]`` logical slot, or the
            full ``[num_blocks, block_size, 584]`` MODEL1 cache.
        block_idx/block_offset: when provided, read the slot from the full
            cache using MODEL1's split physical layout.

    Returns:
        ``[512]`` bf16 — K vector reconstructed from the packed slot.
    """
    if block_idx is None:
        assert slot_or_cache.dtype == torch.uint8 and slot_or_cache.shape == (
            ENTRY_BYTES,
        )
        nope_rope = slot_or_cache[:NOPE_ROPE_STRIDE]
        scale_bytes = slot_or_cache[NOPE_ROPE_STRIDE:]
    else:
        assert block_offset is not None
        nope_rope, scale_bytes = _model1_slot_views(
            slot_or_cache,
            int(block_idx),
            int(block_offset),
            block_size,
        )

    # NoPE: 7 tiles. Match the input device — slot_or_cache comes from kv_cache_fp8
    # which is on the model's CUDA device; allocating ``nope_out`` on the
    # default (CPU) device caused ``torch.cat([nope_out, rope_out])`` to fail.
    nope_out = torch.empty(NOPE_DIM, dtype=torch.bfloat16, device=slot_or_cache.device)
    for tile_idx in range(NOPE_TILES):
        scale = _ue8m0_byte_to_scale(int(scale_bytes[tile_idx].item()))
        tile_quant = nope_rope[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE]
        tile_fp8 = tile_quant.view(torch.float8_e4m3fn)
        nope_out[tile_idx * TILE_SIZE : (tile_idx + 1) * TILE_SIZE] = (
            tile_fp8.float() * scale
        ).to(torch.bfloat16)

    # RoPE: bf16 view of the 128-byte slice
    rope_out = (
        nope_rope[NOPE_BYTES : NOPE_BYTES + ROPE_BYTES]
        .contiguous()
        .view(torch.bfloat16)
    )

    return torch.cat([nope_out, rope_out.clone()], dim=0)
