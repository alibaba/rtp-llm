"""PPU MLA FP8 cache packing with the SGL quantization arithmetic."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_mla_fp8(
    kv_ptr, pe_ptr, cache_ptr, slots_ptr,
    kv_stride: tl.constexpr, pe_stride: tl.constexpr,
    cache_block_stride: tl.constexpr, cache_row_stride: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot >= 0:
        dst = cache_ptr + slot // block_size * cache_block_stride + slot % block_size * cache_row_stride
        offsets = tl.arange(0, 128)
        if tile < 4:
            values = tl.load(kv_ptr + row * kv_stride + tile * 128 + offsets).to(tl.float32)
            # Match quantize_k_cache_separate: divide amax, then multiply by
            # the reciprocal scale. Direct values / scale can change FP8 ties.
            scale = tl.max(tl.abs(values)) / 448.0
            scale = tl.maximum(scale, 1.1754943508222875e-38)
            quantized = tl.clamp(values * (1.0 / scale), -448.0, 448.0).to(tl.float8e4nv)
            tl.store(dst + tile * 128 + offsets, quantized.to(tl.uint8, bitcast=True))
            tl.store((dst + 512).to(tl.pointer_type(tl.float32)) + tile, scale)
        else:
            rope_values = tl.load(pe_ptr + row * pe_stride + offsets, offsets < 64, other=0)
            tl.store((dst + 528).to(tl.pointer_type(tl.bfloat16)) + offsets, rope_values, offsets < 64)


def concat_and_cache_mla_fp8(ckv, kpe, cache, slots):
    """Pack BF16 [T,512]/[T,64] into paged 656-byte MLA cache rows."""
    # Graph batches may pad the input while slot_mapping contains only the
    # real rows, matching the native concat_and_cache_mla contract.
    if slots.numel() == 0:
        return
    if cache.dtype != torch.uint8:
        cache = cache.view(torch.uint8)
    _write_mla_fp8[(slots.numel(), 5)](
        ckv, kpe, cache, slots, ckv.stride(0), kpe.stride(0),
        cache.stride(0), cache.stride(1), cache.shape[1],
    )
