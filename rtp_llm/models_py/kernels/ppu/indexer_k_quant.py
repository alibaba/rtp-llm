"""PPU indexer K-cache writer with reciprocal-then-multiply quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_indexer_k(
    key_ptr,
    cache_ptr,
    slots_ptr,
    key_stride: tl.constexpr,
    cache_stride: tl.constexpr,
    page_size: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + row).to(tl.int64)
    if slot >= 0:
        offsets = tl.arange(0, 128)
        values = tl.load(key_ptr + row * key_stride + offsets).to(tl.float32)
        scale = tl.div_rn(tl.maximum(tl.max(tl.abs(values)), 1e-4), 448.0)
        # Direct values / scale changes FP8 rounding ties on real rotated K.
        quantized = tl.clamp(values * tl.div_rn(1.0, scale), -448.0, 448.0).to(
            tl.float8e4nv
        )
        page = cache_ptr + slot // page_size * cache_stride
        offset = slot % page_size
        tl.store(page + offset * 128 + offsets, quantized.to(tl.uint8, bitcast=True))
        tl.store(
            (page + page_size * 128 + offset * 4).to(tl.pointer_type(tl.float32)), scale
        )


def quantize_and_cache_indexer_k(key, cache, slots):
    """Write BF16 [T,128] into grouped [pages,page_size,132] byte storage."""
    assert key.ndim == 2 and key.shape[1] == 128 and key.dtype == torch.bfloat16
    assert key.stride(1) == 1
    assert cache.ndim == 3 and cache.shape[2] == 132 and cache.dtype == torch.uint8
    assert cache.is_contiguous()
    assert slots.ndim == 1 and slots.dtype == torch.int64 and slots.is_contiguous()
    assert slots.numel() <= key.shape[0]
    if slots.numel() == 0:
        return
    _write_indexer_k[(slots.numel(),)](
        key, cache, slots, key.stride(0), cache.stride(0), cache.shape[1]
    )
