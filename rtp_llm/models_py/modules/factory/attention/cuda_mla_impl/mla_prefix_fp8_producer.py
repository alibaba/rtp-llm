"""Paged latent gather with group128 quantization for the KV-B projection."""

import math

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    allocate_quantized,
    retained_bf16,
)
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_producers import store_group128


@triton.jit(do_not_specialize=["M", "BATCH"])
def _gather_quantized(
    C,
    R,
    CACHE,
    PAGES,
    INFO,
    QI,
    Y,
    S,
    OR,
    M,
    BATCH,
    PAGE: tl.constexpr,
    CP: tl.constexpr,
    CT: tl.constexpr,
    CS: tl.constexpr,
    RS: tl.constexpr,
    OS: tl.constexpr,
    OD: tl.constexpr,
    K: tl.constexpr,
    ROPE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = 0
    for batch in range(BATCH):
        start = tl.load(QI + batch)
        end = tl.load(QI + batch + 1)
        reuse = tl.load(INFO + batch * 4 + 1)
        count = reuse + end - start
        if (row >= offset) & (row < offset + count):
            local = row - offset
            if local < reuse:
                page_offset = tl.load(INFO + batch * 4 + 2)
                page = tl.load(PAGES + page_offset + local // PAGE)
                base = page.to(tl.int64) * CP + (local % PAGE) * CT
                latent = (
                    tl.load(CACHE + base + col, col < K, other=0.0).to(tl.float32)
                    * SCALE
                )
                rope = (
                    tl.load(CACHE + base + K + col, col < ROPE, other=0.0).to(
                        tl.float32
                    )
                    * SCALE
                )
            else:
                source = start + local - reuse
                latent = tl.load(C + source * CS + col, col < K, other=0.0).to(
                    tl.float32
                )
                rope = tl.load(R + source * RS + col, col < ROPE, other=0.0).to(
                    tl.float32
                )
            # Old gather wrote BF16 before the independent GEMM quantizer.
            store_group128(latent, Y, S, row, M, K, BLOCK)
            tl.store(OR + row * OS + col * OD, rope.to(tl.bfloat16), col < ROPE)
        offset += count


class Fp8MlaPrefixGather:
    def __init__(self, cache_scale=1.0):
        if not math.isfinite(cache_scale) or cache_scale <= 0:
            raise ValueError("cache scale must be finite and positive")
        self.cache_scale = cache_scale

    def __call__(self, out_ckv, out_rope, ckv, rope, cache, pages, info, qi, page_size):
        ckv = retained_bf16(ckv)
        if cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16):
            raise TypeError("FP8 producer requires ordinary E4M3 or BF16 cache")
        tensors = (out_ckv, out_rope, ckv, rope, cache, pages, info, qi)
        if any(not x.is_cuda or x.device != cache.device for x in tensors):
            raise ValueError("prefix inputs must share a CUDA device")
        if ckv.dtype != torch.bfloat16 or rope.dtype != torch.bfloat16:
            raise TypeError("current latent and suffix must retain BF16 values")
        m, k = out_ckv.shape
        if (
            ckv.ndim != 2
            or ckv.shape[1] != k
            or ckv.stride(1) != 1
            or rope.stride(1) != 1
        ):
            raise ValueError("invalid current latent layout")
        if (
            cache.ndim != 3
            or cache.shape[1:] != (page_size, k + rope.shape[1])
            or cache.stride(-1) != 1
        ):
            raise ValueError("invalid ordinary MLA cache layout")
        if info.ndim != 2 or info.shape[1] != 4 or qi.numel() != info.shape[0] + 1:
            raise ValueError("invalid prefix metadata")
        if any(
            x.dtype != torch.int32 or not x.is_contiguous() for x in (pages, info, qi)
        ):
            raise ValueError("prefix metadata must be contiguous int32")
        if out_rope.shape != (m, rope.shape[1]) or out_rope.dtype != torch.bfloat16:
            raise ValueError("invalid prefix suffix output")
        result = allocate_quantized(m, k, cache.device)
        if m:
            _gather_quantized[(m,)](
                ckv,
                rope,
                cache,
                pages,
                info,
                qi,
                result.values,
                result.scale_wire,
                out_rope,
                m,
                info.shape[0],
                page_size,
                cache.stride(0),
                cache.stride(1),
                ckv.stride(0),
                rope.stride(0),
                out_rope.stride(0),
                out_rope.stride(1),
                k,
                rope.shape[1],
                self.cache_scale,
                max(512, triton.next_power_of_2(k)),
            )
        return result
