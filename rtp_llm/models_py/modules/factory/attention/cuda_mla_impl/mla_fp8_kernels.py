"""Ordinary E4M3 MLA conversion and paged prefix reads.

Scales express dequantization (real = FP8 * scale). These kernels do not
implement the unrelated 656-byte, dynamic-group MLA cache representation.
"""

import math
import json
import logging
import os
from typing import Optional

import torch
import triton
import triton.language as tl


# Diagnostic runs only: this synchronizes observations to the CPU and must
# remain disabled during timing. Captured graphs contain no host observations.
_FP8_DIAGNOSTICS = os.environ.get("KIMI_K3_MLA_FP8_DIAGNOSTICS", "0") == "1"


def observe_fp8_input(x: torch.Tensor, scale: float, name: str) -> None:
    if not _FP8_DIAGNOSTICS or not x.numel() or torch.cuda.is_current_stream_capturing():
        return
    values = x.detach().float()
    finite = torch.isfinite(values)
    maximum = torch.where(finite, values.abs(), 0.0).amax()
    counts = torch.stack((maximum, (~finite).sum().float(),
                          (finite & (values.abs() > 448.0 * scale)).sum().float()))
    absmax, nonfinite, saturated = counts.cpu().tolist()
    logging.info("K3_MLA_FP8_RANGE %s", json.dumps(dict(
        operand=name, shape=list(x.shape), device=str(x.device), scale=scale,
        finite_absmax=absmax, elements=x.numel(), nonfinite=int(nonfinite),
        clipped=int(saturated), clipped_fraction=saturated / x.numel(),
        scope="eager observation; graph replay is not sampled")))


@triton.jit(do_not_specialize=["N", "SHAPE", "STRIDES"])
def _quantize(
    X, Y, N, INV_SCALE: tl.constexpr, BLOCK: tl.constexpr,
    SHAPE, STRIDES, CONTIGUOUS: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if CONTIGUOUS:
        src = offsets
    else:
        src = tl.full((BLOCK,), 0, tl.int64)
        remaining = offsets
        for dim in tl.static_range(len(SHAPE) - 1, -1, -1):
            src += (remaining % SHAPE[dim]).to(tl.int64) * STRIDES[dim]
            remaining = remaining // SHAPE[dim]
    x = tl.load(X + src, offsets < N, other=0.0).to(tl.float32) * INV_SCALE
    # Match CUDA's saturating E4M3 conversion; an unclamped torch cast can
    # generate NaNs on overflow.
    x = tl.where(x != x, x, tl.minimum(tl.maximum(x, -448.0), 448.0))
    tl.store(Y + offsets, x, offsets < N)


def quantize_fp8(
    x: torch.Tensor, scale: float = 1.0, out: Optional[torch.Tensor] = None,
    *, name: str = "attention_operand",
) -> torch.Tensor:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FP8 MLA scale must be finite and positive")
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"FP8 MLA quantizer requires floating activations, got {x.dtype}")
    if out is None:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    if (out.shape != x.shape or out.dtype != torch.float8_e4m3fn
            or out.device != x.device or not out.is_contiguous()):
        raise ValueError("FP8 MLA quantization output buffer mismatch")
    if _FP8_DIAGNOSTICS:
        observe_fp8_input(x, scale, name)
    if x.numel():
        # Runtime tuples avoid exact-shape specialization. Triton may still
        # keep a bounded set of tuple alignment variants for strided layouts.
        _quantize[(triton.cdiv(x.numel(), 1024),)](
            x, out, x.numel(), 1.0 / scale, 1024,
            () if x.is_contiguous() else tuple(x.shape),
            () if x.is_contiguous() else tuple(x.stride()), x.is_contiguous(),
        )
    return out


@triton.jit(do_not_specialize=["BATCH"])
def _gather_prefix(
    O_C, O_R, C, R, CACHE, PAGES, INFO, Q_IND,
    BATCH, PAGE_SIZE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr, CACHE_TOKEN_STRIDE: tl.constexpr,
    C_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
    LATENT: tl.constexpr, ROPE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offset = 0
    # Metadata follows reuse_kv_cache_indexed_batched: [unused, reuse_len,
    # flattened_page_offset, unused]. Each batch emits its prefix then suffix.
    for batch in range(BATCH):
        q_start = tl.load(Q_IND + batch)
        q_end = tl.load(Q_IND + batch + 1)
        reuse = tl.load(INFO + batch * 4 + 1)
        final_len = reuse + q_end - q_start
        if row >= offset and row < offset + final_len:
            local = row - offset
            col = tl.arange(0, BLOCK)
            if local < reuse:
                page_offset = tl.load(INFO + batch * 4 + 2)
                page = tl.load(PAGES + page_offset + local // PAGE_SIZE)
                base = page.to(tl.int64) * CACHE_PAGE_STRIDE + (local % PAGE_SIZE) * CACHE_TOKEN_STRIDE
                values = tl.load(CACHE + base + col, col < LATENT + ROPE, other=0.0).to(tl.float32) * SCALE
                tl.store(O_C + row * LATENT + col, values, col < LATENT)
                tl.store(O_R + row * ROPE + col - LATENT, values,
                         (col >= LATENT) & (col < LATENT + ROPE))
            else:
                src = q_start + local - reuse
                c = tl.load(C + src * C_STRIDE + col, col < LATENT, other=0)
                r = tl.load(R + src * R_STRIDE + col, col < ROPE, other=0)
                tl.store(O_C + row * LATENT + col, c, col < LATENT)
                tl.store(O_R + row * ROPE + col, r, col < ROPE)
        offset += final_len


def gather_fp8_prefix(
    out_ckv, out_rope, ckv, rope, cache, pages, batch_info, qo_indptr,
    page_size, *, scale,
):
    if cache.dtype != torch.float8_e4m3fn:
        raise TypeError("ordinary MLA FP8 prefix gather requires E4M3 cache")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FP8 MLA prefix scale must be finite and positive")
    tensors = (out_ckv, out_rope, ckv, rope, pages, batch_info, qo_indptr)
    if not cache.is_cuda or any(t.device != cache.device for t in tensors):
        raise ValueError("FP8 MLA prefix tensors must share one CUDA device")
    for tensor in (out_ckv, out_rope, ckv, rope):
        if tensor.ndim != 2 or tensor.dtype != torch.bfloat16 or tensor.stride(1) != 1:
            raise ValueError("FP8 MLA prefix activations require BF16 matrices with contiguous features")
    if (batch_info.ndim != 2 or batch_info.shape[1] != 4
            or qo_indptr.numel() != batch_info.shape[0] + 1):
        raise ValueError("FP8 MLA prefix metadata shape mismatch")
    for tensor in (pages, batch_info, qo_indptr):
        if tensor.dtype != torch.int32 or not tensor.is_contiguous():
            raise ValueError("FP8 MLA prefix metadata requires contiguous int32 tensors")
    latent, rope_dim = ckv.shape[-1], rope.shape[-1]
    if (out_ckv.shape[0] != out_rope.shape[0] or out_ckv.shape[1] != latent
            or out_rope.shape[1] != rope_dim or ckv.shape[0] != rope.shape[0]):
        raise ValueError("FP8 MLA prefix output/input shape mismatch")
    if cache.shape[-1] != latent + rope_dim or cache.stride(-1) != 1:
        raise ValueError("ordinary MLA FP8 cache width/stride mismatch")
    if cache.ndim != 3 or cache.shape[1] != page_size:
        raise ValueError("ordinary MLA FP8 cache must have [pages, page_size, features] layout")
    if not out_ckv.is_contiguous() or not out_rope.is_contiguous():
        raise ValueError("MLA prefix gather outputs must be contiguous")
    if not batch_info.is_contiguous() or not qo_indptr.is_contiguous():
        raise ValueError("MLA prefix gather metadata must be contiguous")
    if out_ckv.shape[0]:
        _gather_prefix[(out_ckv.shape[0],)](
            out_ckv, out_rope, ckv, rope, cache, pages, batch_info, qo_indptr,
            batch_info.numel() // 4, page_size, cache.stride(0), cache.stride(1),
            ckv.stride(0), rope.stride(0), latent, rope_dim, scale,
            triton.next_power_of_2(latent + rope_dim),
        )
