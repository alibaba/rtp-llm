"""Contiguous FP8 K workspace for SM90 DeepGEMM V4.1 prefill indexer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class PrefillFP8IndexerKeys:
    quant: torch.Tensor
    scale: torch.Tensor

    def __len__(self) -> int:
        return self.quant.shape[0]


@triton.jit
def _gather_raw_fp8_pages(pool, slots, out, PAGE_STRIDE: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    page = slot // 64
    offset = slot % 64
    byte = tl.arange(0, 256)
    source = pool + page * PAGE_STRIDE + tl.where(
        byte < 128, offset * 128 + byte, 64 * 128 + offset * 4 + byte - 128
    )
    value = tl.load(source, (slot >= 0) & (byte < 132), other=0)
    tl.store(out + token * 132 + byte, value, byte < 132)


def gather_raw_fp8_indexer(pages: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    """Gather `[K128|scale4]` bytes in request order, preserving CP transport."""
    if pages.dtype != torch.uint8 or pages.shape[1:] != (64, 132):
        raise ValueError("FP8 indexer pages must be uint8 [P,64,132]")
    slots = slots.contiguous().view(-1)
    raw = torch.empty((slots.numel(), 132), dtype=torch.uint8, device=pages.device)
    if slots.numel():
        _gather_raw_fp8_pages[(slots.numel(),)](
            pages, slots, raw, PAGE_STRIDE=64 * 132, num_warps=4
        )
    return raw


@triton.jit
def _gather_fp8_pages(pool, slots, quant, scales, PAGE_STRIDE: tl.constexpr):
    token = tl.program_id(0)
    slot = tl.load(slots + token)
    page = slot // 64
    offset = slot % 64
    byte = tl.arange(0, 128)
    payload = tl.load(
        pool + page * PAGE_STRIDE + offset * 128 + byte,
        slot >= 0, other=0,
    )
    scale = tl.load(
        (pool + page * PAGE_STRIDE + 64 * 128 + offset * 4).to(tl.pointer_type(tl.float32)),
        slot >= 0, other=1.0,
    )
    tl.store((quant + token * 128 + byte).to(tl.pointer_type(tl.uint8)), payload)
    tl.store(scales + token, scale)


def gather_fp8_indexer(pages: torch.Tensor, slots: torch.Tensor) -> PrefillFP8IndexerKeys:
    """Direct non-CP gather into DeepGEMM's two contiguous K operands."""
    if pages.dtype != torch.uint8 or pages.shape[1:] != (64, 132):
        raise ValueError("FP8 indexer pages must be uint8 [P,64,132]")
    slots = slots.contiguous().view(-1)
    quant = torch.empty((slots.numel(), 128), dtype=torch.float8_e4m3fn, device=pages.device)
    scale = torch.empty((slots.numel(),), dtype=torch.float32, device=pages.device)
    if slots.numel():
        _gather_fp8_pages[(slots.numel(),)](
            pages, slots, quant, scale, PAGE_STRIDE=64 * 132, num_warps=4
        )
    return PrefillFP8IndexerKeys(quant, scale)


def unpack_raw_fp8_indexer(raw: torch.Tensor) -> PrefillFP8IndexerKeys:
    if raw.dtype != torch.uint8 or raw.ndim != 2 or raw.shape[1] != 132:
        raise ValueError("raw FP8 indexer keys must be uint8 [N,132]")
    quant = raw[:, :128].contiguous().view(torch.float8_e4m3fn)
    scale = raw[:, 128:].contiguous().view(torch.float32).view(-1)
    return PrefillFP8IndexerKeys(quant, scale)


@triton.jit(do_not_specialize=["N"])
def _quantize_contiguous_k_kernel(keys, quant, scales, N):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, 128)
    x = tl.load(keys + row * 128 + d, row < N, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1e-12)
    packed = (x / scale).to(tl.float8e4nv)
    tl.store(
        (quant + row * 128 + d).to(tl.pointer_type(tl.uint8)),
        packed.to(tl.uint8, bitcast=True), row < N,
    )
    tl.store(scales + row, scale, row < N)


def quantize_contiguous_fp8_indexer(keys: torch.Tensor) -> PrefillFP8IndexerKeys:
    """Warm/no-cache path with the same per-key FP8 codec as the cache writer."""
    if keys.ndim != 2 or keys.shape[1] != 128 or keys.dtype != torch.bfloat16:
        raise ValueError("contiguous indexer K must be BF16 [N,128]")
    keys = keys.contiguous()
    count = keys.shape[0]
    quant = torch.empty((count, 128), dtype=torch.float8_e4m3fn, device=keys.device)
    scale = torch.empty((count,), dtype=torch.float32, device=keys.device)
    if count:
        _quantize_contiguous_k_kernel[(count,)](
            keys, quant, scale, count, num_warps=4
        )
    return PrefillFP8IndexerKeys(quant, scale)


def score_fp8_prefill_chunk(
    q_fp8: torch.Tensor,
    w_fold: torch.Tensor,
    keys: PrefillFP8IndexerKeys,
    visible: torch.Tensor,
) -> torch.Tensor:
    import deep_gemm

    starts = torch.zeros_like(visible, dtype=torch.int32)
    ends = visible.to(torch.int32).contiguous()
    return deep_gemm.fp8_fp4_mqa_logits(
        (q_fp8.contiguous(), None),
        (keys.quant, keys.scale),
        w_fold.float().contiguous(), starts, ends,
        len(keys),
    )
