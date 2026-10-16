"""Vectorized GDN gating with optional FP32 beta for FlashInfer prefill."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.prefill_input_log import trace_triton


@triton.jit
def _gdn_gating_flat(
    A,
    B,
    AL,
    DT,
    G,
    BET,
    COUNT,
    H: tl.constexpr,
    SA: tl.constexpr,
    SB: tl.constexpr,
    SOFT_BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLOCK: tl.constexpr,
    STABLE_SOFTPLUS: tl.constexpr,
    FLASHINFER: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // H
    head = i % H
    mask = i < COUNT
    a = tl.load(A + row.to(tl.int64) * SA + head, mask, 0).to(tl.float32)
    b = tl.load(B + row.to(tl.int64) * SB + head, mask, 0).to(tl.float32)
    al = tl.load(AL + head).to(tl.float32)
    dt = tl.load(DT + head).to(tl.float32)
    x = a + dt
    if STABLE_SOFTPLUS and SOFT_BETA == 1.0:
        # Align FP32 softplus with vLLM's FlashInfer prefill operation order.
        sp = tl.where(x > 0, x + tl.log(1.0 + tl.exp(-x)), tl.log(1.0 + tl.exp(x)))
        sp = tl.where(x <= THRESHOLD, sp, x)
    else:
        sp = tl.where(
            SOFT_BETA * x <= THRESHOLD,
            (1 / SOFT_BETA) * tl.log(1 + tl.exp(SOFT_BETA * x)),
            x,
        )
    g = -tl.exp(al) * sp
    beta_value = tl.sigmoid(b)
    if FLASHINFER:
        g = tl.extra.cuda.libdevice.exp(g)
        # vLLM instead stores FP32 log-g first, then calls torch.exp separately.
    tl.store(G + i, g, mask)
    tl.store(BET + i, beta_value, mask)


def gdn_gating_prefill(
    alog,
    a,
    b,
    dt_bias,
    beta=1.0,
    threshold=20.0,
    *,
    block=256,
    flashinfer=False,
    beta_fp32=False,
    stable_softplus=False,
):
    if a.ndim != 2 or b.shape != a.shape or a.stride(1) != 1 or b.stride(1) != 1:
        raise ValueError("Expected matching token/head matrices with contiguous heads")
    n, h = a.shape
    if alog.shape != (h,) or dt_bias.shape != (h,):
        raise ValueError("Invalid gate parameters")
    if not a.is_cuda or torch.version.hip is not None:
        raise ValueError("CUDA required")
    g = torch.empty((1, n, h), device=a.device, dtype=torch.float32)
    # Align FlashInfer prefill beta with vLLM: store the FP32 sigmoid
    # result directly, without an intermediate BF16 rounding.
    bet = torch.empty(
        (1, n, h),
        device=a.device,
        dtype=torch.float32 if flashinfer or beta_fp32 else b.dtype,
    )
    if n:
        trace_triton(
            "gdn_gating_prefill:_gdn_gating_flat[triton.cdiv(n * h, block),]",
            _gdn_gating_flat,
            (triton.cdiv(n * h, block),),
            a,
            b,
            alog,
            dt_bias,
            g,
            bet,
            n * h,
            h,
            a.stride(0),
            b.stride(0),
            beta,
            threshold,
            block,
            stable_softplus or flashinfer,
            flashinfer,
            num_warps=4,
        )
    return g, bet


def supports_gdn_gating_prefill(alog, a, b, dt_bias):
    return (
        a.is_cuda
        and torch.version.hip is None
        and a.ndim == 2
        and a.shape[0] >= 2048
        and a.shape[1] > 0
        and b.shape == a.shape
        and a.dtype == b.dtype == torch.bfloat16
        and alog.dtype in (torch.bfloat16, torch.float32)
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and alog.device == dt_bias.device == b.device == a.device
        and alog.shape == dt_bias.shape == (a.shape[1],)
        and a.stride(1) == b.stride(1) == alog.stride(0) == dt_bias.stride(0) == 1
        and torch.cuda.get_device_capability(a.device)[0] == 10
    )
