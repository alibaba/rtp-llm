# SPDX-License-Identifier: Apache-2.0
"""PLE gate consumer, following vLLM's fused gate with RTP rounding boundaries.

Reference: vLLM models/qwen4_exp/nvidia/ops/ple.py, _ple_gate_kernel.
Projection quantization and page ownership stay with the caller.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _ple_gate(
    Key,
    Value,
    Hidden,
    GK,
    GQ,
    GC,
    Gated,
    Normed,
    KS,
    VS,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    PROJECTED_FP32: tl.constexpr = False,
):
    row, branch = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < H
    hc_col = branch * H + col
    k = tl.load(Key + row * KEY_STRIDE + hc_col, valid, other=0).to(tl.float32)
    if PROJECTED_FP32:
        k = (k * tl.load(KS + row // 128)).to(tl.bfloat16).to(tl.float32)
    q = tl.load(Hidden + row * HIDDEN_STRIDE + hc_col, valid, other=0).to(tl.float32)
    gk = tl.load(GK + hc_col, valid, other=0).to(tl.float32)
    gq = tl.load(GQ + hc_col, valid, other=0).to(tl.float32)
    kn = (k * tl.rsqrt(tl.sum(k * k, 0) / H + EPS) * (1 + gk)).to(tl.bfloat16)
    qn = (q * tl.rsqrt(tl.sum(q * q, 0) / H + EPS) * (1 + gq)).to(tl.bfloat16)
    product = (kn.to(tl.float32) * qn.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    dot = tl.sum(product, 0).to(tl.bfloat16).to(tl.float32)
    agreement = (dot / tl.sqrt(float(H))).to(tl.bfloat16).to(tl.float32)
    # clamp_min is a BF16 operation in the reference, before BF16 sqrt.
    floor = tl.full((), 1e-6, tl.bfloat16).to(tl.float32)
    magnitude = (
        tl.sqrt(tl.maximum(tl.abs(agreement), floor)).to(tl.bfloat16).to(tl.float32)
    )
    sign = tl.where(agreement > 0, 1.0, tl.where(agreement < 0, -1.0, 0.0))
    signed = (magnitude * sign).to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(signed).to(tl.bfloat16).to(tl.float32)
    value = tl.load(Value + row * VALUE_STRIDE + col, valid, other=0).to(tl.float32)
    if PROJECTED_FP32:
        value = (value * tl.load(VS + row // 128)).to(tl.bfloat16).to(tl.float32)
    gated = (gate * value).to(tl.bfloat16)
    f = gated.to(tl.float32)
    gc = tl.load(GC + hc_col, valid, other=0).to(tl.float32)
    normed = f * tl.rsqrt(tl.sum(f * f, 0) / H + EPS) * (1 + gc)
    tl.store(Gated + row * WIDTH + hc_col, gated, valid)
    tl.store(Normed + row * WIDTH + hc_col, normed, valid)


def try_ple_gate(
    key, value, hidden, norm_key, norm_query, norm_conv, eps, projection_scales=None
):
    projected = projection_scales is not None
    if projected and not (
        hidden.shape[0] == 1
        and len(projection_scales) == 2
        and all(
            s.shape == (triton.cdiv(hidden.shape[1], 128),)
            and s.device == hidden.device
            and s.dtype == torch.float32
            and s.is_contiguous()
            for s in projection_scales
        )
    ):
        return None
    if not (
        hidden.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(hidden.device) == (12, 0)
        and hidden.ndim == 3
        and (
            hidden.shape[0] == 1 or all(x.is_contiguous() for x in (key, value, hidden))
        )
        and hidden.dtype == torch.bfloat16
        and key.dtype == value.dtype == (torch.float32 if projected else torch.bfloat16)
        and hidden.shape[-1] == key.shape[-1] == 10240
        and value.shape[-1] == 2560
        and key.shape[:-1] == value.shape[:-1] == hidden.shape[:-1]
        and all(x.device == hidden.device and x.stride(-1) == 1 for x in (key, value))
        and hidden.stride(-1) == 1
        and all(
            g.shape == (10240,)
            and g.device == hidden.device
            and g.dtype == torch.bfloat16
            and g.is_contiguous()
            for g in (norm_key, norm_query, norm_conv)
        )
        and math.isfinite(eps)
        and eps >= 0
        and not any(
            x.requires_grad
            for x in (key, value, hidden, norm_key, norm_query, norm_conv)
        )
    ):
        return None
    gated = torch.empty(hidden.shape, device=hidden.device, dtype=hidden.dtype)
    normed = torch.empty_like(gated)
    if hidden.shape[1]:
        _ple_gate[(hidden.shape[0] * hidden.shape[1], 4)](
            key,
            value,
            hidden,
            norm_key,
            norm_query,
            norm_conv,
            gated,
            normed,
            *(projection_scales if projected else (None, None)),
            WIDTH=10240,
            H=2560,
            EPS=eps,
            KEY_STRIDE=key.stride(1),
            VALUE_STRIDE=value.stride(1),
            HIDDEN_STRIDE=hidden.stride(1),
            BLOCK=4096,
            num_warps=4,
            PROJECTED_FP32=projected,
        )
    return gated, normed
