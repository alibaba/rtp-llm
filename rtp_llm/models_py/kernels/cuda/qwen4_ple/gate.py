# SPDX-License-Identifier: Apache-2.0
"""PLE gate consumer, following vLLM's fused gate with RTP rounding boundaries.

Reference: vLLM models/qwen4_exp/common/ops/ple.py, _ple_gate_kernel.
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
    DotBuffer,
    WIDTH: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    KEY_STRIDE: tl.constexpr,
    VALUE_STRIDE: tl.constexpr,
    HIDDEN_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    PROJECTED_FP32: tl.constexpr = False,
    STAGE: tl.constexpr = "fused",
):
    # A long prefill can address more than 2^31 activation elements.
    row = tl.program_id(0).to(tl.int64)
    branch = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    valid = col < H
    hc_col = branch * H + col
    if STAGE != "epilogue":
        k = tl.load(Key + row * KEY_STRIDE + hc_col, valid, other=0).to(tl.float32)
        if PROJECTED_FP32:
            k = (k * tl.load(KS + row // 128)).to(tl.bfloat16).to(tl.float32)
        q = tl.load(Hidden + row * HIDDEN_STRIDE + hc_col, valid, other=0).to(
            tl.float32
        )
        gk = tl.load(GK + hc_col, valid, other=0).to(tl.float32)
        gq = tl.load(GQ + hc_col, valid, other=0).to(tl.float32)
        kn = (k * tl.rsqrt(tl.sum(k * k, 0) / H + EPS) * (1 + gk)).to(tl.bfloat16)
        qn = (q * tl.rsqrt(tl.sum(q * q, 0) / H + EPS) * (1 + gq)).to(tl.bfloat16)
        product = (kn.to(tl.float32) * qn.to(tl.float32)).to(tl.bfloat16)
        if STAGE == "products":
            tl.store(DotBuffer + row * WIDTH + hc_col, product, valid)
    if STAGE != "products":
        if STAGE == "epilogue":
            dot = tl.load(DotBuffer + row * 4 + branch).to(tl.float32)
        else:
            dot = tl.sum(product.to(tl.float32), 0).to(tl.bfloat16).to(tl.float32)
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
    key,
    value,
    hidden,
    norm_key,
    norm_query,
    norm_conv,
    eps,
    projection_scales=None,
    *,
    reuse_key=False,
    batch_invariant=False,
):
    if not (hidden.is_cuda and torch.version.hip is None and hidden.ndim == 3):
        return None
    width = hidden.shape[-1]
    capability = torch.cuda.get_device_capability(hidden.device)
    projected = projection_scales is not None
    if not (
        (capability == (12, 0) and width == 10240)
        or (capability in ((10, 0), (10, 3)) and width == 32768 and not projected)
    ):
        return None
    if batch_invariant and (capability != (10, 3) or width != 32768):
        return None
    group_size = width // 4
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
        (hidden.shape[0] == 1 or all(x.is_contiguous() for x in (key, value, hidden)))
        and hidden.dtype == torch.bfloat16
        and key.dtype == value.dtype == (torch.float32 if projected else torch.bfloat16)
        and key.shape[-1] == width
        and value.shape[-1] == group_size
        and key.shape[:-1] == value.shape[:-1] == hidden.shape[:-1]
        and all(x.device == hidden.device and x.stride(-1) == 1 for x in (key, value))
        and hidden.stride(-1) == 1
        and all(
            g.shape == (width,)
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
    # The owner may donate a fresh BF16 projection. Each product CTA replaces
    # only the key elements for its own row and branch.
    gated = (
        key
        if reuse_key
        and not batch_invariant
        and group_size == 8192
        and key.is_contiguous()
        else torch.empty(hidden.shape, device=hidden.device, dtype=hidden.dtype)
    )
    normed = torch.empty_like(gated)
    batch, length, _ = hidden.shape
    rows = batch * length
    if (
        not batch_invariant
        and capability == (10, 3)
        and group_size == 8192
        and (rows >= 4 or (length == 1 and batch > 0))
        and all(x.is_contiguous() for x in (key, value, hidden))
    ):
        from .gate_gluon import m11_gate

        # ATen uses fewer Y lanes when reducing fewer than 16 branches.
        block_x = 128 if rows == 1 else 64 if rows < 4 else 32
        m11_gate[(rows, 4)](
            key,
            value,
            hidden,
            norm_key,
            norm_query,
            norm_conv,
            gated,
            normed,
            EPS=eps,
            BLOCK_X=block_x,
            num_warps=4,
        )
        return gated, normed
    if batch and length:
        # Contiguous tensors may have arbitrary strides on size-one axes.
        row_axis = 0 if length == 1 else 1
        kernel = _ple_gate[(batch * length, 4)]
        args = (
            key,
            value,
            hidden,
            norm_key,
            norm_query,
            norm_conv,
            gated,
            normed,
            *(projection_scales if projected else (None, None)),
        )
        config = dict(
            WIDTH=width,
            H=group_size,
            EPS=eps,
            KEY_STRIDE=key.stride(row_axis),
            VALUE_STRIDE=value.stride(row_axis),
            HIDDEN_STRIDE=hidden.stride(row_axis),
            BLOCK=triton.next_power_of_2(group_size),
            num_warps=4,
            PROJECTED_FP32=projected,
        )
        if group_size == 8192 and not batch_invariant:
            # The BF16 agreement can cross a rounding midpoint if Triton
            # sums the products in a different order than the original op.
            products = gated.view(batch, length, 4, group_size)
            kernel(*args, products, **config, STAGE="products")
            dot = products.sum(dim=-1)
            kernel(*args, dot, **config, STAGE="epilogue")
        else:
            kernel(*args, None, **config)
    return gated, normed
