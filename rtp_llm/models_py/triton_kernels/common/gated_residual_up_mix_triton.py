"""Preserve the small HC GEMV and BF16 gate/product boundaries in one launch."""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.common.fp8_gemv import is_supported


@triton.jit
def _fp8_hc_up_mix_kernel(X, W, S, NORM, Y, M: tl.constexpr, BN: tl.constexpr):
    program = tl.program_id(0)
    row = program % M
    column = program // M * BN + tl.arange(0, BN)
    k = tl.arange(0, 512)
    x = tl.load(X + row * 320 + k, k < 320, other=0).to(tl.float32)
    scale = tl.load(S)
    accumulator = tl.full((BN,), 0, tl.float32)
    weight_bytes = W.to(tl.pointer_type(tl.uint8))
    for branch in tl.static_range(4):
        c = column + branch * 2560
        weights = (
            tl.load(
                weight_bytes + c[:, None] * 320 + k[None, :],
                (column[:, None] < 2560) & (k[None, :] < 320),
                other=0,
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        logits = (
            (tl.sum(weights * x[None, :], 1) * scale).to(tl.bfloat16).to(tl.float32)
        )
        gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
        normed = tl.load(NORM + row * 10240 + c, column < 2560, other=0).to(tl.float32)
        product = (gate * normed).to(tl.bfloat16).to(tl.float32)
        accumulator += product
    tl.store(Y + row * 2560 + column, accumulator / 4.0, column < 2560)


@triton.jit
def _fp8_hc_up_mix_four_rows(X, W, S, NORM, Y):
    columns = tl.program_id(0) * 2 + tl.arange(0, 2)
    k = tl.arange(0, 512)
    rows = tl.arange(0, 4)
    accumulator = tl.zeros((4, 2), tl.float32)
    scale = tl.load(S)
    for branch in tl.static_range(4):
        c = columns + branch * 2560
        weight = (
            tl.load(
                W.to(tl.pointer_type(tl.uint8)) + c[:, None] * 320 + k[None, :],
                k[None, :] < 320,
                other=0,
            )
            .to(tl.float8e4nv, bitcast=True)
            .to(tl.float32)
        )
        # Reuse each FP8 weight load, retaining the per-row reduction and
        # every BF16 boundary of the single-row kernel.
        for row in tl.static_range(4):
            x = tl.load(X + row * 320 + k, k < 320, other=0).to(tl.float32)
            logits = (
                (tl.sum(weight * x[None, :], 1) * scale).to(tl.bfloat16).to(tl.float32)
            )
            gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
            normed = tl.load(NORM + row * 10240 + c).to(tl.float32)
            product = (gate * normed).to(tl.bfloat16).to(tl.float32)
            accumulator += tl.where(rows[:, None] == row, product[None, :], 0.0)
    tl.store(Y + rows[:, None] * 2560 + columns[None, :], accumulator / 4.0)


def try_fp8_up_mix(x, weight, scale, normed, branches):
    if x.ndim < 2 or x.shape[-1] != 320 or not x.is_contiguous():
        return None
    rows = x.reshape(-1, x.shape[-1])
    if rows.shape[0] > 4:
        from rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton import (
            try_fp8_hc_medium_up_mix,
        )

        return try_fp8_hc_medium_up_mix(x, weight, scale, normed, branches)
    if not (
        branches == 4
        and is_supported(rows, weight, scale)
        and normed.shape == (*x.shape[:-1], 10240)
        and normed.dtype == torch.bfloat16
        and normed.device == x.device
        and normed.is_contiguous()
        and not any(t.requires_grad for t in (x, weight, scale, normed))
    ):
        return None
    output = x.new_empty((*x.shape[:-1], 2560))
    if rows.shape[0] == 4:
        _fp8_hc_up_mix_four_rows[(1280,)](
            rows, weight, scale, normed, output, num_warps=1
        )
        return output
    _fp8_hc_up_mix_kernel[(triton.cdiv(2560, 4) * rows.shape[0],)](
        rows,
        weight,
        scale,
        normed,
        output,
        M=rows.shape[0],
        BN=4,
        num_warps=1 if rows.shape[0] > 1 else 4,
    )
    return output
