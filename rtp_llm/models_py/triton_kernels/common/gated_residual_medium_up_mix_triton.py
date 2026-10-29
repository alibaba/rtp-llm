"""SM120 multi-row FP8 HC up projection with BF16 gate/product boundaries."""

import torch
import triton
import triton.language as tl


@triton.jit
def _hc_medium_up_mix(
    A,
    W,
    SA,
    SW,
    NORM,
    Y,
    M: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    packed = tl.arange(0, 4 * BN)
    c = tl.program_id(1) * BN + packed % BN + (packed // BN) * 2560
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, 4 * BN), tl.float32)
    for block in range(tl.cdiv(320, BK)):
        kk = block * BK + k
        a = tl.load(
            A.to(tl.pointer_type(tl.uint8)) + r[:, None] * 320 + kk[None, :],
            (r[:, None] < M) & (kk[None, :] < 320),
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        w = tl.load(
            W.to(tl.pointer_type(tl.uint8)) + c[None, :] * 320 + kk[:, None],
            (c[None, :] < 10240) & (kk[:, None] < 320),
            other=0,
        ).to(tl.float8e4nv, bitcast=True)
        acc = tl.dot(a, w, acc)
    logits = (acc * (tl.load(SA) * tl.load(SW))).to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
    norm = tl.load(
        NORM + r[:, None] * 10240 + c[None, :],
        (r[:, None] < M) & (c[None, :] < 10240),
        other=0,
    ).to(tl.float32)
    product = (gate * norm).to(tl.bfloat16).to(tl.float32).reshape(BM, 4, BN)
    branches = tl.arange(0, 4)
    output = tl.zeros((BM, BN), tl.float32)
    for branch in tl.static_range(4):
        output += tl.sum(tl.where(branches[None, :, None] == branch, product, 0.0), 1)
    tl.store(
        Y + r[:, None] * 2560 + n[None, :],
        output / 4.0,
        (r[:, None] < M) & (n[None, :] < 2560),
    )


def run_quantized(a, weight, input_scale, weight_scale, normed):
    m = normed.shape[0]
    out = normed.new_empty((m, 2560))
    _hc_medium_up_mix[(triton.cdiv(m, 16), 80)](
        a,
        weight,
        input_scale,
        weight_scale,
        normed,
        out,
        M=m,
        BM=16,
        BN=32,
        BK=64,
        num_warps=4,
        num_stages=2,
        enable_fp_fusion=False,
    )
    return out


def try_fp8_hc_medium_up_mix(x, weight, scale, normed, branches):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim >= 2
        and x.shape[-1] == 320
        and 8 <= x.numel() // 320 <= 32
        and branches == 4
        and x.dtype == torch.bfloat16
        and tuple(weight.shape) == (10240, 320)
        and weight.dtype == torch.float8_e4m3fn
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and normed.shape == (*x.shape[:-1], 10240)
        and normed.dtype == torch.bfloat16
        and x.device == weight.device == scale.device == normed.device
        and all(
            t.is_contiguous() and not t.requires_grad
            for t in (x, weight, scale, normed)
        )
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return None
    from rtp_llm.models_py.modules.base.common.fp8_linear import (
        _quantize_projection_input,
    )

    rows = x.reshape(-1, 320)
    padding = (-rows.shape[0]) % 16
    padded = (
        torch.empty(
            (rows.shape[0] + padding, 320), device=x.device, dtype=torch.float8_e4m3fn
        )
        if padding
        else None
    )
    quantized, input_scale = _quantize_projection_input(
        rows, role="DECODE", output=padded
    )
    return run_quantized(
        quantized, weight, input_scale, scale, normed.reshape(-1, 10240)
    ).reshape(*x.shape[:-1], 2560)
