"""FP8-only producer kernels. BF16 dispatch keeps the original kernels."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.modules.factory.linear.quantized_activation import (
    allocate_quantized,
)


@triton.jit
def store_group128(
    values, out, scales, token, M, K: tl.constexpr, BLOCK: tl.constexpr
):
    """Preserve BF16 rounding; keep the request-dependent scale pitch dynamic."""
    cols = tl.arange(0, BLOCK)
    values = values.to(tl.bfloat16).to(tl.float32)
    grouped = tl.reshape(tl.where(cols < K, values, 0.0), (BLOCK // 128, 128))
    amax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-4)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(amax / 448.0, 1.0e-10))))
    quantized = tl.reshape(
        tl.minimum(tl.maximum(grouped / scale[:, None], -448.0), 448.0), (BLOCK,)
    )
    tl.store(out + token * K + cols, quantized, cols < K)
    groups = tl.arange(0, BLOCK // 128)
    exp = (scale.to(tl.int32, bitcast=True) >> 23) & 255
    exp = tl.where(groups < K // 128, exp, 127)
    # BLOCK is >=512 so all four bytes have a unique writer.
    exp4 = tl.reshape(exp, (BLOCK // 512, 4))
    shifts = tl.arange(0, 4) * 8
    packed = tl.sum(exp4 << shifts[None, :], axis=1)
    pg = tl.arange(0, BLOCK // 512)
    aligned_m = tl.cdiv(M, 4) * 4
    tl.store(scales + pg * aligned_m + token, packed, pg < triton.cdiv(K, 512))
    if token == 0:
        padding = M + tl.arange(0, 4)
        tl.store(
            scales + pg[:, None] * aligned_m + padding[None, :],
            0x7F7F7F7F,
            (pg[:, None] < triton.cdiv(K, 512)) & (padding[None, :] < aligned_m),
        )


@triton.jit
def rms_values(
    X,
    W,
    token,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + token * STRIDE + cols, cols < K, other=0.0).to(tl.float32)
    w = tl.load(W + cols, cols < K, other=0.0).to(tl.float32)
    # Match FlashInfer's vec8 accumulation, warp reduction, then warp sums.
    # A single flat reduction can move BF16 rounding boundaries in cache data.
    THREADS: tl.constexpr = min(1024, triton.cdiv(K, 8 * 32) * 32)
    PAD_THREADS: tl.constexpr = triton.next_power_of_2(THREADS)
    threads = tl.arange(0, PAD_THREADS)
    acc = tl.full((PAD_THREADS,), 0.0, tl.float32)
    for batch in tl.static_range(triton.cdiv(K, THREADS * 8)):
        for lane in tl.static_range(8):
            feature = (batch * THREADS + threads) * 8 + lane
            v = tl.load(
                X + token * STRIDE + feature,
                (threads < THREADS) & (feature < K),
                other=0.0,
            ).to(tl.float32)
            acc = tl.fma(v, v, acc)
    warp_sums = tl.sum(tl.reshape(acc, (PAD_THREADS // 32, 32)), axis=1)
    inv = tl.rsqrt(tl.sum(warp_sums, axis=0) / K + EPS)
    # FlashInfer RMSNorm rounds once after multiplying the FP32 weight.
    return (x * inv) * w


@triton.jit(do_not_specialize=["M"])
def _rmsnorm_fp8(
    X,
    W,
    Y,
    S,
    M,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    v = rms_values(X, W, token, STRIDE, K, EPS, BLOCK)
    store_group128(v, Y, S, token, M, K, BLOCK)


@triton.jit(do_not_specialize=["M"])
def _rmsnorm_bf16_fp8(
    X,
    W,
    Y,
    S,
    B,
    M,
    STRIDE: tl.constexpr,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    v = rms_values(X, W, token, STRIDE, K, EPS, BLOCK)
    cols = tl.arange(0, BLOCK)
    tl.store(B + token * K + cols, v.to(tl.bfloat16), cols < K)
    store_group128(v, Y, S, token, M, K, BLOCK)


def rmsnorm_fp8(x, weight, eps, *, retain_bf16=False):
    if x.ndim != 2 or x.dtype != torch.bfloat16 or not x.is_cuda or x.stride(1) != 1:
        raise ValueError("FP8 RMSNorm requires row-strided CUDA BF16 [M,K]")
    m, k = x.shape
    out = allocate_quantized(m, k, x.device, retain_bf16=retain_bf16)
    if m:
        args = (x, weight, out.values, out.scale_wire)
        kernel = _rmsnorm_bf16_fp8 if retain_bf16 else _rmsnorm_fp8
        if retain_bf16:
            args += (out.bf16,)
        kernel[(m,)](*args, m, x.stride(0), k, eps, max(512, triton.next_power_of_2(k)))
    return out


@triton.jit(do_not_specialize=["M"])
def _sigmoid_gate_fp8(
    X,
    G,
    Y,
    S,
    M,
    K: tl.constexpr,
    XS: tl.constexpr,
    GS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + token * XS + cols, cols < K, other=0.0).to(tl.float32)
    gate = tl.load(G + token * GS + cols, cols < K, other=0.0).to(tl.float32)
    # torch.sigmoid(BF16) rounds before the separate multiply.
    v = x * tl.sigmoid(gate).to(tl.bfloat16).to(tl.float32)
    store_group128(v, Y, S, token, M, K, BLOCK)


def sigmoid_gate_fp8(x, gate):
    gate = gate.reshape(x.shape)
    if x.ndim != 2 or x.dtype != torch.bfloat16 or gate.dtype != x.dtype:
        raise ValueError("gate FP8 requires BF16 [M,K] inputs")
    m, k = x.shape
    out = allocate_quantized(m, k, x.device)
    if m:
        _sigmoid_gate_fp8[(m,)](
            x,
            gate,
            out.values,
            out.scale_wire,
            m,
            k,
            x.stride(0),
            gate.stride(0),
            max(512, triton.next_power_of_2(k)),
        )
    return out


@triton.jit(do_not_specialize=["M", "SEQ", "XB", "GB"])
def _kda_output_prefill_fp8(
    X,
    G,
    W,
    Y,
    S,
    M,
    K: tl.constexpr,
    SEQ,
    XB,
    XT: tl.constexpr,
    XH: tl.constexpr,
    GB,
    GT: tl.constexpr,
    GH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    heads, dims = cols // 128, cols % 128
    xb = tl.load(
        X + token // SEQ * XB + token % SEQ * XT + heads * XH + dims,
        cols < K,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        G + token // SEQ * GB + token % SEQ * GT + heads * GH + dims,
        cols < K,
        other=0.0,
    ).to(tl.float32)
    gamma = tl.load(W + dims).to(tl.float32)
    grouped = tl.reshape(xb, (BLOCK // 128, 128))
    variance = tl.sum(grouped * grouped, axis=1) / 128.0
    inverse = 1.0 / tl.sqrt(variance + EPS)
    norm = tl.reshape(grouped * inverse[:, None], (BLOCK,))
    v = norm * gamma * tl.sigmoid(gate)
    store_group128(v, Y, S, token, M, K, BLOCK)


@triton.jit(do_not_specialize=["M", "SEQ", "XB", "GB"])
def _kda_output_decode_fp8(
    X,
    G,
    W,
    Y,
    S,
    M,
    K: tl.constexpr,
    SEQ,
    XB,
    XT: tl.constexpr,
    XH: tl.constexpr,
    GB,
    GT: tl.constexpr,
    GH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    heads, dims = cols // 128, cols % 128
    xb = tl.load(
        X + token // SEQ * XB + token % SEQ * XT + heads * XH + dims,
        cols < K,
        other=0.0,
    ).to(tl.float32)
    gate = tl.load(
        G + token // SEQ * GB + token % SEQ * GT + heads * GH + dims,
        cols < K,
        other=0.0,
    ).to(tl.float32)
    gamma = tl.load(W + dims).to(tl.float32)
    grouped = tl.reshape(xb, (BLOCK // 128, 128))
    # Torch mean over 128 contiguous FP32 squares uses vec4 per thread.
    # Preserve its sequential four-value sum before the warp reduction.
    even, odd = tl.split(tl.reshape(grouped * grouped, (BLOCK // 128, 32, 2, 2)))
    v0, v2 = tl.split(even)
    v1, v3 = tl.split(odd)
    thread_sum = ((v0 + v1) + v2) + v3
    lanes = tl.arange(0, 32)
    # Explicit shfl-down tree: tl.sum may choose a different tree when the
    # scale-packing consumer changes the inferred tensor layout.
    for step in tl.static_range(5):
        source_lane = tl.minimum(lanes + (16 >> step), 31)
        other = tl.gather(
            thread_sum,
            tl.broadcast_to(source_lane[None, :], (BLOCK // 128, 32)),
            axis=1,
        )
        thread_sum = thread_sum + other
    variance = tl.sum(tl.where(lanes[None, :] == 0, thread_sum, 0.0), axis=1) / 128.0
    inverse = tl.rsqrt(variance + EPS)
    norm = tl.reshape(grouped * inverse[:, None], (BLOCK,))
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    v = norm * gamma * sigmoid
    store_group128(v, Y, S, token, M, K, BLOCK)


def kda_output_fp8(x, gate, weight, eps, *, mode):
    from .rms_norm_gate import _bthd_strides

    if x.shape != gate.shape or x.shape[-1] != 128 or x.dtype != torch.bfloat16:
        raise ValueError(
            "K3 FP8 output norm requires matching BF16 tensors with head_dim=128"
        )
    if x.ndim not in (3, 4):
        raise ValueError("K3 output norm requires [T,H,D] or [B,T,H,D]")
    m, k = x.numel() // (x.shape[-2] * 128), x.shape[-2] * 128
    out = allocate_quantized(m, k, x.device)
    if m:
        xs, gs = _bthd_strides(x), _bthd_strides(gate)
        kernel = _kda_output_decode_fp8 if mode == "decode" else _kda_output_prefill_fp8
        kernel[(m,)](
            x,
            gate,
            weight,
            out.values,
            out.scale_wire,
            m,
            k,
            x.shape[-3],
            *xs[:3],
            *gs[:3],
            eps,
            max(512, triton.next_power_of_2(k)),
            enable_fp_fusion=False
        )
    return out
