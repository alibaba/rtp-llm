# SPDX-License-Identifier: Apache-2.0
"""Fixed-tile projections for DeepSeek checkpoint weights."""

import torch
import triton
import triton.language as tl


@triton.jit
def _load_e2m1(weight, k, n, stride_n, stride_k):
    """Decode packed K-axis E2M1 nibbles into one [K, N] weight tile."""
    byte = tl.load(weight + k[:, None] // 2 * stride_k + n[None, :] * stride_n)
    code = (byte.to(tl.int32) >> (4 * (k[:, None] % 2))) & 15
    mag = code & 7
    value = tl.where(mag < 4, mag * 0.5, (2 + (mag & 1)) << (mag >= 6).to(tl.int32))
    return tl.where(code < 8, value, -value).to(tl.float16)


@triton.jit
def _linear_kernel(
    x,
    weight,
    bias,
    out,
    sx,
    sw,
    rows,
    N: tl.constexpr,
    K: tl.constexpr,
    WN: tl.constexpr,
    WK: tl.constexpr,
    s_m,
    s_k,
    w_n: tl.constexpr,
    w_k: tl.constexpr,
    PACKED: tl.constexpr,
    SCALE_N: tl.constexpr,
    MXFP4_GROUP: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    n = tl.program_id(1).to(tl.int64) * BN + tl.arange(0, BN)
    acc = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        k = tile * BK + tl.arange(0, BK)
        a = tl.load(
            x + m[:, None] * K + k[None, :],
            (m[:, None] < rows) & (k[None, :] < K),
            other=0.0,
        )
        if MXFP4_GROUP:
            b = _load_e2m1(weight, k, n % N, WN, WK)
            group = tile * BK // MXFP4_GROUP
            packed = tl.load(sx + m * s_m + group // 4 * s_k, m < rows, other=0)
            a_bits = ((packed >> (8 * (group % 4))) & 255) << 23
            b_bits = tl.load(sw + (n % N) * w_n + tile * w_k).to(tl.int32) << 23
            a_scale = a_bits.to(tl.float32, bitcast=True)
            b_scale = b_bits.to(tl.float32, bitcast=True)
            acc += tl.dot(a.to(tl.float16), b) * (a_scale[:, None] * b_scale[None, :])
        else:
            b = tl.load(
                weight + k[:, None] * WK + n[None, :] * WN,
                (k[:, None] < K) & (n[None, :] < N),
                other=0.0,
            )
        if sx is not None and not MXFP4_GROUP:
            col = tile // 4 if PACKED else tile
            a_scale = tl.load(sx + m * s_m + col * s_k, m < rows, other=0)
            b_scale = tl.load(sw + n // SCALE_N * w_n + col * w_k, n < N, other=0)
            if PACKED:
                shift = (tile % 4) * 8
                a_scale = (((a_scale >> shift) & 255) << 23).to(
                    tl.float32, bitcast=True
                )
                b_scale = (((b_scale >> shift) & 255) << 23).to(
                    tl.float32, bitcast=True
                )
            acc += tl.dot(a.to(tl.float16), b.to(tl.float16)) * (
                a_scale[:, None] * b_scale[None, :]
            )
        elif sx is None:
            if a.dtype != b.dtype:
                a = a.to(tl.float32)
                b = b.to(tl.float32)
            acc = tl.dot(a, b, acc, input_precision="ieee")
    if bias is not None:
        acc += tl.load(bias + n, n < N, other=0)[None, :]
    tl.store(
        out + m[:, None] * N + n[None, :], acc, (m[:, None] < rows) & (n[None, :] < N)
    )


def _launch_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    x_scales: torch.Tensor | None = None,
    weight_scales: torch.Tensor | None = None,
    *,
    group_size: int = 128,
    scale_group_n: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    n, k = weight.shape
    mxfp4 = weight.dtype == torch.int8
    if mxfp4:
        k *= 2
    if k <= 0 or x.shape[-1] != k or not x.is_cuda or weight.device != x.device:
        raise ValueError(
            "DeepSeek projection requires matching CUDA operands and input width"
        )
    if bias is not None:
        if bias.numel() != n or bias.device != x.device:
            raise ValueError(
                "DeepSeek projection bias must match the output width and device"
            )
        bias = bias.reshape(n).contiguous()
    rows = x.numel() // k
    packed = False
    block_k = 32
    s_m = s_k = w_n = w_k = 0
    if mxfp4:
        if (
            x.dtype != torch.float8_e4m3fn
            or group_size not in (32, 128)
            or k % group_size
            or x_scales is None
            or x_scales.dtype != torch.int32
            or x_scales.shape != (rows, triton.cdiv(k // group_size, 4))
            or weight_scales is None
            or weight_scales.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or weight_scales.shape != (n, k // 32)
            or x_scales.device != x.device
            or weight_scales.device != x.device
        ):
            raise ValueError(
                "MXFP4 projection requires E4M3 rows and group-32 UE8M0 weight scales"
            )
        weight_scales = weight_scales.view(torch.uint8)
        s_m, s_k = x_scales.stride()
        w_n, w_k = weight_scales.stride()
    elif x_scales is not None or weight_scales is not None:
        if x_scales is None or weight_scales is None:
            raise ValueError(
                "FP8 projection requires both activation and weight scales"
            )
        packed = weight_scales.dtype == torch.int32
        scale_group_n = scale_group_n or (1 if packed else 128)
        block_k = group_size
        if group_size not in (32, 128) or scale_group_n <= 0 or k % group_size:
            raise ValueError("unsupported FP8 block dimensions")
        columns = triton.cdiv(k // group_size, 4 if packed else 1)
        if (
            x.dtype != torch.float8_e4m3fn
            or weight.dtype != x.dtype
            or weight_scales.dtype not in (torch.float32, torch.int32)
            or x_scales.dtype != weight_scales.dtype
            or x_scales.device != x.device
            or weight_scales.device != x.device
            or x_scales.shape != (rows, columns)
            or weight_scales.ndim != 2
            or weight_scales.size(0) < triton.cdiv(n, scale_group_n)
            or weight_scales.size(1) != columns
        ):
            raise ValueError(
                "FP8 projection scales do not match the checkpoint block layout"
            )
        s_m, s_k = x_scales.stride()
        w_n, w_k = weight_scales.stride()
    shape = (*x.shape[:-1], n)
    if out is not None and (
        out.shape != shape or out.device != x.device or out.dtype != out_dtype
    ):
        raise ValueError(
            "projection output buffer must match the result shape, dtype and device"
        )
    output = (
        out
        if out is not None and out.is_contiguous()
        else torch.empty(shape, device=x.device, dtype=out_dtype)
    )
    if rows and n:
        with torch.cuda.device(x.device):
            _linear_kernel[(triton.cdiv(rows, 16), triton.cdiv(n, 64))](
                x.contiguous(),
                weight,
                bias,
                output,
                x_scales,
                weight_scales,
                rows,
                n,
                k,
                *weight.stride(),
                s_m,
                s_k,
                w_n,
                w_k,
                packed,
                scale_group_n or 1,
                group_size if mxfp4 else 0,
                16,
                64,
                block_k,
                num_warps=4,
            )
    return output if out is None or output is out else out.copy_(output)


def linear(x, weight, bias=None, *, out_dtype=None):
    """Project rows in fixed K order, promoting mixed operands to FP32."""
    supported = (torch.bfloat16, torch.float16, torch.float32)
    if x.dtype not in supported or weight.dtype not in supported:
        raise ValueError("DeepSeek dense projections require BF16, FP16 or FP32")
    return _launch_linear(x, weight, bias, out_dtype or x.dtype)


def quantize_fp8_blocks(rows, packed, group_size=128, *, eps=1e-4):
    """Quantize one token and K group per scale at every batch size."""
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )
    from rtp_llm.ops.compute_ops import per_token_group_quant_fp8_v2

    if rows.ndim != 2 or group_size not in (32, 128) or rows.size(1) % group_size:
        raise ValueError(
            "FP8 activation rows must align to their checkpoint group size"
        )
    rows = rows.contiguous()
    quantized = torch.empty_like(rows, dtype=torch.float8_e4m3fn)
    scales = create_per_token_group_quant_fp8_output_scale(
        rows.shape,
        rows.device,
        group_size,
        column_major_scales=packed,
        scale_tma_aligned=packed,
        scale_ue8m0=packed,
    )
    if rows.size(0):
        per_token_group_quant_fp8_v2(
            rows, quantized, scales, group_size, eps, -448.0, 448.0, packed, False, None
        )
    return quantized, scales


def mxfp4_linear(x, weight, weight_scales, *, group_size=32, out_dtype=None):
    """Apply one native packed E2M1 weight with its checkpoint group-32 scales."""
    if weight.dtype != torch.int8:
        raise ValueError("an MXFP4 weight must store two E2M1 values per int8")
    quantized, scales = quantize_fp8_blocks(
        x.reshape(-1, x.size(-1)),
        True,
        group_size,
        eps=torch.finfo(torch.float32).tiny if group_size == 32 else 1e-4,
    )
    return _launch_linear(
        quantized.view(x.shape),
        weight,
        None,
        out_dtype or x.dtype,
        scales,
        weight_scales,
        group_size=group_size,
    )


def pack_checkpoint_scales(weight, scales, block):
    """Pack four UE8M0 exponents and broadcast each checkpoint N block."""
    n, k = weight.shape
    if scales.dtype != torch.float8_e8m0fnu or scales.shape != (
        triton.cdiv(n, block),
        triton.cdiv(k, block),
    ):
        raise ValueError("UE8M0 scale shape must match the FP8 weight block layout")
    exponents = scales.view(torch.uint8)
    padded = torch.nn.functional.pad(exponents, (0, -scales.size(1) % 4))
    rows = torch.arange(n, device=scales.device) // block
    return padded.contiguous().view(torch.int32).index_select(0, rows)


class BlockFp8Linear(torch.nn.Module):
    """DeepSeek 128-wide checkpoint projection with reusable quantized input."""

    supports_fused_silu_quant = False

    def __init__(self, weight, scales):
        super().__init__()
        if scales.dtype == torch.float8_e8m0fnu:
            scales = pack_checkpoint_scales(weight, scales, 128)
        self.weight = weight
        self.weight_scales = scales
        self.N, self.K = weight.shape
        self.scale_ue8m0 = scales.dtype == torch.int32

    def quantize_input(self, x):
        return quantize_fp8_blocks(x.reshape(-1, self.K), self.scale_ue8m0)

    def forward_quantized(self, x, scales, out=None):
        return _launch_linear(
            x, self.weight, None, torch.bfloat16, scales, self.weight_scales, out=out
        )

    def forward(self, x, out=None):
        quantized, scales = self.quantize_input(x)
        return self.forward_quantized(quantized.view(x.shape), scales, out)


@triton.jit
def _rms_kernel(
    x, weight, out, N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr
):
    start = tl.program_id(0).to(tl.int64) * N
    k = tl.arange(0, BLOCK)
    value = tl.load(x + start + k, k < N, other=0).to(tl.float32)
    value *= tl.rsqrt(tl.sum(value * value, axis=0) / N + EPS)
    if weight is not None:
        value *= tl.load(weight + k, k < N, other=0).to(tl.float32)
    tl.store(out + start + k, value, k < N)


def rms_norm(x, weight=None, eps=1e-6):
    x = x.contiguous()
    if weight is not None:
        if weight.numel() != x.shape[-1] or weight.device != x.device:
            raise ValueError("RMSNorm weight must match the input width and device")
        weight = weight.contiguous()
    out = torch.empty_like(x)
    if x.numel():
        size = x.shape[-1]
        with torch.cuda.device(x.device):
            _rms_kernel[(x.numel() // size,)](
                x, weight, out, size, eps, triton.next_power_of_2(size)
            )
    return out
