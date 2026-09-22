"""Stride-aware BF16 latent norm, KDA norm/gate, and MLA gate producers."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from rtp_llm.models_py.triton_kernels.common.cached_launch import CachedLaunch


def _bf16_cuda_pair(x, other):
    return (
        x.is_cuda
        and other.is_cuda
        and x.device == other.device
        and x.dtype == other.dtype == torch.bfloat16
        and x.stride(-1) == other.stride(-1) == 1
    )


def supports_kda_prefill_norm_gate(x, gate, weight):
    return (
        x.ndim in (3, 4)
        and (x.ndim == 3 or x.shape[0] == 1)
        and x.shape == gate.shape
        and x.shape[-1] == 128
        and x.shape[-2] > 0
        and _bf16_cuda_pair(x, gate)
        and weight.shape == (128,)
        and weight.device == x.device
        and weight.dtype in (torch.bfloat16, torch.float32)
        and weight.stride(0) == 1
    )


def supports_sigmoid_gate(x, gate):
    return (
        x.ndim in (2, 3)
        and gate.ndim in (2, 3)
        and x.shape[0] == gate.shape[0]
        and (x.ndim == 2 or x.shape[-1] == 128)
        and (gate.ndim == 2 or gate.shape[-1] == 128)
        and x.shape[-1] > 0
        and (x.ndim == 2 or x.shape[1] > 0)
        and x.numel() == gate.numel()
        and (x.shape[-1] % 128 == 0)
        and _bf16_cuda_pair(x, gate)
    )


@triton.jit
def _kda_norm_gate(
    X,
    G,
    W,
    Y,
    K: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    GT: tl.constexpr,
    GH: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A valid strided view can exceed int32 offsets even with very few tokens.
    # Widen before multiplication, including the dense output's token offset.
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    heads, dims = (cols // 128).to(tl.int64), cols % 128
    x = tl.load(X + token * XT + heads * XH + dims, cols < K, other=0).to(tl.float32)
    gate = tl.load(G + token * GT + heads * GH + dims, cols < K, other=0).to(tl.float32)
    gamma = tl.load(W + dims).to(tl.float32)
    grouped = tl.reshape(x, (BLOCK // 128, 128))
    # Match the eager Torch decode reduction: four sequential values per lane,
    # then a shfl-down reduction, as in _kda_output_decode_fp8.
    even, odd = tl.split(tl.reshape(grouped * grouped, (BLOCK // 128, 32, 2, 2)))
    v0, v2 = tl.split(even)
    v1, v3 = tl.split(odd)
    thread_sum = ((v0 + v1) + v2) + v3
    lanes = tl.arange(0, 32)
    for step in tl.static_range(5):
        source_lane = tl.minimum(lanes + (16 >> step), 31)
        other = tl.gather(
            thread_sum,
            tl.broadcast_to(source_lane[None, :], (BLOCK // 128, 32)),
            axis=1,
        )
        thread_sum = thread_sum + other
    variance = tl.sum(tl.where(lanes[None, :] == 0, thread_sum, 0.0), axis=1) / 128.0
    norm = tl.reshape(grouped * tl.rsqrt(variance + EPS)[:, None], (BLOCK,))
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    tl.store(Y + token * K + cols, norm * gamma * sigmoid, cols < K)


_launch_kda_norm_gate = CachedLaunch(_kda_norm_gate, enable_fp_fusion=False)


def kda_norm_gate(x, gate, weight, eps):
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    rows, width = x.shape[-3], x.shape[-2] * 128
    if rows:
        _launch_kda_norm_gate(
            (rows, 1, 1),
            (x, gate, weight, output),
            (
                width,
                x.stride(-3),
                x.stride(-2),
                gate.stride(-3),
                gate.stride(-2),
                eps,
                max(512, 1 << (width - 1).bit_length()),
            ),
        )
    return output


@triton.jit
def _latent_rmsnorm(
    X,
    W,
    Y,
    N: tl.constexpr,
    STRIDE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Prefill can exceed 2**31 elements even when each stride fits in int32.
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * STRIDE + cols, cols < N, other=0).to(tl.float32)
    gamma = tl.load(W + cols, cols < N, other=0).to(tl.float32)
    inverse = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)
    tl.store(Y + row * N + cols, x * inverse * gamma, cols < N)


_launch_latent_rmsnorm = CachedLaunch(_latent_rmsnorm, enable_fp_fusion=False)


def latent_rmsnorm(x, weight, eps):
    output = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.shape[0]:
        _launch_latent_rmsnorm(
            (x.shape[0], 1, 1),
            (x, weight, output),
            (x.shape[1], x.stride(0), eps, 1 << (x.shape[1] - 1).bit_length()),
        )
    return output


@triton.jit
def _sigmoid_gate(
    X,
    G,
    Y,
    K: tl.constexpr,
    XT: tl.constexpr,
    XH: tl.constexpr,
    GT: tl.constexpr,
    GH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    heads, dims = (cols // 128).to(tl.int64), cols % 128
    x = tl.load(X + token * XT + heads * XH + dims, cols < K, other=0).to(tl.float32)
    gate = tl.load(G + token * GT + heads * GH + dims, cols < K, other=0).to(tl.float32)
    # Torch's BF16 sigmoid materializes before multiplication. Preserve it.
    sigmoid = tl.div_rn(1.0, 1.0 + libdevice.exp(-gate)).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + token * K + cols, x * sigmoid, cols < K)


_launch_sigmoid_gate = CachedLaunch(_sigmoid_gate, enable_fp_fusion=False)


def sigmoid_gate(x, gate):
    rows = x.shape[0]
    width = x.shape[1] if x.ndim == 2 else x.shape[1] * 128
    output = torch.empty((rows, width), dtype=x.dtype, device=x.device)
    if rows:
        _launch_sigmoid_gate(
            (rows, (width + 1023) // 1024, 1),
            (x, gate, output),
            (
                width,
                x.stride(0),
                128 if x.ndim == 2 else x.stride(1),
                gate.stride(0),
                128 if gate.ndim == 2 else gate.stride(1),
                1024,
            ),
        )
    return output
