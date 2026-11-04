"""Qwen4 Prefill HC up projection and BF16 branch consumer.

The FP8 producer uses the existing per-tensor quantizer and MMA. BF16 weights
and activations stay BF16. Both paths retain the materialized BF16 logits,
sigmoid and product boundaries of the ordinary HC implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _bf16_up_mix(A, W, NORM, Y, M: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    packed = tl.arange(0, 4 * BN)
    c = tl.program_id(1) * BN + packed % BN + (packed // BN) * 2560
    k = tl.arange(0, 64)
    acc = tl.zeros((BM, 4 * BN), tl.float32)
    for block in range(5):
        kk = block * 64 + k
        a = tl.load(A + r[:, None] * 320 + kk[None, :], r[:, None] < M, other=0)
        w = tl.load(W + c[None, :] * 320 + kk[:, None])
        acc = tl.dot(a, w, acc)
    logits = acc.to(tl.bfloat16).to(tl.float32)
    gates = tl.sigmoid(logits).to(tl.bfloat16).to(tl.float32)
    norm = tl.load(NORM + r[:, None] * 10240 + c[None, :], r[:, None] < M, other=0).to(
        tl.float32
    )
    product = (gates * norm).to(tl.bfloat16).to(tl.float32).reshape(BM, 4, BN)
    branches = tl.arange(0, 4)
    output = tl.zeros((BM, BN), tl.float32)
    for branch in tl.static_range(4):
        output += tl.sum(tl.where(branches[None, :, None] == branch, product, 0.0), 1)
    tl.store(Y + r[:, None] * 2560 + n[None, :], output / 4.0, r[:, None] < M)


def supports_hc_prefill(normed, weight, scale):
    return (
        normed.is_cuda
        and torch.version.hip is None
        and torch.cuda.get_device_capability(normed.device) == (12, 0)
        and normed.ndim == 2
        and normed.shape[1] == 10240
        and 4093 <= normed.shape[0] <= 16384
        and weight.shape == (10240, 320)
        and normed.dtype == torch.bfloat16
        and weight.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and weight.device == normed.device
        and all(t.is_contiguous() and not t.requires_grad for t in (weight, normed))
        and (
            weight.dtype == torch.bfloat16
            or (
                isinstance(scale, torch.Tensor)
                and scale.device == normed.device
                and scale.dtype == torch.float32
                and scale.numel() == 1
            )
        )
    )


def supports_hc_up_mix(mix, weight, normed, scale):
    return (
        supports_hc_prefill(normed, weight, scale)
        and mix.ndim == 2
        and mix.shape == (normed.shape[0], 320)
        and mix.dtype == torch.bfloat16
        and mix.device == normed.device
        and mix.is_contiguous()
        and not mix.requires_grad
    )


def hc_up_mix(mix, weight, normed, scale=None):
    if not supports_hc_up_mix(mix, weight, normed, scale):
        raise ValueError("unsupported Qwen4 Prefill HC up/mix geometry")
    rows = mix.shape[0]
    output = mix.new_empty((rows, 2560))
    grid = (triton.cdiv(rows, 32), 40)
    if weight.dtype == torch.bfloat16:
        _bf16_up_mix[grid](
            mix,
            weight,
            normed,
            output,
            M=rows,
            BM=32,
            BN=64,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    else:
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
            scaled_fp8_per_tensor_quant,
        )
        from rtp_llm.models_py.triton_kernels.common.gated_residual_medium_up_mix_triton import (
            _hc_medium_up_mix,
        )

        padding = (-rows) % 16
        padded = (
            torch.empty((rows + padding, 320), device=mix.device, dtype=weight.dtype)
            if padding
            else None
        )
        quantized, input_scale = scaled_fp8_per_tensor_quant(mix, output=padded)
        _hc_medium_up_mix[grid](
            quantized,
            weight,
            input_scale,
            scale,
            normed,
            output,
            M=rows,
            BM=32,
            BN=64,
            BK=64,
            num_warps=4,
            num_stages=2,
            enable_fp_fusion=False,
        )
    return output
