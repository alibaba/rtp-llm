"""Exact native dynamic FP8 quantization for measured small SM120 projections."""

import os

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _small_fp8_quant(
    x, q, scale, N: tl.constexpr, OUT: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < N, other=0).to(tl.float32)
    absolute = tl.abs(values)
    # CUDA fmaxf ignores NaNs during the native amax reduction.
    maximum = tl.max(tl.where(absolute == absolute, absolute, 0), 0)
    output_scale = maximum * (1.0 / 448.0)
    # The native CUDA build flushes subnormal FP32 scales to zero. Preserve
    # the stored scale itself, even when both paths quantize the values to zero.
    output_scale = tl.where(output_scale < 1.1754943508222875e-38, 0, output_scale)
    if tl.program_id(0) == 0:
        tl.store(scale, output_scale)
    inverse = tl.div_rn(1.0, tl.maximum(output_scale, 1.0e-9))

    # Replicate the small reduction across output tiles. This parallelizes the
    # padded GEMM rows without a second launch or an inter-CTA dependency.
    rows = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    values = tl.load(x + rows, rows < N, other=0).to(tl.float32) * inverse
    # Native fmin(value, 448) maps NaNs, including Inf * 0, to positive 448.
    values = tl.where(values == values, values, 448.0)
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    tl.store(q + rows, tl.where(rows < N, values, 0).to(tl.float8e4nv), rows < OUT)


def is_supported(x, output=None):
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
        and 1 <= x.shape[0] <= 32
        and (
            x.shape[1] in (320, 2560, 10240)
            or (
                x.shape[1] == 640
                and x.shape[0] in (4, 8, 12, 16)
                and os.environ.get("QWEN4_FP8_SHARED_DOWN_QUANT", "0") == "1"
            )
        )
        and x.numel() <= 65536
        and torch.cuda.get_device_capability(x.device) == (12, 0)
    ):
        return False
    return output is None or (
        isinstance(output, torch.Tensor)
        and output.device == x.device
        and output.dtype == torch.float8_e4m3fn
        and output.is_contiguous()
        and output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
        and tuple(output.shape)
        in (tuple(x.shape), (tr.cdiv(x.shape[0], 16) * 16, x.shape[1]))
    )


def try_small_fp8_quant(x, *, output=None):
    if not is_supported(x, output):
        return None
    if output is None:
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((1,), device=x.device, dtype=torch.float32)
    block = tr.next_power_of_2(x.numel())
    _small_fp8_quant[(tr.cdiv(output.numel(), 4096),)](
        x,
        output,
        scale,
        x.numel(),
        output.numel(),
        block,
        num_warps=4 if block <= 8192 else 16,
        enable_fp_fusion=False,
    )
    return output, scale
