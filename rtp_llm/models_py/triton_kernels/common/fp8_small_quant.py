"""Exact native dynamic FP8 quantization for qualified SM90/SM120 projections."""

import os

import torch
import triton as tr
import triton.language as tl

_SMALL_QUANT_MAX_ELEMENTS = 65536


@tr.jit
def _fp8_absmax(values):
    absolute = tl.abs(values)
    # CUDA fmaxf ignores NaNs during the native amax reduction.
    return tl.max(tl.where(absolute == absolute, absolute, 0), 0)


@tr.jit
def _quantize_projection_tile(x, q, scale, maximum, N: tl.constexpr, OUT: tl.constexpr):
    # The native per-tensor absmax divides by FP8_E4M3_MAX directly; a
    # multiply by the rounded reciprocal diverges by 1 ULP on some inputs.
    output_scale = tl.div_rn(maximum, 448.0)
    # The native CUDA build flushes subnormal FP32 scales to zero. Preserve
    # the stored scale itself, even when both paths quantize the values to zero.
    output_scale = tl.where(output_scale < 1.1754943508222875e-38, 0, output_scale)
    if tl.program_id(0) == 0:
        tl.store(scale, output_scale)
    inverse = tl.div_rn(1.0, tl.maximum(output_scale, 1.0e-9))

    offsets = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    values = tl.load(x + offsets, offsets < N, other=0).to(tl.float32) * inverse
    # Native fmin(value, 448) maps NaNs, including Inf * 0, to positive 448.
    values = tl.where(values == values, values, 448.0)
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    tl.store(
        q + offsets, tl.where(offsets < N, values, 0).to(tl.float8e4nv), offsets < OUT
    )


@tr.jit
def _small_fp8_quant(
    x, q, scale, N: tl.constexpr, OUT: tl.constexpr, BLOCK: tl.constexpr
):
    # Replicate the small reduction across output tiles to avoid a second launch.
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < N, other=0).to(tl.float32)
    _quantize_projection_tile(x, q, scale, _fp8_absmax(values), N, OUT)


@tr.jit
def _fp8_partial_amax(x, partial, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(x + offsets, offsets < N, other=0).to(tl.float32)
    # Every invocation overwrites the workspace; no atomic/reset is needed.
    tl.store(partial + tl.program_id(0), _fp8_absmax(values))


@tr.jit
def _fp8_finish_quant(
    x,
    partial,
    q,
    scale,
    N: tl.constexpr,
    OUT: tl.constexpr,
    PARTS: tl.constexpr,
    REDUCE: tl.constexpr,
):
    ids = tl.arange(0, REDUCE)
    maxima = tl.load(partial + ids, ids < PARTS, other=0)
    _quantize_projection_tile(x, q, scale, tl.max(maxima, 0), N, OUT)


@tr.jit
def _silu_mul_fp8_quant(
    gate_up,
    q,
    scale,
    N: tl.constexpr,
    D: tl.constexpr,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    row = offsets // D
    col = offsets % D
    gate = tl.load(gate_up + row * (2 * D) + col, offsets < N, other=0).to(tl.float32)
    up = tl.load(gate_up + row * (2 * D) + D + col, offsets < N, other=0).to(tl.float32)
    activated = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
    absolute = tl.abs(activated)
    maximum = tl.max(tl.where(absolute == absolute, absolute, 0), 0)
    output_scale = tl.div_rn(maximum, 448.0)
    output_scale = tl.where(output_scale < 1.1754943508222875e-38, 0, output_scale)
    if tl.program_id(0) == 0:
        tl.store(scale, output_scale)
    inverse = tl.div_rn(1.0, tl.maximum(output_scale, 1.0e-9))

    positions = tl.program_id(0) * 4096 + tl.arange(0, 4096)
    row = positions // D
    col = positions % D
    gate = tl.load(gate_up + row * (2 * D) + col, positions < N, other=0).to(tl.float32)
    up = tl.load(gate_up + row * (2 * D) + D + col, positions < N, other=0).to(
        tl.float32
    )
    activated = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
    values = activated * inverse
    values = tl.where(values == values, values, 448.0)
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    tl.store(
        q + positions,
        tl.where(positions < N, values, 0).to(tl.float8e4nv),
        positions < OUT,
    )


# One CTA retains the activated values through amax instead of recomputing
# SiLU per output tile. Only the gated SM90 shared FFN uses this path.
@tr.jit
def _silu_mul_fp8_quant_shared(
    gate_up,
    q,
    scale_output,
    N: tl.constexpr,
    D: tl.constexpr,
    OUT: tl.constexpr,
    BLOCK: tl.constexpr,
    PAD: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK)
    row = offsets // D
    col = offsets % D
    gate = tl.load(gate_up + row * 2 * D + col, offsets < N, other=0).to(tl.float32)
    up = tl.load(gate_up + row * 2 * D + D + col, offsets < N, other=0).to(tl.float32)
    activated = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
    absolute = tl.abs(activated)
    maximum = tl.max(tl.where(absolute == absolute, absolute, 0), 0)
    scale = tl.div_rn(maximum, 448.0)
    scale = tl.where(scale < 1.1754943508222875e-38, 0, scale)
    tl.store(scale_output, scale)
    inverse = tl.div_rn(1.0, tl.maximum(scale, 1.0e-9))
    values = activated * inverse
    values = tl.where(values == values, values, 448.0)
    values = tl.maximum(-448.0, tl.minimum(448.0, values))
    tl.store(q + offsets, values.to(tl.float8e4nv), offsets < N)
    if OUT > N:
        padding_positions = tl.arange(0, PAD)
        tl.store(
            q + padding_positions,
            tl.full((PAD,), 0, tl.float32).to(tl.float8e4nv),
            (padding_positions >= N) & (padding_positions < OUT),
        )


# B64 is too large for the existing single-CTA SiLU/amax fusion.
# Reduce while producing BF16 activation, then reuse the small finalizer.
@tr.jit
def _silu_mul_fp8_partial_amax(
    x, activated, partial, N: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    col = i % D
    gate = tl.load(x + row * 2 * D + col, i < N, other=0).to(tl.float32)
    value = tl.load(x + row * 2 * D + D + col, i < N, other=0).to(tl.float32)
    a = ((gate * tl.sigmoid(gate)) * value).to(tl.bfloat16).to(tl.float32)
    absolute = tl.abs(a)
    maximum = tl.max(tl.where(absolute == absolute, absolute, 0), 0)
    tl.store(activated + i, a, i < N)
    tl.store(partial + tl.program_id(0), maximum)


def try_silu_mul_fp8_quant(gate_up, *, output=None):
    """Fuse BF16 SiLU×Mul and dynamic FP8 quant for the small shared FFN."""
    if not (
        isinstance(gate_up, torch.Tensor)
        and gate_up.is_cuda
        and torch.version.hip is None
        and gate_up.dtype == torch.bfloat16
        and gate_up.ndim == 2
        and gate_up.is_contiguous()
        and (
            (gate_up.shape[0] in (4, 8, 12, 16) and gate_up.shape[1] == 1280)
            or (
                tuple(gate_up.shape) == (64, 640)
                and torch.cuda.is_current_stream_capturing()
                and torch.cuda.get_device_capability(gate_up.device) == (9, 0)
            )
            or (
                os.environ.get("QWEN_MOE_FP8_SMALL_ROWS_FUSION", "0") == "1"
                and 1 <= gate_up.shape[0] <= 32
                and gate_up.shape[1] == 640
                and torch.cuda.get_device_capability(gate_up.device) == (9, 0)
            )
        )
        and torch.cuda.get_device_capability(gate_up.device) in ((9, 0), (12, 0))
    ):
        return None
    rows, width = gate_up.shape[0], gate_up.shape[1] // 2
    if output is not None and not (
        isinstance(output, torch.Tensor)
        and output.device == gate_up.device
        and output.dtype == torch.float8_e4m3fn
        and output.is_contiguous()
        and output.untyped_storage().data_ptr() != gate_up.untyped_storage().data_ptr()
        and tuple(output.shape) in ((rows, width), (tr.cdiv(rows, 16) * 16, width))
    ):
        return None
    if output is None:
        output = torch.empty(
            (rows, width), device=gate_up.device, dtype=torch.float8_e4m3fn
        )
    scale = torch.empty((1,), device=gate_up.device, dtype=torch.float32)
    block = tr.next_power_of_2(rows * width)
    if rows == 64 and width == 320:
        activated = torch.empty(
            (rows, width), device=gate_up.device, dtype=torch.bfloat16
        )
        parts = tr.cdiv(rows * width, 128)
        partial = torch.empty((parts,), device=gate_up.device, dtype=torch.float32)
        _silu_mul_fp8_partial_amax[(parts,)](
            gate_up,
            activated,
            partial,
            rows * width,
            width,
            128,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _fp8_finish_quant[(tr.cdiv(output.numel(), 4096),)](
            activated,
            partial,
            output,
            scale,
            rows * width,
            output.numel(),
            parts,
            tr.next_power_of_2(parts),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output, scale
    if width == 320:
        warps = 4 if rows <= 2 else 8 if rows <= 8 or rows > 16 else 32
        _silu_mul_fp8_quant_shared[(1,)](
            gate_up,
            output,
            scale,
            rows * width,
            width,
            output.numel(),
            block,
            tr.next_power_of_2(output.numel()),
            num_warps=warps,
            enable_fp_fusion=False,
        )
        return output, scale
    _silu_mul_fp8_quant[(tr.cdiv(output.numel(), 4096),)](
        gate_up,
        output,
        scale,
        rows * width,
        width,
        output.numel(),
        block,
        num_warps=4 if block <= 8192 else 16,
        enable_fp_fusion=False,
    )
    return output, scale


def is_supported(x, output=None):
    if not (
        isinstance(x, torch.Tensor)
        and x.is_cuda
        and torch.version.hip is None
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.is_contiguous()
    ):
        return False
    rows, width = x.shape
    small = (
        1 <= rows <= 32
        and (width in (320, 2560, 10240) or (width == 640 and rows in (4, 8, 12, 16)))
        and x.numel() <= _SMALL_QUANT_MAX_ELEMENTS
    )
    # These shapes save GPU time on H20, but the extra dispatch costs more in eager.
    if not small and not (
        (
            (width == 320 and rows == 64)
            or (width == 640 and rows in (32, 64))
            or (width == 2560 and rows in (32, 64, 256, 1024))
            or (width == 10240 and rows in (16, 32, 64))
        )
        and torch.cuda.is_current_stream_capturing()
    ):
        return False
    capability = torch.cuda.get_device_capability(x.device)
    if not (capability == (9, 0) or (small and capability == (12, 0))):
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
    numel = x.numel()
    output_numel = output.numel()
    grid = (tr.cdiv(output_numel, 4096),)
    if numel > _SMALL_QUANT_MAX_ELEMENTS:
        parts = tr.cdiv(numel, 4096)
        partial = torch.empty((parts,), device=x.device, dtype=torch.float32)
        _fp8_partial_amax[(parts,)](
            x,
            partial,
            numel,
            4096,
            num_warps=4,
        )
        _fp8_finish_quant[grid](
            x,
            partial,
            output,
            scale,
            numel,
            output_numel,
            parts,
            tr.next_power_of_2(parts),
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output, scale
    block = tr.next_power_of_2(numel)
    _small_fp8_quant[grid](
        x,
        output,
        scale,
        numel,
        output_numel,
        block,
        num_warps=4 if block <= 8192 else 16,
        enable_fp_fusion=False,
    )
    return output, scale
