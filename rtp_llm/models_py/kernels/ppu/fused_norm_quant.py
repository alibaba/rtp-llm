"""PPU RMSNorm fusions with BF16 outputs and ordinary FP32 FP8 scales."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _sum32_descending(x, ROWS: tl.constexpr):
    a, b = tl.split(tl.permute(x.reshape((ROWS, 2, 16)), (0, 2, 1)))
    x16 = a + b
    a, b = tl.split(tl.permute(x16.reshape((ROWS, 2, 8)), (0, 2, 1)))
    x8 = a + b
    a, b = tl.split(tl.permute(x8.reshape((ROWS, 2, 4)), (0, 2, 1)))
    x4 = a + b
    a, b = tl.split(tl.permute(x4.reshape((ROWS, 2, 2)), (0, 2, 1)))
    x2 = a + b
    a, b = tl.split(x2)
    return a + b


@triton.jit
def _rmsnorm_quant(
    x_ptr, residual_ptr, weight_ptr, out_ptr, q_ptr, scales_ptr,
    H: tl.constexpr, EPS: tl.constexpr,
    X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr, ADD_RESIDUAL: tl.constexpr, QUANTIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lanes = tl.arange(0, BLOCK // 8)
    vec = tl.arange(0, 8)
    cols_vec = vec[:, None] + lanes[None, :] * 8
    vectors = tl.load(x_ptr + row * X_STRIDE + cols_vec, cols_vec < H, other=0).to(tl.float32)
    if ADD_RESIDUAL:
        residual = tl.load(residual_ptr + row * R_STRIDE + cols_vec, cols_vec < H, other=0).to(tl.float32)
        vectors = vectors + residual
        # The normalization consumes the FP32 sum; only the residual store
        # rounds to BF16, matching fused_add_rmsnorm.
        tl.store(residual_ptr + row * R_STRIDE + cols_vec, vectors, cols_vec < H)
    # FlashInfer accumulates eight adjacent values per lane before its warp
    # reduction. Preserve that association instead of reducing x*x directly.
    lane_sum = tl.full((BLOCK // 8,), 0, tl.float32)
    for j in tl.static_range(8):
        value = tl.gather(vectors, tl.full((1, BLOCK // 8), j, tl.int32), 0).reshape((BLOCK // 8,))
        lane_sum = tl.fma(value, value, lane_sum)
    warp_sum = _sum32_descending(lane_sum.reshape((BLOCK // 256, 32)), BLOCK // 256)
    warps = tl.arange(0, 32)
    warp_padded = tl.where(warps < BLOCK // 256,
                           tl.gather(warp_sum, tl.minimum(warps, BLOCK // 256 - 1), 0), 0)
    sum_sq = _sum32_descending(warp_padded.reshape((1, 32)), 1).reshape(())
    inv_rms = tl.rsqrt(sum_sq / H + EPS)
    cols = tl.arange(0, BLOCK)
    x = tl.trans(vectors).reshape((BLOCK,))
    weight = tl.load(weight_ptr + cols, cols < H, other=0).to(tl.float32)
    normed = (x * inv_rms * weight).to(tl.bfloat16)
    tl.store(out_ptr + row * H + cols, normed, cols < H)
    if QUANTIZE:
        # Keep the BF16 boundary before quantization and the PPU quantizer's
        # direct 448/amax multiplier; CUDA's val/scale path is different.
        values = normed.to(tl.float32).reshape((BLOCK // 128, 128))
        amax = tl.maximum(tl.max(tl.abs(values), 1), 1e-10)
        scale = amax / 448.0
        quantized = tl.clamp(values * (448.0 / amax[:, None]), -448.0, 448.0)
        quantized = quantized.to(q_ptr.dtype.element_ty).reshape((BLOCK,))
        tl.store(q_ptr + row * H + cols, quantized, cols < H)
        groups = tl.arange(0, BLOCK // 128)
        tl.store(scales_ptr + row * (H // 128) + groups, scale, groups < H // 128)


def _run(x: torch.Tensor, weight: torch.Tensor, eps: float,
         residual: Optional[torch.Tensor], quantize: bool):
    if x.ndim != 2:
        raise ValueError("PPU fused RMSNorm expects [M, H]")
    rows, hidden = x.shape
    if hidden <= 0 or weight.shape != (hidden,):
        raise ValueError("PPU fused RMSNorm requires positive H and weight [H]")
    if residual is not None and (residual.shape != x.shape or residual.dtype != x.dtype):
        raise ValueError("residual must match the input shape and dtype")
    # H=512/2048 pass both exact-output and latency gates. H=6144 retains
    # native normalization: its exact fused reduction is slower on PPU.
    eligible = (
        x.dtype == weight.dtype == torch.bfloat16 and x.stride(-1) == 1
        and hidden in (512, 2048)
        and (residual is None or residual.stride(-1) == 1)
    )
    if not eligible:
        import flashinfer

        if residual is None:
            out = flashinfer.norm.rmsnorm(x.contiguous(), weight, eps=eps)
        else:
            out = x if x.stride(-1) == 1 else x.contiguous()
            flashinfer.norm.fused_add_rmsnorm(out, residual, weight, eps=eps)
        if quantize:
            from rtp_llm.models_py.kernels.ppu.fp8_quant import fp8_quantize_128

            q, scales = fp8_quantize_128(out.contiguous())
            return out, q, scales
        return out, out, out
    out = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
    q = torch.empty_like(out, dtype=torch.float8_e4m3fn) if quantize else out
    scales = torch.empty((rows, hidden // 128), device=x.device, dtype=torch.float32) if quantize else out
    if rows:
        _rmsnorm_quant[(rows,)](
            x, residual if residual is not None else x, weight, out, q, scales,
            hidden, eps, x.stride(0), residual.stride(0) if residual is not None else 0,
            triton.next_power_of_2(hidden), residual is not None, quantize,
            num_warps=4 if hidden <= 2048 else 8, enable_fp_fusion=False,
        )
    return out, q, scales


def fused_strided_rmsnorm(x, weight, eps=1e-6):
    return _run(x, weight, eps, None, False)[0]


def fused_strided_rmsnorm_per_token_fp8_quant_with_bf16_output(
    x, weight, eps=1e-6, group_size=128, scale_ue8m0=False,
):
    if group_size != 128 or scale_ue8m0:
        raise ValueError("PPU dense FP8 uses groups of 128 with FP32 scales")
    return _run(x, weight, eps, None, True)


def fused_add_rmsnorm_fp8_quant_with_bf16_output(
    x, residual, weight, eps=1e-6, group_size=128, scale_ue8m0=False,
):
    if group_size != 128 or scale_ue8m0:
        raise ValueError("PPU dense FP8 uses groups of 128 with FP32 scales")
    return _run(x, weight, eps, residual, True)


def fused_add_rmsnorm_fp8_quant(x, residual, weight, eps=1e-6,
                              group_size=128, scale_ue8m0=False):
    _, q, scales = fused_add_rmsnorm_fp8_quant_with_bf16_output(
        x, residual, weight, eps, group_size, scale_ue8m0,
    )
    return q, scales
