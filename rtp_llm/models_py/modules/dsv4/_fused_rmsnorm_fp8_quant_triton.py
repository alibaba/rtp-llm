"""DSV4 fused RMSNorm + FP8 UE8M0 per-token group quantization."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M", "out_scale_stride_k"])
def _rmsnorm_fp8_quant_kernel(
    x_ptr,
    w_ptr,
    out_norm_ptr,
    out_q_ptr,
    out_scale_ptr,
    M,
    x_stride_m,
    out_norm_stride_m,
    out_q_stride_m,
    out_scale_stride_k,
    EPS: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_PACKS: tl.constexpr,
    STORE_NORM: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    if not STORE_NORM:
        # Match the K3.1 Prefill norm's four-row tile and round to BF16 in
        # registers before computing scales. Both Q projections consume FP8.
        rows = (tl.program_id(0) * 4 + tl.arange(0, 4)).to(tl.int64)
        mask = (rows[:, None] < M) & n_mask[None, :]
        x = tl.load(x_ptr + rows[:, None] * x_stride_m + offs_n[None, :],
                    mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        inv = tl.rsqrt(tl.sum(x * x, axis=1) / N + EPS)
        y = (x * inv[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
        groups = tl.reshape(y, (4, BLOCK_N // GROUP_SIZE, GROUP_SIZE))
        amax = tl.maximum(tl.max(tl.abs(groups), axis=2), CLAMP_EPS)
        # UE8M0 is the FP32 exponent rounded up to the next power of two.
        raw_bits = (amax * (1.0 / FP8_MAX)).to(tl.int32, bitcast=True)
        exponent = ((raw_bits >> 23) & 0xFF) + ((raw_bits & 0x7FFFFF) != 0).to(tl.int32)
        scale = (exponent << 23).to(tl.float32, bitcast=True)
        q = tl.reshape(tl.clamp(groups / scale[:, :, None], FP8_MIN, FP8_MAX), (4, BLOCK_N))
        tl.store(out_q_ptr + rows[:, None] * out_q_stride_m + offs_n[None, :], q, mask=mask)

        group_ids = tl.arange(0, BLOCK_N // GROUP_SIZE)
        exponent = tl.where(group_ids[None, :] < N // GROUP_SIZE, exponent, 0)
        packed = tl.sum(
            tl.reshape(exponent, (4, BLOCK_N // GROUP_SIZE // 4, 4))
            << (tl.arange(0, 4)[None, None, :] * 8),
            axis=2,
        )
        packs = tl.arange(0, BLOCK_N // GROUP_SIZE // 4)
        tl.store(out_scale_ptr + rows[:, None] + packs[None, :] * out_scale_stride_k,
                 packed, (rows[:, None] < M) & (packs[None, :] < SCALE_PACKS))
        return

    pid_m = tl.program_id(0).to(tl.int64)
    if pid_m >= M:
        return
    x_row = x_ptr + pid_m * x_stride_m
    x = tl.load(x_row + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(x * x, axis=0) / N + EPS)

    offs_g = tl.arange(0, GROUP_SIZE)
    out_norm_row = out_norm_ptr + pid_m * out_norm_stride_m
    out_q_row = out_q_ptr + pid_m * out_q_stride_m

    for pack_id in tl.static_range(0, SCALE_PACKS):
        packed_scale = tl.zeros((), dtype=tl.int32)
        for pack_idx in tl.static_range(0, 4):
            group_id = pack_id * 4 + pack_idx
            col_base = group_id * GROUP_SIZE
            cols = col_base + offs_g
            mask = cols < N
            x_g = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            w_g = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y = (x_g * inv * w_g).to(tl.bfloat16)
            tl.store(out_norm_row + cols, y, mask=mask)

            y_f32 = y.to(tl.float32)
            absmax = tl.max(tl.abs(y_f32), axis=0)
            scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            q = tl.clamp(y_f32 / scale, FP8_MIN, FP8_MAX)
            tl.store(out_q_row + cols, q.to(out_q_ptr.dtype.element_ty), mask=mask)

            exponent_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            packed_scale = packed_scale | (exponent_biased << (pack_idx * 8))

        tl.store(out_scale_ptr + pack_id * out_scale_stride_k + pid_m, packed_scale)


def _make_ue8m0_scale_like(
    x_shape: torch.Size | tuple[int, int],
    *,
    device: torch.device,
    group_size: int,
) -> torch.Tensor:
    m, n = int(x_shape[0]), int(x_shape[1])
    scale_cols = n // group_size
    aligned_m = triton.cdiv(m, 4) * 4
    aligned_k = triton.cdiv(scale_cols, 4) * 4
    return torch.empty(
        (aligned_k // 4, aligned_m), device=device, dtype=torch.int32
    ).transpose(-1, -2)[:m, :]


def rmsnorm_fp8_quant_ue8m0(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
    group_size: int = 128,
    clamp_eps: float = 1.0e-4,
    out_norm: Optional[torch.Tensor] = None,
    store_norm: bool = True,
) -> tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Normalize a BF16 matrix and quantize it; optionally omit the BF16 output."""
    if x.dim() != 2:
        raise ValueError(f"x must be [M,N], got {tuple(x.shape)}")
    if x.dtype != torch.bfloat16:
        raise ValueError(f"x must be bf16, got {x.dtype}")
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"weight must be bf16, got {weight.dtype}")
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("rmsnorm_fp8_quant_ue8m0 requires CUDA tensors")
    if store_norm:
        if not x.is_contiguous():
            raise ValueError("x must be contiguous")
    elif x.stride(1) != 1 or (x.shape[0] > 1 and x.stride(0) < x.shape[1]):
        raise ValueError("quant-only RMSNorm requires contiguous features and nonoverlapping rows")
    m, n = x.shape
    if weight.shape != (n,) or not weight.is_contiguous():
        raise ValueError(f"weight must be contiguous [{n}], got {tuple(weight.shape)}")
    if n % group_size != 0:
        raise ValueError(f"N={n} must be divisible by group_size={group_size}")

    if not store_norm:
        if out_norm is not None or group_size != 128:
            raise ValueError("quant-only RMSNorm requires group 128 and no BF16 output")
    elif out_norm is None:
        out_norm = torch.empty_like(x)
    else:
        if out_norm.shape != x.shape:
            raise ValueError(
                f"out_norm must have shape {tuple(x.shape)}, got {tuple(out_norm.shape)}"
            )
        if out_norm.dtype != torch.bfloat16:
            raise ValueError(f"out_norm must be bf16, got {out_norm.dtype}")
        if out_norm.device != x.device:
            raise ValueError("out_norm must be on the same device as x")
        if not out_norm.is_contiguous():
            raise ValueError("out_norm must be contiguous")

    out_q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    out_scale = _make_ue8m0_scale_like(x.shape, device=x.device, group_size=group_size)
    if m == 0:
        return out_norm, out_q, out_scale

    finfo = torch.finfo(torch.float8_e4m3fn)
    scale_packs = (n // group_size + 3) // 4
    block_n = triton.next_power_of_2(n) if store_norm else max(512, triton.next_power_of_2(n))
    _rmsnorm_fp8_quant_kernel[(m if store_norm else triton.cdiv(m, 4),)](
        x,
        weight,
        x if out_norm is None else out_norm,
        out_q,
        out_scale,
        m,
        x.stride(0),
        0 if out_norm is None else out_norm.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
        EPS=eps,
        CLAMP_EPS=clamp_eps,
        FP8_MIN=finfo.min,
        FP8_MAX=finfo.max,
        N=n,
        BLOCK_N=block_n,
        GROUP_SIZE=group_size,
        SCALE_PACKS=scale_packs,
        STORE_NORM=store_norm,
        num_warps=8 if not store_norm or block_n >= 4096 else 4,
        num_stages=2,
    )
    return out_norm, out_q, out_scale
