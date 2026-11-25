"""Fused DeepSeek-V4 mHC pre + RMSNorm (+ optional FP8 UE8M0 quant) & copy kernels."""

from __future__ import annotations

import functools
from typing import Any, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit

_MHC_MULT = 4
_MHC_MULT2 = 16
_MHC_MULT3 = 24
_MHC_POST_MULT = 2.0
_MAX_FUSED_HIDDEN = 8192
_UE8M0_GROUP_SIZE = 128


@functools.cache
def _is_sm100_device_index(device_idx: int) -> bool:
    major, _minor = torch.cuda.get_device_capability(device_idx)
    return major == 10


def _is_sm100_device(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    device_idx = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return _is_sm100_device_index(device_idx)


@functools.cache
def _has_tf32_hc_prenorm_gemm() -> bool:
    from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper

    if not deepgemm_wrapper.has_deep_gemm():
        return False
    if deepgemm_wrapper._tf32_hc_prenorm_gemm_impl is None:
        try:
            deepgemm_wrapper._lazy_init_deep_gemm(["tf32_hc_prenorm_gemm"])
        except Exception:
            return False
    return deepgemm_wrapper._tf32_hc_prenorm_gemm_impl is not None


@functools.cache
def _device_sm_count(device_idx: int) -> int:
    return int(torch.cuda.get_device_properties(device_idx).multi_processor_count)


def _compute_num_split(
    block_k: int, k: int, grid_size: int, device: torch.device
) -> int:
    device_idx = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    n_sms = _device_sm_count(device_idx)
    split_k = n_sms // max(grid_size, 1)
    num_block_k = (k + block_k - 1) // block_k
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def _make_ue8m0_scale_like(
    m: int,
    n: int,
    *,
    device: torch.device,
    group_size: int = _UE8M0_GROUP_SIZE,
) -> torch.Tensor:
    scale_cols = n // group_size
    aligned_m = ((m + 3) // 4) * 4
    aligned_k = ((scale_cols + 3) // 4) * 4
    return torch.empty(
        (aligned_k // 4, aligned_m), device=device, dtype=torch.int32
    ).transpose(-1, -2)[:m, :]


if triton is not None:

    @triton.jit(
        do_not_specialize=["M", "n_splits", "sinkhorn_repeat", "out_scale_stride_k"]
    )
    def _mhc_pre_big_fuse_rmsnorm_kernel(
        gemm_out_mul_ptr,
        gemm_out_sqrsum_ptr,
        mhc_scale_ptr,
        mhc_base_ptr,
        residual_ptr,
        norm_weight_ptr,
        post_mix_ptr,
        comb_mix_ptr,
        out_norm_ptr,
        out_q_ptr,
        out_scale_ptr,
        M,
        n_splits,
        sinkhorn_repeat,
        out_scale_stride_k,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_H: tl.constexpr,
        MHC_MULT: tl.constexpr,
        MHC_MULT3: tl.constexpr,
        MHC_NORM_EPS: tl.constexpr,
        MHC_PRE_EPS: tl.constexpr,
        MHC_SINKHORN_EPS: tl.constexpr,
        MHC_POST_MULT: tl.constexpr,
        RMS_EPS: tl.constexpr,
        QUANT_UE8M0: tl.constexpr,
        CLAMP_EPS: tl.constexpr,
        FP8_MIN: tl.constexpr,
        FP8_MAX: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_NUM_GROUPS: tl.constexpr,
        SCALE_PACKS: tl.constexpr,
        ACTIVE_SCALE_PACKS: tl.constexpr,
    ):
        pid = tl.program_id(0).to(tl.int64)
        M_i64 = M.to(tl.int64)
        if pid >= M_i64:
            return

        offs_4 = tl.arange(0, 4)
        offs_4_2d = offs_4[:, None] * 4 + offs_4[None, :]
        sqrsum = 0.0
        mix_pre = tl.zeros((4,), dtype=tl.float32)
        mix_post = tl.zeros((4,), dtype=tl.float32)
        mix_comb = tl.zeros((4, 4), dtype=tl.float32)
        for i_split in range(n_splits):
            split_idx = i_split.to(tl.int64)
            sqrsum += tl.load(gemm_out_sqrsum_ptr + split_idx * M_i64 + pid)
            mul_base = gemm_out_mul_ptr + (split_idx * M_i64 + pid) * MHC_MULT3
            mix_pre += tl.load(mul_base + offs_4)
            mix_post += tl.load(mul_base + 4 + offs_4)
            mix_comb += tl.load(mul_base + 8 + offs_4_2d)

        inv_hc = tl.rsqrt(sqrsum / (MHC_MULT * HIDDEN_SIZE) + MHC_NORM_EPS)
        mix_pre *= inv_hc
        mix_post *= inv_hc
        mix_comb *= inv_hc

        s0 = tl.load(mhc_scale_ptr + 0)
        s1 = tl.load(mhc_scale_ptr + 1)
        s2 = tl.load(mhc_scale_ptr + 2)
        b_pre = tl.load(mhc_base_ptr + offs_4)
        b_post = tl.load(mhc_base_ptr + 4 + offs_4)
        b_comb = tl.load(mhc_base_ptr + 8 + offs_4_2d)

        pre_mix = tl.sigmoid(mix_pre * s0 + b_pre) + MHC_PRE_EPS
        post_mix = tl.sigmoid(mix_post * s1 + b_post) * MHC_POST_MULT
        tl.store(post_mix_ptr + pid * MHC_MULT + offs_4, post_mix)

        cm = mix_comb * s2 + b_comb
        row_max = tl.max(cm, axis=1)[:, None]
        cm = tl.exp(cm - row_max)
        row_sum = tl.sum(cm, axis=1)[:, None]
        cm = cm / row_sum + MHC_SINKHORN_EPS
        col_sum = tl.sum(cm, axis=0)[None, :]
        cm = cm / (col_sum + MHC_SINKHORN_EPS)
        for _ in range(sinkhorn_repeat - 1):
            row_sum = tl.sum(cm, axis=1)[:, None]
            cm = cm / (row_sum + MHC_SINKHORN_EPS)
            col_sum = tl.sum(cm, axis=0)[None, :]
            cm = cm / (col_sum + MHC_SINKHORN_EPS)
        tl.store(comb_mix_ptr + pid * 16 + offs_4_2d, cm)

        offs_h = tl.arange(0, BLOCK_H)
        h_mask = offs_h < HIDDEN_SIZE
        ol = tl.zeros((BLOCK_H,), dtype=tl.float32)
        res_base = residual_ptr + pid * (MHC_MULT * HIDDEN_SIZE)
        for lane in tl.static_range(MHC_MULT):
            p_lane = tl.sum(tl.where(offs_4 == lane, pre_mix, 0.0), axis=0)
            x_lane = tl.load(
                res_base + lane * HIDDEN_SIZE + offs_h, mask=h_mask, other=0.0
            ).to(tl.float32)
            ol += p_lane * x_lane

        # Preserve exact BF16 intermediate rounding between mHC pre and RMSNorm.
        ol_bf16 = ol.to(tl.bfloat16).to(tl.float32)
        var = tl.sum(ol_bf16 * ol_bf16, axis=0) / HIDDEN_SIZE
        rrms = tl.rsqrt(var + RMS_EPS)
        w = tl.load(norm_weight_ptr + offs_h, mask=h_mask, other=0.0).to(tl.float32)
        y = (ol_bf16 * rrms * w).to(tl.bfloat16)
        tl.store(out_norm_ptr + pid * HIDDEN_SIZE + offs_h, y, mask=h_mask)

        if QUANT_UE8M0:
            y_f32 = y.to(tl.float32)
            y_grouped = tl.reshape(y_f32, (BLOCK_NUM_GROUPS, GROUP_SIZE))
            absmax = tl.max(tl.abs(y_grouped), axis=1)
            scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            q = tl.clamp(y_grouped / scale[:, None], FP8_MIN, FP8_MAX).to(
                out_q_ptr.dtype.element_ty
            )
            tl.store(
                out_q_ptr + pid * HIDDEN_SIZE + offs_h,
                tl.reshape(q, (BLOCK_H,)),
                mask=h_mask,
            )
            exp_biased = tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32)
            exp_packs = tl.reshape(exp_biased, (SCALE_PACKS, 4))
            pack_shifts = tl.arange(0, 4)[None, :] * 8
            packed_scales = tl.sum(exp_packs << pack_shifts, axis=1)
            offs_sp = tl.arange(0, SCALE_PACKS)
            tl.store(
                out_scale_ptr + offs_sp.to(tl.int64) * out_scale_stride_k + pid,
                packed_scales,
                mask=offs_sp < ACTIVE_SCALE_PACKS,
            )

    @triton.jit(
        do_not_specialize=[
            "rows",
            "in_row_stride",
            "in_hc_stride",
            "out_row_stride",
            "col_offset",
        ]
    )
    def _mhc_mean_copy_kernel(
        hidden_ptr,
        out_ptr,
        rows,
        dim,
        in_row_stride,
        in_hc_stride,
        out_row_stride,
        col_offset,
        MHC_MULT: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        if pid_m >= rows:
            return
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < dim
        in_base = hidden_ptr + pid_m * in_row_stride + cols
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for lane in tl.static_range(MHC_MULT):
            acc += tl.load(in_base + lane * in_hc_stride, mask=mask, other=0.0).to(
                tl.float32
            )
        pooled = (acc * 0.25).to(hidden_ptr.dtype.element_ty)
        tl.store(
            out_ptr + pid_m * out_row_stride + col_offset + cols,
            pooled.to(out_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit(do_not_specialize=["rows", "src_stride_m", "dst_stride_m"])
    def _triton_copy_2d_kernel(
        src_ptr,
        dst_ptr,
        rows,
        dim,
        src_stride_m,
        dst_stride_m,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1).to(tl.int64)
        if pid_m >= rows:
            return
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < dim
        val = tl.load(src_ptr + pid_m * src_stride_m + cols, mask=mask, other=0.0)
        tl.store(dst_ptr + pid_m * dst_stride_m + cols, val, mask=mask)


def _check_unit_and_norm_unpatched(hc_unit: Any, norm: Any) -> bool:
    if type(hc_unit) is not TileLangHCUnit:
        return False
    if getattr(hc_unit, "_pre_impl_prefill_fast", None) is not None:
        return False
    if (
        "_pre_impl" in hc_unit.__dict__
        or "pre" in hc_unit.__dict__
        or "_pre_impl_prefill_fast" in hc_unit.__dict__
        or type(hc_unit)._pre_impl is not TileLangHCUnit._pre_impl
        or type(hc_unit).pre is not HCUnitBase.pre
    ):
        return False
    if (
        norm.__class__ is not RMSNorm
        or norm.__class__.__module__ != "rtp_llm.models_py.modules.base.cuda.norm"
        or "forward" in norm.__dict__
        or type(norm).forward is not RMSNorm.forward
    ):
        return False
    return True


def can_use_fused_mhc_pre_norm(
    x: torch.Tensor,
    hc_unit: Any,
    norm: Any,
    *,
    quant_ue8m0: bool = False,
) -> bool:
    """Return True when mHC pre + RMSNorm (+ optional UE8M0 quant) can fuse."""
    if triton is None or torch.is_grad_enabled():
        return False
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return False
    if x.dtype != torch.bfloat16 or x.numel() == 0 or not x.is_contiguous():
        return False
    if quant_ue8m0:
        if x.dim() != 3:
            return False
    elif x.dim() not in (3, 4):
        return False
    if not _check_unit_and_norm_unpatched(hc_unit, norm):
        return False
    if int(hc_unit.hc_mult) != _MHC_MULT or int(hc_unit.hc_sinkhorn_iters) < 1:
        return False
    dim = int(hc_unit.dim)
    if int(x.shape[-2]) != _MHC_MULT or int(x.shape[-1]) != dim:
        return False
    if (_MHC_MULT * dim) % 256 != 0 or dim > _MAX_FUSED_HIDDEN:
        return False
    if quant_ue8m0 and dim % _UE8M0_GROUP_SIZE != 0:
        return False

    fn = hc_unit.fn
    base = hc_unit.base
    scale = hc_unit.scale
    weight = getattr(getattr(norm, "weight", None), "data", None)
    if (
        not isinstance(fn, torch.Tensor)
        or not isinstance(base, torch.Tensor)
        or not isinstance(scale, torch.Tensor)
        or not isinstance(weight, torch.Tensor)
    ):
        return False
    if (
        fn.device != x.device
        or base.device != x.device
        or scale.device != x.device
        or weight.device != x.device
    ):
        return False
    if (
        fn.dtype != torch.float32
        or base.dtype != torch.float32
        or scale.dtype != torch.float32
        or weight.dtype != torch.bfloat16
    ):
        return False
    if (
        not fn.is_contiguous()
        or not base.is_contiguous()
        or not scale.is_contiguous()
        or not weight.is_contiguous()
    ):
        return False
    if (
        tuple(fn.shape) != (_MHC_MULT3, _MHC_MULT * dim)
        or tuple(base.shape) != (_MHC_MULT3,)
        or tuple(scale.shape) != (3,)
        or tuple(weight.shape) != (dim,)
    ):
        return False
    return _is_sm100_device(x.device) and _has_tf32_hc_prenorm_gemm()


def _run_fused_mhc_pre_norm(
    x: torch.Tensor,
    hc_unit: TileLangHCUnit,
    norm: RMSNorm,
    *,
    quant_ue8m0: bool,
    group_size: int = _UE8M0_GROUP_SIZE,
    clamp_eps: float = 1.0e-4,
):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

    if triton is None:
        raise RuntimeError("triton is unavailable")
    outer_shape = x.shape[:-2]
    hidden_size = int(hc_unit.dim)
    mhc_hidden_size = _MHC_MULT * hidden_size
    residual_flat = x.reshape(-1, _MHC_MULT, hidden_size)
    num_tokens = int(residual_flat.shape[0])

    n_splits = _compute_num_split(
        64, mhc_hidden_size, (num_tokens + 63) // 64, x.device
    )
    gemm_out_mul = torch.empty(
        n_splits, num_tokens, _MHC_MULT3, dtype=torch.float32, device=x.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=x.device
    )
    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, mhc_hidden_size),
        hc_unit.fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    post_mix = torch.empty(num_tokens, _MHC_MULT, dtype=torch.float32, device=x.device)
    comb_mix = torch.empty(num_tokens, _MHC_MULT2, dtype=torch.float32, device=x.device)
    out_norm = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device
    )

    block_h = max(512, triton.next_power_of_2(hidden_size))
    block_num_groups = block_h // group_size
    scale_packs = block_num_groups // 4

    if quant_ue8m0:
        out_q = torch.empty(
            num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=x.device
        )
        out_scale = _make_ue8m0_scale_like(
            num_tokens, hidden_size, device=x.device, group_size=group_size
        )
        out_scale_stride_k = int(out_scale.stride(1))
        active_scale_packs = (hidden_size // group_size + 3) // 4
        finfo = torch.finfo(torch.float8_e4m3fn)
        fp8_min = float(finfo.min)
        fp8_max = float(finfo.max)
    else:
        out_q = out_norm
        out_scale = post_mix
        out_scale_stride_k = 1
        active_scale_packs = 1
        fp8_min = -448.0
        fp8_max = 448.0

    _mhc_pre_big_fuse_rmsnorm_kernel[(num_tokens,)](
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_unit.scale,
        hc_unit.base,
        residual_flat,
        norm.weight.data,
        post_mix,
        comb_mix,
        out_norm,
        out_q,
        out_scale,
        num_tokens,
        n_splits,
        int(hc_unit.hc_sinkhorn_iters),
        out_scale_stride_k,
        HIDDEN_SIZE=hidden_size,
        BLOCK_H=block_h,
        MHC_MULT=_MHC_MULT,
        MHC_MULT3=_MHC_MULT3,
        MHC_NORM_EPS=float(hc_unit.norm_eps),
        MHC_PRE_EPS=float(hc_unit.hc_eps),
        MHC_SINKHORN_EPS=float(hc_unit.hc_eps),
        MHC_POST_MULT=_MHC_POST_MULT,
        RMS_EPS=float(norm.variance_epsilon),
        QUANT_UE8M0=quant_ue8m0,
        CLAMP_EPS=float(clamp_eps),
        FP8_MIN=fp8_min,
        FP8_MAX=fp8_max,
        GROUP_SIZE=group_size,
        BLOCK_NUM_GROUPS=block_num_groups,
        SCALE_PACKS=scale_packs,
        ACTIVE_SCALE_PACKS=active_scale_packs,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )

    out_norm_view = out_norm.view(*outer_shape, hidden_size)
    post_mix_view = post_mix.view(*outer_shape, _MHC_MULT, 1)
    comb_mix_view = comb_mix.view(*outer_shape, _MHC_MULT, _MHC_MULT)
    if quant_ue8m0:
        return out_norm_view, post_mix_view, comb_mix_view, (out_q, out_scale)
    return out_norm_view, post_mix_view, comb_mix_view


def fused_mhc_pre_norm(
    x: torch.Tensor,
    hc_unit: TileLangHCUnit,
    norm: RMSNorm,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused mHC pre + RMSNorm and return (x_pre_normed, post_mix, comb_mix)."""
    return _run_fused_mhc_pre_norm(x, hc_unit, norm, quant_ue8m0=False)


def fused_mhc_pre_norm_quant(
    x: torch.Tensor,
    hc_unit: TileLangHCUnit,
    norm: RMSNorm,
    *,
    group_size: int = _UE8M0_GROUP_SIZE,
    clamp_eps: float = 1.0e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run fused mHC pre + RMSNorm + FP8 UE8M0 quant for attention QKV reuse."""
    return _run_fused_mhc_pre_norm(
        x,
        hc_unit,
        norm,
        quant_ue8m0=True,
        group_size=group_size,
        clamp_eps=clamp_eps,
    )


def mhc_mean_copy(
    hidden: torch.Tensor,
    buf: torch.Tensor,
    *,
    col_offset: int,
    rows: Optional[int] = None,
    dim: Optional[int] = None,
) -> None:
    """Fused mean(dim=-2) across HC lanes + 2D slice copy into buf."""
    if dim is None:
        dim = int(hidden.shape[-1])
    if rows is None:
        rows = int(hidden.numel() // max(int(hidden.shape[-2]) * dim, 1))
    if rows == 0 or dim == 0:
        return
    if (
        triton is not None
        and hidden.is_cuda
        and buf.is_cuda
        and hidden.dim() in (3, 4)
        and int(hidden.shape[-2]) == _MHC_MULT
        and int(hidden.shape[-1]) == dim
        and hidden.is_contiguous()
        and buf.dim() == 2
        and buf.stride(1) == 1
    ):
        block_n = 1024
        grid = (rows, triton.cdiv(dim, block_n))
        _mhc_mean_copy_kernel[grid](
            hidden,
            buf,
            rows,
            dim,
            _MHC_MULT * dim,
            dim,
            buf.stride(0),
            col_offset,
            MHC_MULT=_MHC_MULT,
            BLOCK_N=block_n,
            num_warps=4,
        )
        return
    pooled = hidden.mean(dim=-2)
    flat = pooled.reshape(-1, dim)
    buf[:rows, col_offset : col_offset + dim].copy_(flat)


def copy_2d_buffer(flat: torch.Tensor, buf: torch.Tensor, *, rows: int) -> None:
    """Copy [rows, dim] into preallocated 2D buf without ATen elementwise overhead."""
    if rows == 0:
        return
    dim = int(flat.size(1))
    if dim == 0:
        buf[:rows].copy_(flat)
        return
    if (
        triton is not None
        and flat.is_cuda
        and buf.is_cuda
        and flat.dim() == 2
        and buf.dim() == 2
        and flat.stride(1) == 1
        and buf.stride(1) == 1
        and flat.dtype == buf.dtype
    ):
        if flat.data_ptr() == buf.data_ptr() and flat.stride(0) == buf.stride(0):
            return
        block_n = 1024
        grid = (rows, triton.cdiv(dim, block_n))
        _triton_copy_2d_kernel[grid](
            flat,
            buf,
            rows,
            dim,
            flat.stride(0),
            buf.stride(0),
            BLOCK_N=block_n,
            num_warps=4,
        )
        return
    buf[:rows].copy_(flat)


__all__ = [
    "can_use_fused_mhc_pre_norm",
    "copy_2d_buffer",
    "fused_mhc_pre_norm",
    "fused_mhc_pre_norm_quant",
    "mhc_mean_copy",
]
