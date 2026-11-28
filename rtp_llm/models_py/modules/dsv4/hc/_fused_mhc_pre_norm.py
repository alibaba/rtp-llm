"""Fused DeepSeek-V4 mHC pre + RMSNorm (+ optional FP8 UE8M0 quant) & copy kernels."""

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

try:
    from rtp_llm.models_py.triton_kernels.fast_launch import fast_launch
except ImportError:  # pragma: no cover

    def fast_launch(jit_fn: Any, grid: Any, *args: Any, **kwargs: Any) -> Any:
        return jit_fn[grid](*args, **kwargs)


_MHC_MULT = 4
_MHC_MULT2 = 16
_MHC_MULT3 = 24
_MHC_POST_MULT = 2.0
_MAX_FUSED_HIDDEN = 8192
_UE8M0_GROUP_SIZE = 128

_FP8_E4M3_FINFO = torch.finfo(torch.float8_e4m3fn)
_FP8_E4M3_MIN = float(_FP8_E4M3_FINFO.min)
_FP8_E4M3_MAX = float(_FP8_E4M3_FINFO.max)
_DEEPGEMM_WRAPPER: Any = None

# The fused kernel launches one CTA per token row and processes the whole
# BLOCK_H=4096 row (4 mHC lanes + RMSNorm + optional UE8M0 quant) inside that
# single CTA. Below this token count the kernel is latency/occupancy bound
# (decode M_local<=128 uses too few CTAs out of 148 SMs and is slower on GPU
# kernel time than the unfused TileLang pre + RMSNorm pair). At M >= 256
# (including prefill M=558, -26%..-41% GPU kernel and -35%..-37% E2E on SM103),
# the single-pass-over-residual fusion wins on both launch count and HBM
# traffic while keeping decode (M <= 128) on the unfused path.
_FUSE_MIN_TOKENS = 256


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
        CHUNK_H: tl.constexpr,
        NUM_CHUNKS: tl.constexpr,
        IS_ALIGNED: tl.constexpr,
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
        PACKS_PER_CHUNK: tl.constexpr,
        ACTIVE_GROUPS: tl.constexpr,
        ACTIVE_SCALE_PACKS: tl.constexpr,
        HAS_TAIL_PAD: tl.constexpr,
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
        s0 = tl.load(mhc_scale_ptr + 0)
        s1 = tl.load(mhc_scale_ptr + 1)
        s2 = tl.load(mhc_scale_ptr + 2)
        b_pre = tl.load(mhc_base_ptr + offs_4)
        b_post = tl.load(mhc_base_ptr + 4 + offs_4)
        b_comb = tl.load(mhc_base_ptr + 8 + offs_4_2d)

        pre_mix = tl.sigmoid(mix_pre * s0 + b_pre) + MHC_PRE_EPS
        p0 = tl.sum(tl.where(offs_4 == 0, pre_mix, 0.0), axis=0)
        p1 = tl.sum(tl.where(offs_4 == 1, pre_mix, 0.0), axis=0)
        p2 = tl.sum(tl.where(offs_4 == 2, pre_mix, 0.0), axis=0)
        p3 = tl.sum(tl.where(offs_4 == 3, pre_mix, 0.0), axis=0)

        post_mix = tl.sigmoid((mix_post * inv_hc) * s1 + b_post) * MHC_POST_MULT
        tl.store(post_mix_ptr + pid * MHC_MULT + offs_4, post_mix)

        cm = (mix_comb * inv_hc) * s2 + b_comb
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

        res_base = residual_ptr + pid * (MHC_MULT * HIDDEN_SIZE)
        norm_row_ptr = out_norm_ptr + pid * HIDDEN_SIZE
        q_row_ptr = out_q_ptr + pid * HIDDEN_SIZE
        offs_c = tl.arange(0, CHUNK_H)

        if NUM_CHUNKS == 1:
            if IS_ALIGNED:
                x0 = tl.load(res_base + offs_c, eviction_policy="evict_first").to(
                    tl.float32
                )
                x1 = tl.load(
                    res_base + HIDDEN_SIZE + offs_c, eviction_policy="evict_first"
                ).to(tl.float32)
                x2 = tl.load(
                    res_base + 2 * HIDDEN_SIZE + offs_c, eviction_policy="evict_first"
                ).to(tl.float32)
                x3 = tl.load(
                    res_base + 3 * HIDDEN_SIZE + offs_c, eviction_policy="evict_first"
                ).to(tl.float32)
                ol = ((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3
                ol_bf16 = ol.to(tl.bfloat16).to(tl.float32)
                var = tl.sum(ol_bf16 * ol_bf16, axis=0) / HIDDEN_SIZE
                rrms = tl.rsqrt(var + RMS_EPS)
                w = tl.load(norm_weight_ptr + offs_c).to(tl.float32)
                y = (ol_bf16 * rrms * w).to(tl.bfloat16)
                tl.store(norm_row_ptr + offs_c, y)
            else:
                c_mask = offs_c < HIDDEN_SIZE
                x0 = tl.load(
                    res_base + offs_c,
                    mask=c_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                x1 = tl.load(
                    res_base + HIDDEN_SIZE + offs_c,
                    mask=c_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                x2 = tl.load(
                    res_base + 2 * HIDDEN_SIZE + offs_c,
                    mask=c_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                x3 = tl.load(
                    res_base + 3 * HIDDEN_SIZE + offs_c,
                    mask=c_mask,
                    other=0.0,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                ol = ((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3
                # Preserve exact BF16 intermediate rounding between mHC pre and RMSNorm.
                ol_bf16 = tl.where(c_mask, ol.to(tl.bfloat16).to(tl.float32), 0.0)
                var = tl.sum(ol_bf16 * ol_bf16, axis=0) / HIDDEN_SIZE
                rrms = tl.rsqrt(var + RMS_EPS)
                w = tl.load(norm_weight_ptr + offs_c, mask=c_mask, other=0.0).to(
                    tl.float32
                )
                y = (ol_bf16 * rrms * w).to(tl.bfloat16)
                tl.store(norm_row_ptr + offs_c, y, mask=c_mask)

            if QUANT_UE8M0:
                y_f32 = y.to(tl.float32)
                y_grouped = tl.reshape(y_f32, (PACKS_PER_CHUNK, 4, GROUP_SIZE))
                absmax = tl.max(tl.abs(y_grouped), axis=2)
                scale_raw = tl.maximum(absmax, CLAMP_EPS) * (1.0 / FP8_MAX)
                raw_bits = scale_raw.to(tl.int32, bitcast=True)
                mant = raw_bits & 0x7FFFFF
                exp_biased = tl.minimum(
                    tl.maximum(
                        ((raw_bits >> 23) & 0xFF) + tl.where(mant != 0, 1, 0), 1
                    ),
                    253,
                )
                inv_scale = ((254 - exp_biased) << 23).to(tl.float32, bitcast=True)
                q = tl.clamp(y_grouped * inv_scale[:, :, None], FP8_MIN, FP8_MAX).to(
                    out_q_ptr.dtype.element_ty
                )
                if IS_ALIGNED:
                    tl.store(q_row_ptr + offs_c, tl.reshape(q, (CHUNK_H,)))
                else:
                    tl.store(
                        q_row_ptr + offs_c,
                        tl.reshape(q, (CHUNK_H,)),
                        mask=offs_c < HIDDEN_SIZE,
                    )
                if HAS_TAIL_PAD:
                    group_ids = (
                        tl.arange(0, PACKS_PER_CHUNK)[:, None] * 4
                        + tl.arange(0, 4)[None, :]
                    )
                    exp_biased = tl.where(group_ids < ACTIVE_GROUPS, exp_biased, 0)
                pack_shifts = tl.arange(0, 4)[None, :] * 8
                packed_scales = tl.sum(exp_biased << pack_shifts, axis=1)
                offs_sp = tl.arange(0, PACKS_PER_CHUNK)
                tl.store(
                    out_scale_ptr + offs_sp.to(tl.int64) * out_scale_stride_k + pid,
                    packed_scales,
                    mask=offs_sp < ACTIVE_SCALE_PACKS,
                )
        else:
            var_vec = tl.zeros((CHUNK_H,), dtype=tl.float32)
            for chunk_id in range(0, NUM_CHUNKS):
                cols = chunk_id * CHUNK_H + offs_c
                if IS_ALIGNED:
                    x0 = tl.load(res_base + cols, eviction_policy="evict_first").to(
                        tl.float32
                    )
                    x1 = tl.load(
                        res_base + HIDDEN_SIZE + cols, eviction_policy="evict_first"
                    ).to(tl.float32)
                    x2 = tl.load(
                        res_base + 2 * HIDDEN_SIZE + cols, eviction_policy="evict_first"
                    ).to(tl.float32)
                    x3 = tl.load(
                        res_base + 3 * HIDDEN_SIZE + cols, eviction_policy="evict_first"
                    ).to(tl.float32)
                    ol_c = ((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3
                    ol_bf16_c = ol_c.to(tl.bfloat16)
                    tl.store(norm_row_ptr + cols, ol_bf16_c)
                    ol_bf16_f32 = ol_bf16_c.to(tl.float32)
                else:
                    c_mask = cols < HIDDEN_SIZE
                    x0 = tl.load(
                        res_base + cols,
                        mask=c_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    x1 = tl.load(
                        res_base + HIDDEN_SIZE + cols,
                        mask=c_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    x2 = tl.load(
                        res_base + 2 * HIDDEN_SIZE + cols,
                        mask=c_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    x3 = tl.load(
                        res_base + 3 * HIDDEN_SIZE + cols,
                        mask=c_mask,
                        other=0.0,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    ol_c = ((p0 * x0 + p1 * x1) + p2 * x2) + p3 * x3
                    # Preserve exact BF16 intermediate rounding between mHC pre and RMSNorm;
                    # stage in out_norm_ptr (L2-hot) so Pass 2 avoids re-reading 4x residual.
                    ol_bf16_c = ol_c.to(tl.bfloat16)
                    tl.store(norm_row_ptr + cols, ol_bf16_c, mask=c_mask)
                    ol_bf16_f32 = tl.where(c_mask, ol_bf16_c.to(tl.float32), 0.0)
                var_vec += ol_bf16_f32 * ol_bf16_f32

            rrms = tl.rsqrt(tl.sum(var_vec, axis=0) / HIDDEN_SIZE + RMS_EPS)
            pack_shifts = tl.arange(0, 4)[None, :] * 8
            for chunk_id in range(0, NUM_CHUNKS):
                cols = chunk_id * CHUNK_H + offs_c
                if IS_ALIGNED:
                    ol_bf16_c = tl.load(norm_row_ptr + cols).to(tl.float32)
                    w_c = tl.load(norm_weight_ptr + cols).to(tl.float32)
                    y_c = (ol_bf16_c * rrms * w_c).to(tl.bfloat16)
                    tl.store(norm_row_ptr + cols, y_c)
                else:
                    c_mask = cols < HIDDEN_SIZE
                    ol_bf16_c = tl.load(norm_row_ptr + cols, mask=c_mask, other=0.0).to(
                        tl.float32
                    )
                    w_c = tl.load(norm_weight_ptr + cols, mask=c_mask, other=0.0).to(
                        tl.float32
                    )
                    y_c = (ol_bf16_c * rrms * w_c).to(tl.bfloat16)
                    tl.store(norm_row_ptr + cols, y_c, mask=c_mask)

                if QUANT_UE8M0:
                    y_f32 = y_c.to(tl.float32)
                    y_grouped = tl.reshape(y_f32, (PACKS_PER_CHUNK, 4, GROUP_SIZE))
                    absmax = tl.max(tl.abs(y_grouped), axis=2)
                    scale_raw = tl.maximum(absmax, CLAMP_EPS) * (1.0 / FP8_MAX)
                    raw_bits = scale_raw.to(tl.int32, bitcast=True)
                    mant = raw_bits & 0x7FFFFF
                    exp_biased = tl.minimum(
                        tl.maximum(
                            ((raw_bits >> 23) & 0xFF) + tl.where(mant != 0, 1, 0), 1
                        ),
                        253,
                    )
                    inv_scale = ((254 - exp_biased) << 23).to(tl.float32, bitcast=True)
                    q_c = tl.clamp(
                        y_grouped * inv_scale[:, :, None], FP8_MIN, FP8_MAX
                    ).to(out_q_ptr.dtype.element_ty)
                    if IS_ALIGNED:
                        tl.store(q_row_ptr + cols, tl.reshape(q_c, (CHUNK_H,)))
                    else:
                        tl.store(
                            q_row_ptr + cols,
                            tl.reshape(q_c, (CHUNK_H,)),
                            mask=cols < HIDDEN_SIZE,
                        )
                    pack_base = chunk_id * PACKS_PER_CHUNK
                    if HAS_TAIL_PAD:
                        group_ids = (pack_base + tl.arange(0, PACKS_PER_CHUNK))[
                            :, None
                        ] * 4 + tl.arange(0, 4)[None, :]
                        exp_biased = tl.where(group_ids < ACTIVE_GROUPS, exp_biased, 0)
                    packed_scales = tl.sum(exp_biased << pack_shifts, axis=1)
                    offs_sp = pack_base + tl.arange(0, PACKS_PER_CHUNK)
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
    min_tokens: int = _FUSE_MIN_TOKENS,
) -> bool:
    """Return True when mHC pre + RMSNorm (+ optional UE8M0 quant) can fuse.

    ``min_tokens`` dispatches the fused kernel away from small batches: the
    per-row CTA design only pays off once enough rows exist to cover the SM
    array, so decode-scale inputs fall back to the unfused TileLang pre +
    RMSNorm chain (see ``_FUSE_MIN_TOKENS``). Tests may pass
    ``min_tokens=0`` to exercise the fused kernel on tiny inputs.
    """
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
    if min_tokens > 0 and x.numel() // (_MHC_MULT * dim) < min_tokens:
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
    out_gemm_mul: Optional[torch.Tensor] = None,
    out_gemm_sqrsum: Optional[torch.Tensor] = None,
    out_post_mix: Optional[torch.Tensor] = None,
    out_comb_mix: Optional[torch.Tensor] = None,
    out_norm: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
):
    global _DEEPGEMM_WRAPPER
    dg = _DEEPGEMM_WRAPPER
    if dg is None:
        from rtp_llm.models_py.kernels.cuda import deepgemm_wrapper as dg

        _DEEPGEMM_WRAPPER = dg
    tf32_hc_prenorm_gemm = dg.tf32_hc_prenorm_gemm

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
    gemm_out_mul = (
        out_gemm_mul.view(n_splits, num_tokens, _MHC_MULT3)
        if out_gemm_mul is not None
        else torch.empty(
            n_splits, num_tokens, _MHC_MULT3, dtype=torch.float32, device=x.device
        )
    )
    gemm_out_sqrsum = (
        out_gemm_sqrsum.view(n_splits, num_tokens)
        if out_gemm_sqrsum is not None
        else torch.empty(n_splits, num_tokens, dtype=torch.float32, device=x.device)
    )
    tf32_hc_prenorm_gemm(
        residual_flat.view(num_tokens, mhc_hidden_size),
        hc_unit.fn,
        gemm_out_mul,
        gemm_out_sqrsum,
        n_splits,
    )

    post_mix = (
        out_post_mix.view(num_tokens, _MHC_MULT)
        if out_post_mix is not None
        else torch.empty(num_tokens, _MHC_MULT, dtype=torch.float32, device=x.device)
    )
    comb_mix = (
        out_comb_mix.view(num_tokens, _MHC_MULT2)
        if out_comb_mix is not None
        else torch.empty(num_tokens, _MHC_MULT2, dtype=torch.float32, device=x.device)
    )
    out_norm_buf = (
        out_norm.view(num_tokens, hidden_size)
        if out_norm is not None
        else torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=x.device)
    )

    if hidden_size <= 512:
        chunk_h = max(
            4 * group_size if quant_ue8m0 else 256,
            triton.next_power_of_2(hidden_size),
        )
    else:
        chunk_h = 512
    num_chunks = (hidden_size + chunk_h - 1) // chunk_h
    packs_per_chunk = max(1, chunk_h // (4 * group_size))
    active_groups = hidden_size // group_size
    if num_chunks == 1:
        num_warps = 8 if chunk_h >= 2048 else (4 if chunk_h >= 1024 else 1)
    elif chunk_h <= 512:
        num_warps = 1 if not quant_ue8m0 else 2
    else:
        num_warps = 2

    if quant_ue8m0:
        out_q_buf = (
            out_q.view(num_tokens, hidden_size)
            if out_q is not None
            else torch.empty(
                num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=x.device
            )
        )
        out_scale_buf = (
            out_scale
            if out_scale is not None
            else _make_ue8m0_scale_like(
                num_tokens, hidden_size, device=x.device, group_size=group_size
            )
        )
        out_scale_stride_k = int(out_scale_buf.stride(1))
        active_scale_packs = (active_groups + 3) // 4
        fp8_min = _FP8_E4M3_MIN
        fp8_max = _FP8_E4M3_MAX
    else:
        out_q_buf = out_norm_buf
        out_scale_buf = post_mix
        out_scale_stride_k = 1
        active_scale_packs = 1
        fp8_min = -448.0
        fp8_max = 448.0

    fast_launch(
        _mhc_pre_big_fuse_rmsnorm_kernel,
        (num_tokens,),
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_unit.scale,
        hc_unit.base,
        residual_flat,
        norm.weight.data,
        post_mix,
        comb_mix,
        out_norm_buf,
        out_q_buf,
        out_scale_buf,
        num_tokens,
        n_splits,
        int(hc_unit.hc_sinkhorn_iters),
        out_scale_stride_k,
        HIDDEN_SIZE=hidden_size,
        CHUNK_H=chunk_h,
        NUM_CHUNKS=num_chunks,
        IS_ALIGNED=(hidden_size % chunk_h == 0),
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
        PACKS_PER_CHUNK=packs_per_chunk,
        ACTIVE_GROUPS=active_groups,
        ACTIVE_SCALE_PACKS=active_scale_packs,
        HAS_TAIL_PAD=(num_chunks * chunk_h) != hidden_size,
        num_warps=num_warps,
        num_stages=1 if num_chunks == 1 else 2,
    )

    out_norm_view = out_norm_buf.view(*outer_shape, hidden_size)
    post_mix_view = post_mix.view(*outer_shape, _MHC_MULT, 1)
    comb_mix_view = comb_mix.view(*outer_shape, _MHC_MULT, _MHC_MULT)
    if quant_ue8m0:
        return out_norm_view, post_mix_view, comb_mix_view, (out_q_buf, out_scale_buf)
    return out_norm_view, post_mix_view, comb_mix_view


def fused_mhc_pre_norm(
    x: torch.Tensor,
    hc_unit: TileLangHCUnit,
    norm: RMSNorm,
    *,
    out_gemm_mul: Optional[torch.Tensor] = None,
    out_gemm_sqrsum: Optional[torch.Tensor] = None,
    out_post_mix: Optional[torch.Tensor] = None,
    out_comb_mix: Optional[torch.Tensor] = None,
    out_norm: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fused mHC pre + RMSNorm and return (x_pre_normed, post_mix, comb_mix)."""
    return _run_fused_mhc_pre_norm(
        x,
        hc_unit,
        norm,
        quant_ue8m0=False,
        out_gemm_mul=out_gemm_mul,
        out_gemm_sqrsum=out_gemm_sqrsum,
        out_post_mix=out_post_mix,
        out_comb_mix=out_comb_mix,
        out_norm=out_norm,
    )


def fused_mhc_pre_norm_quant(
    x: torch.Tensor,
    hc_unit: TileLangHCUnit,
    norm: RMSNorm,
    *,
    group_size: int = _UE8M0_GROUP_SIZE,
    clamp_eps: float = 1.0e-4,
    out_gemm_mul: Optional[torch.Tensor] = None,
    out_gemm_sqrsum: Optional[torch.Tensor] = None,
    out_post_mix: Optional[torch.Tensor] = None,
    out_comb_mix: Optional[torch.Tensor] = None,
    out_norm: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run fused mHC pre + RMSNorm + FP8 UE8M0 quant for attention QKV reuse."""
    return _run_fused_mhc_pre_norm(
        x,
        hc_unit,
        norm,
        quant_ue8m0=True,
        group_size=group_size,
        clamp_eps=clamp_eps,
        out_gemm_mul=out_gemm_mul,
        out_gemm_sqrsum=out_gemm_sqrsum,
        out_post_mix=out_post_mix,
        out_comb_mix=out_comb_mix,
        out_norm=out_norm,
        out_q=out_q,
        out_scale=out_scale,
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
        fast_launch(
            _mhc_mean_copy_kernel,
            grid,
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
        fast_launch(
            _triton_copy_2d_kernel,
            grid,
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
