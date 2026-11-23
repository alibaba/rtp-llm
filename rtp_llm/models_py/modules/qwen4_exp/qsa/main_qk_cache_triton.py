"""Qwen4 uniform target Q/K norm, native MRoPE and FP8 cache writer.

The owner validates its uniform target plan before preparing this operation.
Preparation is read-only; run() is the explicit cache mutation boundary.
The native writer writes page zero during dummy Graph execution too. Readers
continue to reject that allocator sentinel. Existing per-token scales are read
by attention unchanged; only this invocation's K/V entries receive unit scales.
"""

import math
from dataclasses import dataclass

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)

_TORCH_GIT = "70d99e998b4955e0049d13a98d77ae1b14db1f45"


@triton.jit
def _main_qk_norm_rope_cache(
    QKV,
    OUT,
    POS,
    QG,
    KG,
    ROWS: tl.constexpr,
    STRIDE: tl.constexpr,
    Q_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    EPS: tl.constexpr,
    BASE: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    LENS,
    TABLE,
    CACHE,
    SF,
    QUERY: tl.constexpr,
    TW: tl.constexpr,
    CP,
    CKV: tl.constexpr,
    SP: tl.constexpr,
    SKV: tl.constexpr,
    PAGES,
    PAGE_SIZE: tl.constexpr = 128,
    HAS_SCALES: tl.constexpr = True,
    HAS_POS: tl.constexpr = True,
    PDL: tl.constexpr = False,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    column = lane * 4
    base = row * STRIDE + head * 256
    logical_position = tl.load(LENS + row) - 1
    page = logical_position // PAGE_SIZE
    # The table lookup itself must stay inside the block table; a stale or
    # truncated LENS value must not turn into an out-of-bounds TABLE read.
    page_in_table = (page >= 0) & (page < TW)
    physical = tl.load(TABLE + (row // QUERY) * TW + page, page_in_table, other=0)
    valid_page = page_in_table & (physical >= 0) & (physical < PAGES)
    physical_i64 = physical.to(tl.int64)
    page_slot = logical_position % PAGE_SIZE
    frequency = lane
    if HAS_POS:
        axis = tl.where(
            (frequency % 3 == 1) & (frequency < H_SECTION * 3),
            1,
            tl.where((frequency % 3 == 2) & (frequency < W_SECTION * 3), 2, 0),
        )
        position = tl.load(POS + row * 3 + axis).to(tl.float32)
        # Native treats nonpositive per-axis coordinates as the logical
        # row (including the committed prefix), rather than explicit zero.
        position = tl.where(position > 0, position, logical_position.to(tl.float32))
    else:
        position = logical_position.to(tl.float32)
    # Match the native CUDA --use_fast_math sequence, including
    # approximate division and trig. libdevice pow/sin/cos use
    # different rounding and fail BF16 KV byte parity.
    log_base = tl.inline_asm_elementwise(
        "lg2.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[tl.full((), BASE, tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    denominator = tl.exp2((frequency.to(tl.float32) / 32.0) * log_base)
    angle = tl.inline_asm_elementwise(
        "div.approx.ftz.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[position, denominator],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    cosine = tl.inline_asm_elementwise(
        "cos.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[angle],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    sine = tl.inline_asm_elementwise(
        "sin.approx.ftz.f32 $0, $1;",
        constraints="=f,f",
        args=[angle],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    if PDL:
        tl.extra.cuda.gdc_wait()
    a0 = tl.load(QKV + base + column).to(tl.float32)
    a1 = tl.load(QKV + base + column + 1).to(tl.float32)
    a2 = tl.load(QKV + base + column + 2).to(tl.float32)
    a3 = tl.load(QKV + base + column + 3).to(tl.float32)
    b0 = tl.load(QKV + base + column + 128).to(tl.float32)
    b1 = tl.load(QKV + base + column + 129).to(tl.float32)
    b2 = tl.load(QKV + base + column + 130).to(tl.float32)
    b3 = tl.load(QKV + base + column + 131).to(tl.float32)
    s0, s1, s2, s3 = a0 * a0, a1 * a1, a2 * a2, a3 * a3
    t0, t1, t2, t3 = b0 * b0, b1 * b1, b2 * b2, b3 * b3
    # CUDA mean uses 64 reduction lanes for fewer than 16 outputs, otherwise
    # 32. Q and K use their own output counts, including replicated KV heads.
    wide = ((head < Q_HEADS) & (ROWS * Q_HEADS < 16)) | (
        (head >= Q_HEADS) & (ROWS * KV_HEADS < 16)
    )
    sum_wide = (((s0 + s1) + s2) + s3) + (((t0 + t1) + t2) + t3)
    sum_narrow = (((s0 + t0) + (s1 + t1)) + (s2 + t2)) + (s3 + t3)
    variance = tl.sum(tl.where(wide, sum_wide, sum_narrow), 0) / 256
    scale = tl.rsqrt(variance + EPS)
    for j in tl.static_range(4):
        for half in tl.static_range(2):
            rd = column + j + half * 128
            x = tl.load(QKV + base + rd).to(tl.float32)
            qg = tl.load(QG + rd).to(tl.float32)
            kg = tl.load(KG + rd).to(tl.float32)
            g = tl.where(head < Q_HEADS, qg, kg)
            value = ((x * scale) * (1.0 + g)).to(tl.bfloat16).to(tl.float32)
            other_rd = tl.where(rd < 32, rd + 32, rd - 32)
            other_x = tl.load(QKV + base + other_rd, rd < 64, other=0).to(tl.float32)
            other_qg = tl.load(QG + other_rd, rd < 64, other=0).to(tl.float32)
            other_kg = tl.load(KG + other_rd, rd < 64, other=0).to(tl.float32)
            other_g = tl.where(head < Q_HEADS, other_qg, other_kg)
            other_value = (
                ((other_x * scale) * (1.0 + other_g)).to(tl.bfloat16).to(tl.float32)
            )
            cosine_value = tl.gather(cosine, rd % 32, 0)
            sine_value = tl.gather(sine, rd % 32, 0)
            signed_other = tl.where(rd < 32, -other_value, other_value)
            rotated = tl.where(
                rd < 32,
                tl.fma(value, cosine_value, signed_other * sine_value),
                tl.fma(signed_other, sine_value, value * cosine_value),
            )
            rounded = tl.where(rd < 64, rotated, value).to(tl.bfloat16)
            if head < Q_HEADS:
                tl.store(OUT + (row * Q_HEADS + head) * 256 + rd, rounded)
            else:
                tl.store(
                    CACHE + physical_i64 * CP + page_slot * 256 + rd,
                    rounded,
                    valid_page,
                )
            if head == 0:
                v_offset = row * STRIDE + (Q_HEADS + KV_HEADS) * 256 + rd
                v = tl.load(QKV + v_offset)
                tl.store(
                    CACHE + physical_i64 * CP + CKV + page_slot * 256 + rd,
                    v,
                    valid_page,
                )
            if HAS_SCALES and (j == 0) and (half == 0):
                if head == Q_HEADS:
                    tl.store(
                        SF + physical_i64 * SP + page_slot, 1.0, valid_page
                    )
                    tl.store(
                        SF + physical_i64 * SP + SKV + page_slot,
                        1.0,
                        valid_page,
                    )
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()


@dataclass
class MainQkCacheWrite:
    qkv: torch.Tensor
    output: torch.Tensor
    positions: torch.Tensor
    norm: object
    config: object
    cache: torch.Tensor
    scales: Optional[torch.Tensor]
    lengths: torch.Tensor
    table: torch.Tensor
    page_size: int = 128
    cache_page_stride: int = 65536

    def run(self):
        qkv, norm = self.qkv, self.norm
        page_size = self.page_size
        has_scales = self.scales is not None
        has_pos = self.positions is not None
        pdl = sm90_pdl_enabled(qkv.device)
        _main_qk_norm_rope_cache[(qkv.shape[0], norm.head_num + 1)](
            qkv,
            self.output,
            self.positions if has_pos else self.lengths,
            norm.q_weight,
            norm.k_weight,
            ROWS=qkv.shape[0],
            STRIDE=qkv.stride(0),
            Q_HEADS=norm.head_num,
            KV_HEADS=1,
            EPS=norm.eps,
            BASE=float(self.config.base),
            H_SECTION=11,
            W_SECTION=10,
            LENS=self.lengths,
            TABLE=self.table,
            CACHE=self.cache,
            SF=self.scales if has_scales else self.output,
            QUERY=self.lengths.shape[1],
            TW=self.table.stride(0),
            CP=self.cache_page_stride,
            CKV=page_size * 256,
            SP=2 * page_size,
            SKV=page_size,
            PAGES=self.cache.shape[0],
            PAGE_SIZE=page_size,
            HAS_SCALES=has_scales,
            HAS_POS=has_pos,
            PDL=pdl,
            num_warps=1,
            enable_fp_fusion=False,
            launch_pdl=pdl,
        )
        return self.output


def prepare_main_qk_cache(qkv, positions, norm, config, cache, scales, plan, params):
    """Return a read-only prepared writer for the validated uniform target plan.

    Unsupported layout/math contracts fall back before any tensor mutation.
    """
    from rtp_llm.ops import RopeStyle

    rows = qkv.shape[0] if qkv.ndim == 2 else 0
    capability = cached_device_capability(qkv.device) if qkv.is_cuda else None
    if capability == (12, 0):
        arch_ok = (
            getattr(torch.version, "git_version", None) == _TORCH_GIT
            and 4 <= rows <= 32
        )
    elif capability == (9, 0):
        min_rows = 4 if hasattr(params, "max_seq_len") else 1
        arch_ok = torch.version.hip is None and min_rows <= rows <= 512
    else:
        arch_ok = False
    has_pos = positions is not None and positions.numel() > 0
    if not (
        arch_ok
        and qkv.dtype == torch.bfloat16
        and qkv.is_contiguous()
        and not qkv.requires_grad
        and norm.head_num in (3, 6, 12)
        and norm.kv_head_num == 1
        and norm.size_per_head == 256
        and qkv.shape[1] == (norm.head_num + 2) * 256
        and math.isfinite(norm.eps)
        and norm.eps > 0
        and all(
            weight.device == qkv.device
            and weight.shape == (256,)
            and weight.is_contiguous()
            and not weight.requires_grad
            and weight.dtype in (torch.float32, torch.bfloat16)
            for weight in (norm.q_weight, norm.k_weight)
        )
        and config.style == RopeStyle.Mrope
        and config.index_factor == 3
        and config.dim == 64
        and config.base == 10000000.0
        and config.scale == 1.0
        and config.offset == 0
        and (config.mrope_dim1, config.mrope_dim2, config.mrope_dim3) == (11, 11, 10)
        and (
            (
                has_pos
                and positions.device == qkv.device
                and positions.dtype == torch.int32
                and positions.is_contiguous()
                and positions.numel() == rows * 3
            )
            or (not has_pos and capability == (9, 0))
        )
        and cache is not None
        and cache.device == qkv.device
        and plan is not None
    ):
        return None

    page_size = 0
    cache_page_stride = 0
    if cache.dtype == torch.float8_e4m3fn:
        if not cache.is_contiguous():
            return None
        if cache.ndim == 2 and cache.shape[1] in (32768, 65536):
            page_size = cache.shape[1] // 512
        elif (
            cache.ndim == 5
            and cache.shape[1:3] == (2, 1)
            and cache.shape[3] in (64, 128)
            and cache.shape[4] == 256
        ):
            page_size = cache.shape[3]
        if page_size == 0 or not (
            scales is not None
            and scales.device == qkv.device
            and scales.dtype == torch.float32
            and scales.is_contiguous()
            and (
                (
                    scales.ndim == 2
                    and tuple(scales.shape) == (cache.shape[0], 2 * page_size)
                )
                or (
                    scales.ndim == 4
                    and tuple(scales.shape) == (cache.shape[0], 2, 1, page_size)
                )
            )
        ):
            return None
        cache_page_stride = 2 * page_size * 256
    elif cache.dtype == torch.bfloat16 and capability == (9, 0) and scales is None:
        if not cache.is_contiguous():
            return None
        if cache.ndim == 2 and cache.shape[1] in (32768, 65536):
            page_size = cache.shape[1] // 512
        elif (
            cache.ndim == 5
            and cache.shape[1:3] == (2, 1)
            and cache.shape[3] in (64, 128)
            and cache.shape[4] == 256
        ):
            page_size = cache.shape[3]
        if page_size == 0:
            return None
        cache_page_stride = 2 * page_size * 256
    else:
        return None

    lengths, table = plan.kv_lens, plan.block_table
    params_input_lengths = getattr(params, "input_lengths", None)
    params_seq_lengths = getattr(params, "sequence_lengths", None)
    params_ok = (
        getattr(params, "max_seq_len", None) == lengths.shape[1]
        and params_input_lengths is not None
        and params_input_lengths.numel() == lengths.shape[0]
    ) or (
        lengths.shape[1] == 1
        and params_seq_lengths is not None
        and params_seq_lengths.numel() == lengths.shape[0]
    )
    if not (
        lengths.device == qkv.device
        and lengths.dtype == torch.int32
        and lengths.ndim == 2
        and lengths.is_contiguous()
        and 1 <= lengths.shape[1] <= 4
        and lengths.numel() == rows
        and params_ok
        and table.device == qkv.device
        and table.dtype == torch.int32
        and table.ndim == 2
        and table.shape[0] == lengths.shape[0]
        and table.shape[1] > 0
        and table.stride(1) == 1
        and int(getattr(plan, "page_size", page_size)) == page_size
    ):
        return None
    output = torch.empty(
        (rows, norm.head_num * 256), device=qkv.device, dtype=qkv.dtype
    )
    return MainQkCacheWrite(
        qkv,
        output,
        positions,
        norm,
        config,
        cache,
        scales,
        lengths,
        table,
        page_size=page_size,
        cache_page_stride=cache_page_stride,
    )
