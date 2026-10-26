"""Bounded Qwen4 main Q/K norm and native MRoPE preparation candidate."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _main_qk_norm_rope(
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
    ROPE_SCALE: tl.constexpr,
    H_SECTION: tl.constexpr,
    W_SECTION: tl.constexpr,
    POSITION_BASE: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    column = lane * 4
    base = row * STRIDE + head * 256
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
            frequency = rd % 32
            axis = tl.where(
                (frequency % 3 == 1) & (frequency < H_SECTION * 3),
                1,
                tl.where((frequency % 3 == 2) & (frequency < W_SECTION * 3), 2, 0),
            )
            position = tl.load(POS + row * 3 + axis).to(tl.float32)
            # Native treats nonpositive per-axis coordinates as the logical
            # row (including the committed prefix), rather than explicit zero.
            position = tl.where(
                position > 0, position, (row + POSITION_BASE).to(tl.float32)
            )
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
            signed_other = tl.where(rd < 32, -other_value, other_value)
            rotated = tl.where(
                rd < 32,
                tl.fma(value, cosine, signed_other * sine),
                tl.fma(signed_other, sine, value * cosine),
            )
            tl.store(OUT + base + rd, tl.where(rd < 64, rotated, value))
            if head == 0:
                v_offset = row * STRIDE + (Q_HEADS + KV_HEADS) * 256 + rd
                tl.store(OUT + v_offset, tl.load(QKV + v_offset))


def try_main_qk_norm_rope(qkv, positions, norm, config, *, position_base=0):
    """Prepare true TP2/4/8 Q/K without changing any KV writer/state contract.

    The model selects only the measured ordinary unpadded 4K prefill domain.
    Position IDs follow its nonnegative text/mRoPE contract. Small decode and
    12K retain the original complete consumer; explicit tests may call them.
    """
    from rtp_llm.models_py.triton_kernels.common.zero_centered_rms_norm import (
        is_small_qk_supported,
    )
    from rtp_llm.ops import RopeStyle

    if not (
        qkv.ndim == 2
        and (
            is_small_qk_supported(
                qkv,
                norm.q_weight,
                norm.k_weight,
                norm.head_num,
                norm.kv_head_num,
                norm.size_per_head,
            )
            or (
                qkv.is_cuda
                and qkv.is_contiguous()
                and qkv.dtype == torch.bfloat16
                and torch.cuda.get_device_capability(qkv.device) == (12, 0)
                and norm.head_num in (3, 6, 12)
                and norm.kv_head_num == 1
                and norm.size_per_head == 256
                and qkv.shape[1] == (norm.head_num + 2) * 256
                and 33 <= qkv.shape[0] <= 32768
            )
        )
        and all(
            weight.is_cuda
            and weight.device == qkv.device
            and weight.shape == (256,)
            and weight.is_contiguous()
            and weight.dtype in (torch.bfloat16, torch.float32)
            and not weight.requires_grad
            for weight in (norm.q_weight, norm.k_weight)
        )
        and math.isfinite(norm.eps)
        and norm.eps > 0
        and isinstance(position_base, int)
        and 0 <= position_base <= 2**31 - 1 - qkv.shape[0]
        and positions.is_cuda
        and positions.device == qkv.device
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and positions.numel() == qkv.shape[0] * 3
        and config.style == RopeStyle.Mrope
        and config.index_factor == 3
        and config.dim == 64
        and config.base == 10000000.0
        and config.scale == 1.0
        and config.offset == 0
        and (config.mrope_dim1, config.mrope_dim2, config.mrope_dim3) == (11, 11, 10)
        and min(config.mrope_dim1, config.mrope_dim2, config.mrope_dim3) >= 0
        and config.mrope_dim1 + config.mrope_dim2 + config.mrope_dim3 == 32
    ):
        return None
    out = torch.empty_like(qkv)
    _main_qk_norm_rope[(qkv.shape[0], norm.head_num + norm.kv_head_num)](
        qkv,
        out,
        positions,
        norm.q_weight,
        norm.k_weight,
        ROWS=qkv.shape[0],
        STRIDE=qkv.stride(0),
        Q_HEADS=norm.head_num,
        KV_HEADS=norm.kv_head_num,
        EPS=norm.eps,
        BASE=float(config.base),
        ROPE_SCALE=float(config.scale),
        H_SECTION=config.mrope_dim2,
        W_SECTION=config.mrope_dim3,
        POSITION_BASE=position_base,
        num_warps=1,
        enable_fp_fusion=False,
    )
    return out
