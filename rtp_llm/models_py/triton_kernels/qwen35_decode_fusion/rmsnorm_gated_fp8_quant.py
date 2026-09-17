"""Fused RmsNormGated + per-token-group FP8 UE8M0 quant (linear-attn o_proj).

Replaces the two-kernel linear-attention epilogue (45 LA layers)::

    y = RmsNormGated(weight, group_size=128, eps=1e-6, activation="silu")(x, z)
    # NORM_BEFORE_GATE / IS_RMS_NORM: y = rmsnorm_group(x)*w * silu(z)
    fp8, scale = sgl_per_token_group_quant_fp8(
        y, 128, eps=1e-4,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )

Norm math is the IS_RMS_NORM + NORM_BEFORE_GATE + silu path from
``triton_kernels/common/layernorm_gated.py``
(``_layer_norm_fwd_1pass_kernel``). UE8M0 packing / FP8 clamp is the
loop from ``modules/dsv4/_fused_rmsnorm_fp8_quant_triton.py``. Do not
rewrite either formula.

397B decode: ``x``, ``z`` are ``[M, 8192]`` BF16 contiguous. ``weight``
is typically shared ``[128]`` (``RmsNormGated`` ``group_size=head_v_dim``).
``z`` is only read. Production skips the BF16 ``y`` store (o_proj is FP8).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.env import (
    is_decode_fusion_enabled,
)
from rtp_llm.models_py.triton_kernels.qwen35_decode_fusion.fp8_scale import (
    make_ue8m0_scale_like,
)

_GROUP_SIZE = 128
_NORM_EPS = 1.0e-6
_CLAMP_EPS = 1.0e-4
_FP8_DTYPE = torch.float8_e4m3fn


def _fusion_enabled() -> bool:
    return is_decode_fusion_enabled()


def _tensors_supported(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int,
    activation: str,
    bias: Optional[torch.Tensor],
) -> bool:
    if bias is not None:
        return False
    if activation != "silu":
        return False
    if group_size != _GROUP_SIZE:
        return False
    if not isinstance(x, torch.Tensor) or not isinstance(z, torch.Tensor):
        return False
    if not isinstance(weight, torch.Tensor):
        return False
    if x.dim() != 2 or z.dim() != 2 or weight.dim() != 1:
        return False
    if x.shape != z.shape:
        return False
    if x.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        return False
    if weight.dtype != torch.bfloat16:
        return False
    if (not x.is_cuda) or (not z.is_cuda) or (not weight.is_cuda):
        return False
    if x.device != z.device or x.device != weight.device:
        return False
    if (not x.is_contiguous()) or (not z.is_contiguous()):
        return False
    if not weight.is_contiguous():
        return False
    n = x.shape[-1]
    if n == 0 or n % group_size != 0:
        return False
    # SHARED_WEIGHT iff weight.shape == (group_size,); else per-column [N].
    if weight.shape not in ((group_size,), (n,)):
        return False
    return True


def is_supported(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int = _GROUP_SIZE,
    activation: str = "silu",
    bias: Optional[torch.Tensor] = None,
) -> bool:
    """Host-only checks; CUDA-graph safe (no ``.item()`` / device sync)."""
    if not _fusion_enabled():
        return False
    return _tensors_supported(
        x, z, weight, group_size=group_size, activation=activation, bias=bias
    )


# Norm: ``_layer_norm_fwd_1pass_kernel`` with IS_RMS_NORM / NORM_BEFORE_GATE /
# silu / optional SHARED_WEIGHT. Quant pack: DSV4 ``_rmsnorm_fp8_quant_kernel``.
# Production discards BF16 y (o_proj consumes FP8 only); WRITE_Y is optional.
@triton.jit(do_not_specialize=["M", "output_scale_stride_k"])
def _rmsnorm_gated_fp8_quant_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    output_y_ptr,
    output_q_ptr,
    output_scale_ptr,
    M,
    x_stride_m,
    z_stride_m,
    output_y_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    N: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
    pid_pack = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    m_offset = pid_m * BLOCK_M
    if m_offset >= M:
        return

    offs_m = tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = tl.arange(0, GROUP_SIZE)
    row_mask = (m_offset + offs_m) < M

    x_base = (m_offset + offs_m[:, None]) * x_stride_m
    z_base = (m_offset + offs_m[:, None]) * z_stride_m
    y_base = (m_offset + offs_m[:, None]) * output_y_stride_m
    q_base = (m_offset + offs_m[:, None]) * output_q_stride_m

    w_shared = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
    if SHARED_WEIGHT:
        w_shared = tl.load(w_ptr + offs_n, mask=offs_n < GROUP_SIZE, other=0.0).to(
            tl.float32
        )

    packed_scale = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for pack_idx in tl.static_range(4):
        group_id = pid_pack * 4 + pack_idx
        if group_id < NUM_GROUPS:
            n_offset = group_id * GROUP_SIZE
            cols = n_offset + offs_n
            mask = row_mask[:, None] & (cols[None, :] < N)

            x = tl.load(
                x_ptr + x_base + cols[None, :],
                mask=mask,
                other=0.0,
                eviction_policy="evict_first",
            ).to(tl.float32)
            xbar = tl.where(mask, x, 0.0)
            rstd = tl.rsqrt(tl.sum(xbar * xbar, axis=1) / GROUP_SIZE + EPS)
            if SHARED_WEIGHT:
                w = w_shared
            else:
                w = tl.load(w_ptr + cols, mask=cols < N, other=0.0).to(tl.float32)
            y = x * rstd[:, None] * w[None, :]
            z = tl.load(z_ptr + z_base + cols[None, :], mask=mask, other=0.0).to(
                tl.float32
            )
            y_bf16 = (y * (z * tl.sigmoid(z))).to(tl.bfloat16)
            if WRITE_Y:
                tl.store(output_y_ptr + y_base + cols[None, :], y_bf16, mask=mask)

            y_f32 = y_bf16.to(tl.float32)
            absmax = tl.max(tl.abs(y_f32), axis=1)
            scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
            exponent = tl.ceil(tl.log2(scale_raw))
            scale = tl.math.exp2(exponent)
            q = tl.clamp(y_f32 / scale[:, None], FP8_MIN, FP8_MAX)
            tl.store(
                output_q_ptr + q_base + cols[None, :],
                q.to(output_q_ptr.dtype.element_ty),
                mask=mask,
            )
            packed_scale = packed_scale | (
                tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32) << (pack_idx * 8)
            )

    tl.store(
        output_scale_ptr + pid_pack * output_scale_stride_k + m_offset + offs_m,
        packed_scale,
        mask=row_mask,
    )


# One CTA per token. Wide tile + reshape to (TILE_GROUPS, 128) — same layout
# as G's register quant, but RMS is per group (not full-row). ``n_tiles`` is
# a runtime scalar so the N-loop is not unrolled (avoids the 50us spill).
@triton.jit(do_not_specialize=["M", "output_scale_stride_k", "n_tiles"])
def _rmsnorm_gated_fp8_quant_tile_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    output_y_ptr,
    output_q_ptr,
    output_scale_ptr,
    M,
    n_tiles,
    x_stride_m,
    z_stride_m,
    output_y_stride_m,
    output_q_stride_m,
    output_scale_stride_k,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SCALE_PACKS: tl.constexpr,
    EPS: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    CLAMP_EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILE_GROUPS: tl.constexpr,
    TILE_PACKS: tl.constexpr,
    SHARED_WEIGHT: tl.constexpr,
    WRITE_Y: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    if pid_m >= M:
        return

    offs_tg = tl.arange(0, TILE_GROUPS)[:, None]
    offs_g = tl.arange(0, GROUP_SIZE)[None, :]
    offs_pack = tl.arange(0, TILE_PACKS)
    x_row = x_ptr + pid_m * x_stride_m
    z_row = z_ptr + pid_m * z_stride_m
    y_row = output_y_ptr + pid_m * output_y_stride_m
    q_row = output_q_ptr + pid_m * output_q_stride_m
    w_shared = tl.zeros((GROUP_SIZE,), dtype=tl.float32)
    if SHARED_WEIGHT:
        w_shared = tl.load(
            w_ptr + tl.arange(0, GROUP_SIZE),
            mask=tl.arange(0, GROUP_SIZE) < GROUP_SIZE,
            other=0.0,
        ).to(tl.float32)

    for tile in range(0, n_tiles):
        n_offset = tile * BLOCK_N
        cols = n_offset + offs_tg * GROUP_SIZE + offs_g
        mask = cols < N
        x_g = tl.load(
            x_row + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        z_g = tl.load(
            z_row + cols, mask=mask, other=0.0, eviction_policy="evict_first"
        ).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(x_g * x_g, axis=1) / GROUP_SIZE + EPS)
        if SHARED_WEIGHT:
            y_g = x_g * rstd[:, None] * w_shared[None, :] * (z_g * tl.sigmoid(z_g))
        else:
            w_g = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            y_g = x_g * rstd[:, None] * w_g * (z_g * tl.sigmoid(z_g))
        y_bf16 = y_g.to(tl.bfloat16)
        if WRITE_Y:
            tl.store(y_row + cols, y_bf16, mask=mask)

        y_f32 = y_bf16.to(tl.float32)
        absmax = tl.max(tl.abs(y_f32), axis=1)
        scale_raw = tl.maximum(absmax, CLAMP_EPS) / FP8_MAX
        exponent = tl.ceil(tl.log2(scale_raw))
        scale = tl.math.exp2(exponent)
        q = tl.clamp(y_f32 / scale[:, None], FP8_MIN, FP8_MAX)
        tl.store(q_row + cols, q.to(output_q_ptr.dtype.element_ty), mask=mask)

        e4 = tl.reshape(
            tl.clamp(exponent + 127.0, 0.0, 255.0).to(tl.int32),
            (TILE_PACKS, 4),
        )
        packed = tl.sum(e4 << (tl.arange(0, 4) * 8)[None, :], axis=1)
        pack_ids = tile * TILE_PACKS + offs_pack
        tl.store(
            output_scale_ptr + pack_ids * output_scale_stride_k + pid_m,
            packed,
            mask=pack_ids < SCALE_PACKS,
        )


def rmsnorm_gated_fp8_quant(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int = _GROUP_SIZE,
    eps: float = _NORM_EPS,
    quant_eps: float = _CLAMP_EPS,
    activation: str = "silu",
    out_y: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_scale: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    write_y: bool = True,
    variant: Optional[str] = None,
    block_n: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse grouped RMSNorm * silu(z) and UE8M0 FP8 quant.

    Args:
        x: ``[M, N]`` BF16 contiguous CUDA residual / GDN output.
        z: same shape / dtype / device / layout as ``x`` (not overwritten).
        weight: shared ``[group_size]`` or per-column ``[N]`` BF16.
        write_y: write BF16 gated-RMS. Production o_proj only needs FP8.
        variant: ``"tile"`` (1 CTA/token, wide reshape) or ``"pack"`` (split N).

    Returns:
        y: ``[M, N]`` BF16 gated-RMS output (unwritten if ``write_y`` is False)
        fp8: ``[M, N]`` ``float8_e4m3fn``
        scale: TMA-aligned MN-major packed UE8M0 int32 from
            ``make_ue8m0_scale_like``
    """
    if not _tensors_supported(
        x, z, weight, group_size=group_size, activation=activation, bias=None
    ):
        raise ValueError(
            "rmsnorm_gated_fp8_quant expects 2D contiguous CUDA BF16 x/z, "
            f"weight [group_size] or [N], silu, group_size={_GROUP_SIZE}; got "
            f"x={tuple(getattr(x, 'shape', ()))} {getattr(x, 'dtype', None)} "
            f"z={tuple(getattr(z, 'shape', ()))} {getattr(z, 'dtype', None)} "
            f"w={tuple(getattr(weight, 'shape', ()))} act={activation}"
        )

    m, n = x.shape
    if out_y is None:
        out_y = torch.empty_like(x)
    elif out_y.shape != x.shape or out_y.dtype != torch.bfloat16:
        raise ValueError(
            f"out_y must be bf16 {tuple(x.shape)}, got {out_y.dtype} {tuple(out_y.shape)}"
        )
    elif not out_y.is_contiguous() or out_y.device != x.device:
        raise ValueError("out_y must be contiguous on the same device as x")

    if out_q is None:
        out_q = torch.empty((m, n), device=x.device, dtype=_FP8_DTYPE)
    if out_scale is None:
        out_scale = make_ue8m0_scale_like(x.shape, device=x.device, group_size=group_size)
    if m == 0:
        return out_y, out_q, out_scale

    finfo = torch.finfo(_FP8_DTYPE)
    num_groups = n // group_size
    num_packed = (num_groups + 3) // 4
    shared_weight = weight.shape == (group_size,)
    if variant is None:
        variant = "pack"
    if variant not in ("tile", "pack"):
        raise ValueError(f"variant must be 'tile' or 'pack', got {variant}")
    # Graph-tuned on L20D: BLOCK_M=4 / warps=4 / stages=3 wins at decode M<16
    # (~4.6us). BLOCK_M=1 is much slower; BLOCK_M=16 wastes masked rows at M=1.
    if block_m is None:
        block_m = 4 if m < 16 else 8
    if block_n is None:
        block_n = 4096
    if block_n % (group_size * 4) != 0:
        raise ValueError(
            f"block_n must be a multiple of {group_size * 4} for UE8M0 packs, got {block_n}"
        )
    if num_warps is None:
        num_warps = 4
    if num_stages is None:
        num_stages = 3 if m < 16 else 2
    common = dict(
        EPS=eps,
        FP8_MIN=float(finfo.min),
        FP8_MAX=float(finfo.max),
        CLAMP_EPS=quant_eps,
        SHARED_WEIGHT=shared_weight,
        WRITE_Y=write_y,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    args = (
        x,
        z,
        weight,
        out_y,
        out_q,
        out_scale,
        m,
        x.stride(0),
        z.stride(0),
        out_y.stride(0),
        out_q.stride(0),
        out_scale.stride(1),
    )
    if variant == "tile":
        _rmsnorm_gated_fp8_quant_tile_kernel[(m,)](
            x,
            z,
            weight,
            out_y,
            out_q,
            out_scale,
            m,
            triton.cdiv(n, block_n),
            x.stride(0),
            z.stride(0),
            out_y.stride(0),
            out_q.stride(0),
            out_scale.stride(1),
            N=n,
            GROUP_SIZE=group_size,
            SCALE_PACKS=num_packed,
            BLOCK_N=block_n,
            TILE_GROUPS=block_n // group_size,
            TILE_PACKS=block_n // (group_size * 4),
            **common,
        )
    else:
        _rmsnorm_gated_fp8_quant_kernel[(num_packed, triton.cdiv(m, block_m))](
            *args,
            N=n,
            NUM_GROUPS=num_groups,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            **common,
        )
    return out_y, out_q, out_scale


def maybe_rmsnorm_gated_fp8_quant(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    *,
    group_size: int = _GROUP_SIZE,
    eps: float = _NORM_EPS,
    quant_eps: float = _CLAMP_EPS,
    activation: str = "silu",
    bias: Optional[torch.Tensor] = None,
    out_y: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return ``(y, fp8, scale)`` when supported; otherwise ``None`` (old path).

    Production ``o_proj`` only reads FP8, so BF16 ``y`` is not written.
    """
    if not is_supported(
        x, z, weight, group_size=group_size, activation=activation, bias=bias
    ):
        return None
    return rmsnorm_gated_fp8_quant(
        x,
        z,
        weight,
        group_size=group_size,
        eps=eps,
        quant_eps=quant_eps,
        activation=activation,
        out_y=out_y,
        write_y=False,
    )
